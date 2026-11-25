# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from types import SimpleNamespace
from unittest.mock import patch

from pyworkflow.object import Set
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_trigger_data import XmippProtTriggerData


class _FakeImage:
    def __init__(self, objId, creation='2026-09-17 08:00:00'):
        self._objId = objId
        self._creation = creation

    def clone(self):
        return _FakeImage(self._objId, self._creation)

    def getObjId(self):
        return self._objId

    def getObjCreation(self):
        return self._creation


class _FakeInputSet:
    def __init__(self, items, streamClosed=False):
        self._items = list(items)
        self._streamClosed = streamClosed
        self.closed = False
        self.whereCalls = []

    def loadAllProperties(self):
        pass

    def getUniqueValues(self, field, where=None):
        if field != 'id':
            raise AssertionError('Unexpected field: %s' % field)
        self.whereCalls.append(where)
        ids = sorted(item.getObjId() for item in self._items)
        if where:
            threshold = int(where.split('>')[1].strip())
            ids = [itemId for itemId in ids if itemId > threshold]
        return ids

    def iterItems(self, orderBy='id', direction='ASC', where=None, limit=None):
        items = sorted(self._items, key=lambda item: item.getObjId())

        if where and where.startswith('id IN'):
            idsStr = where[where.index('(') + 1: where.index(')')]
            wantedIds = {
                int(itemId.strip())
                for itemId in idsStr.split(',')
                if itemId.strip()
            }
            items = [
                item
                for item in items
                if item.getObjId() in wantedIds
            ]

        if direction == 'DESC':
            items = list(reversed(items))
        if limit is not None and limit > 0:
            items = items[:limit]
        return iter(items)

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closed = True


class _FakeOutputSet:
    STREAM_OPEN = Set.STREAM_OPEN

    def __init__(self, filename=None):
        self.ids = {1}
        self.appended = []
        self.streamState = None

    def loadAllProperties(self):
        pass

    def enableAppend(self):
        pass

    def setStreamState(self, state):
        self.streamState = state

    def copyInfo(self, inputs):
        pass

    def getSize(self):
        return len(self.ids)

    def getIdSet(self):
        return set(self.ids)

    def append(self, image):
        self.ids.add(image.getObjId())
        self.appended.append(image.getObjId())


class TestXmippTriggerDataRegression(BaseTest):
    """Regression tests for Trigger Data streaming and Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self, **kwargs):
        params = {'outputSize': 2, 'allImages': True, 'splitImages': False, 'delay': 4}
        params.update(kwargs)
        prot = self.newProtocol(XmippProtTriggerData, **params)
        prot.images = []
        prot.splitedImages = []
        prot.outputCount = 0
        prot.finished = False
        return prot

    def testNewInputDoesNotDependOnSqliteMtimeAndKeepsEqualCreationIds(self):
        prot = self._newProtocol()
        creation = '2026-09-17 08:00:00'
        image1 = _FakeImage(1, creation)
        image2 = _FakeImage(2, creation)
        inputSet = _FakeInputSet([image1, image2])

        prot.images = [image1]
        prot.check = creation
        prot.inputImages = SimpleNamespace(get=lambda: inputSet)
        prot._fillingOutput = lambda: None

        with patch('xmipp3.protocols.protocol_trigger_data.os.path.getmtime', side_effect=AssertionError('Streaming input must not depend on SQLite mtime.')):
            prot._checkNewInput()

        self.assertEqual([2], [image.getObjId() for image in prot.newImages])
        self.assertEqual([1, 2], [image.getObjId() for image in prot.images])
        self.assertIn('id > 0', inputSet.whereCalls)
        self.assertTrue(inputSet.closed)

    def testLoadOutputSetSkipsAlreadyPersistedIds(self):
        prot = self._newProtocol()
        prot.inputImages = SimpleNamespace(get=lambda: object())
        prot.getOututName = lambda: 'outputParticles'
        prot._create_FakeOutputSet = lambda *args: _FakeOutputSet()

        outputSet = prot._loadOutputSet(
            _FakeOutputSet,
            'outputParticles',
            [_FakeImage(1), _FakeImage(2)],
        )

        self.assertEqual({1, 2}, outputSet.getIdSet())
        self.assertEqual([2], outputSet.appended)

    def testLoadOutputSetReusesLogicalOutputWithoutBackingFile(self):
        # Regression test: an output that Scipion already knows about
        # (protocol.outputParticles is set) must be reused directly as the
        # logical output, without depending on any backing-file identity.
        prot = self._newProtocol()
        prot.inputImages = SimpleNamespace(get=lambda: object())
        prot.getOututName = lambda: 'outputParticles'

        existingOutputSet = _FakeOutputSet()
        existingOutputSet.ids = {1, 2}
        prot.outputParticles = existingOutputSet

        outputSet = prot._loadOutputSet(
            _FakeOutputSet,
            'outputParticles',
            [_FakeImage(2), _FakeImage(3)],
        )

        self.assertIs(existingOutputSet, outputSet)
        self.assertEqual({1, 2, 3}, outputSet.getIdSet())
        self.assertEqual([3], outputSet.appended)

    def testStaticOutputIsRepublishedWhenSqliteAlreadyExists(self):
        prot = self._newProtocol(allImages=False, outputSize=1)
        prot.images = [_FakeImage(1)]
        prot.allImages.set(False)
        prot.outputSize.set(1)
        prot.getImagesType = lambda letters='default': 'particles' if letters == 'lower' else 'Particles'
        prot.getOututName = lambda: 'outputParticles'
        prot.getImagesClass = lambda: _FakeOutputSet
        prot._getPath = lambda name: name

        outputSet = object()
        prot._loadOutputSet = lambda *args, **kwargs: outputSet
        updates = []
        prot._updateOutputSet = lambda name, output, streamMode: updates.append((name, output, streamMode))

        prot._fillingOutput()

        self.assertEqual([('outputParticles', outputSet, Set.STREAM_CLOSED)], updates)

    def testResumeRestoresOutputCountAndSkipsPersistedIdsInSplitMode(self):
        # Regression test: on Resume, self.outputCount (used to name the
        # next batch, e.g. outputParticles3) must be reconstructed from
        # how many batch outputs are actually already persisted - not left
        # at 0, which would recreate outputParticles1 and risk mixing
        # already-published items into a differently-named new batch.
        prot = self._newProtocol(splitImages=True)
        prot.getOututName = lambda: 'outputParticles'
        prot.outputParticles1 = _FakeOutputSet()
        prot.outputParticles1.ids = {1, 2}
        prot.outputParticles2 = _FakeOutputSet()
        prot.outputParticles2.ids = {3, 4}

        prot._restoreStreamingState()

        self.assertEqual(2, prot.outputCount)
        self.assertEqual(
            {1, 2, 3, 4},
            {image.getObjId() for image in prot.images},
        )
        self.assertEqual([], prot.splitedImages)

    def testResumeRestoresPersistedIdsInFullStreamingMode(self):
        prot = self._newProtocol(splitImages=False)
        prot.getOututName = lambda: 'outputParticles'
        prot.outputParticles = _FakeOutputSet()
        prot.outputParticles.ids = {1, 2}

        prot._restoreStreamingState()

        self.assertEqual(0, prot.outputCount)
        self.assertEqual(
            {1, 2},
            {image.getObjId() for image in prot.images},
        )

    def testInsertAllStepsRestoresStateOnlyWhenContinued(self):
        prot = self._newProtocol()
        prot.getOututName = lambda: 'outputParticles'
        prot.outputParticles = _FakeOutputSet()
        prot.outputParticles.ids = {1, 2}
        prot.isContinued = lambda: True
        prot.setImagesClass = lambda: None
        prot.setImagesType = lambda: None

        prot._prepareStreamingGenerator()

        self.assertEqual(
            {1, 2},
            {image.getObjId() for image in prot.images},
        )

    def testInsertAllStepsDoesNotRestoreOnFreshRun(self):
        prot = self._newProtocol()
        prot.getOututName = lambda: 'outputParticles'
        prot.outputParticles = _FakeOutputSet()
        prot.outputParticles.ids = {1, 2}
        prot.isContinued = lambda: False
        prot.setImagesClass = lambda: None
        prot.setImagesType = lambda: None

        prot._prepareStreamingGenerator()

        self.assertEqual([], prot.images)
        self.assertEqual(0, prot.outputCount)

# Finalization regression: the executor performs one last stepsCheck callback
# after it has already found no pending steps.
from unittest.mock import Mock

from pyworkflow.tests import BaseTest, setupTestProject
from xmipp3.protocols.protocol_trigger_data import XmippProtTriggerData


class TestXmippTriggerDataFinalizationRegression(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        prot = self.newProtocol(XmippProtTriggerData)
        prot.finished = True
        prot._prepareStreamingGenerator = Mock()
        prot._checkNewInput = Mock()
        prot._checkNewOutput = Mock()
        prot._insertFunctionStep = Mock(return_value=1)
        prot.createOutputStep = Mock()

        prot.stepsGeneratorStep()

        prot._checkNewInput.assert_not_called()
        prot._checkNewOutput.assert_not_called()
        prot._insertFunctionStep.assert_called_once()


class _RefreshRequiredTriggerOutput:
    def __init__(self, ids):
        self.ids = set(ids)
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True

    def getIdSet(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted TriggerData output must be refreshed before reading ids.'
            )
        return set(self.ids)


class TestXmippTriggerDataLogicalOutputRestore(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testResumeRefreshesPersistedOutputBeforeRestoringIds(self):
        prot = self.newProtocol(
            XmippProtTriggerData,
            outputSize=2,
            allImages=True,
            splitImages=False,
            delay=4,
        )
        prot.getOututName = lambda: 'outputParticles'
        prot.outputParticles = _RefreshRequiredTriggerOutput({1, 2})

        persistedIds, batchCount = prot._getPersistedOutputIds()

        self.assertTrue(prot.outputParticles.loaded)
        self.assertEqual({1, 2}, persistedIds)
        self.assertEqual(0, batchCount)
