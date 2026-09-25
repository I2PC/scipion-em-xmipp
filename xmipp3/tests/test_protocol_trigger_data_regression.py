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

    def iterItems(self, orderBy='id', direction='ASC', where=None, limit=None):
        self.whereCalls.append(where)
        items = sorted(self._items, key=lambda item: (item.getObjCreation(), item.getObjId()))
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
        prot.inputImages = SimpleNamespace(get=lambda: SimpleNamespace(getFileName=lambda: 'input.sqlite'))
        prot._inputClass = lambda filename=None: inputSet
        prot._fillingOutput = lambda: None

        with patch('xmipp3.protocols.protocol_trigger_data.os.path.getmtime', side_effect=AssertionError('Streaming input must not depend on SQLite mtime.')):
            prot._checkNewInput()

        self.assertEqual([2], [image.getObjId() for image in prot.newImages])
        self.assertEqual([1, 2], [image.getObjId() for image in prot.images])
        self.assertTrue(any(where is not None and '>=' in where for where in inputSet.whereCalls))
        self.assertTrue(inputSet.closed)

    def testLoadOutputSetSkipsAlreadyPersistedIds(self):
        prot = self._newProtocol()
        prot.inputImages = SimpleNamespace(get=lambda: object())
        prot._getPath = lambda name: name

        with patch('xmipp3.protocols.protocol_trigger_data.os.path.exists', return_value=True):
            outputSet = prot._loadOutputSet(_FakeOutputSet, 'particles.sqlite', [_FakeImage(1), _FakeImage(2)])

        self.assertEqual({1, 2}, outputSet.getIdSet())
        self.assertEqual([2], outputSet.appended)

    def testLoadOutputSetReusesLogicalOutputWithoutBackingFile(self):
        # Regression test: an output that Scipion already knows about
        # (protocol.outputParticles is set) must be reused even when its
        # backing file was never materialized on disk yet. Falling through
        # to "no backing file -> build a fresh, empty Set" would silently
        # discard whatever was already appended to the real logical output.
        prot = self._newProtocol()
        prot.inputImages = SimpleNamespace(get=lambda: object())
        prot._getPath = lambda name: name

        existingOutputSet = _FakeOutputSet()
        existingOutputSet.ids = {1, 2}
        prot.outputParticles = existingOutputSet

        with patch('xmipp3.protocols.protocol_trigger_data.os.path.exists', return_value=False):
            outputSet = prot._loadOutputSet(
                _FakeOutputSet, 'particles.dat', [_FakeImage(2), _FakeImage(3)],
                outputName='outputParticles',
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

        with patch('xmipp3.protocols.protocol_trigger_data.os.path.exists', return_value=True):
            prot._fillingOutput()

        self.assertEqual([('outputParticles', outputSet, Set.STREAM_CLOSED)], updates)

# Finalization regression: the executor performs one last stepsCheck callback
# after it has already found no pending steps.
from unittest.mock import Mock

from pyworkflow.tests import BaseTest, setupTestProject
from xmipp3.protocols.protocol_trigger_data import XmippProtTriggerData


class TestXmippTriggerDataFinalizationRegression(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testFinishedStepsCheckIsNoOp(self):
        prot = self.newProtocol(XmippProtTriggerData)
        prot.finished = True
        prot._checkNewInput = Mock()
        prot._checkNewOutput = Mock()

        prot._stepsCheck()

        prot._checkNewInput.assert_not_called()
        prot._checkNewOutput.assert_not_called()

