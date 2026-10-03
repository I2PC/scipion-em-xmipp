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

from pyworkflow.object import Float
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_screen_particles import XmippProtScreenParticles
from xmipp3.tests.streaming_test_utils import LogicalOutputSetProbe, FreshOutputSetProbe
from xmipp3.tests.streaming_test_utils import FakeOutputSet as _FakeOutputSet


class _FakeParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _FakeInputSet:
    def __init__(self, ids, streamClosed=False, gettableIds=None):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self._gettableIds = set(ids) if gettableIds is None else set(gettableIds)
        self.closed = False

    def loadAllProperties(self):
        pass

    def __contains__(self, itemId):
        return itemId in self._gettableIds

    def getUniqueValues(self, attr, where=None):
        if where is None:
            return sorted(self._ids)
        threshold = int(where.split('>')[1].strip())
        return sorted(itemId for itemId in self._ids if itemId > threshold)

    def getItem(self, field, value):
        return _FakeParticle(value)

    def isStreamClosed(self):
        return self._streamClosed

    def getSize(self):
        return len(self._ids)

    def close(self):
        self.closed = True


class _FakePointer:
    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj


class TestXmippScreenParticlesRegression(BaseTest):
    """Regression tests for Screen Particles streaming and Continue handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtScreenParticles)
        prot.fnInputMd = prot._getExtraPath('input.xmd')
        prot.fnInputOldMd = prot._getExtraPath('inputOld.xmd')
        prot.fnOutputMd = prot._getExtraPath('output.xmd')
        prot.fnProcessedIds = prot._getExtraPath('processed_ids.txt')
        return prot

    def testLoadInputUsesProcessedIdsInsteadOfCreationTimestamp(self):
        prot = self._newProtocol()
        inputSet = _FakeInputSet([1, 2, 3])
        prot.inputParticles = _FakePointer(inputSet)
        prot._lastInputId = 0
        prot._getKnownProcessedParticleIds = lambda: {1, 2}
        written = {}

        def captureWrite(items, filename, **kwargs):
            written[filename] = [item.getObjId() for item in items]

        with patch('xmipp3.protocols.protocol_screen_particles.writeSetOfParticles', side_effect=captureWrite):
            inputSize, streamClosed = prot._loadInput()

        self.assertEqual([3], written[prot.fnInputMd])
        self.assertEqual([1, 2], written[prot.fnInputOldMd])
        self.assertEqual(3, inputSize)
        self.assertFalse(streamClosed)
        self.assertTrue(inputSet.closed)

    def testLoadInputDefersParticleNotYetVisibleWithoutPermanentLoss(self):
        # Regression test: Set.getItem raises (UnboundLocalError) rather
        # than returning None for a row it cannot find, and
        # _discoverIdsAfter already advances the id watermark past a
        # discovered id regardless of whether it is actually selectable
        # yet. A momentarily-invisible particle must be kept pending and
        # retried on a later check, not dropped - dropping it would lose
        # it forever, since the watermark never revisits an id once
        # passed.
        prot = self._newProtocol()
        inputSet = _FakeInputSet([2], gettableIds=set())
        prot.inputParticles = _FakePointer(inputSet)
        prot._lastInputId = 0
        prot._getKnownProcessedParticleIds = lambda: set()
        written = {}

        def captureWrite(items, filename, **kwargs):
            written[filename] = [item.getObjId() for item in items]

        with patch(
                'xmipp3.protocols.protocol_screen_particles.writeSetOfParticles',
                side_effect=captureWrite,
        ):
            prot._loadInput()  # must not raise

        self.assertEqual([], written.get(prot.fnInputMd, []))
        self.assertEqual({2}, prot._pendingParticleIds)

        # The particle becomes visible on a later check.
        inputSet._gettableIds = {2}
        with patch(
                'xmipp3.protocols.protocol_screen_particles.writeSetOfParticles',
                side_effect=captureWrite,
        ):
            prot._loadInput()

        self.assertEqual([2], written[prot.fnInputMd])
        self.assertEqual(set(), prot._pendingParticleIds)

    def testNewInputDoesNotDependOnSqliteMtime(self):
        prot = self._newProtocol()
        prot._loadInput = lambda: (3, False)
        prot._insertNewPartsSteps = lambda: []
        prot.newDeps = []
        prot.updateSteps = lambda: None

        with patch('xmipp3.protocols.protocol_screen_particles.os.path.getmtime', side_effect=AssertionError('Streaming input must not depend on SQLite mtime.')), patch('xmipp3.protocols.protocol_screen_particles.os.path.exists', return_value=False), patch('xmipp3.protocols.protocol_screen_particles.isEmpty', return_value=False):
            prot._checkNewInput()

    def testAppendNewParticlesSkipsAlreadyPersistedIds(self):
        prot = self._newProtocol()
        outputSet = _FakeOutputSet([1])
        prot._appendNewParticles(outputSet, [_FakeParticle(1), _FakeParticle(2)])
        self.assertEqual({1, 2}, outputSet.ids)
        self.assertEqual([2], outputSet.appended)

    def testAppendNewParticlesDoesNotQueryIdsOnFreshSet(self):
        prot = self._newProtocol()

        class _FreshOutputSet(_FakeOutputSet):
            def getIdSet(self):
                raise AssertionError('Fresh output Set must not query IDs before its first append.')

        outputSet = _FreshOutputSet()
        prot._appendNewParticles(outputSet, [_FakeParticle(1)])
        self.assertEqual({1}, outputSet.ids)
        self.assertEqual([1], outputSet.appended)

    def testStreamingOutputReusesLogicalSetWithoutLegacySqlite(self):
        class LogicalOutputSet:
            def __init__(self):
                self.enableAppendCalls = 0
                self.copyInfoCalls = 0

            def enableAppend(self):
                self.enableAppendCalls += 1

            def copyInfo(self, inputSet):
                self.copyInfoCalls += 1

        class FreshOutputSet:
            STREAM_OPEN = 1

            def __init__(self, filename=None):
                self.filename = filename

            def setStreamState(self, state):
                self.streamState = state

            def copyInfo(self, inputSet):
                self.inputSet = inputSet

        inputSet = object()
        prot = self._newProtocol()
        logicalOutput = LogicalOutputSetProbe()
        prot.outputParticles = logicalOutput
        prot.inputParticles = SimpleNamespace(get=lambda: inputSet)
        prot._getPath = lambda baseName: '/tmp/' + baseName

        with patch(
            'xmipp3.protocols.protocol_screen_particles.os.path.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSetProbe,
                'outputParticles.sqlite',
            )

        self.assertIs(outputSet, logicalOutput)
        self.assertEqual(1, logicalOutput.enableAppendCalls)
        self.assertEqual(1, logicalOutput.copyInfoCalls)

    def testProcessedCheckpointIsWrittenAfterOutputUpdate(self):
        # Regression test: done-tracking checkpoints (_markOutputIdsPersisted
        # / _markRejectedParticleIds) must only run after the output Set has
        # already been persisted via _updateOutputSet - never before.
        prot = self._newProtocol()
        prot.finished = False
        prot.streamClosed = False
        prot.outputSize = 1
        prot.inputSize = 2
        outputSet = _FakeOutputSet()
        partsSet = _FakeOutputSet()
        events = []

        prot._loadOutputSet = lambda *args: outputSet
        prot._createSetOfParticles = lambda: partsSet
        prot._readMetadataIds = lambda _: [2]
        prot._appendNewParticles = lambda *args: events.append('append')
        prot._recalculateSummaryValues = lambda *args: None
        prot._updateOutputSet = lambda *args: events.append('update')
        prot._defineTransformRelation = lambda *args: events.append('relation')
        prot._markOutputIdsPersisted = lambda *args: events.append('persisted')
        prot._markRejectedParticleIds = lambda *args: events.append('rejected')
        prot._getKnownProcessedParticleIds = lambda: {1, 2}
        prot._store = lambda *args: None

        with patch('xmipp3.protocols.protocol_screen_particles.os.path.exists', return_value=True), patch('xmipp3.protocols.protocol_screen_particles.readSetOfParticles', side_effect=lambda fn, s: s.append(_FakeParticle(2))), patch('xmipp3.protocols.protocol_screen_particles.writeSetOfParticles'), patch('xmipp3.protocols.protocol_screen_particles.cleanPath'):
            prot._checkNewOutput()

        self.assertEqual(
            ['append', 'update', 'relation', 'persisted', 'rejected'],
            events,
        )

    def testContinuePreservesVarianceThreshold(self):
        prot = self._newProtocol()
        prot.minZScore = Float(1.0)
        prot.maxZScore = Float(2.0)
        prot.sumZScore = Float(3.0)
        prot.varThreshold = Float(4.0)
        prot.isContinued = lambda: True
        prot._store = lambda *args: None
        prot._initializeZscores()
        self.assertEqual(4.0, prot.varThreshold.get())


    def testSharedStreamingOutputLoaderContract(self):
        import xmipp3.utils as xmippUtils

        loader = getattr(xmippUtils, 'loadOutputSetForAppend', None)
        self.assertIsNotNone(
            loader,
            'Streaming output Set loading should be shared instead of '
            'duplicated across protocols.',
        )

        class LogicalOutputSet:
            def __init__(self):
                self.enableAppendCalls = 0

            def enableAppend(self):
                self.enableAppendCalls += 1

        class FreshOutputSet:
            STREAM_OPEN = 7

            def __init__(self, filename=None):
                self.filename = filename
                self.streamState = None

            def setStreamState(self, state):
                self.streamState = state

        class Protocol:
            def __init__(self, logicalOutput=None):
                if logicalOutput is not None:
                    self.outputParticles = logicalOutput

            def _getPath(self, baseName):
                return '/tmp/' + baseName

        logicalOutput = LogicalOutputSetProbe()
        outputSet, isNew = loader(
            Protocol(logicalOutput),
            FreshOutputSetProbe,
            'outputParticles.sqlite',
            'outputParticles',
        )

        self.assertIs(outputSet, logicalOutput)
        self.assertFalse(isNew)
        self.assertEqual(1, logicalOutput.enableAppendCalls)

        # No os.path.exists() fallback any more: a missing logical
        # attribute always means "create fresh", never "reopen a raw
        # on-disk file by path" (that path isn't necessarily the
        # authoritative backend under a PostgreSQL-backed compatibility
        # bridge).
        outputSet, isNew = loader(
            Protocol(),
            FreshOutputSetProbe,
            'outputParticles.sqlite',
            'missingOutput',
        )

        self.assertTrue(isNew)
        self.assertEqual('/tmp/outputParticles.sqlite', outputSet.filename)
        self.assertEqual(FreshOutputSetProbe.STREAM_OPEN, outputSet.streamState)

import unittest
from unittest.mock import Mock

from xmipp3.protocols.protocol_screen_particles import XmippProtScreenParticles


class TestXmippScreenParticlesFinalizationRegression(unittest.TestCase):

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        class _Harness:
            finished = True

            def __init__(self):
                self._prepareStreamingGenerator = Mock()
                self._checkNewInput = Mock()
                self._checkNewOutput = Mock()
                self._insertFunctionStep = Mock(return_value=1)
                self.createOutputStep = Mock()

        protocol = _Harness()

        XmippProtScreenParticles.stepsGeneratorStep(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()
        protocol._insertFunctionStep.assert_called_once()

