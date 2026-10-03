# **************************************************************************
# *
# * Regression tests for automatic picking streaming/resume recovery.
# *
# **************************************************************************

import unittest
from unittest.mock import Mock

from pyworkflow.object import Set

from xmipp3.protocols import protocol_particle_pick_automatic as auto_pick
from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase


class _FakeMic:
    def __init__(self, objId, fileName=None):
        self.objId = objId
        self.fileName = fileName or '/tmp/mic_%03d.mrc' % objId

    def getObjId(self):
        return self.objId

    def getFileName(self):
        return self.fileName

    def getMicName(self):
        return 'mic_%03d' % self.objId

    def clone(self):
        return _FakeMic(self.objId, self.fileName)

    def strId(self):
        return str(self.objId)


class _FakeMicSet:
    """Mimics only what _loadLogicalSet/_discoverIdsAfter/
    _loadLogicalSetItemsByIds need - no getFileName()/filename
    reconstruction allowed."""

    def __init__(self, ids, streamClosed=False):
        self._ids = set(ids)
        self._items = {objId: _FakeMic(objId) for objId in ids}
        self._streamClosed = streamClosed
        self.closed = False

    def loadAllProperties(self):
        pass

    def getUniqueValues(self, attr, where=None):
        if where is None:
            return sorted(self._ids)
        threshold = int(where.split('>')[1].strip())
        return sorted(itemId for itemId in self._ids if itemId > threshold)

    def getSize(self):
        return len(self._ids)

    def isStreamClosed(self):
        return self._streamClosed

    def getItem(self, _, itemId):
        return self._items.get(itemId)

    def close(self):
        self.closed = True


class _FakePointer:
    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj


class _OutputCoords:
    def __init__(self, micIds=None):
        self.micIds = set(micIds or [])

    def getSize(self):
        return len(self.micIds)

    def getUniqueValues(self, attr):
        return list(self.micIds)


class _InputHarness(XmippStreamingBase):
    def __init__(self, micSet):
        self._micSet = micSet
        self.micDict = {}
        self._micsWatermark = 0
        self._pendingMicIds = set()
        self.newDeps = []
        self.updated = 0

    def getInputMicrographsPointer(self):
        return _FakePointer(self._micSet)

    def _loadInputList(self):
        return auto_pick.XmippParticlePickingAutomatic._loadInputList(self)

    def _insertNewMicsSteps(self, mics):
        mics = list(mics)
        for mic in mics:
            self.micDict[mic.getMicName()] = mic
        return [mic.getObjId() + 100 for mic in mics]

    def updateSteps(self):
        self.updated += 1


class _PickHarness:
    def __init__(self):
        self.micsToPick = auto_pick.MICS_OTHER
        self.numberOfThreads = 3
        self.jobs = []

    def _getExtraPath(self, name=''):
        return '/tmp/' + name

    def _getBoxSize(self):
        return 128

    def runJob(self, program, args):
        self.jobs.append((program, args))


class _OutputHarness:
    def __init__(self, micIds, processedIds, outputIds, streamClosed=False):
        self.events = []
        self.micDict = {mic.getMicName(): mic for mic in [_FakeMic(objId) for objId in micIds]}
        self.processedKeys = {'mic_%03d' % objId for objId in processedIds}
        self.outputCoordinates = _OutputCoords(outputIds) if outputIds is not None else None
        self.streamClosed = streamClosed
        self.finished = False

    def _getFinishedProcessedMicKeys(self):
        return set(self.processedKeys)

    def _getOutputMicIds(self):
        return auto_pick.XmippParticlePickingAutomatic._getOutputMicIds(self)

    def _updateOutputCoordSet(self, mics, streamMode):
        ids = [mic.getObjId() for mic in mics]
        self.events.append(('output', ids, streamMode))
        if self.outputCoordinates is None:
            self.outputCoordinates = _OutputCoords()
        self.outputCoordinates.micIds.update(ids)
        return mics

    def _updateStreamState(self, streamMode):
        self.events.append(('stream', streamMode))

    def _streamingSleepOnWait(self):
        self.events.append(('sleep',))


class TestXmippAutomaticPickingRegression(unittest.TestCase):
    def testCheckNewInputDiscoversOnlyAboveWatermarkAndSchedulesNewMics(self):
        # Regression test: input discovery must use the watermark/pending-id
        # mechanism (XmippStreamingBase) instead of pwem's
        # _loadInputList/_loadSet, which reconstructs a fresh Set from
        # inputSet.getFileName() and does a full Python-side scan every poll.
        micSet = _FakeMicSet([1, 2], streamClosed=True)
        protocol = _InputHarness(micSet)

        auto_pick.XmippParticlePickingAutomatic._checkNewInput(protocol)

        self.assertEqual(protocol._micsWatermark, 2)
        self.assertEqual(sorted(protocol.micDict), ['mic_001', 'mic_002'])
        self.assertEqual(protocol.newDeps, [101, 102])
        self.assertEqual(protocol.updated, 1)
        self.assertEqual(protocol._pendingMicIds, set())

        # Nothing new on the next poll: no duplicate scheduling.
        auto_pick.XmippParticlePickingAutomatic._checkNewInput(protocol)
        self.assertEqual(protocol.newDeps, [101, 102])

    def testPickMicrographReconstructsBoxSize(self):
        protocol = _PickHarness()
        auto_pick.XmippParticlePickingAutomatic._pickMicrograph(protocol, _FakeMic(1))
        self.assertEqual(1, len(protocol.jobs))
        self.assertIn('--particleSize 128', protocol.jobs[0][1])
        self.assertFalse(hasattr(protocol, 'boxSize'))

    def testPickMicrographListSkipsFailingMicrographAndKeepsProcessingOthers(self):
        # Regression test: with streamingBatchSize > 1, a single
        # corrupted/failing micrograph must not crash the whole batch
        # (and hence the whole protocol) - the base
        # pwem._pickMicrographList loop has no per-mic isolation.
        class _BatchPickHarness:
            def __init__(self):
                self.errors = []
                self.picked = []

            def error(self, msg):
                self.errors.append(msg)

            def _pickMicrograph(self, mic, *args):
                self.picked.append(mic.getObjId())
                if mic.getObjId() == 2:
                    raise ValueError("corrupted micrograph")

        protocol = _BatchPickHarness()
        micList = [_FakeMic(1), _FakeMic(2), _FakeMic(3)]

        auto_pick.XmippParticlePickingAutomatic._pickMicrographList(
            protocol, micList,
        )

        self.assertEqual([1, 2, 3], protocol.picked)
        self.assertEqual(1, len(protocol.errors))

    def testOutputIsPersistedFromRealOutputSetNotSidecar(self):
        # Regression test: which mics still need to be flushed to the output
        # Set must come from the real, persisted outputCoordinates
        # (_getOutputMicIds) and from the persisted step graph
        # (_getFinishedProcessedMicKeys), never from a DONE/mic_NNNNNN.TXT
        # marker file.
        protocol = _OutputHarness([1, 2], [1, 2], [1])
        auto_pick.XmippParticlePickingAutomatic._checkNewOutput(protocol)
        self.assertEqual([('output', [2], Set.STREAM_OPEN)], protocol.events)

    def testNoActionWhenAllProcessedAlreadyPersistedButNotFinished(self):
        protocol = _OutputHarness([1], [1], [1])
        auto_pick.XmippParticlePickingAutomatic._checkNewOutput(protocol)
        self.assertEqual([('sleep',)], protocol.events)

    def testFinishedReplayClosesExistingOutput(self):
        protocol = _OutputHarness([1], [1], [1], streamClosed=True)
        auto_pick.XmippParticlePickingAutomatic._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('stream', Set.STREAM_CLOSED)], protocol.events)


class TestXmippParticlePickingAutomaticFinalizationRegression(unittest.TestCase):

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

        auto_pick.XmippParticlePickingAutomatic.stepsGeneratorStep(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()
        protocol._insertFunctionStep.assert_called_once()


if __name__ == '__main__':
    unittest.main()
