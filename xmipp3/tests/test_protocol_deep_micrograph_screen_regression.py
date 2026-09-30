# **************************************************************************
# *
# * Regression tests for Deep Micrograph Cleaner streaming/resume recovery.
# *
# **************************************************************************

import os
import unittest
from unittest.mock import Mock, patch

from pyworkflow.object import Set

from xmipp3.protocols import protocol_deep_micrograph_screen as deep_screen
from xmipp3.tests.streaming_test_utils import OutputStep as _OutputStep


class _Mic:
    def __init__(self, objId):
        self.objId = objId

    def getObjId(self):
        return self.objId

    def strId(self):
        return str(self.objId)

    def getFileName(self):
        return 'mic_%03d.mrc' % self.objId

    def getMicName(self):
        return 'mic_%03d' % self.objId


class _OutputCoords:
    def __init__(self, micIds=None):
        self.micIds = set(micIds or [])
        self.enabledAppend = False

    def getSize(self):
        return len(self.micIds)

    def getUniqueValues(self, attr):
        return list(self.micIds)

    def enableAppend(self):
        self.enabledAppend = True


class _Harness:
    def __init__(self, micIds, processedIds, outputIds, streamClosed=False, allMicsProcessed=True):
        self.events = []
        self.micDict = {str(objId): _Mic(objId) for objId in micIds}
        self.processedIds = set(processedIds)
        self.outputCoords = _OutputCoords(outputIds) if outputIds is not None else None
        self.streamClosed = streamClosed
        self.allMicsProcessed = allMicsProcessed
        self.finished = False
        self.outputStep = _OutputStep()

    def _isMicDone(self, mic):
        return mic.getObjId() in self.processedIds

    def _readDoneList(self):
        raise AssertionError('DONE_all.TXT must not be used as durable state.')

    def _writeDoneList(self, mics):
        raise AssertionError('DONE_all.TXT must not be written.')

    def _isStreamClosed(self):
        return self.streamClosed

    def _areAllMicsProcessed(self):
        return self.allMicsProcessed

    def getOutput(self):
        return self.outputCoords

    def _getOutputMicIds(self):
        return deep_screen.XmippProtDeepMicrographScreen._getOutputMicIds(self)

    def _updateOutputCoordSet(self, micList, streamMode):
        return deep_screen.XmippProtDeepMicrographScreen._updateOutputCoordSet(self, micList, streamMode)

    def _getExtraPath(self, name):
        return '/tmp/' + name

    def _getScale(self):
        return 1

    def _updateOutputSet(self, outputName, outputCoords, streamMode):
        self.events.append(('output', streamMode))
        self.outputCoords = outputCoords

    def getOutputName(self):
        return 'outputCoordinates_Full'

    def _getFirstJoinStep(self):
        return self.outputStep

    def _streamingSleepOnWait(self):
        self.events.append(('sleep',))

    def debug(self, message):
        pass

    def info(self, message):
        pass


class _InputHarness:
    def __init__(self):
        self.outputStep = _OutputStep()
        self.loaded = 0
        self.updated = 0

    def _loadInputList(self):
        self.loaded += 1
        return {'mic2': _Mic(2)}

    def _getFirstJoinStep(self):
        return self.outputStep

    def _insertNewMicsSteps(self, mics):
        return [mic.getObjId() + 100 for mic in mics]

    def updateSteps(self):
        self.updated += 1


class TestXmippDeepMicrographScreenRegression(unittest.TestCase):
    def testCheckNewInputAlwaysReloadsFreshSnapshot(self):
        protocol = _InputHarness()
        deep_screen.XmippProtDeepMicrographScreen._checkNewInput(protocol)
        self.assertEqual(1, protocol.loaded)
        self.assertEqual([102], protocol.outputStep.prerequisites)
        self.assertEqual(1, protocol.updated)

    def testFinishedRequiresAllPickedMicrographs(self):
        protocol = _Harness([1], [1], [1], streamClosed=True, allMicsProcessed=False)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertFalse(protocol.finished)
        self.assertEqual([('sleep',)], protocol.events)

    def testOutputIsPersistedFromRealOutputSetNotSidecar(self):
        # Regression test: which mics still need to be flushed to the output
        # Set must come from the real, persisted outputCoords (_getOutputMicIds),
        # not from a DONE_all.TXT sidecar - _readDoneList/_writeDoneList raise
        # in this harness to prove they are never touched.
        protocol = _Harness([1, 2], [1, 2], [1])
        def fakeRead(outputDir, mics, outputCoords, scale=1):
            protocol.events.append(('read', [mic.getObjId() for mic in mics]))
            outputCoords.micIds.update(mic.getObjId() for mic in mics)

        with patch.object(deep_screen, 'readSetOfCoordinates', fakeRead):
            deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)

        self.assertEqual([('read', [2]), ('output', Set.STREAM_OPEN)], protocol.events)

    def testNoActionWhenAllProcessedAlreadyPersistedButNotFinished(self):
        protocol = _Harness([1], [1], [1], streamClosed=False)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertEqual([('sleep',)], protocol.events)

    def testFinishedReplayClosesExistingOutput(self):
        protocol = _Harness([1], [1], [1], streamClosed=True, allMicsProcessed=True)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('output', Set.STREAM_CLOSED)], protocol.events)

    def testFinishedStepsCheckIsNoOp(self):
        protocol = _InputHarness()
        protocol.finished = True
        protocol._checkNewInput = Mock()
        protocol._checkNewOutput = Mock()

        deep_screen.XmippProtDeepMicrographScreen._stepsCheck(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()

    def testZeroCoordinateMicrographIsNotReprocessedOnEveryPoll(self):
        # Regression test: a mic that is fully processed (DONE marker
        # exists) but contributes zero coordinates (all filtered out by
        # the model threshold) never appears in outputCoords' _micId
        # values. Without a persisted "known output mic ids" cache, it
        # would be treated as "new" and reprocessed on every subsequent
        # poll forever.
        protocol = _Harness([1], [1], outputIds=[])  # mic 1 processed, 0 coords published

        def fakeUpdate(micList, streamMode):
            protocol.events.append(('update', [mic.getObjId() for mic in micList]))

        protocol._updateOutputCoordSet = fakeUpdate

        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)

        updateCalls = [e for e in protocol.events if e[0] == 'update']
        self.assertEqual(
            1,
            len(updateCalls),
            "A processed mic with zero output coordinates must only be "
            "(attempted to be) published once, not reprocessed on "
            "every poll.",
        )


class TestXmippDeepMicrographScreenThumbnailIsolation(unittest.TestCase):
    def testExtractMicrographListStepOwnSkipsFailingThumbnailAndKeepsProcessingOthers(self):
        # Regression test: a single micrograph whose thumbnail
        # generation fails (e.g. a flat/saturated image) must not crash
        # the whole batch step (and hence the whole protocol) - it
        # should be skipped, logged, and the rest of the batch (and its
        # own DONE marker) must still be processed normally.
        import shutil
        import tempfile

        tmpDir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmpDir, ignore_errors=True))

        class _BoolValue:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class _ThumbnailHarness:
            def __init__(self):
                self.micDict = {'mic1': _Mic(1), 'mic2': _Mic(2)}
                self.coordDict = {1: [], 2: []}
                self.saveMicThumbnailWithMask = _BoolValue(True)
                self.errors = []
                self.thumbnailCalls = []

            def _convertCoordinates(self, mic, coordList):
                pass

            def isContinued(self):
                return False

            def _getMicDone(self, mic):
                return os.path.join(tmpDir, 'done_%d' % mic.getObjId())

            def _computeMaskForMicrographList(self, micList, *args):
                pass

            def _generateThumbnail(self, mic):
                self.thumbnailCalls.append(mic.getObjId())
                if mic.getObjId() == 1:
                    raise ValueError("corrupted thumbnail")

            def error(self, msg):
                self.errors.append(msg)

            def info(self, msg):
                pass

        harness = _ThumbnailHarness()

        deep_screen.XmippProtDeepMicrographScreen.extractMicrographListStepOwn(
            harness, ['mic1', 'mic2'],
        )

        self.assertEqual([1, 2], harness.thumbnailCalls)
        self.assertEqual(1, len(harness.errors))
        self.assertTrue(os.path.exists(os.path.join(tmpDir, 'done_1')))
        self.assertTrue(os.path.exists(os.path.join(tmpDir, 'done_2')))


if __name__ == '__main__':
    unittest.main()
class _BatchMic:
    def __init__(self, objId):
        self.objId = objId

    def getObjId(self):
        return self.objId

    def getMicName(self):
        return "mic_%03d" % self.objId


class _BatchClosureHarness(deep_screen.XmippProtDeepMicrographScreen):
    def __init__(self):
        self.coordsClosed = True
        self.micsClosed = False
        self.ctfsClosed = True
        self.initialIds = []
        self.micDict = {}

    def _getStreamingBatchSize(self):
        return 3


class TestXmippDeepMicrographScreenBatching(unittest.TestCase):
    def testPartialBatchWaitsForOtherMicrographsStream(self):
        protocol = _BatchClosureHarness()
        protocol.streamClosed = protocol._isStreamClosed()

        insertedBatches = []

        def insertSingle(mic, prerequisites, *args):
            raise AssertionError(
                "batchSize=3 must not insert single-micrograph steps"
            )

        def insertBatch(mics, prerequisites, *args):
            insertedBatches.append([mic.getMicName() for mic in mics])
            return 99

        deps = protocol._insertNewMics(
            [_BatchMic(1), _BatchMic(2)],
            lambda mic: mic.getMicName(),
            insertSingle,
            insertBatch,
        )

        self.assertEqual(
            [],
            insertedBatches,
            "A partial batch must remain pending while the alternate "
            "micrographs stream is still open.",
        )
        self.assertEqual([], deps)
        self.assertEqual({}, protocol.micDict)
class _IntValue:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _AutomaticBatchHarness:
    def __init__(self, pickedMics):
        self.streamingBatchSize = _IntValue(-1)
        self.pickedMics = pickedMics

    def isInStreaming(self):
        return False

    def _getPersistedAutomaticBatchSize(self):
        return None

    def _getNumPickedMics(self):
        return self.pickedMics


class TestXmippDeepMicrographScreenAutomaticBatching(unittest.TestCase):
    def testAutomaticStaticBatchGrowsAfterFirstBatch(self):
        protocol = _AutomaticBatchHarness(pickedMics=20)

        firstBatch = deep_screen.XmippProtDeepMicrographScreen._getStreamingBatchSize(
            protocol
        )
        secondBatch = deep_screen.XmippProtDeepMicrographScreen._getStreamingBatchSize(
            protocol
        )

        self.assertEqual(4, firstBatch)
        self.assertEqual(
            20,
            secondBatch,
            "Automatic static batching should use the initial batch of 4 only "
            "once, then switch to min(50, number of picked micrographs).",
        )
class _StoredValue:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _PersistedBatchStep:
    funcName = _StoredValue("extractMicrographListStepOwn")
    argsStr = _StoredValue(
        '[["mic_%03d", "mic_%03d", "mic_%03d", "mic_%03d", '
        '"mic_%03d", "mic_%03d", "mic_%03d", "mic_%03d", '
        '"mic_%03d", "mic_%03d", "mic_%03d", "mic_%03d", '
        '"mic_%03d", "mic_%03d", "mic_%03d", "mic_%03d"]]'
        % tuple(range(1, 17))
    )


class _AutomaticBatchResumeHarness:
    def __init__(self):
        self.streamingBatchSize = _IntValue(-1)

    def isInStreaming(self):
        # Simulate Continue after the input/output streams are no longer open.
        return False

    def isContinued(self):
        return True

    def loadSteps(self):
        return [_PersistedBatchStep()]

    def _getPersistedAutomaticBatchSize(self):
        return (
            deep_screen.XmippProtDeepMicrographScreen
            ._getPersistedAutomaticBatchSize(self)
        )

    def _getNumPickedMics(self):
        return 20


class TestXmippDeepMicrographScreenAutomaticBatchResume(unittest.TestCase):
    def testContinueRecoversPreviousAutomaticStreamingBatchSize(self):
        protocol = _AutomaticBatchResumeHarness()

        batchSize = deep_screen.XmippProtDeepMicrographScreen._getStreamingBatchSize(
            protocol
        )

        self.assertEqual(
            16,
            batchSize,
            "Continue must preserve the automatic streaming batch size already "
            "materialized in persisted batch steps instead of switching to "
            "the static first batch of 4.",
        )
