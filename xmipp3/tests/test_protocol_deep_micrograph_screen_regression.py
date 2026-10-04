# **************************************************************************
# *
# * Regression tests for Deep Micrograph Cleaner streaming/resume recovery.
# *
# **************************************************************************

import json
import unittest
from unittest.mock import Mock, patch

from pyworkflow.object import Set

from xmipp3.protocols import protocol_deep_micrograph_screen as deep_screen


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

    def loadAllProperties(self):
        pass

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

    def _getFinishedProcessedMicKeys(self):
        return {str(objId) for objId in self.processedIds}

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

    def _streamingSleepOnWait(self):
        self.events.append(('sleep',))

    def debug(self, message):
        pass

    def info(self, message):
        pass


class _InputHarness:
    def __init__(self):
        self.newDeps = []
        self.loaded = 0
        self.updated = 0
        self.micDict = {}
        self._pendingMicIds = set()
        self._otherIdByCoordId = {}

    def _loadInputList(self):
        self.loaded += 1
        return {'mic2': _Mic(2)}

    def _insertNewMicsSteps(self, mics):
        return [mic.getObjId() + 100 for mic in mics]

    def _micsOther(self):
        return False

    def updateSteps(self):
        self.updated += 1


class TestXmippDeepMicrographScreenRegression(unittest.TestCase):
    def testCheckNewInputAlwaysReloadsFreshSnapshot(self):
        protocol = _InputHarness()
        deep_screen.XmippProtDeepMicrographScreen._checkNewInput(protocol)
        self.assertEqual(1, protocol.loaded)
        self.assertEqual([102], protocol.newDeps)
        self.assertEqual(1, protocol.updated)

    def testFinishedRequiresAllPickedMicrographs(self):
        # The sleep-between-polls responsibility now lives in
        # stepsGeneratorStep's own loop, not inside _checkNewOutput - a
        # not-yet-finished check with nothing new to publish is a no-op.
        protocol = _Harness([1], [1], [1], streamClosed=True, allMicsProcessed=False)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertFalse(protocol.finished)
        self.assertEqual([], protocol.events)

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
        self.assertEqual([], protocol.events)

    def testFinishedReplayClosesExistingOutput(self):
        protocol = _Harness([1], [1], [1], streamClosed=True, allMicsProcessed=True)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('output', Set.STREAM_CLOSED)], protocol.events)

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        protocol = _InputHarness()
        protocol.finished = True
        protocol._prepareStreamingGenerator = Mock()
        protocol._checkNewInput = Mock()
        protocol._checkNewOutput = Mock()
        protocol._insertFunctionStep = Mock(return_value=1)
        protocol.createOutputStep = Mock()

        deep_screen.XmippProtDeepMicrographScreen.stepsGeneratorStep(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()
        protocol._insertFunctionStep.assert_called_once()

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
        # should be skipped, logged, and the rest of the batch still
        # processed normally. No DONE marker file is involved any more -
        # redundant recomputation is guarded by each real output
        # artifact's own idempotency check instead.
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
                self.convertedMics = []
                self.maskComputedFor = None

            def _convertCoordinates(self, mic, coordList):
                self.convertedMics.append(mic.getObjId())

            def _computeMaskForMicrographList(self, micList, *args):
                self.maskComputedFor = [mic.getObjId() for mic in micList]

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
        self.assertEqual([1, 2], harness.convertedMics)
        self.assertEqual([1, 2], harness.maskComputedFor)


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

    def getRunMode(self):
        from pyworkflow.protocol.constants import MODE_RESUME
        return MODE_RESUME

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


class _FakeMic:
    def __init__(self, objId, micName=None):
        self._objId = objId
        self._micName = micName or ('mic_%03d' % objId)

    def getObjId(self):
        return self._objId

    def getMicName(self):
        return self._micName

    def copyObjId(self, other):
        self._objId = other.getObjId()

    def clone(self):
        clone = _FakeMic(self._objId, self._micName)
        return clone


class _FakeMicSet:
    """Stands in for a real SetOfMicrographs, tracking which query
    mechanisms get used so tests can assert discovery is incremental
    (id > watermark / id IN (...)) and never a full scan or a
    filename-based reconstruction."""

    def __init__(self, mics, streamClosed=False):
        self._mics = {mic.getObjId(): mic for mic in mics}
        self._streamClosed = streamClosed
        self.uniqueCalls = []

    def loadAllProperties(self):
        pass

    def getUniqueValues(self, field, where=None):
        self.uniqueCalls.append((field, where))
        ids = sorted(self._mics.keys())
        if where:
            threshold = int(where.split('>')[1].strip())
            ids = [i for i in ids if i > threshold]
        return ids

    def iterItems(self, orderBy=None, direction=None, where=None):
        if where and where.startswith('id IN'):
            idsStr = where[where.index('(') + 1: where.index(')')]
            wanted = {int(x) for x in idsStr.split(',')}
        else:
            wanted = set(self._mics.keys())
        for i in sorted(wanted):
            if i in self._mics:
                yield self._mics[i]

    def isStreamClosed(self):
        return self._streamClosed

    def getFileName(self):
        raise AssertionError(
            "Mic discovery must not reconstruct a Set from a raw filename."
        )

    def close(self):
        pass


class _FakePointer:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeCoords:
    def __init__(self, micSetPointer):
        self._micSetPointer = micSetPointer

    def getMicrographs(self, asPointer=False):
        return self._micSetPointer if asPointer else self._micSetPointer.get()


class _FakeCoordItem:
    def __init__(self, micId):
        self._micId = micId

    def clone(self):
        return _FakeCoordItem(self._micId)


class _FakeCoordSet:
    def __init__(self, coordsByMicId, streamClosed=False):
        self._coordsByMicId = coordsByMicId
        self._streamClosed = streamClosed
        self.iterCalls = []
        self.closeCalls = 0

    def loadAllProperties(self):
        pass

    def getUniqueValues(self, field, where=None):
        if field == '_micName':
            return sorted(
                'mic_%03d' % micId
                for micId, coords in self._coordsByMicId.items()
                if coords
            )
        raise AssertionError('Unexpected getUniqueValues field: %s' % field)

    def iterItems(self, where=None):
        micId = int(where.split('=')[1])
        self.iterCalls.append(micId)
        return iter(self._coordsByMicId.get(micId, []))

    def isStreamClosed(self):
        return self._streamClosed

    def getFileName(self):
        raise AssertionError(
            "Coordinate loading must not reconstruct a Set from a raw filename."
        )

    def close(self):
        self.closeCalls += 1


class _DeepScreenInputHarness(deep_screen.XmippProtDeepMicrographScreen):
    """Exercises the real _loadInputList/_loadInputCoords/_checkNewInput/
    _areAllMicsProcessed against fake logical Sets, without instantiating a
    real Protocol (same no-super().__init__() pattern as
    _BatchClosureHarness above) - subclassing the real protocol, rather
    than just the XmippStreamingBase mixin, so internal self.xxx() calls
    inside those methods (e.g. _loadInputList calling
    self._loadInputCoords) resolve correctly."""

    def __init__(self, coordMics, coordsByMicId, micsOther=False,
                 otherMics=None, coordsStreamClosed=False,
                 micsStreamClosed=False, otherStreamClosed=False):
        self._micsWatermark = 0
        self._pendingMicIds = set()
        self._otherMicsWatermark = 0
        self._pendingOtherMicIds = set()
        self._otherIdByCoordId = {}
        self.micDict = {}
        self.coordDict = {}
        self.newDeps = []

        self.coordMicsSet = _FakeMicSet(coordMics, streamClosed=micsStreamClosed)
        self._coords = _FakeCoords(_FakePointer(self.coordMicsSet))
        self.coordSet = _FakeCoordSet(coordsByMicId, streamClosed=coordsStreamClosed)
        self.inputCoordinates = _FakePointer(self.coordSet)

        self._micsOtherFlag = micsOther
        if micsOther:
            self.otherMicsSet = _FakeMicSet(otherMics or [], streamClosed=otherStreamClosed)
            self.inputMicrographs = _FakePointer(self.otherMicsSet)

        # Controlled per test: whether _insertNewMics' batching would have
        # actually folded the candidates into self.micDict this round
        # (simulating pwem's own batch-size gating).
        self.scheduleAll = True
        self.insertNewMicsStepsCalls = []

    def getCoords(self):
        return self._coords

    def _micsOther(self):
        return self._micsOtherFlag

    def _insertNewMicsSteps(self, mics):
        mics = list(mics)
        self.insertNewMicsStepsCalls.append([m.getMicName() for m in mics])
        if self.scheduleAll:
            for mic in mics:
                self.micDict[mic.getMicName()] = mic
        return [1]

    def updateSteps(self):
        pass


class TestXmippDeepMicrographScreenInputDiscovery(unittest.TestCase):
    def testLoadInputListDiscoversOnlyAboveWatermarkNotFullScan(self):
        mics = [_FakeMic(1), _FakeMic(2), _FakeMic(3)]
        coords = {
            1: [_FakeCoordItem(1)],
            2: [_FakeCoordItem(2)],
            3: [_FakeCoordItem(3)],
        }
        harness = _DeepScreenInputHarness(mics, coords)

        micDict = deep_screen.XmippProtDeepMicrographScreen._loadInputList(harness)

        self.assertEqual({'mic_001', 'mic_002', 'mic_003'}, set(micDict.keys()))
        self.assertEqual(3, harness._micsWatermark)
        self.assertEqual(
            [('id', 'id > 0')],
            harness.coordMicsSet.uniqueCalls,
            "Discovery must query only ids above the watermark, not scan "
            "everything.",
        )
        self.assertEqual(
            [1, 2, 3],
            sorted(harness.coordSet.iterCalls),
            "Coordinates must be fetched with one targeted per-mic query, "
            "not a full scan of the coordinates Set.",
        )

    def testLoadInputListKeepsUndispatchedMicsPendingAcrossPolls(self):
        # Regression test: a mic discovered via the watermark but not yet
        # folded into self.micDict (because pwem's batch size isn't full
        # yet) must still be retried on the next poll, exactly like the
        # old full-rescan behavior naturally did - without re-scanning
        # from scratch once the watermark has moved past it.
        mics = [_FakeMic(1), _FakeMic(2)]
        coords = {1: [_FakeCoordItem(1)], 2: [_FakeCoordItem(2)]}
        harness = _DeepScreenInputHarness(mics, coords)
        harness.scheduleAll = False

        deep_screen.XmippProtDeepMicrographScreen._checkNewInput(harness)

        self.assertEqual({}, harness.micDict)
        self.assertEqual({1, 2}, harness._pendingMicIds)
        self.assertEqual(2, harness._micsWatermark)

        harness.coordMicsSet.uniqueCalls.clear()
        deep_screen.XmippProtDeepMicrographScreen._checkNewInput(harness)

        self.assertEqual(
            [('id', 'id > 2')],
            harness.coordMicsSet.uniqueCalls,
            "The watermark must not reset to re-scan from the start just "
            "because some mics are still pending.",
        )
        self.assertEqual(
            2,
            len(harness.insertNewMicsStepsCalls[-1]),
            "The still-pending mics must be retried, not dropped.",
        )

    def testMicsOtherCrossReferencePrunesPendingOnlyWhenCoordSideScheduled(self):
        coordMics = [_FakeMic(10, micName='mic_a'), _FakeMic(11, micName='mic_b')]
        otherMics = [_FakeMic(90, micName='mic_a'), _FakeMic(91, micName='mic_b')]
        coords = {10: [_FakeCoordItem(10)], 11: [_FakeCoordItem(11)]}
        harness = _DeepScreenInputHarness(
            coordMics, coords, micsOther=True, otherMics=otherMics,
        )
        harness.scheduleAll = False

        deep_screen.XmippProtDeepMicrographScreen._checkNewInput(harness)

        self.assertEqual({}, harness.micDict)
        self.assertEqual({10, 11}, harness._pendingMicIds)
        self.assertEqual({90, 91}, harness._pendingOtherMicIds)
        self.assertEqual({10: 90, 11: 91}, harness._otherIdByCoordId)

        harness.scheduleAll = True
        deep_screen.XmippProtDeepMicrographScreen._checkNewInput(harness)

        self.assertEqual({'mic_a', 'mic_b'}, set(harness.micDict.keys()))
        self.assertEqual(
            10, harness.micDict['mic_a'].getObjId(),
            "The scheduled mic must carry the coordinates-side id once "
            "matched, even though it originates from the other mics Set.",
        )
        self.assertEqual(set(), harness._pendingMicIds)
        self.assertEqual(set(), harness._pendingOtherMicIds)
        self.assertEqual({}, harness._otherIdByCoordId)

    def testAreAllMicsProcessedUsesPointerNotFilename(self):
        mics = [_FakeMic(1), _FakeMic(2)]
        coords = {1: [_FakeCoordItem(1)], 2: [_FakeCoordItem(2)]}
        harness = _DeepScreenInputHarness(mics, coords)
        harness.micDict = {'mic_001': _FakeMic(1), 'mic_002': _FakeMic(2)}

        result = deep_screen.XmippProtDeepMicrographScreen._areAllMicsProcessed(harness)

        self.assertTrue(result)
        self.assertEqual(1, harness.coordSet.closeCalls)

    def testGetFinishedProcessedMicKeysOnlyCountsFinishedExtractSteps(self):
        class _FakeFuncName:
            def __init__(self, name):
                self._name = name

            def get(self):
                return self._name

        class _FakeArgsStr:
            def __init__(self, value):
                self._value = value

            def get(self, default=None):
                return self._value

        class _FakeStep:
            def __init__(self, funcName, args, finished=True):
                self.funcName = _FakeFuncName(funcName)
                self.argsStr = _FakeArgsStr(json.dumps(args))
                self._finished = finished

            def isFinished(self):
                return self._finished

        harness = Mock()
        harness._iterKnownStreamingSteps = lambda: [
            _FakeStep('extractMicrographListStepOwn', [['mic_001', 'mic_002']]),
            _FakeStep('extractMicrographListStepOwn', [['mic_003']], finished=False),
            _FakeStep('someOtherStep', [['mic_999']]),
        ]

        keys = deep_screen.XmippProtDeepMicrographScreen._getFinishedProcessedMicKeys(harness)

        self.assertEqual({'mic_001', 'mic_002'}, keys)


class _RefreshRequiredOutputCoords:
    def __init__(self, micIds):
        self.micIds = set(micIds)
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True

    def getSize(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted coordinates must be refreshed before reading their size.'
            )
        return len(self.micIds)

    def getUniqueValues(self, attr):
        if not self.loaded:
            raise AssertionError(
                'Persisted coordinates must be refreshed before reading their micrograph ids.'
            )
        if attr != '_micId':
            raise AssertionError('Unexpected attribute: %s' % attr)
        return list(self.micIds)


class TestXmippDeepMicrographScreenLogicalOutputRestore(unittest.TestCase):
    def testGetOutputMicIdsRefreshesPersistedLogicalOutput(self):
        outputCoords = _RefreshRequiredOutputCoords([3, 7])

        class _Harness:
            def getOutput(self):
                return outputCoords

        protocol = _Harness()

        micIds = deep_screen.XmippProtDeepMicrographScreen._getOutputMicIds(protocol)

        self.assertEqual({3, 7}, micIds)
        self.assertTrue(outputCoords.loaded)
