# **************************************************************************
# *
# * Regression tests for extract particles streaming/resume recovery.
# *
# **************************************************************************

import unittest

from pyworkflow.object import Set

from xmipp3.protocols import protocol_extract_particles as extract_particles


class _Mic:
    def __init__(self, objId):
        self.objId = objId

    def getObjId(self):
        return self.objId

    def getMicName(self):
        return 'mic_%03d' % self.objId


class _Coord:
    def __init__(self, micId):
        self.micId = micId

    def getMicId(self):
        return self.micId

    def clone(self):
        return _Coord(self.micId)


class _CoordSet:
    def __init__(self, coords):
        self.coords = coords
        self.fullScans = 0
        self.indexedQueries = 0

    def iterItems(self, where=None):
        if where is None:
            self.fullScans += 1
            return iter(self.coords)

        self.indexedQueries += 1
        micId = int(where.split('=')[1])
        return (coord for coord in self.coords if coord.getMicId() == micId)


class _MicSetInfo:
    def __init__(self, size):
        self.size = size

    def getSize(self):
        return self.size


class _CoordsInfo:
    def __init__(self, micCount):
        self.mics = _MicSetInfo(micCount)

    def getMicrographs(self):
        return self.mics


class _CoordLoadHarness:
    def __init__(self, totalMicCount):
        self.coordDict = {}
        self.coords = _CoordsInfo(totalMicCount)

    def getCoords(self):
        return self.coords

    def _shouldBulkLoadCoords(self, micDict):
        return extract_particles.XmippProtExtractParticles._shouldBulkLoadCoords(self, micDict)


class _OutputParts:
    def __init__(self, micIds=None):
        self.micIds = set(micIds or [])
        self.uniqueCalls = 0

    def getSize(self):
        return len(self.micIds)

    def getUniqueValues(self, attr):
        self.uniqueCalls += 1
        return list(self.micIds)


class _InputHarness:
    def __init__(self):
        self.updated = 0
        self.loadCalls = 0
        self.micDict = {}
        self.newDeps = []
        self._pendingMicIds = set()
        self._otherIdByCoordId = {}
        self._pendingCtfIds = set()
        self._ctfIdByCoordId = {}

    def _loadInputList(self):
        self.loadCalls += 1
        return {'mic_002': _Mic(2)}

    def _insertNewMicsSteps(self, mics):
        return [mic.getObjId() + 100 for mic in mics]

    def _micsOther(self):
        return False

    def _useCTF(self):
        return False

    def updateSteps(self):
        self.updated += 1


class _OutputHarness:
    def __init__(self, micIds, processedIds, outputIds, streamClosed=False, allMicsProcessed=True):
        self.events = []
        self.micDict = {mic.getMicName(): mic for mic in [_Mic(objId) for objId in micIds]}
        self.processedIds = set(processedIds)
        self.outputParticles = _OutputParts(outputIds) if outputIds is not None else None
        self.streamClosed = streamClosed
        self.allMicsProcessed = allMicsProcessed
        self.allMicsProcessedCalls = 0
        self.finished = False

    def _getFinishedProcessedMicKeys(self):
        return {'mic_%03d' % objId for objId in self.processedIds}

    def _readDoneList(self):
        raise AssertionError('DONE_all.TXT must not be used as durable state.')

    def _writeDoneList(self, mics):
        raise AssertionError('DONE_all.TXT must not be written.')

    def _isStreamClosed(self):
        return self.streamClosed

    def _areAllMicsProcessed(self):
        self.allMicsProcessedCalls += 1
        return self.allMicsProcessed

    def _getOutputMicIds(self):
        return extract_particles.XmippProtExtractParticles._getOutputMicIds(self)

    def _updateOutputPartSet(self, mics, streamMode):
        ids = [mic.getObjId() for mic in mics]
        self.events.append(('output', ids, streamMode))
        if self.outputParticles is None:
            self.outputParticles = _OutputParts()
        self.outputParticles.micIds.update(ids)

    def _streamingSleepOnWait(self):
        self.events.append(('sleep',))


class TestXmippExtractParticlesRegression(unittest.TestCase):
    def testBulkCoordinateLoadUsesSingleScan(self):
        protocol = _CoordLoadHarness(120)
        micDict = {_Mic(objId).getMicName(): _Mic(objId) for objId in range(1, 101)}
        coordSet = _CoordSet([_Coord(objId) for objId in range(1, 101)] + [_Coord(999)])
        result = extract_particles.XmippProtExtractParticles._loadCoordsForMics(protocol, coordSet, micDict)
        self.assertEqual(100, len(result))
        self.assertEqual(1, coordSet.fullScans)
        self.assertEqual(0, coordSet.indexedQueries)
        self.assertEqual(100, len(protocol.coordDict))

    def testIncrementalCoordinateLoadKeepsIndexedQueries(self):
        protocol = _CoordLoadHarness(1000)
        micDict = {_Mic(objId).getMicName(): _Mic(objId) for objId in (10, 20, 30)}
        coordSet = _CoordSet([_Coord(10), _Coord(20), _Coord(30), _Coord(999)])
        result = extract_particles.XmippProtExtractParticles._loadCoordsForMics(protocol, coordSet, micDict)
        self.assertEqual(3, len(result))
        self.assertEqual(0, coordSet.fullScans)
        self.assertEqual(3, coordSet.indexedQueries)
        self.assertEqual(3, len(protocol.coordDict))

    def testCheckNewInputAlwaysReloadsFreshSnapshot(self):
        protocol = _InputHarness()

        extract_particles.XmippProtExtractParticles._checkNewInput(protocol)
        extract_particles.XmippProtExtractParticles._checkNewInput(protocol)

        self.assertEqual(
            2,
            protocol.loadCalls,
            "Streaming input must be refreshed from the logical Set on "
            "every check, not cached across calls.",
        )
        self.assertEqual(2, protocol.updated)

    def testOutputMicIdsAreLoadedOnce(self):
        protocol = _OutputHarness([1], [1], [1])
        self.assertEqual({1}, extract_particles.XmippProtExtractParticles._getOutputMicIds(protocol))
        self.assertEqual({1}, extract_particles.XmippProtExtractParticles._getOutputMicIds(protocol))
        self.assertEqual(1, protocol.outputParticles.uniqueCalls)

    def testOpenStreamDoesNotScanAllPickedMicrographs(self):
        protocol = _OutputHarness([1], [1], [1], streamClosed=False)
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertEqual(0, protocol.allMicsProcessedCalls)

    def testOutputIsPersistedFromRealOutputSetNotSidecar(self):
        # Regression test: which mics still need to be flushed to the output
        # Set must come from the real, persisted outputParticles
        # (_getOutputMicIds), not from a DONE_all.TXT sidecar -
        # _readDoneList/_writeDoneList raise in this harness to prove they
        # are never touched.
        protocol = _OutputHarness([1, 2], [1, 2], [1])
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertEqual([('output', [2], Set.STREAM_OPEN)], protocol.events)

    def testProcessedMicsIncludeMicsAlreadyInPersistedOutputEvenWithoutMarker(self):
        # Regression test: a mic already reflected in the persisted output
        # Set must still count as processed even if its worker DONE marker
        # file is missing - the real output Set is authoritative, exactly
        # what the DONE_all.TXT sidecar used to (redundantly) guarantee.
        protocol = _OutputHarness([1], [], [1], streamClosed=True, allMicsProcessed=True)
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('output', [], Set.STREAM_CLOSED)], protocol.events)

    def testFinishedReplayClosesExistingOutput(self):
        protocol = _OutputHarness([1], [1], [1], streamClosed=True)
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('output', [], Set.STREAM_CLOSED)], protocol.events)

    def testFinishedRequiresAllPickedMicrographs(self):
        # The sleep-between-polls responsibility now lives in
        # stepsGeneratorStep's own loop, not inside _checkNewOutput - a
        # not-yet-finished check with nothing new to publish is a no-op.
        protocol = _OutputHarness([1], [1], [1], streamClosed=True, allMicsProcessed=False)
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertFalse(protocol.finished)
        self.assertEqual([], protocol.events)

    def testExtractMicrographListSkipsFailingMicrographAndKeepsProcessingOthers(self):
        # Regression test: a single corrupted micrograph must not crash
        # the whole batch (and hence the whole protocol) - the base
        # pwem._extractMicrographList loop has no per-mic isolation.
        class _ExtractHarness:
            def __init__(self):
                self.errors = []
                self.extracted = []

            def error(self, msg):
                self.errors.append(msg)

            def _extractMicrograph(self, mic, *args):
                self.extracted.append(mic.getObjId())
                if mic.getObjId() == 2:
                    raise ValueError("corrupted micrograph")

        harness = _ExtractHarness()
        micList = [_Mic(1), _Mic(2), _Mic(3)]

        extract_particles.XmippProtExtractParticles._extractMicrographList(
            harness, micList,
        )

        self.assertEqual([1, 2, 3], harness.extracted)
        self.assertEqual(1, len(harness.errors))

    def testReadPartsFromMicsSkipsMicrographWhoseCoordinatesFailToParseAndKeepsProcessingOthers(self):
        # Regression test: a single micrograph whose coordinate data
        # fails to parse must not crash readPartsFromMics for the whole
        # batch, and its coordDict entry must still be released so it
        # doesn't linger and leak memory/cause stale re-processing.
        class _ReadPartsHarness:
            def __init__(self, coordDict):
                self.coordDict = coordDict
                self.errors = []

            def getBoxScale(self):
                return 1.0

            def _getPos(self, coord):
                if coord.micId == 2:
                    raise ValueError("corrupted coordinate")
                return (0, 0)

            def _getMicXmd(self, mic):
                return '/nonexistent/%d.xmd' % mic.getObjId()

            def error(self, msg):
                self.errors.append(msg)

        harness = _ReadPartsHarness({
            1: [_Coord(1)], 2: [_Coord(2)], 3: [_Coord(3)],
        })
        outputParts = _OutputParts()
        micList = [_Mic(1), _Mic(2), _Mic(3)]

        extract_particles.XmippProtExtractParticles.readPartsFromMics(
            harness, micList, outputParts,
        )

        self.assertEqual(1, len(harness.errors))
        self.assertEqual({}, harness.coordDict)


if __name__ == '__main__':
    unittest.main()
class _BatchExtractHarness(extract_particles.XmippProtExtractParticles):
    def __init__(self):
        self.coordsClosed = True
        self.micsClosed = True
        self.ctfsClosed = False
        self.initialIds = []
        self.micDict = {}

    def _getStreamingBatchSize(self):
        return 3


class TestXmippExtractParticlesBatching(unittest.TestCase):
    def testPartialBatchWaitsForCtfStream(self):
        protocol = _BatchExtractHarness()
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
            [_Mic(1), _Mic(2)],
            lambda mic: mic.getMicName(),
            insertSingle,
            insertBatch,
        )

        self.assertEqual(
            [],
            insertedBatches,
            "A partial extraction batch must remain pending while the required "
            "CTF stream is still open.",
        )
        self.assertEqual([], deps)
        self.assertEqual({}, protocol.micDict)

import unittest
from unittest.mock import Mock

from xmipp3.protocols.protocol_extract_particles import XmippProtExtractParticles


class TestXmippExtractParticlesFinalizationRegression(unittest.TestCase):

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        class _Harness:
            finished = True
            lenPartsSet = 0

            def __init__(self):
                self._checkNewInput = Mock()
                self._checkNewOutput = Mock()
                self._prepareStreamingGenerator = Mock()
                self._insertFunctionStep = Mock(return_value=1)
                self.createOutputStep = Mock()

        protocol = _Harness()

        XmippProtExtractParticles.stepsGeneratorStep(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()
        protocol._insertFunctionStep.assert_called_once()

