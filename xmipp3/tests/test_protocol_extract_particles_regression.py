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


class _OutputStep:
    def __init__(self):
        self.prerequisites = []
        self.status = None

    def addPrerequisites(self, *deps):
        self.prerequisites.extend(deps)

    def isWaiting(self):
        return True

    def setStatus(self, status):
        self.status = status


class _InputHarness:
    def __init__(self):
        self.outputStep = _OutputStep()
        self.updated = 0
        self.loadCalls = 0
        self.signature = ('initial',)

    def _getInputSignature(self):
        return self.signature

    def _loadInputList(self):
        self.loadCalls += 1
        return {'mic_002': _Mic(2)}

    def _getFirstJoinStep(self):
        return self.outputStep

    def _insertNewMicsSteps(self, mics):
        return [mic.getObjId() + 100 for mic in mics]

    def updateSteps(self):
        self.updated += 1


class _OutputHarness:
    def __init__(self, micIds, processedIds, doneIds, outputIds, streamClosed=False, allMicsProcessed=True):
        self.events = []
        self.micDict = {mic.getMicName(): mic for mic in [_Mic(objId) for objId in micIds]}
        self.processedIds = set(processedIds)
        self.doneIds = set(doneIds)
        self.outputParticles = _OutputParts(outputIds) if outputIds is not None else None
        self.streamClosed = streamClosed
        self.allMicsProcessed = allMicsProcessed
        self.allMicsProcessedCalls = 0
        self.finished = False
        self.outputStep = _OutputStep()

    def _isMicDone(self, mic):
        return mic.getObjId() in self.processedIds

    def _readDoneList(self):
        return list(self.doneIds)

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

    def _writeDoneList(self, mics):
        ids = [mic.getObjId() for mic in mics]
        self.doneIds.update(ids)
        self.events.append(('checkpoint', ids))

    def _getFirstJoinStep(self):
        return self.outputStep

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

    def testCheckNewInputSkipsUnchangedSnapshot(self):
        protocol = _InputHarness()
        extract_particles.XmippProtExtractParticles._checkNewInput(protocol)
        extract_particles.XmippProtExtractParticles._checkNewInput(protocol)
        self.assertEqual(1, protocol.loadCalls)
        self.assertEqual(1, protocol.updated)

        protocol.signature = ('changed',)
        extract_particles.XmippProtExtractParticles._checkNewInput(protocol)
        self.assertEqual(2, protocol.loadCalls)
        self.assertEqual(2, protocol.updated)

    def testOutputMicIdsAreLoadedOnce(self):
        protocol = _OutputHarness([1], [1], [1], [1])
        self.assertEqual({1}, extract_particles.XmippProtExtractParticles._getOutputMicIds(protocol))
        self.assertEqual({1}, extract_particles.XmippProtExtractParticles._getOutputMicIds(protocol))
        self.assertEqual(1, protocol.outputParticles.uniqueCalls)

    def testOpenStreamDoesNotScanAllPickedMicrographs(self):
        protocol = _OutputHarness([1], [1], [1], [1], streamClosed=False)
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertEqual(0, protocol.allMicsProcessedCalls)

    def testOutputIsPersistedBeforeCheckpoint(self):
        protocol = _OutputHarness([1, 2], [1, 2], [1], [1])
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertEqual([('output', [2], Set.STREAM_OPEN), ('checkpoint', [2])], protocol.events)

    def testReplayRepairsCheckpointWithoutDuplicateOutput(self):
        protocol = _OutputHarness([1], [1], [], [1])
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertEqual([('checkpoint', [1])], protocol.events)

    def testFinishedReplayClosesExistingOutput(self):
        protocol = _OutputHarness([1], [1], [1], [1], streamClosed=True)
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('output', [], Set.STREAM_CLOSED)], protocol.events)

    def testFinishedRequiresAllPickedMicrographs(self):
        protocol = _OutputHarness([1], [1], [1], [1], streamClosed=True, allMicsProcessed=False)
        extract_particles.XmippProtExtractParticles._checkNewOutput(protocol)
        self.assertFalse(protocol.finished)
        self.assertEqual([('sleep',)], protocol.events)


if __name__ == '__main__':
    unittest.main()
