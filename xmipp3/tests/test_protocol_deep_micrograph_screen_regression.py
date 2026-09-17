# **************************************************************************
# *
# * Regression tests for Deep Micrograph Cleaner streaming/resume recovery.
# *
# **************************************************************************

import unittest
from unittest.mock import patch

from pyworkflow.object import Set

from xmipp3.protocols import protocol_deep_micrograph_screen as deep_screen


class _Mic:
    def __init__(self, objId):
        self.objId = objId

    def getObjId(self):
        return self.objId

    def strId(self):
        return str(self.objId)


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


class _Harness:
    def __init__(self, micIds, processedIds, doneIds, outputIds, streamClosed=False, allMicsProcessed=True):
        self.events = []
        self.micDict = {str(objId): _Mic(objId) for objId in micIds}
        self.processedIds = set(processedIds)
        self.doneIds = set(doneIds)
        self.outputCoords = _OutputCoords(outputIds) if outputIds is not None else None
        self.streamClosed = streamClosed
        self.allMicsProcessed = allMicsProcessed
        self.finished = False
        self.outputStep = _OutputStep()

    def _isMicDone(self, mic):
        return mic.getObjId() in self.processedIds

    def _readDoneList(self):
        return list(self.doneIds)

    def _isStreamClosed(self):
        return self.streamClosed

    def _areAllMicsProcessed(self):
        return self.allMicsProcessed

    def getOutput(self):
        return self.outputCoords

    def _getOutputMicIds(self):
        return deep_screen.XmippProtDeepMicrographScreen._getOutputMicIds(self)

    def _getExtraPath(self, name):
        return '/tmp/' + name

    def _getScale(self):
        return 1

    def _updateOutputSet(self, outputName, outputCoords, streamMode):
        self.events.append(('output', streamMode))
        self.outputCoords = outputCoords

    def getOutputName(self):
        return 'outputCoordinates_Full'

    def _writeDoneList(self, mics):
        ids = [mic.getObjId() for mic in mics]
        self.doneIds.update(ids)
        self.events.append(('checkpoint', ids))

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
        protocol = _Harness([1], [1], [1], [1], streamClosed=True, allMicsProcessed=False)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertFalse(protocol.finished)
        self.assertEqual([('sleep',)], protocol.events)

    def testOutputIsPersistedBeforeCheckpoint(self):
        protocol = _Harness([1, 2], [1, 2], [1], [1])
        def fakeRead(outputDir, mics, outputCoords, scale=1):
            protocol.events.append(('read', [mic.getObjId() for mic in mics]))
            outputCoords.micIds.update(mic.getObjId() for mic in mics)

        with patch.object(deep_screen, 'readSetOfCoordinates', fakeRead):
            deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)

        self.assertEqual([('read', [2]), ('output', Set.STREAM_OPEN), ('checkpoint', [2])], protocol.events)

    def testReplayRepairsCheckpointWithoutDuplicateOutput(self):
        protocol = _Harness([1], [1], [], [1])
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertEqual([('checkpoint', [1])], protocol.events)

    def testFinishedReplayClosesExistingOutput(self):
        protocol = _Harness([1], [1], [1], [1], streamClosed=True, allMicsProcessed=True)
        deep_screen.XmippProtDeepMicrographScreen._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('output', Set.STREAM_CLOSED)], protocol.events)


if __name__ == '__main__':
    unittest.main()
