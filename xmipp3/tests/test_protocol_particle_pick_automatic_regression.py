# **************************************************************************
# *
# * Regression tests for automatic picking streaming/resume recovery.
# *
# **************************************************************************

import unittest

from pyworkflow.object import Set

from xmipp3.protocols import protocol_particle_pick_automatic as auto_pick


class _Mic:
    def __init__(self, objId, fileName=None):
        self.objId = objId
        self.fileName = fileName or '/tmp/mic_%03d.mrc' % objId

    def getObjId(self):
        return self.objId

    def getFileName(self):
        return self.fileName

    def getMicName(self):
        return 'mic_%03d' % self.objId

    def strId(self):
        return str(self.objId)


class _OutputCoords:
    def __init__(self, micIds=None):
        self.micIds = set(micIds or [])

    def getSize(self):
        return len(self.micIds)

    def getUniqueValues(self, attr):
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
        self.streamClosed = False

    def _loadInputList(self):
        return {'mic_002': _Mic(2)}, True

    def _getFirstJoinStep(self):
        return self.outputStep

    def _insertNewMicsSteps(self, mics):
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
    def __init__(self, micIds, processedIds, doneIds, outputIds, streamClosed=False):
        self.events = []
        self.micDict = {mic.getMicName(): mic for mic in [_Mic(objId) for objId in micIds]}
        self.processedIds = set(processedIds)
        self.doneIds = set(doneIds)
        self.outputCoordinates = _OutputCoords(outputIds) if outputIds is not None else None
        self.streamClosed = streamClosed
        self.finished = False
        self.outputStep = _OutputStep()

    def _isMicDone(self, mic):
        return mic.getObjId() in self.processedIds

    def _readDoneList(self):
        return list(self.doneIds)

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

    def _writeDoneList(self, mics):
        ids = [mic.getObjId() for mic in mics]
        self.doneIds.update(ids)
        self.events.append(('checkpoint', ids))

    def _getFirstJoinStep(self):
        return self.outputStep

    def _streamingSleepOnWait(self):
        self.events.append(('sleep',))


class TestXmippAutomaticPickingRegression(unittest.TestCase):
    def testCheckNewInputAlwaysReloadsFreshSnapshot(self):
        protocol = _InputHarness()
        auto_pick.XmippParticlePickingAutomatic._checkNewInput(protocol)
        self.assertTrue(protocol.streamClosed)
        self.assertEqual([102], protocol.outputStep.prerequisites)
        self.assertEqual(1, protocol.updated)

    def testPickMicrographReconstructsBoxSize(self):
        protocol = _PickHarness()
        auto_pick.XmippParticlePickingAutomatic._pickMicrograph(protocol, _Mic(1))
        self.assertEqual(1, len(protocol.jobs))
        self.assertIn('--particleSize 128', protocol.jobs[0][1])
        self.assertFalse(hasattr(protocol, 'boxSize'))

    def testOutputIsPersistedBeforeCheckpoint(self):
        protocol = _OutputHarness([1, 2], [1, 2], [1], [1])
        auto_pick.XmippParticlePickingAutomatic._checkNewOutput(protocol)
        self.assertEqual([('output', [2], Set.STREAM_OPEN), ('checkpoint', [2])], protocol.events)

    def testReplayRepairsCheckpointWithoutDuplicateOutput(self):
        protocol = _OutputHarness([1], [1], [], [1])
        auto_pick.XmippParticlePickingAutomatic._checkNewOutput(protocol)
        self.assertEqual([('checkpoint', [1])], protocol.events)

    def testFinishedReplayClosesExistingOutput(self):
        protocol = _OutputHarness([1], [1], [1], [1], streamClosed=True)
        auto_pick.XmippParticlePickingAutomatic._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('stream', Set.STREAM_CLOSED)], protocol.events)


if __name__ == '__main__':
    unittest.main()
