# **************************************************************************
# *
# * The streaming generator must abandon its loop once the run has failed.
# *
# **************************************************************************

"""A failed step makes pyworkflow mark the protocol FAILED and the executor
break out of its own loop - and then join every running thread, the
generator's among them. A generator that keeps polling is never joined, so
the whole run hangs with nothing left to do.
"""

import threading
import unittest

import pyworkflow.protocol.constants as cons
from pyworkflow.object import String

from xmipp3.protocols.protocol_streaming_base import (
    XmippStreamingBase, XmippStreamingMoviesMixin)


class _StatusCarrier(XmippStreamingBase):
    """Carries the status attribute the way a real Protocol does."""

    def __init__(self):
        self.status = String()


class TestStreamingMustStopFlag(unittest.TestCase):
    def testRunningProtocolKeepsPolling(self):
        protocol = _StatusCarrier()
        protocol.status.set(cons.STATUS_RUNNING)

        self.assertFalse(protocol._streamingMustStop())

    def testFailedProtocolStopsTheGenerator(self):
        protocol = _StatusCarrier()
        protocol.status.set(cons.STATUS_FAILED)

        self.assertTrue(protocol._streamingMustStop())

    def testAbortedProtocolStopsTheGenerator(self):
        protocol = _StatusCarrier()
        protocol.status.set(cons.STATUS_ABORTED)

        self.assertTrue(protocol._streamingMustStop())

    def testUnsetStatusDoesNotStopTheGenerator(self):
        # A protocol whose status has not been written yet must not be
        # mistaken for a failed one.
        self.assertFalse(_StatusCarrier()._streamingMustStop())


class _GeneratorHarness(XmippStreamingBase):
    """Drives a real generator loop and fails it on the third poll."""

    FAIL_ON_POLL = 3

    def __init__(self, failAfterStart=True):
        self.status = String()
        self.status.set(cons.STATUS_RUNNING)
        self.finished = False
        self.finish = False
        self.newDeps = []
        self.polls = 0
        self._failAfterStart = failAfterStart

    def initializeParams(self):
        pass

    def _checkNewInput(self):
        self.polls += 1

        if self._failAfterStart and self.polls == self.FAIL_ON_POLL:
            # What pyworkflow does to this very object when a step fails.
            self.status.set(cons.STATUS_FAILED)

    def _checkNewOutput(self):
        pass

    def _streamingSleepOnWait(self):
        pass

    def createOutputStep(self):
        pass

    def _insertFunctionStep(self, *args, **kwargs):
        # The generator inserts its final step once the loop is over; the
        # harness must let that happen so the thread ends cleanly and the
        # assertion below is about the loop, not about a missing attribute.
        return 1


class TestGeneratorStopsOnFailure(unittest.TestCase):
    """The loop must come back, not spin for ever."""

    def _runGenerator(self, generatorFunc, protocol, timeout=5):
        thread = threading.Thread(target=generatorFunc, args=(protocol,),
                                  daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        return thread.is_alive()

    def testCtfConsensusGeneratorReturnsAfterFailure(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus)

        protocol = _GeneratorHarness()
        stillRunning = self._runGenerator(
            XmippProtCTFConsensus.stepsGeneratorStep, protocol)

        self.assertFalse(
            stillRunning,
            "The generator kept polling after the protocol failed; the "
            "executor would wait for this thread for ever.",
        )
        self.assertEqual(_GeneratorHarness.FAIL_ON_POLL, protocol.polls)


if __name__ == "__main__":
    unittest.main()


class _CoordDictHarness:
    """Drives the step that consumes a micrograph's coordinates."""

    def __init__(self, micCount=3):
        self.micDict = {}
        self.coordDict = {}
        self.converted = []

        for micId in range(1, micCount + 1):
            name = "mic_%03d" % micId
            self.micDict[name] = _FakeMic(micId, name)
            self.coordDict[micId] = ["coord"] * 200

    def _convertCoordinates(self, mic, coordList):
        self.converted.append(mic.getObjId())

    def _computeMaskForMicrographList(self, micList, *args):
        pass

    def info(self, *args, **kwargs):
        pass


class _FakeMic:
    def __init__(self, objId, micName):
        self._objId = objId
        self._micName = micName

    def getObjId(self):
        return self._objId

    def getMicName(self):
        return self._micName

    def getFileName(self):
        return "/data/%s.mrc" % self._micName


class TestCoordinatesAreReleasedAfterUse(unittest.TestCase):
    """coordDict must hold work in flight, not a record of the whole run.

    One list of cloned Coordinate objects per micrograph, kept for ever,
    is hundreds of millions of live objects on a million-micrograph run.
    """

    def testCoordinatesAreDroppedOnceWritten(self):
        from xmipp3.protocols.protocol_deep_micrograph_screen import (
            XmippProtDeepMicrographScreen)

        protocol = _CoordDictHarness(micCount=3)
        protocol.saveMicThumbnailWithMask = _NoValue()

        XmippProtDeepMicrographScreen.extractMicrographListStepOwn(
            protocol, ["mic_001", "mic_002", "mic_003"])

        self.assertEqual([1, 2, 3], protocol.converted)
        self.assertEqual(
            {}, protocol.coordDict,
            "Coordinates already written out must be released, or coordDict "
            "grows with the whole run.",
        )


class _NoValue:
    def get(self):
        return False


class _FinishedStep:
    def __init__(self, index, finished=True):
        self._index = index
        self._finished = finished

    def getIndex(self):
        return self._index

    def isFinished(self):
        return self._finished


class _InsertedStepsHarness(XmippStreamingBase):
    """Reproduces the order in which steps become runnable."""

    def __init__(self):
        self.insertedDict = {}
        self._steps = []

    def insertStepFor(self, itemId, finished=True):
        """What _insertNewMoviesSteps does: the step is runnable at once."""
        self._steps.append(_FinishedStep(len(self._steps) + 1, finished))
        self.insertedDict[itemId] = len(self._steps)

        return len(self._steps)


class TestFinishedStepIsNeverLost(unittest.TestCase):
    """A step is runnable before the generator records who owns it.

    One that finishes inside that window used to be dropped: its item was
    never published and the protocol never reached its end, polling for
    ever with the output incomplete.
    """

    def testItemFinishingBeforeItsOwnerIsRecorded(self):
        protocol = _InsertedStepsHarness()

        # First poll of the generator, which used to consume a one-shot
        # rescue pass.
        protocol._getFinishedInsertedIds()

        stepIndex = protocol.insertStepFor(7)
        # The worker finishes before _trackPendingInsertedIds runs.
        protocol._recordFinishedInsertedStep(stepIndex)
        protocol._trackPendingInsertedIds([7])

        self.assertIn(
            7, protocol._getFinishedInsertedIds(),
            "An item whose step finished before its owner was recorded "
            "must still be published, or the protocol never finishes.",
        )

    def testOwnerRecordedFirstStillWorks(self):
        protocol = _InsertedStepsHarness()
        protocol._getFinishedInsertedIds()

        stepIndex = protocol.insertStepFor(11)
        protocol._trackPendingInsertedIds([11])
        protocol._recordFinishedInsertedStep(stepIndex)

        self.assertIn(11, protocol._getFinishedInsertedIds())

    def testUnfinishedStepIsNotReportedAsDone(self):
        protocol = _InsertedStepsHarness()
        protocol._getFinishedInsertedIds()

        protocol.insertStepFor(13, finished=False)
        protocol._trackPendingInsertedIds([13])

        self.assertNotIn(13, protocol._getFinishedInsertedIds())

    def testAcknowledgedItemsAreDroppedFromTracking(self):
        # The sets must track work in flight, not the whole run.
        protocol = _InsertedStepsHarness()
        stepIndex = protocol.insertStepFor(21)
        protocol._trackPendingInsertedIds([21])
        protocol._recordFinishedInsertedStep(stepIndex)

        self.assertIn(21, protocol._getFinishedInsertedIds())

        protocol._acknowledgeFinishedInsertedIds([21])

        self.assertNotIn(21, protocol._getFinishedInsertedIds())


import os
import tempfile


class _CleanFolderHarness(XmippStreamingMoviesMixin):
    """Protocol whose project path contains a space, as users' often do."""

    def __init__(self, root):
        self._root = root
        self.warnings = []

    def _getTmpPath(self, *parts):
        return os.path.join(self._root, 'Mis Proyectos', 'Runs', 'tmp', *parts)

    def info(self, *args, **kwargs):
        pass

    def warning(self, message):
        self.warnings.append(message)


class TestMovieFolderCleanupIsSafe(unittest.TestCase):
    """pwem removes the folder with os.system('rm -rf %s').

    The path carries the project directory, so a project whose path has a
    space in it makes that command delete something else entirely while
    leaving the real folder behind.
    """

    def testFolderWithASpaceInThePathIsRemovedCorrectly(self):
        with tempfile.TemporaryDirectory() as root:
            protocol = _CleanFolderHarness(root)
            movieFolder = protocol._getTmpPath('movie_000001')
            os.makedirs(movieFolder)

            # A neighbour that the shell would have eaten.
            neighbour = os.path.join(root, 'Mis')
            os.makedirs(neighbour)
            keep = os.path.join(neighbour, 'do_not_delete.txt')
            open(keep, 'w').close()

            protocol._cleanMovieFolder(movieFolder)

            self.assertFalse(os.path.exists(movieFolder),
                             "The movie folder itself must be removed.")
            self.assertTrue(os.path.exists(keep),
                            "Nothing outside the movie folder may be touched.")

    def testFolderOutsideTheWorkspaceIsRefused(self):
        with tempfile.TemporaryDirectory() as root:
            protocol = _CleanFolderHarness(root)
            outsider = os.path.join(root, 'somewhere_else')
            os.makedirs(outsider)

            protocol._cleanMovieFolder(outsider)

            self.assertTrue(os.path.exists(outsider))
            self.assertTrue(protocol.warnings)


class TestMembershipUsesSets(unittest.TestCase):
    """Membership against a list is linear, and these run once per poll.

    Measured: one second per poll at 20k items, which extrapolates to
    tens of minutes per poll at a million.
    """

    def testTiltAnalysisBuildsASetBeforeTestingMembership(self):
        import inspect
        from xmipp3.protocols import protocol_tilt_analysis

        source = inspect.getsource(
            protocol_tilt_analysis.XmippProtTiltAnalysis._checkNewOutput)

        self.assertIn('doneIdSet = set(doneListIds)', source)
        self.assertIn('not in doneIdSet', source)

    def testMovieGainKeepsEstimatedIdsInASet(self):
        import inspect
        from xmipp3.protocols import protocol_movie_gain

        source = inspect.getsource(
            protocol_movie_gain.XmippProtMovieGain._restoreEstimatedIds)

        self.assertIn('set(', source)
        self.assertNotIn('sorted(', source)


class TestTemporaryFilesAreNamedNotGlobbed(unittest.TestCase):
    """Cleaning up with a prefix glob reaches other micrographs.

    "mic1" and "mic10" are extracted by parallel steps; finishing the
    first used to delete the second's working files mid-extraction.
    """

    def testPrefixGlobWouldDeleteAnotherMicrographsFiles(self):
        import pyworkflow.utils as pwutils

        with tempfile.TemporaryDirectory() as root:
            ours = [os.path.join(root, 'mic1_downsampled.xmp'),
                    os.path.join(root, 'mic1_noDust.xmp')]
            theirs = [os.path.join(root, 'mic10_downsampled.xmp'),
                      os.path.join(root, 'mic10.ctfParam')]

            for fn in ours + theirs:
                open(fn, 'w').close()

            # What the protocol does now: delete the files it created.
            for fn in ours:
                pwutils.cleanPath(fn)

            self.assertFalse(any(os.path.exists(fn) for fn in ours))
            self.assertTrue(
                all(os.path.exists(fn) for fn in theirs),
                "Another micrograph's working files must survive.",
            )

    def testExtractMicrographCleansByNameAndNotByPattern(self):
        import inspect
        from xmipp3.protocols import protocol_extract_particles

        source = inspect.getsource(
            protocol_extract_particles.XmippProtExtractParticles
            ._extractMicrograph)

        self.assertNotIn('cleanPattern', source)
        self.assertIn('for fn in micTmpFiles', source)
