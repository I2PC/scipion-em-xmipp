import unittest
from unittest.mock import MagicMock, patch

from xmipp3.protocols.protocol_flexalign import XmippProtFlexAlign
from pwem.protocols import ProtAlignMovies


class TestFlexAlignScientificFailureRegression(unittest.TestCase):
    """Regression coverage for scientific failures inside movie workers."""

    def testAFailedMovieIsLeftOutWithoutFailingTheStep(self):
        # Raised, it would fail the step and the whole run with it.
        protocol = XmippProtFlexAlign()
        protocol.error = MagicMock()
        protocol._getShiftsFile = MagicMock(return_value="/nonexistent/shifts.xmd")
        movie = MagicMock()
        movie.getFileName.return_value = "movie_000005/test.tif"
        movie.getObjId.return_value = 5

        protocol.tryProcessMovie = MagicMock(
            side_effect=RuntimeError("GPUassert: out of memory"))

        protocol._processMovie(movie)

        protocol.tryProcessMovie.assert_called_once_with(movie)
        self.assertEqual({5}, protocol._failedMovieIds())
        self.assertTrue(protocol.error.called)

    def testContinueAlignsAgainAFinishedMovieWithoutShifts(self):
        import os
        import tempfile

        protocol = XmippProtFlexAlign()
        movies = {objId: MagicMock(getObjId=MagicMock(return_value=objId))
                  for objId in (1, 2, 3)}

        with tempfile.TemporaryDirectory() as tmp:
            shifts = {objId: os.path.join(tmp, "shifts_%d.xmd" % objId)
                      for objId in movies}
            open(shifts[2], "w").close()
            protocol._getShiftsFile = lambda movie: shifts[movie.getObjId()]
            protocol._getPersistedCandidateIds = MagicMock(return_value={1})
            protocol._loadLogicalSetItemsByIds = (
                lambda inputSet, ids: [movies[objId] for objId in ids])
            protocol.inputMovies.get = MagicMock()

            retry = protocol._getResumeRetryIds({1, 2, 3})

        # 1 is published, 2 has its result and only needs publishing.
        self.assertEqual({3}, retry)

    def testProcessMovieKeepsSuccessfulProcessingUnchanged(self):
        protocol = XmippProtFlexAlign()
        movie = MagicMock()
        protocol.tryProcessMovie = MagicMock()

        protocol._processMovie(movie)

        protocol.tryProcessMovie.assert_called_once_with(movie)


    def testGPUArgsUsesPerDeviceBenchmarkStorage(self):
        protocol = XmippProtFlexAlign()
        protocol._getControlPoints = MagicMock(return_value=(6, 5, 7))

        args = protocol.getGPUArgs() % {"GPU": "3"}

        self.assertIn("fftBenchmark_3.txt", args)


    def testConvertInputPreparesGainBeforeMovieWorkers(self):
        protocol = XmippProtFlexAlign()
        protocol._prepareGainForAlignment = MagicMock()

        with patch.object(ProtAlignMovies, "_convertInputStep") as parent_convert:
            protocol._convertInputStep()

        parent_convert.assert_called_once_with()
        protocol._prepareGainForAlignment.assert_called_once_with()

    def testAutoControlPointsCanBeComputedWithoutMutatingProtocolParams(self):
        protocol = XmippProtFlexAlign()

        inputMovies = MagicMock()
        inputMovies.getDim.return_value = (4000, 3000, 35)
        inputMovies.getSamplingRate.return_value = 1.0

        protocol.inputMovies = MagicMock()
        protocol.inputMovies.get.return_value = inputMovies
        protocol.autoControlPoints = MagicMock()
        protocol.autoControlPoints.get.return_value = True
        protocol._isInputEer = MagicMock(return_value=False)

        protocol.controlPointX = MagicMock()
        protocol.controlPointY = MagicMock()
        protocol.controlPointT = MagicMock()

        points = protocol._getControlPoints()

        self.assertEqual(points, (6, 5, 7))
        protocol.controlPointX.set.assert_not_called()
        protocol.controlPointY.set.assert_not_called()
        protocol.controlPointT.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
