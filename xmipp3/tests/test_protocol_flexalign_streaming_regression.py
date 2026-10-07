import unittest
from unittest.mock import MagicMock, patch

from xmipp3.protocols.protocol_flexalign import XmippProtFlexAlign
from pwem.protocols import ProtAlignMovies


class TestFlexAlignScientificFailureRegression(unittest.TestCase):
    """Regression coverage for scientific failures inside movie workers."""

    def testProcessMoviePropagatesScientificFailure(self):
        protocol = XmippProtFlexAlign()
        movie = MagicMock()
        movie.getFileName.return_value = "movie_000005/test.tif"

        failure = RuntimeError("GPUassert: out of memory")
        protocol.tryProcessMovie = MagicMock(side_effect=failure)

        with self.assertRaises(RuntimeError) as raised:
            protocol._processMovie(movie)

        self.assertIs(raised.exception, failure)
        protocol.tryProcessMovie.assert_called_once_with(movie)

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
