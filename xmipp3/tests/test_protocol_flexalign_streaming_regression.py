import unittest
from unittest.mock import MagicMock

from xmipp3.protocols.protocol_flexalign import XmippProtFlexAlign


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


if __name__ == "__main__":
    unittest.main()
