# ***************************************************************************
# *
# * Movie gain in streaming: a movie whose gain estimation fails is still
# * published, without its gain, Continue estimates it again, and the run
# * fails only when every estimation failed.
# *
# ***************************************************************************

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from xmipp3.protocols.protocol_movie_gain import XmippProtMovieGain


def _protocol(root):
    protocol = XmippProtMovieGain()
    protocol.movieStep.set(1)
    protocol.estimateGain.set(True)
    protocol.estimateResidualGain.set(False)
    protocol._getExtraPath = lambda *paths: os.path.join(root, *paths)
    return protocol


class TestAFailedGainEstimation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_ItDoesNotFailTheStep(self):
        protocol = _protocol(self._tmp.name)
        protocol.estimatedIds = set()
        protocol.estimatedResIds = set()
        protocol.error = MagicMock()
        protocol.getInputGain = MagicMock(return_value=None)
        protocol.estimateGainFun = MagicMock(side_effect=RuntimeError("boom"))
        movie = MagicMock(getObjId=MagicMock(return_value=3))

        protocol._processMovie(movie)

        self.assertEqual({3}, protocol._failedMovieIds())
        self.assertNotIn(3, protocol.estimatedIds,
                         "Not estimated: Continue estimates it again.")

    def test_TheMovieIsNoLongerExpectedToHaveAGain(self):
        protocol = _protocol(self._tmp.name)
        protocol._failedMovieIds().add(2)

        self.assertEqual({1, 3}, protocol._gainMovieIds({1, 2, 3}))

    def test_ContinueEstimatesAgainAMovieWithoutItsGain(self):
        protocol = _protocol(self._tmp.name)
        open(protocol.getEstimatedGainPath(1), 'w').close()

        self.assertEqual({2}, protocol._getResumeRetryIds({1, 2}))

    def test_EveryEstimationFailedFailsTheRun(self):
        protocol = _protocol(self._tmp.name)
        protocol._failedMovieIds().update({1, 2})

        with self.assertRaisesRegex(Exception, "failed for every one"):
            protocol._failIfEveryGainEstimationFailed({1, 2}, set(), set())

    def test_SomeEstimatedGainsAreNoFailure(self):
        protocol = _protocol(self._tmp.name)
        protocol._failedMovieIds().add(2)

        protocol._failIfEveryGainEstimationFailed({1, 2}, {1}, set())


if __name__ == '__main__':
    unittest.main()
