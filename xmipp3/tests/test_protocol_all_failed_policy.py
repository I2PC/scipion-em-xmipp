# ***************************************************************************
# *
# * Streaming filters fail only when every item failed: tilt analysis and
# * movie dose analysis left them all out and still ended well.
# *
# ***************************************************************************

import unittest
from unittest.mock import Mock

from xmipp3.protocols.protocol_movie_dose_analysis import XmippProtMovieDoseAnalysis
from xmipp3.protocols.protocol_tilt_analysis import XmippProtTiltAnalysis


class TestTiltAnalysisFailsOnlyWhenEveryMicrographFailed(unittest.TestCase):
    def test_EveryMicrographFailedFailsTheRun(self):
        protocol = XmippProtTiltAnalysis()
        protocol._excludedMicIds().update({1, 2})

        with self.assertRaisesRegex(Exception, "failed for every one of the 2"):
            protocol._failIfEveryMicrographFailed(set())

    def test_SomePublishedMicrographsAreNoFailure(self):
        protocol = XmippProtTiltAnalysis()
        protocol._excludedMicIds().add(2)

        protocol._failIfEveryMicrographFailed({1})

    def test_NothingExcludedIsNoFailure(self):
        XmippProtTiltAnalysis()._failIfEveryMicrographFailed(set())


class TestDoseAnalysisFailsOnlyWhenEveryMovieFailed(unittest.TestCase):
    def _protocol(self):
        protocol = XmippProtMovieDoseAnalysis()
        protocol.insertedIds = [1, 2]
        protocol.processedIds = [1, 2]
        protocol._doneIds = set()
        protocol._acceptedIds = set()
        protocol._discardedIds = set()
        protocol._inputSize = 2
        protocol.isStreamClosed = True
        protocol.meanDoseList = []
        protocol.meanDoseById = {}
        protocol._store = Mock()
        return protocol

    def test_EveryMovieFailedFailsTheRunWithoutPublishingThem(self):
        protocol = self._protocol()
        protocol._loadOutputSet = Mock(
            side_effect=AssertionError("Published as discarded, Continue "
                                       "would never try them again."))

        with self.assertRaisesRegex(Exception, "failed for every one of the 2"):
            protocol._checkNewOutput()


if __name__ == '__main__':
    unittest.main()
