# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

import unittest
from unittest.mock import Mock, patch

from xmipp3.protocols.protocol_movie_dose_analysis import XmippProtMovieDoseAnalysis


class _FakeMovie:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _FakeMovie(self._objId)


class _FlakyFakeMovieSet:
    """A fake movie Set where some ids only become visible after a given
    number of reopen attempts, simulating a transient PostgreSQL-bridge
    visibility lag. Each pointer.get() call represents a fresh reopen."""

    def __init__(self, visibleFromAttempt):
        self.visibleFromAttempt = dict(visibleFromAttempt)
        self.attempt = 0
        self.closed = 0

    def loadAllProperties(self):
        pass

    def __contains__(self, movieId):
        return self.attempt >= self.visibleFromAttempt.get(movieId, 0)

    def getItem(self, _, movieId):
        return _FakeMovie(movieId)

    def close(self):
        self.closed += 1


class _FakePointer:
    def __init__(self, movieSet):
        self._movieSet = movieSet

    def get(self):
        self._movieSet.attempt += 1
        return self._movieSet


class TestMovieDoseAnalysisRegression(unittest.TestCase):
    """Regression tests for Movie Dose Analysis streaming/resume handling."""

    def testGetNewDoneIdsDoesNotStallOnOutOfOrderParallelCompletion(self):
        # Regression test: this protocol runs with parallel step execution
        # (STEPS_PARALLEL, batched), so movies do not necessarily finish
        # processing in id order. A single movie that is still pending (or
        # permanently stuck, e.g. a batch step that never completes) must
        # not block every other, already-processed, higher-id movie from
        # ever being counted as done - otherwise the protocol stalls
        # forever, never reaching allDone == maxMicSize even though almost
        # everything actually finished.
        prot = XmippProtMovieDoseAnalysis()
        prot.insertedIds = [1, 2, 3, 4, 5]
        prot.processedIds = [1, 2, 4, 5]  # 3 is still pending/stuck

        newDone = prot._getNewDoneIds(doneListIds=[])

        self.assertEqual(
            [1, 2, 4, 5], newDone,
            "Movies processed out of id order must still be reported as "
            "done instead of being permanently blocked by a pending one.",
        )

    def testGetNewDoneIdsPicksUpPreviouslyStuckIdOnceProcessed(self):
        # Once the previously pending movie finishes and is skipped via the
        # doneIds set on a later check, it must be reported too - no item is
        # permanently lost, it is just deferred to the round where it
        # actually completes.
        prot = XmippProtMovieDoseAnalysis()
        prot.insertedIds = [1, 2, 3, 4, 5]
        prot.processedIds = [1, 2, 3, 4, 5]

        newDone = prot._getNewDoneIds(doneListIds=[1, 2, 4, 5])

        self.assertEqual([3], newDone)


    def testFinishedCheckDoesNotReloadInputWithoutNewMovies(self):
        """A final streaming check must be a no-op when everything is done."""
        prot = XmippProtMovieDoseAnalysis()
        prot.insertedIds = [1]
        prot.processedIds = [1]
        prot._doneIds = {1}
        prot._acceptedIds = {1}
        prot._discardedIds = set()
        prot._inputSize = 1
        prot.isStreamClosed = True
        prot.finished = True
        prot.mu = 1.0
        prot.meanDoseById = {1: 1.0}
        prot._lastPlotCount = 1

        prot._loadMoviesByIds = Mock(return_value={})
        prot._getFirstJoinStep = Mock(return_value=None)
        prot._store = Mock()

        prot._checkNewOutput()

        prot._loadMoviesByIds.assert_not_called()

    def testLoadMoviesByIdsRetriesTransientlyMissingMovie(self):
        # Regression test: Set.getItem raises UnboundLocalError (not None)
        # when a row is not yet selectable, so a movie that was just
        # discovered via a fresh id-watermark scan may still momentarily
        # fail a subsequent getItem lookup - especially plausible under a
        # PostgreSQL-backed compatibility bridge. _loadMoviesByIds must
        # retry rather than crash the whole batch.
        prot = XmippProtMovieDoseAnalysis()
        movieSet = _FlakyFakeMovieSet(visibleFromAttempt={2: 2})
        prot.inputMovies = _FakePointer(movieSet)

        with patch(
                'xmipp3.protocols.protocol_movie_dose_analysis.time.sleep',
                return_value=None,
        ):
            movies = prot._loadMoviesByIds([1, 2])

        self.assertEqual({1, 2}, set(movies.keys()))
        self.assertGreaterEqual(movieSet.attempt, 2)

    def testLoadMoviesByIdsGivesUpGracefullyAfterMaxAttempts(self):
        # A movie that never becomes visible must be left out of the
        # result instead of raising or blocking the caller forever.
        prot = XmippProtMovieDoseAnalysis()
        movieSet = _FlakyFakeMovieSet(visibleFromAttempt={2: 999})
        prot.inputMovies = _FakePointer(movieSet)

        with patch(
                'xmipp3.protocols.protocol_movie_dose_analysis.time.sleep',
                return_value=None,
        ):
            movies = prot._loadMoviesByIds([1, 2])

        self.assertEqual({1}, set(movies.keys()))

    def testProcessMoviesToleratesMovieMissingFromLoadedBatch(self):
        # Regression test: if a single movie in a parallel batch could not
        # be loaded (still missing after _loadMoviesByIds' own retries),
        # the rest of the batch must still be recorded as processed, and
        # the missing movie itself must still be marked processed (with no
        # stats) rather than silently vanishing from processedIds forever -
        # otherwise the sorted-scan in _getNewDoneIds would wait for it
        # indefinitely.
        prot = XmippProtMovieDoseAnalysis()
        prot.stats = {}
        prot.meanDoseById = {}
        prot.processedIds = []
        prot._loadMoviesByIds = Mock(return_value={1: _FakeMovie(1)})
        prot.estimatePoissonCount = Mock(return_value=None)

        prot._processMovies([1, 2])

        self.assertEqual([1, 2], sorted(prot.processedIds))

    def testCheckNewOutputSkipsMovieMissingFromInputWithoutCrashing(self):
        # A movie still unreachable when building the final output must be
        # skipped for this round (retried on the next check) instead of
        # crashing _checkNewOutput or being marked done/finished.
        prot = XmippProtMovieDoseAnalysis()
        prot.insertedIds = [1]
        prot.processedIds = [1]
        prot._doneIds = set()
        prot._acceptedIds = set()
        prot._discardedIds = set()
        prot._inputSize = 1
        prot.isStreamClosed = False
        prot.mu = 1.0
        prot.usingExperimental = False
        prot.stats = {1: {'mean': 1.0, 'std': 0.1, 'min': 0.9, 'max': 1.1}}
        prot.meanDoseById = {1: 1.0}
        prot.medianDifferences = []
        prot.medianDifferenceIds = []
        prot.medianDoseTemporal = []
        prot.framesRange = None
        prot._lastPlotCount = 0
        prot.window = Mock(get=Mock(return_value=50))
        prot.percentage_threshold = Mock(get=Mock(return_value=5))

        prot._loadMoviesByIds = Mock(return_value={})
        prot._getFirstJoinStep = Mock(return_value=None)
        prot._store = Mock()
        prot._updateDosePlots = Mock()

        prot._checkNewOutput()

        self.assertFalse(prot.finished)
        self.assertEqual(set(), prot._doneIds)


if __name__ == "__main__":
    unittest.main()
