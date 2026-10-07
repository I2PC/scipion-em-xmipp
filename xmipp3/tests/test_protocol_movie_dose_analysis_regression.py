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

import pyworkflow.protocol.constants as cons

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
    number of reopen attempts, simulating a transient visibility
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
        # Mirrors the real pyworkflow Set.__getitem__: raises
        # UnboundLocalError (not a None return) for a row that is not
        # yet selectable, matching what _loadMoviesByIds now catches.
        if movieId not in self:
            raise UnboundLocalError(
                "local variable 'item' referenced before assignment")
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
        # backend with delayed visibility. _loadMoviesByIds must
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

    def _makeInputMoviesMock(self):
        return Mock(get=Mock(return_value=Mock(
            getSamplingRate=Mock(return_value=1.0),
            isStreamClosed=Mock(return_value=False),
            getFramesRange=Mock(return_value=None),
            getFirstItem=Mock(return_value=Mock(
                getAcquisition=Mock(return_value=Mock(
                    getDosePerFrame=Mock(return_value=1.0))))),
        )))

    def testInitializeStepDoesNotRestoreStateOnRestartDespiteIsContinuedBeingTrue(self):
        # Regression test: Protocol._runSteps() always forces runMode to
        # MODE_RESUME while executing, even when the user selected Restart
        # ("Always set to resume, even if set to restart" in pyworkflow's
        # own source) - so self.isContinued() alone cannot distinguish a
        # real Continue from a Restart once the protocol is actually
        # running. Only a real Continue/Resume (tracked via
        # _originalRunMode, set once per _runSteps() call before runMode
        # gets overwritten) should restore in-memory scientific state
        # (self.mu, stats, dose history) from the previous outputs.
        prot = XmippProtMovieDoseAnalysis()
        prot.inputMovies = self._makeInputMoviesMock()
        prot._originalRunMode = cons.MODE_RESTART
        prot._restoreRuntimeStateFromOutputs = Mock()

        prot.initializeStep()

        prot._restoreRuntimeStateFromOutputs.assert_not_called()

    def testInitializeStepRestoresStateOnRealContinue(self):
        prot = XmippProtMovieDoseAnalysis()
        prot.inputMovies = self._makeInputMoviesMock()
        prot._originalRunMode = cons.MODE_RESUME
        prot._restoreRuntimeStateFromOutputs = Mock()

        prot.initializeStep()

        prot._restoreRuntimeStateFromOutputs.assert_called_once()

    def testPrepareStreamingGeneratorDelegatesToInitializeStep(self):
        prot = XmippProtMovieDoseAnalysis()
        prot.initializeStep = Mock()

        prot._prepareStreamingGenerator()

        prot.initializeStep.assert_called_once()

    def testFinalizeStreamingGeneratorInsertsCreateOutputStepOnce(self):
        prot = XmippProtMovieDoseAnalysis()
        prot._steps = []
        prot._prevSteps = []
        inserted = []
        prot._insertFunctionStep = Mock(
            side_effect=lambda *a, **kw: (inserted.append((a, kw)), 99)[1])
        prot.updateSteps = Mock()

        prot._finalizeStreamingGenerator()

        self.assertEqual(1, len(inserted))
        prot.updateSteps.assert_called_once()

    def testFinalizeStreamingGeneratorDoesNotReinsertFinishedCreateOutputStepOnContinue(self):
        class FuncName:
            def get(self):
                return 'createOutputStep'

        class FinishedStep:
            funcName = FuncName()

            def isFinished(self):
                return True

        prot = XmippProtMovieDoseAnalysis()
        prot._steps = []
        prot._prevSteps = [FinishedStep()]
        prot._insertFunctionStep = Mock(
            side_effect=AssertionError(
                'A persisted FINISHED createOutputStep must not be '
                'inserted again on Continue.'))
        prot.updateSteps = Mock(
            side_effect=AssertionError(
                'Continue must not update the graph when finalization is '
                'already finished.'))

        # Must not raise.
        prot._finalizeStreamingGenerator()


if __name__ == "__main__":
    unittest.main()


class _PersistedDoseMovie:
    def __init__(self, objId, mean, diff, globalMedian, usingExperimental):
        self._objId = objId
        self._values = {
            '_MEAN_DOSE_PER_ANGSTROM2': mean,
            '_DIFF_TO_DOSE_PER_ANGSTROM2': diff,
            '_GLOBAL_DOSE_PER_ANGSTROM2': globalMedian,
            '_USING_EXPERIMENTAL_DOSE': usingExperimental,
        }

    def getObjId(self):
        return self._objId

    def getAttributeValue(self, name):
        return self._values.get(name)


class _RefreshRequiredDoseOutput:
    def __init__(self, movies):
        self.movies = list(movies)
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True

    def __iter__(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted movie output must be refreshed before restoring runtime state.'
            )
        return iter(self.movies)


class TestMovieDoseAnalysisLogicalOutputRestore(unittest.TestCase):
    def testRestoreRuntimeStateRefreshesPersistedOutputsBeforeIteration(self):
        accepted = _RefreshRequiredDoseOutput([
            _PersistedDoseMovie(1, 1.2, 0.1, 1.15, False),
        ])
        discarded = _RefreshRequiredDoseOutput([
            _PersistedDoseMovie(2, 1.8, 0.7, 1.15, False),
        ])

        prot = XmippProtMovieDoseAnalysis()
        prot.outputMovies = accepted
        prot.outputMoviesDiscarded = discarded

        prot._restoreRuntimeStateFromOutputs()

        self.assertTrue(accepted.loaded)
        self.assertTrue(discarded.loaded)
        self.assertEqual({1}, prot._acceptedIds)
        self.assertEqual({2}, prot._discardedIds)
        self.assertEqual({1, 2}, prot._doneIds)
        self.assertEqual(1.15, prot.mu)


class _RefreshRequiredDoseIdOutput:
    def __init__(self, ids):
        self.ids = set(ids)
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True

    def getIdSet(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted dose output must be refreshed before reading ids.'
            )
        return set(self.ids)


class TestMovieDoseAnalysisDoneCacheRestore(unittest.TestCase):
    def testDoneIdsCacheRefreshesPersistedOutputsBeforeReadingIds(self):
        accepted = _RefreshRequiredDoseIdOutput({1, 3})
        discarded = _RefreshRequiredDoseIdOutput({2})

        prot = XmippProtMovieDoseAnalysis()
        prot.outputMovies = accepted
        prot.outputMoviesDiscarded = discarded

        prot._loadDoneIdsCache()

        self.assertTrue(accepted.loaded)
        self.assertTrue(discarded.loaded)
        self.assertEqual({1, 3}, prot._acceptedIds)
        self.assertEqual({2}, prot._discardedIds)
        self.assertEqual({1, 2, 3}, prot._doneIds)
