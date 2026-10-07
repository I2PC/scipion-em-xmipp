from unittest.mock import Mock, patch
# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from pwem.objects import SetOfMicrographs, SetOfMovies

from xmipp3.protocols.protocol_movie_max_shift import (
    OUTPUT_MICS_DISCARDED,
    OUTPUT_MOVIES,
    XmippProtMovieMaxShift,
)
from xmipp3.tests.streaming_test_utils import (
    FreshOutputSetProbe,
    LogicalOutputSetProbe,
)


class _FakeInputSet:
    def __init__(self, ids, streamClosed=False, items=None):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self._items = items or {}
        self.closed = False

    def loadAllProperties(self):
        pass

    def __contains__(self, itemId):
        return itemId in self._items

    def getUniqueValues(self, attr, where=None):
        if where is None:
            return sorted(self._ids)
        threshold = int(where.split('>')[1].strip())
        return sorted(itemId for itemId in self._ids if itemId > threshold)

    def getIdSet(self):
        return set(self._ids)

    def getSize(self):
        return len(self._ids)

    def isStreamClosed(self):
        return self._streamClosed

    def getItem(self, _, itemId):
        return self._items.get(itemId)

    def close(self):
        self.closed = True


class _StrictFakeInputSet(_FakeInputSet):
    """Mimics the real Set.getItem contract: it raises rather than
    returning None for a row it cannot find (see
    pyworkflow.object.Set.__getitem__), so a caller must check membership
    (`in`) before calling getItem, not after."""

    def getItem(self, _, itemId):
        if itemId not in self._items:
            raise UnboundLocalError(
                "local variable 'item' referenced before assignment"
            )
        return self._items[itemId]


class _FakePointer:
    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj


class _FakeItem:
    def __init__(self, itemId):
        self.itemId = itemId
        self.enabled = True

    def clone(self):
        clone = _FakeItem(self.itemId)
        clone.enabled = self.enabled
        return clone

    def getObjId(self):
        return self.itemId

    def setEnabled(self, enabled):
        self.enabled = enabled


class _FakeOutputSet:
    def __init__(self):
        self.items = []
        self.closed = False

    def append(self, item):
        self.items.append(item)

    def getSize(self):
        return len(self.items)

    def getIdSet(self):
        return {item.getObjId() for item in self.items}

    def close(self):
        self.closed = True


class TestXmippMovieMaxShiftRegression(BaseTest):
    """Regression tests for Movie Max Shift streaming and Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtMovieMaxShift)
        prot.insertedIds = []
        prot.acceptedIds = []
        prot.discardedIds = []
        prot.isStreamClosed = False
        prot.alreadyLoad = True
        prot._lastInputId = 0
        return prot

    def _prepareInputCheck(self, prot, ids, streamClosed=False):
        fakeSet = _FakeInputSet(ids, streamClosed=streamClosed)
        scheduled = []
        updates = []

        prot.inputMovies = _FakePointer(fakeSet)
        prot.newDeps = []

        def insertSteps(newIds):
            newIds = sorted(newIds)
            scheduled.append(newIds)
            prot.insertedIds.extend(newIds)
            return []

        prot._insertNewMoviesSteps = insertSteps
        prot.updateSteps = lambda: updates.append(True)

        return fakeSet, scheduled, updates

    def testCheckNewInputUsesWatermarkNotSqliteMtime(self):
        # Regression test: new-id discovery must come from the id watermark
        # (_lastInputId / getUniqueValues('id', where=...)), not from any
        # file mtime - a Set not backed by a local file can change without its
        # compatibility file's mtime changing.
        prot = self._newProtocol()
        prot.insertedIds = [1]
        prot._lastInputId = 1

        _, scheduled, updates = self._prepareInputCheck(
            prot,
            ids=[1, 2],
            streamClosed=False
        )

        prot._checkNewInput()

        self.assertEqual([[2]], scheduled)
        self.assertEqual(1, len(updates))

    def testResumeSkipsMoviesAlreadyPersistedInOutputs(self):
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESUME
        prot.runMode.set(MODE_RESUME)
        prot._getAllDoneIds = lambda: ([1], 1, [1], [])

        _, scheduled, updates = self._prepareInputCheck(
            prot,
            ids=[1, 2, 3],
            streamClosed=False
        )

        prot._checkNewInput()

        self.assertEqual([[2, 3]], scheduled)
        self.assertEqual([1, 2, 3], sorted(prot.insertedIds))
        self.assertEqual(1, len(updates))

    def testRestartDoesNotReusePreviousOutputs(self):
        prot = self._newProtocol()

        # Reproduce Protocol._runSteps(): runMode is changed to RESUME while
        # _originalRunMode keeps the action originally requested by the user.
        prot._originalRunMode = MODE_RESTART
        prot.runMode.set(MODE_RESUME)

        def failIfDoneOutputsAreRead():
            raise AssertionError('Restart must not restore previous output IDs.')

        prot._getAllDoneIds = failIfDoneOutputsAreRead

        _, scheduled, updates = self._prepareInputCheck(
            prot,
            ids=[1, 2, 3],
            streamClosed=False
        )

        prot._checkNewInput()

        self.assertEqual([[1, 2, 3]], scheduled)
        self.assertEqual(1, len(updates))

    def testDiscardedMicrographsAreDeclaredAsPossibleOutput(self):
        self.assertEqual(
            'outputMicrographsDiscarded',
            OUTPUT_MICS_DISCARDED
        )
        self.assertIn(
            'outputMicrographsDiscarded',
            XmippProtMovieMaxShift._possibleOutputs
        )
        self.assertIs(
            SetOfMicrographs,
            XmippProtMovieMaxShift._possibleOutputs[
                'outputMicrographsDiscarded'
            ]
        )

    def testStreamingMovieOutputReusesLogicalSetWithoutLegacySqlite(self):
        prot = self._newProtocol()

        logicalOutput = LogicalOutputSetProbe()
        prot.outputMovies = logicalOutput
        prot.inputMics = None
        prot.inputMovies = _FakePointer(_FakeInputSet(ids=[1]))
        prot._getPath = (
            lambda baseName:
            '/tmp/' + baseName
        )

        outputSet = prot._loadOutputSet(
            FreshOutputSetProbe,
            'movies.sqlite',
        )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming movie output must reuse the logical output "
            "when the legacy SQLite file is absent.",
        )
        self.assertEqual(
            1,
            logicalOutput.enableAppendCalls,
        )

    def testMovieOutputWorksWithoutAssociatedMicrographs(self):
        prot = self._newProtocol()
        prot.acceptedIds = [1]
        prot.inputMics = None
        prot.outMicName = None
        # Simulate the sibling-micrograph discovery genuinely finding none,
        # without touching the real mapper/parent-protocol lookup.
        prot.setInputMics = lambda: None
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._store = lambda: None
        prot._defineTransformRelation = lambda *args, **kwargs: None

        movie = _FakeItem(1)

        prot.inputMovies = _FakePointer(_FakeInputSet(
            ids=[1],
            streamClosed=False,
            items={1: movie}
        ))

        movieOutput = _FakeOutputSet()

        def loadOutputSet(SetClass, _):
            if SetClass is SetOfMovies:
                return movieOutput
            if SetClass is SetOfMicrographs:
                return None
            raise AssertionError('Unexpected output Set class.')

        prot._loadOutputSet = loadOutputSet

        def failIfMicsAreLoaded():
            raise AssertionError(
                'Associated micrographs must not be loaded when inputMics is None.'
            )

        prot._loadMicAssociatedInputSet = failIfMicsAreLoaded

        updatedOutputs = []
        prot._updateOutputSet = (
            lambda outputName, outputSet, streamMode:
            updatedOutputs.append((outputName, outputSet, streamMode))
        )

        prot._checkNewOutput()

        self.assertEqual(1, movieOutput.getSize())
        self.assertEqual(OUTPUT_MOVIES, updatedOutputs[0][0])

    def testMissingAssociatedMicrographSetReturnsNone(self):
        prot = self._newProtocol()
        prot.outMicName = None

        class FailMapper:
            def getParent(self, _):
                raise AssertionError(
                    'Parent protocol must not be queried without an output name.'
                )

        prot.getMapper = lambda: FailMapper()

        self.assertIsNone(prot._loadMicAssociatedInputSet())

    def testRetriesAssociatedMicrographDiscoveryWhileStillMissing(self):
        # Streaming may expose movies before the sibling mic output exists.
        prot = self._newProtocol()

        # A previous streaming check already tried to discover associated mics
        # and found none. The protocol must not treat that result as permanent.
        prot.alreadyLoad = True
        prot.inputMics = None
        prot.outMicName = None

        prot.setInputMics = Mock()
        prot._insertFunctionStep = Mock(return_value=17)

        deps = prot._insertNewMoviesSteps([2])

        prot.setInputMics.assert_called_once_with()
        self.assertEqual([17], deps)
        self.assertIn(2, prot.insertedIds)

    def testDiscoversUniqueGenericMicrographOutput(self):
        prot = self._newProtocol()

        sourceMics = SetOfMicrographs()

        class ParentProtocol:
            def iterOutputAttributes(self):
                return [('alignedMics', sourceMics)]

        class Mapper:
            def getParent(self, _):
                return ParentProtocol()

        prot.getMapper = lambda: Mapper()

        prot.setInputMics()

        self.assertIs(sourceMics, prot.inputMics)
        self.assertEqual('alignedMics', prot.inputMicName)
        self.assertEqual(
            'outputMicrographs',
            prot.outMicName,
            'A unique generic SetOfMicrographs output should be '
            'consumed without depending on a protocol-specific name.',
        )

    def testDiscoversNewMotionCorrDoseWeightedMicrographs(self):
        prot = self._newProtocol()

        sourceMics = object()

        class ParentProtocol:
            micrographsDW = sourceMics

        class Mapper:
            def getParent(self, _):
                return ParentProtocol()

        prot.getMapper = lambda: Mapper()

        prot.setInputMics()

        self.assertIs(sourceMics, prot.inputMics)
        self.assertEqual('micrographsDW', prot.inputMicName)
        self.assertEqual(
            'outputMicrographsDoseWeighted',
            prot.outMicName,
            'MaxShift must consume New MotionCorr micrographsDW but '
            'publish its own canonical dose-weighted micrograph output.',
        )

    def testBackfillsMicrographPublishedAfterMovieWasAlreadyDone(self):
        prot = self._newProtocol()
        prot.acceptedIds = [1]
        prot.inputMics = object()
        prot.outMicName = 'outputMicrographs'
        prot._getAllDoneIds = lambda: ([1], 1, [1], [])
        prot._store = lambda: None
        prot._defineTransformRelation = lambda *args, **kwargs: None

        movie = _FakeItem(1)
        mic = _FakeItem(1)

        prot.inputMovies = _FakePointer(_FakeInputSet(
            ids=[1], streamClosed=False, items={1: movie}
        ))
        prot._loadMicAssociatedInputSet = lambda: _FakeInputSet(
            ids=[1], streamClosed=False, items={1: mic}
        )

        movieOutput = _FakeOutputSet()
        movieOutput.append(movie.clone())
        micOutput = _FakeOutputSet()

        def loadOutputSet(SetClass, baseName):
            if SetClass is SetOfMovies:
                return movieOutput
            if SetClass is SetOfMicrographs:
                return micOutput
            raise AssertionError('Unexpected output Set class.')

        prot._loadOutputSet = loadOutputSet

        updatedOutputs = []
        prot._updateOutputSet = (
            lambda outputName, outputSet, streamMode:
            updatedOutputs.append((outputName, outputSet, streamMode))
        )

        prot._checkNewOutput()

        self.assertEqual(
            {1},
            movieOutput.getIdSet(),
            'Backfilling a late micrograph must not duplicate the movie.',
        )
        self.assertEqual(
            {1},
            micOutput.getIdSet(),
            'A micrograph that appears after its movie was already persisted '
            'must be backfilled into the corresponding MaxShift output.',
        )
        self.assertTrue(
            any(name == 'outputMicrographs' for name, _, _ in updatedOutputs),
            'The backfilled micrograph Set must be published as outputMicrographs.',
        )

    def testDoesNotFinishWhilePendingMicrographBackfillRemains(self):
        # Regression test: self.finished must not latch True purely from
        # movie counts while a sibling micrograph is still missing for an
        # already-evaluated movie. _stepsCheck's "if finished: return" makes
        # `finished` permanent - once set, this micrograph would never be
        # retried and would be silently lost forever. This matches the
        # reported "movie_max_shift finishing prematurely" symptom.
        prot = self._newProtocol()
        prot.acceptedIds = [1]
        prot.isStreamClosed = True
        prot.inputMics = object()
        prot.outMicName = 'outputMicrographs'
        prot._getAllDoneIds = lambda: ([1], 1, [1], [])
        prot._store = lambda: None
        prot._defineTransformRelation = lambda *args, **kwargs: None

        movie = _FakeItem(1)
        prot.inputMovies = _FakePointer(_FakeInputSet(
            ids=[1], streamClosed=True, items={1: movie}
        ))
        # The sibling micrograph is still not visible for this movie.
        prot._loadMicAssociatedInputSet = lambda: _FakeInputSet(
            ids=[], streamClosed=True, items={}
        )

        movieOutput = _FakeOutputSet()
        movieOutput.append(movie.clone())
        micOutput = _FakeOutputSet()

        def loadOutputSet(SetClass, baseName):
            if SetClass is SetOfMovies:
                return movieOutput
            if SetClass is SetOfMicrographs:
                return micOutput
            raise AssertionError('Unexpected output Set class.')

        prot._loadOutputSet = loadOutputSet
        prot._updateOutputSet = lambda *args, **kwargs: None

        prot._checkNewOutput()

        self.assertFalse(
            prot.finished,
            'A pending sibling micrograph must prevent the protocol from '
            'latching finished=True, or the micrograph would never be '
            'retried and would be permanently lost.',
        )

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        prot = self.newProtocol(XmippProtMovieMaxShift)
        prot.finished = True
        prot.initializeStep = Mock()
        prot._checkNewInput = Mock()
        prot._checkNewOutput = Mock()
        prot._insertFunctionStep = Mock(return_value=1)
        prot.createOutputStep = Mock()

        prot.stepsGeneratorStep()

        prot._checkNewInput.assert_not_called()
        prot._checkNewOutput.assert_not_called()
        prot._insertFunctionStep.assert_called_once()

    def testFillOutputChecksMembershipBeforeGetItem(self):
        # Regression test: Set.getItem raises (UnboundLocalError) rather
        # than returning None for a row it cannot find, so fillOutput must
        # check membership (`in`) BEFORE calling getItem for both the
        # movie and its sibling micrograph - a lenient fake that tolerates
        # getItem(missing) -> None would hide this bug, so this test uses
        # _StrictFakeInputSet to mimic the real contract.
        prot = self._newProtocol()
        prot.acceptedIds = [1]
        prot.isStreamClosed = True
        prot.inputMics = object()
        prot.outMicName = 'outputMicrographs'
        prot._getAllDoneIds = lambda: ([1], 1, [1], [])
        prot._store = lambda: None
        prot._defineTransformRelation = lambda *args, **kwargs: None

        movie = _FakeItem(1)
        prot.inputMovies = _FakePointer(_StrictFakeInputSet(
            ids=[1], streamClosed=True, items={1: movie}
        ))
        # The sibling micrograph is genuinely not visible yet.
        prot._loadMicAssociatedInputSet = lambda: _StrictFakeInputSet(
            ids=[], streamClosed=True, items={}
        )

        movieOutput = _FakeOutputSet()
        micOutput = _FakeOutputSet()

        def loadOutputSet(SetClass, baseName):
            if SetClass is SetOfMovies:
                return movieOutput
            if SetClass is SetOfMicrographs:
                return micOutput
            raise AssertionError('Unexpected output Set class.')

        prot._loadOutputSet = loadOutputSet
        prot._updateOutputSet = lambda *args, **kwargs: None

        prot._checkNewOutput()  # must not raise

        self.assertEqual({1}, movieOutput.getIdSet())
        self.assertEqual(set(), micOutput.getIdSet())
        self.assertFalse(prot.finished)

    def testEvaluateMovieAlignRetriesTransientlyMissingMovie(self):
        # Regression test: a movie just discovered via the id watermark
        # may still momentarily fail a getItem lookup. _evaluateMovieAlign
        # (a one-shot worker step with no natural retry from the polling
        # loop) must retry via _loadMovieForEvaluation rather than
        # crashing the whole batch.
        prot = self._newProtocol()
        prot.samplingRate = 1.0
        prot.rejType = XmippProtMovieMaxShift.REJ_OR
        prot.maxMovieShift = Mock(get=Mock(return_value=45))
        prot.maxFrameShift = Mock(get=Mock(return_value=10))

        class _NoShiftAlignment:
            def getShifts(self):
                return [], []

        class _FlakyMovie(_FakeItem):
            def getAlignment(self):
                return _NoShiftAlignment()

            def clone(self):
                return _FlakyMovie(self.itemId)

        attempts = {'count': 0}

        class _FlakyInputSet(_StrictFakeInputSet):
            def __contains__(self, itemId):
                attempts['count'] += 1
                return attempts['count'] >= 2

        movie = _FlakyMovie(1)
        prot.inputMovies = _FakePointer(_FlakyInputSet(
            ids=[1], streamClosed=True, items={1: movie}
        ))

        with patch(
                'xmipp3.protocols.protocol_movie_max_shift.time.sleep',
                return_value=None,
        ):
            prot._evaluateMovieAlign([1])

        self.assertEqual([1], prot.acceptedIds)
        self.assertEqual([], prot.discardedIds)

    def testEvaluateMovieAlignDiscardsMovieThatNeverBecomesVisible(self):
        prot = self._newProtocol()
        prot.samplingRate = 1.0
        prot.inputMovies = _FakePointer(_StrictFakeInputSet(
            ids=[], streamClosed=True, items={}
        ))

        with patch(
                'xmipp3.protocols.protocol_movie_max_shift.time.sleep',
                return_value=None,
        ):
            prot._evaluateMovieAlign([1])

        self.assertEqual([], prot.acceptedIds)
        self.assertEqual([1], prot.discardedIds)

    def testEvaluateMovieAlignDiscardsMovieWithCorruptedAlignmentAndKeepsProcessingOthers(self):
        # Regression test: a genuine processing failure on one movie (e.g.
        # corrupted/missing alignment data) must not crash the whole batch
        # step - and hence the whole protocol via pyworkflow's
        # fail-on-any-exception step boundary. It must be discarded with a
        # clear message while the rest of the batch is still evaluated.
        prot = self._newProtocol()
        prot.samplingRate = 1.0
        prot.rejType = XmippProtMovieMaxShift.REJ_OR
        prot.maxMovieShift = Mock(get=Mock(return_value=45))
        prot.maxFrameShift = Mock(get=Mock(return_value=10))

        class _NoShiftAlignment:
            def getShifts(self):
                return [], []

        class _GoodMovie(_FakeItem):
            def getAlignment(self):
                return _NoShiftAlignment()

            def clone(self):
                return _GoodMovie(self.itemId)

        class _CorruptedMovie(_FakeItem):
            def getAlignment(self):
                return None  # simulates missing/corrupted alignment data

            def clone(self):
                return _CorruptedMovie(self.itemId)

        prot.inputMovies = _FakePointer(_StrictFakeInputSet(
            ids=[1, 2], streamClosed=True,
            items={1: _CorruptedMovie(1), 2: _GoodMovie(2)},
        ))

        prot._evaluateMovieAlign([1, 2])

        self.assertEqual([1], prot.discardedIds)
        self.assertEqual([2], prot.acceptedIds)
