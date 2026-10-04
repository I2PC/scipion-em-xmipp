# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from types import SimpleNamespace
from unittest.mock import Mock, patch

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_movie_gain import XmippProtMovieGain
from xmipp3.tests.streaming_test_utils import (
    FakeOutputSet as _FakeOutputSet,
    FreshOutputSetProbe,
    LogicalOutputSetProbe,
)


class _FakeFirstMovie:
    def getObjDict(self, includeBasic=True):
        return {}


class _FakeMoviesList:
    """ Minimal stand-in supporting both getFirstItem() (used to seed the
    one-time orientation step) and iteration (used for per-movie steps). """

    def __init__(self, items=None):
        self._items = items or []

    def getFirstItem(self):
        return self._items[0] if self._items else _FakeFirstMovie()

    def __iter__(self):
        return iter(self._items)


class _FakeMovie:
    def __init__(self, objId, samplingRate=1.5):
        self._objId = objId
        self._samplingRate = samplingRate

    def getObjId(self):
        return self._objId

    def getSamplingRate(self):
        return self._samplingRate

    def clone(self):
        return _FakeMovie(self._objId, self._samplingRate)


class _FreshOutputSet(_FakeOutputSet):
    def getIdSet(self):
        raise AssertionError('Fresh output Set must not query IDs before its first append.')


class TestXmippMovieGainRegression(BaseTest):
    """Regression tests for Movie Gain streaming and Continue recovery."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        return self.newProtocol(XmippProtMovieGain, estimateGain=True, estimateResidualGain=True, estimateOrientation=False, normalizeGain=False)

    def testEstimatedIdsAreNotResetByNewStreamingBatch(self):
        prot = self._newProtocol()
        prot.estimatedIds = [1]
        prot.estimatedResIds = [1]
        prot.convertCIStep = []

        prot._insertNewMoviesSteps({1: 10}, [])

        self.assertEqual([1], prot.estimatedIds)
        self.assertEqual([1], prot.estimatedResIds)

    def testNewInputAlwaysUsesReloadedMovieSnapshot(self):
        prot = self._newProtocol()
        prot.insertedDict = {1: 10}
        prot.listOfMovies = [_FakeMovie(1)]
        scheduled = []

        prot._loadLogicalInputMovies = lambda watermark: (
            [_FakeMovie(2)],
            False,
        )
        prot._getFirstJoinStep = lambda: None
        prot._insertNewMoviesSteps = (
            lambda inserted, movies:
            scheduled.extend(
                movie.getObjId()
                for movie in movies
                if movie.getObjId() not in inserted
            ) or []
        )
        prot.updateSteps = lambda: None

        prot._checkNewInput()

        self.assertEqual([2], scheduled)

    def testProcessMoviePropagatesCorruptedGainFailure(self):
        # A corrupted gain must fail the movie step. Swallowing this error
        # would let ProtProcessMovies mark the step as FINISHED even though
        # the logical gain outputs were never produced.
        prot = self._newProtocol()
        prot.estimatedIds = []
        prot.estimatedResIds = []
        prot.doGainProcess = lambda movieId: True
        prot.getInputGain = Mock(
            side_effect=ValueError("corrupted gain header"),
        )

        with self.assertRaises(ValueError):
            prot._processMovie(_FakeMovie(7))

    def testStreamingMovieOutputReusesLogicalSet(self):
        class InputPointer:
            def get(self):
                return object()

        prot = self._newProtocol()
        logicalOutput = LogicalOutputSetProbe()
        prot.outputMovies = logicalOutput
        prot.inputMovies = InputPointer()

        outputSet = prot._loadOutputSet(
            FreshOutputSetProbe,
            'outputMovies',
        )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming movie output must reuse the persisted logical Set.",
        )
        self.assertEqual(1, logicalOutput.enableAppendCalls)

    def testFreshOutputSetDoesNotQueryIds(self):
        self.assertEqual(set(), XmippProtMovieGain._getOutputIds(_FreshOutputSet()))

    def testOutputsAreIdempotentFromPersistedLogicalState(self):
        # Persisted logical outputs are the completion truth. An already
        # published estimated gain must not be appended again, while the
        # missing residual gain and movie are repaired exactly once.
        class _PersistedOutputSet(_FakeOutputSet):
            iterItems = None

        prot = self._newProtocol()
        movie = _FakeMovie(1)
        prot.listOfMovies = [movie]
        prot.streamClosed = False
        prot.doGainProcess = lambda movieId: True
        prot.getEstimatedGainPath = (
            lambda movieId: 'estimated_%d.xmp' % movieId
        )
        prot.getResidualGainPath = (
            lambda movieId: 'residual_%d.xmp' % movieId
        )
        prot._getFirstJoinStep = lambda: None
        prot._getFinishedProcessMovieIds = lambda: {1}

        estimated = _PersistedOutputSet({1})
        residual = _PersistedOutputSet()
        movies = _PersistedOutputSet()

        prot.estimatedGains = estimated
        prot.residualGains = residual
        prot.outputMovies = movies

        events = []

        def loadOutputSet(SetClass, outputName, fixGain=False):
            if outputName == 'estimatedGains':
                return estimated
            if outputName == 'residualGains':
                return residual
            if outputName == 'outputMovies':
                return movies
            raise AssertionError(
                'Unexpected output name: %s' % outputName
            )

        prot._loadOutputSet = loadOutputSet
        prot._updateOutputSet = (
            lambda outputName, outputSet, state:
            events.append(outputName)
        )
        prot._readDoneList = lambda: (_ for _ in ()).throw(
            AssertionError(
                'A sidecar must not be used as durable state.'
            )
        )
        prot._writeDoneList = lambda done: (_ for _ in ()).throw(
            AssertionError(
                'A sidecar must not be written as durable state.'
            )
        )

        prot._checkNewOutput()

        self.assertEqual([], estimated.appended)
        self.assertEqual([1], residual.appended)
        self.assertEqual([1], movies.appended)
        self.assertEqual(
            ['residualGains', 'outputMovies'],
            events,
        )

    def testGetAllDoneIdsReadsRealOutputMovies(self):
        prot = self._newProtocol()
        prot.outputMovies = _FakeOutputSet({1, 2})

        self.assertEqual({1, 2}, prot._getAllDoneIds())

    def testGetAllDoneIdsIsEmptyWithoutOutputYet(self):
        prot = self._newProtocol()

        self.assertEqual(set(), prot._getAllDoneIds())

    def testOrientationStepIsNotReinsertedWhenAlreadyPublished(self):
        # Regression test: on Resume, insertedDict resets to {} just like
        # on a fresh run, so the "insert once" gate alone cannot tell them
        # apart. Restoring must rely on real evidence - the orientedGain
        # output already existing - not on isContinued() (which defaults
        # to MODE_RESUME even on a genuinely fresh launch).
        prot = self._newProtocol()
        prot.estimateOrientation = SimpleNamespace(get=lambda: True)
        prot.normalizeGain = SimpleNamespace(get=lambda: False)
        prot.convertCIStep = []
        prot.orientedGain = _FakeOutputSet({1})

        inserted = []
        prot._insertFunctionStep = (
            lambda name, *args, **kwargs: inserted.append(name) or len(inserted)
        )

        prot._insertNewMoviesSteps({}, _FakeMoviesList())

        self.assertNotIn('estimateOrientationStep', inserted)

    def testOrientationStepIsInsertedWhenNotYetPublished(self):
        prot = self._newProtocol()
        prot.estimateOrientation = SimpleNamespace(get=lambda: True)
        prot.normalizeGain = SimpleNamespace(get=lambda: False)
        prot.convertCIStep = []

        inserted = []
        prot._insertFunctionStep = (
            lambda name, *args, **kwargs: inserted.append(name) or len(inserted)
        )

        prot._insertNewMoviesSteps({}, _FakeMoviesList())

        self.assertIn('estimateOrientationStep', inserted)

    def testNormalizeStepIsNotReinsertedWhenStepAlreadyFinished(self):
        prot = self._newProtocol()
        prot.estimateOrientation = SimpleNamespace(get=lambda: False)
        prot.normalizeGain = SimpleNamespace(get=lambda: True)
        prot.convertCIStep = []
        prot._isFunctionStepFinished = (
            lambda name: name == 'normalizeGainStep'
        )

        inserted = []
        prot._insertFunctionStep = (
            lambda name, *args, **kwargs:
            inserted.append(name) or len(inserted)
        )

        prot._insertNewMoviesSteps({}, _FakeMoviesList())

        self.assertNotIn('normalizeGainStep', inserted)

    def testNormalizeStepIsInsertedWhenStepNotFinished(self):
        prot = self._newProtocol()
        prot.estimateOrientation = SimpleNamespace(get=lambda: False)
        prot.normalizeGain = SimpleNamespace(get=lambda: True)
        prot.convertCIStep = []
        prot._isFunctionStepFinished = lambda name: False

        inserted = []
        prot._insertFunctionStep = (
            lambda name, *args, **kwargs:
            inserted.append(name) or len(inserted)
        )

        prot._insertNewMoviesSteps({}, _FakeMoviesList())

        self.assertIn('normalizeGainStep', inserted)

    def testEstimatedIdsAreRestoredFromRealOutputOnResume(self):
        # Regression test: estimatedIds/estimatedResIds must be
        # reconstructed from the real, persisted gain outputs when
        # missing (e.g. after a genuine Resume), not left empty - which
        # would otherwise cause already-estimated gains to be silently
        # recomputed.
        prot = self._newProtocol()
        prot.estimatedGains = _FakeOutputSet({1, 2})
        prot.residualGains = _FakeOutputSet({1})
        prot.convertCIStep = []
        prot.estimateOrientation = SimpleNamespace(get=lambda: False)
        prot.normalizeGain = SimpleNamespace(get=lambda: False)
        prot._insertFunctionStep = lambda *args, **kwargs: 1

        prot._insertNewMoviesSteps({}, _FakeMoviesList())

        self.assertEqual({1, 2}, set(prot.estimatedIds))
        self.assertEqual({1}, set(prot.estimatedResIds))

# Finalization regression: the executor performs one last stepsCheck callback
# after it has already found no pending steps.
from unittest.mock import Mock

from pyworkflow.tests import BaseTest, setupTestProject
from xmipp3.protocols.protocol_movie_gain import XmippProtMovieGain


class TestXmippMovieGainFinalizationRegression(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testFinishedStepsCheckIsNoOp(self):
        prot = self.newProtocol(XmippProtMovieGain)
        prot.finished = True
        prot._checkNewInput = Mock()
        prot._checkNewOutput = Mock()

        prot._stepsCheck()

        prot._checkNewInput.assert_not_called()
        prot._checkNewOutput.assert_not_called()


class _RefreshRequiredMovieGainOutput:
    def __init__(self, ids):
        self.ids = set(ids)
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True

    def getIdSet(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted MovieGain output must be refreshed before reading ids.'
            )
        return set(self.ids)

    def getSize(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted MovieGain output must be refreshed before reading size.'
            )
        return len(self.ids)


class TestXmippMovieGainLogicalOutputRestore(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testResumeRefreshesPersistedGainOutputsBeforeReadingState(self):
        prot = self.newProtocol(
            XmippProtMovieGain,
            estimateGain=True,
            estimateResidualGain=True,
            estimateOrientation=True,
            normalizeGain=False,
        )

        estimated = _RefreshRequiredMovieGainOutput({1, 2})
        oriented = _RefreshRequiredMovieGainOutput({1})

        prot.estimatedGains = estimated
        prot.orientedGain = oriented

        prot._restoreEstimatedIds('estimatedIds', 'estimatedGains')
        alreadyPublished = prot._isOutputAlreadyPublished('orientedGain')

        self.assertEqual({1, 2}, set(prot.estimatedIds))
        self.assertTrue(alreadyPublished)
        self.assertTrue(estimated.loaded)
        self.assertTrue(oriented.loaded)
