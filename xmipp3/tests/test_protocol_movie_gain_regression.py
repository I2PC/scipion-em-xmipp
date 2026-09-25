# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_movie_gain import XmippProtMovieGain
from xmipp3.tests.streaming_test_utils import (
    FakeOutputSet as _FakeOutputSet,
    FreshOutputSetProbe,
    LogicalOutputSetProbe,
)


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

        def reloadInput():
            prot.listOfMovies = [_FakeMovie(1), _FakeMovie(2)]
            prot.streamClosed = False

        prot._loadInputList = reloadInput
        prot._getFirstJoinStep = lambda: None
        prot._insertNewMoviesSteps = lambda inserted, movies: scheduled.extend(m.getObjId() for m in movies if m.getObjId() not in inserted) or []
        prot.updateSteps = lambda: None

        prot._checkNewInput()

        self.assertEqual([2], scheduled)

    def testStreamingMovieOutputReusesLogicalSetWithoutLegacySqlite(self):
        from unittest.mock import patch

        class InputPointer:
            def get(self):
                return object()

        prot = self._newProtocol()
        logicalOutput = LogicalOutputSetProbe()
        prot.outputMovies = logicalOutput
        prot.inputMovies = InputPointer()
        prot._getPath = lambda baseName: '/tmp/' + baseName

        with patch(
            'xmipp3.protocols.protocol_movie_gain.os.path.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSetProbe,
                'movies.sqlite',
            )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming movie output must reuse the logical Set when "
            "the legacy SQLite file is absent.",
        )
        self.assertEqual(1, logicalOutput.enableAppendCalls)

    def testFreshOutputSetDoesNotQueryIds(self):
        self.assertEqual(set(), XmippProtMovieGain._getOutputIds(_FreshOutputSet()))

    def testOutputsAreIdempotentAndCheckpointIsLast(self):
        prot = self._newProtocol()
        movie = _FakeMovie(1)
        prot.listOfMovies = [movie]
        prot.streamClosed = False
        prot._isMovieDone = lambda movie: True
        prot._readDoneList = lambda: []
        prot.doGainProcess = lambda movieId: True
        prot.getEstimatedGainPath = lambda movieId: 'estimated_%d.xmp' % movieId
        prot.getResidualGainPath = lambda movieId: 'residual_%d.xmp' % movieId
        prot._getFirstJoinStep = lambda: None

        estimated = _FakeOutputSet({1})
        residual = _FakeOutputSet()
        movies = _FakeOutputSet()
        events = []

        def loadOutputSet(SetClass, baseName, fixGain=False):
            if baseName == prot.estimatedDatabase:
                return estimated
            if baseName == prot.residualDatabase:
                return residual
            return movies

        prot._loadOutputSet = loadOutputSet
        prot._updateOutputSet = lambda outputName, outputSet, state: events.append(outputName)
        prot._writeDoneList = lambda done: events.append('done')

        prot._checkNewOutput()

        self.assertEqual([], estimated.appended)
        self.assertEqual([1], residual.appended)
        self.assertEqual([1], movies.appended)
        self.assertEqual(['estimatedGains', 'residualGains', 'outputMovies', 'done'], events)

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

