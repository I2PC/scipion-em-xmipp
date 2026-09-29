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

import numpy as np
from pwem.objects import SetOfMovies
from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_movie_alignment_consensus import (
    ACCEPTED,
    XmippProtConsensusMovieAlignment,
)
from xmipp3.tests.streaming_test_utils import (
    FreshOutputSetProbe,
    LogicalOutputSetProbe,
)


class _FakeMovie:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _FakeMovie(self._objId)


class _FakeMovieSet:
    def __init__(self, ids, streamClosed=False):
        self._movies = [_FakeMovie(objId) for objId in ids]
        self._streamClosed = streamClosed
        self.closed = False

    def iterItems(self):
        return iter(self._movies)

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closed = True


class _FakeAcquisition:
    def clone(self):
        return _FakeAcquisition()


class _FakeOutputSet:
    def __init__(self, ids=None):
        self.ids = set(ids or [])
        self.appended = []
        self.closed = False

    def getIdSet(self):
        return set(self.ids)

    def getSize(self):
        return len(self.ids)

    def append(self, item):
        self.ids.add(item.getObjId())
        self.appended.append(item.getObjId())

    def setSamplingRate(self, value):
        pass

    def setAcquisition(self, acquisition):
        pass

    def close(self):
        self.closed = True


class _FakeAlignment:
    def getShifts(self):
        return [0.0, 1.0], [0.0, 1.0]


class _FakeAlignedMovie(_FakeMovie):
    def __init__(self, objId):
        super().__init__(objId)
        self._alignment = _FakeAlignment()

    def clone(self):
        return _FakeAlignedMovie(self._objId)

    def setEnabled(self, enabled):
        pass

    def getAlignment(self):
        return self._alignment

    def setAlignment(self, alignment):
        self._alignment = alignment


class _FakeMicrograph(_FakeMovie):
    def clone(self):
        return _FakeMicrograph(self._objId)

    def setEnabled(self, enabled):
        pass


class _FakeIndexedSet:
    def __init__(self, items):
        self.items = items
        self.closed = False

    def __getitem__(self, objId):
        return self.items[objId]

    def close(self):
        self.closed = True


class TestXmippMovieAlignmentConsensusRegression(BaseTest):
    """Regression tests for streaming and Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        return self.newProtocol(XmippProtConsensusMovieAlignment)

    def testStreamingInputDoesNotDependOnSqliteMtime(self):
        prot = self._newProtocol()
        prot.movieFn1 = 'movies1.sqlite'
        prot.movieFn2 = 'movies2.sqlite'
        prot.processedDict = []
        prot.insertedDict = {}
        prot.allMovies1 = {}
        prot.allMovies2 = {}

        movieSet1 = _FakeMovieSet([1, 2], streamClosed=False)
        movieSet2 = _FakeMovieSet([2, 3], streamClosed=False)
        movieSets = {
            prot.movieFn1: movieSet1,
            prot.movieFn2: movieSet2,
        }

        prot._loadInputMovieSet = lambda fn: movieSets[fn]
        prot._insertFunctionStep = lambda *args, **kwargs: 7
        prot._getFirstJoinStep = lambda: None

        updates = []
        prot.updateSteps = lambda: updates.append(True)

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.os.path.getmtime',
            side_effect=AssertionError('Streaming input must not depend on SQLite mtime.')
        ):
            prot._checkNewInput()

        self.assertEqual([2], prot.processedDict)
        self.assertEqual({2: 7}, prot.insertedDict)
        self.assertEqual(1, len(updates))
        self.assertTrue(movieSet1.closed)
        self.assertTrue(movieSet2.closed)

    def testResumeRestoresPersistedOutputsFromRealSets(self):
        # Regression test: Resume must reconstruct processedDict purely
        # from the real, persisted output Sets - no sidecar DONE files.
        prot = self._newProtocol()
        prot.outputMovies = _FakeOutputSet(ids=[1, 3])
        prot.outputMoviesDiscarded = _FakeOutputSet(ids=[2])

        prot._restoreStreamingState()

        self.assertEqual([1, 2, 3], prot.processedDict)

    def testResumeSkipsOnlyMoviesWithPersistedOutputs(self):
        prot = self._newProtocol()
        prot.allMovies1 = {1: None}
        prot.allMovies2 = {1: None}
        prot._originalRunMode = MODE_RESUME
        prot._isMovieOutputDone = lambda movieId: True
        prot.info = Mock()

        prot.alignmentCorrelationMovieStep(1)

        prot.info.assert_called_once_with(
            "Skipping movie with ID: 1, output already persisted"
        )

    def testRestartDoesNotReusePersistedOutputCheckpoint(self):
        prot = self._newProtocol()
        prot.allMovies1 = {1: None}
        prot.allMovies2 = {1: None}
        prot._originalRunMode = MODE_RESTART
        prot._isMovieOutputDone = lambda movieId: True
        prot.info = Mock()

        prot.alignmentCorrelationMovieStep(1)

        prot.info.assert_called_once_with(
            'AlignmentCorrelationMovieStep movie1 or movie2 are None'
        )

    def testNanCorrelationIsClassifiedAsDiscarded(self):
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESTART
        prot.minRangeShift.set(0)
        prot.minConsCorrelation.set(0.75)
        prot.allMovies1 = {1: _FakeAlignedMovie(1)}
        prot.allMovies2 = {1: _FakeAlignedMovie(1)}
        prot.stats = {}
        prot._decidedAccepted = []
        prot._decidedDiscarded = []
        prot._store = lambda *args, **kwargs: None

        nanCorrelation = np.array([[1.0, np.nan], [np.nan, 1.0]])
        with patch('xmipp3.protocols.protocol_movie_alignment_consensus.np.corrcoef', return_value=nanCorrelation):
            prot.alignmentCorrelationMovieStep(1)

        self.assertEqual([], prot._decidedAccepted)
        self.assertEqual([1], prot._decidedDiscarded)

    def testHighCorrelationIsClassifiedAsAccepted(self):
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESTART
        prot.minRangeShift.set(0)
        prot.minConsCorrelation.set(0.75)
        prot.allMovies1 = {1: _FakeAlignedMovie(1)}
        prot.allMovies2 = {1: _FakeAlignedMovie(1)}
        prot.stats = {}
        prot._decidedAccepted = []
        prot._decidedDiscarded = []
        prot._store = lambda *args, **kwargs: None

        highCorrelation = np.array([[1.0, 0.99], [0.99, 1.0]])
        with patch('xmipp3.protocols.protocol_movie_alignment_consensus.np.corrcoef', return_value=highCorrelation):
            prot.alignmentCorrelationMovieStep(1)

        self.assertEqual([1], prot._decidedAccepted)
        self.assertEqual([], prot._decidedDiscarded)

    def testMovieWithinRangeShiftThresholdIsRecordedAsAccepted(self):
        # The "flat trajectory, omit from consensus" branch must record
        # its decision through the same pending-list mechanism as the
        # normal accept/discard path.
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESTART
        prot.minRangeShift.set(2)
        prot.allMovies1 = {1: _FakeAlignedMovie(1)}
        prot.allMovies2 = {1: _FakeAlignedMovie(1)}
        prot.stats = {}
        prot._decidedAccepted = []
        prot._decidedDiscarded = []
        prot._store = lambda *args, **kwargs: None

        prot.alignmentCorrelationMovieStep(1)

        self.assertEqual([1], prot._decidedAccepted)
        self.assertEqual([], prot._decidedDiscarded)

    def testDirectMovieSetPointerResolvesParentMicrographs(self):
        prot = self._newProtocol()

        movieSet = SetOfMovies()
        movieSet._objParentId = 101
        prot.inputMovies1.set(movieSet)

        expectedPath = 'reference-micrographs.sqlite'
        parentProtocol = SimpleNamespace(
            outputMicrographs=SimpleNamespace(
                getFileName=lambda: expectedPath
            )
        )

        class _FakeProject:
            def getProtocol(self, protocolId):
                if protocolId != 101:
                    raise AssertionError('Unexpected protocol id.')
                return parentProtocol

        prot.getProject = lambda: _FakeProject()

        self.assertEqual(expectedPath, prot._getMicsPath())

    def testStreamingMovieOutputReusesLogicalSetWithoutLegacySqlite(self):
        class InputPointer:
            def get(self):
                return object()

        prot = self._newProtocol()
        logicalOutput = LogicalOutputSetProbe()
        prot.outputMovies = logicalOutput
        prot.inputMovies1 = InputPointer()
        prot._getPath = lambda baseName: '/tmp/' + baseName

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.os.path.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSetProbe,
                'movies.sqlite',
                fixSampling=False,
            )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming movie output must reuse the logical Set when "
            "the legacy SQLite file is absent.",
        )
        self.assertEqual(
            1,
            logicalOutput.enableAppendCalls,
        )

    def testGetAllDoneIdsReflectsPublishedOutputImmediately(self):
        # Regression test: done-tracking now reads the real output Sets
        # directly, so there is no separate checkpoint step that could get
        # out of order with publishing - a movie becomes "done" exactly
        # when (and only when) it is actually persisted into the output.
        prot = self._newProtocol()
        prot.allMovies1 = {1: object()}
        prot.allMovies2 = {1: object()}
        prot.isStreamClosed = True
        prot.samplingRate = 1.0
        prot.acquisition = _FakeAcquisition()
        prot._decidedAccepted = [1]
        prot._decidedDiscarded = []

        prot._loadOutputSet = lambda *args, **kwargs: _FakeOutputSet(ids=[1])
        prot.fillOutput = lambda *args, **kwargs: None
        prot._getFirstJoinStep = lambda: None
        prot._defineTransformRelation = lambda *args: None

        def realUpdateOutputSet(name, outputSet, streamMode):
            setattr(prot, name, outputSet)

        prot._updateOutputSet = realUpdateOutputSet

        self.assertEqual(([], []), prot._getAllDoneIds())

        prot._checkNewOutput()

        doneAccepted, doneDiscarded = prot._getAllDoneIds()
        self.assertEqual([1], doneAccepted)
        self.assertEqual([], doneDiscarded)

    def testExistingOutputRebuildsRelation(self):
        prot = self._newProtocol()
        prot.allMovies1 = {1: object()}
        prot.allMovies2 = {1: object()}
        prot.isStreamClosed = True
        prot.samplingRate = 1.0
        prot.acquisition = _FakeAcquisition()
        # These attributes already exist (e.g. from a prior movie) but do
        # not yet contain movie 1 - it is still a pending decision.
        prot.outputMovies = _FakeOutputSet(ids=[])
        prot.outputMicrographs = _FakeOutputSet(ids=[])
        prot._decidedAccepted = [1]
        prot._decidedDiscarded = []

        prot._loadOutputSet = lambda *args, **kwargs: _FakeOutputSet(ids=[1])
        prot.fillOutput = lambda *args, **kwargs: None
        prot._getFirstJoinStep = lambda: None

        events = []
        prot.mapper = SimpleNamespace(
            deleteRelations=lambda protocol: events.append(('delete-relations', None)),
            commit=lambda: events.append(('commit-relations', None)),
        )
        prot._updateOutputSet = lambda name, outputSet, streamMode: events.append(('update', name))
        prot._defineTransformRelation = lambda *args: events.append(('relation', None))

        prot._checkNewOutput()

        self.assertIn(('delete-relations', None), events)
        self.assertIn(('commit-relations', None), events)
        self.assertIn(('relation', None), events)

    def testFillOutputIsIdempotentAfterPartialPersistence(self):
        prot = self._newProtocol()
        prot.movieFn1 = 'movies.sqlite'
        prot.micsFn = 'micrographs.sqlite'
        prot.stats = {
            1: {
                'shift_corr': 1.0,
                'rmse_error': 0.0,
                'max_error': 0.0,
            }
        }

        inputMovies = _FakeIndexedSet({1: _FakeAlignedMovie(1)})
        inputMics = _FakeIndexedSet({1: _FakeMicrograph(1)})
        prot._loadInputMovieSet = lambda fn: inputMovies
        prot._loadInputMicrographSet = lambda fn: inputMics
        prot._getEnable = lambda movieId: True

        movieOutput = _FakeOutputSet(ids=[1])
        micOutput = _FakeOutputSet()

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.setAttribute',
            return_value=None,
        ):
            prot.fillOutput(movieOutput, micOutput, [1], ACCEPTED)

        self.assertEqual([], movieOutput.appended)
        self.assertEqual([1], micOutput.appended)

    def testFillOutputDoesNotQueryIdsFromFreshOutputSets(self):
        prot = self._newProtocol()
        prot.movieFn1 = 'movies.sqlite'
        prot.micsFn = 'micrographs.sqlite'
        prot.stats = {1: {'shift_corr': 1.0, 'rmse_error': 0.0, 'max_error': 0.0}}

        inputMovies = _FakeIndexedSet({1: _FakeAlignedMovie(1)})
        inputMics = _FakeIndexedSet({1: _FakeMicrograph(1)})
        prot._loadInputMovieSet = lambda fn: inputMovies
        prot._loadInputMicrographSet = lambda fn: inputMics
        prot._getEnable = lambda movieId: True

        class _FreshOutputSet(_FakeOutputSet):
            def getIdSet(self):
                raise AssertionError('Fresh output Set must not query IDs before its first append.')

        movieOutput = _FreshOutputSet()
        micOutput = _FreshOutputSet()

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.setAttribute',
            return_value=None,
        ):
            prot.fillOutput(movieOutput, micOutput, [1], ACCEPTED)

        self.assertEqual([1], movieOutput.appended)
        self.assertEqual([1], micOutput.appended)

    def testFinishedCheckDoesNotReloadOutputsWithoutNewMovies(self):
        """The final streaming check must not reopen already persisted outputs."""
        prot = self._newProtocol()
        prot.allMovies1 = {1: object()}
        prot.allMovies2 = {1: object()}
        prot.isStreamClosed = True
        prot.samplingRate = 1.0
        prot.acquisition = _FakeAcquisition()
        prot.outputMovies = _FakeOutputSet(ids=[1])
        prot._decidedAccepted = [1]
        prot._decidedDiscarded = []

        prot._loadOutputSet = Mock(return_value=_FakeOutputSet(ids=[1]))
        prot.fillOutput = Mock()
        prot._updateOutputSet = Mock()
        prot._refreshOutputRelations = Mock()
        prot._getFirstJoinStep = lambda: None

        prot._checkNewOutput()

        self.assertTrue(prot.finished)
        prot._loadOutputSet.assert_not_called()
        prot.fillOutput.assert_not_called()
        prot._updateOutputSet.assert_not_called()
        prot._refreshOutputRelations.assert_not_called()
