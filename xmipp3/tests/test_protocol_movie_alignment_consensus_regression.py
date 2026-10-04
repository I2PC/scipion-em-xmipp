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


class _FakePointer:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeWatermarkMovieItem:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _FakeWatermarkMovieItem(self._objId)


class _FakeWatermarkMovieSet:
    """Stands in for a real SetOfMovies, tracking which query mechanisms
    get used so tests can assert discovery is incremental (id > watermark
    / id IN (...)) and never a full scan or a filename reconstruction."""

    def __init__(self, items, streamClosed=False):
        self._items = {item.getObjId(): item for item in items}
        self._streamClosed = streamClosed
        self.uniqueCalls = []

    def loadAllProperties(self):
        pass

    def getUniqueValues(self, field, where=None):
        self.uniqueCalls.append((field, where))
        ids = sorted(self._items.keys())
        if where:
            threshold = int(where.split('>')[1].strip())
            ids = [i for i in ids if i > threshold]
        return ids

    def iterItems(self, orderBy=None, direction=None, where=None):
        if where and where.startswith('id IN'):
            idsStr = where[where.index('(') + 1: where.index(')')]
            wanted = {int(x) for x in idsStr.split(',')}
        else:
            wanted = set(self._items.keys())
        for i in sorted(wanted):
            if i in self._items:
                yield self._items[i]

    def isStreamClosed(self):
        return self._streamClosed

    def getFileName(self):
        raise AssertionError(
            "Movie discovery must not reconstruct a Set from a raw filename."
        )

    def close(self):
        pass


class _FakeIndexedSet:
    def __init__(self, items):
        self.items = items
        self.closed = False

    def __getitem__(self, objId):
        return self.items[objId]

    def __contains__(self, objId):
        return objId in self.items

    def close(self):
        self.closed = True


class TestXmippMovieAlignmentConsensusRegression(BaseTest):
    """Regression tests for streaming and Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        return self.newProtocol(XmippProtConsensusMovieAlignment)

    def testCheckNewInputDiscoversOnlyAboveWatermarkAndIntersectsBothSides(self):
        # Regression test: the old implementation reconstructed both
        # input Sets from a raw filename and fully re-scanned them
        # (iterItems() with no filter) on every single poll. This proves
        # discovery is now incremental (id > watermark) and that only a
        # movie id present on BOTH sides gets scheduled - the other side
        # stays pending and is retried later, never lost.
        prot = self._newProtocol()
        prot.processedDict = []
        prot.insertedDict = {}
        prot.allMovies1 = {}
        prot.allMovies2 = {}
        prot._movies1Watermark = 0
        prot._movies2Watermark = 0
        prot._pendingMovieIds1 = set()
        prot._pendingMovieIds2 = set()
        prot.newDeps = []

        movieSet1 = _FakeWatermarkMovieSet(
            [_FakeWatermarkMovieItem(1), _FakeWatermarkMovieItem(2)]
        )
        movieSet2 = _FakeWatermarkMovieSet(
            [_FakeWatermarkMovieItem(2), _FakeWatermarkMovieItem(3)]
        )
        prot.inputMovies1 = _FakePointer(movieSet1)
        prot.inputMovies2 = _FakePointer(movieSet2)

        prot._insertFunctionStep = lambda *args, **kwargs: 7
        updates = []
        prot.updateSteps = lambda: updates.append(True)

        prot._checkNewInput()

        # Only movie 2 is present on both sides - the only schedulable one.
        self.assertEqual([2], prot.processedDict)
        self.assertEqual({2: 7}, prot.insertedDict)
        self.assertEqual(1, len(updates))
        self.assertEqual(
            {1}, prot._pendingMovieIds1,
            "Movie 1 (only on side 1) must stay pending, not be dropped.",
        )
        self.assertEqual(
            {3}, prot._pendingMovieIds2,
            "Movie 3 (only on side 2) must stay pending, not be dropped.",
        )
        self.assertEqual({1, 2}, set(prot.allMovies1))
        self.assertEqual({2, 3}, set(prot.allMovies2))
        self.assertEqual(
            [('id', 'id > 0')],
            movieSet1.uniqueCalls,
            "Discovery must query only ids above the watermark, not scan "
            "everything.",
        )

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

    def testDirectMovieSetPointerResolvesParentMicrographOutputName(self):
        prot = self._newProtocol()

        movieSet = SetOfMovies()
        movieSet._objParentId = 101
        prot.inputMovies1.set(movieSet)

        parentProtocol = SimpleNamespace(outputMicrographs=object())

        class _FakeProject:
            def getProtocol(self, protocolId):
                if protocolId != 101:
                    raise AssertionError('Unexpected protocol id.')
                return parentProtocol

        prot.getProject = lambda: _FakeProject()

        self.assertEqual('outputMicrographs', prot._resolveMicsOutputName())

    def testStreamingMovieOutputReusesLogicalSet(self):
        class InputPointer:
            def get(self):
                return object()

        prot = self._newProtocol()
        logicalOutput = LogicalOutputSetProbe()
        prot.outputMovies = logicalOutput
        prot.inputMovies1 = InputPointer()

        outputSet = prot._loadOutputSet(
            FreshOutputSetProbe,
            'outputMovies',
            fixSampling=False,
        )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming movie output must reuse the persisted logical Set.",
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
        prot.fillOutput = lambda *args, **kwargs: [1]
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
        prot.fillOutput = lambda *args, **kwargs: [1]

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
        prot.stats = {
            1: {
                'shift_corr': 1.0,
                'rmse_error': 0.0,
                'max_error': 0.0,
            }
        }

        inputMovies = _FakeIndexedSet({1: _FakeAlignedMovie(1)})
        inputMics = _FakeIndexedSet({1: _FakeMicrograph(1)})
        prot._loadInputMovieSet = lambda: inputMovies
        prot._loadInputMicrographSet = lambda: inputMics
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
        prot.stats = {1: {'shift_corr': 1.0, 'rmse_error': 0.0, 'max_error': 0.0}}

        inputMovies = _FakeIndexedSet({1: _FakeAlignedMovie(1)})
        inputMics = _FakeIndexedSet({1: _FakeMicrograph(1)})
        prot._loadInputMovieSet = lambda: inputMovies
        prot._loadInputMicrographSet = lambda: inputMics
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

    def testFillOutputDefersMovieWhoseMicrographIsNotYetVisible(self):
        # Regression test: Set.__getitem__(int) returns None (not a
        # raise) for a missing row, but indexing straight into .clone()
        # would still crash. The micrograph Set is the output of a
        # different, independently-paced protocol, so a decided movieId
        # may not have a visible micrograph row yet - it must be
        # deferred to a later check, not crash the whole protocol.
        prot = self._newProtocol()
        prot.stats = {
            1: {'shift_corr': 1.0, 'rmse_error': 0.0, 'max_error': 0.0},
            2: {'shift_corr': 1.0, 'rmse_error': 0.0, 'max_error': 0.0},
        }

        inputMovies = _FakeIndexedSet({
            1: _FakeAlignedMovie(1),
            2: _FakeAlignedMovie(2),
        })
        inputMics = _FakeIndexedSet({2: _FakeMicrograph(2)})  # mic 1 missing
        prot._loadInputMovieSet = lambda: inputMovies
        prot._loadInputMicrographSet = lambda: inputMics
        prot._getEnable = lambda movieId: True
        prot.info = Mock()

        movieOutput = _FakeOutputSet()
        micOutput = _FakeOutputSet()

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.setAttribute',
            return_value=None,
        ):
            publishedIds = prot.fillOutput(movieOutput, micOutput, [1, 2], ACCEPTED)

        self.assertEqual([2], publishedIds)
        self.assertEqual([2], movieOutput.appended)
        self.assertEqual([2], micOutput.appended)

    def testCheckNewOutputDoesNotFinishPrematurelyWhenAMicrographIsPending(self):
        # Regression test: self.finished must be computed from what
        # fillOutput actually published, not from the originally-decided
        # newDone list - otherwise a movie deferred because its
        # micrograph isn't visible yet would still count as "done" and
        # could make the protocol finish while that movie is still
        # pending.
        prot = self._newProtocol()
        prot.allMovies1 = {1: object()}
        prot.allMovies2 = {1: object()}
        prot.isStreamClosed = True
        prot.samplingRate = 1.0
        prot.acquisition = _FakeAcquisition()
        prot._decidedAccepted = [1]
        prot._decidedDiscarded = []

        prot._loadOutputSet = lambda *args, **kwargs: _FakeOutputSet()
        prot.fillOutput = lambda *args, **kwargs: []  # movie 1 stayed pending
        prot._updateOutputSet = lambda *args, **kwargs: None

        prot._checkNewOutput()

        self.assertFalse(
            prot.finished,
            "A movie that fillOutput deferred (not actually published) "
            "must not be counted as done.",
        )

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

        prot._checkNewOutput()

        self.assertTrue(prot.finished)
        prot._loadOutputSet.assert_not_called()
        prot.fillOutput.assert_not_called()
        prot._updateOutputSet.assert_not_called()
        prot._refreshOutputRelations.assert_not_called()


class _RefreshRequiredConsensusOutput:
    def __init__(self, ids):
        self.ids = set(ids)
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True

    def getIdSet(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted consensus output must be refreshed before reading ids.'
            )
        return set(self.ids)


class TestXmippMovieAlignmentConsensusLogicalOutputRestore(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testGetAllDoneIdsRefreshesPersistedOutputs(self):
        prot = self.newProtocol(XmippProtConsensusMovieAlignment)
        prot.outputMovies = _RefreshRequiredConsensusOutput({1, 3})
        prot.outputMoviesDiscarded = _RefreshRequiredConsensusOutput({2})

        acceptedIds, discardedIds = prot._getAllDoneIds()

        self.assertTrue(prot.outputMovies.loaded)
        self.assertTrue(prot.outputMoviesDiscarded.loaded)
        self.assertEqual({1, 3}, set(acceptedIds))
        self.assertEqual({2}, set(discardedIds))
