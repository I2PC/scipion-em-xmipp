# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pwem.objects import SetOfMovies
from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_movie_alignment_consensus import (
    ACCEPTED,
    DISCARDED,
    XmippProtConsensusMovieAlignment,
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

    def testResumeRestoresPersistedOutputsAndDropsStaleSelections(self):
        prot = self._newProtocol()

        doneAcceptedFn = self.proj.getTmpPath('movie-consensus-done-accepted.txt')
        doneDiscardedFn = self.proj.getTmpPath('movie-consensus-done-discarded.txt')
        selectionAcceptedFn = self.proj.getTmpPath('movie-consensus-selection-accepted.txt')
        selectionDiscardedFn = self.proj.getTmpPath('movie-consensus-selection-discarded.txt')

        with open(doneAcceptedFn, 'w') as f:
            f.write('1\n3\n')
        with open(doneDiscardedFn, 'w') as f:
            f.write('2\n')
        with open(selectionAcceptedFn, 'w') as f:
            f.write('1 T\n99 T\n1 T\n3 T\n')
        with open(selectionDiscardedFn, 'w') as f:
            f.write('2 F\n88 F\n')

        prot._getCertainDone = (
            lambda label: doneAcceptedFn if label == ACCEPTED else doneDiscardedFn
        )
        prot._getMovieSelecFileAccepted = lambda: selectionAcceptedFn
        prot._getMovieSelecFileDiscarded = lambda: selectionDiscardedFn

        prot._restoreStreamingState()

        self.assertEqual([1, 2, 3], prot.processedDict)

        with open(selectionAcceptedFn) as f:
            self.assertEqual(['1 T\n', '3 T\n'], f.readlines())
        with open(selectionDiscardedFn) as f:
            self.assertEqual(['2 F\n'], f.readlines())

    def testResumeSkipsOnlyMoviesWithPersistedOutputs(self):
        prot = self._newProtocol()
        prot.allMovies1 = {1: None}
        prot.allMovies2 = {1: None}
        prot._originalRunMode = MODE_RESUME
        prot._isMovieOutputDone = lambda movieId: True

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.pwutils.cleanPath'
        ) as cleanPath:
            prot.alignmentCorrelationMovieStep(1)

        cleanPath.assert_not_called()

    def testRestartDoesNotReusePersistedOutputCheckpoint(self):
        prot = self._newProtocol()
        prot.allMovies1 = {1: None}
        prot.allMovies2 = {1: None}
        prot._originalRunMode = MODE_RESTART
        prot._isMovieOutputDone = lambda movieId: True

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.pwutils.cleanPath'
        ) as cleanPath:
            prot.alignmentCorrelationMovieStep(1)

        cleanPath.assert_called_once()

    def testNanCorrelationIsClassifiedAsDiscarded(self):
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESTART
        prot.minRangeShift.set(0)
        prot.minConsCorrelation.set(0.75)
        prot.allMovies1 = {1: _FakeAlignedMovie(1)}
        prot.allMovies2 = {1: _FakeAlignedMovie(1)}
        prot.stats = {}

        acceptedFn = self.proj.getTmpPath('movie-consensus-nan-accepted.txt')
        discardedFn = self.proj.getTmpPath('movie-consensus-nan-discarded.txt')
        doneFn = self.proj.getTmpPath('movie-consensus-nan-done.txt')

        prot._getMovieSelecFileAccepted = lambda: acceptedFn
        prot._getMovieSelecFileDiscarded = lambda: discardedFn
        prot._getMovieDone = lambda movieId: doneFn
        prot._store = lambda *args, **kwargs: None

        nanCorrelation = np.array([[1.0, np.nan], [np.nan, 1.0]])
        with patch('xmipp3.protocols.protocol_movie_alignment_consensus.np.corrcoef', return_value=nanCorrelation):
            prot.alignmentCorrelationMovieStep(1)

        self.assertEqual([], prot._readtMovieId(True))
        self.assertEqual([1], prot._readtMovieId(False))

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
        class LogicalOutputSet:
            def __init__(self):
                self.enableAppendCalls = 0

            def enableAppend(self):
                self.enableAppendCalls += 1

        class FreshOutputSet:
            STREAM_OPEN = 1

            def __init__(self, filename=None):
                self.filename = filename

            def setStreamState(self, state):
                self.streamState = state

            def copyInfo(self, inputSet):
                self.inputSet = inputSet

        class InputPointer:
            def get(self):
                return object()

        prot = self._newProtocol()
        logicalOutput = LogicalOutputSet()
        prot.outputMovies = logicalOutput
        prot.inputMovies1 = InputPointer()
        prot._getPath = lambda baseName: '/tmp/' + baseName

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.os.path.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSet,
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

    def testOutputCheckpointIsWrittenAfterOutputSetsAreUpdated(self):
        prot = self._newProtocol()
        prot.allMovies1 = {1: object()}
        prot.allMovies2 = {1: object()}
        prot.isStreamClosed = True
        prot.samplingRate = 1.0
        prot.acquisition = _FakeAcquisition()

        prot._readCertainDoneList = lambda label: []
        prot._readtMovieId = lambda accepted: [1] if accepted else []
        prot._loadOutputSet = lambda *args, **kwargs: _FakeOutputSet()
        prot.fillOutput = lambda *args, **kwargs: None
        prot._getFirstJoinStep = lambda: None

        events = []
        prot._updateOutputSet = (
            lambda name, outputSet, streamMode: events.append(('update', name))
        )
        prot._defineTransformRelation = (
            lambda *args: events.append(('relation', None))
        )
        prot._writeCertainDoneList = (
            lambda movieId, label: events.append(('checkpoint', label, movieId))
        )

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.os.path.exists',
            return_value=False
        ):
            prot._checkNewOutput()

        updateIndexes = [
            index for index, event in enumerate(events) if event[0] == 'update'
        ]
        checkpointIndex = next(
            index for index, event in enumerate(events) if event[0] == 'checkpoint'
        )

        self.assertEqual(2, len(updateIndexes))
        self.assertGreater(checkpointIndex, max(updateIndexes))

    def testExistingOutputRebuildsRelationBeforeCheckpoint(self):
        prot = self._newProtocol()
        prot.allMovies1 = {1: object()}
        prot.allMovies2 = {1: object()}
        prot.isStreamClosed = True
        prot.samplingRate = 1.0
        prot.acquisition = _FakeAcquisition()
        prot.outputMovies = _FakeOutputSet(ids=[1])
        prot.outputMicrographs = _FakeOutputSet(ids=[1])

        prot._readCertainDoneList = lambda label: []
        prot._readtMovieId = lambda accepted: [1] if accepted else []
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
        prot._writeCertainDoneList = lambda movieId, label: events.append(('checkpoint', label, movieId))

        prot._checkNewOutput()

        relationIndex = next(index for index, event in enumerate(events) if event[0] == 'relation')
        checkpointIndex = next(index for index, event in enumerate(events) if event[0] == 'checkpoint')

        self.assertLess(relationIndex, checkpointIndex)
        self.assertIn(('delete-relations', None), events)
        self.assertIn(('commit-relations', None), events)

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
        prot._writeCertainDoneList = (
            lambda *args: (_ for _ in ()).throw(
                AssertionError('Output checkpoint must not be written from fillOutput.')
            )
        )

        movieOutput = _FakeOutputSet(ids=[1])
        micOutput = _FakeOutputSet()

        with patch(
            'xmipp3.protocols.protocol_movie_alignment_consensus.setAttribute',
            lambda *args, **kwargs: None
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

        with patch('xmipp3.protocols.protocol_movie_alignment_consensus.setAttribute', lambda *args, **kwargs: None):
            prot.fillOutput(movieOutput, micOutput, [1], ACCEPTED)

        self.assertEqual([1], movieOutput.appended)
        self.assertEqual([1], micOutput.appended)

    def testSelectionReaderDeduplicatesMovieIds(self):
        prot = self._newProtocol()
        selectionFn = self.proj.getTmpPath('movie-consensus-selection-duplicates.txt')

        with open(selectionFn, 'w') as f:
            f.write('1 T\n2 T\n1 T\n2 T\n')

        prot._getMovieSelecFileAccepted = lambda: selectionFn

        self.assertEqual([1, 2], prot._readtMovieId(True))
