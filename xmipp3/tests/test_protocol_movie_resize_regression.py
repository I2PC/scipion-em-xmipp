# **************************************************************************
# *
# * Regression tests for movie resize streaming/resume recovery.
# *
# **************************************************************************

import unittest
from unittest.mock import patch

from pyworkflow.object import Set

from xmipp3.protocols.protocol_preprocess import protocol_movie_resize as movie_resize


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _InputSet:
    def __init__(self, dim=(400, 400, 10), sampling=1.5, frames=(1, 10, 1)):
        self.dim = dim
        self.sampling = sampling
        self.frames = frames

    def getDim(self):
        return self.dim

    def getSamplingRate(self):
        return self.sampling

    def getFramesRange(self):
        return self.frames


class _Pointer:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Movie:
    def __init__(self, obj_id):
        self.obj_id = obj_id

    def getObjId(self):
        return self.obj_id

    def clone(self):
        return _Movie(self.obj_id)

    def getAcquisition(self):
        return None


class _OutputMovie:
    def __init__(self):
        self.obj_id = None

    def setObjId(self, obj_id):
        self.obj_id = obj_id

    def getObjId(self):
        return self.obj_id

    def setFileName(self, file_name):
        self.file_name = file_name

    def setAcquisition(self, acquisition):
        self.acquisition = acquisition

    def setSamplingRate(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def setFramesRange(self, frames_range):
        self.frames_range = frames_range

    def getDim(self):
        return 128, 128, 10


class _OutputSet:
    def __init__(self, ids, events):
        self.ids = set(ids)
        self.events = events
        self.closed = False

    def getSize(self):
        return len(self.ids)

    def getIdSet(self):
        return set(self.ids)

    def isEmpty(self):
        return not self.ids

    def setDim(self, dim):
        self.dim = dim

    def append(self, movie):
        self.ids.add(movie.getObjId())
        self.events.append(('append', movie.getObjId()))

    def close(self):
        self.closed = True


class _Harness:
    def __init__(self, movie_ids, processed_ids, done_ids, output_ids, stream_closed):
        self.events = []
        self.listOfMovies = [_Movie(obj_id) for obj_id in movie_ids]
        self.processedIds = set(processed_ids)
        self.doneIds = set(done_ids)
        self.streamClosed = stream_closed
        self.finished = False
        self.outputSet = _OutputSet(output_ids, self.events)
        self.inputMovies = _Pointer(_InputSet())

    def _isMovieDone(self, movie):
        return movie.getObjId() in self.processedIds

    def _readDoneList(self):
        return list(self.doneIds)

    def _loadOutputSet(self, SetClass, baseName):
        return self.outputSet

    def _getNewSamplingRate(self):
        return 3.0

    def _getPath(self, name):
        return '/tmp/' + name

    def _updateOutputSet(self, outputName, outputSet, state):
        self.events.append(('output', state))
        outputSet.close()

    def _writeDoneList(self, movies):
        ids = [movie.getObjId() for movie in movies]
        self.doneIds.update(ids)
        self.events.append(('checkpoint', ids))

    def _getFirstJoinStep(self):
        return None


class _InsertHarness:
    def __init__(self):
        self.inserted = []

    def _insertMovieStep(self, movie):
        self.inserted.append(movie.getObjId())
        return movie.getObjId() + 100


class TestXmippMovieResizeRegression(unittest.TestCase):
    def testNewMoviesStepsUsesFreshInputSnapshot(self):
        protocol = _InsertHarness()
        inserted = {1: 101}
        movies = [_Movie(1), _Movie(2), _Movie(3)]
        deps = movie_resize.XmippProtMovieResize._insertNewMoviesSteps(protocol, inserted, movies)
        self.assertEqual([102, 103], deps)
        self.assertEqual([2, 3], protocol.inserted)

    def testOutputIsPersistedBeforeDoneCheckpoint(self):
        protocol = _Harness(movie_ids=[1, 2], processed_ids=[1, 2], done_ids=[1], output_ids=[1], stream_closed=False)
        with patch.object(movie_resize, 'Movie', _OutputMovie):
            movie_resize.XmippProtMovieResize._checkNewOutput(protocol)
        self.assertEqual([('append', 2), ('output', Set.STREAM_OPEN), ('checkpoint', [2])], protocol.events)

    def testReplayRepairsCheckpointWithoutDuplicateAppend(self):
        protocol = _Harness(movie_ids=[1], processed_ids=[1], done_ids=[], output_ids=[1], stream_closed=False)
        with patch.object(movie_resize, 'Movie', _OutputMovie):
            movie_resize.XmippProtMovieResize._checkNewOutput(protocol)
        self.assertEqual([('checkpoint', [1])], protocol.events)
        self.assertEqual({1}, protocol.outputSet.ids)

    def testFinishedReplayClosesOutputStream(self):
        protocol = _Harness(movie_ids=[1], processed_ids=[1], done_ids=[1], output_ids=[1], stream_closed=True)
        with patch.object(movie_resize, 'Movie', _OutputMovie):
            movie_resize.XmippProtMovieResize._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual([('output', Set.STREAM_CLOSED)], protocol.events)

    def testNewSamplingRateIsReconstructedFromParameters(self):
        protocol = type('SamplingHarness', (), {})()
        protocol.inputMovies = _Pointer(_InputSet(dim=(400, 400, 10), sampling=1.5))
        protocol.resizeOption = movie_resize.RESIZE_FACTOR
        protocol.resizeFactor = _Value(2.0)
        protocol.resizeDim = _Value(200)
        protocol.resizeSamplingRate = _Value(3.0)
        self.assertEqual(3.0, movie_resize.XmippProtMovieResize._getNewSamplingRate(protocol))


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import Mock

from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
    XmippProtMovieResize,
)


class TestXmippMovieResizeFinalizationRegression(unittest.TestCase):

    def testFinishedStepsCheckIsNoOp(self):
        class _Harness:
            finished = True

            def __init__(self):
                self._checkNewInput = Mock()
                self._checkNewOutput = Mock()

        protocol = _Harness()

        XmippProtMovieResize._stepsCheck(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()

