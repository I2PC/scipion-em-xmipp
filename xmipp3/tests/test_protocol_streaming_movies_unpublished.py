# ***************************************************************************
# *
# * Streaming movie protocols (FlexAlign, movie gain): a movie whose result
# * cannot be published is left out, and the run still ends.
# *
# ***************************************************************************

import unittest

from xmipp3.protocols.protocol_streaming_base import XmippStreamingMoviesMixin


class _Movie:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _Movies(XmippStreamingMoviesMixin):
    """Publishes every movie but the ones it is told cannot be."""

    def __init__(self, unpublishable):
        self.listOfMovies = [_Movie(1), _Movie(2)]
        self.streamClosed = True
        self.finished = False
        self.persisted = set()
        self.published = []
        self.errors = []
        self._unpublishable = set(unpublishable)

    def _getPersistedOutputMovieIds(self):
        return set(self.persisted)

    def _getFinishedProcessMovieIds(self):
        return {1, 2}

    def _updateOutputSets(self, newDone, streamMode):
        ids = [movie.getObjId() for movie in newDone]
        self.published.append(ids)
        self.persisted.update(set(ids) - self._unpublishable)

    def _doMovieFolderCleanUp(self):
        return False

    def _getFirstJoinStep(self):
        return None

    def error(self, message):
        self.errors.append(message)


class TestAnUnpublishableMovieDoesNotHangTheRun(unittest.TestCase):
    def test_TheRunEndsWithoutIt(self):
        protocol = _Movies(unpublishable={2})

        protocol._checkNewOutput()

        self.assertTrue(protocol.finished)
        self.assertEqual({1}, protocol.persisted)
        self.assertTrue(protocol.errors)

    def test_ItIsNotTriedAgainOnEveryPoll(self):
        protocol = _Movies(unpublishable={2})
        protocol.streamClosed = False

        protocol._checkNewOutput()
        protocol._checkNewOutput()

        self.assertEqual([[1, 2]], protocol.published[:1])
        self.assertNotIn([2], protocol.published[1:])


if __name__ == '__main__':
    unittest.main()
