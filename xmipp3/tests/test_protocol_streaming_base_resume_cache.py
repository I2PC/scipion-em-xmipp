from unittest import TestCase

from pwem.objects import Movie

from xmipp3.protocols.protocol_streaming_base import XmippStreamingMoviesMixin


class TestXmippStreamingMovieResumeCache(TestCase):
    def testContinueGeneratorRestoresPersistedMovieIdsBeforeParallelSteps(self):
        class Harness(XmippStreamingMoviesMixin):
            def __init__(self):
                self.finished = True
                self.restoreCalls = []

            def isContinued(self):
                return True

            def isFailed(self):
                return False

            def _prepareStreamingGenerator(self):
                return None

            def _restorePersistedOutputIds(self, outputName):
                self.restoreCalls.append(outputName)
                self._persistedOutputIds = {outputName: {251}}
                return {251}

            def _restoreFinishedStreamingMovieSteps(self):
                return None

            def _finalizeStreamingGenerator(self):
                return None

        protocol = Harness()

        XmippStreamingMoviesMixin.stepsGeneratorStep(protocol)

        self.assertEqual(["outputMovies"], protocol.restoreCalls)

    def testContinueProcessMovieUsesRestoredMovieIdsWithoutRefreshingOutputSet(self):
        class Harness(XmippStreamingMoviesMixin):
            def __init__(self):
                self._persistedOutputIds = {"outputMovies": {251}}

            def isContinued(self):
                return True

            def _getPersistedOutputMovieIds(self):
                raise AssertionError("processMovieStep must not refresh the persisted output Set from a worker thread")

        movie = Movie()
        movie.setObjId(251)
        movieDict = movie.getObjDict(includeBasic=True)
        protocol = Harness()

        XmippStreamingMoviesMixin.processMovieStep(protocol, movieDict, False)
