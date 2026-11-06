from unittest import TestCase

from xmipp3.protocols.protocol_streaming_base import XmippStreamingMoviesMixin


class TestXmippStreamingGeneratorFailure(TestCase):
    def testGeneratorStopsWhenParallelMovieStepFails(self):
        class Harness(XmippStreamingMoviesMixin):
            def __init__(self):
                self.finished = False
                self.failed = False
                self.finalized = False

            def isContinued(self):
                return False

            def isFailed(self):
                return self.failed

            def _prepareStreamingGenerator(self):
                return None

            def _restoreFinishedStreamingMovieSteps(self):
                return None

            def _checkNewInput(self):
                self.failed = True

            def _checkNewOutput(self):
                return None

            def _getStreamingSleepOnWait(self):
                raise AssertionError("Generator kept running after the protocol failed")

            def _finalizeStreamingGenerator(self):
                self.finalized = True

        protocol = Harness()

        XmippStreamingMoviesMixin.stepsGeneratorStep(protocol)

        self.assertFalse(protocol.finalized)
