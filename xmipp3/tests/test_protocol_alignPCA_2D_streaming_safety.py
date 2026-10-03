from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from pyworkflow.object import Set

from xmipp3.protocols.protocol_alignPCA_2D import XmippProtClassifyPcaStreaming


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _PcaStreamingHarness(XmippProtClassifyPcaStreaming):
    def __init__(self):
        self.classificationLaunch = False
        self.classificationBatch = _Value(100)
        self.lastRound = False

        self.correctCtf = _Value(True)
        self.classificationMPIs = _Value(4)
        self.numberOfMpi = _Value(2)
        self.mode = _Value(self.CREATE_CLASSES)
        self.firstTimeDone = False
        self.sampling = 1.5

        self.runCalls = []

    def runJob(self, program, args, **kwargs):
        self.runCalls.append((program, args, kwargs))


class TestXmippPcaStreamingSafety(TestCase):
    def test_ClosedStreamClassifiesFinalBatchBelowThreshold(self):
        protocol = _PcaStreamingHarness()
        protocol.classificationBatch = _Value(75000)
        protocol.lastRound = True

        self.assertTrue(
            protocol._doClassification([object()] * 17234),
            "Once the input stream is closed, the final non-empty batch "
            "must be classified even when it is smaller than "
            "classificationBatch.",
        )

        protocol.lastRound = False
        self.assertFalse(
            protocol._doClassification([object()] * 17234),
            "While the stream is still open, a batch below "
            "classificationBatch must keep waiting for more particles.",
        )

    def test_CtfCorrectionUsesReservedProtocolMpis(self):
        protocol = _PcaStreamingHarness()

        module = "xmipp3.protocols.protocol_alignPCA_2D"
        with patch(module + ".writeSetOfParticles", return_value=None):
            protocol.convertInputStep(
                object(),
                "/tmp/imagesInput_0.xmd",
                "/tmp/images_0.mrc",
            )

        ctfCalls = [
            kwargs
            for program, _, kwargs in protocol.runCalls
            if program == "xmipp_ctf_correct_wiener2d"
        ]

        self.assertEqual(
            1,
            len(ctfCalls),
            "The CTF correction must be launched exactly once.",
        )
        self.assertEqual(
            2,
            ctfCalls[0].get("numberOfMpi"),
            "The CTF correction must use numberOfMpi, because that is "
            "the MPI resource Scipion reserves for the protocol.",
        )

    def test_DoClassificationRefusesNewRoundWhileOneIsInFlight(self):
        # Regression test: a classification round writes to fixed
        # (non round-versioned) external files (classes_classes.star,
        # classes_images.star). This protocol runs under STEPS_PARALLEL
        # with no prerequisite chaining between rounds, so only this
        # flag prevents a second round's runClassificationSteps from
        # overwriting those files while a previous round's
        # updateOutputSetOfClasses is still reading them.
        protocol = _PcaStreamingHarness()
        protocol.classificationBatch = _Value(100)
        protocol.classificationLaunch = True

        self.assertFalse(
            protocol._doClassification([object()] * 500),
            "A new classification round must not be launched while a "
            "previous round's external files may still be in use.",
        )

    def test_InsertClassificationStepsMarksClassificationAsLaunched(self):
        protocol = _PcaStreamingHarness()
        protocol.imgsOrigXmd = '/tmp/imagesInput_.xmd'
        protocol.imgsXmd = '/tmp/images_.xmd'
        protocol.imgsFn = '/tmp/images_.mrc'
        protocol.classificationRound = 0
        protocol.newDeps = []
        protocol.info = Mock()
        protocol._insertFunctionStep = Mock(side_effect=lambda *a, **k: object())

        protocol._insertClassificationSteps(
            newParticlesSet=[object()],
            lastInputId=7,
        )

        self.assertTrue(
            protocol.classificationLaunch,
            "Inserting a classification round's steps must mark a "
            "round as in-flight, so no other round is launched while "
            "its external files are still being written/read.",
        )

    def test_UpdateOutputSetOfClassesClearsClassificationLaunchAfterReadingExternalFiles(self):
        protocol = _PcaStreamingHarness()
        protocol.classificationLaunch = True
        protocol.classificationRound = 0
        protocol.info = Mock()
        protocol._loadOutputSet = Mock(return_value=(MagicMock(), False))
        protocol._fillClassesFromLevel = Mock()
        protocol._updateOutputSet = Mock()
        protocol._defineSourceRelation = Mock()
        protocol._getInputPointer = Mock(return_value=object())

        protocol.updateOutputSetOfClasses(7, Set.STREAM_OPEN)

        self.assertFalse(
            protocol.classificationLaunch,
            "Once a round's external files have been fully read, the "
            "next round must be allowed to launch.",
        )
