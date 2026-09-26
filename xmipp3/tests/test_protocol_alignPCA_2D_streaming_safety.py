from unittest import TestCase
from unittest.mock import patch

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
