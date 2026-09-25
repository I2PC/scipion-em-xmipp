# **************************************************************************
# *
# * Regression tests for Xmipp CTF streaming/batch failure recovery.
# *
# **************************************************************************

from pathlib import Path
import tempfile
import unittest

from xmipp3.protocols import protocol_ctf_micrographs as ctf_micrographs


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Mic:
    def getFileName(self):
        return "/tmp/mic_001.mrc"

    def getMicName(self):
        return "mic_001"

    def getObjId(self):
        return 1

    def getSamplingRate(self):
        return 1.0


class _CtfFailureHarness:
    def __init__(self, workdir):
        self.workdir = workdir
        self._XmippProtCTFMicrographs__params = {}
        self.windowSize = _Value(512)
        self.doInitialCTF = False
        self.findPhaseShift = False
        self.skipBorders = _Value(True)
        self.doOptimizeDefocus = True
        self._params = {
            "maxDefocus": 40000.0,
            "minDefocus": 5000.0,
        }
        self._args = ""
        self._program = "xmipp_ctf_estimate_from_micrograph"

    def _getMicBase(self, mic):
        return "mic_001"

    def _getMicrographDir(self, mic):
        return self.workdir

    def _getFileName(self, key, **kwargs):
        return str(Path(self.workdir) / (key + ".dat"))

    def _getExtraPath(self):
        return self.workdir

    def _calculateDownsampleList(self, samplingRate):
        return [1.0]

    def runJob(self, program, args):
        raise RuntimeError("synthetic CTF failure")

    def evaluateSingleMicrograph(self, mic):
        return True


class TestXmippCtfStreamingFailures(unittest.TestCase):
    def testCtfFailurePropagatesBeforeDoneCheckpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = _CtfFailureHarness(tmp)

            with self.assertRaisesRegex(RuntimeError, "synthetic CTF failure"):
                ctf_micrographs.XmippProtCTFMicrographs._estimateCTF(
                    protocol, _Mic()
                )


if __name__ == "__main__":
    unittest.main()
