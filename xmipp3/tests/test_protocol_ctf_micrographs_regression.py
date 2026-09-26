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
    def testCtfFailureDoesNotAbortFollowingMicrographs(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempted = []

            class _StreamingHarness(_CtfFailureHarness):
                def _estimateCTF(self, mic, *args):
                    return (
                        ctf_micrographs.XmippProtCTFMicrographs._estimateCTF(
                            self, mic, *args
                        )
                    )

                def runJob(self, program, args):
                    attempted.append(args)
                    if len(attempted) == 1:
                        raise RuntimeError("synthetic CTF failure")

                def _getMicrographDir(self, mic):
                    micDir = Path(tmp) / ("mic_%03d" % mic.getObjId())
                    micDir.mkdir(exist_ok=True)
                    return str(micDir)

                def _getFileName(self, key, **kwargs):
                    root = kwargs.get("root", tmp)
                    return str(Path(root) / (key + ".dat"))

                def evaluateSingleMicrograph(self, mic):
                    return True

            class _MicWithId(_Mic):
                def __init__(self, objId):
                    self.objId = objId

                def getObjId(self):
                    return self.objId

                def getFileName(self):
                    return "/tmp/mic_%03d.mrc" % self.objId

                def getMicName(self):
                    return "mic_%03d" % self.objId

            protocol = _StreamingHarness(tmp)
            protocol._calculateDownsampleList = lambda samplingRate: [1.0]
            protocol.evaluateSingleMicrograph = lambda mic: True

            ctf_micrographs.ProtCTFMicrographs._estimateCtfList(
                protocol,
                [_MicWithId(1), _MicWithId(2)],
            )

            self.assertGreaterEqual(
                len(attempted),
                2,
                "A failed CTF estimation must not abort later micrographs "
                "in the streaming batch.",
            )


if __name__ == "__main__":
    unittest.main()
