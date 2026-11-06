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

    def copyObjId(self, _other):
        pass


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

                def _getMicBase(self, mic):
                    return "mic_%03d" % mic.getObjId()

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


class _PersistedCtfOutput:
    def __init__(self):
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True

    def getSize(self):
        return 0


class TestXmippCtfLogicalOutputResume(unittest.TestCase):
    def testOutputIsRefreshedBeforeTerminalStreamUpdate(self):
        outputSet = _PersistedCtfOutput()

        class _Harness(ctf_micrographs.XmippStreamingBase):
            _failedCtfMicIds = ctf_micrographs.XmippProtCTFMicrographs._failedCtfMicIds

            def __init__(self):
                self.finished = False
                self.streamClosed = True
                self.micDict = {}
                self.outputCTF = outputSet
                self.streamUpdated = False

            def _getFinishedProcessedMicKeys(self):
                return set()

            def _updateStreamState(self, streamMode):
                if not self.outputCTF.loaded:
                    raise AssertionError('Persisted output must be refreshed before the terminal stream update.')
                self.streamUpdated = True

        protocol = _Harness()

        ctf_micrographs.XmippProtCTFMicrographs._checkNewOutput(protocol)

        self.assertTrue(outputSet.loaded)
        self.assertTrue(protocol.finished)
        self.assertTrue(protocol.streamUpdated)


class _CtfPublishingHarness(ctf_micrographs.XmippStreamingBase):
    """Publishes the micrographs it is given; some have no CTF result."""

    _failedCtfMicIds = ctf_micrographs.XmippProtCTFMicrographs._failedCtfMicIds

    def __init__(self, withResult, persisted=()):
        self.finished = False
        self.streamClosed = True
        self.micDict = {"mic_%03d" % i: _NamedMic(i) for i in (1, 2)}
        self._withResult = set(withResult)
        self._persisted = set(persisted)
        self.published = []
        self.errors = []

    def _getFinishedProcessedMicKeys(self):
        return set(self.micDict)

    def _getKnownPersistedOutputIds(self, outputName):
        return set(self._persisted)

    def _ctfResultExists(self, mic):
        return mic.getObjId() in self._withResult

    def _updateOutputCTFSet(self, micList, streamMode):
        self.published.append([mic.getObjId() for mic in micList])
        return micList

    def _markOutputIdsPersisted(self, outputName, itemIds):
        self._persisted.update(itemIds)

    def _updateStreamState(self, streamMode):
        pass

    def error(self, message):
        self.errors.append(message)


class _NamedMic:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def getMicName(self):
        return "mic_%03d" % self._objId


class TestXmippCtfFailedEstimationsAreLeftOut(unittest.TestCase):
    def testAMicrographWithoutCtfIsNotPublishedAndTheRunEnds(self):
        protocol = _CtfPublishingHarness(withResult={1})

        ctf_micrographs.XmippProtCTFMicrographs._checkNewOutput(protocol)

        self.assertEqual([[1]], protocol.published)
        self.assertEqual({2}, protocol._failedCtfMicIds())
        self.assertTrue(protocol.finished)
        self.assertTrue(protocol.errors)

    def testEveryEstimationFailedFailsTheRun(self):
        protocol = _CtfPublishingHarness(withResult=())

        with self.assertRaisesRegex(Exception, "failed for every one of the 2"):
            ctf_micrographs.XmippProtCTFMicrographs._checkNewOutput(protocol)

        self.assertEqual([], protocol.published)

    def testAnEarlierRunsStepDoesNotCountAsThisRunsEstimation(self):
        import json
        from types import SimpleNamespace

        def step(micNames):
            return SimpleNamespace(
                funcName=SimpleNamespace(get=lambda: 'estimateCtfListStepOwn'),
                argsStr=SimpleNamespace(get=lambda default=None: json.dumps([micNames])),
                isFinished=lambda: True)

        protocol = SimpleNamespace(_prevSteps=[step(['mic_001'])],
                                   _steps=[step(['mic_002'])])

        self.assertEqual(
            {'mic_002'},
            ctf_micrographs.XmippProtCTFMicrographs._getFinishedProcessedMicKeys(protocol))

    def testContinueDoesNotEstimateAgainWhatWasPublished(self):
        from types import SimpleNamespace

        scheduled = []
        protocol = SimpleNamespace(
            micDict={}, newDeps=[], _pendingMicIds=set(),
            _loadInputList=lambda: ({"mic_001": _NamedMic(1),
                                     "mic_002": _NamedMic(2)}, False),
            _getKnownPersistedOutputIds=lambda name: {1},
            _insertNewMicsSteps=lambda mics: scheduled.extend(
                mic.getObjId() for mic in mics) or [],
            updateSteps=lambda: None)

        ctf_micrographs.XmippProtCTFMicrographs._checkNewInput(protocol)

        self.assertEqual([2], scheduled)
        self.assertIn("mic_001", protocol.micDict)


class _CtfPlaceholderHarness:
    def __init__(self, workdir, recalculate=False):
        self.workdir = workdir
        self.recalculate = recalculate

    def _getMicBase(self, mic):
        return "mic_%03d" % mic.getObjId()

    def _getExtraPath(self):
        return self.workdir

    def _getFileName(self, key, **kwargs):
        root = kwargs.get("root", self.workdir)
        micBase = kwargs.get("micBase", "mic_001")
        return str(Path(root) / ("%s_%s.xmd" % (micBase, key)))

    def _createErrorCtfParam(self, mic):
        return ctf_micrographs.XmippProtCTFMicrographs._createErrorCtfParam(
            self,
            mic,
        )

    def _setPsdFiles(self, ctfModel):
        pass


class TestXmippCtfDurableFailureOutput(unittest.TestCase):
    def testAMissingCtfResultIsNotPublishedInStreaming(self):
        # Left out, and estimated again on Continue: a disabled placeholder
        # would count as done and never be estimated again.
        with tempfile.TemporaryDirectory() as tmp:
            protocol = _CtfPlaceholderHarness(tmp)

            with self.assertRaisesRegex(RuntimeError, "No CTF was estimated"):
                ctf_micrographs.XmippProtCTFMicrographs._createCtfModel(
                    protocol, _Mic(), updateSampling=False)

    def testRecalculatingStillMarksAMissingCtfResultAsDisabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = _CtfPlaceholderHarness(tmp, recalculate=True)
            mic = _Mic()

            ctfModel = ctf_micrographs.XmippProtCTFMicrographs._createCtfModel(
                protocol,
                mic,
                updateSampling=False,
            )

            self.assertFalse(ctfModel.isEnabled())
            self.assertIs(
                mic,
                ctfModel.getMicrograph(),
            )
