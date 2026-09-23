# ******************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# ******************************************************************************

from unittest.mock import patch

from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols import XmippProtClassifyPcaStreaming


class _LogicalParticles:
    def __init__(self):
        self.loadCalls = 0

    def getFileName(self):
        raise AssertionError(
            "Streaming must not use the Set storage filename as its "
            "authoritative input."
        )

    def loadAllProperties(self):
        self.loadCalls += 1


class _Pointer:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _EmptyParticles:
    def __init__(self):
        self.infoSource = None

    def copyInfo(self, source):
        self.infoSource = source


class _StreamingSetContractHarness:
    def __init__(self, particles):
        self.inputParticles = _Pointer(particles)
        self.inputFn = "/tmp/compatibility.sqlite"
        self.emptyParticles = _EmptyParticles()

    def debug(self, *args, **kwargs):
        pass

    def _createSetOfParticles(self):
        return self.emptyParticles


class TestXmippClassifyPcaResume(BaseTest):
    """Regression tests for PCA2D streaming Continue/Restart handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newPcaProtocol(self):
        return self.newProtocol(XmippProtClassifyPcaStreaming)

    def _prepareGeneratorProtocol(self, originalRunMode, hasCheckpoint):
        prot = self._newPcaProtocol()

        # Reproduce Protocol._runSteps(): Scipion changes runMode to RESUME
        # while _originalRunMode keeps the action selected by the user.
        prot._originalRunMode = originalRunMode
        prot.runMode.set(MODE_RESUME)

        # We only want to exercise the startup/Resume decision here.
        prot.finish = True
        prot._initialStep = lambda: None
        prot._loadEmptyParticleSet = lambda: object()
        prot._hasStreamingCheckpoint = lambda: hasCheckpoint

        resumeCalls = []
        prot._updateVarsToContinue = lambda: resumeCalls.append(True)

        return prot, resumeCalls

    def testStreamingReadsTheLogicalInputSetInsteadOfReconstructingStorage(self):
        particles = _LogicalParticles()
        harness = _StreamingSetContractHarness(particles)

        module = "xmipp3.protocols.protocol_alignPCA_2D"

        with patch(
            module + ".SetOfParticles",
            side_effect=AssertionError(
                "Streaming must not reconstruct SetOfParticles from a "
                "compatibility storage filename."
            ),
        ):
            loadedParticles = (
                XmippProtClassifyPcaStreaming._loadInputParticleSet(harness)
            )
            emptyParticles = (
                XmippProtClassifyPcaStreaming._loadEmptyParticleSet(harness)
            )

        self.assertIs(
            particles,
            loadedParticles,
            "Streaming must refresh and iterate the logical input Set.",
        )
        self.assertIs(
            particles,
            emptyParticles.infoSource,
            "The temporary batch Set must copy metadata from the logical input Set.",
        )
        self.assertGreaterEqual(
            particles.loadCalls,
            2,
            "Both streaming reads must refresh the logical Set state.",
        )

    def testResumeRestoresClassificationCheckpoint(self):
        prot = self._newPcaProtocol()

        prot._hasStreamingCheckpoint = lambda: True
        prot._getLastDone = lambda: "2026-09-16 10:11:12.123456"
        prot._getLastClassificationRound = lambda: 4

        prot._updateVarsToContinue()

        self.assertEqual("2026-09-16 10:11:12.123456", prot.lastCreationTime)
        self.assertEqual(5, prot.classificationRound)

    def testResumeWithoutCheckpointStartsFromInitialState(self):
        prot = self._newPcaProtocol()

        prot.lastCreationTime = "stale-value"
        prot.classificationRound = 99
        prot._hasStreamingCheckpoint = lambda: False

        prot._updateVarsToContinue()

        self.assertEqual("", prot.lastCreationTime)
        self.assertEqual(1, prot.classificationRound)

    def testResumeUpdateClassesPreservesUpdatedReferences(self):
        prot = self._newPcaProtocol()

        prot.mode.set(prot.UPDATE_CLASSES)
        prot.firstTimeDone = False
        prot._hasStreamingCheckpoint = lambda: True
        prot._getLastDone = lambda: "2026-09-16 10:11:12.123456"
        prot._getLastClassificationRound = lambda: 4

        prot._updateVarsToContinue()

        self.assertTrue(prot.firstTimeDone, "Continue in UPDATE_CLASSES mode must preserve the classes produced by previous rounds.")

    def testStepsGeneratorRestoresStateOnRealResume(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESUME, True)

        prot.stepsGeneratorStep()

        self.assertEqual(1, len(resumeCalls), "Continue with a checkpoint must restore the previous PCA2D streaming state.")

    def testStepsGeneratorDoesNotRestoreFreshDefaultResume(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESUME, False)

        prot.stepsGeneratorStep()

        self.assertEqual(0, len(resumeCalls), "A fresh protocol uses MODE_RESUME by default but must not restore nonexistent streaming state.")

    def testStepsGeneratorDoesNotRestoreStateOnRestart(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESTART, True)

        prot.stepsGeneratorStep()

        self.assertEqual(0, len(resumeCalls), "Restart must not restore the previous PCA2D streaming state.")
