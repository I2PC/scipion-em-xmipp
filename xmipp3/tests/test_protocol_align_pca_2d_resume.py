# ******************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# ******************************************************************************

from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols import XmippProtClassifyPcaStreaming


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
