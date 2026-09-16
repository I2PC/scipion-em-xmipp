# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_eliminate_empty_images import (
    XmippProtEliminateEmptyParticles,
    XmippProtEliminateEmptyClasses,
)


class _FakeSet:
    def __init__(self, size, streamClosed=False):
        self._size = size
        self._streamClosed = streamClosed
        self.loaded = False
        self.closed = False

    def __len__(self):
        return self._size

    def loadAllProperties(self):
        self.loaded = True

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closed = True


class TestXmippEliminateEmptyResume(BaseTest):
    """Regression tests for eliminate-empty streaming Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testParticlesResumeUsesParticleOutputs(self):
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)

        self.assertEqual(
            ('outputParticles', 'eliminatedParticles'),
            prot._getResumeOutputNames()
        )

    def testClassesResumeUsesAverageOutputs(self):
        prot = self.newProtocol(XmippProtEliminateEmptyClasses)

        self.assertEqual(
            ('outputAverages', 'eliminatedAverages'),
            prot._getResumeOutputNames()
        )

    def testParticlesRestoreStreamingStateFromPersistedOutputs(self):
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)

        accepted = _FakeSet(7)
        eliminated = _FakeSet(3)
        inputSet = _FakeSet(12, streamClosed=False)

        prot.outputParticles = accepted
        prot.eliminatedParticles = eliminated
        prot.getInput = lambda: inputSet
        prot._getCreationCheckpoint = (
            lambda inputSet, processedCount: 'checkpoint-%d' % processedCount
        )
        prot.info = lambda *args, **kwargs: None

        prot._restoreStreamingState()

        self.assertEqual(10, prot.outputSize)
        self.assertEqual(12, prot.lenPartsSet)
        self.assertFalse(prot.streamClosed)
        self.assertEqual('checkpoint-10', prot.check)
        self.assertEqual(10, prot._scheduledSize)
        self.assertTrue(accepted.loaded)
        self.assertTrue(eliminated.loaded)
        self.assertTrue(accepted.closed)
        self.assertTrue(eliminated.closed)
        self.assertTrue(inputSet.closed)

    def testClassesRestoreStateWithoutCountingClassOutputsTwice(self):
        prot = self.newProtocol(XmippProtEliminateEmptyClasses)

        acceptedAverages = _FakeSet(8)
        eliminatedAverages = _FakeSet(2)
        inputSet = _FakeSet(15, streamClosed=True)

        prot.outputAverages = acceptedAverages
        prot.eliminatedAverages = eliminatedAverages

        # These outputs represent the same processed classes and must not
        # contribute again to outputSize on Resume.
        prot.outputClasses = _FakeSet(8)
        prot.eliminatedClasses = _FakeSet(2)

        prot.getInput = lambda: inputSet
        prot._getCreationCheckpoint = (
            lambda inputSet, processedCount: 'checkpoint-%d' % processedCount
        )
        prot.info = lambda *args, **kwargs: None

        prot._restoreStreamingState()

        self.assertEqual(10, prot.outputSize)
        self.assertEqual(15, prot.lenPartsSet)
        self.assertTrue(prot.streamClosed)
        self.assertEqual('checkpoint-10', prot.check)
        self.assertEqual(10, prot._scheduledSize)

    def testCheckNewInputDoesNotScheduleSameBatchTwice(self):
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)
        prot._scheduledSize = 10

        inputStates = iter([
            (12, False),
            (12, False),
            (15, True),
        ])
        scheduled = []
        updates = []

        prot._getCurrentInputState = lambda: next(inputStates)
        prot._insertNewPartsSteps = lambda: scheduled.append(True) or []
        prot._getFirstJoinStep = lambda: None
        prot.updateSteps = lambda: updates.append(True)

        prot._checkNewInput()
        prot._checkNewInput()
        prot._checkNewInput()

        self.assertEqual(
            2,
            len(scheduled),
            "A batch already represented by _scheduledSize must not be scheduled twice."
        )
        self.assertEqual(2, len(updates))
        self.assertEqual(15, prot._scheduledSize)
        self.assertEqual(15, prot.lenPartsSet)
        self.assertTrue(prot.streamClosed)

    def testCheckNewOutputCanFinishAfterResumeWithoutInputImages(self):
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)
