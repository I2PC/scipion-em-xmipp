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
    ACCEPTED,
)
from xmipp3.tests.streaming_test_utils import LogicalOutputSetProbe, FreshOutputSetProbe


class _FakeCreationParticle:
    def __init__(self, objId, creation):
        self._objId = objId
        self._creation = creation

    def getObjId(self):
        return self._objId

    def getObjCreation(self):
        return self._creation


class _FakePartsSet:
    def __init__(self, items):
        self._items = list(items)
        self.closed = False

    def __len__(self):
        return len(self._items)

    def iterItems(self, orderBy='id', direction='ASC'):
        items = sorted(self._items, key=lambda p: p.getObjId())
        if direction == 'DESC':
            items = list(reversed(items))
        return iter(items)

    def close(self):
        self.closed = True


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

    def testStreamingParticleOutputReusesLogicalSetWithoutLegacySqlite(self):
        from unittest.mock import patch

        class LogicalOutputSet:
            def __init__(self):
                self.enableAppendCalls = 0
                self.copyInfoCalls = 0

            def enableAppend(self):
                self.enableAppendCalls += 1

            def copyInfo(self, inputSet):
                self.copyInfoCalls += 1

        class FreshOutputSet:
            STREAM_OPEN = 1

            def __init__(self, filename=None):
                self.filename = filename

            def setStreamState(self, state):
                self.streamState = state

            def copyInfo(self, inputSet):
                self.inputSet = inputSet

        prot = self.newProtocol(XmippProtEliminateEmptyParticles)
        logicalOutput = LogicalOutputSetProbe()
        prot.outputParticles = logicalOutput
        prot.inputImages = object()
        prot._getPath = lambda baseName: '/tmp/' + baseName

        with patch(
            'xmipp3.protocols.protocol_eliminate_empty_images.os.path.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSetProbe,
                'outputParticles.sqlite',
            )

        self.assertIs(outputSet, logicalOutput)
        self.assertEqual(1, logicalOutput.enableAppendCalls)
        self.assertEqual(1, logicalOutput.copyInfoCalls)

    def testParticlesRestoreStreamingStateFromPersistedOutputs(self):
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)

        accepted = _FakeSet(7)
        eliminated = _FakeSet(3)
        inputSet = _FakeSet(12, streamClosed=False)

        prot.outputParticles = accepted
        prot.eliminatedParticles = eliminated
        prot.getInput = lambda: inputSet
        prot._getIdCheckpoint = (
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

    def testStreamingClassOutputReusesLogicalSetWithoutLegacySqlite(self):
        from unittest.mock import patch
        from pyworkflow.object import Set

        class LogicalClassSet:
            def __init__(self):
                self.enableAppendCalls = 0
                self.closed = False

            def enableAppend(self):
                self.enableAppendCalls += 1

            def copyInfo(self, inputSet):
                self.inputSet = inputSet

            def appendFromClasses(self, inputSet, enableFunc):
                self.enableFunc = enableFunc

            def setStreamState(self, state):
                self.streamState = state

            def write(self):
                pass

            def copy(self, other, copyId=False):
                raise AssertionError(
                    'Logical class output must be reused directly, not replaced.'
                )

            def close(self):
                self.closed = True

        class FakeClass:
            def __init__(self, objId):
                self._objId = objId

            def getObjId(self):
                return self._objId

        inputSet = [FakeClass(1)]
        prot = self.newProtocol(XmippProtEliminateEmptyClasses)
        logicalOutput = LogicalClassSet()
        prot.outputClasses = logicalOutput
        prot.classesDict = {1: 10}
        prot.getInput = lambda: inputSet
        prot._getPath = lambda baseName: '/tmp/' + baseName
        prot._store = lambda *args, **kwargs: None

        with patch(
            'xmipp3.protocols.protocol_eliminate_empty_images.os.path.exists',
            return_value=False,
        ), patch(
            'xmipp3.protocols.protocol_eliminate_empty_images.SetOfClasses2D',
            side_effect=AssertionError(
                'A fresh class Set must not be created when the logical output exists.'
            ),
        ):
            prot.createOutputClasses(
                'output',
                Set.STREAM_OPEN,
                {1: 1},
            )

        self.assertEqual(1, logicalOutput.enableAppendCalls)
        self.assertTrue(logicalOutput.closed)

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
        prot._getIdCheckpoint = (
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

    def testCheckpointIsNotAdvancedWhenEliminationJobFails(self):
        from unittest.mock import patch

        # Regression test: the creation-time checkpoint (self.check) must
        # only advance past a batch once its elimination job has actually
        # completed. Advancing it beforehand and then having the job fail
        # would make those items silently skipped forever - they would
        # never be re-included in a later batch, on retry or on Resume.
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)
        prot.check = None
        prot.fnInputMd = "/tmp/input%d.xmd"
        prot.fnOutputMd = "/tmp/output.xmd"
        prot.fnElimMd = "/tmp/eliminated.xmd"

        partsSet = _FakePartsSet([
            _FakeCreationParticle(1, "2026-09-01 00:00:00"),
            _FakeCreationParticle(2, "2026-09-02 00:00:00"),
        ])
        prot.prepareImages = lambda: partsSet

        def failingRunJob(*args, **kwargs):
            raise RuntimeError("simulated elimination job failure")

        prot.runJob = failingRunJob

        with patch(
            "xmipp3.protocols.protocol_eliminate_empty_images.writeSetOfParticles",
        ):
            with self.assertRaises(RuntimeError):
                prot.eliminationStep(1)

        self.assertIsNone(
            prot.check,
            "The checkpoint must not advance past a batch whose "
            "elimination job never actually completed.",
        )

    def testCheckpointAdvancesAfterEliminationJobSucceeds(self):
        from unittest.mock import patch

        prot = self.newProtocol(XmippProtEliminateEmptyParticles)
        prot.check = None
        prot.fnInputMd = "/tmp/input%d.xmd"
        prot.fnOutputMd = "/tmp/output.xmd"
        prot.fnElimMd = "/tmp/eliminated.xmd"

        partsSet = _FakePartsSet([
            _FakeCreationParticle(1, "2026-09-01 00:00:00"),
            _FakeCreationParticle(2, "2026-09-02 00:00:00"),
        ])
        prot.prepareImages = lambda: partsSet

        runCalls = []
        prot.runJob = lambda *args, **kwargs: runCalls.append(args)

        with patch(
            "xmipp3.protocols.protocol_eliminate_empty_images.writeSetOfParticles",
        ):
            prot.eliminationStep(1)

        self.assertEqual(1, len(runCalls))
        self.assertEqual(2, prot.check)

    def testEliminationStepFiltersInputByIdNotCreationTime(self):
        from unittest.mock import patch

        # Regression test: the previous creation-timestamp watermark had
        # only second-level precision under SQLite (no microseconds), so
        # two particles created within the same second could be silently
        # and permanently skipped once the strict '>' comparison moved
        # past that second. Filtering by id instead is immune to this,
        # since ids are always unique.
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)
        prot.check = 5
        prot.fnInputMd = "/tmp/input%d.xmd"
        prot.fnOutputMd = "/tmp/output.xmd"
        prot.fnElimMd = "/tmp/eliminated.xmd"

        partsSet = _FakePartsSet([
            _FakeCreationParticle(6, "2026-09-01 00:00:00"),
        ])
        prot.prepareImages = lambda: partsSet
        prot.runJob = lambda *args, **kwargs: None

        with patch(
            "xmipp3.protocols.protocol_eliminate_empty_images.writeSetOfParticles",
        ) as mockWrite:
            prot.eliminationStep(1)

        _, kwargs = mockWrite.call_args
        self.assertEqual('id > 5', kwargs.get('where'))
        self.assertEqual('id', kwargs.get('orderBy'))

    def testClassesSpecialBehavoirReturnsPendingCheckWithoutCommitting(self):
        # The classes variant has its own specialBehavoir (it also computes
        # accept/reject decisions via rejectByPopulation). It must follow
        # the same contract as the particles variant: return the pending
        # checkpoint instead of committing it directly to self.check.
        prot = self.newProtocol(XmippProtEliminateEmptyClasses)
        prot.check = None
        prot.classesDict = None

        partSet = _FakePartsSet([
            _FakeCreationParticle(5, "2026-09-01 00:00:00"),
            _FakeCreationParticle(6, "2026-09-03 00:00:00"),
        ])

        pendingCheck = prot.specialBehavoir(partSet)

        self.assertEqual(6, pendingCheck)
        self.assertIsNone(
            prot.check,
            "specialBehavoir must not commit the checkpoint itself - "
            "eliminationStep only commits it after the elimination job "
            "succeeds.",
        )
        self.assertTrue(partSet.closed)
        self.assertEqual({5: ACCEPTED, 6: ACCEPTED}, prot.enableCls)

    def testCheckNewOutputCanFinishAfterResumeWithoutInputImages(self):
        prot = self.newProtocol(XmippProtEliminateEmptyParticles)
