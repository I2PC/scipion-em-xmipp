# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from unittest.mock import Mock, patch

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_tilt_analysis import XmippProtTiltAnalysis
from xmipp3.tests.streaming_test_utils import (
    FakeOutputSet as _FakeOutputSet,
    FreshOutputSetProbe,
    LogicalOutputSetProbe,
)


class _FakeInputSet:
    def __init__(self, ids, streamClosed=False, gettableIds=None):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self._gettableIds = set(ids) if gettableIds is None else set(gettableIds)
        self.closed = False
        self.loadCalls = 0

    def loadAllProperties(self):
        self.loadCalls += 1

    def __contains__(self, itemId):
        return itemId in self._gettableIds

    def getUniqueValues(self, attr, where=None):
        if where is None:
            return sorted(self._ids)
        threshold = int(where.split('>')[1].strip())
        return sorted(itemId for itemId in self._ids if itemId > threshold)

    def getItem(self, field, value):
        return _FakeMicrograph(value)

    def getIdSet(self):
        return set(self._ids)

    def getSize(self):
        return len(self._ids)

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closed = True


class _FakePointer:
    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj


class _FakeMicrograph:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _FakeMicrograph(self._objId)


class TestXmippTiltAnalysisRegression(BaseTest):
    """Regression tests for Tilt Analysis streaming and Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtTiltAnalysis, autoWindow=False, window_size=512, objective_resolution=8)
        prot.micsFn = 'micrographs.sqlite'
        prot.insertedIds = []
        prot.processedIds = []
        prot.stats = {}
        return prot

    def testNewInputDoesNotDependOnSqliteMtime(self):
        prot = self._newProtocol()
        prot.insertedIds = [1]
        prot._lastInputId = 1
        inputSet = _FakeInputSet([1, 2])
        scheduled = []

        prot.inputMicrographs = _FakePointer(inputSet)
        prot.newDeps = []
        prot.isContinued = lambda: False
        prot._insertNewMicrographSteps = lambda ids: scheduled.append(sorted(ids)) or []
        prot.updateSteps = lambda: None

        with patch('xmipp3.protocols.protocol_tilt_analysis.os.path.getmtime', side_effect=AssertionError('Streaming input must not depend on SQLite mtime.')):
            prot._checkNewInput()

        self.assertEqual([[2]], scheduled)
        self.assertTrue(inputSet.closed)

    def testStreamingOutputReusesLogicalSetWithoutLegacySqlite(self):
        class InputPointer:
            def get(self):
                return object()

        prot = self._newProtocol()
        logicalOutput = LogicalOutputSetProbe()
        prot.outputMicrographs = logicalOutput
        prot.inputMicrographs = InputPointer()
        prot._getPath = lambda baseName: '/tmp/' + baseName

        with patch(
            'xmipp3.protocols.protocol_tilt_analysis.os.path.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSetProbe,
                'micrograph.sqlite',
            )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming output must reuse the logical Set when the "
            "legacy SQLite file is absent.",
        )
        self.assertEqual(
            1,
            logicalOutput.enableAppendCalls,
        )

    def testAppendNewMicrographsSkipsAlreadyPersistedIds(self):
        prot = self._newProtocol()
        outputSet = _FakeOutputSet([1])

        prot._appendNewMicrographs(outputSet, [_FakeMicrograph(1), _FakeMicrograph(2)])

        self.assertEqual({1, 2}, outputSet.ids)
        self.assertEqual([2], outputSet.appended)

    def testAppendNewMicrographsDoesNotQueryIdsOnFreshSet(self):
        prot = self._newProtocol()

        class _FreshOutputSet(_FakeOutputSet):
            def getIdSet(self):
                raise AssertionError('Fresh output Set must not query IDs before its first append.')

        outputSet = _FreshOutputSet()
        prot._appendNewMicrographs(outputSet, [_FakeMicrograph(1)])

        self.assertEqual({1}, outputSet.ids)
        self.assertEqual([1], outputSet.appended)

    def testProcessMicrographListStepDiscardsMicThatNeverBecomesVisible(self):
        # Regression test: Set.getItem raises (UnboundLocalError) rather
        # than returning None for a row it cannot find. A micId that never
        # becomes visible must still be recorded in processedIds (with no
        # stats), or it would never contribute to the "done" accounting
        # and the protocol would never reach finished=True.
        prot = self._newProtocol()
        inputSet = _FakeInputSet([1], gettableIds=set())
        prot.inputMicrographs = _FakePointer(inputSet)
        prot._processMicrograph = Mock()

        with patch(
                'xmipp3.protocols.protocol_tilt_analysis.time.sleep',
                return_value=None,
        ):
            prot.processMicrographListStep([1])

        self.assertEqual([1], prot.processedIds)
        self.assertNotIn(1, prot.stats)
        prot._processMicrograph.assert_not_called()

    def testProcessMicrographListStepMarksProcessedOnComputationFailureAndKeepsProcessingOthers(self):
        # Regression test: a genuine processing failure (e.g. a corrupted
        # or unreadable micrograph image) must not crash the whole batch
        # step - and hence the whole protocol via pyworkflow's
        # fail-on-any-exception step boundary. It must be marked processed
        # with no statistics while the rest of the batch is still
        # evaluated.
        prot = self._newProtocol()
        inputSet = _FakeInputSet([1, 2])
        prot.inputMicrographs = _FakePointer(inputSet)

        def fakeProcessMicrograph(micrograph):
            if micrograph.getObjId() == 1:
                raise ValueError("corrupted micrograph image")
            prot.stats[micrograph.getObjId()] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            prot.processedIds.append(micrograph.getObjId())

        prot._processMicrograph = Mock(side_effect=fakeProcessMicrograph)

        prot.processMicrographListStep([1, 2])

        self.assertEqual([1, 2], sorted(prot.processedIds))
        self.assertNotIn(1, prot.stats)
        self.assertIn(2, prot.stats)

    def testProcessMicrographListStepRetriesUntilMicBecomesVisible(self):
        prot = self._newProtocol()
        inputSet = _FakeInputSet([1], gettableIds=set())
        prot.inputMicrographs = _FakePointer(inputSet)
        prot._processMicrograph = Mock()

        def becomeVisibleOnSleep(_delay):
            inputSet._gettableIds = {1}

        with patch(
                'xmipp3.protocols.protocol_tilt_analysis.time.sleep',
                side_effect=becomeVisibleOnSleep,
        ):
            prot.processMicrographListStep([1])

        prot._processMicrograph.assert_called_once()
        self.assertEqual(
            1,
            prot._processMicrograph.call_args[0][0].getObjId(),
        )

    def testCheckNewOutputSkipsMicWithMissingStatsWithoutCrashing(self):
        # Regression test: a mic recorded as processed but with no computed
        # statistics (because processMicrographListStep exhausted its
        # visibility retries) must be skipped gracefully in _checkNewOutput,
        # not crash with a KeyError on self.stats[micId].
        prot = self._newProtocol()
        prot.processedIds = [1]
        prot.isStreamClosed = False
        prot.stats = {}
        inputSet = _FakeInputSet([1])
        prot.inputMicrographs = _FakePointer(inputSet)
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._store = Mock()

        prot._checkNewOutput()  # must not raise

        self.assertFalse(prot.finished)

    def testFinishedCheckDoesNotReloadInputWithoutNewMicrographs(self):
        """A terminal output check must not reload input items when none are new."""
        prot = self._newProtocol()
        prot.processedIds = [1]
        prot.isStreamClosed = True

        inputSet = _FakeInputSet([1], streamClosed=True)
        prot.inputMicrographs = _FakePointer(inputSet)
        prot._getAllDoneIds = lambda: ([1], 1, [1], [])
        prot._loadOutputSet = Mock()
        prot._updateOutputSet = Mock()
        prot._store = Mock()

        prot._checkNewOutput()

        self.assertTrue(prot.finished)
        self.assertEqual(
            1,
            inputSet.loadCalls,
            "The terminal check may read input size once, but must not "
            "reload the Set again when newDone is empty.",
        )
        prot._loadOutputSet.assert_not_called()
        prot._updateOutputSet.assert_not_called()
        prot._store.assert_called_once()

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        prot = self._newProtocol()
        prot.finished = True
        prot.initializeStep = Mock()
        prot._checkNewInput = Mock()
        prot._checkNewOutput = Mock()
        prot._insertFunctionStep = Mock(return_value=1)
        prot.createOutputStep = Mock()

        prot.stepsGeneratorStep()

        prot._checkNewInput.assert_not_called()
        prot._checkNewOutput.assert_not_called()
        prot._insertFunctionStep.assert_called_once()
