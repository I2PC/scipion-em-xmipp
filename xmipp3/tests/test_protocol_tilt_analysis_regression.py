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
    def __init__(self, ids, streamClosed=False):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self.closed = False

    def getIdSet(self):
        return set(self._ids)

    def getSize(self):
        return len(self._ids)

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closed = True


class _FakeMicrograph:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


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
        inputSet = _FakeInputSet([1, 2])
        scheduled = []

        prot._loadInputSet = lambda _: inputSet
        prot._getFirstJoinStep = lambda: None
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


    def testFinishedCheckDoesNotReloadInputWithoutNewMicrographs(self):
        """A terminal output check must not reload input items when none are new."""
        prot = self._newProtocol()
        prot.processedIds = [1]
        prot.isStreamClosed = True

        inputSet = _FakeInputSet([1], streamClosed=True)
        prot._loadInputSet = Mock(return_value=inputSet)
        prot._getAllDoneIds = lambda: ([1], 1, [1], [])
        prot._getFirstJoinStep = lambda: None
        prot._loadOutputSet = Mock()
        prot._updateOutputSet = Mock()
        prot._store = Mock()

        prot._checkNewOutput()

        self.assertTrue(prot.finished)
        self.assertEqual(
            1,
            prot._loadInputSet.call_count,
            "The terminal check may read input size once, but must not "
            "reload the Set again when newDone is empty.",
        )
        prot._loadOutputSet.assert_not_called()
        prot._updateOutputSet.assert_not_called()
        prot._store.assert_called_once()
