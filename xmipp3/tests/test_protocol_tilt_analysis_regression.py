# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from unittest.mock import patch

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_tilt_analysis import XmippProtTiltAnalysis


class _FakeInputSet:
    def __init__(self, ids, streamClosed=False):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self.closed = False

    def getIdSet(self):
        return set(self._ids)

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closed = True


class _FakeMicrograph:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _FakeOutputSet:
    def __init__(self, ids=None):
        self.ids = set(ids or [])
        self.appended = []

    def getSize(self):
        return len(self.ids)

    def getIdSet(self):
        return set(self.ids)

    def append(self, mic):
        self.ids.add(mic.getObjId())
        self.appended.append(mic.getObjId())


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
