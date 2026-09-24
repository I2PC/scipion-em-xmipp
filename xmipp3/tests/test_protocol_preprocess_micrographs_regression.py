# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

import os

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from xmipp3.tests.streaming_test_utils import FakeOutputSet as _FakeOutputSet


class _FakeMicrograph:
    def __init__(self, objId, fileName=None, micName=None):
        self._objId = objId
        self._fileName = fileName or 'mic_%06d.mrc' % objId
        self._micName = micName or 'mic_%06d' % objId

    def getObjId(self):
        return self._objId

    def getFileName(self):
        return self._fileName

    def getMicName(self):
        return self._micName

    def clone(self):
        return _FakeMicrograph(self._objId, self._fileName, self._micName)


class _FreshOutputSet(_FakeOutputSet):
    def getIdSet(self):
        raise AssertionError('Fresh output Set must not query IDs before its first append.')


class _FakeMapper:
    def __init__(self, events):
        self.events = events

    def deleteRelations(self, creator):
        self.events.append('delete')

    def commit(self):
        self.events.append('commit')


class TestXmippPreprocessMicrographsRegression(BaseTest):
    """Regression tests for Preprocess Micrographs streaming and Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        return self.newProtocol(XmippProtPreprocessMicrographs, doInvert=True)

    def testOutputFileAloneDoesNotMarkPipelineDone(self):
        prot = self._newProtocol()
        mic = _FakeMicrograph(1)
        outputFn = prot._getOutputMicrograph(mic)
        doneFn = prot._getMicDoneMarker(mic.getObjId())

        if os.path.exists(doneFn):
            os.remove(doneFn)

        try:
            os.makedirs(os.path.dirname(outputFn), exist_ok=True)
            open(outputFn, 'w').close()
            self.assertFalse(prot._isMicPipelineDone(mic))
            prot.markMicDoneStep(mic.getObjId())
            self.assertTrue(prot._isMicPipelineDone(mic))
        finally:
            if os.path.exists(doneFn):
                os.remove(doneFn)

    def testRestoreInsertedMicsUsesCompletionMarkers(self):
        prot = self._newProtocol()
        prot.insertedDict = {}
        prot._isMicPipelineDone = lambda mic: mic.getObjId() == 1
        prot._restoreInsertedMics([_FakeMicrograph(1), _FakeMicrograph(2)])
        self.assertEqual({1}, set(prot.insertedDict))

    def testNewInputUsesFreshReloadedSnapshot(self):
        prot = self._newProtocol()
        prot.inputMics = [_FakeMicrograph(1)]
        prot.insertedDict = {1: 10}
        prot._loadInputMics = lambda: ([_FakeMicrograph(1), _FakeMicrograph(2)], False)
        prot._getFirstJoinStep = lambda: None
        scheduled = []
        prot._insertNewMicsSteps = lambda inserted, mics: scheduled.extend(m.getObjId() for m in mics) or []
        prot.updateSteps = lambda: None
        prot._checkNewInput()
        self.assertEqual([2], scheduled)

    def testFreshOutputSetDoesNotQueryIds(self):
        prot = self._newProtocol()
        self.assertEqual(set(), prot._getOutputMicIds(_FreshOutputSet()))

    def testOutputAndRelationsArePersistedBeforeDoneCheckpoint(self):
        prot = self._newProtocol()
        prot.SetOfMicrographs = [_FakeMicrograph(1)]
        prot.streamClosed = False
        prot._isMicPipelineDone = lambda mic: True
        prot._readDoneList = lambda: []
        prot._getOutputMicrograph = lambda mic: 'mic_%06d.mrc' % mic.getObjId()
        outputSet = _FakeOutputSet([99])
        prot.getOutputMics = lambda: outputSet
        events = []
        prot._updateOutputSet = lambda *args: events.append('output')
        prot._refreshOutputRelation = lambda *args: events.append('relations')
        prot._writeDoneList = lambda *args: events.append('done')
        prot._getFirstJoinStep = lambda: None
        prot._checkNewOutput()
        self.assertEqual([1], outputSet.appended)
        self.assertEqual(['output', 'relations', 'done'], events)

    def testCheckpointedMicIsRepublishedWhenOutputIsMissing(self):
        prot = self._newProtocol()
        prot.SetOfMicrographs = [_FakeMicrograph(1)]
        prot.streamClosed = False
        prot._isMicPipelineDone = lambda mic: True
        prot._readDoneList = lambda: [1]
        prot._getOutputMicrograph = lambda mic: 'mic_%06d.mrc' % mic.getObjId()
        outputSet = _FakeOutputSet([99])
        prot.getOutputMics = lambda: outputSet
        events = []
        prot._updateOutputSet = lambda *args: events.append('output')
        prot._refreshOutputRelation = lambda *args: events.append('relations')
        prot._writeDoneList = lambda *args: events.append('done')
        prot._getFirstJoinStep = lambda: None
        prot._checkNewOutput()
        self.assertEqual([1], outputSet.appended)
        self.assertEqual(['output', 'relations'], events)

    def testRefreshOutputRelationRebuildsAndCommits(self):
        prot = self._newProtocol()
        events = []
        prot.mapper = _FakeMapper(events)
        prot._defineTransformRelation = lambda *args: events.append('define')
        prot._refreshOutputRelation(object())
        self.assertEqual(['delete', 'define', 'commit'], events)
