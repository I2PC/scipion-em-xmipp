# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

import json
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


class _FakePointer:
    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj


class _FakeInputMicSet:
    def __init__(self, ids, streamClosed=False, items=None):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self._items = items or {}
        self.closed = False

    def loadAllProperties(self):
        pass

    def isStreamClosed(self):
        return self._streamClosed

    def getSize(self):
        return len(self._ids)

    def __contains__(self, itemId):
        return itemId in self._items

    def getUniqueValues(self, attr, where=None):
        if where is None:
            return sorted(self._ids)
        threshold = int(where.split('>')[1].strip())
        return sorted(itemId for itemId in self._ids if itemId > threshold)

    def getItem(self, field, value):
        return self._items[value]

    def close(self):
        self.closed = True


class _FakeFuncName:
    def __init__(self, name):
        self._name = name

    def get(self):
        return self._name


class _FakeArgsStr:
    def __init__(self, args):
        self._argsJson = json.dumps(args)

    def get(self, default=None):
        return self._argsJson


class _FakeStep:
    def __init__(self, funcName, args, finished=True):
        self.funcName = _FakeFuncName(funcName)
        self.argsStr = _FakeArgsStr(args)
        self._finished = finished

    def isFinished(self):
        return self._finished


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
        # Regression test: an output .mrc file existing on disk must not,
        # by itself, be read as "this mic's pipeline is done" - completion
        # comes only from markMicDoneStep's own FINISHED status in the
        # persisted step graph, never a filesystem marker/output file.
        prot = self._newProtocol()
        mic = _FakeMicrograph(1)
        outputFn = prot._getOutputMicrograph(mic)

        try:
            os.makedirs(os.path.dirname(outputFn), exist_ok=True)
            open(outputFn, 'w').close()

            prot._steps = []
            prot._prevSteps = []
            self.assertFalse(prot._isMicPipelineDone(mic))

            prot._steps = [_FakeStep('markMicDoneStep', [1])]
            self.assertTrue(prot._isMicPipelineDone(mic))
        finally:
            if os.path.exists(outputFn):
                os.remove(outputFn)

    def testMarkMicDoneStepWritesNoMarkerFile(self):
        prot = self._newProtocol()
        os.makedirs(prot._getExtraPath(), exist_ok=True)
        before = set(os.listdir(prot._getExtraPath()))

        prot.markMicDoneStep(1)

        after = set(os.listdir(prot._getExtraPath()))
        self.assertEqual(before, after)

    def testRestoreInsertedMicsUsesCompletionMarkers(self):
        prot = self._newProtocol()
        prot.insertedDict = {}
        prot._isMicPipelineDone = lambda mic: mic.getObjId() == 1
        prot._restoreInsertedMics([_FakeMicrograph(1), _FakeMicrograph(2)])
        self.assertEqual({1}, set(prot.insertedDict))

    def testNewInputUsesFreshReloadedSnapshot(self):
        prot = self._newProtocol()
        prot.insertedDict = {1: 10}
        prot._lastInputId = 1
        mic1 = _FakeMicrograph(1)
        mic2 = _FakeMicrograph(2)
        prot.inputMicrographs = _FakePointer(_FakeInputMicSet(
            ids=[1, 2], items={1: mic1, 2: mic2},
        ))
        prot.newDeps = []
        scheduled = []
        prot._insertNewMicsSteps = lambda inserted, mics: scheduled.extend(m.getObjId() for m in mics) or []
        prot.updateSteps = lambda: None
        prot._checkNewInput()
        self.assertEqual([2], scheduled)

    def testCheckNewInputRetriesMicNotYetVisibleWithoutPermanentLoss(self):
        # Regression test: Set.getItem raises (UnboundLocalError) rather
        # than returning None for a row it cannot find, and _discoverIdsAfter
        # already advances the id watermark past a discovered id regardless
        # of whether it is actually selectable yet. A momentarily-invisible
        # mic must therefore be kept pending and retried on a later check,
        # not dropped - dropping it would lose it forever, since the
        # watermark never revisits an id once passed.
        prot = self._newProtocol()
        prot.insertedDict = {}
        prot._lastInputId = 0
        inputSet = _FakeInputMicSet(ids=[2], items={})
        prot.inputMicrographs = _FakePointer(inputSet)
        prot.newDeps = []
        scheduled = []
        prot._insertNewMicsSteps = lambda inserted, mics: scheduled.extend(m.getObjId() for m in mics) or []
        prot.updateSteps = lambda: None

        prot._checkNewInput()  # must not raise

        self.assertEqual([], scheduled)
        self.assertEqual({2}, prot._pendingMicIds)

        # The mic becomes visible on a later check.
        inputSet._items[2] = _FakeMicrograph(2)
        prot._checkNewInput()

        self.assertEqual([2], scheduled)
        self.assertEqual(set(), prot._pendingMicIds)

    def testOutputAndRelationsArePersistedBeforeCheckpoint(self):
        prot = self._newProtocol()
        prot.SetOfMicrographs = [_FakeMicrograph(1)]
        prot.streamClosed = False
        prot._isMicPipelineDone = lambda mic: True
        prot._getOutputMicrograph = lambda mic: 'mic_%06d.mrc' % mic.getObjId()
        outputSet = _FakeOutputSet([99])
        prot.getOutputMics = lambda: outputSet
        events = []
        prot._updateOutputSet = lambda *args: events.append('output')
        prot._refreshOutputRelation = lambda *args: events.append('relations')
        prot._checkNewOutput()
        self.assertEqual([1], outputSet.appended)
        self.assertEqual(['output', 'relations'], events)

    def testAlreadyPersistedMicIsNotRepublished(self):
        # Regression test: done-tracking must come from the real, persisted
        # outputMicrographs Set (via _getKnownPersistedOutputIds), not a
        # DONE_all.TXT-style sidecar - a mic already reflected there must
        # not trigger a redundant append/output rewrite.
        prot = self._newProtocol()
        prot.SetOfMicrographs = [_FakeMicrograph(1)]
        prot.streamClosed = False
        prot._isMicPipelineDone = lambda mic: True
        prot.outputMicrographs = _FakeOutputSet([1])
        prot._getOutputMicrograph = lambda mic: 'mic_%06d.mrc' % mic.getObjId()
        events = []
        prot._updateOutputSet = lambda *args: events.append('output')
        prot._refreshOutputRelation = lambda *args: events.append('relations')
        prot._checkNewOutput()
        self.assertEqual([], events)

    def testRefreshOutputRelationRebuildsAndCommits(self):
        prot = self._newProtocol()
        events = []
        prot.mapper = _FakeMapper(events)
        prot._defineTransformRelation = lambda *args: events.append('define')
        prot._refreshOutputRelation(object())
        self.assertEqual(['delete', 'define', 'commit'], events)

# Finalization regression: the executor performs one last stepsCheck callback
# after it has already found no pending steps.
from unittest.mock import Mock

from pyworkflow.tests import BaseTest, setupTestProject
from xmipp3.protocols.protocol_preprocess_micrographs import (
    XmippProtPreprocessMicrographs,
)


class TestXmippPreprocessMicrographsFinalizationRegression(BaseTest):

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        prot = self.newProtocol(XmippProtPreprocessMicrographs)
        prot.finished = True
        prot._prepareStreamingGenerator = Mock()
        prot._checkNewInput = Mock()
        prot._checkNewOutput = Mock()
        prot._insertFunctionStep = Mock(return_value=1)
        prot.createOutputStep = Mock()

        prot.stepsGeneratorStep()

        prot._checkNewInput.assert_not_called()
        prot._checkNewOutput.assert_not_called()
        prot._insertFunctionStep.assert_called_once()

