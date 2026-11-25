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
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_particle_pick_consensus import (
    XmippProtConsensusPicking,
    consensusWorker,
    getReadyMics,
)


class _FakeOutputSet:
    def __init__(self, micIds=None):
        self.micIds = set(micIds or [])
        self.appended = []
        self.closed = False

    def getSize(self):
        return len(self.micIds)

    def aggregate(self, operations, operationLabel, groupByLabels):
        return [{'_micId': micId} for micId in self.micIds]

    def append(self, coordinate):
        self.appended.append(coordinate)

    def close(self):
        self.closed = True


class _FakeCoordinate:
    def setMicrograph(self, micrograph):
        self.micrograph = micrograph

    def setPosition(self, x, y):
        self.position = (x, y)


class _FakeMic:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _FakeMic(self._objId)


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


class _FakeMicsDict(dict):
    """Stands in for a SetOfMicrographs: supports __contains__/__getitem__
    like a dict already does, plus the loadAllProperties() refresh call."""

    def loadAllProperties(self):
        pass


class TestXmippParticlePickConsensusRegression(BaseTest):
    """Regression tests for Picking Consensus streaming and Continue handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtConsensusPicking)
        prot.checkedMics = set()
        prot.processedMics = set()
        prot.sampligRates = []
        prot.streamClosed = False
        return prot

    def testGetReadyMicsUsesFreshReloadedCoordinateSet(self):
        # Regression test: getReadyMics must work directly off the
        # already-resolved logical coordSet, never reconstruct a Set from
        # a raw sqlite filename (coordSet.getFileName()).
        class _FakeCoordSet:
            def __init__(self):
                self.loadCalls = 0
                self.closed = False

            def getFileName(self):
                raise AssertionError(
                    "getReadyMics must not reconstruct a Set from a raw "
                    "sqlite filename."
                )

            def loadAllProperties(self):
                self.loadCalls += 1

            def isStreamClosed(self):
                return True

            def getUniqueValues(self, attr):
                return [1, 2]

            def close(self):
                self.closed = True

        coordSet = _FakeCoordSet()
        readyMics, streamClosed = getReadyMics(coordSet)

        self.assertEqual({1, 2}, readyMics)
        self.assertTrue(streamClosed)
        self.assertEqual(1, coordSet.loadCalls)
        self.assertTrue(coordSet.closed)

    def testReadyMicsRemainEmptyUntilAllPickersAreReady(self):
        prot = self._newProtocol()
        pointers = [SimpleNamespace(get=lambda: object()), SimpleNamespace(get=lambda: object())]
        prot.inputCoordinates = pointers
        prot._restoreProcessedMics = lambda: None
        prot.insertNewCoorsSteps = lambda mics: self.fail('No mic should be scheduled before all pickers are ready.')

        with patch('xmipp3.protocols.protocol_particle_pick_consensus.getReadyMics', side_effect=[(set(), False), ({1}, False), (set(), False), ({1}, False)]):
            prot._checkNewInput()

        self.assertEqual(set(), prot.checkedMics)

    def testCheckNewInputUsesUnionAcrossAllInputsWhenStreamsClose(self):
        # Regression test: once every input stream is closed, the protocol
        # must schedule the union of mics ready in ANY input picker - the
        # loop above already computes this correctly for N inputs. A
        # leftover 2-input-only override then narrowed it down to just
        # the intersection of inputCoordinates[0] and [1], silently
        # ignoring mics only found by a 3rd+ picker.
        prot = self._newProtocol()
        prot.inputCoordinates = [
            SimpleNamespace(get=lambda: object()),
            SimpleNamespace(get=lambda: object()),
            SimpleNamespace(get=lambda: object()),
        ]
        prot._restoreProcessedMics = lambda: None
        prot.getMainInput = lambda: SimpleNamespace(
            getMicrographs=lambda: _FakeMicsDict({
                1: _FakeMic(1), 2: _FakeMic(2), 3: _FakeMic(3),
            })
        )
        prot.newDeps = []
        prot.updateSteps = lambda: None

        scheduled = []
        prot.insertNewCoorsSteps = (
            lambda mics: scheduled.extend(mic.getObjId() for mic in mics) or []
        )

        with patch(
                'xmipp3.protocols.protocol_particle_pick_consensus.getReadyMics',
                side_effect=[
                    ({1}, True), ({2}, True), ({3}, True),
                    ({1}, True), ({2}, True),
                ],
        ):
            prot._checkNewInput()

        self.assertEqual({1, 2, 3}, set(scheduled))

    def testCheckNewInputDefersMicrographNotYetVisibleWithoutPermanentLoss(self):
        # Regression test: getMainInput().getMicrographs() resolves a
        # Pointer whose cached value is never refreshed. A micId that
        # getReadyMics() (which does reopen fresh) already reports as
        # ready may still be invisible in that cached micrographs Set -
        # it must be deferred (kept out of checkedMics) so it is retried
        # on the next check, not crash or get permanently skipped.
        prot = self._newProtocol()
        prot.inputCoordinates = [SimpleNamespace(get=lambda: object())]
        prot._restoreProcessedMics = lambda: None
        prot.getMainInput = lambda: SimpleNamespace(
            getMicrographs=lambda: _FakeMicsDict({2: _FakeMic(2)})  # mic 1 missing
        )
        prot.newDeps = []
        prot.updateSteps = lambda: None

        scheduled = []
        prot.insertNewCoorsSteps = (
            lambda mics: scheduled.extend(mic.getObjId() for mic in mics) or []
        )

        with patch(
                'xmipp3.protocols.protocol_particle_pick_consensus.getReadyMics',
                return_value=({1, 2}, True),
        ):
            prot._checkNewInput()

        self.assertEqual([2], scheduled)
        self.assertEqual({2}, prot.checkedMics)

    def testCheckNewOutputDefersConsensusResultWhenMicrographNotYetVisible(self):
        # Regression test: a consensus result marker for a micrograph
        # that is not yet visible in the (staleness-prone) micrographs
        # Pointer must not crash and must not be moved out of _tmp -
        # otherwise it would never be retried and that micrograph's
        # coordinates would be silently lost forever. finished must also
        # not latch True while a result is still deferred.
        prot = self._newProtocol()
        prot.checkedMics = {1}
        prot.processedMics = {1}
        prot.streamClosed = True
        outputSet = _FakeOutputSet()
        events = []

        prot._loadOutputSet = lambda *args: outputSet
        prot._updateOutputSet = lambda *args: events.append('output')
        prot._refreshOutputRelations = lambda *args: events.append('relations')
        prot._getTmpPath = lambda *args: 'tmp'
        prot._getExtraPath = lambda *args: 'extra'
        prot.getMainInput = lambda: SimpleNamespace(getMicrographs=lambda: _FakeMicsDict({}))

        with patch('xmipp3.protocols.protocol_particle_pick_consensus.getFiles', return_value=['/run/tmp/consensusCoords_1.txt']), patch('xmipp3.protocols.protocol_particle_pick_consensus.os.path.getsize', return_value=10), patch('xmipp3.protocols.protocol_particle_pick_consensus.np.loadtxt', side_effect=AssertionError('Must not be loaded before checking micrograph visibility.')), patch('xmipp3.protocols.protocol_particle_pick_consensus.moveFile', side_effect=lambda *args: events.append('marker')):
            prot._checkNewOutput()

        self.assertEqual([], outputSet.appended)
        self.assertNotIn('marker', events)
        self.assertFalse(prot.finished)

    def testRestoreProcessedMicsUsesStepGraphNotFileScan(self):
        # Regression test: "has this mic's consensus already been
        # computed" must come from the persisted step graph (durable
        # across Continue), never from scanning tmp/extra for
        # consensusCoords_*.txt files - an empty consensus result is a
        # real, needed file for the data handoff with _checkNewOutput,
        # but it must not also be the completion bookkeeping mechanism.
        prot = self._newProtocol()
        prot._steps = [
            _FakeStep('calculateConsensusStep', [1, 'mic_001.mrc']),
            _FakeStep('calculateConsensusStep', [2, 'mic_002.mrc']),
            _FakeStep('calculateConsensusStep', [3, 'mic_003.mrc'], finished=False),
            _FakeStep('copyInputFilesStep', []),
        ]
        prot._prevSteps = []

        with patch(
                'xmipp3.protocols.protocol_particle_pick_consensus.getFiles',
                side_effect=AssertionError(
                    '_restoreProcessedMics must not scan the filesystem.'
                ),
        ):
            prot._restoreProcessedMics()

        self.assertEqual({1, 2}, prot.checkedMics)
        self.assertEqual({1, 2}, prot.processedMics)

    def testStreamingOutputReusesLogicalSetWithoutLegacySqlite(self):
        class LogicalOutputSet:
            def __init__(self):
                self.enableAppendCalls = 0
                self.micrographs = None

            def enableAppend(self):
                self.enableAppendCalls += 1

            def setMicrographs(self, micrographs):
                self.micrographs = micrographs

        class FreshOutputSet:
            STREAM_OPEN = 1

            def __init__(self, filename=None):
                self.filename = filename

            def setStreamState(self, state):
                self.streamState = state

            def setBoxSize(self, boxSize):
                self.boxSize = boxSize

            def setMicrographs(self, micrographs):
                self.micrographs = micrographs

        class MainInput:
            def getBoxSize(self):
                return 128

            def getMicrographs(self, asPointer=False):
                return object()

        prot = self._newProtocol()
        logicalOutput = LogicalOutputSet()
        prot.consensusCoordinates = logicalOutput
        prot.getMainInput = lambda: MainInput()
        prot._getPath = lambda baseName: '/tmp/' + baseName

        outputSet = prot._loadOutputSet(
            FreshOutputSet,
            'coordinates.sqlite',
        )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming coordinates output must reuse the logical Set when "
            "the legacy SQLite file is absent.",
        )
        self.assertEqual(1, logicalOutput.enableAppendCalls)

    def testCheckNewOutputSkipsAlreadyPersistedMicrograph(self):
        prot = self._newProtocol()
        prot.checkedMics = {1}
        prot.processedMics = {1}
        outputSet = _FakeOutputSet({1})
        events = []

        prot._loadOutputSet = lambda *args: outputSet
        prot._updateOutputSet = lambda *args: events.append('output')
        prot._refreshOutputRelations = lambda *args: events.append('relations')
        prot._getTmpPath = lambda *args: 'tmp'
        prot._getExtraPath = lambda *args: 'extra'

        with patch('xmipp3.protocols.protocol_particle_pick_consensus.getFiles', return_value=['/run/tmp/consensusCoords_1.txt']), patch('xmipp3.protocols.protocol_particle_pick_consensus.os.path.getsize', return_value=10), patch('xmipp3.protocols.protocol_particle_pick_consensus.np.loadtxt', side_effect=AssertionError('Persisted mic must not be loaded again.')), patch('xmipp3.protocols.protocol_particle_pick_consensus.moveFile', side_effect=lambda *args: events.append('marker')):
            prot._checkNewOutput()

        self.assertEqual([], outputSet.appended)
        self.assertEqual(['output', 'relations', 'marker'], events)

    def testMarkerMovesOnlyAfterOutputAndRelations(self):
        prot = self._newProtocol()
        prot.checkedMics = {2}
        prot.processedMics = {2}
        outputSet = _FakeOutputSet()
        events = []

        prot._loadOutputSet = lambda *args: outputSet
        prot._updateOutputSet = lambda *args: events.append('output')
        prot._refreshOutputRelations = lambda *args: events.append('relations')
        prot._getTmpPath = lambda *args: 'tmp'
        prot._getExtraPath = lambda *args: 'extra'
        prot.getMainInput = lambda: SimpleNamespace(getMicrographs=lambda: _FakeMicsDict({2: object()}))

        with patch('xmipp3.protocols.protocol_particle_pick_consensus.getFiles', return_value=['/run/tmp/consensusCoords_2.txt']), patch('xmipp3.protocols.protocol_particle_pick_consensus.os.path.getsize', return_value=10), patch('xmipp3.protocols.protocol_particle_pick_consensus.np.loadtxt', return_value=np.array([10.0, 20.0])), patch('xmipp3.protocols.protocol_particle_pick_consensus.Coordinate', _FakeCoordinate), patch('xmipp3.protocols.protocol_particle_pick_consensus.moveFile', side_effect=lambda *args: events.append('marker')):
            prot._checkNewOutput()

        self.assertEqual(1, len(outputSet.appended))
        self.assertEqual(['output', 'relations', 'marker'], events)

    def testConsensusWorkerWritesEmptyResultMarkerAtomically(self):
        with tempfile.TemporaryDirectory() as tmpDir:
            outputFn = os.path.join(tmpDir, 'consensusCoords_7.txt')
            consensusWorker([np.empty((0, 2)), np.array([[10, 20]])], -1, 10, outputFn)

            self.assertTrue(os.path.exists(outputFn))
            self.assertEqual(0, os.path.getsize(outputFn))
            self.assertFalse(os.path.exists(outputFn + '.tmp'))

    def testRefreshOutputRelationsRebuildsAndCommits(self):
        prot = self._newProtocol()
        events = []
        prot.mapper = SimpleNamespace(deleteRelations=lambda creator: events.append('delete'), commit=lambda: events.append('commit'))
        prot.defineRelations = lambda outputSet: events.append('define')

        prot._refreshOutputRelations(object())

        self.assertEqual(['delete', 'define', 'commit'], events)

import unittest
from unittest.mock import Mock

from xmipp3.protocols.protocol_particle_pick_consensus import (
    XmippProtConsensusPicking,
)


class TestXmippConsensusPickingFinalizationRegression(unittest.TestCase):

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        class _Harness:
            finished = True

            def __init__(self):
                self._prepareStreamingGenerator = Mock()
                self._checkNewInput = Mock()
                self._checkNewOutput = Mock()
                self._insertFunctionStep = Mock(return_value=1)
                self.createOutputStep = Mock()

        protocol = _Harness()

        XmippProtConsensusPicking.stepsGeneratorStep(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()
        protocol._insertFunctionStep.assert_called_once()

