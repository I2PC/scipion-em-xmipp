# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

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
        class StaleCoordSet:
            def getFileName(self):
                return 'coordinates.sqlite'

            def aggregate(self, *args, **kwargs):
                return [{'_micId': 1}]

        class FreshCoordSet:
            def __init__(self, filename):
                self.filename = filename

            def loadAllProperties(self):
                pass

            def isStreamClosed(self):
                return True

            def aggregate(self, *args, **kwargs):
                return [{'_micId': 1}, {'_micId': 2}]

            def close(self):
                pass

        with patch(
                'xmipp3.protocols.protocol_particle_pick_consensus.SetOfCoordinates',
                FreshCoordSet,
        ):
            readyMics, streamClosed = getReadyMics(StaleCoordSet())

        self.assertEqual({1, 2}, readyMics)
        self.assertTrue(streamClosed)

    def testReadyMicsRemainEmptyUntilAllPickersAreReady(self):
        prot = self._newProtocol()
        pointers = [SimpleNamespace(get=lambda: object()), SimpleNamespace(get=lambda: object())]
        prot.inputCoordinates = pointers
        prot._restoreProcessedMics = lambda: None
        prot.insertNewCoorsSteps = lambda mics: self.fail('No mic should be scheduled before all pickers are ready.')

        with patch('xmipp3.protocols.protocol_particle_pick_consensus.getReadyMics', side_effect=[(set(), False), ({1}, False), (set(), False), ({1}, False)]):
            prot._checkNewInput()

        self.assertEqual(set(), prot.checkedMics)

    def testRestoreProcessedMicsIncludesPendingTmpResults(self):
        prot = self._newProtocol()
        prot._getExtraPath = lambda *args: 'extra'
        prot._getTmpPath = lambda *args: 'tmp'

        def fakeGetFiles(folder):
            if folder == 'extra':
                return ['/run/extra/consensusCoords_1.txt']
            return ['/run/tmp/consensusCoords_2.txt', '/run/tmp/consensusCoords_3.txt.tmp', '/run/tmp/other.txt']

        with patch('xmipp3.protocols.protocol_particle_pick_consensus.getFiles', side_effect=fakeGetFiles):
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

        with patch(
            'xmipp3.protocols.protocol_particle_pick_consensus.os.path.exists',
            return_value=False,
        ):
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
        prot.getMainInput = lambda: SimpleNamespace(getMicrographs=lambda: {2: object()})

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

    def testFinishedStepsCheckIsNoOp(self):
        class _Harness:
            finished = True

            def __init__(self):
                self._checkNewInput = Mock()
                self._checkNewOutput = Mock()

        protocol = _Harness()

        XmippProtConsensusPicking._stepsCheck(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()

