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

from xmipp3.protocols.protocol_particle_pick_consensus import XmippProtConsensusPicking, consensusWorker


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
