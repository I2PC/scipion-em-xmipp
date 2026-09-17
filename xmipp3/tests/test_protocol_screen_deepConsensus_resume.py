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

from xmipp3.protocols.protocol_screen_deepConsensus import XmippProtScreenDeepConsensus


class _FakeMics:
    def getSamplingRate(self):
        return 1.5


class _FakeCoord:
    def __init__(self, micId, x, y):
        self.micId = micId
        self.x = x
        self.y = y

    def getMicId(self):
        return self.micId

    def getX(self):
        return self.x

    def getY(self):
        return self.y


class _FakeParticle:
    def __init__(self, coord):
        self.coord = coord

    def getCoordinate(self):
        return self.coord


class _FakeSet:
    def __init__(self, items):
        self.items = items

    def getSize(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


class TestXmippDeepConsensusResume(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        return self.newProtocol(XmippProtScreenDeepConsensus)

    def testStreamingRuntimeStateIsPerProtocol(self):
        prot1 = self._newProtocol()
        prot1._getInputMicrographs = lambda: _FakeMics()
        prot1._resetStreamingState()
        prot1.TO_EXTRACT_MICFNS['OR'].append('mic1.mrc')
        prot1.EXTRACTING['AND'] = True

        prot2 = self._newProtocol()
        prot2._getInputMicrographs = lambda: _FakeMics()
        prot2._resetStreamingState()

        self.assertEqual([], prot2.TO_EXTRACT_MICFNS['OR'])
        self.assertFalse(prot2.EXTRACTING['AND'])
        self.assertEqual(1.5, prot2.inSamplingRate)

    def testLoadTrainedParamsAddsResumeKeys(self):
        prot = self._newProtocol()
        prot.saveTrainedParams({'trainedMicFns': [], 'predictedMicFns': [], 'posParticlesTrained': 0, 'trainingPass': 0, 'predictionPasses': [], 'doneExtraTesting': False, 'firstTraining': True, 'keepTraining': True})
        params = prot.loadTrainedParams()
        self.assertEqual([], params['pendingPredictedMicFns'])
        self.assertIsNone(params['pendingPredictionPass'])

    def testPredictionIsCommittedOnlyAfterOutput(self):
        prot = self._newProtocol()
        params = prot.loadTrainedParams()
        params['pendingPredictedMicFns'] = ['mic1.mrc', 'mic2.mrc']
        prot.saveTrainedParams(params)
        prot._commitPrediction()
        params = prot.loadTrainedParams()
        self.assertEqual(['mic1.mrc', 'mic2.mrc'], params['predictedMicFns'])
        self.assertEqual([], params['pendingPredictedMicFns'])

        params['pendingPredictionPass'] = 2
        prot.saveTrainedParams(params)
        prot._commitPrediction(2)
        params = prot.loadTrainedParams()
        self.assertIn(2, params['predictionPasses'])
        self.assertIsNone(params['pendingPredictionPass'])

    def testOutputCoordinateKeysRecoverPartialAppend(self):
        prot = self._newProtocol()
        coord = _FakeCoord(7, 10.0, 20.0)
        coordKeys = prot._getOutputCoordinateKeys(_FakeSet([coord]))
        particleKeys = prot._getOutputCoordinateKeys(_FakeSet([_FakeParticle(coord)]), particles=True)
        self.assertEqual({(7, 10.0, 20.0)}, coordKeys)
        self.assertEqual(coordKeys, particleKeys)

    def testMeanAccuracyDoesNotRequireTransientTrainingState(self):
        prot = self._newProtocol()
        params = prot.loadTrainedParams()
        params['trainingPass'] = 1
        netDir = prot._getExtraPath(prot.NET_TEMPLATE.format(1))
        os.makedirs(netDir, exist_ok=True)
        with open(os.path.join(netDir, 'netsMeanValAcc.txt'), 'w') as f:
            f.write('mean 0.97\n')
        self.assertEqual(0.97, prot.loadMeanAccuracy(params))

    def testEndProtocolFinalizesSynchronously(self):
        prot = self._newProtocol()
        state = {'trainingPass': 2}
        events = []
        ready = [['mic1.mrc'], ['mic2.mrc'], []]
        prot.loadTrainedParams = lambda: state
        prot.saveTrainedParams = lambda params: state.update(params)
        prot.retrievePreviousPassModel = lambda *args: events.append('model')
        prot.readyToPredictMicFns = lambda: ready.pop(0)
        prot.predictCNN = lambda: events.append('predict')
        prot.createOutputStep = lambda: events.append('output')
        prot.outputCoordinates = object()
        prot.outputParticles = object()
        prot.updateOutput = lambda closeStream=False: events.append('close' if closeStream else 'update')
        prot.endProtocolResumeSafeStep()
        self.assertEqual('', state['trainingPass'])
        self.assertEqual(['model', 'predict', 'output', 'predict', 'output', 'close'], events)
        self.assertTrue(prot.ENDED)
