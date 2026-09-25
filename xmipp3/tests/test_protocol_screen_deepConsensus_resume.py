# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

import os
from types import SimpleNamespace

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

    def testStreamingMicrographsUseFreshCoordinateSnapshot(self):
        class Mic:
            def __init__(self, micId, fileName):
                self.micId = micId
                self.fileName = fileName

            def getFileName(self):
                return self.fileName

            def clone(self):
                return Mic(self.micId, self.fileName)

        class MicSet:
            def __init__(self, mics):
                self.mics = list(mics)

            def __iter__(self):
                return iter(self.mics)

        class CoordSet:
            def __init__(self, filename=None):
                self.fresh = filename is not None

            def getFileName(self):
                return 'coordinates.sqlite'

            def loadAllProperties(self):
                pass

            def getMicrographs(self):
                if not self.fresh:
                    raise AssertionError(
                        'Streaming code must not enumerate micrographs '
                        'from the stale pointer object.'
                    )
                return MicSet([
                    Mic(1, '/data/mic1.mrc'),
                    Mic(2, '/data/mic2.mrc'),
                ])

            def close(self):
                pass

        prot = self._newProtocol()
        prot.inputCoordinates = [SimpleNamespace(get=lambda: CoordSet())]
        prot.waitFreeInputCoords = lambda: None
        prot.waitFreeInputMics = lambda: None
        prot.prunePaths = lambda paths: [os.path.basename(path) for path in paths]

        mics = prot.getAllCoordsInputMicrographs(shared=False)

        self.assertEqual({'mic1.mrc', 'mic2.mrc'}, set(mics))

    def testStreamingCoordinateReadUsesFreshSnapshot(self):
        class Mic:
            def getFileName(self):
                return '/data/mic2.mrc'

        class CoordSet:
            def __init__(self, filename=None):
                self.fresh = filename is not None

            def getFileName(self):
                return 'coordinates.sqlite'

            def loadAllProperties(self):
                pass

            def iterCoordinates(self, mic):
                if not self.fresh:
                    raise AssertionError(
                        'Streaming code must not read coordinates '
                        'from the stale pointer object.'
                    )
                return iter([object()])

            def close(self):
                pass

        prot = self._newProtocol()
        prot.inputCoordinates = [SimpleNamespace(get=lambda: CoordSet())]
        prot.getAllCoordsInputMicrographs = lambda shared: {'mic2.mrc': Mic()}
        prot.waitFreeInputCoords = lambda: None
        prot.prunePaths = lambda paths: [os.path.basename(path) for path in paths]

        self.assertEqual(
            ['mic2.mrc'],
            prot.getMicrographFnsWithCoordinates(shared=True),
        )

    def testParentClosureUsesFreshCoordinateSnapshot(self):
        class CoordSet:
            def __init__(self, filename=None):
                self.fresh = filename is not None

            def getFileName(self):
                return 'coordinates.sqlite'

            def loadAllProperties(self):
                pass

            def isStreamOpen(self):
                return not self.fresh

            def close(self):
                pass

        prot = self._newProtocol()
        prot.inputCoordinates = [SimpleNamespace(get=lambda: CoordSet())]
        prot.waitFreeInputCoords = lambda: None

        self.assertTrue(prot.checkIfParentsFinished())

    def testInputMicrographsAreNotFrozenAtFirstSnapshot(self):
        class MicSet:
            def __init__(self, size):
                self.size = size

            def getSize(self):
                return self.size

        class CoordSet:
            snapshot = 0

            def __init__(self, filename=None):
                self.fresh = filename is not None

            def getFileName(self):
                return 'coordinates.sqlite'

            def loadAllProperties(self):
                pass

            def getMicrographs(self):
                CoordSet.snapshot += 1
                return MicSet(CoordSet.snapshot)

            def close(self):
                pass

        prot = self._newProtocol()
        prot.inputCoordinates = [SimpleNamespace(get=lambda: CoordSet())]
        prot.waitFreeInputCoords = lambda: None
        prot.waitFreeInputMics = lambda: None

        first = prot._getInputMicrographs()
        second = prot._getInputMicrographs()

        self.assertEqual(1, first.getSize())
        self.assertEqual(
            2,
            second.getSize(),
            'Streaming micrographs must be refreshed instead of cached forever.',
        )

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
        os.makedirs(prot._getExtraPath(), exist_ok=True)
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
