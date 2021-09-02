# ***************************************************************************
# *
# * Authors:     Daniel Del Hoyo (daniel.delhoyo.gomez@alumnos.upm.es)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************

import time

from pyworkflow.tests import BaseTest, DataSet
import pyworkflow.tests as tests
from pyworkflow.protocol import getProtocolFromDb
from pyworkflow.object import PointerList, Pointer

from pwem.objects import SetOfCTF, CTFModel, Micrograph, SetOfMicrographs
from pwem.protocols import (ProtImportMicrographs, ProtImportCoordinates,
                            ProtCreateStreamData)
from pwem.protocols.protocol_create_stream_data import SET_OF_MICROGRAPHS, SET_OF_COORDINATES

from pwem import emlib
from xmipp3.protocols import XmippProtScreenDeepConsensus

ADD_MODEL_TRAIN_NEW = 0
ADD_MODEL_TRAIN_PRETRAIN = 1
ADD_MODEL_TRAIN_PREVRUN = 2

class TestXmippProtScreenDeepConsensus(BaseTest):
    @classmethod
    def setUpClass(cls):
        tests.setupTestProject(cls)
        cls.dsXmipp = DataSet.getDataSet('deepConsensusPicking')

    def _runInportMicrographs(self):
      protImport = self.newProtocol(ProtImportMicrographs,
                                    filesPath=self.dsXmipp.getFile('allMics'),
                                    samplingRate=1.237, voltage=300)
      protImport.setObjLabel('Import micrographs from xmipp tutorial ')
      self.launchProtocol(protImport)
      self.assertIsNotNone(protImport.outputMicrographs.getFileName(),
                           "There was a problem with the import")
      return protImport

    def _runImportCoordinates(self, protImportMics, idx = ''):
      prot = self.newProtocol(ProtImportCoordinates,
                               importFrom=ProtImportCoordinates.IMPORT_FROM_XMIPP,
                               filesPath=self.dsXmipp.getFile('autoPickingCoordinates'),
                               filesPattern='*.pos', boxSize=150,
                               scale=1.,
                               invertX=False,
                               invertY=False
                               )
      prot.inputMicrographs.set(protImportMics.outputMicrographs)
      prot.setObjLabel('Import coords from xmipp {}'.format(idx))
      self.launchProtocol(prot)
      self.assertIsNotNone(prot.outputCoordinates,
                           "There was a problem with the import")
      return prot

    def _runStreamCoordinates(self, protImportCoords, idx=''):
      kwargs = {"setof" : SET_OF_COORDINATES,
                "inputCoordinates" : protImportCoords.outputCoordinates,
                "nDim" : 20,
                "creationInterval" : 5,
                "extraRandomInterval" : 5,
                "delay" : 0
                }
      prot = self.newProtocol(ProtCreateStreamData, **kwargs)
      prot.setObjLabel('Streaming coords {}'.format(idx))
      self.proj.launchProtocol(prot, wait=False)
      return prot

    def _runDeepConsensusPicking(self, inpCoords, kwargs, case = ''):
      prot = self.newProtocol(XmippProtScreenDeepConsensus,
                                   inputCoordinates= inpCoords,
                                   **kwargs)

      prot.setObjLabel('Consensus picking {}'.format(case))
      self.launchProtocol(prot, wait=True)
      self.assertIsNotNone(prot.outputCoordinates,
                           "There was a problem with the consensus")
      self.assertIsNotNone(prot.outputParticles,
                           "There was a problem with the consensus")
      return prot

    def getDeepConsensusKwargs(self, case=1, inputCase=1):
      kwargs = {
        'nEpochs' : 1.0,
        'nModels' :2,
        'extractingBatch':3,
        'trainingBatch':3,
        'predictingBatch':3
      }
      #inputCase controls the input model of the protocol: previous protocol model, new model, pretrained
      inputCaseKwargs = {'numberOfThreads': 1} if inputCase<4 else {'numberOfThreads': 4}
      if inputCase in [1, 4]:
        inputCaseKwargs['modelInitialization'] = ADD_MODEL_TRAIN_NEW
      elif inputCase in [2, 5]:
        inputCaseKwargs['modelInitialization'] = ADD_MODEL_TRAIN_PRETRAIN
      elif inputCase in [3, 6]:
        inputCaseKwargs['modelInitialization'] = ADD_MODEL_TRAIN_PREVRUN
        inputCaseKwargs['continueRun'] = self.lastRun

      #case controls the behavior of the protocol. e.g: skip training, do preliminar predictions
      if case == 1:
        caseKwargs = {'doPreliminarPredictions': False, 'skipTraining': False}
      elif case == 2:
        caseKwargs = {'doPreliminarPredictions': False, 'skipTraining': True}
      elif case == 3:
        caseKwargs = {'doPreliminarPredictions': True, 'skipTraining': False}
      elif case == 4:
        caseKwargs = {'doPreliminarPredictions': True, 'skipTraining': True}

      else:
        caseKwargs={}

      kwargs.update(caseKwargs)
      kwargs.update(inputCaseKwargs)
      return kwargs

    def testDeepConsensusNew(self):
      #Testing the protocol on new model
      nCoordinateSets = 3
      inputCase, case = 4, 3

      inpCoords = self.prepareInput(case, nCoordinateSets)
      kwargs = self.getDeepConsensusKwargs(case, inputCase)
      self.lastRun = self._runDeepConsensusPicking(inpCoords, kwargs, case)

    def testDeepConsensusPretrain(self):
      #Testing the protocol on pretrained model
      nCoordinateSets = 3
      inputCase, case = 5, 1

      inpCoords = self.prepareInput(case, nCoordinateSets)
      kwargs = self.getDeepConsensusKwargs(case, inputCase)
      self.lastRun = self._runDeepConsensusPicking(inpCoords, kwargs, case)

    def testDeepConsensusPrevRun(self):
      #Testing the protocol on a model from a previous run
      nCoordinateSets = 3
      inputCase, case = 6, 4

      inpCoords = self.prepareInput(case, nCoordinateSets)
      prev_kwargs = self.getDeepConsensusKwargs(case=2, inputCase=5)
      self.lastRun = self._runDeepConsensusPicking(inpCoords, prev_kwargs, case)

      kwargs = self.getDeepConsensusKwargs(case, inputCase)
      self._runDeepConsensusPicking(inpCoords, kwargs, case)


    def _updateProtocol(self, prot):
      prot2 = getProtocolFromDb(prot.getProject().path,
                                prot.getDbPath(),
                                prot.getObjId())
      # Close DB connections
      prot2.getProject().closeMapper()
      prot2.closeMappers()
      return prot2

    def checkStreamOutput(self, prots, outputName):
      for prot in prots:
        if not prot.hasAttribute(outputName):
          return False
      return True

    def prepareInput(self, case, nCoordSets=3):
      protImpMics = self._runInportMicrographs()
      protImpCoords = self._runImportCoordinates(protImpMics, case)

      protsStreamCoords = []
      for i in range(nCoordSets):
        protsStreamCoords.append(self._runStreamCoordinates(protImpCoords, i))

      for i in range(nCoordSets):
        self._waitOutput(protsStreamCoords[i], "outputCoordinates")

      inpCoords = []
      for prot in protsStreamCoords:
        point = Pointer(prot, extended="outputCoordinates")
        inpCoords.append(point)
      return inpCoords


