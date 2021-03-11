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

    def getDeepConsensusKwargs(self, case=1):
      ADD_MODEL_TRAIN_NEW = 0
      ADD_MODEL_TRAIN_PRETRAIN = 1
      ADD_MODEL_TRAIN_PREVRUN = 2

      ADD_DATA_TRAIN_NONE = 0
      ADD_DATA_TRAIN_PRECOMP = 1
      ADD_DATA_TRAIN_CUST = 2

      ADD_DATA_TRAIN_CUSTOM_OPT_PARTS = 0
      ADD_DATA_TRAIN_CUSTOM_OPT_COORS = 1

      kwargs = {
        'nEpochs' : 2.0,
        'nModels' :2,
        'extractingBatch':3,
        'trainingBatch':3,
        'predictingBatch':3
      }
      if case == 1:
        #Simplest case
        caseKwargs = {}
      elif case == 2:
        #Pretrained model, do testing, additional data particles, do preliminar predictions
        caseKwargs = {'doPreliminarPredictions': True,
                      'modelInitialization': ADD_MODEL_TRAIN_PRETRAIN,
                      'addTrainingData': ADD_DATA_TRAIN_CUST,
                      'trainingDataType':ADD_DATA_TRAIN_CUSTOM_OPT_PARTS,
                      'trainTrueSetOfParticles': self.lastRun.outputParticles,
                      'trainFalseSetOfParticles': self.lastRun.outputParticles
                      }
      elif case == 3:
        #Previous run model, additional data coords
        caseKwargs = {'modelInitialization':ADD_MODEL_TRAIN_PREVRUN,
                      'continueRun':self.lastRun,
                      'doTesting': True,
                      'testTrueSetOfParticles': self.lastRun.outputParticles,
                      'testFalseSetOfParticles': self.lastRun.outputParticles,
                      'addTrainingData': ADD_DATA_TRAIN_CUST,
                      'trainingDataType': ADD_DATA_TRAIN_CUSTOM_OPT_COORS,
                      'trainTrueSetOfCoords': self.lastRun.outputCoordinates,
                      'trainFalseSetOfCoords': self.lastRun.outputCoordinates
                      }
      else:
        caseKwargs={}

      kwargs.update(caseKwargs)
      return kwargs


    def testImportCoordinates(self):
      nCoordinateSets = 3
      protImpMics = self._runInportMicrographs()

      for case in range(1,4):
        protImpCoords = self._runImportCoordinates(protImpMics, case)

        protsStreamCoords = []
        for i in range(nCoordinateSets):
          protsStreamCoords.append(self._runStreamCoordinates(protImpCoords, i))

        for i in range(nCoordinateSets):
          self._waitOutput(protsStreamCoords[i], "outputCoordinates")

        inpCoords = []
        for prot in protsStreamCoords:
          point = Pointer(prot, extended="outputCoordinates")
          inpCoords.append(point)
        kwargs = self.getDeepConsensusKwargs(case)

        self.lastRun = self._runDeepConsensusPicking(inpCoords, kwargs, case)


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


