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
        cls.dsXmipp = DataSet.getDataSet('xmipp_tutorial')

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
                               filesPath=self.dsXmipp.getFile('pickingXmipp'),
                               filesPattern='*.pos', boxSize=550,
                               scale=3.,
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
                "creationInterval" : 10,
                "extraRandomInterval" : 10,
                "delay" : 0
                }
      prot = self.newProtocol(ProtCreateStreamData, **kwargs)
      prot.setObjLabel('Streaming coords {}'.format(idx))
      self.proj.launchProtocol(prot, wait=False)
      return prot

    def _runDeepConsensusPicking(self, inputCoords, kwargs):
      print(kwargs)
      prot = self.newProtocol(XmippProtScreenDeepConsensus,
                              inputCoordinates = inputCoords,
                              **kwargs)
      prot.setObjLabel('Consensus picking')
      self.proj.launchProtocol(prot)
      return prot

    def getDeepConsensusKwargs(self, protsStreamCoords, case=1):
      kwargs = {}
      return kwargs


    def testImportCoordinates(self):
      nCoordinateSets = 3
      protImpMics = self._runInportMicrographs()

      protsImpCoords = []
      for i in range(nCoordinateSets):
        protsImpCoords.append(self._runImportCoordinates(protImpMics, i))

      protsStreamCoords = []
      for i in range(nCoordinateSets):
        protsStreamCoords.append(self._runStreamCoordinates(protsImpCoords[i], i))


      counter=1
      while not self.checkStreamOutput(protsStreamCoords, 'outputCoordinates'):
        time.sleep(2)
        if counter > 50:
          break
        counter+=1
        for i in range(len(protsStreamCoords)):
          protsStreamCoords[i] = self._updateProtocol(protsStreamCoords[i])

      inpCoords = []
      for prot in protsStreamCoords:
        inpCoords.append(prot.outputCoordinates)
      kwargs = self.getDeepConsensusKwargs(protsStreamCoords)
      self._runDeepConsensusPicking(inpCoords, kwargs)

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


