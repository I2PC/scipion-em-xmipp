# **************************************************************************
# *
# * Authors:    Daniel Del Hoyo Gomez (ddelhoyo@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *
# **************************************************************************

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import ProtImportPdb, ProtImportVolumes
from xmipp3.protocols import XmippProtValFit

class TestValidateFSCQ(BaseTest):
  @classmethod
  def setUpClass(cls):
    cls.ds = DataSet.getDataSet('model_building_tutorial')

    setupTestProject(cls)
    cls._runImportPDB()
    cls._runImportVolume()

  @classmethod
  def _runImportPDB(cls):
    protImportPDB = cls.newProtocol(
      ProtImportPdb,
      inputPdbData=0,
      pdbId="5ni1")
    cls.launchProtocol(protImportPDB)
    cls.protImportPDB = protImportPDB

  @classmethod
  def _runImportVolume(cls):
    args = {'filesPath': cls.ds.getFile(
      'volumes/emd_3488.map'),
      'samplingRate': 1.05,
      'setHalfMaps': True,
      'half1map': cls.ds.getFile('volumes/emd_3488_Noisy_half1.vol'),
      'half2map': cls.ds.getFile('volumes/emd_3488_Noisy_half2.vol'),
      'setOrigCoord': True,
      'x': 0.0,
      'y': 0.0,
      'z': 0.0
    }
    protImportVolume = cls.newProtocol(ProtImportVolumes, **args)
    cls.launchProtocol(protImportVolume)
    cls.protImportVolume = protImportVolume

  def _runFSCQ(self):
    protFSCQ = self.newProtocol(
      XmippProtValFit,
      fromFile=False,
      inputPDBObj=self.protImportPDB.outputPdb,
      inputVolume=self.protImportVolume.outputVolume)

    self.launchProtocol(protFSCQ)
    pdbOut = getattr(protFSCQ, 'outputAtomStruct', None)
    self.assertIsNotNone(pdbOut)

  def testFSCQ(self):
    self._runFSCQ()




