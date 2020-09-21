# **************************************************************************
# *
# * Authors:    Jose Luis Vilas Prieto (jlvilas@cnb.csic.es)
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
from pwem.protocols import ProtImportVolumes, exists
from pyworkflow.tests import BaseTest, DataSet, setupTestProject

from xmipp3.protocols import XmippProtMonoDir, XmippProtCreateMask3D


class TestMonoDirBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='resmap'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.map3D = cls.dataset.getFile('betagal')
        cls.mask = cls.dataset.getFile('betagal_mask')

    @classmethod
    def runImportVolumes(cls, pattern, samplingRate):
        """ Run an Import volumes protocol. """
        cls.protImport = cls.newProtocol(ProtImportVolumes,
                                         filesPath=pattern,
                                         samplingRate=samplingRate
                                         )
        cls.launchProtocol(cls.protImport)
        return cls.protImport

    @classmethod
    def runCreateMask(cls, pattern, thr):
        """ Create a volume mask. """
        cls.msk = cls.newProtocol(XmippProtCreateMask3D,
                                  inputVolume=pattern,
                                  volumeOperation=0,  # OPERATION_THRESHOLD,
                                  threshold=thr,
                                  doSmall=False,
                                  smallSize=False,
                                  doBig=False,
                                  doSymmetrize=False,
                                  doMorphological=False,
                                  doInvert=False,
                                  doSmooth=False,
                                  sigmaConvolution=2
                                  )
        cls.launchProtocol(cls.msk)
        return cls.msk


class TestMonoDir(TestMonoDirBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestMonoDirBase.setData()
        cls.protImportVol = cls.runImportVolumes(cls.map3D, 3.54)
        cls.protCreateMask = cls.runCreateMask(cls.protImportVol.outputVolume, 0.02)

    def testMonoDir(self):
        MonoDir = self.newProtocol(XmippProtMonoDir,
                                   objLabel='single volume monodir',
                                   inputVolumes=self.protImportVol.outputVolume,
                                   Mask=self.protCreateMask.outputMask,
				   resstep=0.5,
				   significance=0.95,
				   fast=True,
  				   isPremasked=False
                                   )
        self.launchProtocol(MonoDir)
        self.assertTrue(exists(MonoDir._getExtraPath('meanResdoa.vol')),
                        "MonoDir has failed")
 
