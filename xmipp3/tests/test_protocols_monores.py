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

from xmipp3.protocols import XmippProtMonoRes, XmippProtCreateMask3D


class TestMonoResBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='resmap'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.map3D = cls.dataset.getFile('betagal')
        cls.half1 = cls.dataset.getFile('betagal_half1')
        cls.half2 = cls.dataset.getFile('betagal_half2')
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


class TestMonoRes(TestMonoResBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestMonoResBase.setData()
        cls.protImportVol = cls.runImportVolumes(cls.map3D, 3.54)
        cls.protImportHalf1 = cls.runImportVolumes(cls.half1, 3.54)
        cls.protImportHalf2 = cls.runImportVolumes(cls.half2, 3.54)
        cls.protCreateMask = cls.runCreateMask(cls.protImportVol.outputVolume, 0.02)

    def testMonoRes1(self):
        MonoRes = self.newProtocol(XmippProtMonoRes,
                                   objLabel='single volume monores',
                                   useHalfVolumes=False,
                                   hasHalfVolumesFile=False,
                                   fullMap=self.protImportVol.outputVolume,
                                   mask=self.protCreateMask.outputMask,
                                   minRes=1,
                                   maxRes=25,
                                   )
        self.launchProtocol(MonoRes)
        self.assertTrue(exists(MonoRes._getExtraPath('monoresResolutionMap.mrc')),
                        "MonoRes (no split, no premasked) has failed")
 
    def testMonoRes2(self):
        MonoRes = self.newProtocol(XmippProtMonoRes,
                                   objLabel='two halves monores',
                                   useHalfVolumes=True,
                                   hasHalfVolumesFile=False,
                                   halfMap1=self.protImportHalf1.outputVolume,
                                   halfMap2=self.protImportHalf2.outputVolume,
                                   provideMaskInHalves=True,
                                   mask=self.protCreateMask.outputMask,
                                   minRes=1,
                                   maxRes=25,
                                   )
        self.launchProtocol(MonoRes)
        self.assertTrue(exists(MonoRes._getExtraPath('monoresResolutionMap.mrc')),
                        "MonoRes (split, pre-masked, no filter) has failed")
 

