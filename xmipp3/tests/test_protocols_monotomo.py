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

import unittest, sys

from pyworkflow.em import exists
from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from pyworkflow.em.protocol import ProtImportVolumes

from xmipp3.protocols import XmippProtMonoTomo, XmippProtCropResizeVolumes


class TestMonoTomoBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='resmap'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.map3D = cls.dataset.getFile('betagal')
        cls.half1 = cls.dataset.getFile('betagal_half1')
        cls.half2 = cls.dataset.getFile('betagal_half2')

    @classmethod
    def runImportVolumes(cls, pattern, samplingRate):
        """ Run an Import volumes protocol. """
        cls.protImport = cls.newProtocol(ProtImportVolumes,
                                         filesPath=pattern,
                                         samplingRate=samplingRate
                                         )
        cls.launchProtocol(cls.protImport)
	cls.protResize = cls.newProtocol(XmippProtCropResizeVolumes,
                                        inputVolumes = cls.protImport.outputVolume,
					doResize = True,
					resizeOption = 2,
					resizeFactor = 5)
	cls.launchProtocol(cls.protResize)
        return cls.protResize

class TestMonoTomo(TestMonoTomoBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestMonoTomoBase.setData()
        cls.protImportHalf1 = cls.runImportVolumes(cls.half1, 3.54)
        cls.protImportHalf2 = cls.runImportVolumes(cls.half2, 3.54)


    def testMonoTomo(self):
        MonoTomo = self.newProtocol(XmippProtMonoTomo,
                                   objLabel='two halves monores',
                                   inputVolume=self.protImportHalf1.outputVol,
                                   inputVolume2=self.protImportHalf2.outputVol,
                                   provideMaskInHalves=True,
                                   useMask=False,
                                   minRes=1,
                                   maxRes=25,
                                   )
        self.launchProtocol(MonoTomo)
        self.assertTrue(exists(MonoTomo._getExtraPath('mgresolution.mrc')),
                        "MonoTomo has failed")
 
