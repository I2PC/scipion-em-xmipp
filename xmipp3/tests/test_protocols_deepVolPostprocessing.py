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

from xmipp3.protocols import XmippProtDeepVolPostProc, XmippProtCreateMask3D
from xmipp3.protocols.protocol_postProcessing_deepPostProcessing import POSTPROCESS_VOL_BASENAME


class TestDeepVolPostProcessingBase(BaseTest):
    '''
    scipion tests xmipp3.tests.test_protocols_deepVolPostprocessing
    '''

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
    def runImportVolAndHalf(cls, patternVol, fnameHalf1, fnameHalf2, samplingRate):
        """ Run an Import volumes protocol. """
        cls.protImport = cls.newProtocol(ProtImportVolumes,
                                         filesPath=patternVol,
                                         setHalfMaps=True,
                                         half1map=fnameHalf1,
                                         half2map=fnameHalf2,
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



class TestDeepVolPostProcessing(TestDeepVolPostProcessingBase):
    @classmethod
    def setUpClass(cls, samplingRate=1): # Actual samplingRate=3.54 but using 1 to speed up computations
        setupTestProject(cls)
        TestDeepVolPostProcessingBase.setData()
        cls.protImportVol = cls.runImportVolumes(cls.map3D, samplingRate)
        cls.protImportHalf1 = cls.runImportVolumes(cls.half1, samplingRate)
        cls.protImportHalf2 = cls.runImportVolumes(cls.half2, samplingRate)
        cls.protImportVolWithHalfs= cls.runImportVolAndHalf(cls.map3D, cls.half1, cls.half2, samplingRate)
        # cls.protCreateMask = cls.runCreateMask(cls.protImportVol.outputVolume, 0.02)

    def _checkMeanVal(self, fname, expectedMean, r_delta=0.5):
      import numpy as np
      from pwem.emlib import Image
      imgHandler = Image()
      imgHandler.read(fname)
      obtainedVol = imgHandler.getData()
      central_cube_slice= obtainedVol[40:60,40:60,40:60]
      self.assertAlmostEqual( np.mean(central_cube_slice), expectedMean, delta=r_delta*abs(expectedMean), msg="Error expected mean volume does not match")

    def testDeepVolPostpro1(self):
        deepPostProc = self.newProtocol(XmippProtDeepVolPostProc,
                                   inputVolume=self.protImportVol.outputVolume,
                                   normalization= XmippProtDeepVolPostProc.NORMALIZATION_AUTO
                                   )
        self.launchProtocol(deepPostProc)
        self.assertTrue(exists(deepPostProc._getExtraPath(POSTPROCESS_VOL_BASENAME)),
                        "Deep Volume Postprocessing has failed")

        self._checkMeanVal( deepPostProc._getExtraPath(POSTPROCESS_VOL_BASENAME), 0.1224, r_delta=0.5)

    def testDeepVolPostpro2(self):
        deepPostProc = self.newProtocol(XmippProtDeepVolPostProc,
                                        useHalfMapsInsteadVol=True,
                                        halfMapsAttached=False,
                                        inputHalf1=self.protImportHalf1.outputVolume,
                                        inputHalf2=self.protImportHalf2.outputVolume,
                                        normalization= XmippProtDeepVolPostProc.NORMALIZATION_AUTO
                                   )
        self.launchProtocol(deepPostProc)
        self.assertTrue(exists(deepPostProc._getExtraPath(POSTPROCESS_VOL_BASENAME)),
                        "Deep Volume Postprocessing has failed")

        self._checkMeanVal(deepPostProc._getExtraPath(POSTPROCESS_VOL_BASENAME), 0.1102, r_delta=0.25)

    def testDeepVolPostpro3(self):
        deepPostProc = self.newProtocol(XmippProtDeepVolPostProc,
                                        useHalfMapsInsteadVol=True,
                                        halfMapsAttached=True,
                                        inputVolume=self.protImportVolWithHalfs.outputVolume,
                                        normalization= XmippProtDeepVolPostProc.NORMALIZATION_AUTO
                                   )
        self.launchProtocol(deepPostProc)
        self.assertTrue(exists(deepPostProc._getExtraPath(POSTPROCESS_VOL_BASENAME)),
                        "Deep Volume Postprocessing has failed")
        self._checkMeanVal( deepPostProc._getExtraPath(POSTPROCESS_VOL_BASENAME), 0.1102, r_delta=0.5)
