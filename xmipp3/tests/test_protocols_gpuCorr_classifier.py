# ***************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
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
# ***************************************************************************/

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pyworkflow.plugin import Domain
from pwem.protocols import ProtImportAverages, ProtImportParticles
from xmipp3.protocols.protocol_preprocess.protocol_crop_resize import XmippProtCropResizeParticles

from xmipp3.protocols.protocol_classification_gpuCorr import *


class TestGpuCorrClassifier(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')

    def importParticles(self, path):
        """ Import an EMX file with Particles and defocus
        """
        prot = self.newProtocol(ProtImportParticles,
                                objLabel='from relion',
                                importFrom=ProtImportParticles.IMPORT_FROM_RELION,
                                starFile=self.dsRelion.getFile(path),
                                magnification=10000,
                                samplingRate=7.08,
                                haveDataBeenPhaseFlipped=True)
        self.launchProtocol(prot)

        return prot

    def resizeParticles(self, particles):
        protResize = self.newProtocol(XmippProtCropResizeParticles,
                                         doResize=True,
                                         resizeOption=1,
                                         resizeDim=60)
        protResize.inputParticles.set(particles)
        self.launchProtocol(protResize)
        return protResize

    def importAverages(self):
        prot = self.newProtocol(ProtImportAverages,
                                filesPath=self.dsRelion.getFile(
                                    'import/averages.mrcs'),
                                samplingRate=1.0)
        self.launchProtocol(prot)

        return prot

    def runClassify(self, inputParts):
        numClasses = int(inputParts.getSize()/1000)
        if numClasses<=2:
            numClasses=4
        protClassify = self.newProtocol(XmippProtGpuCrrCL2D,
                                        useReferenceImages=False,
                                        numberOfClasses=numClasses)
        protClassify.inputParticles.set(inputParts)
        self.launchProtocol(protClassify)

        return protClassify, numClasses

    def runClassify2(self, inputParts, inputRefs):
        numClasses = int(inputParts.getSize()/1000)
        if numClasses<=inputRefs.getSize():
            numClasses=inputRefs.getSize()
        protClassify = self.newProtocol(XmippProtGpuCrrCL2D,
                                        useReferenceImages=True,
                                        numberOfClasses=numClasses)
        protClassify.inputParticles.set(inputParts)
        protClassify.referenceImages.set(inputRefs)
        self.launchProtocol(protClassify)

        return protClassify, numClasses

    def runClassify3(self, inputParts, inputRefs):
        numClasses = inputRefs.getSize()+2
        protClassify = self.newProtocol(XmippProtGpuCrrCL2D,
                                        useReferenceImages=True,
                                        numberOfClasses=numClasses)
        protClassify.inputParticles.set(inputParts)
        protClassify.referenceImages.set(inputRefs)
        protClassify.useAttraction.set(False)
        self.launchProtocol(protClassify)

        return protClassify, numClasses

    def test_pattern(self):

        protImportAvgs = self.importAverages()
        self.assertFalse(protImportAvgs.isFailed(), 'ImportAverages has failed.')

        protResize = self.resizeParticles(protImportAvgs.outputAverages)
        self.assertFalse(protResize.isFailed(), 'Resize has failed.')

        path = 'import/case2/relion_it015_data.star'
        protImportParts = self.importParticles(path)
        self.assertFalse(protImportParts.isFailed(), 'Importparticles has failed.')


        protClassify, numClasses = self.runClassify(protImportParts.outputParticles)
        self.assertFalse(protClassify.isFailed(), 'GL2D (1) has failed.')
        self.assertTrue(protClassify.hasAttribute('outputClasses'),
                        'GL2D (1) has no outputClasses.')
        self.assertEqual(protClassify.outputClasses.getSize(), numClasses,
                         'GL2D (1) returned a wrong number of classes.')

        protClassify2, numClasses2 = self.runClassify2(
            protImportParts.outputParticles, protResize.outputAverages)
        self.assertFalse(protClassify2.isFailed(), 'GL2D (2) has failed.')
        self.assertTrue(protClassify2.hasAttribute('outputClasses'),
                        'GL2D (2) has no outputClasses.')
        self.assertEqual(protClassify2.outputClasses.getSize(), numClasses2,
                         'GL2D (2) returned a wrong number of classes.')

        protClassify3, numClasses3 = self.runClassify3(
            protImportParts.outputParticles, protResize.outputAverages)
        self.assertFalse(protClassify3.isFailed(), 'GL2D (3) has failed.')
        self.assertTrue(protClassify3.hasAttribute('outputClasses'),
                        'GL2D (3) has no outputClasses.')
        self.assertEqual(protClassify3.outputClasses.getSize(), numClasses3,
                         'GL2D (3) returned a wrong number of classes.')
