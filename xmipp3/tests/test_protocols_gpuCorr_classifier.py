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
from pwem.protocols import ProtImportAverages, ProtImportMicrographs, ProtSubSet

from xmipp3.protocols import XmippProtPreprocessMicrographs
from xmipp3.protocols.protocol_extract_particles import *
from xmipp3.protocols.protocol_classification_gpuCorr import *

ProtCTFFind = Domain.importFromPlugin('grigoriefflab.protocols', 'ProtCTFFind',
                                      doRaise=True)
SparxGaussianProtPicking = Domain.importFromPlugin('eman2.protocols',
                                                   'SparxGaussianProtPicking',
                                                   doRaise=True)


# Number of mics to be processed
NUM_MICS = 5 #maximum the number of mics in the relion set

class TestGpuCorrClassifier(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')

    def importAverages(self):
        prot = self.newProtocol(ProtImportAverages,
                                filesPath=self.dsRelion.getFile(
                                    'import/averages.mrcs'),
                                samplingRate=1.0)
        self.launchProtocol(prot)

        return prot

    def importMicrographs(self):
        prot = self.newProtocol(ProtImportMicrographs,
                                filesPath=self.dsRelion.getFile('micrographs'),
                                filesPattern='*.mrc',
                                samplingRateMode=1,
                                magnification=79096,
                                scannedPixelSize=56, voltage=300,
                                sphericalAberration=2.0)
        self.launchProtocol(prot)

        return prot


    def subsetMics(self, inputMics):
        protSubset = ProtSubSet()
        protSubset.inputFullSet.set(inputMics)
        protSubset.chooseAtRandom.set(True)
        protSubset.nElements.set(NUM_MICS)
        self.launchProtocol(protSubset)

        return protSubset

    def invertContrast(self, inputMics):
        protInvCont = XmippProtPreprocessMicrographs(doInvert=True)
        protInvCont.inputMicrographs.set(inputMics)
        self.launchProtocol(protInvCont)

        return protInvCont

    def calculateCtf(self, inputMics):
        protCTF = ProtCTFFind(useCftfind4=True, numberOfThreads=6)
        protCTF.inputMicrographs.set(inputMics)
        protCTF.ctfDownFactor.set(1.0)
        protCTF.lowRes.set(0.05)
        protCTF.highRes.set(0.5)
        self.launchProtocol(protCTF)

        return protCTF


    def runPicking(self, inputMicrographs):
        """ Run a particle picking. """
        protPicking = SparxGaussianProtPicking(boxSize=64,
                                               numberOfThreads=6,
                                               numberOfMpi=1,
                                               lowerThreshold=0.01)
        protPicking.inputMicrographs.set(inputMicrographs)
        self.launchProtocol(protPicking)

        return protPicking

    def runExtractParticles(self, inputCoord, setCtfs):
        protExtract = self.newProtocol(XmippProtExtractParticles,
                                       boxSize=64,
                                       numberOfThreads=6,
                                       doInvert = False,
                                       doFlip = False)

        protExtract.inputCoordinates.set(inputCoord)
        protExtract.ctfRelations.set(setCtfs)

        self.launchProtocol(protExtract)

        return protExtract

    def runClassify(self, inputParts):
        numClasses = int(inputParts.getSize()/1000)
        if numClasses<2:
            numClasses=4
        protClassify = self.newProtocol(XmippProtGpuCrrCL2D,
                                        useReferenceImages=False,
                                        numberOfClasses=numClasses)
        protClassify.inputParticles.set(inputParts)
        self.launchProtocol(protClassify)

        return protClassify, numClasses

    def runClassify2(self, inputParts, inputRefs):
        numClasses = int(inputParts.getSize()/1000)
        if numClasses<inputRefs.getSize():
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

        protImportMics = self.importMicrographs()
        self.assertFalse(protImportMics.isFailed(), 'ImportMics has failed.')

        protImportAvgs = self.importAverages()
        self.assertFalse(protImportAvgs.isFailed(), 'ImportAverages has failed.')

        if NUM_MICS<20:
            protSubsetMics = self.subsetMics(protImportMics.outputMicrographs)
            self.assertFalse(protSubsetMics.isFailed(), 'protSubsetMics has failed.')
            outMics = protSubsetMics.outputMicrographs

        protInvContr = self.invertContrast(outMics)
        self.assertFalse(protInvContr.isFailed(), 'protInvContr has failed.')
        outMics = protInvContr.outputMicrographs

        protCtf = self.calculateCtf(outMics)
        self.assertFalse(protCtf.isFailed(), 'CTFfind4 has failed.')

        protPicking = self.runPicking(outMics)
        self.assertFalse(protPicking.isFailed(), 'Eamn Sparx has failed.')

        protExtract = self.runExtractParticles(protPicking.outputCoordinates,
                                               protCtf.outputCTF)
        self.assertFalse(protExtract.isFailed(), 'Extract particles has failed.')

        protClassify, numClasses = self.runClassify(protExtract.outputParticles)
        self.assertFalse(protClassify.isFailed(), 'GL2D (1) has failed.')
        self.assertTrue(protClassify.hasAttribute('outputClasses'),
                        'GL2D (1) has no outputClasses.')
        self.assertEqual(protClassify.outputClasses.getSize(), numClasses,
                         'GL2D (1) returned a wrong number of classes.')

        protClassify2, numClasses2 = self.runClassify2(
            protExtract.outputParticles, protImportAvgs.outputAverages)
        self.assertFalse(protClassify2.isFailed(), 'GL2D (2) has failed.')
        self.assertTrue(protClassify2.hasAttribute('outputClasses'),
                        'GL2D (2) has no outputClasses.')
        self.assertEqual(protClassify2.outputClasses.getSize(), numClasses2,
                         'GL2D (2) returned a wrong number of classes.')

        protClassify3, numClasses3 = self.runClassify3(
            protExtract.outputParticles, protImportAvgs.outputAverages)
        self.assertFalse(protClassify3.isFailed(), 'GL2D (3) has failed.')
        self.assertTrue(protClassify3.hasAttribute('outputClasses'),
                        'GL2D (3) has no outputClasses.')
        self.assertEqual(protClassify3.outputClasses.getSize(), numClasses3,
                         'GL2D (3) returned a wrong number of classes.')
