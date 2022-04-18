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
from pwem.protocols import ProtImportMicrographs, ProtSubSet

from xmipp3.protocols.protocol_extract_particles import *
from xmipp3.protocols.protocol_cl2d import *
from xmipp3.protocols.protocol_center_particles import *

ProtCTFFind = Domain.importFromPlugin('cistem.protocols', 'CistemProtCTFFind',
                                      doRaise=True)
EmanProtAutopick = Domain.importFromPlugin('eman2.protocols',
                                                   'EmanProtAutopick',
                                           doRaise=True)


# Number of mics to be processed
NUM_MICS = 5

class TestCenterParticles(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')

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


    def calculateCtf(self, inputMics):
        protCTF = ProtCTFFind()
        protCTF.inputMicrographs.set(inputMics)
        # Gone in new version: protCTF.ctfDownFactor.set(1.0)
        protCTF.lowRes.set(44)
        protCTF.highRes.set(15)
        self.launchProtocol(protCTF)

        return protCTF


    def runPicking(self, inputMicrographs):
        """ Run a particle picking. """
        protPicking = EmanProtAutopick(boxSize=64,
                                       numberOfThreads=1,
                                       numberOfMpi=1,
                                       boxerMode=3,
                                       gaussLow=0.001)
        protPicking.inputMicrographs.set(inputMicrographs)
        self.launchProtocol(protPicking)

        return protPicking

    def runExtractParticles(self, inputCoord, setCtfs):
        protExtract = self.newProtocol(XmippProtExtractParticles,
                                       boxSize=64,
                                       doInvert = True,
                                       doFlip = False)

        protExtract.inputCoordinates.set(inputCoord)
        protExtract.ctfRelations.set(setCtfs)

        self.launchProtocol(protExtract)

        return protExtract

    def runClassify(self, inputParts):
        numClasses = int(inputParts.getSize()/1000)
        if numClasses<=2:
            numClasses=4
        protClassify = self.newProtocol(XmippProtCL2D,
                                        numberOfIterations = 2,
                                        numberOfClasses=numClasses,
                                        numberOfInitialClasses=numClasses)
        protClassify.inputParticles.set(inputParts)
        self.launchProtocol(protClassify)

        return protClassify, numClasses

    def runRealign(self, inputClasses, inputMics):
        protRealing = self.newProtocol(XmippProtCenterParticles)
        protRealing.inputClasses.set(inputClasses)
        protRealing.inputMics.set(inputMics)
        self.launchProtocol(protRealing)

        return protRealing



    def test_pattern(self):

        protImportMics = self.importMicrographs()
        if protImportMics.isFailed():
            self.assertTrue(False)

        if NUM_MICS<20:
            protSubsetMics = self.subsetMics(protImportMics.outputMicrographs)
            if protSubsetMics.isFailed():
                self.assertTrue(False)
            outMics = protSubsetMics.outputMicrographs

        protCtf = self.calculateCtf(outMics)
        if protCtf.isFailed():
            self.assertTrue(False)

        protPicking = self.runPicking(outMics)
        if protPicking.isFailed():
            self.assertTrue(False)

        protExtract = self.runExtractParticles\
            (protPicking.outputCoordinates,
             protCtf.outputCTF)
        if protExtract.isFailed():
            self.assertTrue(False)

        protClassify, numClasses = self.runClassify(protExtract.outputParticles)
        if protClassify.isFailed():
            self.assertTrue(False)
        if not protClassify.hasAttribute('outputClasses'):
            self.assertTrue(False)
        if protClassify.outputClasses.getSize() != numClasses:
            self.assertTrue(False)

        protRealing = self.runRealign(protClassify.outputClasses,
                                               outMics)
        if protRealing.isFailed():
            self.assertTrue(False)
        if not protRealing.hasAttribute('outputClasses') or not \
                protRealing.hasAttribute('outputParticles'):
            self.assertTrue(False)
        if protRealing.outputClasses.getSize() != numClasses:
            self.assertTrue(False)





