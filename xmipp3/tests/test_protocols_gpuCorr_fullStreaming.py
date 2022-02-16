# ***************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
# *              David Strelak (dstrelak@cnb.csic.es)
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

import time

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pyworkflow.plugin import Domain
from pwem.protocols import ProtImportMicrographs, ProtCreateStreamData
from pwem.protocols.protocol_create_stream_data import SET_OF_MICROGRAPHS
from pyworkflow.protocol import getProtocolFromDb

from xmipp3.protocols import XmippProtPreprocessMicrographs
from xmipp3.protocols.protocol_extract_particles import *
from xmipp3.protocols.protocol_classification_gpuCorr_full import *

ProtCTFFind = Domain.importFromPlugin('cistem.protocols', 'CistemProtCTFFind',
                                      doRaise=True)
EmanProtAutopick = Domain.importFromPlugin('eman2.protocols',
                                                   'EmanProtAutopick',
                                                   doRaise=True)


# Number of mics to be processed
NUM_MICS = 5

class GpuCorrCommon(BaseTest):
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

    def importMicrographsStr(self, fnMics):
        kwargs = {'inputMics': fnMics,
                  'nDim': NUM_MICS,
                  'creationInterval': 15,
                  'delay': 10,
                  'setof': SET_OF_MICROGRAPHS  # SetOfMics
                  }
        protStream = self.newProtocol(ProtCreateStreamData, **kwargs)
        protStream.setObjLabel('create Stream Mic')
        self.proj.launchProtocol(protStream, wait=False)

        return protStream

    def invertContrast(self, inputMics):
        protInvCont = XmippProtPreprocessMicrographs(doInvert=True)
        protInvCont.inputMicrographs.set(inputMics)
        self.proj.launchProtocol(protInvCont, wait=False)

        return protInvCont

    def calculateCtf(self, inputMics):
        protCTF = ProtCTFFind(useCftfind4=True)
        protCTF.inputMicrographs.set(inputMics)
        # Gone in new version: protCTF.ctfDownFactor.set(1.0)
        protCTF.lowRes.set(44)
        protCTF.highRes.set(15)
        self.proj.launchProtocol(protCTF, wait=False)

        return protCTF


    def runPicking(self, inputMicrographs):
        """ Run a particle picking. """
        protPicking = EmanProtAutopick(boxSize=64, boxerMode=3, gaussLow=0.01)
        protPicking.inputMicrographs.set(inputMicrographs)
        self.proj.launchProtocol(protPicking, wait=False)

        return protPicking

    def runExtractParticles(self, inputCoord, setCtfs):
        self.protExtract = self.newProtocol(XmippProtExtractParticles,
                                       boxSize=64,
                                       doInvert = False,
                                       doFlip = False)

        self.protExtract.inputCoordinates.set(inputCoord)
        self.protExtract.ctfRelations.set(setCtfs)

        self.proj.launchProtocol(self.protExtract, wait=False)

        return self.protExtract

    def runClassify(self, inputParts):
        protClassify = self.newProtocol(XmippProtStrGpuCrrCL2D,
                                        numberOfMpi=4)

        protClassify.inputParticles.set(inputParts)
        self.proj.launchProtocol(protClassify, wait=False)

        return protClassify


    def _updateProtocol(self, prot):
            prot2 = getProtocolFromDb(prot.getProject().path,
                                      prot.getDbPath(),
                                      prot.getObjId())
            # Close DB connections
            prot2.getProject().closeMapper()
            prot2.closeMappers()
            return prot2


    def run_common_workflow(self):
        protImportMics = self.importMicrographs()
        self.assertFalse(protImportMics.isFailed(), 'ImportMics has failed.')

        protImportMicsStr = self.importMicrographsStr(protImportMics.outputMicrographs)
        self._waitOutput(protImportMicsStr,'outputMicrographs')

        protInvContr = self.invertContrast(protImportMicsStr.outputMicrographs)
        self._waitOutput(protInvContr,'outputMicrographs')

        protCtf = self.calculateCtf(protInvContr.outputMicrographs)
        self._waitOutput(protCtf,'outputCTF')

        protPicking = self.runPicking(protInvContr.outputMicrographs)
        self._waitOutput(protPicking,'outputCoordinates')

        self.runExtractParticles(protPicking.outputCoordinates,
                                               protCtf.outputCTF)
        self._waitOutput(self.protExtract,'outputParticles')

        self.verify_classification()


class TestGpuCorrFullStreaming(GpuCorrCommon):
    def test_full_streaming(self):
        self.run_common_workflow()

    def verify_classification(self):
        protClassify = self.runClassify(self.protExtract.outputParticles)

        self._waitOutput(protClassify, "outputClasses", timeOut=180)

        self.assertTrue(protClassify.hasAttribute('outputAverages'),
                        'GL2D-streaming has no outputAverages at the end.')
        self.assertTrue(protClassify.hasAttribute('outputClasses'),
                        'GL2D-streaming has no outputClasses at the end.')
