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

import time

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pyworkflow.plugin import Domain
from pwem.protocols import (ProtImportAverages, ProtImportMicrographs,
                            ProtCreateStreamData)
from pwem.protocols.protocol_create_stream_data import SET_OF_MICROGRAPHS
from pyworkflow.protocol import getProtocolFromDb

from xmipp3.protocols import XmippProtPreprocessMicrographs
from xmipp3.protocols.protocol_extract_particles import *
from xmipp3.protocols.protocol_classification_gpuCorr_semi import *

ProtCTFFind = Domain.importFromPlugin('cistem.protocols', 'CistemProtCTFFind',
                                      doRaise=True)
SparxGaussianProtPicking = Domain.importFromPlugin('eman2.protocols',
                                                   'SparxGaussianProtPicking',
                                                   doRaise=True)


# Number of mics to be processed
NUM_MICS = 5

class TestGpuCorrSemiStreaming(BaseTest):
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

    def importMicrographsStr(self, fnMics):
        kwargs = {'inputMics': fnMics,
                  'nDim': NUM_MICS,
                  'creationInterval': 30,
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
        protPicking = SparxGaussianProtPicking(boxSize=64, lowerThreshold=0.01)
        protPicking.inputMicrographs.set(inputMicrographs)
        self.proj.launchProtocol(protPicking, wait=False)

        return protPicking

    def runExtractParticles(self, inputCoord, setCtfs):
        protExtract = self.newProtocol(XmippProtExtractParticles,
                                       boxSize=64,
                                       doInvert = False,
                                       doFlip = False)

        protExtract.inputCoordinates.set(inputCoord)
        protExtract.ctfRelations.set(setCtfs)

        self.proj.launchProtocol(protExtract, wait=False)

        return protExtract

    def runClassify(self, inputParts, inputAvgs):
        protClassify = self.newProtocol(XmippProtStrGpuCrrSimple,
                                        useAsRef=REF_AVERAGES)

        protClassify.inputParticles.set(inputParts)
        protClassify.inputRefs.set(inputAvgs)
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



    def test_pattern(self):

        protImportAvgs = self.importAverages()
        if protImportAvgs.isFailed():
            self.assertTrue(False)
        protImportMics = self.importMicrographs()
        self.assertFalse(protImportMics.isFailed(), 'ImportMics has failed.')

        protImportMicsStr = self.importMicrographsStr\
            (protImportMics.outputMicrographs)
        counter = 1
        while not protImportMicsStr.hasAttribute('outputMicrographs'):
            time.sleep(2)
            protImportMicsStr = self._updateProtocol(protImportMicsStr)
            if counter > 100:
                break
            counter += 1
        self.assertFalse(protImportMicsStr.isFailed(), 'Create stream data has failed.')
        self.assertTrue(protImportMicsStr.hasAttribute('outputMicrographs'),
                        'Create stream data has no outputMicrographs in more than 3min.')

        protInvContr = self.invertContrast(protImportMicsStr.outputMicrographs)
        counter = 1
        while not protInvContr.hasAttribute('outputMicrographs'):
            time.sleep(2)
            protInvContr = self._updateProtocol(protInvContr)
            if counter > 100:
                break
            counter += 1
        self.assertFalse(protInvContr.isFailed(), 'protInvContr has failed.')
        self.assertTrue(protInvContr.hasAttribute('outputMicrographs'),
                        'protInvContr has no outputMicrographs in more than 3min.')

        protCtf = self.calculateCtf(protInvContr.outputMicrographs)
        counter = 1
        while not protCtf.hasAttribute('outputCTF'):
            time.sleep(2)
            protCtf = self._updateProtocol(protCtf)
            if counter > 100:
                break
            counter += 1
        self.assertFalse(protCtf.isFailed(), 'CTFfind4 has failed.')
        self.assertTrue(protCtf.hasAttribute('outputCTF'),
                        'CTFfind4 has no outputCTF in more than 3min.')

        protPicking = self.runPicking(protInvContr.outputMicrographs)
        counter = 1
        while not protPicking.hasAttribute('outputCoordinates'):
            time.sleep(2)
            protPicking = self._updateProtocol(protPicking)
            if counter > 100:
                break
            counter += 1
        self.assertFalse(protPicking.isFailed(), 'Eman Sparx has failed.')
        self.assertTrue(protPicking.hasAttribute('outputCoordinates'),
                        'Eman Sparx has no outputCoordinates in more than 3min.')

        protExtract = self.runExtractParticles(protPicking.outputCoordinates,
                                               protCtf.outputCTF)
        counter = 1
        while not protExtract.hasAttribute('outputParticles'):
            time.sleep(2)
            protExtract = self._updateProtocol(protExtract)
            if counter > 100:
                break
            counter += 1
        self.assertFalse(protExtract.isFailed(), 'Extract particles has failed.')
        self.assertTrue(protExtract.hasAttribute('outputParticles'),
                        'Extract particles has no outputParticles in more than 3min.')

        protClassify = self.runClassify(protExtract.outputParticles,
                                        protImportAvgs.outputAverages)
        counter = 1
        while protClassify.getStatus()!=STATUS_FINISHED:
            time.sleep(2)
            protClassify = self._updateProtocol(protClassify)
            self.assertFalse(protClassify.isFailed(), 'GL2D-static has failed.')
            if counter > 100:
                self.assertTrue(protClassify.hasAttribute('outputClasses'),
                                'GL2D-static has no outputClasses in more than 3min.')
                self.assertEqual(protClassify.outputClasses.getSize(),
                                 protImportAvgs.outputAverages.getSize(),
                                 'GL2D-static returned a wrong number of classes.')
            counter += 1

        self.assertTrue(protClassify.hasAttribute('outputClasses'),
                        'GL2D-static has no outputClasses at the end')
        self.assertEqual(protClassify.outputClasses.getSize(),
                         protImportAvgs.outputAverages.getSize(),
                         'GL2D-static returned a wrong number of classes at the end.')
