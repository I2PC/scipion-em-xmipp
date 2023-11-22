# ***************************************************************************
# *
# * Authors:     Roberto Marabini (roberto@cnb.csic.es)
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
# ***************************************************************************

import time

from pyworkflow.tests import BaseTest, DataSet
import pyworkflow.tests as tests

from pwem.objects import SetOfCTF, CTFModel, Micrograph, SetOfMicrographs, Pointer
from pwem.protocols import (ProtImportMicrographs, ProtImportCTF,
                            ProtCreateStreamData)
from pwem.protocols.protocol_create_stream_data import SET_OF_MICROGRAPHS


from pwem import emlib
from xmipp3.protocols import XmippProtCTFConsensus, XmippProtCTFMicrographs


class TestXmippCTFConsensusBase(BaseTest):
    """ Testing the non consensus part of the protocol (former ctf-selection)
    """
    @classmethod
    def setUpClass(cls):
        tests.setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('xmipp_tutorial')
        cls.micsFn = cls.dataset.getFile('allMics')

    def _getCTFModel(self, defocusU, defocusV, defocusAngle, resol, psdFile):
        ctf = CTFModel()
        ctf.setStandardDefocus(defocusU, defocusV, defocusAngle)
        ctf.setResolution(resol)
        ctf.setPsdFile(psdFile)

        return ctf

    def checkOutputSize(self, ctfConsensusProt):
        inSize = ctfConsensusProt.inputCTF.get().getSize()
        outSize = 0
        outSize += ctfConsensusProt.outputCTF.getSize() \
                       if hasattr(ctfConsensusProt, 'outputCTF') else 0
        outSize += ctfConsensusProt.outputCTFDiscarded.getSize() \
                       if hasattr(ctfConsensusProt, 'outputCTFDiscarded') else 0
        self.assertEqual(inSize, outSize, "Input size must be equal to the sum"
                                          " of the outputs!")

    def testCtfConsensus1(self):
        # create one micrograph set
        fnMicSet = self.proj.getTmpPath("mics.sqlite")
        fnMic = self.proj.getTmpPath("mic.mrc")
        mic = Micrograph()
        mic.setFileName(fnMic)
        micSet = SetOfMicrographs(filename=fnMicSet)

        # create two CTFsets
        fnCTF1 = self.proj.getTmpPath("ctf1.sqlite")
        ctfSet1 = SetOfCTF(filename=fnCTF1)

        # create one fake micrographs image
        projSize = 32
        img = emlib.Image()
        img.setDataType(emlib.DT_FLOAT)
        img.resize(projSize, projSize)
        img.write(fnMic)

        # fill the sets
        for i in range(1, 4):
            mic = Micrograph()
            mic.setFileName(fnMic)
            micSet.append(mic)

            defocusU = 4000+10*i
            defocusV = 4000+i
            defocusAngle = i*10
            resolution = 2
            psdFile = "psd_1%04d" % i
            ctf = self._getCTFModel(defocusU,
                                    defocusV,
                                    defocusAngle,
                                    resolution,
                                    psdFile)
            ctf.setMicrograph(mic)
            ctfSet1.append(ctf)

        ctfSet1.write()
        micSet.write()

        # import micrograph set
        args = {'importFrom': ProtImportMicrographs.IMPORT_FROM_SCIPION,
                'sqliteFile': fnMicSet,
                'amplitudConstrast': 0.1,
                'sphericalAberration': 2.,
                'voltage': 100,
                'samplingRate': 2.1
                }

        protMicImport = self.newProtocol(ProtImportMicrographs, **args)
        protMicImport.setObjLabel('import micrographs from sqlite ')
        self.launchProtocol(protMicImport)

        # import ctfsets
        protCTF1 = \
            self.newProtocol(ProtImportCTF,
                             importFrom=ProtImportCTF.IMPORT_FROM_SCIPION,
                             filesPath=fnCTF1)
        protCTF1.inputMicrographs.set(protMicImport.outputMicrographs)
        protCTF1.setObjLabel('import ctfs from scipion_1 ')
        self.launchProtocol(protCTF1)

        # launch CTF consensus protocol
        protCtfConsensus = self.newProtocol(XmippProtCTFConsensus)
        protCtfConsensus.inputCTF.set(protCTF1.outputCTF)
        protCtfConsensus.setObjLabel('ctf consensus')
        self.launchProtocol(protCtfConsensus)
        self.checkOutputSize(protCtfConsensus)
        ctf0 = protCtfConsensus.outputCTF.getFirstItem()
        resolution = int(ctf0.getResolution())
        defocusU = int(ctf0.getDefocusU())
        self.assertEqual(resolution, 2)
        self.assertEqual(defocusU, 4010)

    def testCtfConsensus2(self):
        # Import a set of micrographs
        protImport = self.newProtocol(ProtImportMicrographs,
                                      filesPath=self.micsFn,
                                      samplingRate=1.237,
                                      voltage=300)
        self.launchProtocol(protImport)
        self.assertIsNotNone(protImport.outputMicrographs,
                             "There was a problem with the import")

        MICS = protImport.outputMicrographs.getSize()

        # create input micrographs
        kwargs = {'nDim': MICS,
                  'creationInterval': 5,
                  'delay': 0,
                  'setof': SET_OF_MICROGRAPHS
                  }
        protStream = self.newProtocol(ProtCreateStreamData, **kwargs)
        protStream.inputMics = Pointer(protImport, extended="outputMicrographs")
        protStream.setObjLabel('create Stream Mic')
        self.proj.launchProtocol(protStream, wait=False)

        self._waitOutput(protStream, 'outputMicrographs')

        protCTF = self.newProtocol(XmippProtCTFMicrographs)
        protCTF.inputMicrographs = Pointer(protStream, extended="outputMicrographs")
        self.proj.launchProtocol(protCTF, wait=False)

        self._waitOutput(protCTF, 'outputCTF')

        # First consensus
        kwargsCons1 = {
            'maxDefocus': 23000,
            'minDefocus': 1000,
            'astigmatism': 1000,
            'resolution': 6
        }
        protCTFSel = self.newProtocol(XmippProtCTFConsensus, **kwargsCons1)
        protCTFSel.inputCTF = Pointer(protCTF, extended="outputCTF")
        self.proj.launchProtocol(protCTFSel, wait=False)

        self._waitOutput(protCTFSel, 'outputCTF')

        # Second consensus
        kwargsCons2 = {
            'maxDefocus': 40000,
            'minDefocus': 1000,
            'astigmatism': 1000,
            'resolution': 7,
        }
        protCTFSel2 = self.newProtocol(XmippProtCTFConsensus, **kwargsCons2)
        protCTFSel2.inputCTF = Pointer(protCTFSel, extended="outputCTF")
        self.proj.launchProtocol(protCTFSel2)

        self._waitOutput(protCTFSel2, 'outputCTF')
        self._waitOutput(protCTFSel2, 'outputMicrographs')

        micSetDiscarded1 = protCTFSel.outputMicrographsDiscarded
        micSet1 = protCTFSel.outputMicrographs
        counter = 1
        while not ((micSetDiscarded1.getSize() + micSet1.getSize()) == MICS):
            time.sleep(2)
            micSetDiscarded1.load()
            micSet1.load()
            if counter > 100:
                self.assertTrue(False, "Timeout to process all CTFs "
                                       "for the first consensus.")
            counter += 1

        ctfSet2 = protCTFSel2.outputCTF
        counter = 1
        while not (ctfSet2.getSize() == micSet1.getSize()):
            time.sleep(2)
            ctfSet2.load()
            if counter > 100:
                self.assertTrue(False, "Timeout to process all CTFs "
                                       "for the second consensus.")
            counter += 1

        for ctf in ctfSet2:
            defocusU = ctf.getDefocusU()
            defocusV = ctf.getDefocusV()
            astigm = defocusU - defocusV
            resol = ctf.getResolution()

            ok = (defocusU < kwargsCons1["minDefocus"] or
                  defocusU > kwargsCons1["maxDefocus"] or
                  defocusV < kwargsCons1["minDefocus"] or
                  defocusV > kwargsCons1["maxDefocus"] or
                  astigm > kwargsCons1["astigmatism"] or
                  resol > kwargsCons1["resolution"])
            self.assertFalse(ok, "A CTF without the correct parameters"
                                 " is included in the output set")
