# ***************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

from pyworkflow.tests import BaseTest, DataSet
import pyworkflow.tests as tests

from xmipp3.protocols import XmippProtPhantom, XmippProtCreateGallery, XmippProtSimulateCTF, \
    XmippProtLocalCTF, XmippProtConsensusLocalCTF, XmippProtAnalyzeLocalCTF
from pwem.protocols import ProtImportMicrographs, ProtImportParticles, ProtImportVolumes


class TestXmippLocalDefocusEstimation(BaseTest):
    """ Testing local defocus estimation """
    @classmethod
    def setUpClass(cls):
        tests.setupTestProject(cls)
        cls.protCreatePhantom = cls.newProtocol(XmippProtPhantom)
        cls.launchProtocol(cls.protCreatePhantom)
        cls.assertIsNotNone(cls.protCreatePhantom.getFiles(), "There was a problem with phantom creation")
        cls.protCreateGallery = cls.newProtocol(XmippProtCreateGallery,
                                                inputVolume=cls.protCreatePhantom.outputVolume,
                                                rotStep=15.0,
                                                tiltStep=90.0)
        cls.launchProtocol(cls.protCreateGallery)
        cls.assertIsNotNone(cls.protCreateGallery.getFiles(), "There was a problem with create gallery")

    def testXmippLocalDefocus(self):
        protSimulateCTF = self.newProtocol(XmippProtSimulateCTF)
        protSimulateCTF.inputParticles.set(self.protCreateGallery.outputReprojections)
        self.launchProtocol(protSimulateCTF)
        self.assertIsNotNone(protSimulateCTF.outputParticles, "There was a problem with CTF simulation")

        protEstimateLocalDefocus = self.newProtocol(XmippProtLocalCTF,
                                                    inputSet=protSimulateCTF.outputParticles,
                                                    inputVolume=self.protCreatePhantom.outputVolume,
                                                    sameDefocus=True)
        self.launchProtocol(protEstimateLocalDefocus)
        self.assertIsNotNone(protEstimateLocalDefocus.outputParticles.getFiles(),
                             "There was a problem with local CTF estimation")
        self.assertEqual(protEstimateLocalDefocus.outputParticles.getDim(), (40, 40, 181),
                         "There was a problem with size of set of particles")
        self.assertEqual(protEstimateLocalDefocus.outputParticles.getFirstItem().getSamplingRate(), 4,
                         "There was a problem with the sampling rate value of output particles")
        self.assertEqual(protEstimateLocalDefocus.outputParticles.getFirstItem().getCTF().getDefocusU(),
                         protEstimateLocalDefocus.outputParticles.getFirstItem().getCTF().getDefocusV(),
                         "Defocus U and defocus V should be equal if sameDefocus option is set to True")


class TestXmippLocalDefocusEstimationConsensus(TestXmippLocalDefocusEstimation):
    """ Testing local defocus estimation consensus """

    def testXmippLocalDefocusConsensus(self):
        protSimulateCTF2 = self.newProtocol(XmippProtSimulateCTF,
                                            inputParticles=self.protCreateGallery.outputReprojections,
                                            astig=True)
        self.launchProtocol(protSimulateCTF2)
        self.assertIsNotNone(protSimulateCTF2.outputParticles, "There was a problem with second CTF simulation")

        protEstimateLocalDefocus2 = self.newProtocol(XmippProtLocalCTF,
                                                     inputSet=protSimulateCTF2.outputParticles,
                                                     inputVolume=self.protCreatePhantom.outputVolume)
        self.launchProtocol(protEstimateLocalDefocus2)
        self.assertIsNotNone(protEstimateLocalDefocus2.outputParticles.getFiles(),
                             "There was a problem with second CTF estimation")

        protSimulateCTF3 = self.newProtocol(XmippProtSimulateCTF,
                                            inputParticles=self.protCreateGallery.outputReprojections,
                                            astig=True)
        self.launchProtocol(protSimulateCTF3)
        self.assertIsNotNone(protSimulateCTF3.outputParticles, "There was a problem with third CTF simulation")

        protEstimateLocalDefocus3 = self.newProtocol(XmippProtLocalCTF,
                                                     inputSet=protSimulateCTF3.outputParticles,
                                                     inputVolume=self.protCreatePhantom.outputVolume)
        self.launchProtocol(protEstimateLocalDefocus3)
        self.assertIsNotNone(protEstimateLocalDefocus3.outputParticles.getFiles(),
                             "There was a problem with third CTF estimation")

        protConsensusLocalDefocus = self.newProtocol(XmippProtConsensusLocalCTF,
                                                     inputSet=protSimulateCTF2.outputParticles,
                                                     inputSets=[protEstimateLocalDefocus2.outputParticles,
                                                                protEstimateLocalDefocus3.outputParticles])
        self.launchProtocol(protConsensusLocalDefocus)
        self.assertIsNotNone(protConsensusLocalDefocus.outputParticles.getFiles(),
                             "There was a problem with consensus of local defocus estimation")
        self.assertEqual(protConsensusLocalDefocus.outputParticles.getDim(), (40, 40, 181),
                         "There was a problem with size of set of particles after consensus")
        self.assertEqual(protConsensusLocalDefocus.outputParticles.getFirstItem().getSamplingRate(), 4,
                         "There was a problem with the sampling rate value of output consensus particles")


class TestXmippAnalyzeLocalDefocus(BaseTest):
    """ Testing protocol analyze local defocus """

    @classmethod
    def setUpClass(cls):
        tests.setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('xmipp_tutorial')
        cls.micsFn = cls.dataset.getFile('micrographs/BPV_1386.mrc')
        cls.partsFn = cls.dataset.getFile('particles/BPV_particles.sqlite')
        cls.volFn = cls.dataset.getFile('volumes/BPV_scale_filtered_windowed_64.vol')

    def testXmippAnalyzeLocalDefocus(self):
        protImportMics = self.newProtocol(ProtImportMicrographs,
                                          filesPath=self.micsFn,
                                          samplingRate=1.237,
                                          voltage=300)
        self.launchProtocol(protImportMics)
        self.assertIsNotNone(protImportMics.outputMicrographs, "There was a problem with the import of micrographs")

        protImportParticles = self.newProtocol(ProtImportParticles,
                                               importFrom=4,
                                               sqliteFile=self.partsFn,
                                               samplingRate=1.237)
        self.launchProtocol(protImportParticles)
        self.assertIsNotNone(protImportParticles.getFiles(), "There was a problem with the import of particles")

        protAnalyzeLocalDefocus = self.newProtocol(XmippProtAnalyzeLocalCTF,
                                                   inputMics=protImportMics.outputMicrographs,
                                                   inputSet=protImportParticles.outputParticles)
        self.launchProtocol(protAnalyzeLocalDefocus)
        self.assertIsNotNone(protAnalyzeLocalDefocus.outputMicrographs.getFiles(),
                             "There was a problem with analysis of local CTF estimation")
        self.assertEqual(protAnalyzeLocalDefocus.outputMicrographs.getDim(), (9216, 9441, 1),
                         "There was a problem with size of set of particles after analyze local defocus")
        self.assertEqual(protAnalyzeLocalDefocus.outputMicrographs.getFirstItem().getSamplingRate(), 1.237,
                         "There was a problem with the sampling rate value of output analyze local defocus particles")
