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

from xmipp3.protocols import XmippProtPhantom, XmippProtCreateGallery, XmippProtSimulateCTF, XmippProtLocalCTF
from pwem.protocols import ProtImportMicrographs


class TestXmippLocalDefocusEstimation(BaseTest):
    """ Testing local defocus estimation
    """
    @classmethod
    def setUpClass(cls):
        tests.setupTestProject(cls)

    def testXmippLocalDefocus(self):
        protCreatePhantom = self.newProtocol(XmippProtPhantom)
        self.launchProtocol(protCreatePhantom)
        self.assertIsNotNone(protCreatePhantom.getFiles(),  "There was a problem with phantom creation")

        protCreateGallery = self.newProtocol(XmippProtCreateGallery,
                                             inputVolume=protCreatePhantom.outputVolume,
                                             rotStep=15.0,
                                             tiltStep=90.0)
        self.launchProtocol(protCreateGallery)
        self.assertIsNotNone(protCreateGallery.getFiles(), "There was a problem with create gallery")

        protSimulateCTF = self.newProtocol(XmippProtSimulateCTF,
                                           inputParticles=protCreateGallery.outputReprojections)
        self.launchProtocol(protSimulateCTF)
        self.assertIsNotNone(protSimulateCTF.outputParticles, "There was a problem with CTF simulation")

        protEstimateLocalDefocus = self.newProtocol(XmippProtLocalCTF,
                                                    inputSet=protSimulateCTF.outputParticles,
                                                    inputVolume=protCreatePhantom.outputVolume)
        self.launchProtocol(protEstimateLocalDefocus)
        self.assertIsNotNone(protEstimateLocalDefocus.outputParticles.getFiles(),
                             "There was a problem with CTF estimation")
        self.assertEqual(protEstimateLocalDefocus.outputParticles.getDim(), (40, 40, 181),
                         "There was a problem with size of set of particles")
        self.assertEqual(protEstimateLocalDefocus.outputParticles.getFirstItem().getSamplingRate(), 4,
                         "There was a problem with the sampling rate value of output particles")


class TestXmippAnalyzeLocalDefocus(BaseTest):
    """ Testing protocol analyze local defocus
    """
    @classmethod
    def setUpClass(cls):
        tests.setupTestOutput(cls)
        cls.dataset = DataSet.getDataSet('xmipp_tutorial')
        cls.micsFn = cls.dataset.getFile('allMics')

    def testXmippAnalyzeLocalDefocus(self):
        protImportMics = self.newProtocol(ProtImportMicrographs,
                                          filesPath=self.micsFn,
                                          samplingRate=1.237,
                                          voltage=300)
        self.launchProtocol(protImportMics)
        self.assertIsNotNone(protImportMics.outputMicrographs, "There was a problem with the import")
