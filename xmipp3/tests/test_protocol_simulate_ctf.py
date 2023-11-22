# ***************************************************************************
# *
# * Authors:     Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es)
# *              David Herreros Calero (dherreros@cnb.csic.es)
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

import os

from pyworkflow.tests import BaseTest, DataSet
import pyworkflow.tests as tests

import pyworkflow.utils as pwutils

from pwem.protocols import ProtImportVolumes
from xmipp3.protocols import XmippProtSimulateCTF, XmippProtCreateGallery

OUTPARTERROR = "There was a problem with particles output"
class TestXmippProtSimulateCTF(BaseTest):
    """ Testing CTF simulation
    """
    @classmethod
    def setUpClass(cls):
        tests.setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('relion_tutorial')
        cls.volume = cls.dataset.getFile(os.path.join('volumes', 'reference_masked.vol'))

    def runImportVolume(self, pattern, sampling):
        """ Run an Import volume protocol. """
        protImport = self.newProtocol(ProtImportVolumes,
                                     filesPath=pattern,
                                     samplingRate=sampling,
                                     objLabel='relion_tutorial volume ' + pwutils.removeBaseExt(pattern))
        self.launchProtocol(protImport)
        if protImport.outputVolume is None:
            raise Exception('Import of volume: %s, failed. outputVolume is None.' % pattern)
        return protImport.outputVolume

    def runCreateGallery(self, volume):
        protCreateGallery = self.newProtocol(XmippProtCreateGallery,
                                             inputVolume=volume,
                                             objLabel='Imported Volume Projections')
        self.launchProtocol(protCreateGallery)
        if protCreateGallery.outputReprojections is None:
            raise Exception('Create Gallery faile: no output produced')
        return protCreateGallery.outputReprojections

    def runSimulateCTF(self, projections, defocus0=5000, defocusF=25000):
        protSimulateCTF = self.newProtocol(XmippProtSimulateCTF,
                                           inputParticles=projections,
                                           Defocus0=defocus0,
                                           DefocusF=defocusF,
                                           objLabel='Projections with CTF - [%.2f, %.2f]' % (defocus0, defocusF))

        self.launchProtocol(protSimulateCTF)
        return protSimulateCTF.outputParticles

    def runSimulateCTFAstig(self, projections, defocus0=5000, defocusF=25000):
        protSimulateCTFAstig = self.newProtocol(XmippProtSimulateCTF,
                                           inputParticles=projections,
                                           Defocus0=defocus0,
                                           DefocusF=defocusF,
                                           astig=True,
                                           objLabel='Projections with CTF Astig - [%.2f, %.2f]' % (defocus0, defocusF))

        self.launchProtocol(protSimulateCTFAstig)
        return protSimulateCTFAstig.outputParticles

    def testSimulateCTF(self):
        importedVolume = self.runImportVolume(self.volume, 1)
        projectionsGallery = self.runCreateGallery(importedVolume)

        # Check default defoci values
        projectionsCTFDefault = self.runSimulateCTF(projectionsGallery)
        self.assertTrue(projectionsCTFDefault,
                        OUTPARTERROR)
        self.assertEqual(projectionsCTFDefault.getSize(),1647,
                        OUTPARTERROR)
        self.assertEqual(projectionsCTFDefault.getXDim(),60,
                        "Unexpected particle size in output particles")
        self.assertEqual(projectionsCTFDefault.getSamplingRate(),1,
                        "Unexpected sampling rate in output particles")
        for projection in projectionsCTFDefault.iterItems():
            defocusU = projection.getCTF().getDefocusU()
            defocusV = projection.getCTF().getDefocusV()
            self.assertEqual(defocusU,defocusV,'DefocusU and DeofcusV are not equal but they should be')
            self.assertGreaterEqual(defocusU,5000, 'DefocusU and DefocusV values are outside the specified range:'
                                    '[%f, %f]' % (5000, 25000))
            self.assertLessEqual(defocusU,25000, 'DefocusU and DefocusV values are outside the specified range:'
                                    '[%f, %f]' % (5000, 25000))


    def testSimulateCTFAstig(self):
        importedVolume = self.runImportVolume(self.volume, 1)
        projectionsGallery = self.runCreateGallery(importedVolume)

        # Check default defoci values
        projectionsCTFDefaultAstig = self.runSimulateCTFAstig(projectionsGallery)
        self.assertTrue(projectionsCTFDefaultAstig,
                        OUTPARTERROR)
        self.assertEqual(projectionsCTFDefaultAstig.getSize(),1647,
                        OUTPARTERROR)
        self.assertEqual(projectionsCTFDefaultAstig.getXDim(),60,
                        "Unexpected particle size in output particles")
        self.assertEqual(projectionsCTFDefaultAstig.getSamplingRate(),1,
                        "Unexpected sampling rate in output particles")
        for projection in projectionsCTFDefaultAstig.iterItems():
            defocusU = projection.getCTF().getDefocusU()
            defocusV = projection.getCTF().getDefocusV()
            angle = projection.getCTF().getDefocusAngle()
            self.assertNotEqual(defocusU,defocusV,'DefocusU and DeofcusV are equal but they should be')
            self.assertGreaterEqual(defocusU,5000, 'DefocusU values are outside the specified range:'
                                    '[%f, %f]' % (5000, 25000))
            self.assertLessEqual(defocusU,25000, 'DefocusU values are outside the specified range:'
                                    '[%f, %f]' % (5000, 25000))
            self.assertGreaterEqual(defocusV,(5000-500), 'DefocusV values are outside the specified range:'
                                    '[%f, %f]' % (5000-500, 25000+500))
            self.assertLessEqual(defocusV,(25000+500), 'DefocusV values are outside the specified range:'
                                    '[%f, %f]' % (5000-500, 25000+500))
            self.assertGreaterEqual(angle,40, 'Defocus angle values are outside the specified range:'
                                    '[%f, %f]' % (40, 50))
            self.assertLessEqual(angle,50, 'Defocus angle values are outside the specified range:'
                                    '[%f, %f]' % (40, 50))
