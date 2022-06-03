# ***************************************************************************
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *
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


from pyworkflow.tests import BaseTest, setupTestProject

from pwem import emlib

from xmipp3.protocols import XmippProtReconstructFourier, \
                             XmippProtCreateMask3D, XmippProtCreateGallery


class TestReconstructFourier(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def checkOutput(self, prot, outName, conditions=[]):
        """ Check that an ouput was generated and
        the condition is valid.
        """
        o = getattr(prot, outName, None)
        locals()[outName] = o
        self.assertIsNotNone(o, "Output: %s is None" % outName)
        for c in conditions:
            self.assertTrue(eval(c), 'Condition failed: ' + c)

    def createPhantomCube(self):
        spherePhantom = self.newProtocol(XmippProtCreateMask3D,
                                         objLabel='Target volume',
                                         source=1,
                                         geo=1,
                                         size=128)
        self.launchProtocol(spherePhantom)
        self.assertIsNotNone(spherePhantom.outputMask)
        return spherePhantom

    def createProjections(self, volume):
        projections = self.newProtocol(XmippProtCreateGallery,
                                       objLabel='Projections',
                                       inputVolume=volume,
                                       rotStep=10,
                                       tiltStep=10)
        self.launchProtocol(projections)
        self.assertIsNotNone(projections.outputReprojections)
        return projections

    def test_reconstruct_fourier(self):
        cubePhantom = self.createPhantomCube()
        projections = self.createProjections(cubePhantom.outputMask)

        validateProt = self.newProtocol(XmippProtReconstructFourier,
                                objLabel='Fourier reconstruction',
                                inputParticles=projections.outputReprojections)

        self.launchProtocol(validateProt)
        self.checkOutput(validateProt, 'outputVolume')

        # Check if cubes are similar
        sphere_phantom = emlib.Image(cubePhantom.outputMask.getFileName())
        sphere_reconstructed = emlib.Image(validateProt.outputVolume.getFileName())
        corr = sphere_phantom.correlation(sphere_reconstructed)
        self.assertAlmostEqual(corr, 0.98, delta=0.01, msg="There was an error during the reconstruction process")
