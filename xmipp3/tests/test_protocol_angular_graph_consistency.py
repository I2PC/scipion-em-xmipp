# ***************************************************************************
# * Authors:     Jeison Méndez García (jmendez@utp.edu.co)
# *                based on TestMultireferenceAlignability (Javier Vargas, J.M. De la Rosa Trevin)
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

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import ProtImportParticles, ProtImportVolumes, ProtSubSet

from xmipp3.protocols import XmippProtAngularGraphConsistency


class TestAngularGraphConsistency(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')

    def checkOutput(self, prot, outName, conditions=[]):
        """ Check that an ouput was generated and
        the condition is valid.
        """
        o = getattr(prot, outName, None)
        locals()[outName] = o
        self.assertIsNotNone(o, "Output: %s is None" % outName)
        for c in conditions:
            self.assertTrue(eval(c), 'Condition failed: ' + c)

    def importVolumes(self):
        importProt = self.newProtocol(ProtImportVolumes,
                                objLabel='import volume',
                                filesPath=self.dsRelion.getFile('volumes/reference.mrc'),
                                samplingRate=7.08)
        self.launchProtocol(importProt)

        return importProt

    def importParticles(self, numberOfParticles, path):
        """ Import an EMX file with Particles and defocus
        """
        importProt = self.newProtocol(ProtImportParticles,
                                 objLabel='from relion (auto-refine 3d)',
                                 importFrom=ProtImportParticles.IMPORT_FROM_RELION,
                                 starFile=self.dsRelion.getFile(path),
                                 magnification=10000,
                                 samplingRate=7.08,
                                 haveDataBeenPhaseFlipped=True
                                 )
        self.launchProtocol(importProt)
        self.checkOutput(importProt, 'outputParticles',
                         ['outputParticles.hasAlignmentProj()',
                          'outputParticles.isPhaseFlipped()'])
        # subset to speed-up following processes
        subsetProt = self.newProtocol(ProtSubSet,
                                      objLabel='subset of particles',
                                      chooseAtRandom=True,
                                      nElements=numberOfParticles)
        subsetProt.inputFullSet.set(importProt.outputParticles)

        self.launchProtocol(subsetProt)
        self.checkOutput(subsetProt, 'outputParticles',
                         ['outputParticles.getSize() == %d'
                          % numberOfParticles])

        return subsetProt

    def test_validate(self):
        protImportVols = self.importVolumes()
        pathToFile = 'import/case2/relion_it015_data.star'
        importParticlesProt = self.importParticles(2000,pathToFile)

        validateProt = self.newProtocol(XmippProtAngularGraphConsistency,
                                objLabel='angular graph consistency',
                                angularSampling=3,
                                maximumTargetResolution=10,
                                numberOfMpi=4, numberOfThreads=1)
        validateProt.inputParticles.set(importParticlesProt.outputParticles)
        validateProt.inputVolume.set(protImportVols.outputVolume)

        self.launchProtocol(validateProt)
        self.checkOutput(validateProt, 'outputParticles')       
