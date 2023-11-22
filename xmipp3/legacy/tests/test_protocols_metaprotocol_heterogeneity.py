# **************************************************************************
# *
# * Authors:    Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *
# **************************************************************************

'''
from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import ProtImportParticles, ProtImportVolumes, ProtSubSet

from xmipp3.protocols import XmippMetaProtDiscreteHeterogeneityScheduler


class TestMetaprotHeterogeneity(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')

    def importVolume(self):
        prot = self.newProtocol(ProtImportVolumes,
                                objLabel='import volume',
                                filesPath=self.dsRelion.getFile(
                                    'volumes/reference.mrc'),
                                samplingRate=7.08)
        self.launchProtocol(prot)

        return prot

    def importParticles(self, path):
        """ Import an EMX file with Particles and defocus
        """
        prot = self.newProtocol(ProtImportParticles,
                                objLabel='from relion',
                                importFrom=ProtImportParticles.IMPORT_FROM_RELION,
                                starFile=self.dsRelion.getFile(path),
                                magnification=10000,
                                samplingRate=7.08,
                                haveDataBeenPhaseFlipped=True)
        self.launchProtocol(prot)

        return prot

    def test(self):
        protImportVol = self.importVolume()
        path = 'import/case2/relion_it015_data.star'
        protImportParts = self.importParticles(path)

        protHeterogeneity = self.newProtocol(XmippMetaProtDiscreteHeterogeneityScheduler,
                                           inputVolume=protImportVol.outputVolume,
                                           inputParticles=protImportParts.outputParticles,
                                           maxNumClasses=2,
                                           symmetryGroup='d2',
                                           numberOfMpi=8)
        self.launchProtocol(protHeterogeneity)
        self.assertFalse(protHeterogeneity.isFailed(), 'Metaprotocol Heterogeneity has failed.')

'''
