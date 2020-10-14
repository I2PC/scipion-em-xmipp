# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
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

from pyworkflow.tests import BaseTest, DataSet, setupTestProject

from pwem.protocols import ProtImportVolumes, ProtImportParticles, ProtSubSet

from xmipp3.protocols import (XmippProtAngularAlignmentSPH,
                              XmippProtVolumeDeformSPH,
                              XmippProtStructureMapSPH)

class TestSPHBase(BaseTest):
    @classmethod
    def setData(cls):
        cls.dataset = DataSet.getDataSet('relion_tutorial')
        cls.particles = cls.dataset.getFile('import/case2/particles.sqlite')
        cls.volParticles = cls.dataset.getFile('volume')
        cls.volInput = cls.dataset.getFile('volumes/reference_rotated.vol')
        cls.volRef = cls.dataset.getFile('volumes/reference.mrc')

    @classmethod
    def runImportVolume(cls, pattern, sampling):
        """ Run an Import volume protocol. """
        protImport = cls.newProtocol(ProtImportVolumes,
                                          filesPath=pattern,
                                          samplingRate=sampling)
        cls.launchProtocol(protImport)
        if protImport.outputVolume is None:
            raise Exception('Import of volume: %s, failed. outputVolume is None.' % pattern)
        return protImport

    @classmethod
    def runImportParticles(cls, pattern, sampling):
        """ Run an Import particles protocol. """
        protImport = cls.newProtocol(ProtImportParticles,
                                 objLabel='Particles from scipion',
                                 importFrom=ProtImportParticles.IMPORT_FROM_SCIPION,
                                 sqliteFile=pattern,
                                 magnification=50000,
                                 samplingRate=sampling,
                                 haveDataBeenPhaseFlipped=True
                                 )
        cls.launchProtocol(protImport)
        if protImport.outputParticles is None:
            raise Exception('Import of particles: %s, failed. outputParticles is None.' % pattern)
        return protImport



class TestAngularAlignmentSPH(TestSPHBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestSPHBase.setData()
        cls.protImportPart = cls.runImportParticles(cls.particles, 7.08)
        cls.protImportVol = cls.runImportVolume(cls.volParticles, 7.08)

    def test(self):
        subset = self.newProtocol(ProtSubSet,
                                  inputFullSet=self.protImportPart.outputParticles,
                                  chooseAtRandom=True,
                                  nElements=10)
        self.launchProtocol(subset)

        alignSph = self.newProtocol(XmippProtAngularAlignmentSPH,
                                   inputParticles=subset.outputParticles,
                                   inputVolume=self.protImportVol.outputVolume,
                                   depth=1,
                                   targetResolution = 4.0,
                                   numberOfMpi=8)
        self.launchProtocol(alignSph)
        self.assertIsNotNone(alignSph.outputParticles,
                             "There was a problem with AngularAlignmentSPH")


class TestVolumeDeformSPH(TestSPHBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestSPHBase.setData()
        cls.protImportVol = cls.runImportVolume(cls.volInput, 1.0)
        cls.protImportVolRef = cls.runImportVolume(cls.volRef, 1.0)

    def test(self):

        volDeformSph = self.newProtocol(XmippProtVolumeDeformSPH,
                                   inputVolume=self.protImportVol.outputVolume,
                                   refVolume = self.protImportVolRef.outputVolume,
                                   depth=2,
                                   targetResolution = 4.0)
        self.launchProtocol(volDeformSph)
        self.assertIsNotNone(volDeformSph.outputVolume,
                             "There was a problem with VolumeDeformSPH")



class TestStructureMapSPH(TestSPHBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestSPHBase.setData()
        cls.protImportVol1 = cls.runImportVolume(cls.volInput, 1.0)
        cls.protImportVol2 = cls.runImportVolume(cls.volRef, 1.0)

    def test(self):

        structureMap = self.newProtocol(XmippProtStructureMapSPH,
                                   depth=2,
                                   targetResolution = 4.0)
        volumeList = [self.protImportVol1.outputVolume, self.protImportVol2.outputVolume]
        structureMap.inputVolumes.set(volumeList)
        self.launchProtocol(structureMap)
        self.assertTrue(structureMap.isFinished(), "There was a problem with StructureMapSPH")