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

from xmipp3.protocols import (XmippProtAngularAlignmentZernike3D,
                              XmippProtVolumeDeformZernike3D,
                              XmippProtStructureMapZernike3D)

class TestZernike3DBase(BaseTest):
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



class TestAngularAlignmentZernike3D(TestZernike3DBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestZernike3DBase.setData()
        cls.protImportPart = cls.runImportParticles(cls.particles, 7.08)
        cls.protImportVol = cls.runImportVolume(cls.volParticles, 7.08)

    def test(self):
        subset = self.newProtocol(ProtSubSet,
                                  inputFullSet=self.protImportPart.outputParticles,
                                  chooseAtRandom=True,
                                  nElements=10)
        self.launchProtocol(subset)

        # Test for CPU implementation
        alignZernike = self.newProtocol(XmippProtAngularAlignmentZernike3D,
                                        inputParticles=subset.outputParticles,
                                        inputVolume=self.protImportVol.outputVolume,
                                        l1=1,
                                        l2=1,
                                        targetResolution=7.08*4,
                                        useGpu=False,
                                        numberOfMpi=4,
                                        objLabel="Angular Align Zernike3D - CPU")
        self.launchProtocol(alignZernike)
        self.assertIsNotNone(alignZernike.outputParticles,
                             "There was a problem with AngularAlignmentZernike3D - CPU Version")

        # Test for GPU implementation
        alignZernike = self.newProtocol(XmippProtAngularAlignmentZernike3D,
                                        inputParticles=subset.outputParticles,
                                        inputVolume=self.protImportVol.outputVolume,
                                        l1=1,
                                        l2=1,
                                        targetResolution=7.08*4,
                                        useGpu=True,
                                        numberOfMpi=1,
                                        objLabel="Angular Align Zernike3D - GPU")
        self.launchProtocol(alignZernike)
        self.assertIsNotNone(alignZernike.outputParticles,
                             "There was a problem with AngularAlignmentZernike3D - GPU Version")


class TestVolumeDeformZernike3D(TestZernike3DBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestZernike3DBase.setData()
        cls.protImportVol = cls.runImportVolume(cls.volInput, 1.0)
        cls.protImportVolRef = cls.runImportVolume(cls.volRef, 1.0)

    def test(self):
        # Test for CPU implementation
        volDeformZernike = self.newProtocol(XmippProtVolumeDeformZernike3D,
                                            inputVolume=self.protImportVol.outputVolume,
                                            refVolume=self.protImportVolRef.outputVolume,
                                            l1=2,
                                            l2=2,
                                            targetResolution=4.0,
                                            useGpu=False,
                                            numberOfThreads=4,
                                            objLabel="Volume Deform Zernike3D - CPU")
        self.launchProtocol(volDeformZernike)
        self.assertIsNotNone(volDeformZernike.outputVolume,
                             "There was a problem with VolumeDeformZernike3D - CPU Version")

        # Test for GPU implementation
        volDeformZernike = self.newProtocol(XmippProtVolumeDeformZernike3D,
                                            inputVolume=self.protImportVol.outputVolume,
                                            refVolume=self.protImportVolRef.outputVolume,
                                            l1=2,
                                            l2=2,
                                            targetResolution=4.0,
                                            useGpu=True,
                                            numberOfThreads=1,
                                            objLabel="Volume Deform Zernike3D - GPU")
        self.launchProtocol(volDeformZernike)
        self.assertIsNotNone(volDeformZernike.outputVolume,
                             "There was a problem with VolumeDeformZernike3D - GPU Version")



class TestStructureMapZernike3D(TestZernike3DBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestZernike3DBase.setData()
        cls.protImportVol1 = cls.runImportVolume(cls.volInput, 1.0)
        cls.protImportVol2 = cls.runImportVolume(cls.volRef, 1.0)

    def test(self):
        # Test for CPU implementation
        structureMap = self.newProtocol(XmippProtStructureMapZernike3D,
                                        l1=2,
                                        l2=2,
                                        targetResolution=4.0,
                                        useGpu=False,
                                        numberOfThreads=4,
                                        objLabel="StructMap Zernike3D - CPU")
        volumeList = [self.protImportVol1.outputVolume, self.protImportVol2.outputVolume]
        structureMap.inputVolumes.set(volumeList)
        self.launchProtocol(structureMap)
        self.assertTrue(structureMap.isFinished(), "There was a problem with StructureMapZernike3D - CPU Version")

        # Test for GPU implementation
        structureMap = self.newProtocol(XmippProtStructureMapZernike3D,
                                        l1=2,
                                        l2=2,
                                        targetResolution=4.0,
                                        useGpu=True,
                                        numberOfThreads=1,
                                        objLabel="StructMap Zernike3D - GPU")
        volumeList = [self.protImportVol1.outputVolume, self.protImportVol2.outputVolume]
        structureMap.inputVolumes.set(volumeList)
        self.launchProtocol(structureMap)
        self.assertTrue(structureMap.isFinished(), "There was a problem with StructureMapZernike3D - GPU Version")