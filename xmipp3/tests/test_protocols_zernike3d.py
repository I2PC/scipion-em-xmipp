# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
# *             David Herreros Calero (dherreros@cnb.csic.es)
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


# import os
# import numpy as np
#
# from pyworkflow.tests import BaseTest, DataSet, setupTestProject
#
# from pwem.protocols import ProtImportVolumes, ProtImportParticles, ProtSubSet
# import pwem.emlib.metadata as md
#
# from xmipp3.protocols import (XmippProtAngularAlignmentZernike3D,
#                               XmippProtVolumeDeformZernike3D,
#                               XmippProtStructureMapZernike3D,
#                               XmippProtStructureMap)
#
#
# class TestZernike3DBase(BaseTest):
#     @classmethod
#     def setData(cls):
#         cls.dataset = DataSet(name='test_zernike3d',
#                               folder='test_zernike3d',
#                               files={'particles': 'particles/images_1720_norm.xmd',
#                                      'volumes': 'volumes/*.vol',
#                                      '1720': 'volumes/EMD-1720_norm.vol',
#                                      '1723': 'volumes/EMD-1723_norm.vol'})
#         cls.dataset = DataSet.getDataSet('test_zernike3d')
#         cls.particles = cls.dataset.getFile('particles')
#         cls.volumes = cls.dataset.getFile('volumes')
#         cls.volume_1720 = cls.dataset.getFile('1720')
#         cls.volume_1723 = cls.dataset.getFile('1723')
#         cls.clnm_pd_cpu_gold = cls.dataset.getFile('gold_standard_pd/CPU/Volumes_clnm.txt')
#         cls.clnm_pd_gpu_gold = cls.dataset.getFile('gold_standard_pd/GPU/Volumes_clnm.txt')
#
#     @classmethod
#     def runImportVolumes(cls, pattern):
#         """ Run an Import volumes protocol. """
#         protImport = cls.newProtocol(ProtImportVolumes,
#                                      filesPath=pattern,
#                                      samplingRate=1.0,
#                                      objLabel="Maps - Ribosomes 1718-1724")
#         cls.launchProtocol(protImport)
#         cls.assertIsNotNone(protImport.outputVolumes,
#                             'Import of volumes: %s, failed. outputVolumes is None.' % pattern)
#         return protImport
#
#     @classmethod
#     def runImportVolume(cls, pattern, label):
#         protImport = cls.newProtocol(ProtImportVolumes,
#                                      filesPath=pattern,
#                                      samplingRate=1.0,
#                                      objLabel="Map - Ribosome %s" % label)
#         cls.launchProtocol(protImport)
#         cls.assertIsNotNone(protImport.outputVolume,
#                             'Import of volume: %s, failed. outputVolumes is None.' % pattern)
#         return protImport
#
#     @classmethod
#     def runImportParticles(cls, pattern):
#         """ Run an Import particles protocol. """
#         print(pattern)
#         protImport = cls.newProtocol(ProtImportParticles,
#                                      objLabel='Particles - Ribosome 1720',
#                                      importFrom=ProtImportParticles.IMPORT_FROM_XMIPP3,
#                                      mdFile=pattern,
#                                      magnification=50000,
#                                      samplingRate=1.0,
#                                      haveDataBeenPhaseFlipped=True
#                                      )
#         cls.launchProtocol(protImport)
#         cls.assertIsNotNone(protImport.outputParticles,
#                             'Import of particles: %s, failed. outputParticles is None.' % pattern)
#         return protImport
#
#     @classmethod
#     def readZernikeParams(cls, filename):
#         with open(filename, 'r') as fid:
#             lines = fid.readlines()
#         basis_params = np.fromstring(lines[0].strip('\n'), sep=' ')
#         coeffs = np.fromstring(lines[1].strip('\n'), sep=' ')
#         size = int(coeffs.size / 3)
#         coeffs = np.asarray([coeffs[:size], coeffs[size:2 * size], coeffs[2 * size:]])
#         return [int(basis_params[0]), int(basis_params[1]), basis_params[2], coeffs]
#
#     @classmethod
#     def readMetadata(cls, file):
#         mdOut = md.MetaData(file)
#         return np.vstack(mdOut.getColumnValues(md.MDL_SPH_COEFFICIENTS))[:, :-8]
#
#
# class TestProtocolsZernike3D(TestZernike3DBase):
#     @classmethod
#     def setUpClass(cls):
#         setupTestProject(cls)
#         TestZernike3DBase.setData()
#         cls.protImportPart = cls.runImportParticles(cls.particles)
#         cls.protImportVols = cls.runImportVolumes(cls.volumes)
#         cls.protImportVol_1720 = cls.runImportVolume(cls.volume_1720, '1720')
#         cls.protImportVol_1723 = cls.runImportVolume(cls.volume_1723, '1723')
#         cls.CORR_FILE = 'CoordinateMatrixCorr3.txt'
#         cls.EXPECTED_CORR_MAP = 'Unexpected correlation structure mapping'
#
#     def testPairAlignmentZernike3DCPU(self):
#         # Test for CPU implementation
#         alignZernike = self.newProtocol(XmippProtVolumeDeformZernike3D,
#                                         inputVolume=self.protImportVol_1720.outputVolume,
#                                         refVolume=self.protImportVol_1723.outputVolume,
#                                         l1=3,
#                                         l2=2,
#                                         targetResolution=6,
#                                         useGpu=False,
#                                         numberOfThreads=4,
#                                         objLabel="Pair Alignment Zernike3D - CPU")
#         self.launchProtocol(alignZernike)
#         self.assertIsNotNone(alignZernike.outputVolume,
#                              "There was a problem with XmippProtVolumeDeformZernike3D - CPU Version")
#         # Check coefficients and parameters are the ones expected
#         clnm_gold = self.readZernikeParams(self.clnm_pd_cpu_gold)
#         clnm_test = self.readZernikeParams(alignZernike._getExtraPath('Volumes_clnm.txt'))
#         self.assertEqual(clnm_gold[0], clnm_test[0], "There is a problem with Zernike degree")
#         self.assertEqual(clnm_gold[1], clnm_test[1], "There is a problem with Spherical Harmonics degree")
#         self.assertEqual(clnm_gold[2], clnm_test[2], "There is a problem with sphere radius")
#         diff_clnm = np.sqrt(np.sum((clnm_gold[3] - clnm_test[3]) ** 2) / clnm_gold[3].size)
#         self.assertAlmostEqual(diff_clnm, 0, delta=0.5,
#                                msg="Zernike coefficients do not have the expected value")
#
#     def testPairAlignmentZernike3DGPU(self):
#         # Test for GPU implementation
#         alignZernike = self.newProtocol(XmippProtVolumeDeformZernike3D,
#                                         inputVolume=self.protImportVol_1720.outputVolume,
#                                         refVolume=self.protImportVol_1723.outputVolume,
#                                         l1=3,
#                                         l2=2,
#                                         targetResolution=3,
#                                         useGpu=True,
#                                         numberOfThreads=1,
#                                         objLabel="Pair Alignment Zernike3D - GPU")
#         self.launchProtocol(alignZernike)
#         self.assertIsNotNone(alignZernike.outputVolume,
#                              "There was a problem with XmippProtVolumeDeformZernike3D - GPU Version")
#         # Check coefficients and parameters are the ones expected
#         clnm_gold = self.readZernikeParams(self.clnm_pd_gpu_gold)
#         clnm_test = self.readZernikeParams(alignZernike._getExtraPath('Volumes_clnm.txt'))
#         self.assertEqual(clnm_gold[0], clnm_test[0], "There is a problem with Zernike degree")
#         self.assertEqual(clnm_gold[1], clnm_test[1], "There is a problem with Spherical Harmonics degree")
#         self.assertEqual(clnm_gold[2], clnm_test[2], "There is a problem with sphere radius")
#         diff_clnm = np.sqrt(np.sum((clnm_gold[3] - clnm_test[3]) ** 2) / clnm_gold[3].size)
#         self.assertAlmostEqual(diff_clnm, 0, delta=0.5,
#                                msg="Zernike coefficients do not have the expected value")
#
#     def testStructMapZernike3DCPU(self):
#         # Test for CPU implementation
#         structureMap = self.newProtocol(XmippProtStructureMapZernike3D,
#                                         inputVolumes=[self.protImportVols.outputVolumes],
#                                         l1=3,
#                                         l2=2,
#                                         targetResolution=12.0,
#                                         useGpu=False,
#                                         numberOfThreads=4,
#                                         objLabel="StructMap Zernike3D - CPU")
#         self.launchProtocol(structureMap)
#         self.assertTrue(structureMap.isFinished(), "There was a problem with StructureMapZernike3D - CPU Version")
#         # Check if structure mappings (3D) are fine
#         sm_gold_def = np.loadtxt(self.dataset.getFile('gold_standard_sm/CPU/CoordinateMatrix3.txt'),
#                                      delimiter=' ')
#         sm_test_def = np.loadtxt(structureMap._getExtraPath('CoordinateMatrix3.txt'),
#                                      delimiter=' ')
#         diff_sm_def = np.sqrt(np.sum((sm_gold_def - sm_test_def) ** 2) / sm_gold_def.size)
#         self.assertAlmostEqual(diff_sm_def, 0, delta=0.5,
#                                msg="Unexpected deformation structure mapping")
#         sm_cpu_gold_corr = np.loadtxt(self.dataset.getFile('gold_standard_sm/CPU/CoordinateMatrixCorr3.txt'),
#                                      delimiter=' ')
#         sm_cpu_test_corr = np.loadtxt(structureMap._getExtraPath(self.CORR_FILE),
#                                      delimiter=' ')
#         diff_sm_corr = np.sqrt(np.sum((sm_cpu_gold_corr - sm_cpu_test_corr) ** 2) / sm_cpu_gold_corr.size)
#         self.assertAlmostEqual(diff_sm_corr, 0, delta=0.5,
#                                msg=self.EXPECTED_CORR_MAP)
#         sm_cpu_gold_cons = np.loadtxt(self.dataset.getFile('gold_standard_sm/CPU/ConsensusMatrix3.txt'),
#                                      delimiter=' ')
#         sm_cpu_test_cons = np.loadtxt(structureMap._getExtraPath('ConsensusMatrix3.txt'),
#                                      delimiter=' ')
#         diff_sm_cons = np.sqrt(np.sum((sm_cpu_gold_cons - sm_cpu_test_cons) ** 2) / sm_cpu_gold_cons.size)
#         self.assertAlmostEqual(diff_sm_cons, 0, delta=0.5,
#                                msg="Unexpected consensus structure mapping")
#
#     def testStructMapCPU(self):
#         # Test for CPU implementation
#         structureMap = self.newProtocol(XmippProtStructureMap,
#                                         inputVolumes=self.protImportVols.outputVolumes,
#                                         targetResolution=12.0,
#                                         numberOfThreads=4,
#                                         objLabel="StructMap - CPU")
#         self.launchProtocol(structureMap)
#         self.assertTrue(structureMap.isFinished(), "There was a problem with StructureMap - CPU Version")
#         # Check if structure mappings (3D) are fine
#         sm_cpu_gold_corr = np.loadtxt(self.dataset.getFile('gold_standard_sm/No_Zernike/CoordinateMatrixCorr3.txt'),
#                                      delimiter=' ')
#         sm_cpu_test_corr = np.loadtxt(structureMap._getExtraPath(self.CORR_FILE),
#                                      delimiter=' ')
#         diff_sm_corr = np.sqrt(np.sum((sm_cpu_gold_corr - sm_cpu_test_corr) ** 2) / sm_cpu_gold_corr.size)
#         self.assertAlmostEqual(diff_sm_corr, 0, delta=0.5,
#                                msg=self.EXPECTED_CORR_MAP)
#
#
#     def testStructMapZernike3DGPU(self):
#         # Test for GPU implementation
#         structureMap = self.newProtocol(XmippProtStructureMapZernike3D,
#                                         inputVolumes=[self.protImportVols.outputVolumes],
#                                         l1=3,
#                                         l2=2,
#                                         targetResolution=3.0,
#                                         useGpu=True,
#                                         numberOfThreads=1,
#                                         objLabel="StructMap Zernike3D - GPU")
#         self.launchProtocol(structureMap)
#         self.assertTrue(structureMap.isFinished(), "There was a problem with StructureMapZernike3D - GPU Version")
#         # Check if structure mappings (3D) are fine
#         sm_gold_def = np.loadtxt(self.dataset.getFile('gold_standard_sm/GPU/CoordinateMatrix3.txt'),
#                                      delimiter=' ')
#         sm_test_def = np.loadtxt(structureMap._getExtraPath('CoordinateMatrix3.txt'),
#                                      delimiter=' ')
#         diff_sm_def = np.sqrt(np.sum((sm_gold_def - sm_test_def) ** 2) / sm_gold_def.size)
#         self.assertAlmostEqual(diff_sm_def, 0, delta=0.5,
#                                msg="Unexpected deformation structure mapping")
#         sm_cpu_gold_corr = np.loadtxt(self.dataset.getFile('gold_standard_sm/GPU/CoordinateMatrixCorr3.txt'),
#                                      delimiter=' ')
#         sm_cpu_test_corr = np.loadtxt(structureMap._getExtraPath(self.CORR_FILE),
#                                      delimiter=' ')
#         diff_sm_corr = np.sqrt(np.sum((sm_cpu_gold_corr - sm_cpu_test_corr) ** 2) / sm_cpu_gold_corr.size)
#         self.assertAlmostEqual(diff_sm_corr, 0, delta=0.5,
#                                msg=self.EXPECTED_CORR_MAP)
#         sm_cpu_gold_cons = np.loadtxt(self.dataset.getFile('gold_standard_sm/GPU/ConsensusMatrix3.txt'),
#                                      delimiter=' ')
#         sm_cpu_test_cons = np.loadtxt(structureMap._getExtraPath('ConsensusMatrix3.txt'),
#                                      delimiter=' ')
#         diff_sm_cons = np.sqrt(np.sum((sm_cpu_gold_cons - sm_cpu_test_cons) ** 2) / sm_cpu_gold_cons.size)
#         self.assertAlmostEqual(diff_sm_cons, 0, delta=0.5,
#                                msg="Unexpected consensus structure mapping")
#
#     def testAngularAlignZernike3DCPU(self):
#         # Test for CPU implementation
#         alignZernike = self.newProtocol(XmippProtAngularAlignmentZernike3D,
#                                         inputParticles=self.protImportPart.outputParticles,
#                                         inputVolume=self.protImportVol_1723.outputVolume,
#                                         l1=3,
#                                         l2=2,
#                                         targetResolution=12,
#                                         useGpu=False,
#                                         numberOfMpi=4,
#                                         objLabel="Angular Align Zernike3D - CPU")
#         self.launchProtocol(alignZernike)
#         self.assertIsNotNone(alignZernike.outputParticles,
#                              "There was a problem with AngularAlignmentZernike3D - CPU Version")
#         # Check if coefficients are fine
#         aa_gold_clnm = self.readMetadata(self.dataset.getFile('gold_standard_aa/CPU/output_particles.xmd'))
#         aa_test_clnm = self.readMetadata(alignZernike._getExtraPath('output_particles.xmd'))
#         diff_clnm = np.sqrt(np.sum((aa_gold_clnm - aa_test_clnm) ** 2) / aa_gold_clnm.size)
#         self.assertAlmostEqual(diff_clnm, 0, delta=0.5,
#                                msg="Unexpected Zernike coefficients")
#
#     def testAngularAlignZernike3DGPU(self):
#         # Test for GPU implementation
#         alignZernike = self.newProtocol(XmippProtAngularAlignmentZernike3D,
#                                         inputParticles=self.protImportPart.outputParticles,
#                                         inputVolume=self.protImportVol_1723.outputVolume,
#                                         l1=3,
#                                         l2=2,
#                                         targetResolution=3,
#                                         useGpu=True,
#                                         numberOfMpi=1,
#                                         objLabel="Angular Align Zernike3D - GPU")
#         self.launchProtocol(alignZernike)
#         self.assertIsNotNone(alignZernike.outputParticles,
#                              "There was a problem with AngularAlignmentZernike3D - GPU Version")
#         # Check if coefficients are fine
#         aa_gold_clnm = self.readMetadata(self.dataset.getFile('gold_standard_aa/GPU/output_particles.xmd'))
#         aa_test_clnm = self.readMetadata(alignZernike._getExtraPath('output_particles.xmd'))
#         diff_clnm = np.sqrt(np.sum((aa_gold_clnm - aa_test_clnm) ** 2) / aa_gold_clnm.size)
#         self.assertAlmostEqual(diff_clnm, 0, delta=0.5,
#                                msg="Unexpected Zernike coefficients")