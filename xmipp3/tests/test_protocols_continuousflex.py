# **************************************************************************
# * Authors:     Mohamad Harastani (mohamad.harastani@upmc.fr)
# * IMPMC, Sorbonne University
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
# **************************************************************************
from continuousflex.protocols import FlexProtAlignmentNMAVol
from pwem.protocols import ProtImportPdb
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet
from continuousflex.protocols import (FlexProtNMA, FlexProtSynthesizeSubtomo, FlexProtSynthesizeImages, NMA_CUTOFF_ABS,
                                      FlexProtAlignmentNMA, FlexProtSubtomogramAveraging)
from continuousflex.protocols.protocol_subtomograms_synthesize import MODE_RELATION_3CLUSTERS

files_dictionary = {'pdb': 'pdb/AK.pdb', 'particles': 'particles/img.stk', 'vol': 'volumes/AK_LP10.vol',
                    'precomputed_atomic': 'gold/images_WS_atoms.xmd',
                    'precomputed_pseudoatomic': 'gold/images_WS_pseudoatoms.xmd',
                    'small_stk': 'test_alignment_10images/particles/smallstack_img.stk',
                    'subtomograms':'HEMNMA_3D/subtomograms/*.vol',
                    'precomputed_HEMNMA3D_atoms':'HEMNMA_3D/gold/precomputed_atomic.xmd',
                    'precomputed_HEMNMA3D_pseudo':'HEMNMA_3D/gold/precomputed_pseudo.xmd',

                    'charmm_prm':'genesis/par_all36_prot.prm',
                    'charmm_top':'genesis/top_all36_prot.rtf',
                    '1ake_pdb':'genesis/1ake.pdb',
                    '1ake_vol':'genesis/1ake.mrc',
                    '4ake_pdb':'genesis/4ake.pdb',
                    '4ake_aa_pdb':'genesis/4ake_aa.pdb',
                    '4ake_aa_psf':'genesis/4ake_aa.psf',
                    '4ake_ca_pdb':'genesis/4ake_ca.pdb',
                    '4ake_ca_top':'genesis/4ake_ca.top',
                    }
DataSet(name='nma_V2.0', folder='nma_V2.0', files=files_dictionary,
        url='https://raw.githubusercontent.com/continuousflex-org/testdata-continuousflex/main')



class TestContinuousFlexBasics(TestWorkflow):
    """ Test protocol for HEMNMA, HEMNMA-3D, StructMap, TomoFlow and STA backend functions. """
    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')

    def test_all(self):
        # ------------------------------------------------
        # Import a Pdb -> NMA
        # ------------------------------------------------
        protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('pdb'))
        protImportPdb.setObjLabel('AK.pdb')
        self.launchProtocol(protImportPdb)
        # Launch NMA for PDB imported
        protNMA = self.newProtocol(FlexProtNMA,
                                   cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protImportPdb.outputPdb)
        protNMA.setObjLabel('NMA')
        self.launchProtocol(protNMA)
        # ------------------------------------------------
        # Create few volumes for testing
        # ------------------------------------------------
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                          modeList='7-8',
                                          numberOfVolumes=2,
                                          modeRelationChoice=MODE_RELATION_3CLUSTERS,
                                          noiseCTFChoice=1, # not adding noise
                                          volumeSize=16)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('test volumes')
        self.launchProtocol(protSynthesize)
        # ------------------------------------------------
        # Lunch HEMNMA-3D (this will test xmipp_nma_alignment_vol, also useful for StructMap)
        # ------------------------------------------------
        protAlignment = self.newProtocol(FlexProtAlignmentNMAVol,
                                         modeList='7-8')
        protAlignment.inputModes.set(protNMA.outputModes)
        protAlignment.inputVolumes.set(protSynthesize.outputVolumes)
        protAlignment.setObjLabel('HEMNMA-3D & StructMap backend')
        self.launchProtocol(protAlignment)
        # ------------------------------------------------
        # Lunch STA on the synthesized subtomograms (this will test xmipp_volumeset_align, also useful for TomoFlow)
        # ------------------------------------------------
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                   NumOfIters=1)
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.setObjLabel('STA and TomoFlow backend')
        self.launchProtocol(protStA)
        # ------------------------------------------------
        # Create few images for testing
        # ------------------------------------------------
        protSynthesize = self.newProtocol(FlexProtSynthesizeImages,
                                          modeList='7-8',
                                          numberOfVolumes=2,
                                          modeRelationChoice=MODE_RELATION_3CLUSTERS,
                                          noiseCTFChoice=1, # not adding noise
                                          volumeSize=32)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('test images')
        self.launchProtocol(protSynthesize)
        # ------------------------------------------------
        # Lunch HEMNMA on the synthesized images (this will test xmipp_nma_alignment)
        # ------------------------------------------------
        protAlignment = self.newProtocol(FlexProtAlignmentNMA,
                                         modeList='7-8')
        protAlignment.inputModes.set(protNMA.outputModes)
        protAlignment.inputParticles.set(protSynthesize.outputImages)
        protAlignment.setObjLabel('HEMNMA backend')
        self.launchProtocol(protAlignment)