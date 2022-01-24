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
from continuousflex.protocols.protocol_subtomogrmas_synthesize import MODE_RELATION_3CLUSTERS


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