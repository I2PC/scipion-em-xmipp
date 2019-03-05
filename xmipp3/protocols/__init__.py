# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *              David Maluenda Niubo (dmaluenda@cnb.csic.es) [2]
# *
# * [1] SciLifeLab, Stockholm University
# * [2] Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

from .nma import *
from .pdb import *
from .protocol_preprocess import *
from .protocol_3dbionotes import XmippProt3DBionotes
from .protocol_assignment_tilt_pair import XmippProtAssignmentTiltPair
from .protocol_align_volume import XmippProtAlignVolume, XmippProtAlignVolumeForWeb
from .protocol_preprocess.protocol_add_noise import (XmippProtAddNoiseVolumes,
                                                     XmippProtAddNoiseParticles)
from .protocol_apply_alignment import XmippProtApplyAlignment
from .protocol_apply_transformation_matrix import XmippProtApplyTransformationMatrix
from .protocol_break_symmetry import XmippProtAngBreakSymmetry
from .protocol_cl2d_align import XmippProtCL2DAlign
from .protocol_cl2d import XmippProtCL2D
from .protocol_classify_kmeans2d import XmippProtKmeansClassif2D
from .protocol_cltomo import XmippProtCLTomo
from .protocol_classification_gpuCorr import XmippProtGpuCrrCL2D
from .protocol_classification_gpuCorr_semi import XmippProtStrGpuCrrSimple
from .protocol_classification_gpuCorr_full import XmippProtStrGpuCrrCL2D
from .protocol_ctf_defocus_group import XmippProtCTFDefocusGroup
from .protocol_compare_reprojections import XmippProtCompareReprojections
from .protocol_compare_angles import XmippProtCompareAngles
from .protocol_create_gallery import XmippProtCreateGallery
from .protocol_ctf_consensus import XmippProtCTFConsensus
from .protocol_ctf_micrographs import XmippProtCTFMicrographs
from .protocol_ctf_correct_wiener2d import XmippProtCTFCorrectWiener2D
from .protocol_consensus_classes3D import XmippProtConsensusClasses3D
from .protocol_subtract_projection import XmippProtSubtractProjection
from .protocol_denoise_particles import XmippProtDenoiseParticles
from .protocol_eliminate_empty_images import XmippProtEliminateEmptyParticles, \
                                             XmippProtEliminateEmptyClasses
from .protocol_extract_particles import XmippProtExtractParticles
from .protocol_extract_particles_movies import XmippProtExtractMovieParticles
from .protocol_extract_particles_pairs import XmippProtExtractParticlesPairs
from .protocol_extract_unit_cell import XmippProtExtractUnit
from .protocol_helical_parameters import XmippProtHelicalParameters
from .protocol_kerdensom import XmippProtKerdensom
from .protocol_ml2d import XmippProtML2D
from .protocol_movie_gain import XmippProtMovieGain
from .protocol_mltomo import XmippProtMLTomo
from .protocol_movie_average import XmippProtMovieAverage
from .protocol_movie_correlation import XmippProtMovieCorr
from .protocol_movie_opticalflow import XmippProtOFAlignment, ProtMovieAlignment
from .protocol_movie_max_shift import XmippProtMovieMaxShift
from .protocol_multiple_fscs import XmippProtMultipleFSCs
from .protocol_multireference_alignability import XmippProtMultiRefAlignability
from .protocol_normalize_strain import XmippProtNormalizeStrain
from .protocol_particle_pick_automatic import XmippParticlePickingAutomatic
from .protocol_particle_pick_consensus import XmippProtConsensusPicking
from .protocol_pick_noise import XmippProtPickNoise
from .protocol_particle_boxsize import XmippProtParticleBoxsize
from .protocol_particle_pick import XmippProtParticlePicking
from .protocol_particle_pick_pairs import XmippProtParticlePickingPairs
from .protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from .protocol_projmatch import XmippProtProjMatch
from .protocol_random_conical_tilt import XmippProtRCT
from .protocol_ransac import XmippProtRansac
from .protocol_center_particles import XmippProtCenterParticles
from .protocol_reconstruct_fourier import XmippProtReconstructFourier
from .protocol_reconstruct_highres import XmippProtReconstructHighRes
from .protocol_reconstruct_significant import XmippProtReconstructSignificant
from .protocol_reconstruct_swarm import XmippProtReconstructSwarm
from .protocol_resolution3d import XmippProtResolution3D
from .protocol_resolution_directional import XmippProtMonoDir
from .protocol_resolution_monogenic_signal import XmippProtMonoRes
from .protocol_volume_local_sharpening import XmippProtLocSharp
from .protocol_resolution_monotomo import XmippProtMonoTomo
from .protocol_rotational_spectra import XmippProtRotSpectra
from .protocol_rotational_symmetry import XmippProtRotationalSymmetry
from .protocol_screen_particles import XmippProtScreenParticles
from .protocol_screen_deepConsensus import XmippProtScreenDeepConsensus, XmippProtDeepConsSubSet
from .protocol_screen_deeplearning import XmippProtScreenDeepLearning
from .protocol_solid_angles import XmippProtSolidAngles
from .protocol_split_volume import XmippProtSplitvolume
from .protocol_trigger_data import XmippProtTriggerData
from .protocol_validate_nontilt import XmippProtValidateNonTilt
from .protocol_validate_overfitting import XmippProtValidateOverfitting
from .protocol_volume_strain import XmippProtVolumeStrain
from .protocol_enrich import XmippProtEnrich
from .protocol_write_testC import XmippProtWriteTestC
from .protocol_write_testP import XmippProtWriteTestP
from .protocol_deep_denoising import XmippProtDeepDenoising
from .protocol_generate_reprojections import XmippProtGenerateReprojections
from .protocol_split_volume_hierarchical_cluster import XmippProtSplitVolumeHierarchical
from .protocol_reconstruct_heterogeneous import XmippProtReconstructHeterogeneous
from .protocol_metaprotocol_create_subset import XmippMetaProtCreateSubset
from .protocol_metaprotocol_create_output import XmippMetaProtCreateOutput
from .protocol_metaprotocol_discrete_heterogeneity_scheduler import XmippMetaProtDiscreteHeterogeneityScheduler # keep this after XmippMetaProtCreateSubset and XmippMetaProtCreateOutput

from .protocol_align_volume_and_particles import XmippProtAlignVolumeParticles
