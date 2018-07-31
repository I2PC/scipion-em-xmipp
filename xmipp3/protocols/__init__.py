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

# some sub-packages




#FIXME: Uncomment protocols and fix the import issues

from nma import *
from pdb import *
from xmipp3.protocols.protocol_preprocess import *


#ROB no file protocol_3dbionotes in devel
#from xmipp3.protocols.protocol_3dbionotes import XmippProt3DBionotes

from xmipp3.protocols.protocol_assignment_tilt_pair import XmippProtAssignmentTiltPair
from xmipp3.protocols.protocol_align_volume import XmippProtAlignVolume, XmippProtAlignVolumeForWeb
from xmipp3.protocols.protocol_preprocess.protocol_add_noise import (
    XmippProtAddNoiseVolumes, XmippProtAddNoiseParticles)
from xmipp3.protocols.protocol_apply_alignment import XmippProtApplyAlignment
from xmipp3.protocols.protocol_apply_transformation_matrix import XmippProtApplyTransformationMatrix
from xmipp3.protocols.protocol_break_symmetry import XmippProtAngBreakSymmetry
from xmipp3.protocols.protocol_cl2d_align import XmippProtCL2DAlign
from xmipp3.protocols.protocol_cl2d import XmippProtCL2D
from xmipp3.protocols.protocol_cltomo import XmippProtCLTomo
#AJ
from xmipp3.protocols.protocol_classification_gpuCorr import XmippProtGpuCrrCL2D
from xmipp3.protocols.protocol_classification_gpuCorr_semi import XmippProtStrGpuCrrSimple
from xmipp3.protocols.protocol_classification_gpuCorr_full import XmippProtStrGpuCrrCL2D
#END
# from xmipp3.protocols.protocol_ctf_defocus_group import XmippProtCTFDefocusGroup
from xmipp3.protocols.protocol_compare_reprojections import XmippProtCompareReprojections
from xmipp3.protocols.protocol_compare_angles import XmippProtCompareAngles
from xmipp3.protocols.protocol_create_gallery import XmippProtCreateGallery
from xmipp3.protocols.protocol_ctf_discrepancy import XmippProtCTFDiscrepancy
from xmipp3.protocols.protocol_ctf_micrographs import XmippProtCTFMicrographs
from xmipp3.protocols.protocol_ctf_correct_wiener2d import XmippProtCTFCorrectWiener2D
from xmipp3.protocols.protocol_consensus_classes3D import XmippProtConsensusClasses3D
from xmipp3.protocols.protocol_subtract_projection import XmippProtSubtractProjection
from xmipp3.protocols.protocol_denoise_particles import XmippProtDenoiseParticles
from xmipp3.protocols.protocol_eliminate_empty_particles import XmippProtEliminateEmptyParticles
from xmipp3.protocols.protocol_extract_particles import XmippProtExtractParticles
from xmipp3.protocols.protocol_extract_particles_movies import XmippProtExtractMovieParticles
from xmipp3.protocols.protocol_extract_particles_pairs import XmippProtExtractParticlesPairs
from xmipp3.protocols.protocol_extract_unit_cell import XmippProtExtractUnit
from xmipp3.protocols.protocol_helical_parameters import XmippProtHelicalParameters
from xmipp3.protocols.protocol_kerdensom import XmippProtKerdensom
from xmipp3.protocols.protocol_ml2d import XmippProtML2D
from xmipp3.protocols.protocol_movie_gain import XmippProtMovieGain
from xmipp3.protocols.protocol_mltomo import XmippProtMLTomo
from xmipp3.protocols.protocol_movie_average import XmippProtMovieAverage
from xmipp3.protocols.protocol_movie_correlation import XmippProtMovieCorr
from xmipp3.protocols.protocol_movie_opticalflow import XmippProtOFAlignment, ProtMovieAlignment
from xmipp3.protocols.protocol_movie_max_shift import XmippProtMovieMaxShift
from xmipp3.protocols.protocol_multiple_fscs import XmippProtMultipleFSCs
from xmipp3.protocols.protocol_multireference_alignability import XmippProtMultiRefAlignability
from xmipp3.protocols.protocol_normalize_strain import XmippProtNormalizeStrain
from xmipp3.protocols.protocol_particle_pick_automatic import XmippParticlePickingAutomatic
from xmipp3.protocols.protocol_particle_pick_consensus import XmippProtConsensusPicking
from xmipp3.protocols.protocol_particle_pick import XmippProtParticlePicking
from xmipp3.protocols.protocol_particle_pick_pairs import XmippProtParticlePickingPairs
from xmipp3.protocols.protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from xmipp3.protocols.protocol_projmatch import XmippProtProjMatch, XmippProjMatchViewer
from xmipp3.protocols.protocol_random_conical_tilt import XmippProtRCT
from xmipp3.protocols.protocol_ransac import XmippProtRansac
from xmipp3.protocols.protocol_reconstruct_fourier import XmippProtReconstructFourier
from xmipp3.protocols.protocol_reconstruct_highres import XmippProtReconstructHighRes
from xmipp3.protocols.protocol_reconstruct_significant import XmippProtReconstructSignificant
from xmipp3.protocols.protocol_reconstruct_swarm import XmippProtReconstructSwarm
from xmipp3.protocols.protocol_resolution3d import XmippProtResolution3D
from xmipp3.protocols.protocol_resolution_monogenic_signal import XmippProtMonoRes
from xmipp3.protocols.protocol_rotational_spectra import XmippProtRotSpectra
from xmipp3.protocols.protocol_rotational_symmetry import XmippProtRotationalSymmetry
from xmipp3.protocols.protocol_screen_particles import XmippProtScreenParticles
from xmipp3.protocols.protocol_solid_angles import XmippProtSolidAngles
from xmipp3.protocols.protocol_split_volume import XmippProtSplitvolume
from xmipp3.protocols.protocol_trigger_data import XmippProtTriggerData
from xmipp3.protocols.protocol_validate_nontilt import XmippProtValidateNonTilt
from xmipp3.protocols.protocol_validate_overfitting import XmippProtValidateOverfitting
# from xmipp3.protocols.protocol_validate_tilt import XmippProtValidateTilt
from xmipp3.protocols.protocol_volume_strain import XmippProtVolumeStrain
from xmipp3.protocols.protocol_volume_homogenizer import XmippProtVolumeHomogenizer
from xmipp3.protocols.protocol_write_testC import XmippProtWriteTestC
from xmipp3.protocols.protocol_write_testP import XmippProtWriteTestP
from xmipp3.protocols.protocol_ctf_selection import XmippProtCTFSelection
#AJ
from xmipp3.protocols.protocol_realignment_classes import XmippProtReAlignClasses
