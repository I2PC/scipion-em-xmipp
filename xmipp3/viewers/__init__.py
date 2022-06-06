# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *
# * [1] SciLifeLab, Stockholm University
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

from .viewer import XmippViewer
from .plotter import XmippPlotter

from .viewer_cl2d import XmippCL2DViewer
from .viewer_ctf_consensus import XmippCTFConsensusViewer
from .viewer_deep_consensus import XmippDeepConsensusViewer
from .viewer_deep_micrograph_cleaner import XmippDeepMicrographViewer
from .viewer_ml2d import XmippML2DViewer
from .viewer_mltomo import XmippMLTomoViewer
from .viewer_movie_alignment import XmippMovieAlignViewer, XmippMovieMaxShiftViewer
from .viewer_normalize_strain import XmippNormalizeStrainViewer
from .viewer_resolution3d import XmippResolution3DViewer
from .viewer_resolution_bfactor import XmippBfactorResolutionViewer
from .viewer_resolution_directional import XmippMonoDirViewer
from .viewer_resolution_fso import XmippProtFSOViewer
from .viewer_resolution_monogenic_signal import XmippMonoResViewer
from .viewer_resolution_deepres import XmippResDeepResViewer
from .viewer_validate_nontilt import XmippValidateNonTiltViewer
from .viewer_validate_fscq import XmippProtValFitViewer
from .viewer_split_volume import XmippViewerSplitVolume
from .viewer_validate_overfitting import XmippValidateOverfittingViewer
from .viewer_volume_strain import XmippVolumeStrainViewer
from .viewer_reconstruct_highres import XmippReconstructHighResViewer
from .viewer_solid_angles import SolidAnglesViewer
from .viewer_consensus_classes3D import XmippConsensusClasses3DViewer
from .viewer_classes3D import XmippClasses3DViewer
from .viewer_extract_asymmetric_unit import viewerXmippProtExtractUnit
from .viewer_deepEMHancer import viewerXmippProtDeepVolPostProc
from .viewer_local_sharpening import viewerXmippProtLocSharp
from .viewer_volume_subtraction import XmippProtVolSubtractionViewer
from .viewer_volume_consensus import XmippVolumeConsensusViewer

#from .viewer_combine_pdb import XmippProtCombinePdbViewer

from .viewer_projmatch import XmippProjMatchViewer

from .viewer_volume_deform_zernike3d import XmippVolumeDeformZernike3DViewer
from .viewer_structure_map import XmippProtStructureMapViewer
from .viewer_pdb_deform_zernike3d import XmippPDBDeformViewer
from .viewer_metaprotocol_golden_highres import XmippMetaprotocolGoldenHighResViewer
from .viewer_deep_align import XmippDeepAlignViewer

# TODO: Import from continuousflex the modules needed to create clusters (soft dependency)
from .viewer_angular_alignment_sph import XmippAngularAlignmentSphViewer
from .viewer_analyze_local_ctf import XmippAnalyzeLocalCTFViewer
