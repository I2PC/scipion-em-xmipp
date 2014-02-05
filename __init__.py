# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
This sub-package will contains Xmipp3.0 specific protocols
"""

_logo = "xmipp_logo.png"
_referencesDict = {
                   'DelaRosa2013_Xmipp':'[[http://www.ncbi.nlm.nih.gov/pubmed/24075951][De la Rosa-Trevin, et.al, JSB (2013)]]',
                   'Sorzano2013_Protocols':'[[http://www.ncbi.nlm.nih.gov/pubmed/23086876][Sorzano, et.al, Meth.Mol.Biol. (2013)]]',
                   'Otsu1979_Segmentation':'[[http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=4310076][Otsu, IEEE. Trans. SCM. (1979)]]'
                   }
                #'[[http://www.ncbi.nlm.nih.gov/pubmed/15477099][Xmipp: Sorzano, JSB (2004)]]',
                #'[[http://www.ncbi.nlm.nih.gov/pubmed/8812978][Xmipp: Marabini, JSB (1996)]]',
                #'[[http://www.sciencedirect.com/science/article/pii/B9780124059146000160][Protocols: Devaux, Meth.Cell.Biol. (2012)]]',
                #'[[http://www.ncbi.nlm.nih.gov/pubmed/18536645][Protocols: Scheres, Nat.Prot. (2008)]]',
_references = [_referencesDict['DelaRosa2013_Xmipp'], _referencesDict['Sorzano2013_Protocols']]

from xmipp3 import *
from convert import *
from viewer import XmippViewer
from viewer_ml2d import XmippML2DViewer
from viewer_cl2d import XmippCL2DViewer
from viewer_ml3d import XmippML3DViewer
from viewer_nma import XmippNMAViewer
from viewer_nma_alignment import XmippAlignmentNMAViewer

from plotter import XmippPlotter
from protocol_preprocess_micrographs import XmippProtPreprocessMicrographs
from protocol_preprocess_volumes import XmippProtPreprocessVolumes
from protocol_ctf_micrographs import XmippProtCTFMicrographs
from protocol_particle_pick import XmippProtParticlePicking 
from protocol_extract_particles import XmippProtExtractParticles
from protocol_ml2d import XmippProtML2D
from protocol_cl2d import XmippProtCL2D
from protocol_cl2d_align import XmippProtCL2DAlign
from protocol_kerdensom import XmippProtKerdensom
from protocol_rotational_spectra import XmippProtRotSpectra 
from protocol_ml3d import XmippProtML3D
from protocol_projmatch import XmippProtProjMatch
from protocol_filters import XmippProtFilter
from protocol_filters import XmippProtMask, XmippProtResize
from protocol_create_mask import XmippProtCreateMask3D
#from protocol_apply_mask import XmippProtApplyMask3D
from protocol_particle_pick_automatic import XmippParticlePickingAutomatic
from protocol_screen_particles import XmippProtScreenParticles
from protocol_simulated_annealing import XmippProtInitVolSimAnneal
from protocol_ransac import XmippProtRansac
from protocol_convert_pdb import XmippProtConvertPdb
from protocol_join_sets import XmippProtJoinSets
from protocol_nma import XmippProtNMA
from protocol_nma_alignment import XmippProtAlignmentNMA
from protocol_create_mask import XmippProtCreateMask3D
from protocol_align_volume import XmippProtAlignVolume
from protocol_screen_classes import XmippProtScreenClasses

# Wizards
from wizard import *
