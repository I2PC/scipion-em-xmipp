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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
"""
This modules contains constants related to Xmipp3 protocols
"""

# ------------------ Constants values --------------------------------------
XMIPP_GIT_URL = 'https://github.com/I2PC/xmipp.git'
XMIPP_URL = 'https://github.com/i2pc/scipion-em-xmipp'
XMIPP_HOME = 'XMIPP_HOME'
NMA_HOME = 'NMA_HOME'
XMIPP_DLTK_NAME = 'deepLearningToolkit'  # consider to change it to xmipp_DLTK to make short it
XMIPP_CUDA_BIN = 'XMIPP_CUDA_BIN'
XMIPP_CUDA_LIB = 'XMIPP_CUDA_LIB'

MASK_FILL_VALUE = 0
MASK_FILL_MIN = 1
MASK_FILL_MAX = 2
MASK_FILL_AVG = 3

PROJECT_FOURIER = 0
PROJECT_REALSPACE = 1

KERNEL_NEAREST = 0
KERNEL_LINEAR = 1
KERNEL_BSPLINE = 2

SELECT_NONE = 0
SELECT_MAXCC = 1
SELECT_PERCENTAGE = 2
SELECT_CLASSPERCENTAGE = 3

RECONSTRUCT_FOURIER = 0
RECONSTRUCT_ART = 1
RECONSTRUCT_WBP = 2

FILTER_SPACE_FOURIER = 0
FILTER_SPACE_REAL = 1
FILTER_SPACE_WAVELET = 2

# Rotational spectra find center mode
ROTSPECTRA_CENTER_MIDDLE = 0
ROTSPECTRA_CENTER_FIRST_HARMONIC = 1

# 3D Geometrical Mask
MASK3D_SPHERE = 0
MASK3D_BOX = 1
MASK3D_CROWN = 2
MASK3D_CYLINDER = 3
MASK3D_GAUSSIAN = 4
MASK3D_RAISED_COSINE = 5
MASK3D_RAISED_CROWN = 6

# Mask Choice
SOURCE_GEOMETRY = 0
SOURCE_MASK = 1

# 2D Geometrical Mask
MASK2D_CIRCULAR = 0
MASK2D_BOX = 1
MASK2D_CROWN = 2
MASK2D_GAUSSIAN = 3
MASK2D_RAISED_COSINE = 4
MASK2D_RAISED_CROWN = 5

# Threshold value substitute
FILL_VALUE = 0
FILL_BINARIZE = 1
FILL_AVG = 2

# Reconstruction methods
RECONSTRUCT_FOURIER = 0
RECONSTRUCT_WSLART = 1

# Micrograph type constants for particle extraction
SAME_AS_PICKING = 0
OTHER = 1

SYM_URL = "[[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]]"

# symmetry dictionary
#FIXME: This should not be imported here and exposed as this module constants
from pwem.constants import (
    SYM_CYCLIC, SYM_DIHEDRAL_X, SYM_TETRAHEDRAL, SYM_OCTAHEDRAL, SYM_I222,
    SYM_I222r, SYM_In25, SYM_In25r)


XMIPP_TO_SCIPION = {}
XMIPP_CYCLIC = 0
XMIPP_DIHEDRAL_X = 1
XMIPP_TETRAHEDRAL = 2
XMIPP_OCTAHEDRAL = 3
XMIPP_I222 = 4
XMIPP_I222r = 5
XMIPP_In25 = 6
XMIPP_In25r = 7

XMIPP_TO_SCIPION[XMIPP_CYCLIC] = SYM_CYCLIC
XMIPP_TO_SCIPION[XMIPP_DIHEDRAL_X] = SYM_DIHEDRAL_X
XMIPP_TO_SCIPION[XMIPP_TETRAHEDRAL] = SYM_TETRAHEDRAL
XMIPP_TO_SCIPION[XMIPP_OCTAHEDRAL] = SYM_OCTAHEDRAL
XMIPP_TO_SCIPION[XMIPP_I222] = SYM_I222
XMIPP_TO_SCIPION[XMIPP_I222r] = SYM_I222r
XMIPP_TO_SCIPION[XMIPP_In25] = SYM_In25
XMIPP_TO_SCIPION[XMIPP_In25r] = SYM_In25r

XMIPP_SYM_NAME = {}
XMIPP_SYM_NAME[XMIPP_CYCLIC] = 'Cn'
XMIPP_SYM_NAME[XMIPP_DIHEDRAL_X] = 'Dn'
XMIPP_SYM_NAME[XMIPP_TETRAHEDRAL] = 'T'
XMIPP_SYM_NAME[XMIPP_OCTAHEDRAL] = 'O'
XMIPP_SYM_NAME[XMIPP_I222] = 'I1'
XMIPP_SYM_NAME[XMIPP_I222r] = 'I2'
XMIPP_SYM_NAME[XMIPP_In25] = 'I3'
XMIPP_SYM_NAME[XMIPP_In25r] = 'I4'


# Xmipp programs
CUDA_ALIGN_SIGNIFICANT = "xmipp_cuda_align_significant"
