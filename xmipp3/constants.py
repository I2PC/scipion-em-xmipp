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
import enum

XMIPP_URL = 'https://github.com/i2pc/scipion-em-xmipp'
XMIPP_HOME = 'XMIPP_HOME'
NMA_HOME = 'NMA_HOME'
XMIPP_DLTK_NAME = 'deepLearningToolkit'  # consider to change it to xmipp_DLTK to make short it

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

class XMIPPCOLUMNS(enum.Enum):
    xcoor = 'xcoor'  # MDL_XCOOR
    ycoor = 'ycoor'  # MDL_YCOOR
    image = 'image'  # MDL_IMAGE
    scoreByVariance = 'scoreByVariance'  # MDL_SCORE_BY_VAR
    scoreByGiniCoeff = 'scoreByGiniCoeff'  # MDL_SCORE_BY_GINI
    zScoreDeepLearning1 = 'zScoreDeepLearning1'  # MDL_ZSCORE_DEEPLEARNING1
    enabled = 'enabled'   # MDL_ENABLED




"""
///==== Add labels entries from here in the SAME ORDER as declared in ENUM ==========
        //The label MDL_OBJID is special and should not be used
        MDL::addLabel(MDL_OBJID, LABEL_SIZET, "objId");
        //The label MDL_GATHER_ID is special and should not be used
        MDL::addLabel(MDL_GATHER_ID, LABEL_SIZET, "gatherId");

        //MDL::addLabel(MDL_ANGLE_COMPARISON, LABEL_VECTOR_DOUBLE, "angle_comparison");
        //MDL::addLabelAlias(MDL_ANGLE_COMPARISON, "angleComparison"); //3.0

        MDL::addLabel(MDL_ANGLE_PSI, LABEL_DOUBLE, "anglePsi");
        MDL::addLabelAlias(MDL_ANGLE_PSI, "psi");
        MDL::addLabel(MDL_ANGLE_PSI2, LABEL_DOUBLE, "anglePsi2");
        MDL::addLabelAlias(MDL_ANGLE_PSI2, "psi2");
        MDL::addLabel(MDL_ANGLE_PSI_DIFF, LABEL_DOUBLE, "anglePsiDiff");
        MDL::addLabel(MDL_ANGLE_ROT, LABEL_DOUBLE, "angleRot");
        MDL::addLabelAlias(MDL_ANGLE_ROT, "rot");
        MDL::addLabel(MDL_ANGLE_ROT2, LABEL_DOUBLE, "angleRot2");
        MDL::addLabelAlias(MDL_ANGLE_ROT2, "rot2");
        MDL::addLabel(MDL_ANGLE_ROT_DIFF, LABEL_DOUBLE, "angleRotDiff");
        MDL::addLabel(MDL_ANGLE_TILT, LABEL_DOUBLE, "angleTilt");
        MDL::addLabelAlias(MDL_ANGLE_TILT, "tilt");
        MDL::addLabel(MDL_ANGLE_TILT2, LABEL_DOUBLE, "angleTilt2");
        MDL::addLabelAlias(MDL_ANGLE_TILT2, "tilt2");
        MDL::addLabel(MDL_ANGLE_TILT_DIFF, LABEL_DOUBLE, "angleTiltDiff");
        MDL::addLabel(MDL_ANGLE_DIFF0, LABEL_DOUBLE, "angleDiff0");
        MDL::addLabel(MDL_ANGLE_DIFF, LABEL_DOUBLE, "angleDiff");
        MDL::addLabel(MDL_ANGLE_DIFF2, LABEL_DOUBLE, "angleDiff2");
        MDL::addLabel(MDL_ANGLE_Y, LABEL_DOUBLE, "angleY");
        MDL::addLabel(MDL_ANGLE_Y2, LABEL_DOUBLE, "angleY2");
        MDL::addLabel(MDL_ANGLE_TEMPERATURE, LABEL_DOUBLE, "angleTemp");

        MDL::addLabel(MDL_APPLY_SHIFT, LABEL_BOOL, "applyShift");
        MDL::addLabel(MDL_AVG, LABEL_DOUBLE, "avg");
        MDL::addLabel(MDL_AVG_CHANGES_ORIENTATIONS, LABEL_DOUBLE, "avgChanOrient");
        MDL::addLabel(MDL_AVG_CHANGES_OFFSETS, LABEL_DOUBLE, "avgChanOffset");
        MDL::addLabel(MDL_AVG_CHANGES_CLASSES, LABEL_DOUBLE, "avgChanClass");
        MDL::addLabel(MDL_AVGPMAX, LABEL_DOUBLE, "avgPMax");

        MDL::addLabel(MDL_BFACTOR, LABEL_DOUBLE, "bFactor");
        MDL::addLabel(MDL_BGMEAN, LABEL_DOUBLE, "bgMean");
        MDL::addLabel(MDL_BLOCK_NUMBER, LABEL_INT, "blockNumber");

        MDL::addLabel(MDL_CL2D_CHANGES, LABEL_INT, "cl2dChanges");
        MDL::addLabel(MDL_CL2D_SIMILARITY, LABEL_DOUBLE, "cl2dSimilarity");
        MDL::addLabel(MDL_CLASS_COUNT, LABEL_SIZET, "classCount");
        MDL::addLabelAlias(MDL_CLASS_COUNT, "class_count"); //3.0
        MDL::addLabel(MDL_CLASS_PERCENTAGE, LABEL_DOUBLE, "classPercentage");
        MDL::addLabel(MDL_CLASSIFICATION_DATA, LABEL_VECTOR_DOUBLE, "classificationData");
        MDL::addLabelAlias(MDL_CLASSIFICATION_DATA, "ClassificationData");
        MDL::addLabel(MDL_CLASSIFICATION_DATA_SIZE, LABEL_SIZET, "classificationDatasize");
        MDL::addLabelAlias(MDL_CLASSIFICATION_DATA_SIZE, "ClassificationDataSize");
        MDL::addLabel(MDL_CLASSIFICATION_DPR_05, LABEL_DOUBLE, "classificationDPR05");
        MDL::addLabelAlias(MDL_CLASSIFICATION_DPR_05, "ClassificationDPR05");
        MDL::addLabel(MDL_CLASSIFICATION_FRC_05, LABEL_DOUBLE, "classificationFRC05");
        MDL::addLabelAlias(MDL_CLASSIFICATION_FRC_05, "ClassificationFRC05");
        MDL::addLabel(MDL_CLASSIFICATION_INTRACLASS_DISTANCE, LABEL_DOUBLE, "classificationIntraclassDistance");
        MDL::addLabelAlias(MDL_CLASSIFICATION_INTRACLASS_DISTANCE, "ClassificationIntraclassDistance");
        MDL::addLabel(MDL_COLOR, LABEL_INT, "color");
        MDL::addLabel(MDL_COMMENT, LABEL_STRING, "comment");
        MDL::addLabel(MDL_COST, LABEL_DOUBLE, "cost");
        MDL::addLabel(MDL_COST_PERCENTILE, LABEL_DOUBLE, "costPerc");
        MDL::addLabel(MDL_COORD_CONSENSUS_SCORE, LABEL_DOUBLE, "CoordConsScore");
        MDL::addLabel(MDL_COUNT2, LABEL_SIZET, "count2");
        MDL::addLabel(MDL_COUNT, LABEL_SIZET, "count");
        MDL::addLabel(MDL_CORR_DENOISED_PROJECTION, LABEL_DOUBLE, "corrDenoisedProjection");
        MDL::addLabel(MDL_CORR_DENOISED_NOISY, LABEL_DOUBLE, "corrDenoisedNoisy");
        MDL::addLabel(MDL_CORRELATION_IDX, LABEL_DOUBLE, "corrIdx");
        MDL::addLabel(MDL_CORRELATION_MASK, LABEL_DOUBLE, "corrMask");
        MDL::addLabel(MDL_CORRELATION_WEIGHT, LABEL_DOUBLE, "corrWeight");

        MDL::addLabel(MDL_CRYSTAL_CELLX, LABEL_INT, "crystalCellx");
        MDL::addLabel(MDL_CRYSTAL_CELLY, LABEL_INT, "crystalCelly");
        MDL::addLabel(MDL_CRYSTAL_DISAPPEAR_THRE, LABEL_DOUBLE, "crystalDisthresh");
        MDL::addLabel(MDL_CRYSTAL_LATTICE_A, LABEL_VECTOR_DOUBLE, "crystalLatticeA");
        MDL::addLabel(MDL_CRYSTAL_LATTICE_B, LABEL_VECTOR_DOUBLE, "crystalLatticeB");
        MDL::addLabel(MDL_CRYSTAL_ORTHO_PRJ, LABEL_BOOL, "crystalOrthoProj");
        MDL::addLabel(MDL_CRYSTAL_PROJ, LABEL_BOOL, "crystalProj");
        MDL::addLabel(MDL_CRYSTAL_SHFILE, LABEL_STRING, "crystalShiftFile");
        MDL::addLabel(MDL_CRYSTAL_SHIFTX, LABEL_DOUBLE, "crystalShiftX");
        MDL::addLabel(MDL_CRYSTAL_SHIFTY, LABEL_DOUBLE, "crystalShiftY");
        MDL::addLabel(MDL_CRYSTAL_SHIFTZ, LABEL_DOUBLE, "crystalShiftZ");
        MDL::addLabel(MDL_CRYSTAL_NOISE_SHIFT,LABEL_VECTOR_DOUBLE, "crystalNoiseShift");
        MDL::addLabel(MDL_CTF_BG_BASELINE, LABEL_DOUBLE, "ctfBgBaseline");
        MDL::addLabelAlias(MDL_CTF_BG_BASELINE, "CTFBG_Baseline");//3.0
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN2_ANGLE, LABEL_DOUBLE, "ctfBgGaussian2Angle");
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN2_ANGLE, "CTFBG_Gaussian2_Angle"); //3.0
        MDL::addLabel(MDL_CTF_BG_R1, LABEL_DOUBLE, "ctfBgR1");
        MDL::addLabel(MDL_CTF_BG_R2, LABEL_DOUBLE, "ctfBgR2");
        MDL::addLabel(MDL_CTF_BG_R3, LABEL_DOUBLE, "ctfBgR3");

        MDL::addLabel(MDL_CTF_DATA_PHASE_FLIPPED, LABEL_BOOL, "ctfPhaseFlipped");
        MDL::addLabel(MDL_CTF_CORRECTED, LABEL_BOOL, "ctfCorrected");
        MDL::addLabel(MDL_CTF_X0, LABEL_DOUBLE, "ctfX0");
        MDL::addLabel(MDL_CTF_XF, LABEL_DOUBLE, "ctfXF");
        MDL::addLabel(MDL_CTF_Y0, LABEL_DOUBLE, "ctfY0");
        MDL::addLabel(MDL_CTF_YF, LABEL_DOUBLE, "ctfYF");
        MDL::addLabel(MDL_CTF_DEFOCUS_PLANEUA, LABEL_DOUBLE, "ctfDefocusPlaneUA");
        MDL::addLabel(MDL_CTF_DEFOCUS_PLANEUB, LABEL_DOUBLE, "ctfDefocusPlaneUB");
        MDL::addLabel(MDL_CTF_DEFOCUS_PLANEUC, LABEL_DOUBLE, "ctfDefocusPlaneUC");
        MDL::addLabel(MDL_CTF_DEFOCUS_PLANEVA, LABEL_DOUBLE, "ctfDefocusPlaneVA");
        MDL::addLabel(MDL_CTF_DEFOCUS_PLANEVB, LABEL_DOUBLE, "ctfDefocusPlaneVB");
        MDL::addLabel(MDL_CTF_DEFOCUS_PLANEVC, LABEL_DOUBLE, "ctfDefocusPlaneVC");


        MDL::addLabel(MDL_CTF_BG_GAUSSIAN2_CU, LABEL_DOUBLE, "ctfBgGaussian2CU");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN2_CV, LABEL_DOUBLE, "ctfBgGaussian2CV");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN2_K, LABEL_DOUBLE, "ctfBgGaussian2K");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN2_SIGMAU, LABEL_DOUBLE, "ctfBgGaussian2SigmaU");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN2_SIGMAV, LABEL_DOUBLE, "ctfBgGaussian2SigmaV");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN_ANGLE, LABEL_DOUBLE, "ctfBgGaussianAngle");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN_CU, LABEL_DOUBLE, "ctfBgGaussianCU");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN_CV, LABEL_DOUBLE, "ctfBgGaussianCV");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN_K, LABEL_DOUBLE, "ctfBgGaussianK");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN_SIGMAU, LABEL_DOUBLE, "ctfBgGaussianSigmaU");
        MDL::addLabel(MDL_CTF_BG_GAUSSIAN_SIGMAV, LABEL_DOUBLE, "ctfBgGaussianSigmaV");
        MDL::addLabel(MDL_CTF_PHASE_SHIFT, LABEL_DOUBLE, "ctfVPPphaseshift");
        MDL::addLabel(MDL_CTF_VPP_RADIUS, LABEL_DOUBLE, "ctfVPPRadius");

        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN2_CU, "CTFBG_Gaussian2_CU");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN2_CV, "CTFBG_Gaussian2_CV");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN2_K, "CTFBG_Gaussian2_K");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN2_SIGMAU, "CTFBG_Gaussian2_SigmaU");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN2_SIGMAV, "CTFBG_Gaussian2_SigmaV");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN_ANGLE, "CTFBG_Gaussian_Angle");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN_CU, "CTFBG_Gaussian_CU");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN_CV, "CTFBG_Gaussian_CV");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN_K, "CTFBG_Gaussian_K");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN_SIGMAU, "CTFBG_Gaussian_SigmaU");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_GAUSSIAN_SIGMAV, "CTFBG_Gaussian_SigmaV");//3.0

        MDL::addLabel(MDL_CTF_BG_SQRT_ANGLE, LABEL_DOUBLE, "ctfBgSqrtAngle");
        MDL::addLabel(MDL_CTF_BG_SQRT_K, LABEL_DOUBLE, "ctfBgSqrtK");
        MDL::addLabel(MDL_CTF_BG_SQRT_U, LABEL_DOUBLE, "ctfBgSqrtU");
        MDL::addLabel(MDL_CTF_BG_SQRT_V, LABEL_DOUBLE, "ctfBgSqrtV");
        MDL::addLabelAlias(MDL_CTF_BG_SQRT_ANGLE, "CTFBG_Sqrt_Angle");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_SQRT_K, "CTFBG_Sqrt_K");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_SQRT_U, "CTFBG_Sqrt_U");//3.0
        MDL::addLabelAlias(MDL_CTF_BG_SQRT_V, "CTFBG_Sqrt_V");  //3.0

        MDL::addLabel(MDL_CONTINUOUS_X, LABEL_DOUBLE, "continuousX");
        MDL::addLabel(MDL_CONTINUOUS_Y, LABEL_DOUBLE, "continuousY");
        MDL::addLabel(MDL_CONTINUOUS_FLIP, LABEL_BOOL, "continuousFlip");
        MDL::addLabel(MDL_CONTINUOUS_GRAY_A, LABEL_DOUBLE, "continuousA");
        MDL::addLabel(MDL_CONTINUOUS_GRAY_B, LABEL_DOUBLE, "continuousB");
        MDL::addLabel(MDL_CONTINUOUS_SCALE_ANGLE, LABEL_DOUBLE, "continuousScaleAngle");
        MDL::addLabel(MDL_CONTINUOUS_SCALE_X, LABEL_DOUBLE, "continuousScaleX");
        MDL::addLabel(MDL_CONTINUOUS_SCALE_Y, LABEL_DOUBLE, "continuousScaleY");
        MDL::addLabel(MDL_CTF_CA, LABEL_DOUBLE, "ctfChromaticAberration");
        MDL::addLabel(MDL_CTF_CONVERGENCE_CONE, LABEL_DOUBLE, "ctfConvergenceCone");
        MDL::addLabel(MDL_CTF_CRIT_NONASTIGMATICVALIDITY, LABEL_DOUBLE, "ctfCritNonAstigmaticValidty");
        MDL::addLabel(MDL_CTF_CRIT_DAMPING, LABEL_DOUBLE, "ctfCritDamping");
        MDL::addLabel(MDL_CTF_CRIT_FIRSTZEROAVG, LABEL_DOUBLE, "ctfCritFirstZero");
        MDL::addLabel(MDL_CTF_CRIT_FIRSTZERODISAGREEMENT, LABEL_DOUBLE, "ctfCritDisagree");
        MDL::addLabel(MDL_CTF_CRIT_MAXFREQ, LABEL_DOUBLE, "ctfCritMaxFreq");
        MDL::addLabel(MDL_CTF_CRIT_FIRSTZERORATIO, LABEL_DOUBLE, "ctfCritfirstZeroRatio");
        MDL::addLabel(MDL_CTF_CRIT_FIRSTMINIMUM_FIRSTZERO_RATIO, LABEL_DOUBLE, "ctfCritFirstMinFirstZeroRatio");
        MDL::addLabel(MDL_CTF_CRIT_FIRSTMINIMUM_FIRSTZERO_DIFF_RATIO, LABEL_DOUBLE, "ctfCritCtfMargin");
        MDL::addLabel(MDL_CTF_CRIT_FITTINGCORR13, LABEL_DOUBLE, "ctfCritCorr13");
        MDL::addLabel(MDL_CTF_CRIT_FITTINGSCORE, LABEL_DOUBLE, "ctfCritFitting");
        MDL::addLabel(MDL_CTF_CRIT_ICENESS, LABEL_DOUBLE, "ctfCritIceness");
        MDL::addLabel(MDL_CTF_CRIT_NORMALITY, LABEL_DOUBLE, "ctfCritNormality");
        MDL::addLabel(MDL_CTF_CRIT_PSDCORRELATION90, LABEL_DOUBLE, "ctfCritPsdCorr90");
        MDL::addLabel(MDL_CTF_CRIT_PSDPCA1VARIANCE, LABEL_DOUBLE, "ctfCritPsdPCA1");
        MDL::addLabel(MDL_CTF_CRIT_PSDPCARUNSTEST, LABEL_DOUBLE, "ctfCritPsdPCARuns");
        MDL::addLabel(MDL_CTF_CRIT_PSDRADIALINTEGRAL, LABEL_DOUBLE, "ctfCritPsdInt");
        MDL::addLabel(MDL_CTF_CRIT_PSDVARIANCE, LABEL_DOUBLE, "ctfCritPsdStdQ");
        MDL::addLabel(MDL_CTF_CS, LABEL_DOUBLE, "ctfSphericalAberration");
        MDL::addLabel(MDL_CTF_DEFOCUSA, LABEL_DOUBLE, "ctfDefocusA");//average defocus
        MDL::addLabel(MDL_CTF_DEFOCUS_ANGLE, LABEL_DOUBLE, "ctfDefocusAngle");
        MDL::addLabel(MDL_CTF_DEFOCUSU, LABEL_DOUBLE, "ctfDefocusU");
        MDL::addLabel(MDL_CTF_DEFOCUSV, LABEL_DOUBLE, "ctfDefocusV");
        MDL::addLabel(MDL_CTF_DEFOCUS_CHANGE, LABEL_DOUBLE, "ctfDefocusChange");
        MDL::addLabel(MDL_CTF_DEFOCUS_R2, LABEL_DOUBLE, "ctfDefocusR2");
        MDL::addLabel(MDL_CTF_DEFOCUS_RESIDUAL, LABEL_DOUBLE, "ctfDefocusResidual");
        MDL::addLabel(MDL_CTF_DEFOCUS_COEFS, LABEL_VECTOR_DOUBLE, "ctfDefocusCoeficients");
        MDL::addLabel(MDL_CTF_DIMENSIONS, LABEL_VECTOR_DOUBLE, "ctfDimensions");
        MDL::addLabel(MDL_CTF_DOWNSAMPLE_PERFORMED, LABEL_DOUBLE, "CtfDownsampleFactor");
        MDL::addLabel(MDL_CTF_ENERGY_LOSS, LABEL_DOUBLE, "ctfEnergyLoss");
        MDL::addLabel(MDL_CTF_ENVELOPE, LABEL_DOUBLE, "ctfEnvelope");
        MDL::addLabel(MDL_CTF_ENVELOPE_PLOT, LABEL_STRING, "ctfEnvelopePlot");
        MDL::addLabel(MDL_CTF_GROUP, LABEL_INT, "ctfGroup");
        MDL::addLabel(MDL_CTF_INPUTPARAMS, LABEL_STRING, "ctfInputParams", TAGLABEL_TEXTFILE);
        MDL::addLabel(MDL_CTF_K, LABEL_DOUBLE, "ctfK");
        MDL::addLabel(MDL_CTF_ENV_R0, LABEL_DOUBLE, "ctfEnvR0");
        MDL::addLabel(MDL_CTF_ENV_R1, LABEL_DOUBLE, "ctfEnvR1");
        MDL::addLabel(MDL_CTF_ENV_R2, LABEL_DOUBLE, "ctfEnvR2");
        MDL::addLabel(MDL_CTF_LAMBDA, LABEL_DOUBLE, "ctfLambda");
        MDL::addLabel(MDL_CTF_LENS_STABILITY, LABEL_DOUBLE, "ctfLensStability");
        MDL::addLabel(MDL_CTF_LONGITUDINAL_DISPLACEMENT, LABEL_DOUBLE, "ctfLongitudinalDisplacement");
        MDL::addLabel(MDL_CTF_MODEL2, LABEL_STRING, "ctfModel2", TAGLABEL_CTFPARAM);
        MDL::addLabel(MDL_CTF_MODEL, LABEL_STRING, "ctfModel", TAGLABEL_CTFPARAM);
        MDL::addLabel(MDL_CTF_Q0, LABEL_DOUBLE, "ctfQ0");
        MDL::addLabel(MDL_CTF_SAMPLING_RATE, LABEL_DOUBLE, "ctfSamplingRate");
        MDL::addLabel(MDL_CTF_SAMPLING_RATE_Z, LABEL_DOUBLE, "ctfSamplingRateZ");
        MDL::addLabel(MDL_CTF_TRANSVERSAL_DISPLACEMENT, LABEL_DOUBLE, "ctfTransversalDisplacement");
        MDL::addLabel(MDL_CTF_VOLTAGE, LABEL_DOUBLE, "ctfVoltage");
        MDL::addLabel(MDL_CTF_XRAY_LENS_TYPE, LABEL_STRING, "ctfXrayLensType");
        MDL::addLabel(MDL_CTF_XRAY_OUTER_ZONE_WIDTH, LABEL_DOUBLE, "ctfXrayOuterZoneWidth");
        MDL::addLabel(MDL_CTF_XRAY_ZONES_NUMBER, LABEL_DOUBLE, "ctfXrayZonesN");
        MDL::addLabel(MDL_CUMULATIVE_SSNR, LABEL_DOUBLE, "cumulativeSSNR");

        MDL::addLabelAlias(MDL_CTF_CA, "CTF_Chromatic_aberration"); //3.0
        MDL::addLabelAlias(MDL_CTF_CONVERGENCE_CONE, "CTF_Convergence_cone"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_DAMPING, "CTFCrit_damping"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_FIRSTZEROAVG, "CTFCrit_firstZero"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_FIRSTZERODISAGREEMENT, "CTFCrit_disagree"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_FIRSTZERORATIO, "CTFCrit_firstZeroRatio"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_FITTINGCORR13, "CTFCrit_Corr13"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_FITTINGSCORE, "CTFCrit_Fitting"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_NORMALITY, "CTFCrit_Normality"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_PSDCORRELATION90, "CTFCrit_psdcorr90"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_PSDPCA1VARIANCE, "CTFCrit_PSDPCA1"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_PSDPCARUNSTEST, "CTFCrit_PSDPCARuns"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_PSDRADIALINTEGRAL, "CTFCrit_psdint"); //3.0
        MDL::addLabelAlias(MDL_CTF_CRIT_PSDVARIANCE, "CTFCrit_PSDStdQ"); //3.0
        MDL::addLabelAlias(MDL_CTF_CS, "CTF_Spherical_aberration"); //3.0
        MDL::addLabelAlias(MDL_CTF_DEFOCUSA, "CTF_Defocus_A"); //3.0//average defocus
        MDL::addLabelAlias(MDL_CTF_DEFOCUS_ANGLE, "CTF_Defocus_angle"); //3.0
        MDL::addLabelAlias(MDL_CTF_DEFOCUSU, "CTF_Defocus_U"); //3.0
        MDL::addLabelAlias(MDL_CTF_DEFOCUSV, "CTF_Defocus_V"); //3.0
        MDL::addLabelAlias(MDL_CTF_DIMENSIONS, "CTF_Xray_dimensions"); //3.0
        MDL::addLabelAlias(MDL_CTF_DOWNSAMPLE_PERFORMED, "CTFDownsampleFactor"); //3.0
        MDL::addLabelAlias(MDL_CTF_ENERGY_LOSS, "CTF_Energy_loss"); //3.0
        MDL::addLabelAlias(MDL_CTF_GROUP, "CTFGroup"); //3.0
        MDL::addLabelAlias(MDL_CTF_INPUTPARAMS, "CTFInputParams"); //3.0
        MDL::addLabelAlias(MDL_CTF_K, "CTF_K"); //3.0
        MDL::addLabelAlias(MDL_CTF_LAMBDA, "CTF_Xray_lambda"); //3.0
        MDL::addLabelAlias(MDL_CTF_LENS_STABILITY, "CTF_Lens_stability"); //3.0
        MDL::addLabelAlias(MDL_CTF_LONGITUDINAL_DISPLACEMENT, "CTF_Longitudinal_displacement"); //3.0
        MDL::addLabelAlias(MDL_CTF_MODEL2, "CTFModel2"); //3.0
        MDL::addLabelAlias(MDL_CTF_MODEL, "CTFModel"); //3.0
        MDL::addLabelAlias(MDL_CTF_Q0, "CTF_Q0"); //3.0
        MDL::addLabelAlias(MDL_CTF_SAMPLING_RATE, "CTF_Sampling_rate"); //3.0
        MDL::addLabelAlias(MDL_CTF_SAMPLING_RATE_Z, "CTF_Sampling_rate_z"); //3.0
        MDL::addLabelAlias(MDL_CTF_TRANSVERSAL_DISPLACEMENT, "CTF_Transversal_displacement"); //3.0
        MDL::addLabelAlias(MDL_CTF_VOLTAGE, "CTF_Voltage"); //3.0
        MDL::addLabelAlias(MDL_CTF_XRAY_LENS_TYPE, "CTF_Xray_lens_type"); //3.0
        MDL::addLabelAlias(MDL_CTF_XRAY_OUTER_ZONE_WIDTH, "CTF_Xray_OuterZoneWidth"); //3.0
        MDL::addLabelAlias(MDL_CTF_XRAY_ZONES_NUMBER, "CTF_Xray_ZonesN"); //3.0

        MDL::addLabel(MDL_DATATYPE, LABEL_INT, "datatype");
        MDL::addLabel(MDL_DEFGROUP, LABEL_INT, "defocusGroup");
        MDL::addLabel(MDL_DIMENSIONS_2D, LABEL_VECTOR_DOUBLE, "dimensions2D");
        MDL::addLabel(MDL_DIMENSIONS_3D, LABEL_VECTOR_DOUBLE, "dimensions3D");
        MDL::addLabel(MDL_DIMRED, LABEL_VECTOR_DOUBLE, "dimredCoeffs");
        MDL::addLabel(MDL_DIRECTION, LABEL_VECTOR_DOUBLE, "direction");
        MDL::addLabel(MDL_DM3_IDTAG, LABEL_INT, "dm3IdTag");
        MDL::addLabel(MDL_DM3_NODEID, LABEL_INT, "dm3NodeId");
        MDL::addLabel(MDL_DM3_NUMBER_TYPE, LABEL_INT, "dm3NumberType");
        MDL::addLabel(MDL_DM3_PARENTID, LABEL_INT, "dm3ParentID");
        MDL::addLabel(MDL_DM3_SIZE, LABEL_INT, "dm3Size");
        MDL::addLabel(MDL_DM3_TAGCLASS, LABEL_STRING, "dm3TagClass");
        MDL::addLabel(MDL_DM3_TAGNAME, LABEL_STRING, "dm3TagName");
        MDL::addLabel(MDL_DM3_VALUE, LABEL_VECTOR_DOUBLE, "dm3Value");

        MDL::addLabel(MDL_ENABLED, LABEL_INT, "enabled");

        //MDL_EXECUTION_DATE so far an string but may change...
        MDL::addLabel(MDL_DATE, LABEL_STRING, "date");
        MDL::addLabel(MDL_TIME, LABEL_DOUBLE, "time");

        MDL::addLabel(MDL_FLIP, LABEL_BOOL, "flip");
        MDL::addLabelAlias(MDL_FLIP, "Flip");
        MDL::addLabel(MDL_FOM, LABEL_DOUBLE, "fom");
        MDL::addLabel(MDL_FRAME_ID, LABEL_SIZET, "frameId");

        MDL::addLabel(MDL_IDX, LABEL_SIZET, "index");
        MDL::addLabel(MDL_IMAGE1, LABEL_STRING, "image1", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE2, LABEL_STRING, "image2", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE3, LABEL_STRING, "image3", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE4, LABEL_STRING, "image4", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE5, LABEL_STRING, "image5", TAGLABEL_IMAGE);

        MDL::addLabelAlias(MDL_IMAGE1, "associatedImage1"); //3.0
        MDL::addLabelAlias(MDL_IMAGE2, "associatedImage2"); //3.0
        MDL::addLabelAlias(MDL_IMAGE3, "associatedImage3"); //3.0
        MDL::addLabelAlias(MDL_IMAGE4, "associatedImage4"); //3.0
        MDL::addLabelAlias(MDL_IMAGE5, "associatedImage5"); //3.0

        MDL::addLabel(MDL_IMAGE, LABEL_STRING, "image", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE_COVARIANCE, LABEL_STRING, "imageCovariance", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE_IDX, LABEL_SIZET, "imageIndex");
        MDL::addLabel(MDL_IMAGE_ORIGINAL, LABEL_STRING, "imageOriginal", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE_REF, LABEL_STRING, "imageRef", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE_RESIDUAL, LABEL_STRING, "imageResidual", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_IMAGE_TILTED, LABEL_STRING, "imageTilted", TAGLABEL_IMAGE);

        MDL::addLabelAlias(MDL_IMAGE_ORIGINAL, "original_image"); //3.0
        MDL::addLabelAlias(MDL_IMAGE_TILTED, "tilted_image"); //3.0

        MDL::addLabel(MDL_IMED, LABEL_DOUBLE, "imedValue");

        MDL::addLabel(MDL_IMGMD, LABEL_STRING, "imageMetaData", TAGLABEL_METADATA);
        MDL::addLabel(MDL_INTSCALE, LABEL_DOUBLE, "intScale");

        MDL::addLabel(MDL_ITEM_ID, LABEL_SIZET, "itemId");
        MDL::addLabel(MDL_ITER, LABEL_INT, "iterationNumber");

        MDL::addLabel(MDL_KERDENSOM_FUNCTIONAL, LABEL_DOUBLE, "kerdensomFunctional");
        MDL::addLabel(MDL_KERDENSOM_REGULARIZATION, LABEL_DOUBLE, "kerdensomRegularization");
        MDL::addLabel(MDL_KERDENSOM_SIGMA, LABEL_DOUBLE, "kerdensomSigma");

        MDL::addLabel(MDL_KEYWORDS, LABEL_STRING, "keywords");
        MDL::addLabel(MDL_KSTEST, LABEL_DOUBLE, "kstest");
        MDL::addLabel(MDL_LL, LABEL_DOUBLE, "logLikelihood");
        MDL::addLabelAlias(MDL_LL, "LL");
        MDL::addLabel(MDL_LOCAL_ALIGNMENT_PATCHES, LABEL_VECTOR_SIZET, "localAlignmentPatches");
        MDL::addLabel(MDL_LOCAL_ALIGNMENT_COEFFS_X, LABEL_VECTOR_DOUBLE, "localAlignmentCoeffsX");
        MDL::addLabel(MDL_LOCAL_ALIGNMENT_COEFFS_Y, LABEL_VECTOR_DOUBLE, "localAlignmentCoeffsY");
        MDL::addLabel(MDL_LOCAL_ALIGNMENT_CONF_2_5_PERC, LABEL_DOUBLE, "localAlignmnentConf25Perc");
        MDL::addLabel(MDL_LOCAL_ALIGNMENT_CONF_97_5_PERC, LABEL_DOUBLE, "localAlignmnentConf955Perc");
        MDL::addLabel(MDL_LOCAL_ALIGNMENT_CONTROL_POINTS, LABEL_VECTOR_SIZET, "localAlignmentControlPoints");
        MDL::addLabel(MDL_MACRO_CMD, LABEL_STRING, "macroCmd");
        MDL::addLabel(MDL_MACRO_CMD_ARGS, LABEL_STRING, "macroCmdArgs");
        MDL::addLabel(MDL_MAGNIFICATION, LABEL_DOUBLE, "magnification");
        MDL::addLabel(MDL_MAPTOPOLOGY, LABEL_STRING, "mapTopology");
        MDL::addLabel(MDL_MASK, LABEL_STRING, "mask", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_MAXCC, LABEL_DOUBLE, "maxCC");
        MDL::addLabel(MDL_MAXCC_PERCENTILE, LABEL_DOUBLE, "maxCCPerc");
        MDL::addLabel(MDL_MAX, LABEL_DOUBLE, "max");
        MDL::addLabel(MDL_MICROGRAPH_ID, LABEL_SIZET, "micrographId");
        MDL::addLabel(MDL_MICROGRAPH, LABEL_STRING, "micrograph", TAGLABEL_MICROGRAPH);
        MDL::addLabel(MDL_MICROGRAPH_MOVIE_ID, LABEL_SIZET, "micrographMovieId");
        MDL::addLabel(MDL_MICROGRAPH_MOVIE, LABEL_STRING, "movie", TAGLABEL_MICROGRAPH);
        MDL::addLabel(MDL_MICROGRAPH_PARTICLES, LABEL_STRING, "micrographParticles", TAGLABEL_MICROGRAPH);
        MDL::addLabel(MDL_MICROGRAPH_ORIGINAL, LABEL_STRING, "micrographOriginal", TAGLABEL_MICROGRAPH);
        MDL::addLabel(MDL_MICROGRAPH_TILTED, LABEL_STRING, "micrographTilted", TAGLABEL_MICROGRAPH);
        MDL::addLabel(MDL_MICROGRAPH_TILTED_ORIGINAL, LABEL_STRING, "micrographTiltedOriginal", TAGLABEL_MICROGRAPH);
        MDL::addLabel(MDL_MIN, LABEL_DOUBLE, "min");
        MDL::addLabel(MDL_MIRRORFRAC, LABEL_DOUBLE, "mirrorFraction");
        MDL::addLabel(MDL_MISSINGREGION_NR, LABEL_INT, "missingRegionNumber");
        MDL::addLabelAlias(MDL_MISSINGREGION_NR, "Wedge");
        MDL::addLabel(MDL_MISSINGREGION_THX0, LABEL_DOUBLE, "missingRegionThetaX0");
        MDL::addLabel(MDL_MISSINGREGION_THXF, LABEL_DOUBLE, "missingRegionThetaXF");

        MDL::addLabel(MDL_MLF_CTF,    LABEL_DOUBLE, "mlfCtf");
        MDL::addLabel(MDL_MLF_WIENER, LABEL_DOUBLE, "mlfWiener");
        MDL::addLabel(MDL_MLF_SIGNAL, LABEL_DOUBLE, "mlfSignal");
        MDL::addLabel(MDL_MLF_NOISE,  LABEL_DOUBLE, "mlfNoise");

        MDL::addLabel(MDL_MISSINGREGION_THY0, LABEL_DOUBLE, "missingRegionThetaY0");
        MDL::addLabel(MDL_MISSINGREGION_THYF, LABEL_DOUBLE, "missingRegionThetaYF");
        MDL::addLabel(MDL_MISSINGREGION_TYPE, LABEL_STRING, "missingRegionType");
        MDL::addLabel(MDL_MODELFRAC, LABEL_DOUBLE, "modelFraction");
        MDL::addLabel(MDL_NEIGHBORHOOD_RADIUS, LABEL_DOUBLE, "neighborhoodRadius");
        MDL::addLabel(MDL_NEIGHBOR, LABEL_SIZET, "neighbor");
        MDL::addLabel(MDL_NEIGHBORS, LABEL_VECTOR_SIZET, "neighbors");
        MDL::addLabel(MDL_NMA, LABEL_VECTOR_DOUBLE, "nmaDisplacements");
        MDL::addLabelAlias(MDL_NMA, "NMADisplacements");//3.0
        MDL::addLabel(MDL_NMA_ATOMSHIFT, LABEL_DOUBLE, "nmaAtomShift");
        MDL::addLabel(MDL_NMA_COLLECTIVITY, LABEL_DOUBLE, "nmaCollectivity");
        MDL::addLabel(MDL_NMA_ENERGY, LABEL_DOUBLE, "nmaEnergy");
        MDL::addLabel(MDL_NMA_MINRANGE, LABEL_DOUBLE, "nmaMin");
        MDL::addLabel(MDL_NMA_MAXRANGE, LABEL_DOUBLE, "nmaMax");
        MDL::addLabel(MDL_NMA_MODEFILE, LABEL_STRING, "nmaModefile", TAGLABEL_TEXTFILE);
        MDL::addLabelAlias(MDL_NMA_MODEFILE, "NMAModefile");//3.0
        MDL::addLabel(MDL_NMA_SCORE, LABEL_DOUBLE, "nmaScore");
        MDL::addLabel(MDL_NOISE_ANGLES, LABEL_VECTOR_DOUBLE, "noiseAngles");
        MDL::addLabel(MDL_NOISE_COORD, LABEL_VECTOR_DOUBLE, "noiseCoord");
        MDL::addLabel(MDL_NOISE_PARTICLE_COORD, LABEL_VECTOR_DOUBLE, "noiseParticleCoord");
        MDL::addLabel(MDL_NOISE_PIXEL_LEVEL, LABEL_VECTOR_DOUBLE, "noisePixelLevel");
        MDL::addLabel(MDL_ORDER, LABEL_SIZET, "order_");
        MDL::addLabel(MDL_ORIGIN_X, LABEL_DOUBLE, "originX");
        MDL::addLabel(MDL_ORIGIN_Y, LABEL_DOUBLE, "originY");
        MDL::addLabel(MDL_ORIGIN_Z, LABEL_DOUBLE, "originZ");
        MDL::addLabel(MDL_PARTICLE_ID, LABEL_SIZET, "particleId");
        MDL::addLabel(MDL_PHANTOM_BGDENSITY, LABEL_DOUBLE, "phantomBGDensity");
        MDL::addLabel(MDL_PHANTOM_FEATURE_CENTER, LABEL_VECTOR_DOUBLE, "featureCenter");
        MDL::addLabel(MDL_PHANTOM_FEATURE_DENSITY, LABEL_DOUBLE, "featureDensity");
        MDL::addLabel(MDL_PHANTOM_FEATURE_OPERATION, LABEL_STRING, "featureOperation");
        MDL::addLabel(MDL_PHANTOM_FEATURE_SPECIFIC, LABEL_VECTOR_DOUBLE, "featureSpecificVector");
        MDL::addLabel(MDL_PHANTOM_FEATURE_TYPE, LABEL_STRING, "featureType");
        MDL::addLabel(MDL_PHANTOM_SCALE, LABEL_DOUBLE, "phantomScale");

        MDL::addLabel(MDL_OPTICALFLOW_MEANX, LABEL_DOUBLE, "opticalMeanX");
        MDL::addLabel(MDL_OPTICALFLOW_MEANY, LABEL_DOUBLE, "opticalMeanY");
        MDL::addLabel(MDL_OPTICALFLOW_STDX, LABEL_DOUBLE, "opticalStdX");
        MDL::addLabel(MDL_OPTICALFLOW_STDY, LABEL_DOUBLE, "opticalStdY");

        MDL::addLabel(MDL_PICKING_STATE, LABEL_STRING, "pickingState");
        MDL::addLabelAlias(MDL_PICKING_STATE, "picking_state");//3.0
        MDL::addLabel(MDL_PICKING_MICROGRAPH_STATE, LABEL_STRING, "pickingMicrographState");
        MDL::addLabelAlias(MDL_PICKING_MICROGRAPH_STATE, "micrograph_state");//3.0
        MDL::addLabel(MDL_PICKING_PARTICLE_SIZE, LABEL_INT, "particleSize");
        MDL::addLabel(MDL_PICKING_AUTOPICKPERCENT, LABEL_INT, "autopickPercent");
        MDL::addLabel(MDL_PICKING_TEMPLATES, LABEL_INT, "templatesNum");
        MDL::addLabel(MDL_PICKING_AUTOPARTICLES_SIZE, LABEL_INT, "autoParticlesNum");
        MDL::addLabel(MDL_PICKING_MANUALPARTICLES_SIZE, LABEL_INT, "manualParticlesNum");

        MDL::addLabel(MDL_PMAX, LABEL_DOUBLE, "pMax");
        MDL::addLabelAlias(MDL_PMAX, "Pmax");
        MDL::addLabelAlias(MDL_PMAX, "sumP");
        MDL::addLabel(MDL_POINTSASYMETRICUNIT, LABEL_SIZET, "pointsAsymmetricUnit");
        MDL::addLabelAlias(MDL_POINTSASYMETRICUNIT, "pointsasymmetricUnit");
        MDL::addLabel(MDL_PRJ_ANGFILE, LABEL_STRING, "projAngleFile");
        MDL::addLabelAlias(MDL_PRJ_ANGFILE, "angleFile");//3.0
        MDL::addLabel(MDL_PRJ_DIMENSIONS, LABEL_VECTOR_DOUBLE, "projDimensions");
        MDL::addLabel(MDL_PRJ_PSI_NOISE, LABEL_VECTOR_DOUBLE, "projPsiNoise");
        MDL::addLabel(MDL_PRJ_PSI_RANDSTR, LABEL_STRING, "projPsiRandomness");
        MDL::addLabel(MDL_PRJ_PSI_RANGE, LABEL_VECTOR_DOUBLE, "projPsiRange");
        MDL::addLabel(MDL_PRJ_ROT_NOISE, LABEL_VECTOR_DOUBLE, "projRotNoise");
        MDL::addLabel(MDL_PRJ_ROT_RANDSTR, LABEL_STRING, "projRotRandomness");
        MDL::addLabel(MDL_PRJ_ROT_RANGE, LABEL_VECTOR_DOUBLE, "projRotRange");
        MDL::addLabel(MDL_PRJ_TILT_NOISE, LABEL_VECTOR_DOUBLE, "projTiltNoise");
        MDL::addLabel(MDL_PRJ_TILT_RANDSTR, LABEL_STRING, "projTiltRandomness");
        MDL::addLabel(MDL_PRJ_TILT_RANGE, LABEL_VECTOR_DOUBLE, "projTiltRange");
        MDL::addLabel(MDL_PRJ_VOL, LABEL_STRING, "projVolume", TAGLABEL_VOLUME);

        MDL::addLabel(MDL_PROGRAM, LABEL_STRING, "program");
        MDL::addLabel(MDL_USER, LABEL_STRING, "user");

        MDL::addLabel(MDL_PSD_ENHANCED, LABEL_STRING, "psdEnhanced", TAGLABEL_IMAGE);
        MDL::addLabelAlias(MDL_PSD_ENHANCED, "enhancedPowerSpectrum");//3.0
        MDL::addLabel(MDL_PSD, LABEL_STRING, "psd", TAGLABEL_PSD);
        MDL::addLabelAlias(MDL_PSD, "powerSpectrum");//3.0
        MDL::addLabel(MDL_RANDOMSEED, LABEL_INT, "randomSeed");
        MDL::addLabel(MDL_REF2, LABEL_INT, "ref2");
        MDL::addLabel(MDL_REF3D, LABEL_INT, "ref3d");
        MDL::addLabel(MDL_REF, LABEL_INT, "ref");
        MDL::addLabelAlias(MDL_REF, "Ref");
        MDL::addLabel(MDL_REFMD, LABEL_STRING, "referenceMetaData", TAGLABEL_METADATA);

        MDL::addLabel(MDL_RESIDUE, LABEL_INT, "residue");
        MDL::addLabel(MDL_RESOLUTION_DPR, LABEL_DOUBLE, "resolutionDPR");
        MDL::addLabel(MDL_RESOLUTION_ERRORL2, LABEL_DOUBLE, "resolutionErrorL2");
        MDL::addLabel(MDL_RESOLUTION_FSO, LABEL_DOUBLE, "resolutionFSO");
        MDL::addLabel(MDL_RESOLUTION_FRC, LABEL_DOUBLE, "resolutionFRC");
        MDL::addLabel(MDL_RESOLUTION_FRCRANDOMNOISE, LABEL_DOUBLE, "resolutionFRCRandomNoise");
        MDL::addLabel(MDL_RESOLUTION_FREQ, LABEL_DOUBLE, "resolutionFreqFourier");
        MDL::addLabel(MDL_RESOLUTION_FREQ2, LABEL_DOUBLE, "resolutionFreqFourier2");
        MDL::addLabel(MDL_RESOLUTION_FREQREAL, LABEL_DOUBLE, "resolutionFreqReal");
        MDL::addLabel(MDL_RESOLUTION_LOCAL_RESIDUE, LABEL_DOUBLE, "localresolutionResidue");
        MDL::addLabel(MDL_RESOLUTION_LOG_STRUCTURE_FACTOR, LABEL_DOUBLE, "resolutionLogStructure");
        MDL::addLabel(MDL_RESOLUTION_STRUCTURE_FACTOR, LABEL_DOUBLE, "resolutionStructure");
        MDL::addLabel(MDL_RESOLUTION_SSNR, LABEL_DOUBLE, "resolutionSSNR");
        MDL::addLabel(MDL_RESOLUTION_RFACTOR, LABEL_DOUBLE, "resolutionRfactor");

        MDL::addLabelAlias(MDL_RESOLUTION_DPR, "DPR");
        MDL::addLabelAlias(MDL_RESOLUTION_ERRORL2, "Error_l2");
        MDL::addLabelAlias(MDL_RESOLUTION_FRC, "FRC");
        MDL::addLabelAlias(MDL_RESOLUTION_FRCRANDOMNOISE, "FRC_random_noise");
        MDL::addLabelAlias(MDL_RESOLUTION_FREQ, "Resol_Inverse_Ang");
        MDL::addLabelAlias(MDL_RESOLUTION_FREQREAL, "Resol_Ang");

        MDL::addLabel(MDL_SAMPLINGRATE, LABEL_DOUBLE, "samplingRate");
        MDL::addLabel(MDL_SAMPLINGRATE_ORIGINAL, LABEL_DOUBLE, "samplingRateOriginal");
        MDL::addLabel(MDL_SAMPLINGRATE_X, LABEL_DOUBLE, "samplingRateX");
        MDL::addLabel(MDL_SAMPLINGRATE_Y, LABEL_DOUBLE, "samplingRateY");
        MDL::addLabel(MDL_SAMPLINGRATE_Z, LABEL_DOUBLE, "samplingRateZ");

        MDL::addLabelAlias(MDL_SAMPLINGRATE, "sampling_rate"); //3.0
        MDL::addLabelAlias(MDL_SAMPLINGRATE_ORIGINAL, "sampling_rate_original"); //3.0
        MDL::addLabelAlias(MDL_SAMPLINGRATE_X, "sampling_rateX"); //3.0
        MDL::addLabelAlias(MDL_SAMPLINGRATE_Y, "sampling_rateY"); //3.0
        MDL::addLabelAlias(MDL_SAMPLINGRATE_Z, "sampling_rateZ"); //3.0

        MDL::addLabel(MDL_SCALE, LABEL_DOUBLE, "scale");
        MDL::addLabel(MDL_SCORE_BY_PCA_RESIDUAL_PROJ, LABEL_DOUBLE, "scoreByPcaResidualProj");
        MDL::addLabel(MDL_SCORE_BY_PCA_RESIDUAL_EXP, LABEL_DOUBLE, "scoreByPcaResidualExp");
        MDL::addLabel(MDL_SCORE_BY_PCA_RESIDUAL, LABEL_DOUBLE, "scoreByPcaResidual");
        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY, LABEL_DOUBLE, "scoreByAlignability");
        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY_PRECISION, LABEL_DOUBLE, "scoreByAlignabilityPrecision");
        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY_ACCURACY, LABEL_DOUBLE, "scoreByAlignabilityAccuracy");
        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY_PRECISION_EXP, LABEL_DOUBLE, "scoreByAlignabilityPrecisionExp");
        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY_PRECISION_REF, LABEL_DOUBLE, "scoreByAlignabilityPrecisionRef");
        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY_ACCURACY_EXP, LABEL_DOUBLE, "scoreByAlignabilityAccuracyExp");
        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY_ACCURACY_REF, LABEL_DOUBLE, "scoreByAlignabilityAccuracyRef");

        MDL::addLabel(MDL_SCORE_BY_ALIGNABILITY_NOISE, LABEL_DOUBLE, "scoreByAlignabilityNoise");
        MDL::addLabel(MDL_SCORE_BY_EMPTINESS, LABEL_DOUBLE, "scoreEmptiness");
        MDL::addLabel(MDL_SCORE_BY_ENTROPY, LABEL_VECTOR_DOUBLE, "entropyFeatures");
        MDL::addLabel(MDL_SCORE_BY_GRANULO, LABEL_VECTOR_DOUBLE, "granuloFeatures");
        MDL::addLabel(MDL_SCORE_BY_HISTDIST, LABEL_VECTOR_DOUBLE, "histdistFeatures");
        MDL::addLabel(MDL_SCORE_BY_LBP, LABEL_VECTOR_DOUBLE, "lbpFeatures");
        MDL::addLabel(MDL_SCORE_BY_MIRROR, LABEL_DOUBLE, "scoreByMirror");
        MDL::addLabel(MDL_SCORE_BY_RAMP, LABEL_VECTOR_DOUBLE, "rampCoefficients");
        MDL::addLabel(MDL_SCORE_BY_SCREENING, LABEL_VECTOR_DOUBLE, "screenFeatures");
        MDL::addLabel(MDL_SCORE_BY_VARIANCE, LABEL_VECTOR_DOUBLE, "varianceFeatures");
        MDL::addLabel(MDL_SCORE_BY_VAR, LABEL_DOUBLE, "scoreByVariance");
        MDL::addLabel(MDL_SCORE_BY_GINI, LABEL_DOUBLE, "scoreByGiniCoeff");
        MDL::addLabel(MDL_SCORE_BY_ZERNIKE, LABEL_VECTOR_DOUBLE, "zernikeMoments");
        MDL::addLabel(MDL_SCORE_BY_ZSCORE, LABEL_DOUBLE, "scoreByZScore");

        MDL::addLabelAlias(MDL_SCALE, "Scale");
        MDL::addLabel(MDL_SELFILE, LABEL_STRING, "selfile", TAGLABEL_METADATA);
        MDL::addLabel(MDL_SERIE, LABEL_STRING, "serie");

        MDL::addLabel(MDL_SHIFT_X, LABEL_DOUBLE, "shiftX");
        MDL::addLabelAlias(MDL_SHIFT_X, "Xoff");
        MDL::addLabel(MDL_SHIFT_X2, LABEL_DOUBLE, "shiftX2");
        MDL::addLabel(MDL_SHIFT_X_DIFF, LABEL_DOUBLE, "shiftXDiff");
        MDL::addLabel(MDL_SHIFT_Y, LABEL_DOUBLE, "shiftY");
        MDL::addLabelAlias(MDL_SHIFT_Y, "Yoff");
        MDL::addLabel(MDL_SHIFT_Y2, LABEL_DOUBLE, "shiftY2");
        MDL::addLabel(MDL_SHIFT_Y_DIFF, LABEL_DOUBLE, "shiftYDiff");
        MDL::addLabel(MDL_SHIFT_Z, LABEL_DOUBLE, "shiftZ");
        MDL::addLabelAlias(MDL_SHIFT_Z, "Zoff");
        MDL::addLabel(MDL_SHIFT_DIFF0, LABEL_DOUBLE, "shiftDiff0");
        MDL::addLabel(MDL_SHIFT_DIFF, LABEL_DOUBLE, "shiftDiff");
        MDL::addLabel(MDL_SHIFT_DIFF2, LABEL_DOUBLE, "shiftDiff2");
        MDL::addLabel(MDL_SIGMANOISE, LABEL_DOUBLE, "sigmaNoise");
        MDL::addLabel(MDL_SIGMAOFFSET, LABEL_DOUBLE, "sigmaOffset");
        MDL::addLabel(MDL_SIGNALCHANGE, LABEL_DOUBLE, "signalChange");
        MDL::addLabel(MDL_SPH_COEFFICIENTS, LABEL_VECTOR_DOUBLE, "sphCoefficients");
        MDL::addLabel(MDL_SPH_DEFORMATION, LABEL_DOUBLE, "sphDeformation");
        MDL::addLabel(MDL_SPH_TSNE_COEFF1D, LABEL_DOUBLE, "sphTsne1D");
        MDL::addLabel(MDL_SPH_TSNE_COEFF2D, LABEL_VECTOR_DOUBLE, "sphTsne2D");
        MDL::addLabel(MDL_STDDEV, LABEL_DOUBLE, "stddev");
        MDL::addLabel(MDL_STAR_COMMENT, LABEL_STRING, "starComment");
        MDL::addLabel(MDL_SUM, LABEL_DOUBLE, "sum");
        MDL::addLabel(MDL_SUMWEIGHT, LABEL_DOUBLE, "sumWeight");
        MDL::addLabel(MDL_SYMNO, LABEL_INT, "symNo");

        MDL::addLabel(MDL_TOMOGRAM_VOLUME, LABEL_STRING, "tomogramVolume", TAGLABEL_IMAGE);
        MDL::addLabel(MDL_TOMOGRAMMD, LABEL_STRING, "tomogramMetadata", TAGLABEL_METADATA);

        MDL::addLabel(MDL_TRANSFORM_MATRIX, LABEL_STRING, "transformMatrix");

        MDL::addLabel(MDL_TEST_SIZE, LABEL_INT, "testSize");

        MDL::addLabel(MDL_VOLUME_SCORE_SUM, LABEL_DOUBLE, "volScoreSum");
        MDL::addLabel(MDL_VOLUME_SCORE_MEAN, LABEL_DOUBLE, "volScoreMean");
        MDL::addLabel(MDL_VOLUME_SCORE_MIN, LABEL_DOUBLE, "volScoreMin");
        MDL::addLabel(MDL_VOLUME_SCORE_SUM_TH, LABEL_DOUBLE, "volScoreSumTh");
        MDL::addLabel(MDL_VOLUME_SCORE1, LABEL_DOUBLE, "volScore1");
        MDL::addLabel(MDL_VOLUME_SCORE2, LABEL_DOUBLE, "volScore2");
        MDL::addLabel(MDL_VOLUME_SCORE3, LABEL_DOUBLE, "volScore3");
        MDL::addLabel(MDL_VOLUME_SCORE4, LABEL_DOUBLE, "volScore4");
        MDL::addLabel(MDL_WEIGHT, LABEL_DOUBLE, "weight");
        MDL::addLabel(MDL_WEIGHT_P, LABEL_DOUBLE, "weight_clusterability");
        MDL::addLabelAlias(MDL_WEIGHT, "Weight");
        MDL::addLabel(MDL_WEIGHT_CONTINUOUS2, LABEL_DOUBLE, "weightContinuous2");
        MDL::addLabel(MDL_WEIGHT_JUMPER0, LABEL_DOUBLE, "weightJumper0");
        MDL::addLabel(MDL_WEIGHT_JUMPER, LABEL_DOUBLE, "weightJumper");
        MDL::addLabel(MDL_WEIGHT_JUMPER2, LABEL_DOUBLE, "weightJumper2");
        MDL::addLabel(MDL_WEIGHT_SIGNIFICANT, LABEL_DOUBLE, "weightSignificant");
        MDL::addLabel(MDL_WEIGHT_SSNR, LABEL_DOUBLE, "weightSSNR");

        MDL::addLabel(MDL_WEIGHT_PRECISION_ALIGNABILITY, LABEL_DOUBLE, "weightPrecisionAlignability");
        MDL::addLabel(MDL_WEIGHT_ACCURACY_ALIGNABILITY, LABEL_DOUBLE, "weightAccuracyAlignability");
        MDL::addLabel(MDL_WEIGHT_ALIGNABILITY, LABEL_DOUBLE, "weightAlignability");
        MDL::addLabel(MDL_WEIGHT_PRECISION_MIRROR, LABEL_DOUBLE, "weightPrecisionMirror");

        MDL::addLabel(MDL_WROBUST, LABEL_DOUBLE, "wRobust");
        MDL::addLabel(MDL_XCOOR, LABEL_INT, "xcoor");
        MDL::addLabel(MDL_XCOOR_TILT, LABEL_INT, "xcoorTilt");

        MDL::addLabel(MDL_X, LABEL_DOUBLE, "x");
        MDL::addLabel(MDL_XSIZE, LABEL_SIZET, "xSize");
        MDL::addLabel(MDL_YCOOR, LABEL_INT, "ycoor");
        MDL::addLabel(MDL_YCOOR_TILT, LABEL_INT, "ycoorTilt");
        MDL::addLabel(MDL_Y, LABEL_DOUBLE, "y");
        MDL::addLabel(MDL_YSIZE, LABEL_SIZET, "ySize");
        MDL::addLabel(MDL_ZCOOR, LABEL_INT, "zcoor");
        MDL::addLabel(MDL_Z, LABEL_DOUBLE, "z");
        MDL::addLabel(MDL_ZSCORE, LABEL_DOUBLE, "zScore");
        MDL::addLabel(MDL_ZSCORE_HISTOGRAM, LABEL_DOUBLE, "zScoreHistogram");
        MDL::addLabel(MDL_ZSCORE_RESMEAN, LABEL_DOUBLE, "zScoreResMean");
        MDL::addLabel(MDL_ZSCORE_RESVAR, LABEL_DOUBLE, "zScoreResVar");
        MDL::addLabel(MDL_ZSCORE_RESCOV, LABEL_DOUBLE, "zScoreResCov");
        MDL::addLabel(MDL_ZSCORE_SHAPE1, LABEL_DOUBLE, "zScoreShape1");
        MDL::addLabel(MDL_ZSCORE_SHAPE2, LABEL_DOUBLE, "zScoreShape2");
        MDL::addLabel(MDL_ZSCORE_SNR1, LABEL_DOUBLE, "zScoreSNR1");
        MDL::addLabel(MDL_ZSCORE_SNR2, LABEL_DOUBLE, "zScoreSNR2");
        MDL::addLabel(MDL_ZSCORE_DEEPLEARNING1, LABEL_DOUBLE, "zScoreDeepLearning1");
        MDL::addLabel(MDL_GOOD_REGION_SCORE, LABEL_DOUBLE, "goodRegionScore");
        MDL::addLabel(MDL_ZSIZE, LABEL_SIZET, "zSize");

        MDL::addLabelAlias(MDL_XCOOR, "Xcoor");//3.0
        MDL::addLabelAlias(MDL_XCOOR, "<X position>");
        MDL::addLabelAlias(MDL_XCOOR_TILT, "XcoorTilt");//3.0

        MDL::addLabelAlias(MDL_X, "X"); //3.0
        MDL::addLabelAlias(MDL_XSIZE, "Xsize"); //3.0
        MDL::addLabelAlias(MDL_YCOOR, "Ycoor"); //3.0
        MDL::addLabelAlias(MDL_YCOOR, "<Y position>");
        MDL::addLabelAlias(MDL_YCOOR_TILT, "YcoorTilt"); //3.0
        MDL::addLabelAlias(MDL_Y, "Y"); //3.0
        MDL::addLabelAlias(MDL_YSIZE, "Ysize"); //3.0
        MDL::addLabelAlias(MDL_ZCOOR, "Zcoor"); //3.0
        MDL::addLabelAlias(MDL_Z, "Z"); //3.0
        MDL::addLabelAlias(MDL_ZSCORE, "Zscore"); //3.0
        MDL::addLabelAlias(MDL_ZSIZE, "Zsize"); //3.0


"""
