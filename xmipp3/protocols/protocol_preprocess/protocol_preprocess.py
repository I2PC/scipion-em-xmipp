# ******************************************************************************
# *
# * Authors:     Jose Gutierrez (jose.gutierrez@cnb.csic.es)
# *              Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
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
# ******************************************************************************

import os
from pyworkflow.utils import *
from pyworkflow.protocol.params import *
from pyworkflow.utils.path import cleanPath

from pwem.objects import Volume, SetOfParticles, SetOfClasses2D

from pwem import emlib
from xmipp3.constants import *
from xmipp3.convert import  writeSetOfParticles
from xmipp3.base import isXmippCudaPresent
from .protocol_process import XmippProcessParticles,\
    XmippProcessVolumes
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippPreprocessHelper:
    """ 
    Helper class that contains some Protocol utilities methods
    used by both  XmippProtPreprocessParticles and XmippProtPreprocessVolumes.
    """
    
    #--------------------------- DEFINE param functions ------------------------
    @classmethod
    def _defineProcessParams(cls, form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        # Invert Contrast
        form.addParam('doInvert', BooleanParam, default=False,
                      label='Invert contrast',
                      help='Invert the contrast if your particles are black over'
                           'a white background.')
        # Threshold
        form.addParam('doThreshold', BooleanParam, default=False,
                      label="Threshold",
                      help='Remove voxels below a certain value.')
        form.addParam('thresholdType', EnumParam, condition='doThreshold',
                      choices=['abs_below', 'below', 'above'],
                      default=MASK_FILL_VALUE,
                      label="Fill with ", display=EnumParam.DISPLAY_COMBO,
                      help='Select how are you going to fill the pixel values'
                           'outside the mask.')
        form.addParam('threshold', FloatParam, default=0,
                      label="Threshold value", condition='doThreshold',
                      help='Grey value below which all voxels should be set to 0.')
        form.addParam('fillType', EnumParam, condition='doThreshold',
                      choices=['value', 'binarize', 'avg'],
                      default=FILL_VALUE,
                      label="Substitute by", display=EnumParam.DISPLAY_COMBO,
                      help='If you select: value: Selected are substitute by a desired value.'
                           '            binarize: Selected are set to 0, non-selected to 1.'
                           '                 avg: Average of non-selected.')
        form.addParam('fillValue', IntParam, default=0,
                      condition='doThreshold and fillType == %d' % FILL_VALUE,
                      help=' Substitute selected pixels by this value.',
                      label='Fill value')

    #--------------------------- INSERT steps functions ------------------------
    @classmethod
    def _insertCommonSteps(cls, protocol, changeInserts):
        if protocol.doInvert:
            args = protocol._argsInvert()
            protocol._insertFunctionStep("invertStep", args, changeInserts)

        if protocol.doThreshold:
            args = protocol._argsThreshold()
            protocol._insertFunctionStep("thresholdStep", args, changeInserts)

    #--------------------------- UTILS functions -------------------------------
    @classmethod
    def _argsCommonInvert(cls):
        args = ' --mult -1'
        return args
    
    @classmethod
    def _argsCommonThreshold(cls, protocol):
        args = " --select %s %f" % (protocol.getEnumText('thresholdType'),
                                    protocol.threshold)
        fillStr = protocol.getEnumText('fillType')
        args += " --substitute %s " % fillStr
        
        if protocol.fillType == MASK_FILL_VALUE:
            args += " %f" % protocol.fillValue
        return args
    

class XmippProtPreprocessParticles(XmippProcessParticles):
    """Preprocesses particle images by applying the optional steps:  dust
    removal, randomize phase, normalize, center images, phase flip images,
    invert contrast, threshold, fill with below, abs_bellow or above, a
    threshold grey value below which all voxels should be set to 0, fill
    value or substitute by value binarize or  average. This cleaning stage
    improves particle quality and consistency for downstream tasks.

    AI Generated

    ## Overview

    The Preprocess Particles protocol applies a sequence of optional preprocessing
    operations to a set of particle images.

    Particle preprocessing is often used to clean images, standardize contrast,
    normalize background statistics, remove extreme pixel values, phase-flip CTF
    effects, or prepare particles for later alignment, classification, and
    reconstruction.

    The protocol can apply several operations, including:

    - dust removal;
    - phase randomization beyond a selected resolution;
    - normalization;
    - image centering;
    - phase flipping;
    - contrast inversion;
    - thresholding.

    The selected operations are applied in the order defined by the protocol. The
    main output is a new set of preprocessed particles.

    ## Inputs and General Workflow

    The input is a set of particles.

    The protocol converts the input particle set to Xmipp metadata format. It then
    runs the selected preprocessing operations sequentially. Each operation uses
    the output of the previous one, so the order of selected operations matters.

    The output particle set preserves the input metadata when possible and points
    to the processed particle images.

    ## Input Particles

    The **Input particles** parameter defines the particle set to be preprocessed.

    The protocol does not change the biological identity of the particles. It
    changes the image values and, for phase flipping, updates the phase-flipped
    status of the output set.

    The output particles can be used in downstream protocols such as 2D
    classification, alignment, reconstruction, screening, or additional cleaning.

    ## Dust Removal

    The **Dust removal** option detects pixels with unusually large absolute values
    and replaces them with random values from a Gaussian distribution with zero
    mean and unit standard deviation.

    This is useful for removing isolated very bright or very dark artifacts, such
    as detector spikes, hot pixels, or other extreme outliers.

    The **Threshold for dust removal** parameter controls how extreme a pixel must
    be to be treated as dust. Pixels with signal higher or lower than this value
    times the image standard deviation are affected.

    The default value, 3.5, is suggested for cryo-EM. For high-contrast negative
    stain images, a higher value may be preferable because real signal can be much
    stronger.

    ## Randomize Phases

    The **Randomize phases** option randomizes Fourier phases beyond a selected
    resolution.

    The **Maximum Resolution** parameter defines the resolution, in angstroms,
    beyond which phases are randomized.

    This operation is useful for control experiments or for removing high-frequency
    phase information beyond a chosen limit. It should not be used as routine
    particle cleaning unless the user has a specific validation or preprocessing
    reason.

    ## Normalize

    The **Normalize** option standardizes particle intensity values.

    The protocol supports three normalization modes:

    **OldXmipp** sets the whole-image mean to 0 and standard deviation to 1.

    **NewXmipp** sets the background mean to 0 and background standard deviation to
    1.

    **Ramp** subtracts a background ramp and then applies the NewXmipp background
    normalization.

    Normalization is useful because many downstream algorithms assume comparable
    particle intensity statistics.

    ## Background Radius

    The **Background radius** parameter is used for NewXmipp and Ramp
    normalization.

    Pixels outside this circle are treated as background. Their statistics are used
    to normalize the particle images. If the radius is less than or equal to 0, the
    protocol uses half the particle box size.

    The radius should be chosen so that the background region does not include
    significant particle density. If the radius is too small or too large,
    background statistics may be biased.

    The protocol validates that the background radius is not larger than the
    particle half-size.

    ## Center Images

    The **Center images** option recenters particle images using the Xmipp image
    centering tool.

    This can be useful when particles are approximately centered but still show
    small systematic shifts.

    Centering should be used carefully. If particles contain multiple components,
    strong background artifacts, or highly asymmetric density, automatic centering
    may not correspond to the desired biological center.

    ## Phase Flip Images

    The **Phase flip images** option applies CTF phase flipping to the particle
    images.

    The protocol uses the particle sampling rate and the CTF information associated
    with the input particles.

    Phase flipping changes the phase-flipped status of the output particle set.
    The output set is marked as phase flipped if the input was not phase flipped,
    and vice versa.

    This option is useful when the user wants to correct CTF phase inversions while
    keeping a relatively simple CTF treatment.

    ## Invert Contrast

    The **Invert contrast** option multiplies particle images by -1.

    This is useful when the particles have the opposite contrast convention from
    what downstream protocols expect. For example, it can convert black particles
    on a white background into white particles on a darker background, or the other
    way around.

    Incorrect contrast inversion can make later processing fail, so the output
    should be inspected visually.

    ## Threshold

    The **Threshold** option replaces selected pixel values according to a
    threshold rule.

    The protocol first selects pixels using one of three modes:

    **abs_below** selects pixels whose absolute value is below the threshold.

    **below** selects pixels below the threshold.

    **above** selects pixels above the threshold.

    Then it substitutes the selected pixels using the selected substitution mode.

    ## Threshold Value

    The **Threshold value** parameter defines the gray-value cutoff used by the
    threshold operation.

    The meaning of this value depends on the selected threshold type. For example,
    in **below** mode, pixels below this value are selected. In **above** mode,
    pixels above this value are selected.

    Thresholding can be useful for cleaning or binarization-like operations, but it
    can also remove weak signal if used too aggressively.

    ## Substitute By

    The **Substitute by** parameter controls how selected pixels are replaced.

    The options are:

    **value**, where selected pixels are replaced by the user-provided fill value;

    **binarize**, where selected and non-selected pixels are converted into a
    binary representation;

    **avg**, where selected pixels are replaced by the average of the non-selected
    pixels.

    This parameter determines whether thresholding behaves as intensity clipping,
    mask-like binarization, or background substitution.

    ## Fill Value

    The **Fill value** parameter is used when **Substitute by = value**.

    It defines the numerical value assigned to selected pixels.

    A common value is 0, but other values may be useful depending on the image
    normalization and background convention.

    ## Output Particles

    The main output is **outputParticles**.

    This output contains the preprocessed particle images after all selected
    operations have been applied.

    The output particle set can be used directly in downstream Scipion protocols.
    When phase flipping is applied, the phase-flipped state of the output set is
    updated accordingly.

    ## Operation Order

    The protocol applies operations in a defined order:

    1. dust removal;
    2. phase randomization;
    3. normalization;
    4. centering;
    5. phase flipping;
    6. contrast inversion;
    7. thresholding.

    This order matters. For example, inverting contrast before thresholding may
    produce a different result than thresholding before inversion. Users should
    therefore select operations with the intended sequence in mind.

    ## Interpreting the Result

    The output particles should be interpreted as processed versions of the input
    particles.

    Preprocessing can improve consistency, remove artifacts, and prepare images
    for later algorithms. However, it can also remove useful signal or introduce
    bias if parameters are inappropriate.

    The processed particles should be inspected visually, especially when using
    thresholding, contrast inversion, phase randomization, or aggressive dust
    removal.

    ## Practical Recommendations

    Use dust removal to suppress isolated extreme pixels.

    Use normalization for most particle-processing workflows, especially before
    classification or alignment.

    Use Ramp normalization when background gradients are visible.

    Use phase flipping only when CTF metadata are reliable and when the workflow
    requires phase-flipped particles.

    Use contrast inversion only after confirming the expected contrast convention.

    Use thresholding cautiously and inspect the output.

    Apply only the operations that are needed. Avoid unnecessary preprocessing
    that may alter the particle signal.

    ## Final Perspective

    Preprocess Particles is a general particle-cleaning and normalization protocol.

    For biological users, its value is that it prepares particle images for later
    processing by standardizing intensity, correcting contrast conventions,
    removing extreme artifacts, and optionally applying phase flipping or
    thresholding.

    The protocol is a preprocessing utility. Its output should improve technical
    consistency, but it should always be checked before being used in major
    downstream steps.
    """
    _label = 'preprocess particles'
    _devStatus = PROD
    _possibleOutputs = {"outputParticles": SetOfParticles}

    # Normalization enum constants
    NORM_OLD = 0
    NORM_NEW = 1
    NORM_RAMP =2

    def __init__(self, **kwargs):
        XmippProcessParticles.__init__(self, **kwargs)
    
    #--------------------------- DEFINE param functions ------------------------
    def _defineProcessParams(self, form):
        form.addParam('doRemoveDust', BooleanParam, default=False,
                      label='Dust removal', 
                      help='Sets pixels with unusually large values to'
                           'random values from a Gaussian '
                           'with zero-mean and unity-standard deviation.')
        form.addParam('thresholdDust', FloatParam, default=3.5,
                      condition='doRemoveDust', expertLevel=LEVEL_ADVANCED,
                      label='Threshold for dust removal',
                      help='Pixels with a signal higher or lower than'
                           'this value times the standard deviation of the image'
                           'will be affected. For cryo, 3.5 is a good value.'
                           'For high-contrast negative stain, the signal itself'
                           'may be affected so that a higher value may be preferable.')
        form.addParam('doRandomize', BooleanParam, default=False,
                      label="Randomize phases",
                      help='Randomize phases beyond a certain frequency.')
        form.addParam('maxResolutionRandomize', FloatParam, default=40,
                      label="Maximum Resolution", condition='doRandomize',
                      help='Angstroms.')
        form.addParam('doNormalize', BooleanParam, default=False,
                      label='Normalize', 
                      help='It subtracts a ramp in the gray values and normalizes '
                           'so that in the background there is 0 mean and '
                           'standard deviation 1.')
        form.addParam('normType', EnumParam, condition='doNormalize',
                      label='Normalization type', expertLevel=LEVEL_ADVANCED,
                      choices=['OldXmipp','NewXmipp','Ramp'],
                      default=self.NORM_RAMP,display=EnumParam.DISPLAY_COMBO, 
                      help='OldXmipp: mean(Image)=0, stddev(Image)=1\n'
                           'NewXmipp: mean(background)=0, stddev(background)=1\n'
                           'Ramp: subtract background + NewXmipp')
        form.addParam('backRadius', IntParam, default=-1, condition='doNormalize',
                      label='Background radius', expertLevel=LEVEL_ADVANCED,
                      help='Pixels outside this circle are assumed to be noise and '
                           'their stddev is set to 1. Radius for background '
                           'circle definition (in pix.). '
                           'If this value is less than or equal to 0, then half the box size is used.')
        form.addParam('doCenter', BooleanParam, default=False,
                      label='Center images')
        form.addParam('doPhaseFlip', BooleanParam, default=False,
                      label='Phase flip images')
        XmippPreprocessHelper._defineProcessParams(form)

    #--------------------------- INSERT steps functions ------------------------
    def _insertProcessStep(self):
        self.isFirstStep = True
        # this is for when the options selected has changed and the protocol is resumed
        changeInserts = [self.doRemoveDust.get(), self.doNormalize.get(), self.doInvert.get(),
                         self.doThreshold.get(), self.doCenter.get(), self.doPhaseFlip.get()]
        
        if self.doRemoveDust:
            args = self._argsRemoveDust()
            self._insertFunctionStep("removeDustStep", args, changeInserts)

        if self.doRandomize:
            args = self._argsRandomize()
            self._insertFunctionStep("randomizeStep", args, changeInserts)

        if self.doNormalize:
            args = self._argsNormalize()
            self._insertFunctionStep("normalizeStep", args, changeInserts)
        
        if self.doCenter:
            args = self._argsCenter()
            self._insertFunctionStep("centerStep", args, changeInserts)

        if self.doPhaseFlip:
            args = self._argsPhaseFlip()
            self._insertFunctionStep("phaseFlipStep", args, changeInserts)

        XmippPreprocessHelper._insertCommonSteps(self, changeInserts)
        
    #--------------------------- STEPS functions -------------------------------
    def invertStep(self, args, changeInserts):
        self.runJob('xmipp_image_operate', args)

    def randomizeStep(self, args, changeInserts):
        self.runJob("xmipp_transform_randomize_phases", args)

    def thresholdStep(self, args, changeInserts):
        self.runJob("xmipp_transform_threshold", args)
        
    def removeDustStep(self, args, changeInserts):
        self.runJob('xmipp_transform_filter', args)
    
    def normalizeStep(self, args, changeInserts):
        self.runJob("xmipp_transform_normalize", args)
    
    def centerStep(self, args, changeInserts):
        self.runJob("xmipp_transform_center_image", args % locals())

    def phaseFlipStep(self, args, changeInserts):
        self.runJob("xmipp_ctf_correct_phase", args % locals())

    def sortImages(self, outputFn, outputMd):
        pass

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        validateMsgs = []
        
        if self.doNormalize.get() and self.normType.get() != 0:
            size = self._getSize()
            if self.backRadius.get() > size:
                validateMsgs.append('Set a valid Background radius less than %d'
                                                                         % size)
        return validateMsgs
    
    def _summary(self):
        summary = []
        summary.append("Input particles: %s"
                        % self.inputParticles.get().getFileName())
        
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            summary.append("Dust removal: %s" % self.doRemoveDust)
            summary.append("Normalize the background: %s" % self.doNormalize)
            summary.append("Invert contrast: %s" % self.doInvert)
            summary.append("Remove voxels with threshold: %s" %self.doThreshold)
        return summary
    
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("Input particles %s of %s elements"
                            % (self.getObjectTag('inputParticles'),
                               self.inputParticles.get().getSize()))
            if self.doNormalize:
                methods.append("The background was normalized with %s method."
                                % self.getEnumText('normType'))
            if self.doInvert:
                methods.append("The contrast was inverted")
            if self.doThreshold:
                methods.append("Pixels with values below %f was removed"
                               % self.threshold.get())
            methods.append('Output set: %s'%self.getObjectTag('outputParticles'))
        return methods
    
    #--------------------------- UTILS functions -------------------------------
    def _argsRemoveDust(self):
        if self.isFirstStep:
            args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                   % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        args += " --bad_pixels outliers %f" % self.thresholdDust.get()
        return args

    def _argsRandomize(self):
        if self.isFirstStep:
            args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                   % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        samplingRate = self.inputParticles.get().getSamplingRate()
        args+= " --freq continuous %f %f"%(float(self.maxResolutionRandomize.get()),samplingRate)
        return args

    def _argsNormalize(self):
        if self.isFirstStep:
            args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                   % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        
        normType = self.normType.get()
        bgRadius = self.backRadius.get()
        radii = self._getSize()
        if bgRadius <= 0:
            bgRadius = int(radii)

        if normType == self.NORM_OLD:
            args += " --method OldXmipp"
        elif normType == self.NORM_NEW:
            args += " --method NewXmipp --background circle %d" % bgRadius
        else:
            args += " --method Ramp --background circle %d" % bgRadius
        return args
    
    def _argsInvert(self):
        if self.isFirstStep:
            args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                   % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        args += XmippPreprocessHelper._argsCommonInvert()
        return args
    
    def _argsThreshold(self):
        if self.isFirstStep:
            args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                   % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        args += XmippPreprocessHelper._argsCommonThreshold(self)
        return args
    
    def _argsCenter(self):
        if self.isFirstStep:
            args = "-i %s -o %s --save_metadata_stack %s" \
                   % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        return args

    def _argsPhaseFlip(self):
        if self.isFirstStep:
            args = "-i %s -o %s --save_metadata_stack %s" \
                   % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputMd
        args+=" --sampling_rate %f"%self.inputParticles.get().getSamplingRate()
        return args

    def _getSize(self):
        """ get the size of SetOfParticles object"""
        Xdim = self.inputParticles.get().getDimensions()[0]
        size = int(Xdim/2)
        return size
    
    def _setFalseFirstStep(self):
        if self.isFirstStep:
                self.isFirstStep = False

    def _postprocessOutput(self, outputSet):
        if self.doPhaseFlip.get():
            outputSet.setIsPhaseFlipped(not self.inputParticles.get().isPhaseFlipped())


class XmippProtPreprocessVolumes(XmippProcessVolumes):
    """Preprocesses 3D volumes using Xmipp tools to prepare them for further
    analysis. Operations include: normalization, change hand, change
    icosahedral orientation, randomize phase, symmetry, symmetry group,
    aggregation mode, wrap, apply Laplacian, mask volume. adjust gray value,
    segment, normalize background, invert contrast and threshold.

    AI Generated

    ## Overview

    The Preprocess Volumes protocol applies a sequence of optional preprocessing
    operations to one volume or to a set of volumes.

    Volume preprocessing is useful for changing map handedness, changing
    icosahedral conventions, randomizing phases, applying symmetry, denoising,
    adjusting gray levels, segmenting the molecule, normalizing the background,
    inverting contrast, or thresholding voxel values.

    The protocol can be used as a general preparation tool before map comparison,
    alignment, validation, visualization, masking, subtraction, or refinement.

    The main output is a processed volume or processed set of volumes.

    ## Inputs and General Workflow

    The input can be a single volume or a set of volumes.

    The selected operations are applied sequentially. Each operation acts on the
    result of the previous operation. For volume sets, the operation is applied to
    each volume in the set whenever applicable.

    The protocol writes the processed volume or volume set and registers it as the
    output.

    ## Input Volumes

    The **Input volumes** parameter defines the map or maps to be processed.

    The input may be a single volume or a set of volumes.

    The protocol does not decide which operations are biologically appropriate. The
    user selects the intended preprocessing steps and should inspect the result
    afterwards.

    ## Change Hand

    The **Change hand** option applies a mirror transformation along the X axis.

    This changes the handedness of the map.

    Changing hand is useful when a reconstruction is known to have the wrong
    handedness or when comparing maps that use different hand conventions. It
    should not be applied casually, because it changes the chirality of the
    structure.

    ## Change Icosahedral Orientation

    The **Change icosahedral orientation** option converts a volume from one
    standard icosahedral orientation convention to another.

    The user selects the source convention in **from** and the target convention in
    **to**. The available conventions are:

    - i1;
    - i2;
    - i3;
    - i4.

    This option is useful when maps with icosahedral symmetry need to be converted
    between Xmipp-supported symmetry conventions.

    ## Randomize Phases

    The **Randomize phases** option randomizes Fourier phases beyond a selected
    resolution.

    The **Maximum Resolution** parameter defines the resolution, in angstroms,
    beyond which phases are randomized.

    This operation can be useful for control experiments or for suppressing
    high-frequency phase information beyond a chosen limit. It should be used with
    a clear validation or preprocessing purpose.

    ## Symmetrize

    The **Symmetrize** option applies a symmetry group to the input volume.

    The **Symmetry group** parameter defines the Xmipp symmetry to apply. The user
    should provide a valid symmetry group, such as a cyclic, dihedral, or
    icosahedral group. If no symmetry should be applied, the Symmetrize option
    should be disabled rather than setting the group to c1.

    The protocol validates that c1 is not used as the symmetry group when
    symmetrization is requested.

    ## Aggregation Mode

    The **Aggregation mode** parameter controls how symmetrized copies are combined.

    There are two options:

    **Average** averages the symmetry-related copies.

    **Sum** sums them.

    Average is usually appropriate when the goal is to produce a symmetrized map
    with comparable intensity scale. Sum may be useful in technical workflows where
    the accumulated density is desired.

    ## Wrap

    The **Wrap** option controls whether density is wrapped around the box during
    symmetrization.

    When enabled, wrapping is allowed. When disabled, the protocol uses a
    non-wrapping behavior.

    Disabling wrap can help avoid artificial density appearing on the opposite side
    of the box when transformed density crosses a boundary.

    ## Mask Volume for Symmetrization or Laplacian

    The **Mask volume** parameter can be used with symmetrization or Laplacian
    filtering.

    For symmetrization, the mask can restrict the region used during the operation.
    For Laplacian filtering, it can provide a mask for the denoising step.

    The mask should be in the same coordinate frame as the input volume.

    ## Apply Laplacian

    The **Apply Laplacian** option applies a Laplacian-like denoising operation
    using Xmipp filtering.

    The protocol uses a Retinex-style parameter internally and can optionally use a
    volume mask.

    This operation is intended to enhance or denoise volume features, but it should
    be inspected carefully because filtering can alter map appearance.

    ## Adjust Gray Values

    The **Adjust gray values** option adjusts the volume gray values so that they
    are compatible with a set of projection images.

    The user provides **Set of particles**, which may be a set of particles, a set
    of averages, or a set of 2D classes. The protocol uses up to 200 images for the
    adjustment.

    If the images already have projection alignment, those alignments are used. If
    not, the protocol estimates orientations against the input volume using
    significant alignment, with CPU or GPU execution depending on the settings.

    The adjustment then modifies volume gray levels according to the image set.

    ## Images for Gray-Value Adjustment

    The **Set of particles** parameter provides the images used for gray-level
    adjustment.

    These images should have the final pixel size and final image size expected
    for the model.

    The image set should correspond to projections of the same structure. If the
    images are unrelated, poorly aligned, or strongly heterogeneous, the gray-level
    adjustment may be unreliable.

    ## Symmetry Group for Significant Alignment

    The **Symmetry group** parameter under gray-value adjustment defines the
    symmetry used when assigning orientations to the input images.

    If no symmetry is present, use **c1**.

    This setting is used only when the input images do not already have projection
    alignment and the protocol must estimate orientations for the adjustment step.

    ## GPU Execution for Significant Alignment

    The protocol includes hidden GPU settings for the significant-alignment step
    used during gray-value adjustment.

    If GPU execution is requested, the protocol checks that the required Xmipp CUDA
    programs are available. If they are not found, validation reports an error.

    GPU execution can accelerate orientation assignment when adjustment images do
    not already have projection alignment.

    ## Segment

    The **Segment** option separates the molecule from the background.

    The protocol first creates a mask using the selected segmentation method, then
    applies that binary mask to the volume.

    Segmentation is useful when the user wants to remove background or keep only
    the molecular region.

    ## Segmentation Type

    The **Segmentation Type** parameter controls how the segmentation mask is
    created.

    The available options are:

    **Voxel mass**, where the user provides the target number of voxels.

    **Aminoacid mass**, where the user provides an approximate number of amino
    acids.

    **Dalton mass**, where the user provides the molecular mass in daltons.

    **Automatic**, where the protocol uses Otsu thresholding.

    The **Molecule Mass** parameter is used for all segmentation types except
    Automatic.

    ## Normalize Background

    The **Normalize background** option normalizes the volume background so that it
    has zero mean and standard deviation 1.

    The **Mask Radius** parameter defines the radius, in pixels, used to identify
    the region outside the molecule as background. If it is set to -1, the protocol
    uses half the size of the volume.

    The protocol validates that the radius is not larger than the allowed volume
    half-size.

    ## Invert Contrast

    The **Invert contrast** option multiplies the volume values by -1.

    This can be useful when the map contrast convention is opposite to what a
    downstream protocol expects.

    Because it changes the sign of density, it should be used only when the
    contrast convention is clearly known.

    ## Threshold

    The **Threshold** option replaces selected voxel values according to a
    threshold rule.

    The selection can be:

    **abs_below**, selecting voxels whose absolute value is below the threshold;

    **below**, selecting voxels below the threshold;

    **above**, selecting voxels above the threshold.

    Selected voxels are then substituted according to the chosen substitution mode.

    ## Substitute By

    The **Substitute by** parameter controls how threshold-selected voxels are
    replaced.

    The options are:

    **value**, replacing selected voxels with the user-provided fill value;

    **binarize**, converting selected and non-selected voxels into a binary
    representation;

    **avg**, replacing selected voxels by the average of the non-selected voxels.

    This option determines whether thresholding behaves as clipping, binarization,
    or background replacement.

    ## Operation Order

    The selected operations are applied in the following order:

    1. change hand;
    2. change icosahedral orientation;
    3. randomize phases;
    4. symmetrize;
    5. Laplacian filtering;
    6. gray-value adjustment;
    7. segmentation;
    8. background normalization;
    9. contrast inversion;
    10. thresholding.

    The order matters. For example, segmenting before normalizing may produce a
    different result than normalizing before segmenting.

    ## Output Volumes

    The main output is the processed volume or volume set.

    For a single input volume, the output is one processed map. For a set of
    volumes, the output contains the processed version of each input map.

    The output can be used in downstream protocols such as alignment, comparison,
    masking, filtering, subtraction, validation, or visualization.

    ## Validation Rules

    The protocol checks that an input volume or volume set has been provided.

    If background normalization is selected, the background radius must be valid
    for the input volume size.

    If symmetrization is selected, the symmetry group must not be c1. To avoid
    symmetrization, the user should disable the Symmetrize option.

    If GPU execution is requested for significant alignment and the required CUDA
    programs are not available, the protocol reports a validation error.

    ## Interpreting the Result

    The output should be interpreted as a processed version of the input volume or
    volumes.

    Some operations are purely geometrical, such as changing hand or icosahedral
    orientation. Others change intensities, such as normalization, gray-value
    adjustment, thresholding, or contrast inversion. Others change structural
    support, such as segmentation or masking.

    Because these operations can strongly affect map appearance, the output should
    always be inspected together with the original input.

    ## Practical Recommendations

    Use change hand only when the map handedness is known to be wrong.

    Use icosahedral orientation conversion only for maps with icosahedral symmetry
    and known convention differences.

    Use symmetrization only when symmetry is biologically justified.

    Use gray-value adjustment when a map must be made compatible with a set of
    projection images.

    Use segmentation and thresholding cautiously, especially when weak or flexible
    density is important.

    Use background normalization to standardize maps before downstream analysis.

    Inspect the output after each major preprocessing workflow. When uncertain,
    run separate preprocessing protocols for individual operations so that their
    effects can be checked independently.

    ## Final Perspective

    Preprocess Volumes is a general map-preparation protocol.

    For biological users, its value is that it gathers several common volume
    preprocessing operations in one place: changing hand, changing symmetry
    orientation, randomizing phases, symmetrizing, denoising, adjusting gray
    values, segmenting, normalizing, inverting contrast, and thresholding.

    The protocol is powerful but should be used carefully. Each selected operation
    changes the map in a specific way, and biological interpretation should always
    be based on the processed output together with the original map and the
    processing context.
    """
    _label = 'preprocess volumes'
    _devStatus = PROD
    # Aggregation constants
    AGG_AVERAGE=0
    AGG_SUM=1
    
    # Segmentation type
    SEG_VOXEL=0
    SEG_AMIN=1
    SEG_DALTON=2
    SEG_AUTO=3


    def __init__(self, **kwargs):
        XmippProcessVolumes.__init__(self, **kwargs)
    
    #--------------------------- DEFINE param functions ------------------------
    def _defineProcessParams(self, form):
        # Change hand
        form.addParam('doChangeHand', BooleanParam, default=False,
                      label="Change hand", 
                      help='Change hand by applying a mirror along X.')
        # Change from one icosahedral standard orientation to another
        form.addParam('doRotateIco', BooleanParam, default=False,
                      label="Change icosahedral orientation", 
                      help='Change from one icosahedral standard orientation to another.')
        form.addParam('rotateFromIco', EnumParam, choices=['i1','i2','i3','i4'],
                       display=EnumParam.DISPLAY_COMBO, default=0,
                       label='from', condition='doRotateIco', 
                       help='See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]] '
                            'for a description of the symmetry groups format.')
        form.addParam('rotateToIco', EnumParam, choices=['i1','i2','i3','i4'],
                       label='to', default=1, display=EnumParam.DISPLAY_COMBO,
                       condition='doRotateIco')
        # Randomize the phases
        form.addParam('doRandomize', BooleanParam, default=False,
                      label="Randomize phases", 
                      help='Randomize phases beyond a certain frequency.')
        # ToDo: add wizard
        form.addParam('maxResolutionRandomize', FloatParam, default=40,
                      label="Maximum Resolution", condition='doRandomize',
                      help='Angstroms.')
        # Symmetrization
        form.addParam('doSymmetrize', BooleanParam, default=False,
                      label="Symmetrize", 
                      help='Symmetrize the input model.')
        form.addParam('symmetryGroup', StringParam, default='i1',
                      label="Symmetry group", condition='doSymmetrize',
                      help='See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]] '
                      'for a description of the symmetry groups format.'
                      'If no symmetry is present, set the Symmetrize field to not.')
        form.addParam('aggregation', EnumParam, choices=['Average', 'Sum'], 
                      display=EnumParam.DISPLAY_COMBO, condition = 'doSymmetrize',
                      default=0, label='Aggregation mode', 
                      help='Symmetrized volumes can be averaged or summed.')
        form.addParam('doWrap', BooleanParam, default=True,
                      label="Wrap", condition='doSymmetrize',
                      help='by default, the image/volume is wrapped')
        # Filtering
        form.addParam('doLaplacian', BooleanParam, default=False,
                      help="Laplacian denoising", label="Apply Laplacian")
        form.addParam('volumeMask', PointerParam, pointerClass='VolumeMask',
                      label='Mask volume', condition='doSymmetrize or doLaplacian',
                      allowsNull=True)

        # Adjust gray values
        form.addParam('doAdjust', BooleanParam, default=False,
                      label="Adjust gray values", 
                      help='Adjust input gray values so that it is compatible'
                           'with a set of projections.') 
        form.addParam('inputImages', PointerParam,
                      pointerClass='SetOfParticles, SetOfAverages, SetOfClasses2D',
                      label="Set of particles", condition='doAdjust',
                      help='Set of images to which the model should conform.'
                           'The set of images should have the final pixel size'
                           'and the final size of the model.')
        form.addParam('sigSymGroup', TextParam, default='c1',
                      label="Symmetry group", condition='doAdjust',
                      help='See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]]'
                           'for a description of the symmetry groups format.'
                           'If no symmetry is present, give c1.')  
        # Segment
        form.addParam('doSegment', BooleanParam,
                      default=False, label="Segment", 
                      help='Separate the molecule from its background.')
        form.addParam('segmentationType', EnumParam, condition='doSegment',
                      choices=['Voxel mass', 'Aminoacid mass','Dalton mass','Automatic'],
                      default=3, display=EnumParam.DISPLAY_COMBO,
                      label="Segmentation Type", help='Type of segmentation.')
        form.addParam('segmentationMass', FloatParam, label="Molecule Mass",
                      default=-1, condition='doSegment and segmentationType != 3',
                      help='In automatic segmentation, set it to -1.')
        # Normalize background
        form.addParam('doNormalize', BooleanParam, default=False,
                      label="Normalize background", 
                      help='Set background to have zero mean and standard deviation 1.')
        form.addParam('backRadius', FloatParam, default=-1,
                      label="Mask Radius", condition='doNormalize',
                      help='In pixels. Set to -1 for half of the size of the volume.')
        XmippPreprocessHelper._defineProcessParams(form)

    #--------------------------- INSERT steps functions ------------------------
    def _insertProcessStep(self):
        self.isFirstStep = True
        # this is for when the options selected has changed and the protocol is resumed
        changeInserts = [self.doChangeHand, self.doRotateIco, self.doRandomize, 
                         self.doSymmetrize, self.doLaplacian, self.doAdjust, 
                         self.doSegment, self.doInvert, self.doNormalize, 
                         self.doThreshold]

        if self.doChangeHand:
            args = self._argsChangeHand()
            self._insertFunctionStep("changeHandStep", args, changeInserts)

        if self.doRotateIco:
            args = self._argsRotateIco()
            self._insertFunctionStep("rotateIcoStep", args, changeInserts)   
        
        if self.doRandomize:
            args = self._argsRandomize()
            self._insertFunctionStep("randomizeStep", args, changeInserts)
        
        if self.doSymmetrize:
            args = self._argsSymmetrize()
            self._insertFunctionStep("symmetrizeStep", args, changeInserts)
        
        if self.doLaplacian:
            args = self._argsLaplacian()
            self._insertFunctionStep("laplacianStep", args, changeInserts)
        
        if self.doAdjust:
            self._insertFunctionStep("projectionStep", changeInserts)
            self._insertFunctionStep("adjustStep",self.isFirstStep,changeInserts)
            if self.isFirstStep:
                self.isFirstStep = False

        if self.doSegment:
            args = self._argsSegment()
            self._insertFunctionStep("segmentStep", args, changeInserts)
            if self.isFirstStep:
                self.isFirstStep = False
        
        if self.doNormalize:
            args = self._argsNormalize()
            self._insertFunctionStep("normalizeStep", args, changeInserts)
        
        XmippPreprocessHelper._insertCommonSteps(self, changeInserts)

    #--------------------------- STEPS functions -------------------------------
    def invertStep(self, args, changeInserts):
        self.runJob('xmipp_image_operate', args)
    
    def thresholdStep(self, args, changeInserts):
        self.runJob("xmipp_transform_threshold", args)
        
    def removeDustStep(self, args, changeInserts):
        self.runJob('xmipp_transform_filter', args)
    
    def normalizeStep(self, args, changeInserts):
        self.runJob("xmipp_transform_normalize", args)
        
    def changeHandStep(self, args, changeInserts):
        self.runJob("xmipp_transform_mirror", args)

    def rotateIcoStep(self, args, changeInserts):
        self.runJob("xmipp_transform_geometry", args)  
    
    def randomizeStep(self, args, changeInserts):
        self.runJob("xmipp_transform_randomize_phases", args)
    
    def symmetrizeStep(self, args, changeInserts):
        self.runJob("xmipp_transform_symmetrize", args)
    
    def laplacianStep(self, args, changeInserts):
        self.runJob("xmipp_transform_filter", args, numberOfMpi=1)
    
    def projectionStep(self, changeInserts):
        partSet = self.inputImages.get()
        imgsFn = self._getTmpPath('input_images.xmd')
        
        if partSet.getSize() > 200:
            newPartSet = self._getRandomSubset(partSet, 200)
        else:
            newPartSet = partSet
            
        writeSetOfParticles(newPartSet, imgsFn)
        
        if not partSet.hasAlignmentProj():
            if not self.useGpu.get():
                params = {'imgsFn': imgsFn,
                          'dir': self._getTmpPath(),
                          'vols': self.inputFn,
                          'symmetryGroup': self.sigSymGroup.get(),
                          }
                sigArgs = '-i %(imgsFn)s --initvolumes %(vols)s --odir %(dir)s' \
                          ' --sym %(symmetryGroup)s --alpha0 0.005 --dontReconstruct ' \
                          % params
                self.runJob("xmipp_reconstruct_significant", sigArgs)
            else:
                fnGallery = self._getExtraPath('gallery.stk')
                fnGalleryMd = self._getExtraPath('gallery.doc')
                angleStep = 5
                args = "-i %s -o %s --sampling_rate %f --sym %s --min_tilt_angle 0 --max_tilt_angle 180 " % \
                       (self.inputFn, fnGallery, angleStep,
                        self.sigSymGroup.get())
                self.runJob("xmipp_angular_project_library", args,
                            numberOfMpi=min(self.numberOfMpi.get(), 24))

                count=0
                GpuListCuda=''
                if self.useQueueForSteps() or self.useQueue():
                    GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                    GpuList = GpuList.split(",")
                    for elem in GpuList:
                        GpuListCuda = GpuListCuda+str(count)+' '
                        count+=1
                else:
                    GpuListAux = ''
                    for elem in self.getGpuList():
                        GpuListCuda = GpuListCuda+str(count)+' '
                        GpuListAux = GpuListAux+str(elem)+','
                        count+=1
                    os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux

                fnAngles = 'images_iter001_00.xmd'
                args = '-i %s -r %s -o %s --odir %s --keepBestN 1 --dev %s ' % (
                imgsFn, fnGalleryMd, fnAngles, self._getTmpPath(), GpuListCuda)
                self.runJob(CUDA_ALIGN_SIGNIFICANT, args, numberOfMpi=1)

    def adjustStep(self, isFirstStep, changeInserts):
        if isFirstStep:
            inputFn = self.inputFn
        else:
            if self._isSingleInput():
                inputFn = self.outputStk
            else:
                inputFn = self.outputMd

        if self._isSingleInput():
            args = self._argsAdjust(0)
            localArgs = self._adjustLocalArgs(inputFn, self.outputStk, args)
            self.runJob("xmipp_transform_adjust_volume_grey_levels", localArgs)
        else:
            volMd = emlib.MetaData(self.inputFn)
            outVolMd = emlib.MetaData(self.outputMd)
            for objId in volMd:
                args = self._argsAdjust(objId-1)
                inputVol = volMd.getValue(emlib.MDL_IMAGE, objId)
                outputVol = outVolMd.getValue(emlib.MDL_IMAGE, objId)
                localArgs = self._adjustLocalArgs(inputVol, outputVol, args)
                self.runJob("xmipp_transform_adjust_volume_grey_levels", localArgs)
    
    def segmentStep(self, args, changeInserts):
        fnMask = self._getTmpPath("mask.vol")
        if self.isFirstStep:
            inputFn = self.inputFn
        else:
            if self._isSingleInput():
                inputFn = self.outputStk
            else:
                inputFn = self.outputMd
        
        if self._isSingleInput():
            localArgs = self._segmentLocalArgs(inputFn, fnMask, args)
            maskArgs = self._segMentMaskArgs(inputFn, self.outputStk, fnMask)
            self._segmentVolume(localArgs, maskArgs, fnMask)
        else:
            volMd = emlib.MetaData(inputFn)
            outVolMd = emlib.MetaData(self.outputMd)
            for objId in volMd:
                inputVol = volMd.getValue(emlib.MDL_IMAGE, objId)
                outputVol = outVolMd.getValue(emlib.MDL_IMAGE, objId)
                localArgs = self._segmentLocalArgs(inputVol, fnMask, args)
                maskArgs = self._segMentMaskArgs(inputVol, outputVol, fnMask)
                self._segmentVolume(localArgs, maskArgs, fnMask)
    
    def _segmentVolume(self, localArgs, maskArgs, fnMask):
        self.runJob("xmipp_volume_segment", localArgs)
        if exists(fnMask):
            self.runJob("xmipp_transform_mask", maskArgs)
            cleanPath(fnMask)
    
    #--------------------------- INFO functions --------------------------------
    def _argsChangeHand(self):
        if self.isFirstStep:
            if self._isSingleInput():
                args = "-i %s -o %s" % (self.inputFn, self.outputStk)
            else:
                args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                       % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        args += " --flipX"
        
        return args

    def _argsRotateIco(self):
        if self.isFirstStep:
            if self._isSingleInput():
                args = "-i %s -o %s" % (self.inputFn, self.outputStk)
            else:
                args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                       % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        args += " --rotate_volume icosahedral i%d i%s --dont_wrap" \
                % (self.rotateFromIco.get()+1, self.rotateToIco.get()+1)

        return args
    
    def _argsRandomize(self):
        if self.isFirstStep:
            if self._isSingleInput():
                args = "-i %s -o %s" % (self.inputFn, self.outputStk)
            else:
                args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                       % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        
        samplingRate = self.inputVolumes.get().getSamplingRate()
        resol = self.maxResolutionRandomize.get()
        args += " --freq continuous %f %f" % (float(resol),samplingRate)
        return args
    
    def _argsSymmetrize(self):
        if self.isFirstStep:
            if self._isSingleInput():
                args = "-i %s -o %s" % (self.inputFn, self.outputStk)
            else:
                args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                       % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk

        symmetry   = self.symmetryGroup.get()
        doWrap     = self.doWrap.get()
        if self.volumeMask.get() is not None:
            fnVolumeMask = self.volumeMask.get().getFileName()
            doVolumeMask = True
        else:
            doVolumeMask = False

        ###########FILEFILEFILE
        symmetryAggregation = self.aggregation.get()

        # Validation done in the _validate function
        #         if symmetry != 'c1':
        args += " --sym %s " % symmetry

        if symmetryAggregation == "sum":
            args += " --sum"

        if not doWrap:
            args += " --dont_wrap "

        if doVolumeMask:
            if exists(fnVolumeMask):
                args += " --mask_in %s "%fnVolumeMask
            else:
                print('Error: mask %s does not exists'%fnVolumeMask)
        return args
    
    def _argsLaplacian(self):
        args = ""
        if self._isSingleInput():
            args = "-i %s -o %s --retinex %f" \
                   % (self.inputFn, self.outputStk, 0.9)

        if self.volumeMask.get() is not None:
            fnVolumeMask = self.volumeMask.get().getFileName()
            if exists(fnVolumeMask):
                args += " %s"%fnVolumeMask

        return args
    
    def _argsAdjust(self, number):
        if self.inputImages.get().hasAlignmentProj():
            fn = "input_images.xmd"
        else:
            fn = "images_iter001_%02d.xmd" % number
        args = " -m %s" % self._getTmpPath(fn)
        return args
    
    def _adjustLocalArgs(self, inputVol, outputVol, args):
            localArgs = "-i %s -o %s" % (inputVol, outputVol) + args
            return localArgs
    
    def _argsSegment(self):
        segmentationType = self.segmentationType.get()
        segmentationMass = self.segmentationMass.get()
        ts = self._getSize()
        
        args = " --method "
        if segmentationType == "Voxel mass":
            args += "voxel_mass %d" % (int(segmentationMass))
        elif segmentationType == "Aminoacid mass":
            args += "aa_mass %d %f" % (int(segmentationMass),float(ts))
        elif segmentationType == "Dalton mass":
            args += "dalton_mass %d %f" % (int(segmentationMass),float(ts))
        else:
            args += "otsu"
        return args
    
    def _segmentLocalArgs(self, inputVol, fnMask, args):
        return "-i %s -o %s " % (inputVol, fnMask) + args
    
    def _segMentMaskArgs(self, inputVol, outputVol, fnMask):
        print("self.isFirstStep, ", self.isFirstStep)
        if self.isFirstStep:
            maskArgs = "-i %s -o %s" % (inputVol, outputVol)
            self._setFalseFirstStep()
        else:
            maskArgs = "-i %s" % outputVol
        maskArgs += " --mask binary_file %s" % fnMask
        return maskArgs
    
    def _argsNormalize(self):
        if self.isFirstStep:
            if self._isSingleInput():
                args = "-i %s -o %s" % (self.inputFn, self.outputStk)
            else:
                args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                       % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        
        maskRadius = self.backRadius.get()
        if maskRadius <= 0:
            size = self._getSize()
            maskRadius = size/2
        
        args += " --method NewXmipp --background circle %d" % int(maskRadius)
        return args
    
    def _argsInvert(self):
        if self.isFirstStep:
            if self._isSingleInput():
                args = "-i %s -o %s" % (self.inputFn, self.outputStk)
            else:
                args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                       % (self.inputFn, self.outputStk, self.outputMd)
            self._setFalseFirstStep()
        else:
            args = "-i %s" % self.outputStk
        args += XmippPreprocessHelper._argsCommonInvert()
        return args
    
    def _argsThreshold(self):
        if self.isFirstStep:
            if self._isSingleInput():
                args = "-i %s -o %s" % (self.inputFn, self.outputStk)
            else:
                args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" \
                       % (self.inputFn, self.outputStk, self.outputMd)
        else:
            args = "-i %s" % self.outputStk
        args += XmippPreprocessHelper._argsCommonThreshold(self)
        return args

    def _validate(self):
        validateMsgs = []
        
        if not self.inputVolumes.hasValue():
            validateMsgs.append('Please provide an initial volume(s).')
            
        if self.doNormalize.get():
            size = int(self._getSize()/2)
            
            if self.backRadius.get() > size:
                validateMsgs.append('Set a valid Background radius less than %d'
                                                                         % size)
        
        if self.doSymmetrize.get():
            if self.symmetryGroup.get() == 'c1':
                validateMsgs.append('c1 is not a valid symmetry group.'
                                    'If you do not want to symmetrize set'
                                    'the field Symmetrize to not.')

        if self.useGpu and not isXmippCudaPresent(CUDA_ALIGN_SIGNIFICANT):
            validateMsgs.append("You asked to use GPU, but I cannot find Xmipp cuda programs in the path")

        return validateMsgs
    
    def _summary(self):
        summary = []

        if self.inputVolumes.get() is None:
            return summary

        summary.append("Input volumes:  %s"%self.inputVolumes.get().getNameId())
        
        if not hasattr(self, 'outputVol'):
            summary.append("Output volumes not ready yet.")
        else:
            summary.append("Output volumes: %s" % self.outputVol)
        
        return summary
    
    def _methods(self):
        return self._summary()

    #--------------------------- UTILS functions -------------------------------
    def _getSize(self):
        """ get the size of either Volume or SetOfVolumes object"""
        if isinstance(self.inputVolumes.get(), Volume):
            Xdim = self.inputVolumes.get().getDim()[0]
        else:
            Xdim = self.inputVolumes.get().getDimensions()[0]
        return Xdim
    
    def _setFalseFirstStep(self):
        if self.isFirstStep:
                self.isFirstStep = False

    def _getRandomSubset(self, imgSet, numOfParts):
        if isinstance(imgSet, SetOfClasses2D):
            partSet = self._createSetOfParticles("_averages")
            for i, cls in enumerate(imgSet):
                img = cls.getRepresentative()
                img.setSamplingRate(cls.getSamplingRate())
                img.setObjId(i+1)
                partSet.append(img)
        else:
            partSet = imgSet
        
        if partSet.getSize() > numOfParts:
            newPartSet = SetOfParticles(filename=self._getTmpPath("particles_tmp.sqlite"))
            counter = 0
            for part in partSet.iterItems(orderBy='RANDOM()', direction='ASC'):
                if counter < numOfParts:
                    newPartSet.append(part)
                    counter += 1
                else:
                    break
        else:
            newPartSet = partSet
        
        return newPartSet
    
