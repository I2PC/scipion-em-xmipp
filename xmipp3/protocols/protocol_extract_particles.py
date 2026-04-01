# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Laura del Cano (ldelcano@cnb.csic.es)
# *              Adrian Quintana (aquintana@cnb.csic.es)
# *              Javier Vargas (jvargas@cnb.csic.es)
# *              Grigory Sharov (sharov@igbmc.fr)
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

from os.path import exists

import pwem.emlib.metadata as md
import pyworkflow.utils as pwutils
from pyworkflow.object import Integer
from pyworkflow.protocol.constants import (STEPS_PARALLEL, LEVEL_ADVANCED,
                                           STATUS_FINISHED)
import pyworkflow.protocol.params as params
from pwem.protocols import ProtExtractParticles
from pwem.objects import Particle

from xmipp3.base import XmippProtocol
from xmipp3.convert import (micrographToCTFParam, writeMicCoordinates,
                            xmippToLocation, setXmippAttributes)
from xmipp3.constants import OTHER
from pyworkflow import BETA, UPDATED, NEW, PROD

FACTOR_BOXSIZE = 1.5

class XmippProtExtractParticles(ProtExtractParticles, XmippProtocol):
    """Extracts particle images from micrographs based on provided coordinates.
    This essential step prepares particle stacks for further processing such as
    classification and reconstruction.

    AI Generated:

    ## Overview

    The Extract Particles protocol cuts out particle images from micrographs
    using a set of input coordinates. This is one of the central steps in a
    cryo-EM workflow, because it transforms the coordinate list produced by
    picking into an actual stack of particle images that can later be
    classified, aligned, and reconstructed.

    In practical use, extraction is much more than simple cropping. The
    protocol can also rescale the particles, remove extreme pixel outliers,
    invert contrast, apply phase flipping, normalize the images, and annotate
    each particle with local quality descriptors. These operations make the
    extracted particles more suitable for downstream analysis and can strongly
    influence the quality of 2D classes and later 3D reconstructions.

    For a biological user, the main idea is that this protocol defines exactly
    what the “working particle images” will look like in the rest of the
    project. Choices made here affect both interpretability and the behavior of
    later algorithms.

    ## Inputs and General Workflow

    The protocol starts from a **set of coordinates**, and from the micrographs
    associated with those coordinates. In some workflows, the extraction can
    also be performed from a different set of micrographs, provided that the
    relationship between the coordinates and the micrographs is still valid.

    For each micrograph, the protocol reads the coordinate positions,
    optionally rescales them if the extraction micrographs differ from the
    picking micrographs, and cuts particle boxes centered at those positions.
    Before or during extraction, the protocol may apply several preprocessing
    operations to the micrograph or to the particle images, such as dust
    removal, downsampling, phase flipping, contrast inversion, and
    normalization.

    The final output is a **set of particles** ready for downstream processing.

    ## Choosing the Particle Box Size

    The **particle box size** is one of the most important parameters. It
    defines how large the extracted particle image will be, in pixels.

    Biologically and practically, the box should be large enough to contain the
    full particle plus a surrounding region of background. The background is
    important because many downstream methods use it to estimate noise
    statistics and to stabilize alignment. If the box is too small, part of the
    particle may be cut off, which can strongly damage classification and
    refinement. If it is too large, unnecessary background is added, increasing
    computational cost and sometimes making alignment harder.

    The protocol suggests a box size close to **1.5 times the picking box size**,
    which is often a good starting point. In routine workflows, users should
    visually inspect a few extracted particles to verify that the particle fits
    comfortably within the box without wasting too much empty area.

    ## Rescaling and Downsampling

    The protocol can optionally **rescale particles during extraction**. This
    can be done by specifying a downsampling factor, a target output box size,
    or a target sampling rate.

    This option is especially useful in early stages of processing. Larger
    particles at very fine sampling may be unnecessarily expensive for initial
    2D classification, whereas a moderate downsampling often speeds up
    processing considerably without compromising early decisions.

    When particles are rescaled, the protocol also adjusts the coordinate
    scaling and the extraction box size accordingly. An advanced option allows
    the user to **force the output particle box size**, which can be useful in
    specialized workflows but should be used carefully because it may break the
    original size ratio between picking and extraction.

    From a biological point of view, rescaling does not change the underlying
    specimen, but it changes how finely the image is sampled. A strong
    downsampling can remove high-resolution information, so it is usually
    appropriate for exploratory or initial classification steps rather than for
    the final high-resolution stages.

    ## Border Handling

    By default, Xmipp skips particles whose boxes would extend outside the
    micrograph borders. This is often a sensible behavior because border
    particles are incomplete and may be problematic in downstream analysis.

    However, the protocol allows the user to **fill pixels outside the borders**
    with the nearest available pixel values. This makes it possible to retain
    particles near the edges of the micrograph.

    In practice, this option is only useful if those edge particles are expected
    to be important. In most workflows, excluding border particles is safer,
    since partially boxed particles often behave poorly in classification.

    ## Dust Removal

    The protocol can perform **dust removal**, which replaces extreme pixel
    outliers with values drawn from a more normal image background distribution.

    This step is recommended in many cryo-EM workflows because micrographs may
    contain very bright or very dark artifacts caused by detector defects,
    contaminants, or occasional image anomalies. Such pixels can interfere with
    normalization and classification.

    The **dust threshold** controls how extreme a pixel must be before it is
    replaced. For cryo-EM data, moderate thresholds are usually appropriate.
    For negative stain or other higher-contrast images, users may need to be
    more conservative, since the true signal itself may become extreme.

    Biologically, dust removal is not meant to alter the particle signal. Its
    purpose is to eliminate obvious non-biological artifacts.

    ## Contrast Inversion

    The protocol can **invert image contrast** during extraction. This is often
    necessary because different software packages expect particles with
    opposite contrast conventions.

    In most standard cryo-EM workflows using Xmipp, Relion, Spider, or EMAN
    conventions, particles are usually expected to be
    **white on a dark background**. If the raw micrographs show particles as
    dark features on a bright background, inversion is recommended.

    This step does not change the structure, only the sign convention of the
    image. Nevertheless, it is important because using the wrong contrast
    convention can confuse downstream methods or make visual inspection
    misleading.

    ## Phase Flipping

    The protocol can apply **CTF phase flipping** during extraction, provided
    that CTF information is available for the micrographs.

    Phase flipping compensates for the phase reversals introduced by the
    contrast transfer function. This is a relatively simple form of CTF
    correction and is useful in some processing traditions, particularly in
    Xmipp and EMAN workflows.

    However, not all packages expect or require phase-flipped particles. Some
    modern refinement programs prefer to work with unflipped particles and
    model the CTF internally. Therefore, this option should be chosen according
    to the overall workflow.

    From a biological perspective, phase flipping can improve the coherence of
    particle images, but only if the CTF estimation is reliable. It should not
    be used unless valid CTF information is attached to the micrographs.

    ## Normalization

    The protocol can **normalize the extracted particles**, which is strongly
    recommended in most workflows. Normalization places the particle images on
    a more consistent intensity scale and typically forces the background to
    have approximately zero mean and unit variance.

    Several normalization strategies are available. The simplest normalizes the
    whole image. More refined methods estimate the background from pixels
    outside a given circular radius and normalize with respect to that
    background. The **Ramp** option also subtracts a background trend before
    normalization.

    The **background radius** determines which pixels are considered background.
    If set automatically, the protocol uses approximately half the box size.
    This works well in many cases, but users should ensure that the chosen
    background region is mostly outside the particle.

    Normalization is extremely important biologically and computationally
    because it makes particles more comparable across micrographs and imaging
    conditions.

    ## Patch Size and Local Micrograph Quality Measures

    An additional feature of this protocol is that it computes local descriptors
    of the micrograph region around each coordinate, including measures related
    to **local variance** and **Gini coefficient**. These can help characterize
    whether a particle lies in a noisy or heterogeneous local region of the
    micrograph.

    The **patch size** controls the neighborhood used for this local analysis.
    If not set manually, the protocol uses a value related to the particle box
    size.

    These descriptors do not directly alter the particles, but they are useful
    for later quality analysis or filtering. In practice, they provide extra
    information about whether a coordinate came from a locally stable or
    problematic region of the micrograph.

    ## Coordinate Scaling and Different Micrograph Sources

    If extraction is performed from a micrograph set different from the one
    used for picking, the protocol automatically handles the difference in
    sampling rate and rescales the coordinates accordingly.

    This is important in workflows where picking was done on one representation
    of the micrographs and extraction is performed on another. The protocol is
    designed to preserve geometric consistency, but users should still make
    sure that the coordinate-to-micrograph relationship is correct.

    From a practical point of view, this feature is very useful in multiscale
    workflows, where picking may happen on downsampled micrographs and
    extraction on full-resolution ones.


    ## Outputs and Their Interpretation

    The protocol produces a **set of extracted particles**. Each particle
    retains its coordinate and micrograph association, and when available it
    may also carry CTF information and local quality descriptors.

    These particles are the main working dataset for downstream cryo-EM
    analysis. Their appearance, sampling, normalization state, and possible
    phase flipping all depend on the choices made in this protocol.

    Biologically, the output should be understood as the standardized image
    representation of the experimental particle projections that will be used
    for all later computational interpretation.

    ## Practical Recommendations

    A good general strategy is to begin with a reasonable box size that
    comfortably contains the particle plus background, and to keep normalization
    enabled. Dust removal is also usually beneficial.

    If the dataset is large or the particle is big, moderate rescaling can
    greatly accelerate initial classification. For final high-resolution work,
    however, it is often preferable to return to particles extracted at the
    most appropriate sampling.

    Contrast inversion should be chosen according to the requirements of the
    downstream software. Phase flipping should only be used when it is
    consistent with the rest of the workflow and when reliable CTF estimates
    are available.

    Users should also inspect a representative subset of extracted particles
    visually. This remains one of the best ways to verify that the box size,
    normalization, and contrast choices are sensible.

    ## Final Perspective

    The Extract Particles protocol is one of the key points where raw cryo-EM
    data become a workable particle dataset. It does not just crop boxes: it
    defines the image representation that downstream algorithms will receive.

    For most cryo-EM users, careful configuration of this protocol is essential.
    Good extraction choices can make the rest of the workflow easier, cleaner,
    and more robust, while poor choices can propagate problems throughout
    classification and reconstruction.
    """
    _label = 'extract particles'
    _devStatus = PROD
    RESIZE_FACTOR = 0
    RESIZE_DIMENSIONS = 1
    RESIZE_SAMPLINGRATE = 2
    
    def __init__(self, **kwargs):
        ProtExtractParticles.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL

    #--------------------------- DEFINE param functions ------------------------
    def _definePreprocessParams(self, form):
        form.addParam('boxSize', params.IntParam,
                      label='Particle box size (px)', allowsPointers=True, default=-1,
                      # validators=[params.Positive],
                      help='This is the size of the boxed particles (in pixels). '
                           'Note that if the downsample option is used, the particles will be boxed using a '
                           'downsampled box size that maintains the ratio with the original extraction box size.'
                           'Use the wizard to select a box size automatically. '
                           'This is calculated by multiplying 1.5 * box size used for picking.')

        form.addParam('doResize', params.BooleanParam, default=False,
                      label='Rescale particles?',
                      help='If you set to *Yes*, you should provide a resize option.')

        form.addParam('resizeOption', params.EnumParam,
                      choices=['Factor', 'Dimensions', 'Sampling Rate'],
                      condition='doResize',
                      default=self.RESIZE_FACTOR,
                      label="Resize option", display=params.EnumParam.DISPLAY_COMBO,
                      help='Select an option to resize the images: \n '
                           '_Factor_: Set a resize factor to resize. \n '
                           '_Dimensions_: Set the output dimensions in pixels. \n'
                           '_Sampling Rate_: Set the desire sampling rate to resize.')

        # downFactor should always be 1.0 or greater
        geOne = params.GE(1.0, error='Value should be greater or equal than 1.0')

        form.addParam('downFactor', params.FloatParam, default=1.0,
                      validators=[geOne],
                      label='Downsampling factor',
                      condition='doResize and resizeOption==%d' % self.RESIZE_FACTOR,
                      help='Select a value greater than 1.0 to reduce the size '
                           'of micrographs before extracting the particles. '
                           'If 1.0 is used, no downsample is applied. '
                           'Non-integer downsample factors are possible. ')

        form.addParam('resizeDim', params.IntParam, default=0,
                      condition='doResize and resizeOption==%d' % self.RESIZE_DIMENSIONS,
                      allowsPointers=True,
                      label='Re-scaled size (px)',
                      help='Size in pixels of the particle images <x> <y=x> <z=x>.')

        form.addParam('resizeSamplingRate', params.FloatParam, default=1.0,
                      condition='doResize and resizeOption==%d' % self.RESIZE_SAMPLINGRATE,
                      allowsPointers=True,
                      label='Resize sampling rate (Å/px)',
                      help='Set the new output sampling rate.')

        form.addParam('boxSizeForced', params.IntParam,
                      label='Force output particle box size (px)', default=-1,
                      condition='doResize and resizeOption!=%d' % self.RESIZE_DIMENSIONS,
                      expertLevel=LEVEL_ADVANCED,
                      help='By default, this is set to -1, meaning that if the downsample option is used, the particles '
                           'will be boxed using a downsampled box size that maintains the ratio with the '
                           'original extraction box size. If a value is specified (forced), the output extraction '
                           'box size will be applied directly to the downsampled micrographs, '
                           'losing the original size ratio.')

        form.addParam('doBorders', params.BooleanParam, default=False,
                      label='Fill pixels outside borders',
                      help='Xmipp by default skips particles whose boxes fall '
                           'outside of the micrograph borders. Set this '
                           'option to True if you want those pixels outside '
                           'the borders to be filled with the closest pixel '
                           'value available')
        
        form.addSection(label='Preprocess')

        form.addParam('doRemoveDust', params.BooleanParam, default=True,
                      label='Dust removal (Recommended)', important=True,
                      help='Sets pixels with unusually large values to random '
                           'values from a Gaussian with zero-mean and '
                           'unity-standard deviation.')

        form.addParam('thresholdDust', params.FloatParam, default=5,
                      condition='doRemoveDust', expertLevel=LEVEL_ADVANCED,
                      label='Threshold for dust removal',
                      help='Pixels with a signal higher or lower than this '
                           'value times the standard deviation of the image '
                           'will be affected. For cryo, 3.5 is a good value. '
                           'For high-contrast negative stain, the signal '
                           'itself may be affected so that a higher value may '
                           'be preferable.')

        form.addParam('doInvert', params.BooleanParam, default=True,
                      label='Invert contrast', 
                      help='Invert the contrast if your particles are black '
                           'over a white background.  Xmipp, Spider, Relion '
                           'and Eman require white particles over a black '
                           'background. Frealign (up to v9.07) requires black '
                           'particles over a white background')
        
        form.addParam('doFlip', params.BooleanParam, default=False,
                      label='Phase flipping',
                      help='Use the information from the CTF to compensate for '
                           'phase reversals.\n'
                           'Phase flip is recommended in Xmipp or Eman\n'
                           '(even Wiener filtering and bandpass filter are '
                           'recommended for obtaining better 2D classes)\n'
                           'Otherwise (Frealign, Relion, Spider, ...), '
                           'phase flip is not recommended.')

        form.addParam('doNormalize', params.BooleanParam, default=True,
                      label='Normalize (Recommended)', 
                      help='It subtract a ramp in the gray values and '
                           'normalizes so that in the background there is 0 '
                           'mean and standard deviation 1.')
        form.addParam('normType', params.EnumParam,
                      choices=['OldXmipp','NewXmipp','Ramp'], default=2,
                      condition='doNormalize', expertLevel=LEVEL_ADVANCED,
                      display=params.EnumParam.DISPLAY_COMBO,
                      label='Normalization type', 
                      help='OldXmipp (mean(Image)=0, stddev(Image)=1). \n'
                           'NewXmipp (mean(background)=0, '
                           'stddev(background)=1) \n  '
                           'Ramp (subtract background+NewXmipp).')
        form.addParam('backRadius', params.IntParam, default=-1,
                      condition='doNormalize',
                      label='Background radius (px)', expertLevel=LEVEL_ADVANCED,
                      help='Pixels outside this circle are assumed to be noise '
                           'and their stddev is set to 1. Radius for '
                           'background circle definition (in pix.). If this '
                           'value is less than or equal to 0, then half the box size is used.')
        
        form.addParam('patchSize', params.IntParam, default=-1, 
                      label='Patch size for the variance filter (px)', 
                      expertLevel=LEVEL_ADVANCED,
                      help='Windows size to make the variance filter and '
                           'compute the Gini coeff. A twice of the particle '
                           'size is recommended. Set at -1 applies 1.5*BoxSize.')

        form.addParallelSection(threads=4, mpi=1)
    
    #--------------------------- INSERT steps functions ------------------------
    def _insertInitialSteps(self):
        # Just overwrite this function to load some info
        # before the actual processing
        self._setupBasicProperties()

        return []

    def _getExtractArgs(self):
        """ Should be implemented in sub-classes to define the argument
        list that should be passed to the picking step function.
        """
        return [self.doInvert.get(),
                self._getNormalizeArgs(),
                self.doBorders.get()]
    
    #--------------------------- STEPS functions -------------------------------
    def _extractMicrograph(self, mic, doInvert, normalizeArgs, doBorders):
        """ Extract particles from one micrograph """
        fnLast = mic.getFileName()
        baseMicName = pwutils.removeBaseExt(fnLast)
        outputRoot = str(self._getExtraPath(baseMicName))
        fnPosFile = self._getMicPos(mic)
        boxSize = self._getExtractBoxSize()
        downFactor = self._getDownFactor()
        patchSize = self.patchSize.get() if self.patchSize.get() > 0 \
                    else int(boxSize*1.5*downFactor)

        particlesMd = 'particles@%s' % fnPosFile
        # If it has coordinates extract the particles
        if exists(fnPosFile):
            # Create a list with micrographs operations (programs in xmipp) and
            # the required command line parameters (except input/ouput files)
            micOps = []

            try:
                # Compute the variance and Gini coeff. of the part. and mic., resp.
                args  =  '--pos %s' % fnPosFile
                args += ' --mic %s' % fnLast
                args += ' --patchSize %d' % patchSize
                self.runJob('xmipp_coordinates_noisy_zones_filter', args)
            except:
                print("'xmipp_coordinates_noisy_zones_filter' have failed for "
                      "%s micrograph. We continue..." % mic.getMicName())

            def getMicTmp(suffix):
                return self._getTmpPath(baseMicName + suffix)

            # Check if it is required to downsample our micrographs
            if self.notOne(downFactor):
                fnDownsampled = getMicTmp("_downsampled.xmp")
                args = "-i %s -o %s --step %f --method fourier"
                self.runJob('xmipp_transform_downsample',
                            args % (fnLast, fnDownsampled, downFactor))
                fnLast = fnDownsampled

            if self.doRemoveDust:
                fnNoDust = getMicTmp("_noDust.xmp")
                args = " -i %s -o %s --bad_pixels outliers %f"
                self.runJob('xmipp_transform_filter',
                            args % (fnLast, fnNoDust, self.thresholdDust))
                fnLast = fnNoDust

            if self._useCTF():
                # We need to write a Xmipp ctfparam file
                # to perform the phase flip on the micrograph
                fnCTF = self._getTmpPath("%s.ctfParam" % baseMicName)
                micrographToCTFParam(mic, fnCTF)
                # Insert step to flip micrograph
                if self.doFlip:
                    fnFlipped = getMicTmp('_flipped.xmp')
                    args = " -i %s -o %s --ctf %s --sampling %f"
                    self.runJob('xmipp_ctf_phase_flip',
                                args % (fnLast, fnFlipped, fnCTF,
                                        self._getNewSampling()))
                    fnLast = fnFlipped
            else:
                fnCTF = None

            args = " -i %s --pos %s" % (fnLast, particlesMd)
            args += " -o %s.mrcs --Xdim %d" % (outputRoot, boxSize)

            if doInvert:
                args += " --invert"

            if fnCTF:
                args += " --ctfparam " + fnCTF
            
            if doBorders:
                args += " --fillBorders"

            self.runJob("xmipp_micrograph_scissor", args)

            # Normalize
            if normalizeArgs:
                self.runJob('xmipp_transform_normalize',
                            '-i %s.mrcs %s' % (outputRoot, normalizeArgs))
        else:
            self.warning("The micrograph %s hasn't coordinate file! "
                         % baseMicName)
            self.warning("Maybe you picked over a subset of micrographs")

        # Let's clean the temporary mrc micrographs
        if not pwutils.envVarOn("SCIPION_DEBUG_NOCLEAN"):
            pwutils.cleanPattern(self._getTmpPath(baseMicName) + '*')

    def _getNormalizeArgs(self):
        if not self.doNormalize:
            return ''

        normType = self.getEnumText("normType")
        args = "--method %s " % normType

        if normType != "OldXmipp":
            bgRadius = self.backRadius.get()
            if bgRadius <= 0:
                bgRadius = int(self._getExtractBoxSize() / 2)
            args += " --background circle %d" % bgRadius

        return args

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []
        if self.doResize:
            downFactor = self._getDownFactor()
            if downFactor < 1:
                errors.append('The new particles size should be smaller than the current one')

        if self.boxSize.get() == -1:
            self.boxSize.set(self.getBoxSize())

        if self.boxSize <= 0:
            errors.append('Box size must be positive.')
        else:
            self.boxSize.set(self.getEven(self.boxSize))

        if self.doNormalize:
            if self.backRadius >= int(self.boxSize.get() / 2):
                errors.append("Background radius for normalization should be "
                              "equal or less than half of the box size.")

        # doFlip can only be selected if CTF information
        # is available on picked micrographs
        if self.doFlip and not self._useCTF():
            errors.append('Phase flipping cannot be performed unless '
                          'CTF information is provided.')

        # We cannot check this if the protocol is in streaming.

        #self._setupCtfProperties() # setup self.micKey among others
        # if self._useCTF() and self.micKey is None:
        #     errors.append('Some problem occurs matching micrographs and CTF.\n'
        #                   'There were micrographs for which CTF was not found '
        #                   'either using micName or micId.\n')

        # Clear the CTFs if micrograph source is "same as picking" to avoid unconsistencies
        if not self._micsOther():
            self.inputMicrographs.set(None)

        return errors
    
    def _citations(self):
        return ['Vargas2013b']
    
    def _summary(self):
        summary = []
        summary.append("Micrographs source: %s"
                       % self.getEnumText("downsampleType"))
        summary.append("Particle box size: %d" % self.boxSize)
        
        if not hasattr(self, 'outputParticles'):

            summary.append("Output images not ready yet.") 
        else:
            summary.append("Particles extracted: %d" %
                           self.outputParticles.getSize())
        
        return summary
    
    def _methods(self):
        methodsMsgs = []

        if self.getStatus() == STATUS_FINISHED:
            msg = ("A total of %d particles of size %d were extracted"
                   % (self.getOutput().getSize(), self.boxSize))
            
            if self._micsOther():
                msg += (" from another set of micrographs: %s"
                        % self.getObjectTag('inputMicrographs'))
            
            msg += " using coordinates %s" % self.getObjectTag('inputCoordinates')
            msg += self.methodsVar.get('')
            methodsMsgs.append(msg)

            if self.doRemoveDust:
                methodsMsgs.append("Removed dust over a threshold of %s."
                                   % self.thresholdDust)
            if self.doInvert:
                methodsMsgs.append("Inverted contrast on images.")
            if self._doDownsample():
                methodsMsgs.append("Particles downsampled by a factor of %0.2f."
                                   % self._getDownFactor())
            if self.doNormalize:
                methodsMsgs.append("Normalization: %s."
                                   % self.getEnumText('normType'))

        return methodsMsgs

    # --------------------------- UTILS functions ------------------------------
    def _convertCoordinates(self, mic, coordList):
        writeMicCoordinates(mic, coordList, self._getMicPos(mic),
                            getPosFunc=self._getPos)
    
    def _micsOther(self):
        """ Return True if other micrographs are used for extract. """
        return self.downsampleType == OTHER

    def _useCTF(self):
        return self.ctfRelations.hasValue()

    def _doDownsample(self):
        return self._getDownFactor() > 1.0

    def notOne(self, value):
        return abs(value - 1) > 0.0001

    def _getNewSampling(self):
        newSampling = self.samplingMics

        if self._doDownsample():
            # Set new sampling, it should be the input sampling of the used
            # micrographs multiplied by the downFactor
            newSampling *= self._getDownFactor()

        return newSampling

    def _getDownFactor(self):
        downFactor = 1
        if self.doResize:
            if self.resizeOption == self.RESIZE_FACTOR:
                downFactor = self.downFactor.get()
            elif self.resizeOption == self.RESIZE_SAMPLINGRATE:
                downFactor = self.resizeSamplingRate.get()/self.getCoordSampling()
            elif self.resizeOption == self.RESIZE_DIMENSIONS:
                downFactor = self.boxSize.get()/self.resizeDim.get()

        return float(downFactor)

    def _getExtractBoxSize(self):
        if self.boxSize.get() == -1:
            boxSize = int(self.getBoxSize())
        else:
            boxSize = int(self.boxSize.get())

        downFactor =  self._getDownFactor()
        if downFactor > 1:
            newBoxSize = self.getEven(boxSize/downFactor)
        else:
            newBoxSize = boxSize

        if self.boxSizeForced.get() != -1: # Force output box size
            newBoxSize = self.boxSizeForced.get()

        return int(newBoxSize)

    def _setupBasicProperties(self):
        # Set sampling rate (before and after doDownsample) and inputMics
        # according to micsSource type
        inputCoords = self.getCoords()
        mics = inputCoords.getMicrographs()
        self.samplingInput = inputCoords.getMicrographs().getSamplingRate()
        self.samplingMics = self.getInputMicrographs().getSamplingRate()
        self.samplingFactor = float(self.samplingMics / self.samplingInput)

        scale = self.getBoxScale()
        self.debug("Scale: %f" % scale)
        if self.notOne(scale):
            # If we need to scale the box, then we need to scale the coordinates
            getPos = lambda coord: (int(coord.getX() * scale),
                                    int(coord.getY() * scale))
        else:
            getPos = lambda coord: coord.getPosition()
        # Store the function to be used for scaling coordinates
        self._getPos = getPos

    def getInputMicrographs(self):
        """ Return the micrographs associated to the SetOfCoordinates or
        Other micrographs. """
        if not self._micsOther():
            return self.inputCoordinates.get().getMicrographs()
        else:
            return self.inputMicrographs.get()

    def _storeMethodsInfo(self, fnImages):
        """ Store some information when the protocol finishes. """
        mdImgs = md.MetaData(fnImages)
        total = mdImgs.size()
        mdImgs.removeDisabled()
        zScoreMax = mdImgs.getValue(md.MDL_ZSCORE, mdImgs.lastObject())
        numEnabled = mdImgs.size()
        numRejected = total - numEnabled
        msg = ""

        if self.doFlip:
            msg += "\nPhase flipping was performed."

        self.methodsVar.set(msg)

    def getCoords(self):
        return self.inputCoordinates.get()

    def getOutput(self):
        if (self.hasAttribute('outputParticles') and
            self.outputParticles.hasValue()):
            return self.outputParticles
        else:
            return None

    def getCoordSampling(self):
        return self.getCoords().getMicrographs().getSamplingRate()

    def getMicSampling(self):
        return self.getInputMicrographs().getSamplingRate()

    def getBoxScale(self):
        """ Computing the sampling factor between input and output.
        We should take into account the differences in sampling rate between
        micrographs used for picking and the ones used for extraction.
        The downsampling factor could also affect the resulting scale.
        """
        samplingPicking = self.getCoordSampling()
        samplingExtract = self.getMicSampling()
        f = float(samplingPicking) / samplingExtract
        return f / self._getDownFactor() if self._doDownsample() else f

    def getEven(self, boxSize):
        return Integer(int(int(boxSize)/2+0.75)*2)

    def getBoxSize(self):
        # This function is needed by the wizard and for auto-boxSize selection
        return self.getEven(self.getCoords().getBoxSize()*FACTOR_BOXSIZE)

    def _getOutputImgMd(self):
        return self._getPath('images.xmd')

    def createParticles(self, item, row):
        from ..convert import rowToParticle
        
        particle = rowToParticle(row, readCtf=self._useCTF())
        coord = particle.getCoordinate()
        item.setY(coord.getY())
        item.setX(coord.getX())
        particle.setCoordinate(item)
        self.imgSet.append(particle)
        item._appendItem = False

    def readPartsFromMics(self, micList, outputParts):
        """ Read the particles extract for the given list of micrographs
        and update the outputParts set with new items.
        """
        p = Particle()
        for mic in micList:
            # We need to make this dict because there is no ID in the .xmd file
            coordDict = {}
            for coord in self.coordDict[mic.getObjId()]:
                pos = self._getPos(coord)
                if pos in coordDict:
                    print("WARNING: Ignoring duplicated coordinate: %s, id=%s" %
                          (coord.getObjId(), pos))
                coordDict[pos] = coord

            added = set() # Keep track of added coords to avoid duplicates
            fnMicXmd = self._getMicXmd(mic)
            if exists(fnMicXmd):
                for row in md.iterRows(fnMicXmd):
                    pos = (row.getValue(md.MDL_XCOOR), row.getValue(md.MDL_YCOOR))
                    coord = coordDict.get(pos, None)
                    if coord is not None and coord.getObjId() not in added:
                        # scale the coordinates according to particles dimension.
                        coord.scale(self.getBoxScale())
                        p.copyObjId(coord)
                        p.setLocation(xmippToLocation(row.getValue(md.MDL_IMAGE)))
                        p.setCoordinate(coord)
                        p.setMicId(mic.getObjId())
                        p.setCTF(mic.getCTF())
                        # adding the variance and Gini coeff. value of the mic zone
                        setXmippAttributes(p, row, md.MDL_SCORE_BY_VAR)
                        setXmippAttributes(p, row, md.MDL_SCORE_BY_GINI)
                        setXmippAttributes(p, row, md.MDL_LOCAL_AVERAGE)
                        if row.containsLabel(md.MDL_ZSCORE_DEEPLEARNING1):
                            setXmippAttributes(p, row, md.MDL_ZSCORE_DEEPLEARNING1)

                        # disabled particles (in metadata) should not add to the
                        # final set
                        if row.getValue(md.MDL_ENABLED) > 0:
                            outputParts.append(p)
                            added.add(coord.getObjId())

            # Release the list of coordinates for this micrograph since it
            # will not be longer needed
            del self.coordDict[mic.getObjId()]

    def _getMicPos(self, mic):
        """ Return the corresponding .pos file for a given micrograph. """
        micBase = pwutils.removeBaseExt(mic.getFileName())
        return self._getExtraPath(micBase + ".pos")

    def _getMicXmd(self, mic):
        """ Return the corresponding .xmd with extracted particles
        for this micrograph. """
        micBase = pwutils.removeBaseExt(mic.getFileName())
        return self._getExtraPath(micBase + ".xmd")
