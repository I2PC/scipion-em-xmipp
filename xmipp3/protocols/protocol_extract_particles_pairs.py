# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Laura del Cano (ldelcano@cnb.csic.es)
# *              Adrian Quintana (aquintana@cnb.csic.es)
# *              Javier Vargas (jvargas@cnb.csic.es)
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

try:
    from itertools import izip
except ImportError:
    izip = zip

from os.path import exists

import pwem.emlib.metadata as md
import pyworkflow.utils as pwutils
from pyworkflow.protocol.constants import (STEPS_PARALLEL, LEVEL_ADVANCED,
                                           STATUS_FINISHED)
from pyworkflow.protocol.params import (PointerParam, EnumParam, FloatParam,
                                        IntParam, BooleanParam, Positive, GE)
from pwem.protocols import ProtExtractParticlesPair
from pwem.objects import ParticlesTiltPair, TiltPair, SetOfParticles

from xmipp3.base import XmippProtocol
from xmipp3.convert import (writeSetOfCoordinates, readSetOfParticles,
                            micrographToCTFParam)
from xmipp3.constants import OTHER


class XmippProtExtractParticlesPairs(ProtExtractParticlesPair, XmippProtocol):
    """Extracts particles based on coordinates from tilted pairs of micrographs.

    AI Generated

    ## Overview

    The Extract Particle Pairs protocol extracts particle images from pairs of
    tilted and untilted micrographs using a previously defined set of paired
    coordinates. It is intended for workflows in which each particle has two
    corresponding views: one acquired from an untilted micrograph and one from a
    tilted micrograph.

    This type of data is commonly used in random conical tilt, validation,
    orientation assignment, and other workflows where the geometrical relationship
    between tilted and untilted views is important. The central purpose of this
    protocol is therefore not only to crop particle images, but also to preserve the
    pairing between the two views of the same particle.

    The output is a set of particle pairs. Each pair contains one untilted particle
    and its corresponding tilted particle. This paired structure is essential for
    downstream protocols that rely on the tilt geometry or on the comparison
    between the two acquisition conditions.

    ## Inputs and General Workflow

    The main input is a set of **tilted-pair coordinates**. This object contains
    two linked coordinate sets: one for the untilted micrographs and one for the
    tilted micrographs. Each coordinate in the untilted set is associated with the
    corresponding coordinate in the tilted set.

    By default, particles are extracted from the same micrographs that were used
    during picking. Alternatively, the user may provide another set of tilted-pair
    micrographs. This is useful when coordinates were picked on one version of the
    micrographs, for example a downsampled or preprocessed version, but extraction
    should be performed from another version.

    For each micrograph, the protocol writes the corresponding coordinate file,
    optionally applies preprocessing operations, extracts particle boxes centered
    on the coordinates, normalizes the extracted images if requested, and finally
    reconstructs the particle-pair object by matching the untilted and tilted
    particles.

    The protocol only keeps pairs for which both members are valid. If either the
    untilted or tilted particle is rejected or cannot be extracted, the pair is not
    included in the final output.

    ## Coordinates Tilted Pairs

    The **Coordinates tilted pairs** input defines the geometrical correspondence
    between the two views. This is the most important input of the protocol.

    Each pair represents the same physical particle observed in two different
    micrographs: one untilted and one tilted. Preserving this relationship is the
    main difference between this protocol and standard single-micrograph particle
    extraction.

    Users should ensure that the coordinate-pair object is correct before running
    the protocol. If the pairing is wrong, downstream tilted-pair analysis will be
    affected even if the extracted particle images look visually reasonable.

    ## Micrographs Source

    The **Micrographs source** option controls which micrographs are used for
    extraction.

    If **same as picking** is selected, the protocol extracts the particles from
    the micrographs associated with the input tilted-pair coordinates. This is the
    simplest and safest option, and it is appropriate when picking was performed
    directly on the micrographs that should be used for extraction.

    If **other** is selected, the user must provide another set of tilted-pair
    micrographs. This option is useful in multiscale workflows, for example when
    particles were picked on downsampled micrographs but should be extracted from
    higher-resolution micrographs.

    When another micrograph set is used, the protocol attempts to match the
    coordinates and micrographs by micrograph name. Differences in pixel size are
    handled automatically by rescaling the coordinates. However, the biological
    meaning of the extraction still depends on the micrographs being correctly
    matched. Users should therefore verify that the tilted and untilted micrographs
    correspond to the same acquisition pairs.

    ## CTF Information for Untilited and Tilted Micrographs

    The protocol allows the user to provide separate CTF estimations for the
    untilted and tilted micrographs.

    CTF information has two roles. First, it can be associated with the extracted
    particles, so that downstream protocols know the optical parameters of each
    image. Second, it is required if the user wants to perform phase flipping during
    extraction.

    Because tilted-pair data contain two different micrograph sets, the protocol
    asks separately for:

    - CTF estimation for untilted micrographs.
    - CTF estimation for tilted micrographs.

    If CTF information is available for only one of the two views, the protocol can
    still use it for that view. However, for complete paired downstream analysis, it
    is usually preferable to provide CTF information for both untilted and tilted
    micrographs.

    Phase flipping cannot be performed unless CTF information is provided.

    ## Particle Box Size

    The **particle box size** defines the size, in pixels, of the extracted
    particles.

    The box should be large enough to contain the full particle in both the
    untilted and tilted views, plus a surrounding region of background. This is
    especially important for tilted particles, where the apparent projection may
    occupy a slightly different region of the box because of the tilt geometry and
    coordinate-pairing accuracy.

    If the box is too small, part of the particle may be cut off, damaging
    classification, alignment, or reconstruction. If the box is too large, the
    particles will contain excessive background, increasing computational cost and
    possibly making alignment less stable.

    When downsampling is used, the box size should be interpreted as the final
    particle box size after the intended scaling. Users should visually inspect the
    output particles to confirm that both untilted and tilted views are properly
    centered and fully contained.

    ## Downsampling Factor

    The **downsampling factor** allows the user to reduce the size of the
    micrographs before extraction. A value of 1.0 means that no downsampling is
    applied. Values greater than 1.0 produce particles at a coarser sampling rate.

    Downsampling is useful when the goal is to perform exploratory analysis,
    initial classification, or faster screening of the paired dataset. It reduces
    image size and computational cost.

    However, strong downsampling removes high-resolution information. Therefore, it
    should be used carefully if the extracted particle pairs will be used for
    high-resolution interpretation.

    When downsampling is applied, the protocol updates the output sampling rate and
    rescales the coordinates accordingly, so that the extracted particles remain
    geometrically consistent with the input coordinate pairs.

    ## Border Handling

    By default, Xmipp skips particles whose boxes would fall outside the micrograph
    boundaries. This is usually the safest behavior because particles close to the
    edge may be incomplete.

    The option **Fill pixels outside borders** allows the protocol to keep such
    particles by filling the missing pixels with the closest available pixel values.
    This may be useful if the user wants to retain as many particle pairs as
    possible, but it should be used with caution.

    For tilted-pair data, a particle pair is only useful if both views are valid.
    Keeping edge particles may therefore introduce incomplete or artificial image
    regions into one member of the pair. In most routine workflows, excluding
    border particles is preferable.

    ## Dust Removal

    The protocol can perform **dust removal**, which replaces unusually extreme
    pixel values with values compatible with a normal background distribution.

    This operation is recommended for many cryo-EM datasets because detector
    artifacts, hot pixels, contamination, or other image anomalies may introduce
    very bright or very dark pixels. These outliers can interfere with
    normalization and later classification.

    The **threshold for dust removal** defines how extreme a pixel must be before
    it is corrected. The default value is appropriate for many cryo-EM datasets.
    For high-contrast data, such as negative stain, a higher threshold may be safer
    because true signal can be more intense.

    Dust removal should be understood as an artifact-suppression step. It is not
    intended to modify the biological particle signal.

    ## Contrast Inversion

    The **Invert contrast** option changes the sign convention of the extracted
    images.

    Different cryo-EM packages use different contrast conventions. Xmipp, Spider,
    Relion, and EMAN usually expect particles to appear as white densities on a
    dark background. If the input micrographs show particles as dark features on a
    bright background, contrast inversion should be enabled.

    This operation does not change the structural information, but it is important
    for compatibility with downstream processing. Using the wrong contrast
    convention may lead to confusing visual inspection and poor behavior in some
    algorithms.

    In tilted-pair workflows, the same contrast convention should normally be used
    for both untilted and tilted particles.

    ## Phase Flipping

    The protocol can perform **CTF phase flipping** before particle extraction,
    provided that CTF information is available.

    Phase flipping compensates for the phase reversals introduced by the microscope
    contrast transfer function. It is a simple form of CTF correction and may be
    useful in workflows based on Xmipp or EMAN conventions.

    However, not all downstream programs expect phase-flipped particles. Some
    modern refinement workflows prefer to keep the original images and account for
    the CTF internally. Therefore, phase flipping should be selected according to
    the requirements of the downstream tilted-pair workflow.

    For this protocol, phase flipping may depend on whether CTF estimations are
    available for the untilted micrographs, the tilted micrographs, or both. Users
    should be careful when only one member of the pair has CTF information, because
    this can lead to different correction states for the two views.

    ## Normalization

    The **Normalize** option is recommended in most workflows. It places extracted
    particles on a comparable intensity scale, usually by making the background
    have approximately zero mean and unit standard deviation.

    The protocol offers several normalization types:

    **OldXmipp** normalizes using the statistics of the whole image.

    **NewXmipp** estimates normalization statistics from the background region
    outside a circular mask.

    **Ramp** subtracts a background ramp and then applies the NewXmipp-style
    background normalization.

    For most biological users, the Ramp option is a good default because it helps
    remove slow background variations while producing particles with comparable
    intensity statistics.

    The **background radius** defines the circular region used to separate particle
    and background. Pixels outside this circle are treated as background. If the
    value is not explicitly set, the protocol uses approximately half the box size.
    The radius should not be larger than half the particle box size.

    A good normalization is particularly important for paired-particle workflows
    because the untilted and tilted views should be comparable in intensity scale
    as far as possible.

    ## Preservation of Particle Pairs

    A key feature of this protocol is that it preserves the relationship between
    untilted and tilted particles.

    Internally, the protocol extracts particles from both micrograph sets and then
    reconstructs the final paired object. A pair is included in the output only when
    both the untilted and tilted particle are present and enabled.

    This behavior is important. It avoids producing incomplete pairs, which would
    not be usable in downstream tilted-pair analysis. As a consequence, the final
    number of particle pairs may be smaller than the number of initial coordinate
    pairs if some particles are rejected, fall outside micrograph borders, or fail
    during extraction.

    ## Outputs and Their Interpretation

    The main output is a **ParticlesTiltPair** object.

    This output contains two linked particle sets:

    - the untilted particle set;
    - the tilted particle set.

    Each entry in the output corresponds to a pair formed by one untilted particle
    and its associated tilted particle. The output also keeps the relationship to
    the original tilted-pair coordinates.

    If CTF information was provided, it may be associated with the corresponding
    particles. If phase flipping was performed, the output particle sets are marked
    accordingly.

    The output should be interpreted as the paired image dataset that will be used
    by downstream protocols requiring tilted-pair information.

    ## Practical Recommendations

    For routine use, start by extracting from the same micrographs used for picking,
    unless there is a clear reason to use another micrograph set.

    Choose a particle box size large enough to contain the particle in both views.
    Tilted particles may require slightly more care because their projection and
    centering can differ from the untilted view.

    Keep dust removal and normalization enabled in most cases. These operations
    usually make the particle pairs more stable for downstream processing.

    Use downsampling for exploratory analysis or fast initial screening, but avoid
    excessive downsampling if high-resolution information will be needed later.

    Only enable phase flipping if CTF information is available and if the downstream
    workflow expects phase-flipped particles.

    After extraction, visually inspect both the untilted and tilted particle sets.
    Check that particles are centered, have the expected contrast, are not cut by
    the box boundaries, and remain correctly paired.

    ## Final Perspective

    The Extract Particle Pairs protocol is the tilted-pair counterpart of standard
    particle extraction. Its purpose is not simply to crop particles from
    micrographs, but to produce a clean, geometrically consistent dataset of paired
    particle views.

    Good parameter choices at this stage are important because errors in box size,
    contrast, normalization, CTF handling, or micrograph-coordinate matching can
    propagate into all later tilted-pair analyses.

    For biological users, the most important point is that the protocol defines the
    working paired-particle dataset. The quality and consistency of this output
    will strongly influence the reliability of any downstream interpretation based
    on tilted and untilted views.
    """
    _label = 'extract particle pairs'

    def __init__(self, **kwargs):
        ProtExtractParticlesPair.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinatesTiltedPairs', PointerParam,
                      important=True, label="Coordinates tilted pairs",
                      pointerClass='CoordinatesTiltPair',
                      help='Select the CoordinatesTiltPairs')
        form.addParam('downsampleType', EnumParam,
                      choices=['same as picking', 'other'],
                      default=0, important=True, label='Micrographs source',
                      display=EnumParam.DISPLAY_HLIST,
                      help='By default the particles will be extracted '
                           'from the micrographs used in the picking '
                           'step ( _same as picking_ option ).\n'
                           'If you select _other_ option, you must provide '
                           'a different set of micrographs to extract from.\n'
                           '*Note*: In the _other_ case, ensure that provided '
                           'micrographs and coordinates are related '
                           'by micName or by micId. Difference in pixel size '
                           'will be handled automatically.')
        form.addParam('inputMicrographsTiltedPair', PointerParam,
                      pointerClass='MicrographsTiltPair',
                      condition='downsampleType != 0',
                      important=True, label='Input tilt pair micrographs',
                      help='Select the tilt pair micrographs from which to '
                           'extract.')
        form.addParam('ctfUntilt', PointerParam, allowsNull=True,
                      # expertLevel=LEVEL_ADVANCED,
                      pointerClass='SetOfCTF',
                      label='CTF estimation (untilted mics)',
                      help='Choose some CTF estimation related to input '
                           'UNTILTED micrographs. \n CTF estimation is needed '
                           'if you want to do phase flipping or you want to '
                           'associate CTF information to the particles.')
        form.addParam('ctfTilt', PointerParam, allowsNull=True,
                      # expertLevel=LEVEL_ADVANCED,
                      pointerClass='SetOfCTF',
                      label='CTF estimation (tilted mics)',
                      help='Choose some CTF estimation related to input '
                           'TILTED micrographs. \n CTF estimation is needed '
                           'if you want to do phase flipping or you want to '
                           'associate CTF information to the particles.')

        # downFactor should always be 1.0 or greater
        geOne = GE(1.0, error='Value should be greater or equal than 1.0')

        form.addParam('downFactor', FloatParam, default=1.0,
                      validators=[geOne],
                      label='Downsampling factor',
                      help='Select a value greater than 1.0 to reduce the size '
                           'of micrographs before extracting the particles. '
                           'If 1.0 is used, no downsample is applied. '
                           'Non-integer downsample factors are possible. ')
        form.addParam('boxSize', IntParam, default=0,
                      label='Particle box size', validators=[Positive],
                      help='In pixels. The box size is the size of the boxed '
                           'particles, actual particles may be smaller than '
                           'this. If you do downsampling after extraction, '
                           'provide final box size here.')
        form.addParam('doBorders', BooleanParam, default=False,
                      label='Fill pixels outside borders',
                      help='Xmipp by default skips particles whose boxes fall '
                           'outside of the micrograph borders. Set this '
                           'option to True if you want those pixels outside '
                           'the borders to be filled with the closest pixel '
                           'value available')

        form.addSection(label='Preprocess')
        form.addParam('doRemoveDust', BooleanParam, default=True,
                      important=True, label='Dust removal (Recommended)',
                      help='Sets pixels with unusually large values to random '
                           'values from a Gaussian with zero-mean and '
                           'unity-standard deviation.')
        form.addParam('thresholdDust', FloatParam, default=3.5,
                      condition='doRemoveDust', expertLevel=LEVEL_ADVANCED,
                      label='Threshold for dust removal',
                      help='Pixels with a signal higher or lower than this '
                           'value times the standard deviation of the image '
                           'will be affected. For cryo, 3.5 is a good value. '
                           'For high-contrast negative stain, the signal '
                           'itself may be affected so that a higher value may '
                           'be preferable.')
        form.addParam('doInvert', BooleanParam, default=None,
                      label='Invert contrast',
                      help='Invert the contrast if your particles are black '
                           'over a white background. Xmipp, Spider, Relion '
                           'and Eman require white particles over a black '
                           'background, Frealign (up to v9.07) requires black '
                           'particles over a white background')
        form.addParam('doFlip', BooleanParam, default=False,
                      # expertLevel=LEVEL_ADVANCED,
                      label='Phase flipping',
                      help='Use the information from the CTF to compensate for '
                           'phase reversals.\n'
                           'Phase flip is recommended in Xmipp or Eman\n'
                           '(even Wiener filtering and bandpass filter are '
                           'recommended for obtaining better 2D classes)\n'
                           'Otherwise (Frealign, Relion, Spider, ...), '
                           'phase flip is not recommended.')
        form.addParam('doNormalize', BooleanParam, default=True,
                      label='Normalize (Recommended)',
                      help='It subtracts a ramp in the gray values and '
                           'normalizes so that in the background there is 0 '
                           'mean and standard deviation 1.')
        form.addParam('normType', EnumParam,
                      choices=['OldXmipp', 'NewXmipp', 'Ramp'], default=2,
                      condition='doNormalize', expertLevel=LEVEL_ADVANCED,
                      display=EnumParam.DISPLAY_COMBO,
                      label='Normalization type',
                      help='OldXmipp (mean(Image)=0, stddev(Image)=1).\n'
                           'NewXmipp (mean(background)=0, '
                           'stddev(background)=1)\n'
                           'Ramp (subtract background+NewXmipp).')
        form.addParam('backRadius', IntParam, default=-1,
                      condition='doNormalize',
                      label='Background radius', expertLevel=LEVEL_ADVANCED,
                      help='Pixels outside this circle are assumed to be '
                           'noise and their stddev is set to 1. Radius for '
                           'background circle definition (in pix.). If this '
                           'value is 0, then half the box size is used.')
        form.addParallelSection(threads=4, mpi=1)

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._setupBasicProperties()
        # Write pos files for all micrographs
        firstStepId = self._insertFunctionStep('writePosFilesStep')
        # For each micrograph insert the steps, run in parallel
        deps = []

        def _insertMicStep(mic):
            localDeps = [firstStepId]
            fnLast = mic.getFileName()
            micName = pwutils.removeBaseExt(mic.getFileName())

            def getMicTmp(suffix):
                return self._getTmpPath(micName + suffix)

            # Create a list with micrographs operations (programs in xmipp) and
            # the required command line parameters (except input/ouput files)
            micOps = []

            # Check if it is required to downsample your micrographs
            downFactor = self.downFactor.get()

            if self.notOne(downFactor):
                fnDownsampled = getMicTmp("_downsampled.xmp")
                args = "-i %s -o %s --step %f --method fourier"
                micOps.append(('xmipp_transform_downsample',
                               args % (fnLast, fnDownsampled, downFactor)))
                fnLast = fnDownsampled

            if self.doRemoveDust:
                fnNoDust = getMicTmp("_noDust.xmp")
                args = " -i %s -o %s --bad_pixels outliers %f"
                micOps.append(('xmipp_transform_filter',
                               args % (fnLast, fnNoDust, self.thresholdDust)))
                fnLast = fnNoDust

            if self._useCTF() and self.ctfDict[mic] is not None:
                # We need to write a Xmipp ctfparam file
                # to perform the phase flip on the micrograph
                fnCTF = self._getTmpPath("%s.ctfParam" % micName)
                mic.setCTF(self.ctfDict[mic])
                micrographToCTFParam(mic, fnCTF)
                # Insert step to flip micrograph
                if self.doFlip:
                    fnFlipped = getMicTmp('_flipped.xmp')
                    args = " -i %s -o %s --ctf %s --sampling %f"
                    micOps.append(('xmipp_ctf_phase_flip',
                                   args % (fnLast, fnFlipped, fnCTF,
                                           self._getNewSampling())))
                    fnLast = fnFlipped
            else:
                fnCTF = None

            # Actually extract
            deps.append(self._insertFunctionStep('extractParticlesStep',
                                                 mic.getObjId(), micName,
                                                 fnCTF, fnLast, micOps,
                                                 self.doInvert.get(),
                                                 self._getNormalizeArgs(),
                                                 self.doBorders.get(),
                                                 prerequisites=localDeps))

        for mic in self.ctfDict:
            _insertMicStep(mic)

        metaDeps = self._insertFunctionStep('createMetadataImageStep',
                                            prerequisites=deps)

        # Insert step to create output objects
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=[metaDeps], wait=False)

    # --------------------------- STEPS functions ------------------------------
    def writePosFilesStep(self):
        """ Write the pos file for each micrograph in metadata format
        (both untilted and tilted). """
        writeSetOfCoordinates(self._getExtraPath(),
                              self.inputCoords.getUntilted(),
                              scale=self.getBoxScale())
        writeSetOfCoordinates(self._getExtraPath(),
                              self.inputCoords.getTilted(),
                              scale=self.getBoxScale())

        # We need to find the mapping by micName (without ext) between the
        #  micrographs in the SetOfCoordinates and the Other micrographs
        if self._micsOther():
            micDict = {}
            for micU, micT in izip(self.inputCoords.getUntilted().getMicrographs(),
                                   self.inputCoords.getTilted().getMicrographs()):
                micBaseU = pwutils.removeBaseExt(micU.getFileName())
                micPosU = self._getExtraPath(micBaseU + ".pos")
                micDict[pwutils.removeExt(micU.getMicName())] = micPosU
                micBaseT = pwutils.removeBaseExt(micT.getFileName())
                micPosT = self._getExtraPath(micBaseT + ".pos")
                micDict[pwutils.removeExt(micT.getMicName())] = micPosT

            # now match micDict and other mics (in self.ctfDict)
            if any(pwutils.removeExt(mic.getMicName()) in micDict for mic in self.ctfDict):
                micKey = lambda mic: pwutils.removeExt(mic.getMicName())
            else:
                raise Exception('Could not match input micrographs and coordinates '
                                'by micName.')

            for mic in self.ctfDict:
                mk = micKey(mic)
                if mk in micDict:
                    micPosCoord = micDict[mk]
                    if exists(micPosCoord):
                        micBase = pwutils.removeBaseExt(mic.getFileName())
                        micPos = self._getExtraPath(micBase + ".pos")
                        if micPos != micPosCoord:
                            self.info('Moving %s -> %s' % (micPosCoord, micPos))
                            pwutils.moveFile(micPosCoord, micPos)

    def extractParticlesStep(self, micId, baseMicName, fnCTF,
                             micrographToExtract, micOps,
                             doInvert, normalizeArgs, doBorders):
        """ Extract particles from one micrograph """
        outputRoot = str(self._getExtraPath(baseMicName))
        fnPosFile = self._getExtraPath(baseMicName + ".pos")

        # If it has coordinates extract the particles
        particlesMd = 'particles@%s' % fnPosFile
        boxSize = self.boxSize.get()
        boxScale = self.getBoxScale()
        print("boxScale: ", boxScale)

        if exists(fnPosFile):
            # Apply first all operations required for the micrograph
            for program, args in micOps:
                self.runJob(program, args)

            args = " -i %s --pos %s" % (micrographToExtract, particlesMd)
            args += " -o %s --Xdim %d" % (outputRoot, boxSize)

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
                            '-i %s.stk %s' % (outputRoot, normalizeArgs))
        else:
            self.warning("The micrograph %s hasn't coordinate file! "
                         % baseMicName)
            self.warning("Maybe you picked over a subset of micrographs")

        # Let's clean the temporary mrc micrographs
        if not pwutils.envVarOn("SCIPION_DEBUG_NOCLEAN"):
            pwutils.cleanPattern(self._getTmpPath(baseMicName) + '*')

    def createMetadataImageStep(self):
        mdUntilted = md.MetaData()
        mdTilted = md.MetaData()
        # for objId in mdPairs:
        for uMic, tMic in izip(self.uMics, self.tMics):
            umicName = pwutils.removeBaseExt(uMic.getFileName())
            fnMicU = self._getExtraPath(umicName + ".xmd")
            fnPosU = self._getExtraPath(umicName + ".pos")
            # Check if there are picked particles in these micrographs
            if pwutils.exists(fnMicU):
                mdMicU = md.MetaData(fnMicU)
                mdPosU = md.MetaData('particles@%s' % fnPosU)
                mdPosU.merge(mdMicU)
                mdUntilted.unionAll(mdPosU)
                tmicName = pwutils.removeBaseExt(tMic.getFileName())
                fnMicT = self._getExtraPath(tmicName + ".xmd")
                fnPosT = self._getExtraPath(tmicName + ".pos")
                mdMicT = md.MetaData(fnMicT)
                mdPosT = md.MetaData('particles@%s' % fnPosT)
                mdPosT.merge(mdMicT)
                mdTilted.unionAll(mdPosT)

        # Write image metadata (check if it is really necessary)
        fnTilted = self._getExtraPath("images_tilted.xmd")
        fnUntilted = self._getExtraPath("images_untilted.xmd")
        mdUntilted.write(fnUntilted)
        mdTilted.write(fnTilted)

    def createOutputStep(self):
        fnTilted = self._getExtraPath("images_tilted.xmd")
        fnUntilted = self._getExtraPath("images_untilted.xmd")

        # Create outputs SetOfParticles both for tilted and untilted
        imgSetU = self._createSetOfParticles(suffix="Untilted")
        imgSetU.copyInfo(self.uMics)
        imgSetT = self._createSetOfParticles(suffix="Tilted")
        imgSetT.copyInfo(self.tMics)

        sampling = self.getMicSampling() if self._micsOther() else self.getCoordSampling()
        if self._doDownsample():
            sampling *= self.downFactor.get()
        imgSetU.setSamplingRate(sampling)
        imgSetT.setSamplingRate(sampling)

        # set coords from the input, will update later if needed
        imgSetU.setCoordinates(self.inputCoordinatesTiltedPairs.get().getUntilted())
        imgSetT.setCoordinates(self.inputCoordinatesTiltedPairs.get().getTilted())

        # Read untilted and tilted particles on a temporary object (also disabled particles)
        imgSetAuxU = SetOfParticles(filename=':memory:')
        imgSetAuxU.copyInfo(imgSetU)
        readSetOfParticles(fnUntilted, imgSetAuxU, removeDisabled=False)

        imgSetAuxT = SetOfParticles(filename=':memory:')
        imgSetAuxT.copyInfo(imgSetT)
        readSetOfParticles(fnTilted, imgSetAuxT, removeDisabled=False)

        # calculate factor for coords scaling
        factor = 1 / self.samplingFactor
        if self._doDownsample():
            factor /= self.downFactor.get()

        coordsT = self.getCoords().getTilted()
        # For each untilted particle retrieve micId from SetOfCoordinates untilted
        for imgU, coordU in izip(imgSetAuxU, self.getCoords().getUntilted()):
            # FIXME: Remove this check when sure that objIds are equal
            id = imgU.getObjId()
            if id != coordU.getObjId():
                raise Exception('ObjIds in untilted img and coord are not the same!!!!')
            imgT = imgSetAuxT[id]
            coordT = coordsT[id]

            # If both particles are enabled append them
            if imgU.isEnabled() and imgT.isEnabled():
                if self._micsOther() or self._doDownsample():
                    coordU.scale(factor)
                    coordT.scale(factor)
                imgU.setCoordinate(coordU)
                imgSetU.append(imgU)
                imgT.setCoordinate(coordT)
                imgSetT.append(imgT)

        if self.doFlip:
            imgSetU.setIsPhaseFlipped(self.ctfUntilt.hasValue())
            imgSetU.setHasCTF(self.ctfUntilt.hasValue())
            imgSetT.setIsPhaseFlipped(self.ctfTilt.hasValue())
            imgSetT.setHasCTF(self.ctfTilt.hasValue())
        imgSetU.write()
        imgSetT.write()

        # Define output ParticlesTiltPair
        outputset = ParticlesTiltPair(filename=self._getPath('particles_pairs.sqlite'))
        outputset.setTilted(imgSetT)
        outputset.setUntilted(imgSetU)
        for imgU, imgT in izip(imgSetU, imgSetT):
            outputset.append(TiltPair(imgU, imgT))

        outputset.setCoordsPair(self.inputCoordinatesTiltedPairs.get())
        self._defineOutputs(outputParticlesTiltPair=outputset)
        self._defineSourceRelation(self.inputCoordinatesTiltedPairs, outputset)

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        # doFlip can only be selected if CTF information
        # is available on picked micrographs
        if self.doFlip and not self._useCTF():
            errors.append('Phase flipping cannot be performed unless '
                          'CTF information is provided.')

        if self.doNormalize:
            if self.backRadius > int(self.boxSize.get() / 2):
                errors.append("Background radius for normalization should be "
                              "equal or less than half of the box size.")
        return errors

    def _citations(self):
        return ['delaRosaTrevin2013']

    def _summary(self):
        summary = []
        summary.append("Micrographs source: %s"
                       % self.getEnumText('downsampleType'))
        summary.append("Particle box size: %d" % self.boxSize)

        if not hasattr(self, 'outputParticlesTiltPair'):
            summary.append("Output images not ready yet.")
        else:
            summary.append("Particle pairs extracted: %d" %
                           self.outputParticlesTiltPair.getSize())

        if self.doFlip:
            if self.ctfTilt.hasValue() and self.ctfUntilt.hasValue():
                summary.append('Phase flipped for both untilted and tilted particles.')
            elif self.ctfUntilt.hasValue():
                summary.append('Phase flipped only for untilted particles.')
            else:
                summary.append('Phase flipped only for tilted particles.')

        return summary

    def _methods(self):
        methodsMsgs = []

        if self.getStatus() == STATUS_FINISHED:
            msg = "A total of %d particle pairs of size %d were extracted" % \
                  (self.getOutput().getSize(), self.boxSize)
            if self._micsOther():
                msg += " from another set of micrographs: %s" % \
                       self.getObjectTag('inputMicrographsTiltedPair')

            msg += " using coordinates %s" % self.getObjectTag('inputCoordinatesTiltedPairs')
            msg += self.methodsVar.get('')
            methodsMsgs.append(msg)

            if self.doRemoveDust:
                methodsMsgs.append("Removed dust over a threshold of %s." % (self.thresholdDust))
            if self.doInvert:
                methodsMsgs.append("Inverted contrast on images.")
            if self._doDownsample():
                methodsMsgs.append("Particles downsampled by a factor of %0.2f." % self.downFactor)
            if self.doNormalize:
                methodsMsgs.append("Normalization performed of type %s." %
                                   (self.getEnumText('normType')))
            if self.doFlip:
                methodsMsgs.append("Phase flipping was performed.")

        return methodsMsgs

    # --------------------------- UTILS functions ------------------------------
    def _setupBasicProperties(self):
        # Get sampling rate and inputMics according to micsSource type
        self.inputCoords = self.getCoords()
        self.uMics, self.tMics = self.getInputMicrographs()
        ctfUntilt = self.ctfUntilt.get() if self.ctfUntilt.hasValue() else None
        ctfTilt = self.ctfTilt.get() if self.ctfTilt.hasValue() else None
        self.samplingFactor = float(self.getMicSampling() / self.getCoordSampling())

        # create a ctf dict with unt/tilt mics
        self.ctfDict = {}
        for micU in self.uMics:
            if ctfUntilt is not None:
                micBase = pwutils.removeExt(self.getMicNameOrId(micU))
                for ctf in ctfUntilt:
                    ctfMicName = self.getMicNameOrId(ctf.getMicrograph())
                    ctfMicBase = pwutils.removeExt(ctfMicName)
                    if micBase == ctfMicBase:
                        self.ctfDict[micU.clone()] = ctf.clone()
                        break
            else:
                self.ctfDict[micU.clone()] = None

        for micT in self.tMics:
            if ctfTilt is not None:
                micBase = pwutils.removeExt(self.getMicNameOrId(micT))
                for ctf in ctfTilt:
                    ctfMicName = self.getMicNameOrId(ctf.getMicrograph())
                    ctfMicBase = pwutils.removeExt(ctfMicName)
                    if micBase == ctfMicBase:
                        self.ctfDict[micT.clone()] = ctf.clone()
                        break
            else:
                self.ctfDict[micT.clone()] = None

    def getInputMicrographs(self):
        """ Return pairs of micrographs associated to the SetOfCoordinates or
        Other micrographs. """
        if not self._micsOther():

            return self.inputCoordinatesTiltedPairs.get().getUntilted().getMicrographs(), \
                   self.inputCoordinatesTiltedPairs.get().getTilted().getMicrographs()
        else:
            return self.inputMicrographsTiltedPair.get().getUntilted(), \
                   self.inputMicrographsTiltedPair.get().getTilted()

    def getCoords(self):
        return self.inputCoordinatesTiltedPairs.get()

    def getOutput(self):
        if self.hasAttribute('outputParticlesTiltPair') and self.outputParticlesTiltPair.hasValue():
            return self.outputParticlesTiltPair
        else:
            return None

    def getCoordSampling(self):
        return self.getCoords().getUntilted().getMicrographs().getSamplingRate()

    def getMicSampling(self):
        return self.getInputMicrographs()[0].getSamplingRate()

    def _getNewSampling(self):
        newSampling = self.getMicSampling()

        if self._doDownsample():
            # Set new sampling, it should be the input sampling of the used
            # micrographs multiplied by the downFactor
            newSampling *= self.downFactor.get()

        return newSampling

    def notOne(self, value):
        return abs(value - 1) > 0.0001

    def _doDownsample(self):
        return self.downFactor > 1.0

    def _getNormalizeArgs(self):
        if not self.doNormalize:
            return ''

        normType = self.getEnumText("normType")
        args = "--method %s " % normType

        if normType != "OldXmipp":
            bgRadius = self.backRadius.get()
            if bgRadius <= 0:
                bgRadius = int(self.boxSize.get() / 2)
            args += " --background circle %d" % bgRadius

        return args

    def getBoxScale(self):
        """ Computing the sampling factor between input and output.
        We should take into account the differences in sampling rate between
        micrographs used for picking and the ones used for extraction.
        The downsampling factor could also affect the resulting scale.
        """
        f = self.getCoordSampling() / self.getMicSampling()
        return f / self.downFactor.get() if self._doDownsample() else f

    def _micsOther(self):
        """ Return True if other micrographs are used for extract. """
        return self.downsampleType == OTHER

    def _useCTF(self):
        return self.ctfUntilt.hasValue() or self.ctfTilt.hasValue()

    def getMicNameOrId(self, mic):
        return mic.getMicName() or mic.getObjId()
