# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_1_1
from pyworkflow.utils.path import makePath, removeBaseExt

from xmipp3.convert import *
from .protocol_particle_pick_pairs import XmippProtParticlePickingPairs


TYPE_COORDINATES = 0
TYPE_PARTICLES = 1


class XmippProtAssignmentTiltPair(XmippProtParticlePickingPairs):
    """    
    Determines the affine transformation between two sets of points obtained
    from tilted micrograph pairs. This transformation aligns coordinate sets
    for accurate particle matching and analysis.

    AI Generated:

    What this protocol is for

    Assign tiltpairs is designed for workflows that use paired micrographs
    acquired at two tilts (an untilted image and a tilted image of the same
    area). Its goal is to match particles between the untilted and tilted
    views by estimating the geometric relationship between both coordinate
    systems. Concretely, it determines an affine transformation that maps
    points picked in the untilted micrograph to their corresponding points in
    the tilted micrograph (and vice versa), and it uses this to establish
    reliable particle correspondences.

    For a biological user, this is a key step when working with tilt-pair
    strategies (for example, validating orientations, studying handedness,
    improving angular assignment robustness, or performing analyses where you
    need to know which tilted particle corresponds to which untilted particle).
    Instead of relying on manual pairing, the protocol automates matching in a
    way that is tolerant to shift, tilt geometry, and the typical distortions
    introduced by tilting.

    Inputs: what you need to provide

    You start by selecting a MicrographsTiltPair object, i.e., a collection of
    paired micrographs where each pair contains an untilted micrograph and its
    corresponding tilted micrograph. This is the backbone that tells the
    protocol which images belong together.

    Then you provide the particle information in one of two forms, controlled
    by Input type:

    If you choose Coordinates, you will provide two coordinate sets: one for
    the untilted micrographs and one for the tilted micrographs. These are
    typically the outputs of picking steps (manual or automatic) performed
    separately on untilted and tilted images.

    If you choose Particles, you will provide two particle sets instead: an
    untilted particle set and a tilted particle set. In this case, the protocol
    uses the coordinates stored inside each particle set. This option is
    convenient when picking has already been followed by extraction and you
    want to work directly with particle objects rather than coordinate-only
    objects.

    A critical practical requirement is that the untilted and tilted inputs
    must be of the same type (both coordinates or both particles). The protocol
    enforces this because it needs a consistent representation on both sides
    of the pairing.

    What the protocol does during processing

    For each micrograph pair, the protocol takes the two point clouds
    (untilted picks and tilted picks) and searches for the best mapping between
    them under an affine model, allowing for the expected effects of tilting
    plus practical nuisances such as global shifts. Once a mapping is found,
    it establishes a correspondence between points across the pair. In parallel,
    it also estimates the tilt axis from the matched coordinates, which is
    often useful for downstream tilt-pair interpretation.

    From the user perspective, the protocol transforms two separate picking
    results into a coherent “paired” coordinate dataset where the same physical
    particle is recognized across both tilts.

    The parameters you will most likely care about

    The Tilt angle parameter is optional but can be very helpful. If you enter
    a tilt angle (in degrees), the protocol will focus the search around that
    value, exploring an interval of approximately ±15 degrees. This can speed
    up matching and make it more robust when you have prior knowledge from the
    acquisition. If you leave it at the default (−1), the protocol will proceed
    without assuming prior tilt-angle information, which is convenient when
    metadata is missing or uncertain.

    The Threshold value controls how strict the matching is. It is expressed as
    a fraction of the particle box size: a candidate match between a tilted
    point and an untilted point is accepted only if the distance is smaller than
    threshold × particle size. Biologically, this threshold is the knob that
    balances sensitivity and specificity. If it is too permissive, you may
    accept incorrect matches, especially in dense particle fields or
    contaminated areas. If it is too strict, true matches may be rejected when
    there are distortions, picking imprecision, or moderate drift between
    tilts. The default is a moderate value that often works well, but if you
    see too many mismatches or too few matches, this is typically the first
    parameter to adjust.

    The Maximum shift (pixels) specifies how much global displacement is
    allowed between the tilted and untilted micrographs. This matters because
    tilt pairs can be shifted due to stage movement or imperfect re-centering.
    If you set this too low, correct pairings may fail in datasets where the
    two images are significantly shifted. If you set it excessively high, the
    search space becomes larger and matching can become more ambiguous in
    crowded micrographs. If you have a good idea of how well-centered your
    tilt pairs are, tailoring this parameter can improve robustness.

    What is considered “particle size” in this protocol

    The protocol needs a notion of particle size to interpret the threshold.
    If you use particle sets, it uses the particle box dimension directly. If
    you use coordinate sets, it uses the box size stored in the coordinate set.
    In practice, this means your coordinate set should have a meaningful box
    size (typically matching what you would extract later), because it controls
    the distance scale used for matching.

    Outputs and how to use them

    The protocol produces matched outputs via its “tilt-pair coordinate
    registration” step. In Scipion terms, the typical outcome is an output
    tilt-pair coordinate set (often represented as a specialized tilt-pair
    object or a registered coordinate set, depending on the surrounding
    workflow) where correspondences between untilted and tilted picks are
    established and stored. This output is what you use in downstream tilt-pair
    processing steps, including any analysis that compares alignment parameters
    between tilts or uses tilt-pair information to validate projection
    directions.

    Additionally, the estimated tilt axis is computed as part of processing and
    becomes available for interpretation and subsequent steps that benefit
    from knowing the tilt geometry.

    Practical interpretation and biological usage patterns

    A common biological workflow is to run picking (or extraction) independently
    on the untilted and tilted images, then run Assign tiltpairs to match
    particles across the pair. After that, you can proceed with analyses that
    depend on having the same particle observed under two tilts. This can be
    useful for diagnosing orientation assignments, checking for
    mirror/handedness issues, or providing additional constraints in advanced
    angular validation strategies.

    If you find that matching fails or produces suspicious results, the most
    common causes are (i) incorrect pairing of micrographs (wrong
    untilted/tilted association), (ii) different particle populations picked in
    the two images (for example, one image includes many contaminants or a
    different picking threshold), (iii) a threshold that is too strict or too
    permissive, or (iv) a maximum shift that does not reflect the real
    displacement between images.
    """
    _label = 'assign tiltpairs'
    _lastUpdateVersion = VERSION_1_1
    def __init__(self, *args, **kwargs):
        XmippProtParticlePickingPairs.__init__(self, *args, **kwargs)
        self.stepsExecutionMode = params.STEPS_PARALLEL

        def onChangeInputType():
            """ Dynamically change the PointerClass depending on the
             selected input type (either Coordinates or Particles).
            """
            pointerClass = 'SetOf%s' % self.getEnumText('inputType')
            self.getParam('untiltedSet').setPointerClass(pointerClass)
            self.getParam('tiltedSet').setPointerClass(pointerClass)

        self.inputType.trace(onChangeInputType)

        # Force callback function
        onChangeInputType()

    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMicrographsTiltedPair', params.PointerParam,
                      pointerClass='MicrographsTiltPair',
                      important=True,
                      label="Micrograph tilt pair",
                      help='Select micrographs tilt pair.')

        form.addParam('inputType', params.EnumParam,
                      choices=['Coordinates', 'Particles'], default=0,
                      display=params.EnumParam.DISPLAY_COMBO,
                      label='Input type',
                      help='Select a Set of Coordinates or a Set or Particles.')

        form.addParam('untiltedSet', params.PointerParam,
                      pointerClass='SetOfCoordinates,SetOfParticles',
                      label="Untilted input",
                      help='Select the untilted input set, it can be either '
                           'coordinates or particles (that contains coordinates.')

        form.addParam('tiltedSet', params.PointerParam,
                      pointerClass='SetOfCoordinates,SetOfParticles',
                      label="Tilted input",
                      help='Select the tilted input set, it can be either '
                           'coordinates or particles (that contains coordinates. '
                           'It should be of the same type of the input untilted.')

        form.addParam('tiltAngle', params.FloatParam, default=-1,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Tilt angle",
                      help='Tilt angle estimation, the method will look for '
                           'the assignment in the interval of [tilt_angle-15, '
                           'tilt_angle+15].\n By default: tilt angle = -1, if '
                           'there is not any information about the tilt angle')

        form.addParam('threshold', params.FloatParam, default=0.25,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Threshold value",
                      help='Parameter between 0 and 1 that allows to define if \n'
                      'a tilt point can be matched with a certain untilt point.\n'
                      'The matching is performed only if the distance is lesser\n'
                      'than threshold * particlesize.')

        form.addParam('maxShift', params.FloatParam, default=1500,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Maximum shift (pixels)",
                      help='Maximum allowed distance (in pixels) that the tilt '
                           'micrograph can be shifted respect to the untilted '
                           'micrograph')

        form.addParallelSection(threads=1, mpi=1)

    #--------------------------- INSERT steps functions --------------------------------------------

    def _particlesPos(self, *parts):
        return 'particles@%s.pos' % self._getExtraPath(*parts)

    def _micBaseName(self, mic):
        return removeBaseExt(mic.getFileName())

    def _coordsFromParts(self, micSet, partSet, suffix):
        """ Create and fill a SetOfCoordinates from a given SetOfParticles. """
        coordSet = self._createSetOfCoordinates(micSet, suffix='_untilted')

        for particle in partSet:
            coord = particle.getCoordinate().clone()
            coord.copyObjId(particle)
            coordSet.append(coord)
        coordSet.setBoxSize(partSet.getXDim())

        return coordSet

    def _getBoxSize(self):
        untiltedSet = self.untiltedSet.get()
        if isinstance(untiltedSet, SetOfParticles):
            boxSize = untiltedSet.getXDim()
        else:
            boxSize = untiltedSet.getBoxSize() # coordinates
        return boxSize

    def _insertAllSteps(self):
        self.micsFn = self._getPath()
        # Convert input into xmipp Metadata format
        convertId = self._insertFunctionStep('convertInputStep')
        deps = []
        for tiltPair in self.inputMicrographsTiltedPair.get():
            uName = self._micBaseName(tiltPair.getUntilted())
            tName = self._micBaseName(tiltPair.getTilted())
            stepId = self._insertFunctionStep('assignmentStep',
                                              self._particlesPos("untilted", uName),
                                              self._particlesPos("tilted", tName),
                                              tiltPair.getTilted().getFileName(),
                                              self._particlesPos(uName),
                                              self._particlesPos(tName),
                                              prerequisites=[convertId])
            deps.append(stepId)

        self._insertFunctionStep('createOutputStep', prerequisites=deps)

    def convertInputStep(self):
        """ Read the input metadatata. """
        # Get the converted input micrographs in Xmipp format
        makePath(self._getExtraPath("untilted"))
        makePath(self._getExtraPath("tilted"))

        uSet = self.untiltedSet.get()
        tSet = self.tiltedSet.get()

        # Get the untilted and tilted coordinates, depending on the input type
        if isinstance(uSet, SetOfParticles):
            uCoords = uSet.getCoordinates()
            tCoords = tSet.getCoordinates()

            # If there are not Coordinates associated to particles
            # we need to create and fill the set of coordinates
            if uCoords is None or tCoords is None:
                micTiltedPairs = self.inputMicrographsTiltedPair.get()
                uCoords = self._coordsFromParts(micTiltedPairs.getUntilted(),
                                                uSet, '_untilted')
                tCoords = self._coordsFromParts(micTiltedPairs.getTilted(),
                                                tSet, '_tilted')
        else:
            uCoords = uSet
            tCoords = tSet

        writeSetOfCoordinates(self._getExtraPath("untilted"), uCoords)
        writeSetOfCoordinates(self._getExtraPath("tilted"), tCoords)

    def assignmentStep(self, fnuntilt, fntilt, fnmicsize, fnposUntilt, fnposTilt):
        params =  ' --untiltcoor %s' % fnuntilt
        params += ' --tiltcoor %s' % fntilt
        params += ' --tiltmicsize %s' % fnmicsize
        params += ' --maxshift %f' % self.maxShift
        params += ' --particlesize %d' % self._getBoxSize()
        params += ' --threshold %f' % self.threshold
        params += ' --odir %s' % self._getExtraPath()
        self.runJob('xmipp_image_assignment_tilt_pair', params)

        # Estimate the tilt axis
        params =  ' --untilted %s' % fnposUntilt
        params += ' --tilted %s' % fnposTilt
        params += ' -o %s' % self._getPath('input_micrographs.xmd')
        self.runJob('xmipp_angular_estimate_tilt_axis', params)

    def createOutputStep(self):
        self.registerCoords(self._getExtraPath(), store=False)

    #--------------------------- INFO functions -------------------------------------------- 
    def _validate(self):
        errors = []
        uSet = self.untiltedSet.get()
        tSet = self.tiltedSet.get()

        if (uSet is not None and tSet is not None and
            uSet.getClassName() != tSet.getClassName()):
            errors.append('Both untilted and tilted inputs should be of the '
                          'same type. ')

        return errors

    def _methods(self):
        messages = []
        if hasattr(self,'outputCoordinatesTiltPair'):
            messages.append('The assignment has been performed using and '
                            'affinity transformation [Publication: Vilas2016]')
        return messages

    def _citations(self):
        return ['Vilas2016']

