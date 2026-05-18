# coding=utf-8
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *              Carlos Oscar Sanchez Sorzano
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia, CSIC
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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import PointerParam, FloatParam, BooleanParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils.path import cleanPattern
from pwem.protocols import ProtAnalysis3D
from pwem import emlib

from xmipp3.convert import readSetOfParticles, writeSetOfParticles

CITE ='Fernandez-Gimenez2023b'


class XmippProtLocalCTF(ProtAnalysis3D):
    """Compares a set of particles with the corresponding projections of a reference volume.
    The set of particles must have a 3D angular assignment.
    This protocol refines the CTF, computing local defocus change.
    The maximun allowed defocus is a parameter introduced by the user (advanced).
    The protocol gives back the input set of particles with the refine local
    defocus and the defocus change with relation to the global defocus.

    AI Generated

    ## Overview

    The Estimate Local Defocus protocol refines the CTF information of a set of
    particles by estimating a local defocus correction for each particle.

    In standard cryo-EM processing, CTF estimation is usually performed at the
    micrograph level. This means that all particles from the same micrograph are
    initially assigned the same global defocus parameters, or parameters derived
    from a fitted defocus model. However, the true defocus can vary locally across
    the micrograph because of sample tilt, ice curvature, specimen height
    variation, beam-induced deformation, or other acquisition effects.

    This protocol refines the defocus at the particle level by comparing each
    particle with the corresponding projection of a reference volume. The input
    particles must already have 3D projection alignment information, because the
    protocol needs to know which projection direction of the volume corresponds to
    each particle.

    The output is a new set of particles with refined local defocus information and
    an additional defocus-change value relative to the original defocus.

    ## Inputs and General Workflow

    The protocol requires two main inputs:

    - a set of particles with projection alignment;
    - a reference volume.

    The particles are first converted to Xmipp metadata format. The reference
    volume is also converted to Xmipp format and resized if necessary so that its
    box size matches the particle images.

    The protocol then compares each particle with the corresponding projection of
    the reference volume and optimizes the defocus parameters. During this process,
    it can also optimize simple gray-scale differences between the experimental
    particle and the reprojection.

    Finally, the refined CTF columns are merged back into the particle metadata and
    a new Scipion particle set is produced.

    ## Input Images

    The **Input images** parameter should point to a SetOfParticles with 3D
    projection alignment information.

    This requirement is essential. The protocol does not determine the particle
    orientation from scratch. Instead, it assumes that each particle already has a
    valid projection direction and uses that direction to compare the particle with
    the reference volume.

    If the angular assignments are poor, the local defocus refinement may also be
    poor. Apparent CTF differences may then partly reflect alignment errors,
    conformational mismatch, or particle heterogeneity rather than true defocus
    variation.

    This protocol is therefore best used after a reliable 3D refinement or angular
    assignment step.

    ## Reference Volume

    The **Volume to compare images to** is the 3D map used as the structural
    reference during local defocus refinement.

    The volume should represent the same structure as the input particles and
    should be good enough to generate meaningful projections. If the volume is
    low-quality, strongly biased, or corresponds to a different conformational
    state, the defocus refinement may become unreliable.

    Before the refinement, the protocol checks whether the volume and particle
    images have the same box size. If not, the volume is resized to match the
    particle dimensions. This allows projections of the volume to be compared
    directly with the particle images.

    ## Maximum Defocus Change

    The **Maximum defocus change** parameter defines the largest allowed correction
    to the particle defocus, in angstroms.

    This parameter constrains the local refinement around the original defocus
    value. It prevents the optimization from moving to unrealistically large
    defocus changes that may fit noise or compensate for other errors.

    A smaller value makes the refinement more conservative. This is appropriate
    when the original CTF estimation is expected to be accurate and only small
    local deviations are expected.

    A larger value allows more flexibility. This may be useful for tilted samples,
    thick ice, strongly curved specimens, or datasets where substantial local
    defocus variation is expected.

    However, increasing this value too much can make the refinement less stable.
    The local defocus correction should remain physically plausible.

    ## Gray-Scale Optimization

    The protocol can also optimize simple gray-scale differences between each
    particle and the corresponding projection of the volume.

    The reprojection can be modified as:

    \[
    aP + b
    \]

    where \(P\) is the projection, \(a\) is a scale factor, and \(b\) is an
    intensity shift.

    The **Maximum gray scale change** parameter restricts the allowed range of the
    scale factor. The scale factor is constrained to the interval:

    \[
    [1 - \text{maxGrayScaleChange},\; 1 + \text{maxGrayScaleChange}]
    \]

    The **Maximum gray shift change** parameter restricts the allowed additive
    intensity shift to:

    \[
    [-\text{maxGrayShiftChange},\; \text{maxGrayShiftChange}]
    \]

    These options allow the comparison to tolerate moderate intensity differences
    between experimental particles and model projections. This is useful because
    particles may differ in normalization, local background, or contrast strength.

    These are advanced parameters. In most workflows, the default values should be
    used unless there is a clear reason to restrict or relax gray-scale
    optimization.

    ## Same Defocus in U and V

    The **Force defocusV to be equal than defocusU** option constrains the refined
    CTF to have the same defocus in the two principal directions.

    Normally, the CTF may include astigmatism. In that case, defocusU and defocusV
    can be different, corresponding to different defocus values along two
    directions.

    If this option is enabled, the protocol forces both values to be equal. This
    removes astigmatism from the local refinement and estimates only a single
    defocus value per particle.

    This option may be useful when the user wants a simpler local defocus model or
    when astigmatism is expected to be negligible. It should not be used if real
    astigmatism is present and should be preserved.

    ## Phase-Flipped Particles

    If the input particles are marked as phase-flipped, the protocol passes this
    information to the refinement program.

    This is important because phase-flipped images have already been modified to
    compensate for CTF phase reversals. The comparison between particles and
    projections must take this processing state into account.

    Users should therefore ensure that the particle metadata correctly describe
    whether the particles have been phase-flipped.

    ## Local Defocus Change

    One of the most important values produced by the protocol is the local
    **defocus change**.

    This value describes how much the refined particle defocus differs from the
    original defocus value. It can be interpreted as a particle-level correction to
    the global or previously assigned CTF estimate.

    Small defocus changes indicate that the original CTF estimate was already
    consistent with the particle and reference volume. Larger changes may indicate
    local height variation, specimen tilt, local ice curvature, or inaccuracies in
    the original CTF model.

    Very large or spatially inconsistent changes should be interpreted with care,
    because they may also reflect poor alignment, wrong particles, structural
    heterogeneity, or mismatch between particles and the reference volume.

    ## Outputs and Their Interpretation

    The main output is a new **SetOfParticles**.

    This output preserves the information from the input particles but includes
    refined CTF values and the local defocus-change information computed by the
    protocol.

    The output particles can be used in later refinement or reconstruction steps
    that benefit from more accurate particle-level CTF information.

    The output should be interpreted as the same particle dataset with improved
    local CTF metadata, not as a new classification or filtering of particles.

    ## Practical Recommendations

    Use this protocol only after the particles have reliable 3D projection
    alignment. The method depends on comparing each particle with the correct
    projection of the reference volume.

    Use a reference volume that corresponds to the same structural state as the
    particles. If the dataset contains strong conformational heterogeneity, it may
    be better to refine local defocus separately for more homogeneous subsets.

    Start with the default maximum defocus change unless there is a specific reason
    to expect larger local defocus variation.

    Be cautious when allowing very large defocus changes. A better numerical fit
    does not necessarily mean a physically meaningful CTF correction.

    Use the same-defocus option only when astigmatism should be ignored or when a
    single-defocus local model is desired.

    After running the protocol, inspect the distribution of defocus changes. Values
    should be compatible with the expected experimental geometry and sample
    properties.

    ## Final Perspective

    Estimate Local Defocus is a particle-level CTF refinement protocol. It improves
    the CTF description of each particle by using the structural information
    contained in a reference volume and the known projection direction of each
    particle.

    For biological users, the main value of this protocol is that it can correct
    local defocus variations that are not captured by a single micrograph-level CTF
    estimate. This can be especially useful for tilted specimens, thick samples,
    curved ice, or high-resolution projects where small CTF inaccuracies matter.

    The protocol should be used carefully, with a reliable reference volume and
    good angular assignments, because local defocus refinement can otherwise absorb
    errors that come from alignment, heterogeneity, or model mismatch rather than
    from true CTF variation.
    """
    _label = 'estimate local defocus'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images",
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj')
        form.addParam('inputVolume', PointerParam, label="Volume to compare images to",
                      pointerClass='Volume',
                      help='Volume to be used for class comparison')
        form.addParam('maxDefocusChange', FloatParam, label="Maximum defocus change (A)", default=500,
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('maxGrayScaleChange', FloatParam, label="Maximum gray scale change", default=1,
                      expertLevel=LEVEL_ADVANCED, help="The reprojection is modified as a*P+b, a is restricted to the "
                                                       "interval [1-maxGrayScale,1+maxGrayScale]")
        form.addParam('maxGrayShiftChange', FloatParam, label="Maximum gray shift change", default=1,
                      expertLevel=LEVEL_ADVANCED, help="The reprojection is modified as a*P+b, b is restricted to the "
                                                       "interval [-maxGrayShift,maxGrayShift]")
        form.addParam('sameDefocus', BooleanParam, label="Force defocusV to be equal than defocusU", default=False,
                      expertLevel=LEVEL_ADVANCED,
                      help="As the CTF usually suffers from astigmatism (it is not spherical but ellipsoidal), the "
                           "defocus vary if computed in X or Y direction, being defocus U value the defocus in X "
                           "direction and defocus V value the defocus in Y direction.")
        form.addParallelSection(threads=0, mpi=8)
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("refineDefocus")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        """convert input to proper format and dimensions if necessary"""
        imgSet = self.inputSet.get()
        writeSetOfParticles(imgSet, self._getExtraPath('input_imgs.xmd'))
        img = emlib.image.ImageHandler()
        fnVol = self._getExtraPath("volume.vol")
        img.convert(self.inputVolume.get(), fnVol)
        xDimVol=self.inputVolume.get().getDim()[0]
        xDimImg = imgSet.getDim()[0]
        if xDimVol!=xDimImg:
            self.runJob("xmipp_image_resize", "-i %s --dim %d"%(fnVol,xDimImg),numberOfMpi=1)

    def refineDefocus(self):
        """compute local defocus using Xmipp (xmipp_angular_continuous_assign2) and add to metadata columns related to
        defocus"""
        fnVol = self._getExtraPath("volume.vol")
        fnIn = self._getExtraPath('input_imgs.xmd')
        fnOut = self._getExtraPath('output_imgs.xmd')
        anglesOutFn = self._getExtraPath("anglesCont.stk")
        Ts = self.inputSet.get().getSamplingRate()
        args = "-i %s -o %s --ref %s --optimizeDefocus --max_defocus_change %d --sampling %f --optimizeGray " \
               "--max_gray_scale %f --max_gray_shift %f " % \
               (fnIn, anglesOutFn, fnVol, self.maxDefocusChange.get(), Ts, self.maxGrayScaleChange.get(),
                self.maxGrayShiftChange.get())
        if self.inputSet.get().isPhaseFlipped():
            args += " --phaseFlipped"
        if self.sameDefocus.get():
            args += " --sameDefocus"
        self.runJob("xmipp_angular_continuous_assign2", args)

        fnCont = self._getExtraPath('anglesCont.xmd')
        self.runJob("xmipp_metadata_utilities", '-i %s --operate keep_column "itemId ctfDefocusU ctfDefocusV '
                                                'ctfDefocusChange ctfDefocusAngle"'%
                    fnCont, numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities",
                    '-i %s -o %s --operate drop_column "ctfDefocusU ctfDefocusV ctfDefocusChange ctfDefocusAngle"' %
                    (fnIn, fnOut), numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities",
                    "-i %s --set join %s itemId itemId" % (fnOut, fnCont), numberOfMpi=1)

        cleanPattern(self._getExtraPath("anglesCont.*"))

    def createOutputStep(self):
        """create scipion output data from metadata"""
        outputSet = self._createSetOfParticles()
        imgSet = self.inputSet.get()
        outputSet.copyInfo(imgSet)
        readSetOfParticles(self._getExtraPath('output_imgs.xmd'), outputSet,
                           extraLabels=[emlib.MDL_CTF_DEFOCUS_CHANGE])
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Refined defocus of %i particles" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        summary.append("Allowed defocus: %s" % self.maxDefocusChange.get())
        return summary
    
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We refined the defocus %i input particles %s regarding to volume %s, allowing a maximun "
                           "defocus of %s" % (self.inputSet.get().getSize(), self.getObjectTag('inputSet'),
                                              self.getObjectTag('inputVolume'), self.maxDefocusChange.get()))
        return methods
