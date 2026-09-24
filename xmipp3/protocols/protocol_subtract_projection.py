# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *           Federico P. de Isidro-Gomez
# *
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

from os.path import basename
from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, FloatParam, EnumParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pwem import emlib
from pwem.protocols import EMProtocol
from xmipp3.convert import writeSetOfParticles, readSetOfParticles
from pyworkflow import PROD, UPDATED

OUTPUT_MRCS = "output_particles.mrcs"
OUTPUT_XMD = "output_particles.xmd"
CITE = 'Fernandez-Gimenez2023a'

class XmippProtSubtractProjectionBase(EMProtocol):
    """ Helper class that contains some Protocol utilities methods
    used by both  XmippProtSubtractProjection and XmippProtBoostParticles.

    AI Generated

    ## Overview

    The Subtract Projection protocol subtracts projections of a reference volume
    from a set of experimental particles.

    In single-particle cryo-EM, each particle image corresponds to a projection of
    the 3D structure in a particular orientation. If the particles already have
    alignment information, the protocol can project a reference volume using those
    same orientations and compare each experimental particle with its corresponding
    reference projection.

    This protocol uses that idea to remove, or alternatively keep, selected
    regions of the particle signal. It is useful when the user wants to focus
    downstream analysis on a specific part of a complex, remove a dominant region,
    or generate particles where a chosen structural component has been subtracted.

    The protocol can use a circular mask or a 3D protein mask during the numerical
    adjustment, can optionally use a region-of-interest mask, can take CTF into
    account, and can use either Fourier or real-space projection.

    The main output is a new particle set containing the processed particles and
    subtraction-related metadata.

    ## Inputs and General Workflow

    The protocol requires:

    - a set of particles;
    - a reference volume;
    - a mask strategy;
    - optionally, a region-of-interest mask.

    The input particles are first written to Xmipp metadata format. The reference
    volume is then projected using the orientations stored in the particle set.
    For each particle, the protocol compares the experimental image with the
    corresponding projection of the reference volume.

    The projection and particle are numerically adjusted, and the selected
    subtraction or keeping operation is applied. The resulting particles are saved
    as a new particle stack and read back into Scipion as an output particle set.

    The output particles include additional metadata describing the subtraction fit.

    ## Input Particles

    The **Particles** parameter defines the particle set to be processed.

    The particles must have projection-alignment information, because the protocol
    uses the orientation of each particle to generate the corresponding projection
    of the reference volume.

    The input particles and reference volume must have the same X and Y
    dimensions, and they must have the same sampling rate. The protocol validates
    these requirements before running.

    This protocol is therefore intended for particles that have already been
    aligned or refined against a compatible reference.

    ## Reference Volume

    The **Reference volume** parameter defines the 3D map that will be projected
    and compared with the particles.

    For each particle, the protocol generates the projection of this volume using
    the particle orientation. This projected reference is then adjusted and used
    for subtraction or signal keeping.

    The reference volume should correspond to the same structure and coordinate
    system as the input particles. If the reference is shifted, rotated, scaled, or
    sampled differently from the particles, the subtraction will be unreliable.

    ## Mask Used for the Adjustment

    The **Mask** parameter controls which region is considered during the numerical
    adjustment between each particle and its corresponding reference projection.

    There are two options:

    **Circular mask** applies a circular mask to every particle.

    **Protein mask** uses a 3D specimen mask indicating the region of the map that
    should be considered in the analysis.

    Pixels outside the selected mask are ignored during the adjustment. This helps
    avoid fitting the reference projection to solvent, background, or irrelevant
    regions.

    ## Circular Mask Radius

    The **Circular mask radius** parameter is used when the mask option is
    **Circular mask**.

    It defines the radius of the circular mask applied to the particles. The
    purpose of this mask is to avoid edge artifacts and restrict the comparison to
    the central particle region.

    If the value is **-1**, the protocol uses half the X dimension of the input
    particles.

    A smaller radius focuses the fit on the central region. A larger radius includes
    more peripheral density and background.

    ## Protein Mask

    The **Mask** volume parameter is used when the mask option is **Protein mask**.

    This mask should define the region of the input volume that must be considered
    during the particle-projection adjustment.

    The mask must have the same dimensions and sampling rate as the reference
    volume. The protocol validates these conditions.

    A good mask should include the density that should drive the fit while
    excluding solvent and irrelevant background. Masks with values in background
    regions should be avoided when they are not intended to contribute to the
    analysis.

    ## ROI Mask

    The **ROI mask** parameter defines the region that the user wants to keep or
    subtract.

    This mask is different from the main fitting mask. The main mask controls the
    region used for numerical adjustment, while the ROI mask defines the structural
    region affected by the final operation.

    If no ROI mask is provided, the subtraction is performed on the whole particle
    image.

    The ROI mask should correspond to the region of the reference volume that the
    user wants to remove or preserve. It should avoid including background voxels
    as part of the region of interest.

    ## Keep or Subtract

    The **Mask contains the part to** parameter controls how the ROI mask is
    interpreted.

    If **Keep** is selected, the protocol keeps the region represented by the ROI
    mask and removes the rest according to the projection-subtraction operation.

    If **Subtract** is selected, the protocol subtracts the region represented by
    the ROI mask from the particles.

    This option determines whether the ROI mask marks the part of the structure to
    retain or the part to remove.

    ## Ignore CTF

    The **Ignore CTF** option controls whether CTF effects are considered during
    the subtraction.

    If this option is disabled, the protocol considers CTF information when
    matching the reference projections to the particles.

    If this option is enabled, CTF is ignored. This is appropriate when the input
    particles have already been CTF corrected or when the user intentionally wants
    to perform the subtraction without CTF modulation.

    Using the wrong CTF setting can produce incomplete or biased subtraction.

    ## Maximum Resolution

    The **Maximum resolution** parameter defines the highest resolution used in the
    subtraction calculation, in angstroms.

    If the value is **-1**, the protocol uses its default behavior, described in
    the help as approximately the sampling divided by square root of 2.

    Limiting the maximum resolution can make the subtraction more stable by
    avoiding high-frequency noise or details not reliably represented by the
    reference volume.

    The value should be greater than 0 if explicitly provided.

    ## Ignore Particles with Negative beta0 or R2

    The **Ignore particles with negative beta0 or R2?** option controls whether
    particles with problematic fitting parameters are excluded from the output.

    The subtraction procedure estimates numerical parameters describing how the
    reference projection fits each particle. Particles with negative beta0 or R2
    are considered bad particles by the protocol.

    When this option is enabled, these particles are not included in the output
    particle set. Negative beta values also do not contribute to the mean beta when
    the corresponding mean option is used internally.

    This option helps remove particles for which the projection fit is not
    reliable.

    ## Projector

    The **Projector** parameter controls how projections of the reference volume
    are generated.

    There are two options:

    **Fourier** projection is faster but may be more prone to artifacts.

    **Real space** projection is slower but more accurate.

    The mask is always projected in real space.

    For exploratory workflows or large datasets, Fourier projection may be
    convenient. For more accurate subtraction, real-space projection may be
    preferable.

    ## Filter Decay Sigma

    The **Decay of the filter (sigma)** parameter is used with the Fourier
    projector.

    It controls the smoothness of the mask transition. A smoother transition can
    reduce edge artifacts during projection and subtraction.

    This is an advanced parameter. Most users should keep the default unless they
    observe mask-related artifacts or have a specific reason to tune the filter
    decay.

    ## Fourier Padding Factor

    The **Fourier padding factor** parameter is used with the Fourier projector.

    It defines how much the volume is zero-padded before generating projections.
    Padding can improve projection accuracy but increases memory use.

    The protocol warns that if the input particles are very large, for example
    larger than about 750 pixels, users may consider reducing the padding factor to
    1 to save RAM.

    ## Output Particles

    The main output is **outputParticles**.

    This output contains the processed particles after projection subtraction or
    signal keeping. The particles are saved in a new MRC stack and metadata file,
    then read back into Scipion.

    The output particle set preserves input particle information and includes
    additional Xmipp metadata labels related to the subtraction fit, including:

    - subtraction R2;
    - beta0;
    - beta1;
    - beta.

    These values can be useful for inspecting how well each particle was explained
    by the reference projection.

    ## Interpreting the Output

    The output particles should be interpreted as modified images in which a
    reference-derived projection has been used to remove or preserve selected
    signal.

    This operation depends strongly on the quality of the reference volume, the
    accuracy of particle orientations, the correctness of the masks, and the CTF
    setting.

    Incomplete subtraction may occur if the reference does not match the particles,
    if the particles are heterogeneous, if alignments are inaccurate, or if the
    mask is poorly defined.

    Over-subtraction or artifacts may occur if the reference projection is fitted
    too aggressively or if the ROI mask includes inappropriate regions.

    ## Practical Recommendations

    Use this protocol only with particles that have reliable projection-alignment
    parameters.

    Make sure the particle box size and sampling rate match the reference volume.

    Use a reference volume in the same coordinate frame as the input particles.

    Use a protein mask when the fitting should focus on the molecular region.
    Use a circular mask for simpler, central-particle masking.

    Define the ROI mask carefully. It should correspond to the region to keep or
    subtract and should not include background unnecessarily.

    Use **Ignore CTF** only when particles have already been CTF corrected or when
    CTF-aware subtraction is not desired.

    Start with conservative maximum-resolution settings. Avoid relying on noisy
    high-frequency information for subtraction.

    Inspect output particles visually and examine subtraction metadata before using
    the output for downstream classification or refinement.

    ## Final Perspective

    Subtract Projection is a focused particle-processing protocol for removing or
    preserving reference-derived signal in aligned particles.

    For biological users, its main value is that it allows analysis to focus on a
    selected region of a larger structure. For example, a dominant stable domain
    can be subtracted so that variability in another region becomes easier to
    classify, or a region of interest can be kept while the rest of the particle is
    suppressed.

    The quality of the result depends on accurate alignments, an appropriate
    reference volume, correct CTF handling, and carefully designed masks.
    """
    _devStatus = PROD

    # --------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineParams(cls, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', 
                      PointerParam, 
                      pointerClass='SetOfParticles', 
                      label="Particles ",
                      help='Specify a SetOfParticles')
        form.addParam('vol', 
                      PointerParam, 
                      pointerClass='Volume', 
                      label="Reference volume ",
                      help='Specify a volume.')
        form.addParam('maskOption',
                EnumParam,
                choices=['Circular mask', 'Protein mask'],
                default=0,
                label='Mask',
                help='Mask to be applied to particles before subtraction. Pixels out to mask are ignored in the analysis:\n'
                     '- Circular mask: circular mask is applied to every particle.\n'
                     '- Protein mask: specimen mask indicating the region of the map that must be conisdered in the analysis.')
        form.addParam('cirmaskrad', 
                      FloatParam, 
                      label="Circular mask radius: ", 
                      default=-1, 
                      expertLevel=LEVEL_ADVANCED,
                      help='Radius of the circular mask to avoid edge artifacts. '
                           'If -1 it is half the X dimension of the input particles',
                      condition='maskOption==0')
        form.addParam('mask', 
                      PointerParam, 
                      pointerClass='VolumeMask', 
                      label='Mask ', 
                      allowsNull=True,
                      help='Specify a 3D mask for the region of the input volume that must be considered in the analysis.',
                      condition='maskOption==1')
        form.addParam('ignoreCTF',
                      BooleanParam,
                      label="Ignore CTF",
                      default=False,
                      help='Do not consider CTF in the subtraction. Use if particles have been CTF corrected.')
        form.addParam('resol', 
                      FloatParam, 
                      label="Maximum resolution (A)", 
                      default=-1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Maximum resolution in Angtroms up to which the substraction is calculated. By default (-1) it is '
                           ' set to sampling/sqrt(2).')
        form.addParam('nonNegative',
                      BooleanParam,
                      label="Ignore particles with negative beta0 or R2?: ",
                      default=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Particles with negative beta0 or R2 will not appear in the output set as they are '
                           'considered bad particles. Moreover, negative betas will not contribute to mean beta if '
                           '"mean" option is selected')
        form.addParam('realSpaceProjection', 
                      EnumParam,
                      choices=['Fourier ', 'Real space'],
                      default=0,
                      label='Projector',
                      help='Projector for the input volume (mask is always projected in real space):\n'
                           '- Fourier: faster but more artifact prone.\n'
                           '- Real space: slower but more accurate.')
        form.addParam('sigma', 
                      FloatParam, 
                      label="Decay of the filter (sigma): ", 
                      default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Decay of the filter (sigma) to smooth the mask transition',
                      condition='realSpaceProjection==0')
        form.addParam('pad', 
                      IntParam, 
                      label="Fourier padding factor: ", 
                      default=2, 
                      expertLevel=LEVEL_ADVANCED,
                      help='The volume is zero padded by this factor to produce projections',
                      condition='realSpaceProjection==0')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertSubSteps()
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputSet)
        readSetOfParticles(self._getExtraPath(OUTPUT_XMD), outputSet,
                           extraLabels=[emlib.MDL_SUBTRACTION_R2,
                                        emlib.MDL_SUBTRACTION_BETA0,
                                        emlib.MDL_SUBTRACTION_BETA1,
                                        emlib.MDL_SUBTRACTION_B])
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputSet, outputSet)


class XmippProtSubtractProjection(XmippProtSubtractProjectionBase):
    """ This protocol computes the subtraction between particles and a reference volume, by computing its projections with the same angles that input particles have. Then, each particle and the correspondent projection of the reference volume are numerically adjusted and subtracted using a mask which denotes the region to keep. """

    _label = 'subtract projection'
    INPUT_PARTICLES = "input_particles.xmd"

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        XmippProtSubtractProjectionBase._defineParams(form)
        form.addParam('maskRoi', PointerParam, pointerClass='VolumeMask', label='ROI mask ', allowsNull=True,
                      help='Specify a 3D mask for the region of the input volume that you want to keep or subtract, '
                           'avoiding masks with 1s in background. If no mask is given, the subtraction is performed in'
                           ' whole images.')
        form.addParam('subtract', EnumParam, default=0, choices=["Keep", "Subtract"], display=EnumParam.DISPLAY_HLIST,
                      label="Mask contains the part to ")
        form.addParallelSection(threads=0, mpi=4)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertSubSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('subtractionStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath(self.INPUT_PARTICLES))

    def subtractionStep(self):
        vol = self.vol.get().clone()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        args = '-i %s --ref %s -o %s --sampling %f --max_resolution %f --padding %f ' \
               '--sigma %d --save %s --save_metadata_stack %s --keep_input_columns ' % \
               (self._getExtraPath(self.INPUT_PARTICLES), fnVol, self._getExtraPath(OUTPUT_MRCS),
                vol.getSamplingRate(), self.resol.get(), self.pad.get(), self.sigma.get(),
                self._getExtraPath(), self._getExtraPath(OUTPUT_XMD))
        
        if self.maskOption.get() == 0:
            args += " --cirmaskrad %d " % self.cirmaskrad.get()
        else:
            fnMask = self.mask.get().getFileName()
            if fnMask.endswith('.mrc'):
                fnMask += ':mrc'
            args += " --mask %s" % fnMask

        maskRoi = self.maskRoi.get()
        if maskRoi is not None:
            fnMaskRoi = maskRoi.getFileName()
            if fnMaskRoi.endswith('.mrc'):
                fnMaskRoi += ':mrc'
            args += ' --mask_roi %s' % fnMaskRoi
        
        if self.nonNegative.get():
            args += ' --nonNegative'
        
        if self.subtract.get():
            args += ' --subtract'
        
        if self.realSpaceProjection.get() == 1:
            args += ' --realSpaceProjection'

        if self.ignoreCTF.get():
            args += ' --ignoreCTF'
        
        self.runJob("xmipp_subtract_projection", args)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.inputParticles.get().getFirstItem()
        vol = self.vol.get()
        mask = self.mask.get()
        if part.getDim()[0] != vol.getDim()[0]:
            errors.append("Input particles and volume should have same X and Y dimensions")
        if round(part.getSamplingRate(), 2) != round(vol.getSamplingRate(), 2):
            errors.append("Input particles and volume should have same sampling rate")
        if mask:
            if round(vol.getSamplingRate(), 2) != round(mask.getSamplingRate(), 2):
                errors.append("Input volume and mask should have same sampling rate")
            if vol.getDim() != mask.getDim():
                errors.append("Input volume and mask should have same dimensions")
        if self.resol.get() == 0:
            errors.append("Resolution (angstroms) should be bigger than 0")
        return errors

    def _warnings(self):
        part = self.inputParticles.get().getFirstItem()
        if part.getDim()[0] > 750 or part.getDim()[1] > 750:
            return ["Particles are quite big, consider to change 'pad=1' (advanced parameter) in order to save RAM "
                    "(even if your RAM is big)."]

    def _summary(self):
        summary = ["Volume: %s\nSet of particles: %s\nMask: %s" %
                   (self.vol.get().getFileName(), self.inputParticles.get(), self.mask.get().getFileName())]
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append("Volume projections subtracted to particles keeping the region in %s"
                           % basename(self.mask.get().getFileName()))
        return methods


class XmippProtBoostParticles(XmippProtSubtractProjectionBase):
    """ This protocol tries to boost the frequencies of the particles to imporve them, based on an adjustment on its correspondent projections from a reference volume. """

    _label = 'boost particles'
    INPUT_PARTICLES = "input_particles.xmd"

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        XmippProtSubtractProjectionBase._defineParams(form)
        form.addParallelSection(threads=0, mpi=4)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertSubSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('boostingStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath(self.INPUT_PARTICLES))

    def boostingStep(self):
        vol = self.vol.get().clone()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        args = '-i %s --ref %s -o %s --sampling %f --max_resolution %f --padding %f --sigma %d ' \
               '--cirmaskrad %d --boost --save %s --save_metadata_stack %s --keep_input_columns '\
               % (self._getExtraPath(self.INPUT_PARTICLES), fnVol, self._getExtraPath(OUTPUT_MRCS),
                  vol.getSamplingRate(), self.resol.get(), self.pad.get(), self.sigma.get(),
                  self.cirmaskrad.get(), self._getExtraPath(), self._getExtraPath(OUTPUT_XMD))

        if self.nonNegative.get():
            args += ' --nonNegative'
            
        if self.ignoreCTF.get():
            args += ' --ignoreCTF'
            
        self.runJob("xmipp_subtract_projection", args)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.inputParticles.get().getFirstItem()
        vol = self.vol.get()
        if part.getDim()[0] != vol.getDim()[0]:
            errors.append("Input particles and volume should have same X and Y dimensions")
        if round(part.getSamplingRate(), 2) != round(vol.getSamplingRate(), 2):
            errors.append("Input particles and volume should have same sampling rate")
        if self.resol.get() == 0:
            errors.append("Resolution (angstroms) should be bigger than 0")
        return errors

    def _warnings(self):
        part = self.inputParticles.get().getFirstItem()
        if part.getDim()[0] > 750 or part.getDim()[1] > 750:
            return ["Particles are quite big, consider to change 'pad=1' (advanced parameter) in order to save RAM "
                    "(even if your RAM is big)."]

    def _summary(self):
        summary = ["Volume: %s\nSet of particles: %s\n" %
                   (self.vol.get().getFileName(), self.inputParticles.get())]
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append("Particles boosted according to their equivalent projections from a reference volume.")
        return methods
