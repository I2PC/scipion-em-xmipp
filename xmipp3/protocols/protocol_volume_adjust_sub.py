# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, FloatParam, EnumParam, StringParam, FileParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils import replaceBaseExt, removeExt, getExt

from pwem.convert import headers, downloadPdb, cifToPdb
from pwem.objects import Volume, Transform
from pwem.protocols import EMProtocol
from pyworkflow import BETA, UPDATED, NEW, PROD


CITE = 'Fernandez-Gimenez2021'


class XmippProtVolAdjBase(EMProtocol):
    """ Helper class that contains some Protocol utilities methods
    used by both  XmippProtVolSubtraction and XmippProtVolAdjust."""

    _possibleOutputs = {"outputVolume": Volume}
    _devStatus = PROD

    # --------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vol1', PointerParam, pointerClass='Volume', label="Volume 1 (reference)", help='Specify a volume.')
        form.addParam('masks', BooleanParam, label='Mask volumes?', default=True,
                      help='The masks are not mandatory but highly recommendable.')
        form.addParam('mask1', PointerParam, pointerClass='VolumeMask', label="Mask for volume 1",
                      condition='masks', help='Specify a mask for volume 1.')
        form.addParam('resol', FloatParam, label="Filter at resolution: ", default=3, allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Resolution (A) at which subtraction will be performed, filtering the input volumes.'
                           'Value 0 implies no filtering.')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3, condition='resol',
                      help='Decay of the filter (sigma parameter) to smooth the mask transition',
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('iter', IntParam, label="Number of iterations: ", default=5, expertLevel=LEVEL_ADVANCED)
        form.addParam('rfactor', FloatParam, label="Relaxation factor (lambda): ", default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Relaxation factor for Fourier amplitude projector (POCS), it should be between 0 and 1, '
                           'being 1 no relaxation and 0 no modification of volume 2 amplitudes')
        form.addParam('radavg', BooleanParam, label="Match the rotationally averaged Fourier amplitudes?", default=True,
                      help='Match the rotationally averaged Fourier amplitudes when adjusting the amplitudes instead of'
                           ' taking them directly from the reference volume. For subtraction and consensus it is '
                           'recommended to set it to True but for sharpening it is recommended to set it to False')
        form.addParam('computeE', BooleanParam, label="Compute energy?", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Compute energy difference between the different adjustment steps and iterations to see if '
                           'the method reaches convergence')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertAdjSteps()
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def createOutputStep(self):
        vol1 = self.vol1.get()
        volume = Volume()
        volume.setSamplingRate(vol1.getSamplingRate())
        if vol1.getFileName().endswith('mrc'):
            origin = Transform()
            ccp4header = headers.Ccp4Header(vol1.getFileName(), readHeader=True)
            shifts = ccp4header.getOrigin()
            origin.setShiftsTuple(shifts)
            volume.setOrigin(origin)
        volume.setFileName(self._getExtraPath("output_volume.mrc"))
        filename = volume.getFileName()
        if filename.endswith('.mrc') or filename.endswith('.map'):
            volume.setFileName(filename + ':mrc')
        self._defineOutputs(outputVolume=volume)


class XmippProtVolSubtraction(XmippProtVolAdjBase):
    """ This protocol scales a volume in order to adjust it to another one. Then, it can calculate the subtraction
    of the two volumes. Second input can be a pdb. The volumes should be aligned previously and they have to
    be equal in size.

    AI Generated

    ## Overview

    The Volumes Subtraction protocol adjusts one volume to another and then
    subtracts it.

    This protocol is useful when two aligned 3D maps represent related structures
    and the user wants to remove the signal of one map from the other. A common
    case is subtracting a known component, domain, ligand, fitted atomic model, or
    reference density from a larger map in order to highlight remaining density.

    The protocol first adjusts the amplitudes of the second input to make it
    comparable with the first volume. It can then subtract the adjusted second
    volume from the first one. This adjustment step is important because direct
    subtraction of two maps with different scales, filtering, or amplitude behavior
    can produce misleading residual density.

    The main output is a new volume containing the subtraction result.

    ## Inputs and General Workflow

    The protocol requires a first volume, called **Volume 1**, which acts as the
    reference volume.

    The second input can be provided in two ways:

    - as a second volume;
    - as an atomic structure, from which the protocol generates a density volume
      and mask.

    The input volumes should already be aligned and should have the same size. The
    protocol is not intended to perform global map alignment before subtraction.

    If masks are provided, they are used to guide the adjustment and subtraction.
    The protocol can also filter the input volumes to a selected resolution before
    performing the operation.

    After running, the protocol creates an output volume named `output_volume.mrc`.

    ## Volume 1

    The **Volume 1 (reference)** parameter defines the map from which the second
    volume will be subtracted.

    This volume provides the reference sampling rate, origin, and output metadata.
    If the input file is an MRC file, the protocol preserves the origin information
    from the MRC header in the output volume.

    Biologically, this should usually be the map containing the density that the
    user wants to analyze after subtracting another component.

    ## Second Input as a Volume

    If **Is the second input a PDB?** is set to **No**, the user provides **Volume
    2**.

    This volume is adjusted to Volume 1 and then subtracted from it.

    The two volumes should already be in the same coordinate frame, have the same
    box size, and represent comparable density. If the volumes are shifted,
    rotated, or sampled inconsistently, the subtraction result will be unreliable.

    ## Second Input as a PDB

    If **Is the second input a PDB?** is set to **Yes**, the protocol converts an
    atomic structure into a density map.

    The PDB can be provided either as a Scipion atomic-structure object or as a
    local file. The protocol generates a volume from the atomic model using the
    sampling rate, size, and origin of Volume 1. It also generates a mask for the
    PDB-derived volume.

    This option can be convenient, but the protocol help explicitly warns that it
    is not the recommended workflow. Automatic PDB-to-map conversion can be
    sensitive to origin mismatches. A safer workflow is to convert the PDB to a map
    in a separate step, inspect the result, and then use the inspected map as
    Volume 2.

    ## Masks

    The **Mask volumes?** option controls whether masks are used.

    Masks are not mandatory, but they are highly recommended. They define the
    regions of each volume that should guide the adjustment and subtraction.

    When masks are enabled, the user provides:

    - **Mask for volume 1**;
    - **Mask for volume 2**, unless the second input is a PDB, in which case the
      protocol generates the second mask automatically.

    Good masks should include the relevant molecular density and avoid excessive
    background. Poor masks can produce incorrect amplitude adjustment or unwanted
    subtraction artifacts.

    ## Filter Resolution

    The **Filter at resolution** parameter defines the resolution, in angstroms,
    at which the input volumes are filtered before subtraction.

    Filtering can make the subtraction more robust by removing high-frequency
    differences that should not drive the adjustment. This is especially useful
    when the two maps have different noise levels or effective resolutions.

    A value of **0** means that no filtering is applied.

    If a positive value is used, the protocol converts the resolution into a
    Fourier cutoff using the sampling rate of Volume 1.

    ## Filter Decay Sigma

    The **Decay of the filter (sigma)** parameter controls the smoothness of the
    filter transition when filtering is applied.

    A smoother transition can reduce ringing and sharp cutoff artifacts.

    This is an advanced parameter. Most users should keep the default unless they
    have a specific reason to tune the filtering behavior.

    ## Number of Iterations

    The **Number of iterations** parameter controls how many iterations are used by
    the adjustment/subtraction algorithm.

    More iterations allow the adjustment process more opportunity to converge, but
    increase computation time.

    The default value is intended as a practical starting point.

    ## Relaxation Factor

    The **Relaxation factor (lambda)** controls the strength of the Fourier
    amplitude adjustment.

    The value must be between 0 and 1.

    A value of 1 means no relaxation in the update. A value closer to 0 makes the
    amplitude modification weaker. The protocol validates that this value remains
    inside the allowed range.

    This is an advanced parameter and should usually be left at its default unless
    the user understands the adjustment behavior.

    ## Rotationally Averaged Fourier Amplitudes

    The **Match the rotationally averaged Fourier amplitudes?** option controls how
    Fourier amplitudes are adjusted.

    When enabled, the protocol matches rotationally averaged Fourier amplitudes
    rather than directly taking amplitudes from the reference volume.

    For subtraction and consensus-like uses, this option is recommended. It helps
    avoid overly local or direction-specific amplitude transfer and makes the
    adjustment more stable.

    For sharpening-like workflows, the form help indicates that this option is
    usually less appropriate.

    ## Compute Energy

    The **Compute energy?** option asks the protocol to compute the energy
    difference between adjustment steps and iterations.

    This is useful for checking whether the method is converging.

    It is an advanced diagnostic option. It does not change the biological meaning
    of the output, but it can help understand whether the adjustment behaved
    stably.

    ## Save Intermediate Files

    The **Save intermediate files?** option stores additional files:

    - filtered Volume 1;
    - adjusted Volume 2.

    These are the volumes that are actually used in the subtraction.

    Saving them is useful for debugging and interpretation, because it allows the
    user to inspect what was subtracted after filtering and adjustment. The option
    is disabled by default to avoid unnecessary files.

    ## Output Volume

    The main output is **outputVolume**.

    This volume contains the subtraction result and is written as
    `output_volume.mrc`.

    The output volume uses the sampling rate of Volume 1. If Volume 1 is an MRC
    file, the origin from its header is copied to the output.

    The output should be interpreted as Volume 1 after subtracting the adjusted
    second input.

    ## Interpreting the Result

    The subtraction result depends strongly on alignment, scaling, masking, and
    filtering.

    A good result can reveal residual density, flexible regions, missing
    components, ligands, or conformational differences. A poor result may contain
    positive or negative artifacts caused by misalignment, incorrect masks,
    different resolutions, or unsuitable amplitude adjustment.

    The output should therefore always be inspected together with the original
    volumes, masks, and, when saved, the adjusted intermediate maps.

    ## Practical Recommendations

    Use this protocol only when the two inputs are already aligned and have the
    same box size.

    Prefer providing a precomputed and inspected map as Volume 2 rather than
    letting the protocol convert a PDB automatically, unless the coordinate origin
    is very well controlled.

    Use masks whenever possible.

    Use resolution filtering when the two maps have different effective resolution
    or noise behavior.

    Keep rotationally averaged amplitude matching enabled for most subtraction
    workflows.

    Save intermediate files when testing parameters or when the subtraction result
    is difficult to interpret.

    Inspect the output carefully before using it for biological conclusions.

    ## Final Perspective

    Volumes Subtraction is a map-processing protocol for removing one adjusted
    density from another.

    For biological users, its value is that it can isolate residual density or
    focus attention on structural differences between two related maps. The
    protocol is most reliable when the input maps are well aligned, consistently
    sampled, appropriately masked, and filtered to a resolution suitable for the
    comparison.
    """

    _label = 'volumes subtraction'
    IMPORT_OBJ = 0
    IMPORT_FROM_FILES = 1

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        XmippProtVolAdjBase._defineParams(form)
        form.addParam('pdb', BooleanParam, label='Is the second input a PDB?', default=False,
                      help='If yes, the protocol will generate and store in folder "extra" of this protocol '
                           'a volume and a mask from the pdb. This is not the recommended option, as the automatic '
                           'conversion of the PDB into a density map may not be successful due to origin mismatches. '
                           'We recommend to convert previously the PDB, inspect the converted map and use the map as '
                           'input. If not, a second volume has to be input and optionally (but highly recommendable), '
                           'a mask for it.')
        form.addParam('inputPdbData', EnumParam, choices=['object', 'file'], condition='pdb',
                      label="Retrieve PDB from", default=self.IMPORT_OBJ,
                      display=EnumParam.DISPLAY_HLIST,
                      help='Retrieve PDB data from server, use a pdb Object, or a local file')
        form.addParam('pdbObj', PointerParam, pointerClass='AtomStruct',
                      label="Input pdb ", condition='inputPdbData == IMPORT_OBJ and pdb', allowsNull=True,
                      help='Specify a pdb object. This is not the recommended option, as the automatic conversion of '
                           'the PDB into a density map may not be successful due to origin mismatches. We recommend'
                           'to convert previously the PDB, inspect the converted map and use the map as input.')
        form.addParam('pdbFile', FileParam,
                      label="File path", condition='inputPdbData == IMPORT_FROM_FILES and pdb', allowsNull=True,
                      help='Specify a path to desired PDB structure.')
        form.addParam('vol2', PointerParam, pointerClass='Volume', label="Volume 2", condition='pdb == False',
                      help='Specify a volume.')
        form.addParam('mask2', PointerParam, pointerClass='VolumeMask', label="Mask for volume 2",
                      condition='masks and pdb==False', help='Specify a mask for volume 1.')
        form.addParam('saveFiles', BooleanParam, label='Save intermediate files?', default=False,
                      expertLevel=LEVEL_ADVANCED, help='Save input volume 1 filtered and input volume 2 adjusted, which'
                                                       'are the volumes that are really subtracted.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAdjSteps(self):
        if self.pdb:
            self._insertFunctionStep('convertPdbStep')
            self._insertFunctionStep('generateMask2Step')
        self._insertFunctionStep('subtractionStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertPdbStep(self):
        vol1 = self.vol1.get()
        pdbFn = self._getPdbFileName()
        self.outFile = self._getVolName()
        samplingR = vol1.getSamplingRate()
        size = vol1.getDim()
        ccp4header = headers.Ccp4Header(vol1.getFileName(), readHeader=True)
        self.shifts = ccp4header.getOrigin()
        # convert pdb to volume with the size and origin of the input volume
        args = ' -i %s --sampling %f -o %s --size %d %d %d --orig %d %d %d' % \
               (pdbFn, samplingR, removeExt(self.outFile), size[2], size[1], size[0], self.shifts[0]/samplingR,
                self.shifts[1]/samplingR, self.shifts[2]/samplingR)
        program = "xmipp_volume_from_pdb"
        self.runJob(program, args)
        volpdbmrc = "%s.mrc" % removeExt(self.outFile)
        # convert volume from pdb to .mrc in order to store the origin in the mrc header
        args2 = ' -i %s -o %s -t vol' % (self.outFile, volpdbmrc)
        program2 = "xmipp_image_convert"
        self.runJob(program2, args2)
        # write origin in mrc header of volume from pdb
        ccp4headerOut = headers.Ccp4Header(volpdbmrc, readHeader=True)
        ccp4headerOut.setOrigin(self.shifts)

    def generateMask2Step(self):
        args = ' -i %s -o %s --select below 0.010000 --substitute binarize' % (self.outFile,
                                                                               self._getExtraPath("mask2.mrc"))
        program = "xmipp_transform_threshold"
        self.runJob(program, args)
        args2 = ' -i %s --binaryOperation dilation --size 1' % (self._getExtraPath("mask2.mrc"))
        program2 = "xmipp_transform_morphology"
        self.runJob(program2, args2)

    def subtractionStep(self):
        vol1 = self.vol1.get().clone()
        fileName1 = vol1.getFileName()
        if fileName1.endswith('.mrc'):
            fileName1 += ':mrc'
        if self.pdb:
            vol2 = self.outFile
            mask2 = self._getExtraPath("mask2.mrc")
        else:
            vol2 = self.vol2.get().getFileName()
            if vol2.endswith('.mrc'):
                vol2 += ':mrc'
            if self.masks:
                mask2 = self.mask2.get().getFileName()
        resol = self.resol.get()
        iter = self.iter.get()
        program = "xmipp_volume_subtraction"
        args = '--i1 %s --i2 %s -o %s --iter %s --lambda %s --sub' % \
               (fileName1, vol2, self._getExtraPath("output_volume.mrc"), iter, self.rfactor.get())
        if resol:
            fc = vol1.getSamplingRate()/resol
            args += ' --cutFreq %f --sigma %d' % (fc, self.sigma.get())

        if self.masks:
            args += ' --mask1 %s --mask2 %s' % (self.mask1.get().getFileName(), mask2)
        if self.saveFiles:
            args += ' --saveV1 %s --saveV2 %s' % (self._getExtraPath('vol1_filtered.mrc'),
                                                  self._getExtraPath('vol2_adjusted.mrc'))
        if self.radavg:
            args += ' --radavg'
        if self.computeE:
            args += ' --computeEnergy'
        self.runJob(program, args)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = ["Volume 1: %s" % self.vol1.get().getFileName()]
        if self.pdb:
            if self.inputPdbData == self.IMPORT_OBJ:
                summary.append("Input PDB File: %s" % self.pdbObj.get().getFileName())
            else:
                summary.append("Input PDB File: %s" % self.pdbFile.get())
            summary.append("Mask 2 generated")
        else:
            summary.append("Volume 2: %s" % self.vol2.get().getFileName())
        if self.masks:
            summary.append("Input mask 1: %s" % self.mask1.get().getFileName())
            summary.append("Input mask 2: %s" % self.mask2.get().getFileName())
        if self.resol.get() != 0:
            summary.append("Subtraction at resolution %f A" % self.resol.get())
        if self.radavg:
            summary.append("Matching the rotational averaged Fourier amplitudes")
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            methods.append("Volume %s subtracted from volume %s" % (self.vol2.get().getFileName(),
                                                                    self.vol1.get().getFileName()))
            if self.resol.get() != 0:
                methods.append(" at resolution %f A" % self.resol.get())
        return methods

    def _validate(self):
        errors = []
        rfactor = self.rfactor.get()
        if rfactor < 0 or rfactor > 1:
            errors.append('Relaxation factor (lambda) must be between 0 and 1')
        return errors

    def _citations(self):
        return [CITE]

    # --------------------------- UTLIS functions --------------------------------------------
    def _getPdbFileName(self):
        if self.inputPdbData == self.IMPORT_OBJ:
            return self.pdbObj.get().getFileName()
        else:
            return self.pdbFile.get()

    def _getVolName(self):
        return self._getExtraPath(replaceBaseExt(self._getPdbFileName(), "vol"))


class XmippProtVolAdjust(XmippProtVolAdjBase):
    """ This protocol scales a volume in order to assimilate it to another one.
    The volume with the best resolution should be the first one.
    The volumes should be aligned previously and they have to be equal in
    size.

    AI Generated

    ## Overview

    The Volume Adjust protocol modifies one volume so that its amplitudes become
    more comparable to a reference volume.

    Unlike the Volumes Subtraction protocol, this protocol does not subtract the
    second volume from the first one. Instead, it adjusts **Volume 2** to resemble
    **Volume 1** in terms of Fourier amplitude behavior and scaling. The result is
    an adjusted version of Volume 2.

    This is useful when two aligned maps need to be brought to a comparable scale
    before visualization, comparison, consensus analysis, subtraction in a later
    step, or other processing.

    The volume with the best resolution should normally be provided as Volume 1,
    the reference. Volume 2 is the map that will be modified.

    ## Inputs and General Workflow

    The protocol requires two input volumes:

    - **Volume 1**, the reference volume;
    - **Volume 2**, the volume to be modified.

    The volumes should already be aligned and should have the same size. The
    protocol does not perform global alignment.

    If masks are enabled, one mask is provided for each volume. The protocol then
    runs the Xmipp volume-adjustment procedure, optionally filtering the volumes to
    a selected resolution and iterating the adjustment.

    The output is a new adjusted volume written as `output_volume.mrc`.

    ## Volume 1

    The **Volume 1 (reference)** parameter defines the reference map.

    This volume provides the target amplitude behavior for the adjustment. The
    protocol also uses its sampling rate for the output volume. If Volume 1 is an
    MRC file, its origin information is copied to the output.

    The help text indicates that the volume with the best resolution should be used
    as Volume 1.

    ## Volume 2

    The **Volume 2 (to modify)** parameter defines the map that will be adjusted.

    The output volume is an adjusted version of this second map. It is modified so
    that it becomes more comparable to Volume 1 according to the selected
    amplitude-adjustment settings.

    Volume 2 should already be aligned with Volume 1 and should have the same box
    size. If the volumes are not aligned, the adjustment may compensate
    incorrectly and the output may be misleading.

    ## Masks

    The **Mask volumes?** option controls whether masks are used.

    Masks are highly recommended because they restrict the adjustment to the
    relevant molecular regions and reduce the influence of solvent or background.

    When masks are enabled, the user provides:

    - **Mask for volume 1**;
    - **Mask for volume 2**.

    The masks should correspond to the relevant density in each volume. They should
    avoid including large background regions unless those regions are intended to
    contribute to the adjustment.

    ## Filter Resolution

    The **Filter at resolution** parameter defines the resolution, in angstroms,
    used to filter the input volumes during adjustment.

    A value of **0** means no filtering.

    Filtering can make the adjustment more stable by focusing it on a controlled
    frequency range. This is useful when the two maps have different noise levels,
    different effective resolutions, or high-frequency features that should not
    drive the amplitude adjustment.

    ## Filter Decay Sigma

    The **Decay of the filter (sigma)** parameter controls the smoothness of the
    filter transition when filtering is applied.

    A smoother transition can reduce artifacts associated with abrupt Fourier
    cutoffs.

    This is an advanced setting and is usually left at its default value.

    ## Number of Iterations

    The **Number of iterations** parameter controls how many adjustment iterations
    are performed.

    More iterations may allow the procedure to converge more fully, but increase
    runtime. Too many iterations are not necessarily better if the maps are noisy,
    poorly aligned, or poorly masked.

    The default value is a reasonable starting point.

    ## Relaxation Factor

    The **Relaxation factor (lambda)** controls the strength of the Fourier
    amplitude projector used during adjustment.

    The value must be between 0 and 1.

    A value of 1 corresponds to no relaxation, while a value closer to 0 reduces
    the modification applied to the amplitudes. The protocol validates this range
    before running.

    This is an advanced parameter.

    ## Rotationally Averaged Fourier Amplitudes

    The **Match the rotationally averaged Fourier amplitudes?** option controls
    whether the adjustment matches rotationally averaged Fourier amplitudes.

    When enabled, the method adjusts Volume 2 using radial amplitude information
    rather than directly copying directional amplitude information from Volume 1.

    This option is recommended for subtraction and consensus-like workflows. The
    form help notes that for sharpening workflows it is recommended to set this
    option to **False**.

    ## Compute Energy

    The **Compute energy?** option asks the program to compute the energy
    difference between adjustment steps and iterations.

    This can help assess whether the adjustment is converging.

    It is mainly a diagnostic option for advanced users.

    ## Output Volume

    The main output is **outputVolume**.

    This output is the adjusted version of Volume 2, written as
    `output_volume.mrc`.

    The output volume uses the sampling rate of Volume 1. If Volume 1 is an MRC
    file, the origin information from its header is copied to the output.

    The output can be used for comparison, visualization, later subtraction, or
    other downstream protocols requiring the second map to be adjusted to the
    reference map.

    ## Interpreting the Result

    The adjusted volume should be interpreted as Volume 2 after amplitude and scale
    adjustment with respect to Volume 1.

    No biological density is created by the protocol. The output is a transformed
    version of the second map designed to make it more comparable to the reference.

    If the input volumes are poorly aligned, have different masks, contain
    different structures, or differ strongly in resolution, the adjustment may
    produce artifacts or misleading amplitudes.

    ## Practical Recommendations

    Use Volume 1 as the better resolved or more reliable reference map.

    Use Volume 2 as the map that should be modified.

    Make sure both volumes are already aligned and have the same size before
    running the protocol.

    Use masks whenever possible.

    Use filtering when comparing maps with different noise or resolution behavior.

    Keep rotationally averaged amplitude matching enabled for consensus or later
    subtraction workflows. Consider disabling it only for sharpening-oriented use
    cases.

    Inspect the adjusted output together with both input maps and masks.

    ## Final Perspective

    Volume Adjust is a map-normalization and amplitude-adjustment protocol.

    For biological users, its main value is that it prepares one map to be more
    directly comparable with another. This can be useful before subtraction,
    consensus analysis, visualization, or other workflows where differences in map
    scale and amplitude behavior would otherwise complicate interpretation.

    The protocol should be used on already aligned volumes and interpreted as a
    preparatory adjustment step rather than as an independent validation or
    refinement procedure.
    """

    _label = 'volume adjust'
    IMPORT_OBJ = 0
    IMPORT_FROM_FILES = 1

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        XmippProtVolAdjBase._defineParams(form)
        form.addParam('vol2', PointerParam, pointerClass='Volume', label="Volume 2 (to modify)", help='Specify a volume.')
        form.addParam('mask2', PointerParam, pointerClass='VolumeMask', label="Mask for volume 2",
                      condition='masks', help='Specify a mask for volume 1.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAdjSteps(self):
        self._insertFunctionStep('adjustStep')

    # --------------------------- STEPS functions --------------------------------------------
    def adjustStep(self):
        vol1 = self.vol1.get().clone()
        fnVol1 = vol1.getFileName()
        vol2 = self.vol2.get().getFileName()
        if fnVol1.endswith('.mrc'):
            fnVol1 += ':mrc'
        if vol2.endswith('.mrc'):
            vol2 += ':mrc'
        resol = self.resol.get()
        iter = self.iter.get()
        program = "xmipp_volume_subtraction"
        args = '--i1 %s --i2 %s -o %s --iter %s --lambda %s' % \
               (fnVol1, vol2, self._getExtraPath("output_volume.mrc"), iter, self.rfactor.get())
        if resol:
            fc = vol1.getSamplingRate()/resol
            args += ' --cutFreq %f --sigma %d' % (fc, self.sigma.get())
        if self.masks:
            args += ' --mask1 %s --mask2 %s' % (self.mask1.get().getFileName(), self.mask2.get().getFileName())
        if self.radavg:
            args += ' --radavg'
        if self.computeE:
            args += ' --computeEnergy'
        self.runJob(program, args)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = ["Volume 1: %s\nVolume 2: %s" % (self.vol1.get().getFileName(), self.vol2.get().getFileName())]
        if self.masks:
            summary.append("Input mask 1: %s" % self.mask1.get().getFileName())
            summary.append("Input mask 2: %s" % self.mask2.get().getFileName())
        if self.resol.get() != 0:
            summary.append("Filter at resolution %f A" % self.resol.get())
        if self.radavg:
            summary.append("Matching the rotational averaged Fourier amplitudes")
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            methods.append("Volume %s adjusted to volume %s" % (self.vol2.get().getFileName(),
                                                                self.vol1.get().getFileName()))
            if self.resol.get() != 0:
                methods.append(" at resolution %f A" % self.resol.get())
        return methods

    def _validate(self):
        errors = []
        rfactor = self.rfactor.get()
        if rfactor < 0 or rfactor > 1:
            errors.append('Relaxation factor (lambda) must be between 0 and 1')
        return errors

    def _citations(self):
        return [CITE]
