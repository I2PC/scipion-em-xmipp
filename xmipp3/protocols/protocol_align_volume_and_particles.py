# **************************************************************************
# *
# * Authors:     ajimenez@cnb.csic.es
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
import enum

import numpy as np
import pyworkflow.protocol.params as params
from pwem.convert.headers import setMRCSamplingRate
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem.protocols import ProtAlignVolume
from pwem.emlib.image import ImageHandler
from pwem.objects import Transform, Volume, SetOfParticles
from pyworkflow.utils import weakImport

from pyworkflow.utils.path import cleanPath

from xmipp3.constants import SYM_URL
from pyworkflow import BETA, UPDATED, NEW, PROD

ALIGN_MASK_CIRCULAR = 0
ALIGN_MASK_BINARY_FILE = 1

ALIGN_GLOBAL = 0
ALIGN_LOCAL = 1

pointerClasses = [SetOfParticles]
with weakImport("tomo"):
    from tomo.objects import SetOfSubTomograms
    pointerClasses.append(SetOfSubTomograms)

class AlignVolPartOutputs(enum.Enum):
    Volume = Volume
    Particles = SetOfParticles

class XmippProtAlignVolumeParticles(ProtAlignVolume):
    """ 
     Aligns a volume (inputVolume) using a Fast Fourier method
     with respect to a reference one (inputReference).
     The obtained alignment parameters are used to align the set of particles or subtomograms
     (inputParticles) that generated the input volume.

     AI Generated:

        What this protocol does and when to use it

        Align Volume and Particles is designed for a very practical situation
        in cryo-EM (and also subtomogram averaging): you already have an input
        volume reconstructed from a set of particles (or subtomograms), and you
        want that volume to match a reference volume in orientation and
        position. Once the protocol finds the rigid-body transformation that
        aligns the input volume to the reference, it propagates exactly the
        same transformation to the underlying particle orientations/transforms.
        In other words, it is a “volume-level registration” step that
        automatically updates the corresponding particle set so everything
        stays consistent.

        Biologically, this is useful whenever you want to compare, merge, or
        re-express particle metadata in the coordinate system of a reference
        map. Typical examples include bringing results from different
        refinements into the same frame, standardizing orientations before
        downstream classification/analysis, or aligning two reconstructions
        from related conditions so that particle poses become directly
        comparable.

        A key requirement is that the input particles/subtomograms must
        already have alignment information (orientation/shift matrices).
        If the particle set is not aligned yet, the protocol will not run.

        Inputs: what you need to provide

        You will provide three inputs: a reference volume, an input volume,
        and the input particles (or subtomograms) that generated that input
        volume. The alignment is computed only between the two volumes, but
        the output includes both an aligned version of the input volume and a
        transformed particle set whose poses have been updated.

        From a user perspective, the most important conceptual point is
        consistency: the input particle set should correspond to the input
        volume you are aligning. If you accidentally provide a particle set
        that does not match the input volume (for example, from a different
        refinement or different half-set), the protocol may still run but the
        resulting particle transforms will not be meaningful.

        Alignment mode: Global vs Local

        The protocol offers two alignment modes, intended to balance
        robustness and speed.

        If you choose Global alignment, the protocol performs a broad,
        “global” search using the Fast Fourier approach. This is typically
        the right choice when you do not trust the initial relative orientation
        between the two volumes, for example when the input volume comes from
        a different pipeline, a different processing branch, or when you
        suspect the map might be rotated or shifted with respect to the
        reference.

        If you choose Local alignment, the protocol performs a local refinement
        around the current pose. This is faster and is best used when the input
        volume is already close to the reference (for example, when you are
        just making small adjustments, or when both maps come from similar
        processing with only mild differences).

        In practice, many biological users start with Global for safety when
        integrating results across workflows, and switch to Local when
        refining a known alignment.

        Consider mirrors: handling handedness and mirror solutions

        The Consider mirrors option enables the alignment procedure to
        evaluate mirrored solutions. This can matter when there is a
        possibility of a mirror ambiguity between maps, or when you suspect a
        handedness-related discrepancy. In most routine workflows where all
        processing has been consistent, you will keep this disabled. However,
        it becomes relevant when comparing volumes coming from different
        sources or historical datasets, or when you have reasons to suspect
        that the map might be mirrored with respect to the reference.

        Because mirror solutions can sometimes produce deceptively good
        correlation scores, it is good practice to enable this option only when
        you have a concrete motivation and to validate the result visually and
        biologically (for example by checking known chiral features, helix
        handedness, or consistency with atomic models).

        Symmetry group: what it means here

        You can specify a symmetry group (e.g., c1, c2, d7, i1, etc.) using
        Xmipp’s symmetry conventions. Biologically, symmetry can help alignment
        when the object is symmetric, but it also introduces the usual
        interpretation caveat: a symmetric object may have multiple equivalent
        orientations.

        In many workflows you will leave this as c1 unless you are working with
        a truly symmetric assembly and you want the alignment to treat
        symmetry-equivalent poses appropriately. If your goal is to make
        particle poses comparable between two analyses of the same symmetric
        complex, setting the correct symmetry can prevent avoidable mismatches.
        On the other hand, if you care about a specific asymmetric feature
        (bound ligand, partial occupancy, asymmetric subunit), forcing symmetry
        may not be desirable.

        Wrap option: advanced edge behavior during alignment

        The Wrap parameter controls whether the input volume is treated as
        wrapping around at the borders during alignment. Most biological users
        will not need to change this. If your map is well boxed and the object
        is not interacting with the borders, leaving wrap disabled is typically
        safe and avoids artifacts that can appear when densities are allowed
        to “wrap” across edges.

        Masking: focusing the alignment on the biologically relevant region

        As with most volume alignment problems, masking can be decisive.
        This protocol allows you to apply a mask to the volumes during
        alignment, so that the transformation is driven primarily by the
        region you care about.

        If you enable masking, you can choose either a circular (spherical)
        mask or a binary mask file. The circular mask is quick and can work
        well for compact particles. The binary mask is more flexible and is
        usually preferred for real biological cases where only part of the
        structure is stable or informative, such as multi-domain proteins,
        flexible complexes, membrane proteins with large solvent regions, or
        when you want to align using a conserved core while ignoring a
        variable periphery.

        A biologically sensible mask typically includes the rigid, conserved
        domain(s) and excludes highly flexible regions and noise-dominated
        solvent. If the alignment is unstable or “snaps” to the wrong solution,
        applying a focused mask is often the first and most effective remedy.

        What happens internally from the user point of view: resizing and
        consistency

        The protocol ensures that the reference and input volume are
        compatible in size before alignment. If their box dimensions
        differ, it will adjust the reference volume to match the input
        volume dimension for the purpose of the alignment. For a biological
        user, the takeaway is that you should still aim to provide volumes
        with consistent sampling and comparable boxing, because extreme
        differences in box definition, filtering, or masking style can still
        complicate interpretation even if the dimensions are made compatible.

        Outputs: what you get and how to interpret it

        After running, the protocol produces two outputs.

        First, it generates an aligned version of the input volume,
        transformed into the coordinate system of the reference. This output
        is the most direct way to visually check whether the alignment makes
        biological sense: you can overlay it with the reference, compare
        sections, or inspect key landmarks.

        Second, it produces a new aligned particle (or subtomogram) set. In
        \this output set, each particle’s existing transformation matrix is
        updated by composing it with the volume-to-volume alignment
        transformation. Conceptually, you can think of it as re-expressing
        all particle poses in the coordinate frame of the reference map,
        while preserving each particle’s relative orientation/shift within
        the reconstruction.

        This particle output is the main reason the protocol exists: it lets
        you continue downstream processing—classification, refinement,
        comparison, export—without having to manually reframe orientations.

        Practical guidance: common processing scenarios

        If your goal is simply to bring an existing reconstruction into the
        frame of a known reference so that you can compare them or create
        consistent figures, the typical workflow is to run the protocol in
        Global mode, optionally with a mask around the stable core, and then
        verify by overlay in a viewer.

        If your goal is to merge or compare particle metadata across pipelines,
        the particle output becomes the key product. In that case, it is worth
        being conservative: use a mask that focuses on the most reliable region
        and validate the aligned volume carefully, because any mistake at the
        volume alignment step will be propagated to every particle.

        If your volumes are already nearly aligned (for example, coming from
        two very similar refinements or from consecutive steps of the same
        workflow), then Local mode is usually sufficient and faster. This is a
        common choice when you want a gentle correction rather than a full
        re-registration.

        If you suspect a mirror discrepancy or handedness issue between
        datasets, enabling Consider mirrors can help detect it, but it should
        be treated as a hypothesis-testing tool: always validate the chosen
        solution with biological constraints.

        A final note on requirements and validation

        This protocol assumes that the input particles/subtomograms already
        contain alignment transforms. It does not perform particle alignment
        itself; instead, it transfers a volume-derived rigid transformation
        onto existing particle poses. Because of that, a good validation habit
        is to (1) visually check the aligned input volume against the
        reference, and (2) if you intend to use the aligned particles for
        further refinement, run a quick downstream sanity check (for example,
        a short refinement or a quick comparison of angular distributions)
        to confirm that the new frame is consistent.

        If you want, I can also write a short “recommended settings by
        scenario” paragraph specifically for SPA vs subtomogram averaging
        (when subtomograms are present), still in the same textual style.
     """
    _label = 'align volume and particles'
    _possibleOutputs = AlignVolPartOutputs
    _devStatus = UPDATED
    nVols = 0

    
    def __init__(self, **args):
        ProtAlignVolume.__init__(self, **args)

        # These 2 must match the output enum above.
        self.Volume = None
        self.Particles = None

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Volume parameters')
        form.addParam('inputReference', params.PointerParam, pointerClass='Volume', 
                      label="Reference volume", important=True, 
                      help='Reference volume to be used for the alignment.')    
        form.addParam('inputVolume', params.PointerParam, pointerClass='Volume',
                      label="Input volume", important=True, 
                      help='Select one volume to be aligned against the reference volume.')
        form.addParam('inputParticles', params.PointerParam, pointerClass=pointerClasses,
                      label="Input particles", important=True, 
                      help='Select one set of particles to be aligned against '
                           'the reference set of particles using the transformation '
                           'calculated with the reference and input volumes.')
        form.addParam('alignmentMode', params.EnumParam, default=ALIGN_GLOBAL, choices=["Global","Local"],
                      label="Alignment mode")
        form.addParam('considerMirrors', params.BooleanParam, default=False,
                      label='Consider mirrors')
        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry group",
                      help='See %s page for a description of the symmetries '
                           'accepted by Xmipp' % SYM_URL)
        form.addParam('wrap', params.BooleanParam, default=False,
                      label='Wrap', expertLevel=LEVEL_ADVANCED,
                      help='Wrap the input volume when aligning to the reference')
        
        group1 = form.addGroup('Mask')
        group1.addParam('applyMask', params.BooleanParam, default=False, 
                      label='Apply mask?',
                      help='Apply a 3D Binary mask to the volumes')
        group1.addParam('maskType', params.EnumParam,
                        choices=['circular','binary file'],
                        default=ALIGN_MASK_CIRCULAR,
                        label='Mask type', display=params.EnumParam.DISPLAY_COMBO,
                        condition='applyMask',
                      help='Select the type of mask you want to apply')
        group1.addParam('maskRadius', params.IntParam, default=-1,
                        condition='applyMask and maskType==%d' % ALIGN_MASK_CIRCULAR,
                        label='Mask radius',
                        help='Insert the radius for the mask')
        group1.addParam('maskFile', params.PointerParam,
                        condition='applyMask and maskType==%d' % ALIGN_MASK_BINARY_FILE,
                        pointerClass='VolumeMask', label='Mask file',
                        help='Select the volume mask object')

        form.addParallelSection(threads=8, mpi=1)
        
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):

        #Some definitions of filenames
        self.fnRefVol = self._getExtraPath("refVolume.vol")
        self.fnInputVol = self._getExtraPath("inputVolume.vol")

        maskArgs = self._getMaskArgs()

        self._insertFunctionStep(self.convertStep)
        self._insertFunctionStep(self.alignVolumeStep, maskArgs)
        self._insertFunctionStep(self.createOutputStep)
        
    #--------------------------- STEPS functions --------------------------------------------


    def convertStep(self):

        # Resizing inputs
        ih = ImageHandler()
        ih.convert(self.inputReference.get(), self.fnRefVol)
        XdimRef = self.inputReference.get().getDim()[0]
        ih.convert(self.inputVolume.get(), self.fnInputVol)
        XdimInput = self.inputVolume.get().getDim()[0]

        if XdimRef!=XdimInput:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" %
                        (self.fnRefVol, XdimInput), numberOfMpi=1)


    def alignVolumeStep(self, maskArgs):

        fhInputTranMat = self.getTransformationFile()
        outVolFn = self.getOutputAlignedVolumePath()
      
        args = "--i1 %s --i2 %s --apply %s" % \
               (self.fnRefVol, self.fnInputVol, outVolFn)
        args += maskArgs
        if self.alignmentMode.get()==ALIGN_GLOBAL:
            args += " --frm"
        else:
            args += " --local"
        if self.considerMirrors:
            args += " --consider_mirror"
        args += " --copyGeo %s" % fhInputTranMat
        if not self.wrap:
            args += ' --dontWrap'
        self.runJob("xmipp_volume_align", args)
        cleanPath(self.fnRefVol)
        cleanPath(self.fnInputVol)

    def getAlignmentMatrix(self):
        if not (hasattr(self, '_lhsAlignmentMatrix') and hasattr(self, '_rhsAlignmentMatrix')):
            fhInputTranMat = self.getTransformationFile()
            transMatFromFile = np.loadtxt(fhInputTranMat)
            self._lhsAlignmentMatrix = np.reshape(transMatFromFile, (4, 4))
            self._rhsAlignmentMatrix = np.eye(4)
            
            if np.linalg.det(self._lhsAlignmentMatrix[:3,:3]) < 0:
                self._rhsAlignmentMatrix[2,2] = -1

        return self._lhsAlignmentMatrix, self._rhsAlignmentMatrix

    def getTransformationFile(self):
        return self._getExtraPath('transformation-matrix.txt')


    def createOutputStep(self):   

        # VOLUME aligned to the reference
        outVolFn = self.getOutputAlignedVolumePath()
        Ts = self.inputVolume.get().getSamplingRate()
        outVol = Volume()
        outVol.setLocation(outVolFn)
        # Set the mrc header for sampling rate.
        setMRCSamplingRate(outVolFn, Ts)

        # Set transformation matrix
        fhInputTranMat = self.getTransformationFile()
        transMatFromFile = np.loadtxt(fhInputTranMat)
        transformationMat = np.reshape(transMatFromFile,(4,4))
        transform = Transform()
        transform.setMatrix(transformationMat)
        outVol.setTransform(transform)
        outVol.setSamplingRate(Ts)

        outputArgs = {AlignVolPartOutputs.Volume.name: outVol}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputVolume, outVol)

        # PARTICLES ....
        inputParts = self.inputParticles.get()
        outputParticles = inputParts.create(self._getExtraPath())
        outputParticles.copyInfo(self.inputParticles.get())
        outputParticles.setAlignmentProj()

        # Clone set
        #readSetOfParticles(outParticlesFn, outputParticles)
        outputParticles.copyItems(inputParts,updateItemCallback=self._updateParticleTransform)
        outputArgs = {AlignVolPartOutputs.Particles.name: outputParticles}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputParticles, outputParticles)

    def _updateParticleTransform(self, particle, row):
        lhs, rhs = self.getAlignmentMatrix()
        alignment = np.array(particle.getTransform().getMatrix())
        alignment2 = lhs @ alignment @ rhs

        particle.getTransform().setMatrix(alignment2)

    def getOutputAlignedVolumePath(self):
        outVolFn = self._getExtraPath("inputVolumeAligned.mrc")
        return outVolFn

    #--------------------------- INFO functions --------------------------------------------
    
    def _validate(self):
        errors = []
        if self.inputParticles.get().hasAlignment() is False:
            errors.append("Input particles need to be aligned (they should have transformation matrix)")
        return errors
    
    def _summary(self):
        summary = []
        summary.append("Alignment method: %s" % self.getEnumText('alignmentMode'))
        return summary
    
    def _methods(self):
        methods = 'We aligned a volume against a reference volume using '
        methods += ' the Fast Fourier alignment described in [Chen2013].'
        return [methods]
        
    def _citations(self):
        return ['Chen2013']
        
    #--------------------------- UTILS functions -------------------------------
    def _getMaskArgs(self):
        maskArgs = ''
        if self.applyMask:
            if self.maskType == ALIGN_MASK_CIRCULAR:
                maskArgs+=" --mask circular -%d" % self.maskRadius
            else:
                maskArgs+=" --mask binary_file %s" % self.maskFile.get().getFileName()
        return maskArgs

          
