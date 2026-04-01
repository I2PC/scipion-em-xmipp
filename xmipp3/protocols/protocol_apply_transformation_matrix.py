# **************************************************************************
# *
# * Authors:     Mohsen Kazemi (mkazemi@cnb.csic.es)
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

import numpy as np

from pyworkflow import VERSION_1_1
import pyworkflow.protocol.params as params

from pwem.protocols import ProtProcessParticles


from xmipp3.convert import writeSetOfParticles

        
class XmippProtApplyTransformationMatrix(ProtProcessParticles):
    """ 
    Apply transformation matrix  of an aligned volume on 
    a set of particles to modify their angular assignment.
    Note:
    These particles are practically related to the 
    aligned volume (but before alignment).

    AI Generated:

    What this protocol is for

    Apply transformation matrix is a metadata-propagation tool that updates the
    angular assignment (projection alignment transforms) of a particle set
    using the rigid transformation stored in a volume. It is meant for
    situations where you have aligned a volume to another reference volume
    (for example with an “align volume” protocol), and you now want the
    particles that generated the original (pre-alignment) volume to be
    expressed in the new reference frame.

    Biologically, this comes up all the time when comparing reconstructions or
    consolidating multiple processing branches. If you align one map to another,
    the particle orientations associated with the first map become “out of
    frame” relative to the aligned map unless you apply the same transformation
    to the particle metadata. This protocol does exactly that: it takes the
    transformation matrix stored in the aligned volume and composes it onto
    every particle’s projection transform so that the particle angles/shifts
    are updated consistently.

    A useful way to think about it is: “I have particles aligned to a volume;
    I moved/rotated that volume to match another reference; now I need to
    move/rotate the particle orientations in the same way.”

    Inputs and what they must contain

    You provide two inputs.

    The first is an input particle set that must already have projection
    alignment (i.e., assigned angles/transform matrices consistent with a 3D
    reference). In Scipion terms, these are particles with 3D/projection
    alignment metadata, not merely 2D alignment.

    The second is an input volume that must contain a stored transform. In
    practice, this is typically the output aligned volume produced by a
    volume-to-volume alignment protocol (the common example is the output of
    an “align volume” step). The crucial requirement is that the volume’s
    transform represents the mapping you want to apply to the particle poses.

    This protocol assumes that the input particles are indeed the ones
    associated with the pre-aligned version of that volume. If the particle set
    is unrelated (different refinement, different reference, different
    convention), applying the matrix will produce particle orientations that
    are mathematically consistent but biologically meaningless.

    What the protocol does (conceptually)

    For each particle, the protocol takes the particle’s current transform
    matrix (its projection orientation and shift) and left-multiplies it by the
    volume’s transform matrix. The effect is that every particle’s pose is
    rotated/translated according to the same rigid-body transform that was
    applied to the volume. No image pixels are changed; only the particle
    metadata transforms are updated.

    This means the protocol is fast and deterministic: it is not a refinement
    and it does not depend on correlation or optimization. It simply
    re-expresses particle orientations in a new coordinate frame.

    Outputs and how to use them

    The output is a new SetOfParticles (outputParticles) that is identical to
    the input particle set in terms of particle membership and images, but with
    updated projection transforms. You would use this output whenever you want
    downstream steps (for example, further refinement, comparison,
    classification in a common frame, or export) to interpret the particles
    according to the aligned volume’s coordinate system.

    A typical biological workflow is: align volume A to volume B, then apply
    A’s alignment transform to the particles that produced A, so that those
    particles are now consistent with the aligned A (and therefore comparable
    to B). This is especially helpful when you want to merge particle sets,
    compare angular distributions across processing branches, or build
    consistent “state-to-state” comparisons where each state was reconstructed
    independently but later brought into a common frame.

    What this protocol does not do (important for interpretation)

    This protocol does not improve alignment quality and does not validate
    anything. If the volume-to-volume alignment transform is wrong or ambiguous
    (for instance due to symmetry, masking issues, or heterogeneity), then the
    particle updates will also be wrong. In biological practice, it is
    therefore wise to first verify that the aligned volume truly matches the
    intended reference (visual overlay, landmark comparison, fitting of a known
    model, etc.) before propagating the transform to particles and using them
    for further quantitative analysis.

    Also, because this protocol updates only metadata, it is not the right tool
    if your goal is to create a physically transformed particle stack where
    pixels have been rotated/shifted. For that, you would use protocols that
    apply transforms to images (like 2D apply alignment or similar image
    transformation steps). Here, the objective is coordinate-frame consistency
    for projection parameters.

    Practical recommendations

    This protocol is most valuable when you are building multi-branch workflows
    and need all results to live in a shared frame. It is also helpful when
    preparing data for downstream steps that assume one consistent reference
    orientation, such as some comparative analyses or downstream integration
    with external modeling pipelines.

    As a sanity check after running, it is good practice to examine whether
    downstream steps that depend on orientation—such as reprojections, angular
    distributions, or quick refinements—behave as expected. If something looks
    “rotated wrong” globally, that often indicates that the volume transform
    being propagated was not the one you intended (or that symmetry introduced
    an alternative but equivalent solution).
    """
    
    _label = 'apply transformation matrix'
    _lastUpdateVersion = VERSION_1_1
    #--------------------------- DEFINE param functions ------------------------------------
    
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam, 
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",  
                      help="Aligned particles that their  "
                           "angular assignment needs to be modified.")        
        form.addParam('inputVolume', params.PointerParam,
                 pointerClass='Volume',
                 pointerCondition='hasTransform',
                 label='Input volume', 
                 help="Volume that we want to use its transformation matrix "
                      "to modify angular assignment of input particles. "
                      "(This is normally the output volume of protocol_"
                      "align_volume)")
        
        form.addParallelSection(threads=1, mpi=2)    
    #--------------------------- INSERT steps functions ------------------------------------
    
    def _insertAllSteps(self):        
        fnOutputParts = self._getExtraPath('output_particles.xmd')
        inputSet = self.inputParticles.get()
        inputVol = self.inputVolume.get()        
        self._insertFunctionStep('createOutputStep', 
                                 fnOutputParts, inputSet, inputVol)
    #--------------------------- STEPS functions --------------------------------------------        
    
    def createOutputStep(self, fnOutputParts, inputSet, inputVol):        
        volTransformMatrix = np.matrix(inputVol.getTransform().getMatrix())
        
        outputSet = self._createSetOfParticles()
        for part in inputSet.iterItems():
            partTransformMat = part.getTransform().getMatrix()       
            partTransformMatrix = np.matrix(partTransformMat)            
            newTransformMatrix = volTransformMatrix * partTransformMatrix
            part.getTransform().setMatrix(newTransformMatrix)
            outputSet.append(part)       
        
        outputSet.copyInfo(inputSet)        
        self._defineOutputs(outputParticles=outputSet)        
        writeSetOfParticles(outputSet, fnOutputParts)
    #--------------------------- INFO functions --------------------------------------------
    
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            summary.append("Applied alignment to %s particles." % 
                           self.inputParticles.get().getSize())
        return summary

    def _methods(self):
        if not hasattr(self, 'outputParticles'):
            return ["Output particles not ready yet."]
        else:
            return ["We applied alignment to %s particles from %s and produced %s."
                    %(self.inputParticles.get().getSize(), 
                      self.getObjectTag('inputParticles'), 
                      self.getObjectTag('outputParticles'))]
    