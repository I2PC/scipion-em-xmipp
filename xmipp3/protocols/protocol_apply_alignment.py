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
import pyworkflow.protocol.params as params
from pyworkflow.utils.properties import Message

import pwem.emlib.metadata as md
from pwem.emlib.image import ImageHandler
from pwem.objects import Particle
from pwem.protocols import ProtAlign2D
from pwem.constants import ALIGN_NONE

from xmipp3.convert import xmippToLocation, writeSetOfParticles

        
class XmippProtApplyAlignment(ProtAlign2D):
    """ Applies previously calculated alignment parameters to a set of images,
    producing a new aligned dataset. This step is critical for improving the
    consistency and quality of images before further analysis or reconstruction.

    AI Generated:

    What this protocol is for

        Apply Alignment 2D takes a set of particles that already have 2D
        alignment parameters (in-plane rotation and shifts) and applies those
        transforms to the actual particle images, producing a new dataset where
        the images themselves are physically aligned. This is an important
        conceptual distinction for many biological users: after a 2D
        classification or alignment step, particles usually store alignment
        parameters in metadata, but the underlying image pixels remain
        unchanged. This protocol “bakes in” the alignment, generating a new
        aligned image stack that is often easier to use for visualization,
        downstream averaging, and certain workflows where you want aligned
        images without relying on metadata transforms.

        A typical reason to run this protocol is to produce a clean, directly
        comparable particle stack and a representative average image that
        reflects the aligned dataset. It is also useful when exporting
        particles to external tools or sharing a dataset where you want the
        alignment already applied.

        Input requirement

        The protocol accepts a single input: a SetOfParticles that must
        already contain 2D alignment information. In practical terms, the set
        must come from a step that has assigned 2D transforms (for example, a
        2D alignment or 2D classification protocol that computes in-plane
        alignment). If the particle set does not have 2D alignment, there is
        nothing to apply, and this protocol is not applicable.

        From a biological workflow perspective, this is typically used after
        you have obtained a set of well-aligned particles—often after cleaning
        out junk classes—so that the resulting aligned stack reflects
        meaningful particle centering and orientation.

        What happens during processing

        The protocol reads the input particle set, including each particle’s
        2D alignment parameters, and then applies the corresponding geometric
        transform to each particle image. The result is written to a new aligned
        stack (a new set of image files), and the output particle set is
        updated so that each particle now points to its newly transformed image.

        A key point for interpretation is that the protocol then removes the
        2D alignment metadata from the output particles. This is intentional:
        because the image pixels have already been transformed, the output set
        is considered “already aligned” in the image data itself and no longer
        needs 2D alignment parameters in the metadata. For the user, the
        simplest way to think about it is: the output particles are now in a
        consistent orientation/centering without requiring further application
        of transforms.

        Outputs and how to use them

        The protocol produces two outputs.

        The first is outputParticles, a new particle set whose images are the
        aligned versions of the originals. This is the dataset you would use
        if you want a particle stack that is visually coherent, straightforward
        to average, or ready for export. It is also convenient for manual
        inspection because particles appear consistently centered and oriented
        (to the extent that the input alignment was meaningful).

        The second output is outputAverage, which is a single average image
        computed from the aligned output particle set. Biologically, this
        average can be used as a quick quality indicator: if the alignment was
        good and the dataset is reasonably homogeneous in view, the average
        will look sharper and more structured than the average of unaligned
        particles. If the average remains blurry, it may indicate residual
        heterogeneity, poor alignment quality upstream, or that the particles
        represent many different orientations (in which case a single 2D
        average is not expected to be sharp).

        Practical guidance for biological users

        This protocol is most useful when you are at a stage where you trust
        the 2D alignment parameters and you want to materialize them into the
        images. A common pattern is to first run a 2D classification, select
        the good classes (thus filtering out junk and contaminants), and then
        apply alignment to the selected particles to obtain a clean aligned
        stack and a representative average.

        It is less useful if you plan to continue doing 2D classification or
        alignment steps that expect alignment metadata, because after applying
        alignment the output set no longer carries 2D alignment transforms.
        That does not prevent further processing, but it changes the
        interpretation: downstream tools will treat the images as already
        aligned rather than needing to apply stored transforms.

        Another common use is interoperability: if you need to export a
        particle stack for visualization or external processing and you want
        what you export to already reflect the alignment you computed in
        Scipion/Xmipp, this protocol provides a direct way to do that.

        What this protocol does not do

        Apply Alignment 2D does not compute new alignments and does not improve
        alignment by itself; it only applies what was computed previously. If
        the upstream alignment is wrong (for example, dominated by noise, or
        driven by junk particles), applying it will simply produce a
        consistently wrong aligned stack. That is why, in biological practice,
        it is usually best applied after you have already curated the dataset
        to contain mostly valid particles and reasonably consistent alignments.

        Recommended checks after running

        A quick sanity check is to inspect the outputAverage and compare it to
        an average before alignment. If the alignment was meaningful, the
        aligned average should show improved contrast and sharper structural
        features. It is also useful to browse a subset of outputParticles to
        confirm that centering and orientation look consistent and that no
        obvious artifacts were introduced.

        In short, this protocol is a practical “finalization” step that turns
        alignment metadata into an aligned particle stack and a representative
        aligned average, which can simplify downstream handling and improve
        interpretability for many common cryo-EM workflows.
    """
    _label = 'apply alignment 2d'

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam, important=True,  
                      pointerCondition='hasAlignment2D',
                      label=Message.LABEL_INPUT_PART, pointerClass='SetOfParticles',
                      help='Select the particles that you want to apply the'
                           'alignment parameters.')
        form.addParallelSection(threads=0, mpi=4)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    
    def _insertAllSteps(self):
        """ Mainly prepare the command line for call cl2d align program"""
        
        # Create a metadata with the geometrical information 
        # as expected by Xmipp
        imgsFn = self._getPath('input_particles.xmd')
        self._insertFunctionStep('convertInputStep', imgsFn)
        self._insertFunctionStep('applyAlignmentStep', imgsFn)
        self._insertFunctionStep('createOutputStep')        

    #--------------------------- STEPS functions --------------------------------------------        
    
    def convertInputStep(self, outputFn):
        """ Create a metadata with the images and geometrical information. """
        writeSetOfParticles(self.inputParticles.get(), outputFn)
        
        return [outputFn]
    
    def applyAlignmentStep(self, inputFn):
        """ Create a metadata with the images and geometrical information. """
        outputStk = self._getPath('aligned_particles.stk')
        args = '-i %(inputFn)s -o %(outputStk)s --apply_transform ' % locals()
        self.runJob('xmipp_transform_geometry', args)
        
        return [outputStk]
     
    def createOutputStep(self):
        particles = self.inputParticles.get()

        # Generate the SetOfAlignmet
        alignedSet = self._createSetOfParticles()
        alignedSet.copyInfo(particles)

        inputMd = self._getPath('aligned_particles.xmd')
        alignedSet.copyItems(particles,
                             updateItemCallback=self._updateItem,
                             itemDataIterator=md.iterRows(inputMd, sortByLabel=md.MDL_ITEM_ID))
        # Remove alignment 2D
        alignedSet.setAlignment(ALIGN_NONE)

        # Define the output average

        avgFile = self._getExtraPath("average.xmp")

        imgh = ImageHandler()
        avgImage = imgh.computeAverage(alignedSet)

        avgImage.write(avgFile)

        avg = Particle()
        avg.setLocation(1, avgFile)
        avg.copyInfo(alignedSet)

        self._defineOutputs(outputAverage=avg)
        self._defineSourceRelation(self.inputParticles, avg)

        self._defineOutputs(outputParticles=alignedSet)
        self._defineSourceRelation(self.inputParticles, alignedSet)
    
    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        return errors
        
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            summary.append("Applied alignment to %s particles." % self.inputParticles.get().getSize())
        return summary

    def _methods(self):
        if not hasattr(self, 'outputParticles'):
            return ["Output particles not ready yet."]
        else:
            return ["We applied alignment to %s particles from %s and produced %s."
                    % (self.inputParticles.get().getSize(), self.getObjectTag('inputParticles'), self.getObjectTag('outputParticles'))]
    
    #--------------------------- UTILS functions --------------------------------------------
    def _updateItem(self, item, row):
        """ Implement this function to do some
        update actions over each single item
        that will be stored in the output Set.
        """
        # By default update the item location (index, filename) with the new binary data location
        newFn = row.getValue(md.MDL_IMAGE)
        newLoc = xmippToLocation(newFn)
        item.setLocation(newLoc)
        # Also remove alignment info
        item.setTransform(None)

