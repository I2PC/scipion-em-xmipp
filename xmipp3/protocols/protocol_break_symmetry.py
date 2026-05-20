# **************************************************************************
# *
# * Authors:     Roberto Marabini (roberto@cnb.csic.es)
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

from pyworkflow.object import String
from pyworkflow.protocol.params import StringParam

from pwem.protocols import ProtProcessParticles
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import writeSetOfParticles


class XmippProtAngBreakSymmetry(ProtProcessParticles):
    """
    Given an input set of particles with angular assignment, find an
    equivalent angular assignment for a given symmetry.

    Be aware that input symmetry values follows Xmipp conventions as described
    in: https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry

    AI Generated:

    What this protocol is for

    Break symmetry is a protocol for particle sets that already have a 3D
    angular assignment (projection alignment). In symmetric specimens, a given
    3D orientation is not unique: there are multiple symmetry-related equivalent
    orientations that produce the same physical view under the symmetry group.
    Many pipelines store one of these equivalent solutions, but the particular
    representative chosen can vary across runs, software packages, or even
    across different parts of the same workflow.

    This protocol standardizes that situation by replacing each particle’s
    angular assignment with an equivalent assignment under the specified
    symmetry. In other words, it “moves” the particle angles to another
    symmetry-related version that is still valid, but follows a consistent
    convention. This can be very useful when you want to compare angular
    assignments between datasets, merge particles from multiple refinements,
    analyze angular distributions cleanly, or prepare particles for downstream
    steps where a consistent symmetry convention matters.

    Even though the label says “break symmetry”, for most biological users the
    practical interpretation is: choose a symmetry-consistent representative
    for each particle’s orientation, removing arbitrary symmetry-related
    differences in how orientations are expressed.

    When you would use it (typical biological scenarios)

    You usually run this protocol when:

    You have a symmetric complex (Cn, Dn, etc.) and you want to compare or
    combine results from different refinements where symmetry-related angle
    conventions differ.

    You are analyzing angular distributions and want to avoid artificial
    multi-cluster patterns caused purely by symmetry-equivalent representations.

    You are preparing a particle set for a downstream tool that expects a
    particular convention for symmetric orientations.

    You suspect that two particle sets are aligned equivalently but appear
    “different” in angles due to symmetry-related remapping; applying a
    consistent remapping can make them directly comparable.

    This protocol does not change the images or re-align particles by
    correlation; it only changes the stored orientation metadata to an
    equivalent symmetry-related one.

    Input requirement

    The input is the protocol’s inherited inputParticles (a SetOfParticles)
    that must already contain projection alignment (3D angles and shifts
    represented as a transform matrix). If particles do not have an angular
    assignment, there is nothing to remap, and the protocol is not applicable.

    The key parameter: Symmetry group

    You specify the symmetry group as a string using Xmipp symmetry conventions
    (for example c1, c2, c3, d7, etc.). If your specimen has no symmetry or you
    are working in an asymmetric refinement, use c1.

    This parameter must match the symmetry you want to use to define
    equivalences. If you provide the wrong symmetry, the protocol will still
    generate “equivalent” angles under that wrong symmetry, but they will not
    represent the biology of your particle and can confuse downstream analysis.

    A practical hint: use the same symmetry you used in the refinement that
    produced the angular assignments, unless you have a specific reason to
    remap under a different group.

    What the protocol does to your data

    Internally, the protocol exports particle metadata, runs an Xmipp symmetry
    remapping tool, and then rebuilds a new particle set where each particle
    keeps the same identity and image association but has an updated projection
    transform consistent with the remapped angles.

    The important consequence is that the output particle set remains a valid
    aligned dataset, but the angular assignments may look different
    numerically—while still representing the same physical orientations up to
    symmetry.

    Outputs and how to use them

    The protocol produces outputParticles, a new particle set with the remapped
    angular assignments. You would use this output in any step where you want
    symmetry-consistent angles, for example:

    comparing particle poses across runs,

    plotting angular distributions,

    exporting poses to other tools,

    merging aligned particle sets coming from different refinement branches.

    Because only metadata is changed, downstream refinement should remain
    consistent in terms of physical interpretation, but you gain a cleaner,
    more standardized representation of orientations.

    What this protocol does not do

    It does not refine angles, does not assess alignment quality, and does not
    “fix” incorrect orientations. If the original angular assignments are wrong
    due to misalignment or heterogeneity, this protocol will not correct that.
    It also does not change the symmetry of the reconstruction; it only uses
    the symmetry group to choose an equivalent representation of each
    particle’s pose.

    Practical checks after running

    A quick way to confirm the protocol did what you intended is to compare
    angular distribution plots before and after. For symmetric datasets, you
    often see that “symmetry-induced duplicates” in angle space collapse into a
    cleaner pattern. If instead the result looks inconsistent or dramatically
    different in a way that does not match symmetry expectations, the most
    common cause is an incorrect symmetry group string.

    Note on conventions

    Xmipp uses its own symmetry conventions and naming; the protocol explicitly
    follows those conventions. If you are unsure, consult the Xmipp symmetry
    documentation referenced in the protocol help and ensure your symmetry
    string matches that format.
    """
    _label = 'break symmetry'

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help="See https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry"
                           " for a description of the symmetry groups format in Xmipp.\n"
                           "If no symmetry is present, use _c1_.")
    
    def _getDefaultParallel(self):
        """This protocol doesn't have mpi version"""
        return (0, 0)
     
    #--------------------------- INSERT steps functions --------------------------------------------            
    def _insertAllSteps(self):
        """ Mainly prepare the command line for call brak symmetry program"""
        # Create a metadata with the geometrical information
        # as expected by Xmipp
        imgsFn = self._getPath('input_particles.xmd')
        self._insertFunctionStep(self.convertInputStep, imgsFn)
        self._insertFunctionStep(self.breakSymmetryStep, imgsFn)
        self._insertFunctionStep(self.createOutputStep)

    #--------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, outputFn):
        """ Create a metadata with the images and geometrical information. """
        writeSetOfParticles(self.inputParticles.get(), outputFn)

    #--------------------------- STEPS functions --------------------------------------------
    def breakSymmetryStep(self, imgsFn):
        outImagesMd = self._getPath('images.xmd')
        args = "-i Particles@%s --sym %s -o %s" % (imgsFn,
                                                 self.symmetryGroup.get(),
                                                 outImagesMd )
        self.runJob("xmipp_angular_break_symmetry", args)
        self.outputMd = String(outImagesMd)

    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(imgSet)
        partSet.copyItems(imgSet,
                          updateItemCallback=self._createItemMatrix,
                          itemDataIterator=md.iterRows(self.outputMd.get(), sortByLabel=md.MDL_ITEM_ID))
        
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(imgSet, partSet)

    #--------------------------- INFO functions --------------------------------------------                
    def _summary(self):
        import os
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            summary.append("Symmetry: %s"% self.symmetryGroup.get())
        return summary
    
    def _validate(self):
        pass
        
    def _citations(self):
        return []#['Vargas2013b']
    
    def _methods(self):
        methods = []
#        if hasattr(self, 'outputParticles'):
#            outParticles = len(self.outputParticles) if self.outputParticles is not None else None
#            particlesRejected = len(self.inputParticles.get())-outParticles if outParticles is not None else None
#            particlesRejectedText = ' ('+str(particlesRejected)+')' if particlesRejected is not None else ''
#            rejectionText = [
#                             '',# REJ_NONE
#                             ' and removing those not reaching %s%s' % (str(self.maxZscore.get()), particlesRejectedText),# REJ_MAXZSCORE
#                             ' and removing worst %s percent%s' % (str(self.percentage.get()), particlesRejectedText)# REJ_PERCENTAGE
#                             ]
#            methods.append('Input dataset %s of %s particles was sorted by'
#                           ' its ZScore using xmipp_image_sort_by_statistics'
#                           ' program%s. ' % (self.getObjectTag('inputParticles'), len(self.inputParticles.get()), rejectionText[self.autoParRejection.get()]))
#            methods.append('Output set is %s.'%self.getObjectTag('outputParticles'))
        return methods

    #--------------------------- Utils functions --------------------------------------------                
    def _createItemMatrix(self, item, row):
        from xmipp3.convert import createItemMatrix

        createItemMatrix(item, row, align=ALIGN_PROJ)

