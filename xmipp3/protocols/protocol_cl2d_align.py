# ******************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
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
# ******************************************************************************w
import pyworkflow.protocol.params as params

import pwem.emlib.metadata as md
from pwem.objects import Particle, SetOfAverages, SetOfClasses2D
from pwem.protocols import ProtAlign2D
from pwem import ALIGN_NONE, ALIGN_2D

from xmipp3.convert import createItemMatrix, writeSetOfParticles, getImageLocation


class XmippProtCL2DAlign(ProtAlign2D):
    """ Aligns a set of particles using the CL2D algorithm, allowing for an
    optional reference 2D image. Accurate alignment is critical for improving
    class averages and achieving high-quality 2D classifications.

    AI Generated:

    This protocol uses the CL2D engine to perform a pure alignment of a
    particle stack in 2D. Conceptually, it runs CL2D in the special case of
    one reference class (--nref 1), so the algorithm iteratively estimates for
    every particle the in-plane rotation/shift that best matches a reference,
    and it stores those alignment parameters back into the particle
    transformations. The practical point is simple: you end up with a new
    particle set that has a consistent 2D alignment (in-plane rotation + x/y
    shifts), plus an average image that summarizes the aligned data. This can
    be used as a preprocessing step before 2D classification, for diagnostics,
    or for making a clean average quickly.

    What you provide as input

    You provide a SetOfParticles as input. The protocol writes them into an
    Xmipp metadata file (images.xmd) with alignment cleared (ALIGN_NONE) so
    that CL2D starts from the raw images without inheriting previous alignment
    information. The alignment results are then written by CL2D into that
    metadata and later imported back into Scipion objects.

    Reference strategy: with or without a user-provided reference

    The main user choice is whether to align against a specific reference
    image or to let CL2D generate one internally.

    If Use a Reference Image is enabled, you provide one reference image
    through the referenceImage parameter. The protocol accepts three types: a
    single Particle, a SetOfAverages, or a SetOfClasses2D. Internally it always
    resolves this to one image: if you provide a set (averages or classes), it
    simply takes the first item (for classes it takes the representative of the
    first class). This resolved image is passed to CL2D as
    --ref0 <imageLocation>, and all particles will be aligned to it. This
    mode is useful when you already have a trusted target view (for example, a
    good 2D average from a previous run) and you want the whole dataset
    brought into that reference frame.

    If Use a Reference Image is disabled, the protocol does not use --ref0.
    Instead it runs CL2D with --nref0 1, meaning CL2D will build an initial
    reference automatically (in practice, by averaging subsets according to
    its internal initialization). This is the usual “no-prior” mode when you
    just want a stable alignment without deciding in advance what the reference
    should look like.

    A key practical implication: using a reference can make alignment outcomes
    more reproducible and can force alignment toward a particular view, while
    “no reference” tends to produce a reference that reflects the dominant
    signal in your data (which can be good, but may also drift if your dataset
    has mixed views or strong junk).

    How strong the alignment is allowed to be

    Two parameters control the alignment search.

    Maximum shift (px) sets the translation search window. Biologically, this
    should match your expectation of miscentering: if particles are already
    well centered, small values are fine and faster; if extraction/centering is
    rough, too small a maximum shift will prevent correct alignment and you may
    see blurred averages.

    Number of iterations controls how many refinement rounds CL2D will do. More
    iterations give the algorithm more opportunity to converge, particularly
    when SNR is low, but the cost increases and over-refinement of noise is
    possible if the dataset is very heterogeneous or dominated by junk.

    What the protocol runs internally

    After converting input to Xmipp metadata (images.xmd), the protocol runs:

    xmipp_classify_CL2D -i images.xmd --odir <extraDir> --nref 1 --iter <N>
    --maxShift <S> ...

    with either --ref0 <reference> (reference mode) or --nref0 1 (no-reference
    mode). Even though the executable name says “classify”, here it is being
    used as an alignment tool because the classification is restricted to a
    single class; the only thing that changes across iterations is the estimated
    alignment of each particle to that class/reference.

    Outputs you get

    You get two outputs:

    outputAverage: a single Particle pointing to the CL2D-produced class
    average (internally it is taken from level_00/class_classes.stk, image
    index 1). This is the aligned average of your dataset with respect to the
    chosen (or internally generated) reference.

    outputParticles: a new SetOfParticles where each particle has a 2D
    transform imported from the CL2D metadata via createItemMatrix(...,
    align=ALIGN_2D). The set is explicitly marked as ALIGN_2D, and the protocol
    also sets the representative of the particle set to the produced average,
    which is handy for quick visualization.

    Validation checks (what can make it fail)

    The main hard requirement is that MPI must be > 1 (the protocol enforces
    this because it expects CL2D to run in parallel).

    If you use a reference image, the protocol checks that the reference and
    input particles have exactly the same dimensions. If sizes differ, the
    protocol refuses to run because CL2D cannot align images of different box
    sizes.

    When this protocol is most useful in practice

    In a typical workflow, this is useful when you want a quick, consistent 2D
    alignment for: generating a clean average for reporting/diagnostics,
    preparing particles for subsequent 2D classification (especially if your
    next step benefits from roughly aligned inputs), or aligning a subset
    against a known reference view to compare datasets. If you notice that
    results are dominated by a wrong view or drift, switching between
    “reference” and “no reference”, or tightening the maximum shift, are
    """
    _label = 'align with cl2d'
    
    # --------------------------- DEFINE param functions -----------------------
    def _defineAlignParams(self, form):
        form.addParam('useReferenceImage', params.BooleanParam, default=False,
                      label='Use a Reference Image ?',
                      help='If you set to *Yes*, you should provide a '
                           'reference image.\n'
                           'If *No*, the default generation is done by '
                           'averaging subsets of the input images.')
        form.addParam('referenceImage', params.PointerParam,
                      condition='useReferenceImage',
                      pointerClass='Particle, SetOfAverages, SetOfClasses2D', allowsNull=True,
                      label="Reference image",
                      help='Image that will serve as class reference. If the input is a set, then the first image '
                           'will be used as reference.')
        form.addParam('maximumShift', params.IntParam, default=10,
                      label='Maximum shift (px):')
        form.addParam('numberOfIterations', params.IntParam, default=10,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Number of iterations:',
                      help='Maximum number of iterations')
        form.addParallelSection(threads=0, mpi=4)
    
    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """ Mainly prepare the command line for call cl2d align program"""
        
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('images.xmd')
        self._insertFunctionStep('convertInputStep')
        
        # Prepare arguments to call program: xmipp_classify_CL2D
        self._params = {'imgsFn': self.imgsFn,
                        'extraDir': self._getExtraPath(),
                        'maxshift': self.maximumShift.get(),
                        'iter': self.numberOfIterations.get(),
                        }
        args = ('-i %(imgsFn)s --odir %(extraDir)s --nref 1 --iter %(iter)d '
                '--maxShift %(maxshift)d')
        
        if self.useReferenceImage:
            if isinstance(self.referenceImage.get(),Particle):
                args += " --ref0 " + getImageLocation(self.referenceImage.get())
            elif isinstance(self.referenceImage.get(), SetOfClasses2D):
                args += " --ref0 " + getImageLocation(self.referenceImage.get().getFirstItem().getRepresentative())
            else:
                args += " --ref0 " + getImageLocation(self.referenceImage.get().getFirstItem())
        else:
            args += " --nref0 1"
        self._insertRunJobStep("xmipp_classify_CL2D", args % self._params)
        
        self._insertFunctionStep('createOutputStep')
    
    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(), self.imgsFn,
                            alignType=ALIGN_NONE)
    
    def createOutputStep(self):
        """ Store the setOfParticles object
        as result of the protocol.
        """
        particles = self.inputParticles.get()
        # Define the output average
        avgFile = self._getExtraPath('level_00', 'class_classes.stk')
        avg = Particle()
        avg.setLocation(1, avgFile)
        avg.copyInfo(particles)
        self._defineOutputs(outputAverage=avg)
        self._defineSourceRelation(self.inputParticles, avg)
        
        # Generate the Set of Particles with alignment
        alignedSet = self._createSetOfParticles()
        alignedSet.copyInfo(particles)
        alignedSet.setRepresentative(avg)
        alignedSet.copyItems(particles,
                             updateItemCallback=self._createItemMatrix,
                             itemDataIterator=md.iterRows(self.imgsFn,
                                                          sortByLabel=md.MDL_ITEM_ID))
        alignedSet.setAlignment(ALIGN_2D)
        self._defineOutputs(outputParticles=alignedSet)
        self._defineSourceRelation(self.inputParticles, alignedSet)
    
    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        if self.numberOfMpi <= 1:
            errors.append('Mpi needs to be greater than 1.')
        if self.useReferenceImage:
            if self.referenceImage.hasValue():
                refImage = self.referenceImage.get()
                if isinstance(refImage,Particle):
                    [x1, y1, z1] = refImage.getDim()
                else:
                    [x1, y1, z1] = refImage.getFirstItem().getDim()
                [x2, y2, z2] = self.inputParticles.get().getDim()
                if x1 != x2 or y1 != y2 or z1 != z2:
                    errors.append('The input images and the reference image '
                                  'have different sizes')
            else:
                errors.append("Please, enter a reference image")
        return errors
    
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output alignment not ready yet.")
        else:
            summary.append("Input Particles: %s"
                           % self.inputParticles.get().getSize())
            if self.useReferenceImage:
                summary.append("Aligned with reference image: %s"
                               % self.referenceImage.get().getNameId())
            else:
                summary.append("Aligned with no reference image.")
        return summary
    
    def _citations(self):
        return ['Sorzano2010a']
    
    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output alignment not ready yet.")
        else:
            if self.useReferenceImage:
                methods.append(
                    "We aligned images %s with respect to the reference image "
                    "%s using CL2D [Sorzano2010a]"
                    % (self.getObjectTag('inputParticles'),
                       self.getObjectTag('referenceImage')))
            else:
                methods.append(
                    "We aligned images %s with no reference using CL2D "
                    "[Sorzano2010a]" % self.getObjectTag('inputParticles'))
            methods.append(" and produced %s images."
                           % self.getObjectTag('outputParticles'))
        return methods
    
    def _createItemMatrix(self, item, row):
        createItemMatrix(item, row, align=ALIGN_2D)
