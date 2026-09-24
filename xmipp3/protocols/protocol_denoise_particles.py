# **************************************************************************
# *
# * Authors:     I. Foche (ifoche@cnb.csic.es)
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

import pwem.emlib.metadata as md
from pyworkflow.object import String
from pyworkflow.protocol.params import IntParam, PointerParam, LEVEL_ADVANCED
from pwem.protocols import ProtProcessParticles
from pwem.objects import SetOfVolumes
from pwem.objects import SetOfAverages

from xmipp3.convert import (writeSetOfParticles, writeSetOfClasses2D,
                            xmippToLocation)

        
class XmippProtDenoiseParticles(ProtProcessParticles):
    """ Remove particles noise by filtering them. 
    This filtering process is based on a projection over a basis created
    from some averages (extracted from classes). This filtering is not 
    intended for processing particles. The huge filtering they will be 
    passed through is known to remove part of the signal with the noise. 
    However this is a good method for clearly see which particle are we 
    going to process before it's done.

    AI Generated

    ## Overview

    The Denoise Particles protocol produces a strongly filtered version of a
    particle set by projecting the particles onto a reduced basis learned from
    2D class averages or averages supplied by the user. The goal is not to
    create particles for high-resolution reconstruction, but to obtain
    cleaner-looking images that make it easier to visually understand what
    kind of particles are present in the dataset.

    In practical cryo-EM workflows, this protocol is mainly a **visualization
    and inspection tool**. It can help the user assess whether a set of
    particles is broadly consistent with a given 2D classification, whether
    the particles contain recognizable structural signal, or whether large
    fractions of the dataset look suspicious. Because the filtering is
    intentionally strong, it also removes part of the true signal together
    with the noise. For that reason, the output should not be interpreted as a
    faithful restoration of the experimental images.

    For a biological user, the main value of this protocol is that it gives a
    simplified, less noisy view of the data, which is often useful before
    making decisions about subsequent processing.

    ## Inputs and General Workflow

    The protocol requires two inputs: a **set of particles** to be denoised and
    a **set of classes or averages** used to construct the basis for denoising.

    The input particles must already have **alignment information** relative to
    the chosen classes or averages. This is the standard situation after 2D
    classification methods such as CL2D or ML2D. This requirement is important
    because the denoising relies on the particles being expressed in a
    coordinate system that is consistent with the basis learned from the class
    information.

    The workflow has two conceptual stages. First, the protocol builds a
    PCA-like basis from the supplied classes or averages. Second, each particle
    is projected onto a selected number of basis components, and the particle
    is reconstructed from that reduced representation. The reconstructed image
    is smoother and less noisy than the original one, but it is also more
    model-driven.

    ## The Role of the Input Classes or Averages

    The classes or averages define the space of images in which the particles
    will be represented. In other words, the denoising is guided by the
    dominant patterns present in those reference images.

    Biologically, this means that the result depends strongly on the quality
    and relevance of the chosen classes. If the input classes represent the
    real structural views present in the particles, the denoised images can
    become much easier to interpret. If the classes are poor, heterogeneous, or
    not representative of the particles, the denoising may produce misleading
    images.

    This is therefore not a generic image denoiser. It is a
    **class-guided denoising procedure**, and its output reflects the image
    space spanned by the chosen classes or averages.

    ## Basis Construction

    The protocol constructs a reduced basis using a rotational PCA procedure.
    The purpose of this basis is to capture the main modes of variation present
    in the input classes or averages.

    Two parameters are especially relevant here. The
    **maximum number of classes** controls how many reference images are used
    to build the basis. The **maximum number of PCA bases** controls how many
    components are retained during basis construction.

    In practice, using more classes can make the basis more representative of
    the structural diversity in the data, but also potentially more
    heterogeneous. Using fewer classes produces a simpler basis, which may be
    easier to interpret but may not capture all meaningful views.

    Similarly, a larger number of PCA components allows a richer representation,
    while a smaller number enforces stronger simplification.

    ## Denoising by Projection onto the Basis

    Once the basis has been computed, each particle is projected onto a chosen
    number of PCA components, and the image is reconstructed from that
    projection.

    This is the central denoising step. By representing each particle with only
    the main components of the basis, much of the small-scale noise is
    suppressed. At the same time, image details that are not well represented
    by the basis may also disappear.

    The key parameter here is the **number of PCA bases on which to project**.
    Smaller values lead to more aggressive denoising and smoother particles,
    but at the risk of oversimplifying them. Larger values preserve more
    detail, but also keep more noise.

    A practical way to think about this is that the number of projection bases
    controls the balance between interpretability and faithfulness to the
    original data.

    ## Interpretation of the Output

    The output is a new **set of particles** containing denoised images. These
    particles preserve the identity of the input particles, but their image
    content has been replaced by a filtered reconstruction derived from the
    learned basis.

    These denoised particles are useful for:
    * visually inspecting the dominant particle appearance,
    * checking whether the dataset broadly matches the chosen classes,
    * identifying obvious mismatches or poor-quality subsets,
    * preparing exploratory visual summaries of particle content.

    However, they should **not** be used as a substitute for the original
    particles in serious downstream processing such as high-resolution
    refinement or final reconstruction. The protocol documentation is explicit
    on this point: the filtering is strong and removes genuine signal together
    with noise.

    From a biological perspective, the output is best interpreted as an
    illustrative representation of the particles, not as a more accurate

    ## Practical Recommendations

    This protocol is best used after a meaningful 2D classification, when the
    user already has class averages that reflect the major views present in the
    dataset. In that situation, the denoised particles can provide a very
    intuitive picture of what the particle set contains.

    A good starting point is to use a reasonably representative class set and a
    moderate number of PCA components. If the denoised particles still look too
    noisy, the number of projection components can be reduced. If they look
    oversmoothed or overly similar, it may be better to increase the number of
    components or revise the class set used to build the basis.

    Users should be cautious when the dataset contains substantial
    heterogeneity. If the class set represents only one subset of the data,
    particles belonging to other states may be poorly represented and may be
    denoised into misleading images.

    ## Outputs and Their Interpretation

    The protocol produces a new set called **outputParticles**, which contains
    the denoised version of the input particle images.

    The output keeps the general metadata and identity of the original set, but
    the particle images now correspond to the filtered reconstructions. This
    makes it easy to compare the denoised particles with the originals or to
    use them for visual inspection in the same workflow context.

    ## Final Perspective

    The Denoise Particles protocol is a guided image-simplification tool
    designed to help users look at noisy particle data through the lens of an
    existing class-based representation. Its main purpose is not to improve the
    dataset for reconstruction, but to make it easier to understand and inspect.

    For most cryo-EM users, it should be viewed as a convenient visualization
    aid: useful for exploring, checking, and communicating what the particles
    look like, but not as a replacement for the original experimental images in
    quantitative processing.
    """
    _label = 'denoise particles'

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        # First we customize the inputParticles param to fit our needs in this protocol
        form.getParam('inputParticles').pointerCondition = String('hasAlignment')
        form.getParam('inputParticles').help = String('Input images you want to filter. It is important that the images have alignment information with '
                                                      'respect to the chosen set of classes. This is the standard situation '
                                                      'after CL2D or ML2D.')
        form.addParam('inputClasses', PointerParam, label='Input Classes', important=True,
                      pointerClass='SetOfClasses, SetOfAverages',
                      help='Select the input classes for the basis construction against images will be projected to.')
        
        form.addSection(label='Basis construction')
        form.addParam('maxClasses', IntParam, default=128,
                      label='Max. number of classes', expertLevel=LEVEL_ADVANCED,
                      help='Maximum number of classes.')
        form.addParam('maxPCABases', IntParam, default=200,
                      label='Number of PCA bases', expertLevel=LEVEL_ADVANCED,
                      help='Number of PCA bases.')
        form.addSection(label='Denoising')
        form.addParam('PCABases2Project', IntParam, default=200,
                      label='Number of PCA bases on which to project', expertLevel=LEVEL_ADVANCED,
                      help='Number of PCA bases on which to project.')
        
    def _getDefaultParallel(self):
        """ Return the default value for thread and MPI
        for the parallel section definition.
        """
        return (2, 4)
     
    #--------------------------- INSERT steps functions --------------------------------------------            
    def _insertAllSteps(self):
        """ Insert every step of the protocol"""
        
        # Convert input images if necessary
        self._insertFunctionStep('denoiseImages')
        
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions --------------------------------------------
    def denoiseImages(self):
        # We start preparing writing those elements we're using as input to keep them untouched
        imagesMd = self._getPath('images.xmd')
        writeSetOfParticles(self.inputParticles.get(), imagesMd)
        classesMd = self._getPath('classes.xmd')
        if isinstance(self.inputClasses.get(), SetOfAverages):
            writeSetOfParticles(self.inputClasses.get(), classesMd)
        else:
            writeSetOfClasses2D(self.inputClasses.get(), classesMd)

        fnRoot = self._getExtraPath('pca')
        fnRootDenoised = self._getExtraPath('imagesDenoised')

        args = ('-i Particles@%s --oroot %s --eigenvectors %d --maxImages %d'
                % (imagesMd, fnRoot, self.maxPCABases.get(), self.maxClasses.get()))
        self.runJob("xmipp_image_rotational_pca", args)

        N=min(self.maxPCABases.get(), self.PCABases2Project.get())
        args='-i %s -o %s.stk --save_metadata_stack %s.xmd --basis %s.stk %d'\
             % (imagesMd, fnRootDenoised, fnRootDenoised, fnRoot, N)

        self.runJob("xmipp_transform_filter", args)

        self.outputMd = String('%s.stk' % fnRootDenoised)

    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        partSet = self._createSetOfParticles()
        
        partSet.copyInfo(imgSet)
        partSet.copyItems(imgSet,
                            updateItemCallback=self._updateLocation,
                            itemDataIterator=md.iterRows(self.outputMd.get(), sortByLabel=md.MDL_ITEM_ID))
        
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(imgSet, partSet)
    
    #--------------------------- INFO functions --------------------------------------------                
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            summary.append('PCA basis created by using %d classes' % len(self.inputClasses.get()))
            summary.append('Max. number of classes defined for PCA basis creation: %d' % self.maxClasses.get())
            summary.append('Max. number of PCA bases defined for PCA basis creation: %d' % self.maxPCABases.get())
            summary.append('PCA basis on which to project for denoising: %d' % self.PCABases2Project.get())
        return summary
    
    def _validate(self):
        pass
        
    def _citations(self):
        return ['zhao2013', 'ponce2011']
    
    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append('An input dataset of %d particles was filtered creating a PCA basis (%d components) with '
                           'xmipp_image_rotational_pca and projecting the dataset into that base with xmipp_transform_filter.'\
                           % (len(self.inputParticles.get()), len(self.inputClasses.get())))
        return methods
    
    #--------------------------- UTILS functions --------------------------------------------
    def _updateLocation(self, item, row):
        index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
        item.setLocation(index, filename)

