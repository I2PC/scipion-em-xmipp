# *****************************************************************************
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
# *****************************************************************************

from os.path import join, exists

import pwem.emlib.metadata as md
import pyworkflow.utils.path as path
from pwem.objects import SetOfClasses2D
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, BooleanParam, IntParam,
                                        FloatParam)

from pwem.protocols import ProtClassify2D
from pwem.constants import ALIGN_2D


from xmipp3.convert import writeSetOfParticles, rowToAlignment, xmippToLocation


class XmippProtML2D(ProtClassify2D):
    """
    Perform (multi-reference) 2D-alignment using 
    a maximum-likelihood ( *ML* ) target function.
    
    Initial references can be generated from random subsets of the experimental
    images or can be provided by the user (this can introduce bias). The output
    of the protocol consists of the refined 2D classes (weighted averages over 
    all experimental images). The experimental images are not altered at all.    
    
    Although the calculations can be rather time-consuming (especially for 
    many, large experimental images and a large number of references we 
    strongly recommend to let the calculations converge.

    AI Generated

    ## Overview

    The ML2D protocol performs 2D alignment and classification of particle images
    using a maximum-likelihood target function.

    Unlike simple hard-assignment classification methods, maximum-likelihood 2D
    classification treats particle assignment and alignment in a probabilistic
    way. Each particle may contribute to the estimation of the 2D classes according
    to its likelihood under different orientations, shifts, and class references.
    This makes the method more robust when the images are noisy, as is usually the
    case in cryo-EM.

    The output of the protocol is a set of refined 2D classes. The experimental
    particles themselves are not modified; instead, they are assigned to classes
    and associated with refined 2D alignment parameters.

    ML2D is useful for obtaining representative 2D averages, separating different
    views or particle populations, and improving the organization of particle
    datasets before later 3D processing.

    ## Inputs and General Workflow

    The main input is a set of particle images. The protocol converts the input
    particles into Xmipp metadata format and then runs the selected maximum-
    likelihood 2D classification program.

    Initial references can be generated automatically from subsets of the input
    particles, or they can be provided by the user. The algorithm then iteratively
    updates the class references and particle assignments until convergence or
    until the maximum number of iterations is reached.

    At the end, the protocol creates a Scipion **SetOfClasses2D** containing the
    final 2D classes. Each class has a representative image, and particles are
    assigned to classes with their corresponding 2D alignment information.

    ## Input Particles

    The **Input particles** parameter defines the particle set to be classified.

    The particles should be reasonably prepared before running ML2D. In practical
    terms, this usually means that they should have been extracted with a sensible
    box size, normalized, and possibly preprocessed according to the needs of the
    workflow.

    ML2D can perform 2D alignment during classification, so the particles do not
    necessarily need to be perfectly aligned beforehand. However, very poorly
    centered particles, strong contaminants, or highly heterogeneous mixtures can
    make convergence slower and the resulting classes harder to interpret.

    As with all 2D classification methods, the quality of the output depends
    strongly on the quality of the input particle set.

    ## Generating Initial Classes

    The option **Generate classes?** controls how the initial 2D references are
    obtained.

    If this option is set to **Yes**, the protocol generates the initial class
    references automatically, typically by averaging subsets of the experimental
    images. This is often the preferred option when the user wants to reduce
    reference bias.

    If the option is set to **No**, the user must provide a set of class images to
    be used as initial references. This can be useful when good prior references
    are available, for example from a previous classification run or from a related
    dataset.

    However, user-provided references can introduce bias. If the initial references
    are too specific or incorrect, the classification may be guided toward those
    patterns. Automatic reference generation is therefore often safer for
    exploratory analysis.

    ## Number of Classes

    When initial references are generated automatically, the **Number of classes**
    parameter defines how many 2D classes will be produced.

    A small number of classes gives a compact summary of the dataset, but may merge
    different views, conformations, or particle qualities. A large number of
    classes provides more detail, but each class may contain fewer particles and
    therefore have a noisier average.

    The appropriate number depends on the dataset size and heterogeneity. For small
    or homogeneous datasets, fewer classes may be sufficient. For large or
    heterogeneous datasets, more classes may be needed to separate different views
    or particle populations.

    If the user provides initial references, the number of classes is determined by
    the number of provided reference images.

    ## User-Provided Class Images

    The **Class image(s)** parameter is used when automatic class generation is
    disabled.

    These images serve as the initial 2D references for the maximum-likelihood
    classification. They may come from a previous classification, manually selected
    averages, or another related processing workflow.

    This option can accelerate convergence when the references are appropriate.
    However, it can also bias the classification toward the provided references.
    For this reason, it should be used carefully, especially when the goal is to
    discover unexpected heterogeneity.

    ## ML2D and MLF2D

    The option **Use MLF2D instead of ML2D?** switches from the standard ML2D
    method to a Fourier-space maximum-likelihood variant.

    Standard ML2D works in image space. MLF2D performs maximum-likelihood
    classification in Fourier space and can use CTF-related information more
    explicitly.

    When MLF2D is selected, the input particles must contain CTF information. The
    protocol checks this requirement before running.

    The Fourier-space variant is useful when the user wants the classification to
    take microscope transfer effects into account more directly. However, it also
    requires that the particle metadata contain reliable CTF information.

    ## CTF-Amplitude Correction

    When MLF2D is used, the option **Use CTF-amplitude correction?** controls
    whether CTF amplitude correction is applied.

    If enabled, the method uses the CTF information associated with the particles.
    This can make the Fourier-space comparison more physically meaningful.

    If disabled, the protocol does not use this CTF correction in the same way, and
    the image pixel size must be available so that the Fourier-space calculations
    are properly scaled.

    This option is relevant only for MLF2D. In routine use, it should usually be
    left enabled when reliable CTF information is available.

    ## Phase-Flipped Images

    The **Are the images CTF phase flipped?** option tells MLF2D whether the input
    particles have already been phase-flipped.

    This is important because phase flipping changes the relationship between the
    particle images and the CTF model. If this option is set incorrectly, the
    Fourier-space likelihood calculation may use an inconsistent CTF convention.

    Users should check the preprocessing history of the particles before setting
    this option. If particles were extracted or processed with phase flipping, this
    should be indicated here.

    ## High-Resolution Limit

    The **High-resolution limit** parameter is used in MLF2D to exclude frequencies
    beyond a given resolution from the classification.

    For example, a limit of 20 Å means that frequencies higher than 20 Å are not
    used in the likelihood calculation. If the value is set to zero, no such limit
    is imposed.

    Excluding very high-resolution frequencies can be useful because these
    frequencies may be dominated by noise, especially in early processing stages.
    A conservative limit can make classification more stable and less sensitive to
    high-frequency noise.

    This parameter should be chosen according to the quality of the particles and
    the stage of processing. Early exploratory classification usually benefits from
    a relatively low-resolution comparison.

    ## Mirror Alignment

    The option **Also include mirror in the alignment?** allows the alignment
    search to include mirrored versions of the particle images.

    This can be useful when particles may appear in mirror-related views, for
    example when they can adsorb to the grid in different orientations such as
    face-up and face-down.

    Including mirrors makes the search more flexible, but it also increases the
    space of possible transformations. Users should keep it enabled when mirror
    ambiguity is expected and disable it only when such transformations are known
    to be inappropriate for the dataset.

    ## Fast Version

    The **Use the fast version?** option is available for standard ML2D.

    When enabled, the protocol uses a reduced search-space approach to accelerate
    the computation. This can be important because maximum-likelihood
    classification can be computationally expensive, especially with many
    particles, large image boxes, many classes, or fine angular sampling.

    The fast version is usually a good practical choice. However, because it avoids
    searching the complete solution space, users should be cautious when working
    with very difficult datasets or when maximum classification accuracy is more
    important than speed.

    ## Normalization Refinement

    The option **Refine the normalization for each image?** enables a variant of
    the algorithm that accounts for image-normalization errors.

    This can be useful when particles have residual differences in intensity scale
    or background that were not fully corrected during preprocessing. By refining
    normalization at the particle level, the classification may become less
    sensitive to such variations.

    However, this option adds model flexibility and may increase computation time.
    It should be used when normalization differences are suspected to affect the
    classification.

    ## Maximum Number of Iterations

    The **Maximum number of iterations** parameter defines when the iterative
    process should stop if convergence has not already been reached.

    Maximum-likelihood classification can require many iterations, especially for
    large or heterogeneous datasets. Stopping too early may produce poorly
    converged classes, while allowing more iterations gives the algorithm more
    opportunity to stabilize.

    The default value is intended to allow convergence in many cases. Users should
    avoid stopping the protocol prematurely unless there is a clear reason.

    ## In-Plane Rotation Sampling

    The **In-plane rotation sampling** parameter defines the angular step, in
    degrees, used when searching over rotations within the image plane.

    A smaller step gives a finer angular search and can improve alignment accuracy,
    but it increases computation time. A larger step is faster but may give less
    accurate alignments.

    The appropriate value depends on particle size, expected angular precision,
    image quality, and computational resources. The default value is a reasonable
    starting point for many datasets.

    ## Noise and Offset Parameters

    The **Std for pixel noise** parameter defines the expected standard deviation
    of the pixel noise used by the maximum-likelihood model.

    The **Std for origin offset** parameter defines the expected standard deviation
    of particle shifts, in pixels.

    These parameters influence the probabilistic model used during classification.
    They are advanced options and should usually be left at their default values
    unless the user has a specific reason to tune the likelihood model.

    The origin-offset parameter is particularly relevant when particles are not
    well centered. A larger value allows larger shifts, whereas a smaller value
    assumes that particles are already well centered.

    ## Outputs and Their Interpretation

    The main output is a **SetOfClasses2D**.

    Each class contains particles assigned to that class and a representative 2D
    average. The particles also receive 2D alignment information derived from the
    classification.

    The class averages should be inspected visually. Good classes usually show
    clear structural features, consistent particle views, and reasonable particle
    counts. Poor classes may be noisy, contain contaminants, represent badly
    centered particles, or mix incompatible views.

    The output classes can be used for particle cleaning, subset selection,
    exploratory structural analysis, or as preparation for later 3D processing.

    ## Practical Recommendations

    Use automatic reference generation for exploratory classification, especially
    when you want to avoid reference bias.

    Use user-provided references only when they are reliable and when the goal is
    to refine or reproduce a known classification.

    Choose the number of classes according to dataset size and heterogeneity. Too
    few classes may merge important differences; too many classes may produce noisy
    or empty-looking averages.

    Let the protocol converge when possible. Maximum-likelihood classification can
    be time-consuming, but premature stopping may reduce class quality.

    Use MLF2D only when the particles have reliable CTF information and when
    Fourier-space treatment is desired.

    Check whether particles are phase-flipped before using MLF2D, and set the
    corresponding option consistently.

    Inspect the final class averages and particle distribution across classes.
    Classes with clear density and reasonable particle counts are usually more
    trustworthy than very small or noisy classes.

    ## Final Perspective

    ML2D is a probabilistic 2D alignment and classification protocol. Its main
    strength is that it can handle noisy cryo-EM particles by estimating class
    averages and particle alignments within a maximum-likelihood framework.

    For biological users, ML2D is useful for producing interpretable 2D averages,
    evaluating particle quality, separating different views or populations, and
    preparing particle subsets for further 3D analysis.

    The most important practical choices are the number of classes, the use of
    automatic or provided references, the possible use of Fourier-space MLF2D, and
    whether the protocol is allowed enough iterations to converge.

    """
    _label = 'ml2d'
    
    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        
    def _defineFileNames(self):
        """ Centralize how files are called within the protocol. """
        myDict = {
                  'input_particles': self._getTmpPath('input_particles.xmd'),
                  'input_references': self._getTmpPath('input_references.xmd'),
                  'output_classes': self._getOroot() + 'classes.xmd', # self._getExtraPath('classes.xmd'),
                  'final_classes': self._getPath('classes2D.sqlite'),
                  'output_particles':  self._getOroot() + 'images.xmd',# self._getExtraPath('images.xmd'),
                  'classes_scipion': self._getPath('classes_scipion_iter_%(iter)02d.sqlite')
                  }
        self._updateFilenamesDict(myDict)
    
    #--------------------------- DEFINE param functions -----------------------
    
    def _defineParams(self, form):
        form.addSection(label='Params')
        
        group = form.addGroup('Input')
        group.addParam('inputParticles', PointerParam,
                       pointerClass='SetOfParticles',
                       label="Input particles", important=True,
                       help='Select the input images from the project.')        
        group.addParam('doGenerateReferences', BooleanParam, default=True,
                      label='Generate classes?',
                      help='If you set to *No*, you should provide class '
                           'images. If *Yes*, the default generation is done '
                           'by averaging subsets of the input images (less '
                           'bias introduced).')
        group.addParam('numberOfClasses', IntParam, default=3,
                       condition='doGenerateReferences',
                      label='Number of classes:',
                      help='Number of classes to be generated.')
        group.addParam('inputReferences', PointerParam, allowsNull=True,
                       condition='not doGenerateReferences',
                      label="Class image(s)",
                      pointerClass='SetOfParticles',
                      help='Image(s) that will serve as initial 2D classes')
        
        form.addParam('doMlf', BooleanParam, default=False, important=True,
                      label='Use MLF2D instead of ML2D?')
        
        group = form.addGroup('ML-Fourier', condition='doMlf')
        group.addParam('doCorrectAmplitudes', BooleanParam, default=True,
                      label='Use CTF-amplitude correction?',
                      help='If set to *Yes*, the input images file should '
                           'contains.\n If set to *No*, provide the images '
                           'pixel size in Angstrom.')
        group.addParam('areImagesPhaseFlipped', BooleanParam, default=True,
                      label='Are the images CTF phase flipped?',
                      help='You can run MLF with or without having phase flipped the images.')        
        group.addParam('highResLimit', IntParam, default=20,
                      label='High-resolution limit (Ang)',
                      help='No frequencies higher than this limit will be taken into account.\n'
                           'If zero is given, no limit is imposed.')
        
        form.addSection(label='Advanced')
        form.addParam('doMirror', BooleanParam, default=True,
                      label='Also include mirror in the alignment?',
                      help='Including the mirror transformation is useful if your particles'
                           'have a handedness and may fall either face-up or face-down on the grid.'
                           )
        form.addParam('doFast', BooleanParam, default=True, condition='not doMlf',
                      label='Use the fast version?',
                      help='If set to *Yes*, a fast approach will be used to avoid\n'
                           'searching in the whole solutions space.             \n\n'
                           'For details see (and please cite): \n' + self._getCite('Scheres2005b')
                           )
        form.addParam('doNorm', BooleanParam, default=False,
                      label='Refine the normalization for each image?',
                      help='This variant of the algorithm deals with normalization errors. \n\n'
                           'For details see (and please cite): \n ' + self._getCite('Scheres2009b')
                           )             
        # Advance or expert parameters
        form.addParam('maxIters', IntParam, default=100, expertLevel=LEVEL_ADVANCED,
                      label='Maximum number of iterations',
                      help='If the convergence has not been reached after this number'
                           'of iterations, the process will be stopped.')   
        form.addParam('psiStep', FloatParam, default=5.0, expertLevel=LEVEL_ADVANCED,
                      label='In-plane rotation sampling (degrees)',
                      help='In-plane rotation sampling interval (degrees).')          
        form.addParam('stdNoise', FloatParam, default=1.0, expertLevel=LEVEL_ADVANCED,
                      label='Std for pixel noise',
                      help='Expected standard deviation for pixel noise.')               
        form.addParam('stdOffset', FloatParam, default=3.0,
                      expertLevel=LEVEL_ADVANCED, label='Std for origin offset',
                      help='Expected standard deviation for origin offset (pixels).') 
        
        form.addParallelSection(threads=2, mpi=4)
           
    #--------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._defineFileNames()
        partSetObjId = self.inputParticles.get().getObjId()
        self._insertFunctionStep('convertInputStep', partSetObjId)
        program = self._getMLProgram()
        params = self._getMLParams()
        self._insertRunJobStep(program, params)
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self, inputId):
        """ Write the input images as a Xmipp metadata file. """
        writeSetOfParticles(self.inputParticles.get(),
                            self._getFileName('input_particles'))
        # If input references, also convert to xmipp metadata
        if not self.doGenerateReferences:
            writeSetOfParticles(self.inputReferences.get(),
                                self._getFileName('input_references'))

    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        
        classes2DSet = self._createSetOfClasses2D(imgSet)
        
        self._fillClassesFromIter(classes2DSet, "last")
        
        self._defineOutputs(outputClasses=classes2DSet)
        self._defineSourceRelation(self.inputParticles, classes2DSet)
        if not self.doGenerateReferences:
            self._defineSourceRelation(self.inputReferences, classes2DSet)
    
    #--------------------------- INFO functions -------------------------------------------- 
    
    def _validate(self):
        errors = []
        if self.doMlf:
            inputParticles = self.inputParticles.get()
            if inputParticles is not None and not inputParticles.hasCTF():
                errors.append('Input particles does not have CTF information.\n'
                              'This is required when using ML in fourier space.')
        return errors
    
    def _citations(self):
        cites = ['Scheres2005a']
        
        if self.doMlf:
            cites.append('Scheres2007b')
            
        elif self.doFast:
            cites.append('Scheres2005b')
            
        if self.doNorm:
            cites.append('Scheres2009b')
            
        return cites
    
    def _summary(self):
        summary = []
        if hasattr(self, 'outputClasses'):
            summary.append('Input Particles: *%d*' % self.inputParticles.get().getSize())
            summary.append('Classified into *%d* classes' % self.numberOfClasses.get())

            if self.doMlf:
                summary.append('- Used a ML in _Fourier-space_')
            elif self.doFast:
                summary.append('- Used _fast_, reduced search-space approach')

            if self.doNorm:
                summary.append('- Refined _normalization_ for each experimental image')
            
        return summary
    
    def _methods(self):
        methods = []
        if hasattr(self, 'outputClasses'):
            methods.append('Input dataset %s of *%d* images was classified' % (self.getObjectTag('inputParticles'), self.inputParticles.get().getSize()))
            numberOfClasses = self.numberOfClasses.get()
            classesTxt =  'class' if numberOfClasses == 1 else 'classes'
            methods.append('into *%d* 2D %s using Maximum Likelihood (ML) inside Xmipp.' % (numberOfClasses, classesTxt))

            if self.doMlf:
                methods.append('ML was used in _Fourier-space_.')
            elif self.doFast:
                methods.append('Used _fast_, reduced search-space approach.')

            if self.doNorm:
                methods.append('The _normalization_ was refined for each experimental image.')
            methods.append('Output set is %s.'%(self.getObjectTag('outputClasses')))

        return methods
    
    #--------------------------- UTILS functions --------------------------------------------
    def _getMLParams(self):
        """ Mainly prepare the command line for call ml(f)2d program"""
        params = ' -i %s --oroot %s' % (self._getFileName('input_particles'), self._getOroot()) #self._getPath()
        if self.doGenerateReferences:
            params += ' --nref %d' % self.numberOfClasses.get()
            self.inputReferences.set(None)
        else:
            params += ' --ref %s' % self._getFileName('input_references')
            self.numberOfClasses.set(self.inputReferences.get().getSize())
        
        if self.doMlf:
            if not self.doCorrectAmplitudes:
                params += ' --no_ctf'                    
            if not self.areImagesPhaseFlipped:
                params += ' --not_phase_flipped'
            if self.highResLimit > 0:
                params += ' --limit_resolution 0 %f' % self.highResLimit.get()
            params += ' --sampling_rate %f' % self.inputParticles.get().getSamplingRate()
        else:
            if self.doFast:
                params += ' --fast'
            if self.numberOfThreads > 1:
                params += ' --thr %d' % self.numberOfThreads.get()
            
        if self.maxIters != 100:
            params += ' --iter %d' % self.maxIters.get()

        if self.doMirror:
            params += ' --mirror'
            
        if self.doNorm:
            params += ' --norm'

        return params
        
    def _getIterMdClasses(self, it=None, block="classes"):
        """ Return the classes metadata for this iteration.
        block parameter can be 'info' or 'classes'."""
        if it == "last":
            return self._getFileName('output_classes')
        else:
            return self._getIterMdFile("classes", it, block)
    
    def _getIterMdImages(self, it=None, block=None):
        """ Return the images metadata for this iteration."""
        if it == "last":
            return self._getFileName('output_particles')
        else:
            return self._getIterMdFile("images", it, block)
    
    def _getIterMdFile(self, fn, it, block):
        if it is None:
            it = self._lastIteration()
        extra = self._getOroot() + 'extra'
        mdFile = join(extra, 'iter%03d' % it, 'iter_%s.xmd' %fn)
        #if block:
            #mdFile = block + '@' + mdFile

        return mdFile
    
    def _lastIteration(self):
        """ Find the last iteration number """
        it = -1
        while True:
            if not exists(self._getIterMdClasses(it+1)):
                break
            it += 1
        return it        
    
    def _getMLId(self):
        """ Return ml or mlf depending if using fourier or not. """
        if self.doMlf:
            return 'mlf'
        return 'ml'
    
    def _getMLProgram(self):
        """ Return the program to be used, depending if using fourier. """
        return "xmipp_%s_align2d" % self._getMLId()
    
    def _getOroot(self):        
        return self._getPath('%s2d_' % self._getMLId())
    
    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(md.MDL_REF))
        item.setTransform(rowToAlignment(row, ALIGN_2D))
        
    def _updateClass(self, item):
        classId = item.getObjId()
        
        if classId in self._classesInfo:
            index, fn, _ = self._classesInfo[classId]
            item.setAlignment2D()
            item.getRepresentative().setLocation(index, fn)
    
    def _loadClassesInfo(self, filename):
        """ Read some information about the produced 2D classes
        from the metadata file.
        """
        self._classesInfo = {} # store classes info, indexed by class id
        
        mdClasses = md.MetaData(filename)
        
        for classNumber, row in enumerate(md.iterRows(mdClasses)):
            index, fn = xmippToLocation(row.getValue(md.MDL_IMAGE))
            # Store info indexed by id, we need to store the row.clone() since
            # the same reference is used for iteration            
            self._classesInfo[classNumber+1] = (index, fn, row.clone())
    
    def _fillClassesFromIter(self, clsSet, iteration):
        """ Create the SetOfClasses2D from a given iteration. """
        self._loadClassesInfo(self._getIterMdClasses(iteration))
        dataXmd = self._getIterMdImages(iteration)
        clsSet.classifyItems(updateItemCallback=self._updateParticle,
                             updateClassCallback=self._updateClass,
                             itemDataIterator=md.iterRows(dataXmd,
                                                          sortByLabel=md.MDL_ITEM_ID))
    
    def _getIterClasses(self, it, clean=False):
        """ Return a classes .sqlite file for this iteration.
        If the file doesn't exists, it will be created by 
        converting from this iteration iter_images.xmd file.
        """
        dataClasses = self._getFileName('classes_scipion', iter=it)
         
        if clean:
            path.cleanPath(dataClasses)
        
        if not exists(dataClasses):
            clsSet = SetOfClasses2D(filename=dataClasses)
            clsSet.setImages(self.inputParticles.get())
            self._fillClassesFromIter(clsSet, it)
            clsSet.write()
            clsSet.close()
        
        return dataClasses
