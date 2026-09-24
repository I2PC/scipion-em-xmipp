# ******************************************************************************
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
# ******************************************************************************

from os.path import join, dirname, exists
from glob import glob


import pyworkflow.protocol.params as param
import pyworkflow.protocol.constants as const
from pyworkflow.utils.path import cleanPath, makePath

import pwem.emlib.metadata as md
from pwem.protocols import ProtClassify2D
from pwem.objects import SetOfClasses2D
from pwem.constants import ALIGN_NONE, ALIGN_2D
from pyworkflow import BETA, UPDATED, NEW, PROD


from xmipp3.convert import (writeSetOfParticles, createItemMatrix,
                            writeSetOfClasses2D, xmippToLocation,
                            rowToAlignment)


# Comparison methods enum
CMP_CORRELATION = 0
CMP_CORRENTROPY = 1

# Clustering methods enum
CL_CLASSICAL = 0
CL_ROBUST = 1

# Classes keys
OUTPUTCLASSES = 'outputClasses'
CLASSES = ''
CLASSES_CORE = '_core'
CLASSES_STABLE_CORE = '_stable_core'

# Suggested number of images per class
IMAGES_PER_CLASS = 200


class XmippProtCL2D(ProtClassify2D):
    """ Classifies a set of 2D images using clustering algorithms. It
    subdivides the original dataset into user-defined classes, aiding the
    identification of particle heterogeneity or structural variations within
    the data.

    AI Generated:

    What this protocol is for

    CL2D is Xmipp’s classic 2D classification method for single-particle
    datasets. Its goal is to take a set of particle images and organize them
    into 2D classes so that particles with similar views cluster together. In
    practice, this is one of the most useful “dataset-understanding” steps in
    a workflow: it helps you see whether your sample is homogeneous, whether
    you have different conformations or compositions, whether there is
    preferred orientation, and how much junk/contamination is present.

    Compared to very fast “quick look” approaches, CL2D is designed to be a
    robust, iterative classifier that can build structure in the dataset
    gradually. A particularly helpful concept in this protocol is the idea of
    cores: subsets of particles that are most representative of each class (and
    optionally “stable cores”, which are particles that remain consistently
    grouped through the multilevel process). For biological users, cores are
    often the easiest way to obtain a cleaner subset without doing aggressive
    manual rejection.

    Inputs: what you need to provide

    You provide a single mandatory input: a SetOfParticles (your particle
    stack). These particles do not need to be aligned beforehand for CL2D to
    work as a classifier, but the quality of the result always depends on the
    usual prerequisites: reasonable particle extraction, sensible box size,
    and (often) some form of downsampling if the sampling rate is very small
    and you are only aiming at a medium-resolution 2D classification.

    Optionally, you may provide initial classes/averages if you prefer to start
    from known references rather than random initialization. This is useful
    when you want reproducibility, when you are “updating” a previously known
    set of views, or when you already trust a set of 2D averages and want the
    classification to converge around them.

    The most important knob: how many classes to ask for

    The parameter Number of classes controls the granularity of the
    classification. Biologically, this is where you decide whether you want a
    coarse view of the data (fewer classes, more particles per class, easier to
    interpret) or a finer partition (more classes, smaller classes, better at
    separating rare views or subtle heterogeneity but also easier to
    over-fragment noise).

    A practical way to think about it is the “typical number of images per
    class”. If you have N particles and choose K classes, the average class
    size is N/K. CL2D is often most interpretable when classes have enough
    particles to average out noise (hundreds is comfortable for many datasets),
    but this depends strongly on SNR and particle size.

    How initialization works, and when to use each mode

    CL2D can start in two different ways.

    If you enable random initialization, the algorithm will build the
    classification starting from a small number of initial classes (the
    “first level”), and then progressively refine/split them until the final
    number of classes is reached. This is the most common choice when you do
    not have a reliable set of starting references. The key advanced parameter
    here is Number of initial classes: conceptually, you begin with a small
    number of broad groups and then increase complexity in later levels. If you
    set this too low, the early grouping can be overly coarse; if you set it
    too high, you lose the benefit of gradual organization and can become more
    sensitive to noise.

    If you disable random initialization, you provide Initial classes (a set of
    2D classes or averages). In this mode, CL2D starts from those references
    and tends to behave more like “classify relative to these starting views”.
    This is often useful for controlled workflows (e.g., you already have a
    trusted set of views from a previous run or a known sample) or when you
    want more repeatable outcomes between runs.

    Iterations and convergence: what “Number of iterations” really means

    Within each level of the multilevel process, CL2D performs iterative
    refinement. The parameter Number of iterations sets an upper bound for how
    long it will refine at each level. Biologically, increasing iterations can
    help when particles are noisy or when the dataset has subtle differences
    that need more refinement to separate. On the other hand, very large
    iteration counts are not always beneficial: if the dataset contains a lot
    of junk or highly heterogeneous content, extra iterations may mostly refine
    noise patterns rather than producing cleaner biology.

    A common, practical approach is to keep a moderate value for routine runs
    and only increase it if you see unstable or underdeveloped class averages.

    How similarity is measured: correlation vs correntropy

    CL2D lets you choose a comparison method that defines how “close” two
    images are during clustering.

    With correlation, similarity is driven by standard cross-correlation–like
    behavior. This tends to work well for many cryo-EM particle sets and is
    usually a safe default.

    With correntropy, similarity becomes more robust to certain kinds of
    outliers and non-Gaussian noise. Biologically, this option can sometimes
    help when a dataset contains a mixture of good particles and difficult
    contaminants, or when you suspect that outliers are pulling classes in
    unhelpful directions. It is not a magic switch, but it is a reasonable
    alternative if correlation-based grouping produces classes that look
    “washed” or dominated by a few atypical images.

    How clustering is done: classical vs robust

    The clustering method controls the criterion used to build and update
    classes.

    The classical criterion is the standard behavior and is typically the
    first choice.

    The robust option is meant to be less sensitive to problematic particles.
    For biological users, a good rule of thumb is: if the dataset is clean and
    you mostly want to resolve views, classical is usually fine; if you have
    strong junk, variable backgrounds, or particles with systematic artifacts,
    robust clustering may give more stable, interpretable classes.

    Core analysis: extracting the “most representative” particles per class

    One of the most biologically useful features of this protocol is core
    analysis. When core analysis is enabled, CL2D analyzes each class and
    identifies a subset of particles that are close to the class center
    according to internal criteria (including Z-score–like measures and
    PCA-based distances). Intuitively, the core is the set of particles that
    best represent the class signal and are least likely to be junk or
    misassigned.

    The parameters Junk Zscore and PCA Zscore control how strict this selection
    is. Lower thresholds reject more particles (giving smaller, cleaner cores),
    while higher thresholds accept more (giving larger cores that may include
    more borderline particles). If your goal is to build a high-quality subset
    for downstream refinement, stricter cores are often attractive. If your
    goal is to avoid losing rare but real views, you may want to be less strict.

    CL2D can also compute a stable core, which is a stricter idea: particles
    that have remained essentially grouped throughout the multilevel
    classification. The Tolerance parameter controls how many “exceptions” are
    allowed across levels. With tolerance set to zero, stable core membership
    becomes very stringent; with higher tolerance, you keep more particles
    while still favoring stable assignments. For biological workflows, stable
    cores are often a good way to define “high-confidence” subsets for more
    delicate steps, like initial model generation or high-resolution refinement.

    Optional hierarchy: understanding how classes split across levels

    If you enable Compute class hierarchy, CL2D will track how classes relate
    between levels (how groups split and evolve). This is mostly a diagnostic
    feature, but it can be biologically informative: it helps you see whether a
    class is stable, whether it splits into meaningful sub-views, or whether it
    fragments in ways that look like noise.

    Optional analysis of rejected particles

    If you enable Analyze rejected particles, the protocol will generate
    additional information in the run directory showing what was excluded from
    cores (and stable cores). This can be useful when you want to understand
    the nature of the rejected set—whether it is mostly obvious junk, rare
    orientations, contaminants, or simply low-SNR particles.

    Outputs: what you get at the end

    The main output is outputClasses, a SetOfClasses2D with your final
    classification.

    If core analysis is enabled and applicable, you also get
    outputClasses_core, which contains the “core” particles for each class.

    If stable core analysis is enabled and applicable, you also get
    outputClasses_stable_core, which contains the most stable, high-confidence
    particles per class.

    From a biological processing standpoint, these outputs give you three
    practical processing paths: you can continue with the full classified
    dataset, or you can continue with a cleaner “core” subset, or you can
    choose the most conservative “stable core” subset when you want maximum
     reliability at the cost of throwing away more particles.

    Typical processing strategies (how users usually apply CL2D)

    A very common strategy is to run CL2D with a reasonable number of classes
    and then visually inspect the class averages to decide what to keep. In
    parallel, core/stable core outputs can be used as an automated way to
    define a cleaner subset, especially when you want to reduce bias in manual
    selection.

    Another common approach is iterative: run CL2D once to diagnose dataset
    quality and remove obvious junk, then rerun CL2D on the cleaner subset with
    more classes to resolve views and heterogeneity more finely.

    Finally, if you already have a reliable set of 2D averages (from previous
    runs or a known sample), starting from initial classes can make outcomes
    more reproducible and can help the classification converge to the expected
    view set more quickly."""
    
    _label = 'cl2d'
    _devStatus = PROD

    _possibleOutputs = {OUTPUTCLASSES: SetOfClasses2D,
                        OUTPUTCLASSES+CLASSES_CORE: SetOfClasses2D,
                        OUTPUTCLASSES+CLASSES_STABLE_CORE: SetOfClasses2D}
    
    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        if self.numberOfMpi.get() < 2:
            self.numberOfMpi.set(2)

    def _defineFileNames(self):
        """ Centralize how files are called within the protocol. """
        self.levelPath = self._getExtraPath('level_%(level)02d/')
        myDict = {
                  'input_particles': self._getTmpPath('input_particles.xmd'),
                  'input_references': self._getTmpPath('input_references.xmd'),
                  'final_classes': self._getPath('classes2D%(sub)s.sqlite'),
                  'output_particles': self._getExtraPath('images.xmd'),
                  'level_classes' : self.levelPath + 'level_classes%(sub)s.xmd',
                  'level_images' : self.levelPath + 'level_images%(sub)s.xmd',
                  'classes_scipion': (self.levelPath + 'classes_scipion_level_'
                                                   '%(level)02d%(sub)s.sqlite'),
                  'classes_hierarchy': self._getExtraPath("classes%(sub)s"
                                                          "_hierarchy.txt")
                  }
        self._updateFilenamesDict(myDict)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', param.PointerParam,
                      label="Input images",
                      important=True, pointerClass='SetOfParticles',
                      help='Select the input images to be classified.')
        form.addParam('numberOfClasses', param.IntParam, default=64,
                      label='Number of classes:',
                      help='Number of classes (or references) to be generated.')
        form.addParam('randomInitialization', param.BooleanParam, default=True,
                      expertLevel=const.LEVEL_ADVANCED,
                      label='Random initialization of classes:',
                      help="Initialize randomly the first classes. If you "
                           "don't initialize randomly, you must supply a set "
                           "of initial classes")
        form.addParam('initialClasses', param.PointerParam,
                      label="Initial classes",
                      condition="not randomInitialization",
                      pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Set of initial classes to start the classification')
        form.addParam('numberOfInitialClasses', param.IntParam, default=4,
                      expertLevel=const.LEVEL_ADVANCED,
                      label='Number of initial classes:',
                      condition="randomInitialization",
                      help='Initial number of classes used in the first level.')
        form.addParam('numberOfIterations', param.IntParam, default=10,
                      expertLevel=const.LEVEL_ADVANCED,
                      label='Number of iterations:',
                      help='Maximum number of iterations within each level.')
        form.addParam('comparisonMethod', param.EnumParam,
                      choices=['correlation', 'correntropy'],
                      label="Comparison method", default=CMP_CORRELATION,
                      expertLevel=const.LEVEL_ADVANCED,
                      display=param.EnumParam.DISPLAY_COMBO,
                      help='Use correlation or correntropy')
        form.addParam('clusteringMethod', param.EnumParam,
                      choices=['classical', 'robust'],
                      label="Clustering method", default=CL_CLASSICAL,
                      expertLevel=const.LEVEL_ADVANCED,
                      display=param.EnumParam.DISPLAY_COMBO,
                      help='Use the classical clustering criterion or the '
                           'robust')
        form.addParam('extraParams', param.StringParam,
                      expertLevel=const.LEVEL_ADVANCED,
                      label='Additional parameters',
                      help='Additional parameters for classify_CL2D: \n'
                           ' --verbose, --corrSplit, ...')

        form.addSection(label='Core analysis')
        form.addParam('doCore', param.BooleanParam, default=True,
                      label='Perform core analysis',
                      help='An image belongs to the core if it is close (see '
                           'Junk Zscore and PCA Zscore) to the class center')
        form.addParam('thZscore', param.FloatParam, default=3,
                      label='Junk Zscore', expertLevel=const.LEVEL_ADVANCED,
                      condition='doCore',
                      help='Which is the average Z-score to be considered as '
                           'junk. Typical values go from 1.5 to 3. For the '
                           'Gaussian distribution 99.5% of the data is '
                           'within a Z-score of 3. Lower Z-scores reject more '
                           'images. Higher Z-scores accept more images.')
        form.addParam('thPCAZscore', param.FloatParam, default=3,
                      condition='doCore', expertLevel=const.LEVEL_ADVANCED,
                      label='PCA Zscore',
                      help='Which is the PCA Z-score to be considered as junk. '
                           'Typical values go from 1.5 to 3. For the Gaussian '
                           'distribution 99.5% of the data is within a '
                           'Z-score of 3. Lower Z-scores reject more images. '
                           'Higher Z-scores accept more images.')
        form.addParam('doStableCore', param.BooleanParam, default=True,
                      condition='doCore', label='Perform stable core analysis',
                      help='Two images belong to the stable core if they have '
                           'been essentially together along the classification '
                           'process')
        form.addParam('tolerance', param.IntParam, default=1, label='Tolerance',
                      expertLevel=const.LEVEL_ADVANCED,
                      condition='doCore and doStableCore',
                      help='An image belongs to the stable core if it has been '
                           'with other images in the same class in all the '
                           'previous levels except possibly a few of them. '
                           'Tolerance defines how few is few. Tolerance=0 '
                           'means that an image must be in all previous levels '
                           'with the rest of images in the core.',)
        form.addParam("computeHierarchy", param.BooleanParam, default=False,
                      label="Compute class hierarchy",
                      expertLevel=const.LEVEL_ADVANCED)
        form.addParam("analyzeRejected", param.BooleanParam, default=False,
                      label="Analyze rejected particles",
                      expertLevel=const.LEVEL_ADVANCED,
                      help='To see the analysis you need to browse the '
                           'execution directory and go into the different '
                           'levels')
        form.addParallelSection(threads=0, mpi=4)

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        """ Mainly prepare the command line for call cl2d program"""

        # Convert input images if necessary
        self._defineFileNames()

        if self.initialClasses.get():
            initialClassesId = self.initialClasses.get().getObjId()
        else:
            initialClassesId = None
        
        self._insertFunctionStep(self.convertInputStep,
                                 self.inputParticles.get().getObjId(),
                                 initialClassesId)

        self._params = {'imgsFn': self._getFileName('input_particles'),
                        'extraDir': self._getExtraPath(),
                        'nref': self.numberOfClasses.get(),
                        'nref0': self.numberOfInitialClasses.get(),
                        'iter': self.numberOfIterations.get(),
                        'extraParams': self.extraParams.get(''),
                        'thZscore': self.thZscore.get(),
                        'thPCAZscore': self.thPCAZscore.get(),
                        'tolerance': self.tolerance.get(),
                        'initClassesFn': self._getFileName('input_references')
                        }
        
        args = self._defArgsClassify()

        self._insertClassifySteps("xmipp_classify_CL2D", args, subset=CLASSES)

        #TODO: Added this If. Check with COSS error if makes sense.
        #Also, if conditions below are enough to validate that classes core
        # and stable core are not empty

        if not self.randomInitialization:
            self.numberOfInitialClasses.set(self.initialClasses.get().getSize())

        # Analyze cores and stable cores
        if self.numberOfClasses > self.numberOfInitialClasses and self.doCore:
            program = "xmipp_classify_CL2D_core_analysis"
            # core analysis
            args = self._defArgsCoreAnalisys()
            self._insertClassifySteps(program, args, subset=CLASSES_CORE)
            if self.analyzeRejected:
                self._insertFunctionStep(self.analyzeOutOfCores, CLASSES_CORE)

            if (self.numberOfClasses > (2 * self.numberOfInitialClasses.get())
                and self.doStableCore): # Number of levels should be > 2
                
                # stable core analysis
                args = self._defArgsCoreAnalisys("stable")
                self._insertClassifySteps(program, args,
                                          subset=CLASSES_STABLE_CORE)
                if self.analyzeRejected:
                    self._insertFunctionStep(self.analyzeOutOfCores,
                                             CLASSES_STABLE_CORE)

    def _insertClassifySteps(self, program, args, subset=CLASSES):
        """ Defines four steps for the subset:
        1. Run the main program.
        2. Evaluate classes
        3. Sort the classes.
        4. And create output
        """
        self._insertRunJobStep(program, args % self._params)
        self._insertFunctionStep(self.evaluateClassesStep, subset)
        self._insertFunctionStep(self.sortClassesStep, subset)
        self._insertFunctionStep(self.createOutputStep, subset)
    
    #--------------------------- STEPS functions -------------------------------
    def convertInputStep(self, particlesId, classesId):
        writeSetOfParticles(self.inputParticles.get(),
                            self._getFileName('input_particles'),
                            alignType=ALIGN_NONE)
        
        if not self.randomInitialization:
            
            if isinstance(self.initialClasses.get(), SetOfClasses2D):
                writeSetOfClasses2D(self.initialClasses.get(),
                                    self._getFileName('input_references'),
                                    writeParticles=False)
            else:
                writeSetOfParticles(self.initialClasses.get(),
                                    self._getFileName('input_references'))

    def sortClassesStep(self, subset=''):
        """ Sort the classes and provided a quality criterion. """
        levelMdFiles = self._getAllLevelMdFiles(subset)
        for mdFn in levelMdFiles:
            fnRoot = join(dirname(mdFn), "classes%s_sorted" % subset)
            params = "-i classes@%s --oroot %s" % (mdFn, fnRoot)
            self.runJob("xmipp_image_sort", params)
            mdFnOut = fnRoot + ".xmd"
            mdOut = md.MetaData(mdFnOut)
            
            
            for objId in mdOut:
                mdOut.setValue(md.MDL_ITEM_ID,
                               int(mdOut.getValue(md.MDL_REF,objId)),objId)
            mdOut.write("classes_sorted@" + mdFn, md.MD_APPEND)

    def evaluateClassesStep(self, subset=''):
        """ Calculate the FRC and output the hierarchy for
        each level of classes.
        """
        levelMdFiles = self._getAllLevelMdFiles(subset)
        hierarchyFnOut = self._getExtraPath("classes%s_hierarchy.txt" % subset)
        prevMdFn = None
        for mdFn in levelMdFiles:
            self.runJob("xmipp_classify_evaluate_classes",
                        "-i " + mdFn, numberOfMpi=1)
            if self.computeHierarchy and prevMdFn is not None:
                args = "--i1 %s --i2 %s -o %s" % (prevMdFn, mdFn, hierarchyFnOut)
                if exists(hierarchyFnOut):
                    args += " --append"
                self.runJob("xmipp_classify_compare_classes",
                            args, numberOfMpi=1)
            prevMdFn = mdFn

    def createOutputStep(self, subset=''):
        """ Store the SetOfClasses2D object
        resulting from the protocol execution.
        """
       
        level = self._lastLevel()
        
        subsetFn = self._getFileName("level_classes", level=level, sub=subset)
        
        if exists(subsetFn):
            classes2DSet = self._createSetOfClasses2D(self.inputParticles, subset)
            self._fillClassesFromLevel(classes2DSet, "last", subset)
    
            result = {OUTPUTCLASSES + subset: classes2DSet}
            self._defineOutputs(**result)
            self._defineSourceRelation(self.inputParticles, classes2DSet)
    
    def analyzeOutOfCores(self,subset):
        """ Analyze which images are out of cores """
        levelMdFiles = self._getAllLevelMdFiles(subset)
        
        for fn in levelMdFiles:
            mdAll=md.MetaData()
            blocks = md.getBlocksInMetaDataFile(fn)
            fnDir=dirname(fn)
            # Gather all images in block
            for block in blocks:
                if block.startswith('class0'):
                    mdClass=md.MetaData(block+"@"+fn)
                    mdAll.unionAll(mdClass)
            if mdAll.size()>0:
                # Compute difference to images
                fnSubset=join(fnDir,"images%s.xmd"%subset)
                mdAll.write(fnSubset)
                fnOutOfSubset=join(fnDir,"imagesOut.xmd")
                
                inputMd = self._getFileName('input_particles')
                
                args = "-i %s --set subtraction %s -o %s" % (inputMd,
                                                            fnSubset,
                                                            fnOutOfSubset)
                
                self.runJob("xmipp_metadata_utilities", args, numberOfMpi=1,
                            numberOfThreads=1)
                
                # Remove disabled and intermediate files
                mdClass=md.MetaData(fnOutOfSubset)
                mdClass.removeDisabled()
                fnRejected="images_rejected@"+fn
                mdClass.write(fnRejected,md.MD_APPEND)
                cleanPath(fnOutOfSubset)
                cleanPath(fnSubset)
                
                # If enough images, make a small summary
                if mdClass.size()>100:
                    from math import ceil
                    fnRejectedDir=join(fnDir,"rejected%s"%subset)
                    makePath(fnRejectedDir)
                    Nclasses=int(ceil(mdClass.size()/300))
                    self.runJob("xmipp_classify_CL2D",
                                "-i %s --nref0 1 --nref %d --iter 5 --distance "
                                "correlation --classicalMultiref "
                                "--classifyAllImages --odir %s"
                                %( fnRejected, Nclasses, fnRejectedDir))

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        validateMsgs = []
        if self.numberOfMpi <= 1:
            validateMsgs.append('Mpi needs to be greater than 1.')
        if self.numberOfInitialClasses > self.numberOfClasses:
            validateMsgs.append('The number of final classes cannot be smaller'
                                ' than the number of initial classes')
        if isinstance(self.initialClasses.get(), SetOfClasses2D):
            if not self.initialClasses.get().hasRepresentatives():
                validateMsgs.append("The input classes should have "
                                    "representatives.")
        return validateMsgs
    
    def _warnings(self):
        validateMsgs = []
        if self.inputParticles.get().getSamplingRate() < 3:
            validateMsgs.append("The sampling rate is smaller than 3 A/pix, "
                                "consider downsampling the input images to "
                                "speed-up the process. Probably you don't want"
                                " such a precise 2D classification.")
        return validateMsgs

    def _citations(self):
        citations=['Sorzano2010a']
        if self.doCore:
            citations.append('Sorzano2014')
        return citations

    def _summaryLevelFiles(self, summary, levelFiles, subset):
        if levelFiles:
            levels = [i for i in range(self._lastLevel()+1)]
            summary.append('Computed classes%s, levels: %s' % (subset, levels))

    def _summary(self):
        self._defineFileNames()
        summary = []
        levelFiles = self._getAllLevelMdFiles()

        if not hasattr(self, 'outputClasses'):
            summary.append("Output classes not ready yet.")
        elif levelFiles:
            self._summaryLevelFiles(summary, levelFiles, CLASSES)
            self._summaryLevelFiles(summary, self._getAllLevelMdFiles(CLASSES_CORE), CLASSES_CORE)
            self._summaryLevelFiles(summary, self._getAllLevelMdFiles(CLASSES_STABLE_CORE), CLASSES_STABLE_CORE)
        else:
            summary.append("Input Particles: *%d*\nClassified into *%d* classes\n"
                           % (self.inputParticles.get().getSize(),
                              self.numberOfClasses.get()))
            # summary.append('- Used a _clustering_ algorithm to subdivide the original dataset into the given number of classes')
        return summary

    def _methods(self):
        strline = ''
        if hasattr(self, 'outputClasses'):
            strline += 'We classified %d particles from %s ' % (self.inputParticles.get().getSize(),
                                                                self.getObjectTag('inputParticles'))
            strline += 'into %d classes %s using CL2D [Sorzano2010a]. ' % (self.numberOfClasses,
                                                                           self.getObjectTag('outputClasses'))
            strline += '%s method was used to compare images and %s clustering criterion. '%\
                           (self.getEnumText('comparisonMethod'), self.getEnumText('clusteringMethod'))
            if self.numberOfClasses > self.numberOfInitialClasses and self.doCore:
                strline+='We also calculated the class cores %s' % self.getObjectTag('outputClasses_core')
                if self.numberOfClasses > (2 * self.numberOfInitialClasses.get()) and self.doStableCore: # Number of levels should be > 2
                    strline += ' and the class stable cores %s' % self.getObjectTag('outputClasses_stable_core')
                strline+=' [Sorzano2014].'
        return [strline]
    
    #--------------------------- UTILS functions -------------------------------
    def _defArgsClassify(self):
        # Prepare arguments to call program: xmipp_classify_CL2D
        args = '-i %(imgsFn)s --odir %(extraDir)s --oroot level --nref ' \
               '%(nref)d --iter %(iter)d %(extraParams)s'
        if self.comparisonMethod == CMP_CORRELATION:
            args += ' --distance correlation'
        if self.clusteringMethod == CL_CLASSICAL:
            args += ' --classicalMultiref'
        if self.randomInitialization:
            args += ' --nref0 %(nref0)d'
        else:
            args += ' --ref0 %(initClassesFn)s'
        
        return args
    
    def _defArgsCoreAnalisys(self,coreType="core"):
        args = " --dir %(extraDir)s --root level "
        if coreType =="core":
            args += "--computeCore %(thZscore)f %(thPCAZscore)f"
        else:
            args += "--computeStableCore %(tolerance)d"
        return args

    def _getAllLevelMdFiles(self, subset=''):
        """ Grab the metadata class files for each level. """
        levelMdFiles = []
        lastLevel = self._lastLevel()
        for i in range(lastLevel):
            classFn = self._getLevelMdClasses(lev=i, block="", subset=subset)
            if exists(classFn):
                levelMdFiles.append(classFn)
        return levelMdFiles
    
    def _createItemMatrix(self, item, row):
        createItemMatrix(item, row, align=ALIGN_2D)

    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(md.MDL_REF))
        item.setTransform(rowToAlignment(row, ALIGN_2D))

    def _updateClass(self, item):
        classId = item.getObjId()

        if classId in self._classesInfo:
            index, fn, _ = self._classesInfo[classId]
            item.setAlignment2D()
            rep = item.getRepresentative()
            rep.setLocation(index, fn)
            rep.setSamplingRate(self.inputParticles.get().getSamplingRate())

    def _loadClassesInfo(self, filename):
        """ Read some information about the produced 2D classes
        from the metadata file.
        """
        self._classesInfo = {}  # store classes info, indexed by class id

        mdClasses = md.MetaData(filename)

        for classNumber, row in enumerate(md.iterRows(mdClasses)):
            index, fn = xmippToLocation(row.getValue(md.MDL_IMAGE))
            # Store info indexed by id, we need to store the row.clone() since
            # the same reference is used for iteration
            self._classesInfo[classNumber + 1] = (index, fn, row.clone())

    def _fillClassesFromLevel(self, clsSet, level, subset):
        """ Create the SetOfClasses2D from a given iteration. """
        self._loadClassesInfo(self._getLevelMdClasses(lev=level, subset=subset))
        
        if subset == '' and level == "last":
            xmpMd = self._getFileName('output_particles')
            if not exists(xmpMd):
                xmpMd = self._getLevelMdImages(level, subset)
        else:
            xmpMd = self._getLevelMdImages(level, subset)
            
        iterator = md.SetMdIterator(xmpMd, sortByLabel=md.MDL_ITEM_ID,
                                    updateItemCallback=self._updateParticle,
                                    skipDisabled=True)
        
        # itemDataIterator is not neccesary because, the class SetMdIterator
        # contain all the information about the metadata
        clsSet.classifyItems(updateItemCallback=iterator.updateItem,
                             updateClassCallback=self._updateClass)

    def _getLevelMdClasses(self, lev=0, block="classes", subset=""):
        """ Return the classes metadata for this iteration.
        block parameter can be 'info' or 'classes'."""
        if lev == "last":
            lev = self._lastLevel()
        mdFile = self._getFileName('level_classes', level=lev, sub=subset)
        if block:
            mdFile = block + '@' + mdFile
    
        return mdFile

    def _getLevelMdImages(self, level, subset):
        if level == "last":
            level = self._lastLevel()
    
        xmpMd = self._getFileName('level_images', level=level, sub=subset)
        if not exists(xmpMd):
            self._createLevelMdImages(level, subset)
            
        return xmpMd
        
    def _createLevelMdImages(self, level, sub):
        if level == "last":
            level = self._lastLevel()
        mdClassesFn = self._getLevelMdClasses(lev=level, block="", subset=sub)
        mdImgs = md.joinBlocks(mdClassesFn, "class0")
        mdImgs.write(self._getFileName('level_images',level=level, sub=sub))
        
    def _lastLevel(self):
        """ Find the last Level number """
        clsFn = self._getFileName('level_classes', level=0, sub="")
        levelTemplate = clsFn.replace('level_00','level_??')
        lev = len(glob(levelTemplate)) - 1
        return lev

    def _getLevelClasses(self, lev, suffix, clean=False):
        """ Return a classes .sqlite file for this level.
        If the file doesn't exists, it will be created by
        converting from this level level_images.xmd file.
        """
        dataClasses = self._getFileName('classes_scipion', level=lev,
                                         sub=suffix)
        if clean:
            cleanPath(dataClasses)
        
        if not exists(dataClasses):
            clsSet = SetOfClasses2D(filename=dataClasses)
            clsSet.setImages(self.inputParticles.get())
            self._fillClassesFromLevel(clsSet, level=lev, subset=suffix)
            clsSet.write()
            clsSet.close()
        
        return dataClasses
