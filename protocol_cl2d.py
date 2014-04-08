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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
This sub-package contains wrapper around CL2D Xmipp program
"""

from os.path import join, dirname, exists
from pyworkflow.em import *  
import xmipp

from convert import createXmippInputImages, readSetOfClasses2D
#from xmipp3 import XmippProtocol
from glob import glob

# Comparison methods enum
CMP_CORRELATION = 0
CMP_CORRENTROPY = 1

# Clustering methods enum
CL_CLASSICAL = 0
CL_ROBUST = 1

        
        
class XmippProtCL2D(ProtClassify2D):
    """ Classifies a set of images using a clustering algorithm 
            aiming at subdividing the original dataset
             into a given number of subclasses """
    
    _label = 'cl2d'
    
    def __init__(self, **args):
        Protocol.__init__(self, **args) 
        if self.numberOfMpi.get() < 2:
            self.numberOfMpi.set(2)       

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputImages', PointerParam, label="Input images", important=True, 
                      pointerClass='SetOfParticles',
                      help='Select the input images from the project.'
                           'It should be a SetOfImages class')        
        form.addParam('numberOfReferences', IntParam, default=64,
                      label='Number of references:',
                      help='Number of references (or classes) to be generated.')
        form.addParam('numberOfInitialReferences', IntParam, default=4, expertLevel=LEVEL_ADVANCED,
                      label='Number of initial references:',
                      help='Initial number of references used in the first level.')
        form.addParam('numberOfIterations', IntParam, default=4, expertLevel=LEVEL_ADVANCED,
                      label='Number of iterations:',
                      help='Maximum number of iterations within each level.')         
        form.addParam('comparisonMethod', EnumParam, choices=['correlation', 'correntropy'],
                      label="Comparison method", default=CMP_CORRELATION,
                      display=EnumParam.DISPLAY_COMBO,
                      help='Use correlation or correntropy')
        form.addParam('clusteringMethod', EnumParam, choices=['classical', 'robust'],
                      label="Clustering method", default=CL_CLASSICAL,
                      display=EnumParam.DISPLAY_COMBO,
                      help='Use the classical clustering criterion or the robust')
        form.addParam('extraParams', StringParam, expertLevel=LEVEL_EXPERT,
              label='Additional parameters',
              help='Additional parameters for classify_CL2D: \n  --verbose, --corrSplit, ...')   
        
        form.addSection(label='Core analysis')        
        form.addParam('thZscore', FloatParam, default=3,
                      label='Junk Zscore',
                      help='Which is the average Z-score to be considered as junk. Typical values'
                           'go from 1.5 to 3. For the Gaussian distribution 99.5% of the data is'
                           'within a Z-score of 3. Lower Z-scores reject more images. Higher Z-scores'
                           'accept more images.')
        form.addParam('thPCAZscore', FloatParam, default=3,
                      label='PCA Zscore',
                      help='Which is the PCA Z-score to be considered as junk. Typical values'
                           'go from 1.5 to 3. For the Gaussian distribution 99.5% of the data is'
                           'within a Z-score of 3. Lower Z-scores reject more images. Higher Z-scores'
                           'accept more images.')        
        form.addParam('tolerance', IntParam, default=1,
                      label='Tolerance',
                      help='An image belongs to the stable core if it has been with other images in the same class'
                           'in all the previous levels except possibly a few of them. Tolerance defines how few is few.'
                           'Tolerance=0 means that an image must be in all previous levels with the rest of images in'
                           'the core.')          
        
        form.addParallelSection(threads=0, mpi=4)
        
    #--------------------------- INSERT steps functions --------------------------------------------                
    def _insertAllSteps(self):
        """ Mainly prepare the command line for call cl2d program"""
        
        # Convert input images if necessary
        imgsFn = createXmippInputImages(self, self.inputImages.get())
        
        # Prepare arguments to call program: xmipp_classify_CL2D
        self._params = {'imgsFn': imgsFn, 
                        'extraDir': self._getExtraPath(),
                        'nref': self.numberOfReferences.get(), 
                        'nref0': self.numberOfInitialReferences.get(),
                        'iter': self.numberOfIterations.get(),
                        'extraParams': self.extraParams.get(''),
                        'thZscore': self.thZscore.get(),
                        'thPCAZscore': self.thPCAZscore.get(),
                        'tolerance': self.tolerance.get()
                      }
        args = '-i %(imgsFn)s --odir %(extraDir)s --oroot level --nref %(nref)d --iter %(iter)d %(extraParams)s'
        if self.comparisonMethod == CMP_CORRELATION:
            args += ' --distance correlation'
        if self.clusteringMethod == CL_CLASSICAL:
            args += ' --classicalMultiref'
        if not self.extraParams.hasValue() or not '--ref0' in self.extraParams.get():
            args += ' --nref0 %(nref0)d'
    
        self._insertClassifySteps("xmipp_classify_CL2D", args)
        
        # Analyze cores and stable cores
        if self.numberOfReferences > self.numberOfInitialReferences:
            program = "xmipp_classify_CL2D_core_analysis"
            args = "--dir %(extraDir)s --root level "
            # core analysis
            self._insertClassifySteps(program, args + "--computeCore %(thZscore)f %(thPCAZscore)f", subset='_core')
            if self.numberOfReferences > (2 * self.numberOfInitialReferences.get()): # Number of levels should be > 2
                # stable core analysis
                self._insertClassifySteps(program, args + "--computeStableCore %(tolerance)d", subset='_stable_core')
            
    def _insertClassifySteps(self, program, args, subset=''):
        """ Defines four steps for the subset:
        1. Run the main program.
        2. Evaluate classes
        3. Sort the classes.
        4. And create output
        """
        self._insertRunJobStep(program, args % self._params)
        self._insertFunctionStep('evaluateClassesStep', subset)
        self._insertFunctionStep('sortClassesStep', subset)
        self._insertFunctionStep('createOutputStep', subset)        
    
    #--------------------------- STEPS functions --------------------------------------------   
    def sortClassesStep(self, subset=''):
        """ Sort the classes and provided a quality criterion. """
        nproc = self.numberOfMpi.get()
        if nproc < 2:
            nproc = 2 # Force at leat two processor because only MPI version is available
        levelMdFiles = self._getLevelMdFiles(subset)
        for mdFn in levelMdFiles:
            fnRoot = join(dirname(mdFn), "classes%s_sorted" % subset)
            params = "-i classes@%s --oroot %s" % (mdFn, fnRoot)
            self.runJob("xmipp_image_sort", params, numberOfMpi=nproc)
            mdFnOut = fnRoot + ".xmd"
            md = xmipp.MetaData(mdFnOut)
            md.addItemId()
            md.write("classes_sorted@" + mdFn, xmipp.MD_APPEND)
            #deleteFile(log,fnRoot+".xmd")
        
    def evaluateClassesStep(self, subset=''):
        """ Calculate the FRC and output the hierarchy for 
        each level of classes.
        """
        levelMdFiles = self._getLevelMdFiles(subset)
        hierarchyFnOut = self._getExtraPath("classes%s_hierarchy.txt" % subset)
        prevMdFn = None
        for mdFn in levelMdFiles:
            self.runJob("xmipp_classify_evaluate_classes", "-i " + mdFn, numberOfMpi=1)
            if prevMdFn is not None:
                args = "--i1 %s --i2 %s -o %s" % (prevMdFn, mdFn, hierarchyFnOut)
                if exists(hierarchyFnOut):
                    args += " --append"
                self.runJob("xmipp_classify_compare_classes",args, numberOfMpi=1)
            prevMdFn = mdFn
    
    def createOutputStep(self, subset=''):
        """ Store the SetOfClasses2D object  
        resulting from the protocol execution. 
        """
        levelMdFiles = self._getLevelMdFiles(subset)
        lastMdFn = levelMdFiles[-1]
        inputImages = self.inputImages.get()
        classes2DSet = self._createSetOfClasses2D(inputImages, subset)
        readSetOfClasses2D(classes2DSet, lastMdFn, 'classes_sorted')
        result = {'outputClasses' + subset: classes2DSet}
        self._defineOutputs(**result)
        self._defineSourceRelation(inputImages, classes2DSet)
    
    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        validateMsgs = []
        # Some prepocessing option need to be marked
        if self.numberOfMpi <= 1:
            validateMsgs.append('Mpi needs to be greater than 1.')
        return validateMsgs
    
    def _citations(self):
        return ['Sorzano2010a']
    
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputClasses'):
            summary.append("Output classes not ready yet.")
        else:
            summary.append("Input Images: %s" % self.inputImages.get().getName())
            summary.append("Number of references: %d" % self.numberOfReferences.get())
            summary.append("Output classes: %s" % self.outputClasses.getName())
            if self.hasAttribute('outputClasses_core'):
                summary.append("Output classes core: %s" % self.outputClasses_core.getName())
            if self.hasAttribute('outputClasses_stable_core'):
                summary.append("Output classes stable core: %s" % self.outputClasses_stable_core.getName())
        return summary
    
    def _methods(self):
        methods = []
        if hasattr(self, 'outputClasses'):
            methods.append("The SetOfParticles with *%d* particles" % self.inputImages.get().getSize())
            methods.append("has been classified in *%d* classes." % self.numberOfReferences.get())
            methods.append("The method used to compare was *%s*" % self.getEnumText('comparisonMethod'))
            methods.append("and to clustering was *%s*" % self.getEnumText('clusteringMethod'))
        return methods
    
    #--------------------------- UTILS functions --------------------------------------------
    def _getLevelMdFiles(self, subset=''):
        """ Grab the metadata class files for each level. """
        levelMdFiles = glob(self._getExtraPath("level_??/level_classes%s.xmd" % subset))
        levelMdFiles.sort()
        return levelMdFiles        
    
    
    