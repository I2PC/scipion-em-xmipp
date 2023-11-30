# ******************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
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
import os

# import emtable

# from emtable import Table
from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, EnumParam, IntParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils.path import cleanPath, makePath

import pwem.emlib.metadata as md
from pwem.protocols import ProtClassify2D
from pwem.objects import SetOfClasses2D
from pwem.constants import ALIGN_NONE, ALIGN_2D
import xmipp3


from xmipp3.convert import (writeSetOfParticles, createItemMatrix,
                            writeSetOfClasses2D, xmippToLocation,
                            rowToAlignment)




def updateEnviron(gpuNum):
    """ Create the needed environment for pytorch programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)
        
        
class XmippProtClassifyPca(ProtClassify2D, xmipp3.XmippProtocol):
    """ Classifies a set of images. """
    
    _label = '2D classification pca'
    _lastUpdateVersion = VERSION_2_0
    _conda_env = 'xmipp_pyTorch'
    
        #Mode 
    CREATE_CLASSES = 0
    UPDATE_CLASSES = 1
    
    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        # if self.numberOfMpi.get() < 2:
        #     self.numberOfMpi.set(2)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
    
        form.addHidden(GPU_LIST, StringParam, default='0',
               label="Choose GPU ID",
               help="GPU may have several cores. Set it to zero"
                    " if you do not know what we are talking about."
                    " First core index is 0, second 1 and so on.")
    
    
        form.addSection(label='Input')
    
        form.addParam('inputParticles', PointerParam,
                      label="Input images",
                      important=True, pointerClass='SetOfParticles',
                      help='Select the input images to be classified.')
        form.addParam('numberOfClasses', IntParam, default=50,
                      label='Number of classes:',
                      help='Number of classes (or references) to be generated.')
        form.addParam('mode', EnumParam, choices=['create_classes', 'update_classes'],
                      label="Create or update 2D classes?", default=self.CREATE_CLASSES,
                      display=EnumParam.DISPLAY_HLIST, 
                      help='This option allows for either global refinement from an initial volume '
                    ' or just alignment of particles. If the reference volume is at a high resolution, '
                    ' it is advisable to only align the particles and reconstruct at the end of the iterative process.') 
        form.addParam('initialClasses', PointerParam,
                      label="Initial classes",
                      condition="mode",
                      pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Set of initial classes to start the classification')
        form.addParam('correctCtf', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
              label='Correct CTF?',
              help='If you set to *Yes*, the CTF of the experimental particles will be corrected')
        form.addParam('mask', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
              label='Use Gaussian Mask?',
              help='If you set to *Yes*, a gaussian mask is applied to the images.')
        form.addParam('sigma', IntParam, default=-1, expertLevel=LEVEL_ADVANCED,
              label='sigma:', condition="mask",
              help='Sigma is the parameter that controls the dispersion or "width" of the curve..')
    
        form.addSection(label='Pca training')
    
        form.addParam('resolution',FloatParam, label="max resolution", default=8,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef' ,FloatParam, label="% variance", default=0.5, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of coefficients to be considers (between 0-1).'
                      ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('training',IntParam, default=40000,
                      label="particles for training",
                      help='Number of particles for PCA training')
    
    
        form.addParallelSection(threads=1, mpi=4)

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):

        """ Mainly prepare the command line for call classification program"""
    
        updateEnviron(self.gpuList.get())
        
        self.imgsOrigXmd = self._getExtraPath('images_original.xmd')
        self.imgsXmd = self._getTmpPath('images.xmd')
        self.imgsFn = self._getTmpPath('images.mrc')
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getTmpPath('references.mrcs')
        self.sampling = self.inputParticles.get().getSamplingRate()
        self.acquisition = self.inputParticles.get().getAcquisition()
        mask = self.mask.get()
        sigma = self.sigma.get()
        if sigma == -1:
            sigma = self.inputParticles.get().getDimensions()[0]/3
        self.numTrain = min(self.training.get(), self.inputParticles.get().getSize())

    
        self._insertFunctionStep('convertInputStep', 
                                # self.inputParticles.get(), self.imgsOrigXmd, self.imgsXmd) #wiener oier
                                self.inputParticles.get(), self.imgsOrigXmd, self.imgsFn) #convert
        
        self._insertFunctionStep("pcaTraining", self.imgsFn, self.resolution.get(), self.training.get())
        
        self._insertFunctionStep("classification", self.imgsFn, self.numberOfClasses.get(), self.imgsOrigXmd, mask, sigma )
    
        self._insertFunctionStep('createOutputStep')
        

    #--------------------------- STEPS functions -------------------------------
    def convertInputStep(self, input, outputOrig, outputMRC):
        writeSetOfParticles(input, outputOrig)
        
        if self.mode == self.UPDATE_CLASSES: 
            
            if isinstance(self.initialClasses.get(), SetOfClasses2D):
                writeSetOfClasses2D(self.initialClasses.get(),
                                    self.refXmd, writeParticles=False)
            else:
                writeSetOfParticles(self.initialClasses.get(),
                                    self.refXmd)
                       
            args = ' -i  %s -o %s  '%(self.refXmd, self.ref)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)
            
        
        if self.correctCtf: 
            args = ' -i  %s -o %s --sampling_rate %s '%(outputOrig, outputMRC, self.sampling)
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get())
            
                    #WIENER Oier
            # args = ' -i %s  -o %s --pixel_size %s --spherical_aberration %s --voltage %s --batch 1024 --device cuda:0'% \
            #         (outputOrig, outputMRC, self.sampling, self.acquisition.getSphericalAberration(), self.acquisition.getVoltage())
            
            # env = self.getCondaEnv()
            # env['LD_LIBRARY_PATH'] = ''
            # self.runJob("xmipp_swiftalign_wiener_2d", args, numberOfMpi=1, env=env)
        else:      
            args = ' -i  %s -o %s  '%(outputOrig, outputMRC)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1) 
        
        
    def pcaTraining(self, inputIm, resolutionTrain, numTrain):
        args = ' -i %s  -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca  --batchPCA'% \
                (inputIm, self.sampling, resolutionTrain, self.coef.get(), numTrain, self._getExtraPath())

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca_train", args, numberOfMpi=1, env=env)
        
        
    def classification(self, inputIm, numClass, stfile, mask, sigma):
        args = ' -i %s -s %s -c %s -n 15 -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt -o %s/classes -stExp %s' % \
                (inputIm, self.sampling, numClass, self._getExtraPath(), self._getExtraPath(),  self._getExtraPath(),
                 stfile)
        if mask:
            args += ' --mask --sigma %s '%(sigma) 
            
        if self.mode == self.UPDATE_CLASSES:
            args += ' -r %s '%(self.ref)

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca", args, numberOfMpi=1, env=env)
        
        
    def createOutputStep(self):
        """ Store the SetOfClasses2D object
        resulting from the protocol execution.
        """
    
        inputParticles = self.inputParticles#.get()

        classes2DSet = self._createSetOfClasses2D(inputParticles)
        self._fillClassesFromLevel(classes2DSet)

        self._defineOutputs(outputClasses=classes2DSet)
        self._defineSourceRelation(self.inputParticles, classes2DSet)
    
    #--------------------------- INFO functions --------------------------------
    
    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are not errors. If some errors are found, a list with
        the error messages will be returned.
        """
        
        errors = []
        if self.inputParticles.get().getDimensions()[0] > 256:
            errors.append("You should resize the particles."
                                " Sizes smaller than 128 pixels are recommended.")
        er = self.validateDLtoolkit()
        if er:
            errors.append(er)
        return errors


    def _warnings(self):
        validateMsgs = []
        if  self.inputParticles.get().getDimensions()[0] > 128:
            validateMsgs.append("Particle sizes equal to or less"
                                " than 128 pixels are recommended.")
        elif self.inputParticles.get().getDimensions()[0] > 256:
            validateMsgs.append("Particle sizes equal to or less"
                                " than 128 pixels are recommended.")
        return validateMsgs
    #
    # def _citations(self):
    #     citations=['Sorzano2010a']
    #     if self.doCore:
    #         citations.append('Sorzano2014')
    #     return citations
    #
    # def _summaryLevelFiles(self, summary, levelFiles, subset):
    #     if levelFiles:
    #         levels = [i for i in range(self._lastLevel()+1)]
    #         summary.append('Computed classes%s, levels: %s' % (subset, levels))
    #
    # def _summary(self):
    #     self._defineFileNames()
    #     summary = []
    #     levelFiles = self._getAllLevelMdFiles()
    #
    #     if not hasattr(self, 'outputClasses'):
    #         summary.append("Output classes not ready yet.")
    #     elif levelFiles:
    #         self._summaryLevelFiles(summary, levelFiles, CLASSES)
    #         self._summaryLevelFiles(summary, self._getAllLevelMdFiles(CLASSES_CORE), CLASSES_CORE)
    #         self._summaryLevelFiles(summary, self._getAllLevelMdFiles(CLASSES_STABLE_CORE), CLASSES_STABLE_CORE)
    #     else:
    #         summary.append("Input Particles: *%d*\nClassified into *%d* classes\n"
    #                        % (self.inputParticles.get().getSize(),
    #                           self.numberOfClasses.get()))
    #         # summary.append('- Used a _clustering_ algorithm to subdivide the original dataset into the given number of classes')
    #     return summary
    #
    # def _methods(self):
    #     strline = ''
    #     if hasattr(self, 'outputClasses'):
    #         strline += 'We classified %d particles from %s ' % (self.inputParticles.get().getSize(),
    #                                                             self.getObjectTag('inputParticles'))
    #         strline += 'into %d classes %s using CL2D [Sorzano2010a]. ' % (self.numberOfClasses,
    #                                                                        self.getObjectTag('outputClasses'))
    #         strline += '%s method was used to compare images and %s clustering criterion. '%\
    #                        (self.getEnumText('comparisonMethod'), self.getEnumText('clusteringMethod'))
    #         if self.numberOfClasses > self.numberOfInitialClasses and self.doCore:
    #             strline+='We also calculated the class cores %s' % self.getObjectTag('outputClasses_core')
    #             if self.numberOfClasses > (2 * self.numberOfInitialClasses.get()) and self.doStableCore: # Number of levels should be > 2
    #                 strline += ' and the class stable cores %s' % self.getObjectTag('outputClasses_stable_core')
    #             strline+=' [Sorzano2014].'
    #     return [strline]
    #
    # #--------------------------- UTILS functions -------------------------------
        
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
    
    def _fillClassesFromLevel(self, clsSet):
        """ Create the SetOfClasses2D from a given iteration. """
        self._loadClassesInfo(self._getExtraPath('classes_classes.star'))
    
        xmpMd = self._getExtraPath('classes_images.star')
    
        iterator = md.SetMdIterator(xmpMd, sortByLabel=md.MDL_ITEM_ID,
                                    updateItemCallback=self._updateParticle,
                                    skipDisabled=True)
    
        clsSet.classifyItems(updateItemCallback=iterator.updateItem,
                             updateClassCallback=self._updateClass)



