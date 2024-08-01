# **************************************************************************
# *
# * Author:    Laura Baena Márquez
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************

#TODO: Check imports
#TODO: check whether " or ' will be used

#--------------- ATTENTION ----------------

#TODO: IMPORTANT!!!!!!! KEEP IN MIND Noise 0, Particle 1

#--------------- ATTENTION ----------------

import os

from pyworkflow.protocol.params import (Form, PointerParam, FloatParam, EnumParam, FileParam, IntParam, BooleanParam)
from pwem.objects import (SetOfParticles)
from pyworkflow.object import Float, Integer, Boolean
from ..convert import writeSetOfParticles
from pwem import emlib
from pwem.emlib.image import ImageHandler
from math import floor

import xmippLib


#TODO: check what should go in the ()
class XmippProtWrongAssignCheckTrain():

    #TODO: Protocol explanation comment

    # Identification parameters
    _label      = 'deep wrong angle assignment check'
    _devStatus  = BETA
    _conda_env = 'xmipp_DLTK_v1.0'
    #TODO: what is this? (below)
    #_stepsCheckSecs  = 60 # Scipion steps check interval (in seconds)
    #_possibleOutputs = {'output3DCoordinates' : SetOfCoordinates3D}

    #TODO: Evaluate changing name to global variable
    FORM_NN_USAGE_LABELS   = ["Model from scratch", "Use existing model"]
    #TODO: Explain what this is
    SHIFTVALUE = 2

    #TODO: Evaluate changing name to text so it's named coherently
    REF_CORRECT = "okSubset_"
    REF_INCORRECT = "wrongSubset_"

    #TODO: Check if this is correct
    def _init_(self, **kwargs):
        super()._init_(**kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL

    #--------------- DEFINE param functions ---------------
    def _defineParams(self, form: Form):

        #TODO: investigate multipointer param vs pointer param
        #TODO: is it possible to have optional parameters? using default?
        #TODO: investigate addHidden

        form.addSection(label = 'Main')
        form.addParam('assignment', PointerParam, 
                      label = "Input assignment", 
                      important = True, 
                      pointerClass = 'SetOfParticles')
        '''
        form.addParam('assignmentB', PointerParam, 
                      label = "Input assignment 2", 
                      important = True, 
                      pointerClass = 'SetOfParticles')
        '''
        '''#If the user is expected to prepare the input prior to using the protocol this args are not necessary
        form.addParam('angleComparison', PointerParam, 
                      label = "Angle comparison", 
                      important = True, 
                      pointerClass = 'SetOfParticles', 
                      help = 'Angle comparison between the two previous angle assignments')
        
        form.addParam('angleThreshold', FloatParam, 
                      label = "Angle Threshold", 
                      help = 'Value to discriminate angle comparisons between acceptable and not acceptable (too much variation between models)')
        '''
        #TODO: Review help text, copied from original
        #TODO: Not finished
        form.addParam('inputVolume',PointerParam,
                label = "Volume to compare images to",
                important = True,
                pointerClass = 'Volume',
                help = 'Volume to be used for class comparison')

        #TODO: eliminate section
        form.addSection(label = 'NN')
        form.addParam('trainingModel', EnumParam, 
                      label = "Select NN usage", 
                      important = True, 
                      choices = self.FORM_NN_USAGE_LABELS, 
                      default = 0, 
                      help = 'When set to *%s*, a new model will be created from scratch. The option *%s* allows using a pre-existing NN model.' % tuple(self.FORM_NN_USAGE_LABELS))
        form.addParam('modelToUse', FileParam, 
                      label = "Select pretrained model", 
                      condition = 'trainingModel == 1', 
                      help = 'Select the H5 format filename of the trained neural network. Note that the architecture must have been compiled using this same protocol.')
        ''' # There is no longer an option to not train in this protocol
        form.addParam('doTrain', BooleanParam, 
                      label = "Perform training", 
                      condition = 'trainingModel == %s' %self.MODEL_PRETRAIN,
                      default = False, 
                      help = 'When set to "yes" the existing model will also be used for training (not overwritten). If set to "No" the existing model will only be used for inference.')
        '''
        form.addParam('batchSz', IntParam, 
                      label = "Batch size", 
                      expertLevel = LEVEL_ADVANCED, 
                      default = 0, #TODO: maybe no batches should not be the default
                      help = 'Must be an integer. If batch size is bigger than the dataset size only one batch will be used. Also, if batchSize left to 0 only one batch will be used.')
        form.addParam('ensembleNum', IntParam, 
                      label = "Introduce ...", 
                      expertLevel = LEVEL_ADVANCED, 
                      default = 1, 
                      help = '')
        form.addParam('numEpoch', IntParam, 
                      label = "Number of epochs", 
                      expertLevel = LEVEL_ADVANCED, 
                      default = '', 
                      help = '')

        #TODO: Finish Ensemble param when available in program
        #TODO: Decide numEpoch default + include help explanation (Maybe I want this to not be a param and optimize every time?)
        #TODO: add GPU and Threads (form.AddParallelSection?)
    
    #--------------- INSERT steps functions ----------------
    #TODO: Write function defintion
    def _insertAllSteps(self):
        
        self.readInputs()
        self.insertFunctionStep(self.generateSubsetsStep)
        self.insertFunctionStep(self.callProduceResidualsStep)
        self.insertFunctionStep(self.prepocessTrainingInputStep)
        self.insertFunctionStep(self.callTrainingProgramStep)

    #--------------- STEPS functions ----------------

    #TODO: write function definition
    def readInputs(self):

        self.inputAssign : SetOfParticles = self.assignment.get()
        self.modelType = Integer(self.trainingModel.get())

        #TODO: include previous condition for model saving (if no pretrained model is used there is no point in this)
        #TODO: should I cast this into a string?
        self.nnModel = self.modelToUse.get()
        self.doTraining = Boolean(self.doTrain.get())
        self.batchSize = Integer(self.batchSz.get())
        #self.numEnsemble = Integer(self.ensembleNum.get())
        self.numEpochs = Integer(self.numEpoch.get())

        self.inputVol = self.inputVolume.get()

    #TODO: write function definition
    def generateSubsetsStep(self):

        self.okSubset = []
        self.wrongSubset = []
        aux = []

        for particle in self.inputAssign:
            
            self.okSubset.append([particle.getNameId(), particle.getFileName(), particle.getMatrix()])
            self.wrongSubset.append([particle.getNameId(), particle.getFileName()])
            aux.append(particle.getMatrix)

            aux = aux[self.SHIFTVALUE:] + aux[:self.SHIFTVALUE]

        for ptcl, value in zip(self.wrongSubset, aux):

            ptcl.append(value)
        
    #TODO: write function definition
    def callProduceResidualsStep(self):

        fnVol = self._getTmpPath("volume.vol")

        self.produceResiduals(self.okSubset, fnVol, self.REF_CORRECT)
        self.produceResiduals(self.wrongSubset, fnVol, self.REF_INCORRECT)

    #TODO: Evaluate produceResiduals for inference once cleaned
    #TODO: write function definition    
    def produceResiduals(self, imgSet, fnVol, ref):

        #TODO: properly comment this external code (and inform it is external)

        #TODO: Why do I call this residuals if the information here is not residuals (it's only the original info in Xmipp metadata file)
        self.imgsFn = self._getExtraPath(ref + 'residuals.xmd')
        #vol = self.inputVolume.get()
        #imgSet = self.inputSet.get()

        ## This function will write a SetOfParticles as a Xmipp metadata
        writeSetOfParticles(imgSet, self.imgsFn)

        img = ImageHandler()
        #fnVol = self._getTmpPath("volume.vol")
        img.convert(self.inputVolume.get(), fnVol)
        xdim = self.inputVolume.get().getDim()[0]

        #TODO: check the numberOfMpi=1 since this might not be a good practice
        if xdim != imgSet.getDim()[0]:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" % (fnVol, imgSet.getDim()[0]), numberOfMpi=1)

        #fnVol = self._getTmpPath("volume.vol")
        anglesOutFn = self._getExtraPath(ref + "anglesCont.stk")
        projectionsOutFn = self._getExtraPath(ref + "projections.stk")
        self.fnResiduals = self._getExtraPath(ref + "residuals.mrcs")

        program = "xmipp_angular_continuous_assign2"

        args = " -i " + imgSet ## Set of particles with initial alignment
        args += " -o " + anglesOutFn ## Output of the program (Stack of images prepared for 3D reconstruction)
        args += " --ref " + fnVol ## Reference volume
        args += " --optimizeAngles -- optimizeShift" 
        args += " --max_shift" + floor(xdim*0.05) ## Maximum shift allowed in pixels
        args += " --oprojections " + projectionsOutFn ## Output stack of projections
        args += " --sampling " + self.inputVol.getSamplingRate() ## Sampling rate (A/pixel)
        args += " --oresiduals " + self.fnResiduals ## Output stack of residuals
        args += " --optimizeGray --max_gray_scale 0.95 " ## Set values for gray optimization

        self.runJob(program,args)

    #TODO: write function definition
    def extractResiduals(self):

        #TODO: read residuals.mrcs (or residuals.xmd) to extract images fns 
        #TODO: return them in whichever format fits best

        pass
    
    #TODO: where do I include the residuals filenames? Separated file or extra column? 
    #       Substituting the original filename is not an option right? (No, is not)

    #TODO: move to utils since it is not an Step?
    #TODO: write function definition
    def saveInXMD(self, pathToFile):

        #Found uses to adapt to:
            # (deprecated) 1. Generate XMD files for residuals generation (with original images)
            # 2. Generate XMD files for training (with residual images)

        pass

    #TODO: write function definition
    def prepocessTrainingInputStep(self):

        #TODO: Call extract residuals for every subset and join them with new fnImgs
        #TODO: Save every subset in a different XMD and save route (self?)
        #TODO: create output file 

        pass


    #TODO: write funcion definition
    #TODO: evaluate if -> None in function would be useful
    #TODO: remember including ensemble args when implemented
    def callTrainingProgramStep(self):

        #TODO: check if this name should be included elsewhere
        program = "xmipp_deep_wrong_assign_check_tr"

        args = ' -c ' + str(self.) # file containing correct examples
        args += ' -w ' + str(self.) # file containing incorrect examples
        #TODO: when do I create this file?
        args += ' -o ' + str(self.) # file where the final model will be stored
        args += ' -b ' + self.batchSize

        #TODO: evaluate if only one argument could be used instead of a boolean and a fn separately
        #TODO: if pretrained:
        args += ' --pretrained ' 
        # Model could be universal or specific from user
        args += ' -f ' + str (self.) # file where the pretrained model is stored

        args += ' -e ' + self. numEpochs
        #TODO: evaluate including learning rate args += ' -l ' + self.
        #TODO: evaluate including patience args += ' -p ' + self.

        self.runJob(program,args)


    #TODO: Should I do anything else here before the protocol finishes?
    

    #--------------- INFO functions -------------------------
    def _summary(self):
        summary = []
        #TODO: appends to summary with the relevant info

    # ------------- UTILS functions -------------------------


    '''  # This decition has been delegated to the user kept just in case while developing
    #TODO: write function definition
    def divideSubsetsStep(self):

        self.okSubset = []
        self.doubtSubset = []

        for ptcA, ptcB, angPtc in zip(self.assignA.iterItems(), self.assignB.iterItems(), self.compAvsB.iterItems()):
            
            #TODO: [TODO: deprecate] choose final criteria to decide which matrix will be appended (CHECK)
            #TODO: can I force that the output always comes from a specific module (see particle picker)
            # Note that this function is adapted for a Xmipp3-compare-angles-output-like input from Scipion
            if Float(angPtc.getAttributeValue(attrName = "_xmipp_angleDiff")) <= self.threshold:
                self.okSubset.append([ptcA.getNameId(), ptcA.getFileName(), ptcA.getMatrix()])
            else:
                self.doubtSubset.append([ptcA.getNameId(), ptcA.getFileName(), ptcA.getMatrix()])
                self.doubtSubset.append([ptcA.getNameId(), ptcA.getFileName(), ptcB.getMatrix()])
    '''