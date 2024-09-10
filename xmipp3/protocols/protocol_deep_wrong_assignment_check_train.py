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

#TODO: Check imports and organize
#TODO: check whether " or ' will be used
#TODO: Check comments are used cohesively

#--------------- ATTENTION ----------------

#TODO: IMPORTANT!!!!!!! KEEP IN MIND Noise 0, Particle 1

#--------------- ATTENTION ----------------

import os

from pyworkflow.protocol.params import Form, PointerParam, FloatParam, EnumParam, FileParam, IntParam, BooleanParam, StringParam, GPU_LIST
from pwem.objects import SetOfParticles
from pyworkflow.object import Float, Integer, Boolean, String
from ..convert import writeSetOfParticles, locationToXmipp
from pwem import emlib
from pwem.emlib.image import ImageHandler
from math import floor
from pyworkflow.utils.path import cleanPath
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from xmipp3 import XmippProtocol
from pwem.protocols import EMProtocol
from pwem.objects.data import EMObject
#from xmipp3.objects import ModelLol

from pyworkflow import BETA, UPDATED, NEW, PROD

import xmippLib
'''
class ModelObj (EMObject):

    def __init__(self, fileName = None):
        self.fileName = String(fileName)
'''

class XmippProtWrongAssignCheckTrain(EMProtocol, XmippProtocol):

    #TODO: Protocol explanation comment

    # Identification parameters
    _label      = 'deep wrong angle assignment check training'
    _devStatus  = BETA
    _conda_env = 'xmipp_DLTK_v1.0'
    
    #_possibleOutputs = {'finalModel' : ModelObj}

    #TODO: Evaluate changing global variable name
    FORM_NN_USAGE_LABELS   = ["Model from scratch", "Use existing model"]
    #TODO: Explain what this is for
    SHIFTVALUE = 10

    #TODO: Evaluate changing name to text so it's named coherently
    REF_CORRECT = "okSubset"
    REF_INCORRECT = "wrongSubset"

    #TODO: Investigate how to properly prepare this (both)
    def _init_(self, **kwargs):
        super()._init_(**kwargs)

    #--------------- DEFINE param functions ---------------
    def _defineParams(self, form: Form):

        form.addHidden(GPU_LIST, StringParam, 
                        default='',
                        help="Your system may have several GPUs installed, "
                             " choose the one you'd like to use."
                        )
        
        #TODO: Explain the use of this
        form.addParallelSection(threads=6, mpi=0)

        form.addSection(label = 'Input')
        form.addParam('assignment', PointerParam, 
                    label = "Input assignment", 
                    important = True, 
                    pointerClass = 'SetOfParticles')
        form.addParam('inputVolume',PointerParam,
                    label = "Volume to compare images to",
                    important = True,
                    pointerClass = 'Volume',
                    help = 'Volume used for residuals generation')
        form.addParam('trainingModel', EnumParam, 
                    label = "Select NN usage", 
                    important = True, 
                    choices = self.FORM_NN_USAGE_LABELS, 
                    default = 0, 
                    help = 'When set to *%s*, a new model will be created from scratch.' 
                    'The option *%s* allows using a pre-existing NN model.' % tuple(self.FORM_NN_USAGE_LABELS))
        form.addParam('modelToUse', FileParam, 
                    label = "Select pretrained model", 
                    condition = 'trainingModel == 1', 
                    help = 'Select the H5 format filename of the trained neural network.' 
                    'Note that the architecture must have been compiled using this same protocol.')
        #TODO: what should I include as advanced? expertLevel = LEVEL_ADVANCED,
        #TODO: evaluate use of default value vs allows null = TRUE
        form.addParam('batchSz', IntParam, 
                    label = "Batch size", 
                    default = 1, 
                    help = 'Must be an integer. If batch size is bigger than the dataset size or is set to 0 only one batch will be used.')
        #TODO: finish help section once fully developed
        form.addParam('saveInfo', BooleanParam,
                      label = 'Save information about NN',
                      default=False,
                      help = 'When this flag is activated the protocol will save a document contaning information related to the Neural Network(s), monitored values during the training process.')
        #TODO: Finish Ensemble param when available in program
        #TODO: Decide numEpoch default + include help explanation (Maybe I want this to not be a param and optimize every time?)
        """
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
        """

    #--------------- INSERT steps functions ----------------
    
    #TODO: Write function defintion
    def _insertAllSteps(self):
        
        self._insertFunctionStep(self.convertInputs)
        self._insertFunctionStep(self.generateSubsetsStep)
        self._insertFunctionStep(self.useGenerateResidualsStep)
        self._insertFunctionStep(self.callTrainingProgramStep)

    #--------------- STEPS functions ----------------

    #TODO: write function definition
    def convertInputs(self):

        self.nThreads = str(self.numberOfThreads.get())

        self.inputAssgn : SetOfParticles = self.assignment.get()
        self.inputAssignFn = self._getExtraPath("initial.xmd")
        writeSetOfParticles(self.inputAssgn, self.inputAssignFn)
        
        self.inputVol = self.inputVolume.get()
        self.fnVol = self._getTmpPath("volume.vol")
        #self.fnVol = locationToXmipp(1,self.inputVol)
        #TODO: writeSetOfVolumes?

        self.modelType = Integer(self.trainingModel.get())

        if self.modelType == 1:
            self.nnModel = self.modelToUse.get()
        
        self.batchSize = Integer(self.batchSz.get())
        #self.numEnsemble = Integer(self.ensembleNum.get())
        #self.numEpochs = Integer(self.numEpoch.get())

        self.doSaveInfo = self.saveInfo.get()
        

    #TODO: write function definition
    def generateSubsetsStep(self):

        self.okSubset = self._getExtraPath("okSubset.xmd")
        self.wrongSubset = self._getExtraPath("wrongSubset.xmd")

        mdInit = xmippLib.MetaData(self.inputAssignFn)

        listOfLabelsGeneric = [xmippLib.MDL_ITEM_ID, xmippLib.MDL_IMAGE]
        listOfLabelsMatrix = [xmippLib.MDL_ANGLE_PSI, xmippLib.MDL_ANGLE_ROT, xmippLib.MDL_ANGLE_TILT, xmippLib.MDL_SHIFT_X, xmippLib.MDL_SHIFT_Y]

        mdOk = xmippLib.MetaData()
        mdWr = xmippLib.MetaData()

        for lbl in listOfLabelsGeneric:
            mdInit.copyColumnTo(mdOk, lbl, lbl)
            mdInit.copyColumnTo(mdWr, lbl, lbl)

        for lbl in listOfLabelsMatrix:
            mdInit.copyColumnTo(mdOk, lbl, lbl)
            aux = mdInit.getColumnValues(lbl)
            aux = aux[self.SHIFTVALUE:] + aux[:self.SHIFTVALUE]
            mdWr.setColumnValues(lbl,aux)

        mdOk.write(self.okSubset)
        mdWr.write(self.wrongSubset)
        
    #TODO: write function definition
    def useGenerateResidualsStep(self):
    
        self.generateResiduals(self.okSubset, self.inputVol, self.REF_CORRECT)
        self.generateResiduals(self.wrongSubset, self.inputVol, self.REF_INCORRECT)
        
    #TODO: write function definition
    #TODO: remember including ensemble args when implemented
    #TODO: check variable names for inputs (with changes in forms above)
    def callTrainingProgramStep(self):
        
        #TODO: change this into proper path for model output
        outputFn = self._getExtraPath("model.h5")

        program = "xmipp_deep_wrong_assign_check_tr"

        args = ' -c ' + self.okSubset ## File containing correct examples
        args += ' -w ' + self.wrongSubset ## File containing incorrect examples
        args += ' -o ' + outputFn ## File where the final model will be stored
        args += ' -b %d' % self.batchSize ## Number of images to feed the nn per batch (remainder is kept)

        if self.modelType == 1:
            # Model could be universal or specific from user
            args += ' -f ' + str(self.pretrainedMod) # file where the pretrained model is stored

        args += ' -t ' + self.nThreads

        #TODO: fully implement in program
        if self.doSaveInfo:
            #TODO: evaluate file format
            savedInfoFn = self._getExtraPath()
            args += ' -s ' + savedInfoFn

        #TODO: Write in help that the user may not write anything if all gpus are to be used
        if self.getGpuList():
            args += ' --gpus ' + ','.join(map(str, self.getGpuList()))

        #TODO: include param when default defined
        #args += ' -e ' + self. numEpochs
        #TODO: evaluate including learning rate args += ' -l ' + self.
        #TODO: evaluate including patience args += ' -p ' + self.

        self.runJob(program,args, env=self.getCondaEnv())
        
        #self._defineOutputs(finalModel = ModelObj(fileName = outputFn))

    
    #--------------- INFO functions -------------------------
    def _summary(self):
        summary = []
        #TODO: appends to summary with the relevant info

    # ------------- UTILS functions -------------------------
    #TODO: Any change must be also included in scoring protocol
    #TODO: write function definition    
    def generateResiduals(self, fnSet, fnVol, ref):

        #TODO: properly comment this external code (and inform it is external)

        xDim, _, _, _, _ = xmippLib.MetaDataInfo(fnSet)
        xDimVol = self.inputVolume.get().getXDim()

        img = ImageHandler()
        vol = self._getTmpPath("volume.vol")
        img.convert(fnVol,vol)

        if xDimVol != xDim:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" % (self.fnVol, xDim), numberOfMpi=1)

        anglesOutFn = self._getExtraPath("anglesCont_%s.stk" % ref)
        self.fnResiduals = self._getExtraPath("residuals%s.mrcs" % ref)

        program = "xmipp_angular_continuous_assign2"

        args = " -i " + fnSet ## Set of particles with initial alignment
        args += " -o " + anglesOutFn ## Output of the program (Stack of images prepared for 3D reconstruction)
        args += " --ref " + vol ## Reference volume
        args += " --optimizeAngles --optimizeShift"  #TODO: investigate for educated comments
        args += " --max_shift %d" % floor(xDim*0.05) ## Maximum shift allowed in pixels
        args += " --sampling %f " % self.inputVolume.get().getSamplingRate() ## Sampling rate (A/pixel)
        args += " --oresiduals " + self.fnResiduals ## Output stack of residuals
        args += " --ignoreCTF --optimizeGray --max_gray_scale 0.95 " ## Set values for gray optimization
        #TODO: investigate if these are the proper arguments to keep 
        #TODO: investigate CTF for educated comments (DONE, description pending)

        ## Calling the residuals generation cpp program
        self.runJob(program,args)

        ## Preparing residuals info to be manipulated
        mdRes = xmippLib.MetaData(self.fnResiduals) 
        ## Preparing original file info to be manipulated
        mdOrgn = xmippLib.MetaData(fnSet)
        ## Saving the new stack of residuals in the original file
        mdOrgn.setColumnValues(xmippLib.MDL_IMAGE_RESIDUAL,mdRes.getColumnValues(xmippLib.MDL_IMAGE))
        ## Updating the changes to the original file
        mdOrgn.write(fnSet)

        #TODO: Should I include any cleanPath?

##############################################################################################

''' #Previous snippet from generate SubsetStep()
aux = []

for particle in self.inputAssign:
    
    #self.okSubset.append([particle.getNameId(), particle.getFileName(), particle.getMatrix()])
    #self.wrongSubset.append([particle.getNameId(), particle.getFileName()])
    aux.append(particle.getMatrix())

aux = aux[self.SHIFTVALUE:] + aux[:self.SHIFTVALUE]

for ptcl, value in zip(self.wrongSubset, aux):

    ptcl.append(value)
'''

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