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
#TODO: Check comments are used cohesively

import os

from pyworkflow.protocol.params import (Form, PointerParam, FloatParam, EnumParam, FileParam, IntParam, BooleanParam)
from pwem.objects import (SetOfParticles)
from pyworkflow.object import Float, Integer, Boolean
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from xmipp3 import XmippProtocol
from pwem.protocols import EMProtocol #TODO: do I need this?
from ..convert import writeSetOfParticles, locationToXmipp

#TODO: check what should go in the ()
class XmippProtWrongAssignCheckScore(EMProtocol, XmippProtocol):

    #TODO: Protocol explanation comment

    # Identification parameters
    _label      = 'deep wrong angle assignment check scoring'
    _devStatus  = BETA
    #TODO: is this true?
    _conda_env = 'xmipp_DLTK_v1.0'
    _possibleOutputs = {'outputParticles' : SetOfParticles}

    REF_SCORING = "scoringSet"

    #TODO: include init

    #TODO: Include, wherever, inference output filename variable
    #TODO: Include, wherever, final output filename variable 

    #--------------- DEFINE param functions ---------------
    def _defineParams(self, form: Form):
        
        #TODO: Evaluate the use of this
        form.addParallelSection(threads=8, mpi=1)

        form.addSection(label='main')
        form.addParam('inferenceInput', PointerParam,
                    label = "Input particles",
                    important = True,
                    pointerClass = 'SetOfParticles')
        form.addParam('inputVolume',PointerParam,
                    label = "Volume to compare images to",
                    important = True,
                    pointerClass = 'Volume',
                    help = 'Volume used for residuals generation')
        form.addParam('modelToUse', FileParam, 
                    label = "Select pretrained model", 
                    help = 'Select the H5 format filename of the trained neural network.' 
                    'Note that the architecture must have been compiled using this same protocol.')
        #TODO: maybe no batches should not be the default value
        #TODO: evaluate if this param should really be advanced
        form.addParam('batchSz', IntParam, 
                    label = "Batch size", 
                    expertLevel = LEVEL_ADVANCED, 
                    default = 0, 
                    help = 'Must be an integer. If batch size is bigger than the dataset size only one batch will be used. '
                    'Also, if batchSize left to 0 only one batch will be used.')

    #--------------- INSERT steps functions ----------------

    #TODO: Write function definition
    def _insertAllSteps(self):

        self.readInputs()
        self._insertFunctionStep(self.useGenerateResidualsStep)
        self._insertFunctionStep(self.callScoringProgramStep)
        self._insertFunctionStep(self.createOutputStep)

    #--------------- STEPS functions ----------------

    #TODO: evaluate change name into convert to follow "convention"
    #TODO: write function definition
    def readInputs(self):

        self.inputInference : SetOfParticles = self.inferenceInput.get()
        self.inputInferenceFn = self._getExtraPath("prescored.xmd")
        writeSetOfParticles(self.inputInference, self.inputInferenceFn)

        #TODO: Keep in mind possible changes in training protocol
        self.inputVol = self.inputVolume.get()
        self.fnVol = self._getTmpPath("volume.vol")

        self.model = self.modelToUse.get()
        self.batchSize = Integer(self.batchSz.get())

    #TODO: Keep in mind training protocol changes
    #TODO: write function definition
    def useGenerateResidualsStep(self):

        #TODO: Make sure there is no problem with not creating a subset of inference set (since we're including a few new columns)
        self.generateResiduals(self.inputInferenceFn, self.inputVol, self.REF_SCORING)


    #TODO: write funcion definition + args comments
    #TODO: evaluate if -> None in function would be useful
    def callScoringProgramStep(self):

        self.fnOutputFile = self._getExtraPath("scoreOutput.xmd")

        program = "xmipp_deep_wrong_assign_check_sc"

        args = ' -i ' + str(self.inputInference) # file containing images for inference
        args += ' -m ' + str(self.model) # file containing the model to be used
        args += ' -o ' + str(self.fnOutputFile) # 
        args += ' -b ' + self.batchSize

        #TODO: IMPORTANT DON'T FORGET GPU usage

        self.runJob(program,args)

    #TODO: write function definition
    def createOutputStep(self):
        
        outputSet = self._createSetOfParticles(suffix = "_after_inference")
        #TODO: does this file contain all the info I should use?
        outputSet.copyInfo(self.fnOutputFile)
        #TODO: setDim is necessary?
        #TODO: setObjLabel is necessary?
        #TODO: Why md.IterRows ?
        outputSet.copyItems(self.fnOutputFile) #TODO: should I include any oher arg?

        self._defineOutputs(**{"OutputParticles": outputSet})
        self._defineSourceRelation(self.inputInference, outputSet)
    

    #--------------- INFO functions -------------------------
    def _summary(self):
        summary = []
        #TODO: appends to summary with the relevant info

    # ------------- UTILS functions -------------------------

    #TODO: write function definition
    def generateResiduals(self):

        #TODO: include function from training protocol once completed 

        pass

    ##############################################################################################

    ''' # deprecated functionality, check before deleting

    def processInferenceResultStep(self):
        
        #TOD: check possible castings into particle and such with ":"

        #TOD: Keep in mind that infer output does not have MDL_REF (???)
        infmd = xmippLib.MetaData(self.inferOut)

        #TOD: should I include confidence levels as identifiable labels? 
        #TOD: modify how this value is set
        confThreshold = 0.95

        self.processedSet = []

        for i in range(0,len(infmd),2):

            #TOD: make sure this attr is properly visited
            #TOD: check attr name once the label is created
            if infmd[i].getAttributeValue(attrName = "classProbability") >= infmd[i+1].getAttributeValue(attrName = "classProbability"):
                if infmd[i].getAttributeValue(attrName = "classProbability") >= confThreshold:
                    self.processedSet.append(infmd[i])
            else:
                if infmd[i+1].getAttributeValue(attrName = "classProbability") >= confThreshold:
                    self.processedSet.append(infmd[i+1])

        #TOD: is any kind of casting necessary?
        self.finalSet = self.okSubset + self.processedSet
    '''