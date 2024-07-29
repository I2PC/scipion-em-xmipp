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

import os

from pyworkflow.protocol.params import (Form, PointerParam, FloatParam, EnumParam, FileParam, IntParam, BooleanParam)
from pwem.objects import (SetOfParticles)
from pyworkflow.object import Float, Integer, Boolean

#TODO: check what should go in the ()
class XmippProtWrongAssignCheckScore():

    #TODO: Protocol explanation comment

    # Identification parameters
    _label      = 'deep wrong angle assignment check'
    _devStatus  = BETA
    _conda_env = 'xmipp_DLTK_v1.0'
    #TODO: what is this? (below)
    #_stepsCheckSecs  = 60 # Scipion steps check interval (in seconds)
    #_possibleOutputs = {'output3DCoordinates' : SetOfCoordinates3D}

    #TODO: include init

    #TODO: Include, wherever, inference output filename variable
    #TODO: Include, wherever, final output filename variable 

    #--------------- DEFINE param functions ---------------
    #TODO: Scipion Form code function
    def _defineParams(self, form: Form):

        #TODO: keep in mind the other protocols TODO's

        form.addSection(label='main')
        form.addParam('inferenceInput', PointerParam,
                      label = "Input particles",
                      important = True,
                      pointerClass = 'SetOfParticles')
        form.addParam('modelToUse', FileParam, 
                      label = "Select model for inference", 
                      condition = 'trainingModel == %s' %self.MODEL_PRETRAIN, 
                      help = 'Select the H5 format filename of the trained neural network. Note that the architecture must have been compiled using this same protocol.')
        form.addParam('batchSz', IntParam, 
                      label = "Batch size", 
                      expertLevel = LEVEL_ADVANCED, 
                      default = 0, 
                      help = 'Must be an integer. If batch size is bigger than the dataset size only one batch will be used.')
        
        #TODO: Include help info for batches: If batchSize left to 0 only one batch will be used.

    #--------------- INSERT steps functions ----------------

    #TODO: Insert steps function
    def _insertAllSteps(self):

        pass

    #--------------- STEPS functions ----------------

    #TODO: write function definition
    def readInputsStep(self):

        self.inputInference : SetOfParticles = self.inferenceInput.get()
        #TODO: should I check if this is a string?
        self.model = self.modelToUse.get()
        self.batchSize = Integer(self.batchSz.get())

    #TODO: Read inputs step function
    #TODO: Prepare input function, if separate
    #TODO: Perform Inference function
    def callScoringProgramStep():
        
        pass

    #TODO: Prepare inference output function (Copied from previous protocolo)

    #TODO: Write function definition
    def processInferenceResultStep(self):
        
        #TODO: check possible castings into particle and such with ":"

        #TODO: Keep in mind that infer output does not have MDL_REF
        infmd = xmippLib.MetaData(self.inferOut)

        #TODO: should I include confidence levels as identifiable labels? 
        #TODO: modify how this value is set
        confThreshold = 0.95

        self.processedSet = []

        for i in range(0,len(infmd),2):

            #TODO: make sure this attr is properly visited
            #TODO: check attr name once the label is created
            if infmd[i].getAttributeValue(attrName = "classProbability") >= infmd[i+1].getAttributeValue(attrName = "classProbability"):
                if infmd[i].getAttributeValue(attrName = "classProbability") >= confThreshold:
                    self.processedSet.append(infmd[i])
            else:
                if infmd[i+1].getAttributeValue(attrName = "classProbability") >= confThreshold:
                    self.processedSet.append(infmd[i+1])

        #TODO: is any kind of casting necessary?
        self.finalSet = self.okSubset + self.processedSet

    ''' # Moved from previos joined protocol just in case
    #TODO: step to prepare output for Scipion (how do I return it to a set of particles?)
    #TODO: write function definition
    def createOutputStep():

        pass
    '''

    #--------------- INFO functions -------------------------
    def _summary(self):
        summary = []
        #TODO: appends to summary with the relevant info

    # ------------- UTILS functions -------------------------

    pass