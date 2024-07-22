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

    #--------------- DEFINE param functions ---------------
    #TODO: Scipion Form code function
    def _defineParams(self, form: Form):

        pass

    #--------------- INSERT steps functions ----------------

    #TODO: Insert steps function
    def _insertAllSteps(self):

        pass

    #--------------- STEPS functions ----------------

    #TODO: Read inputs step function
    #TODO: Prepare input function, if separate
    #TODO: Perform Inference function
    def callScoringProgram():
        
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

    #TODO: Create output step

    #--------------- INFO functions -------------------------
    def _summary(self):
        summary = []
        #TODO: appends to summary with the relevant info

    # ------------- UTILS functions -------------------------

    pass