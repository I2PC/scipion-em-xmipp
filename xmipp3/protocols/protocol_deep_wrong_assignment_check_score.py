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
from pwem.protocols import EMProtocol
from ..convert import writeSetOfParticles, locationToXmipp
#readSetOfParticles,
from pwem.emlib.image import ImageHandler
from math import floor

from pyworkflow import BETA, UPDATED, NEW, PROD

import xmippLib

from pyworkflow.object import ObjectWrap, String, Float, Integer, Object
from pwem.constants import (NO_INDEX, ALIGN_NONE, ALIGN_PROJ, ALIGN_2D,
                            ALIGN_3D)
from pwem.objects import (Angles, Coordinate, Micrograph, Volume, Particle,
                          MovieParticle, CTFModel, Acquisition, SetOfParticles,
                          Class3D, SetOfVolumes, Transform)
from pwem.emlib.image import ImageHandler
import pwem.emlib.metadata as md

from xmipp3.base import getLabelPythonType, iterMdRows

from pwem import emlib

from xmipp3.convert import rowFromMd, ALIGNMENT_DICT, rowToParticle
from xmipp3.constants import EXTRA_FIELD_CP_SCORE
#, _containsAny

#TODO: check what should go in the ()
class XmippProtWrongAssignCheckScore(EMProtocol, XmippProtocol):

    #TODO: Protocol explanation comment

    # Identification parameters
    _label      = 'deep wrong angle assignment check scoring'
    _devStatus  = BETA
    _conda_env = 'xmipp_DLTK_v1.0'

    _possibleOutputs = {'outputParticles' : SetOfParticles}

    REF_SCORING = "scoringSet"

    #TODO: Investigate how to properly prepare this (both)
    def _init_(self, **kwargs):
        super()._init_(**kwargs)

    #TODO: Include, wherever, inference output filename variable
    #TODO: Include, wherever, final output filename variable 

    #--------------- DEFINE param functions ---------------
    def _defineParams(self, form: Form):
        
        #TODO: Explain the use of this
        form.addParallelSection(threads=8, mpi=0)

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
        #form.addParam('pretrainedModel', PointerParam, label = "Model", pointerClass = "ModelObj")
        
        form.addParam('modelToUse', FileParam, 
                    label = "Select pretrained model", 
                    help = 'Select the H5 format filename of the trained neural network.' 
                    'Note that the architecture must have been compiled using this same protocol.')
        
        #TODO: allowsNull = True (false is default)
        #TODO: maybe no batches should not be the default value
        #TODO: include changes from training protocol (default, expert_level etc)
        form.addParam('batchSz', IntParam, 
                    label = "Batch size", 
                    expertLevel = LEVEL_ADVANCED, 
                    default = 1, 
                    help = 'Must be an integer. If batch size is bigger than the dataset size or is set to 0 only one batch will be used.')

    #--------------- INSERT steps functions ----------------

    #TODO: Write function definition
    def _insertAllSteps(self):

        self._insertFunctionStep(self.convertInputs)
        self._insertFunctionStep(self.useGenerateResidualsStep)
        self._insertFunctionStep(self.callScoringProgramStep)
        self._insertFunctionStep(self.createOutputStep)

    #--------------- STEPS functions ----------------

    #TODO: write function definition
    def convertInputs(self):

        self.inputInference : SetOfParticles = self.inferenceInput.get()
        self.inputInferenceFn = self._getExtraPath("prescored.xmd")
        writeSetOfParticles(self.inputInference, self.inputInferenceFn)

        #TODO: Keep in mind possible changes in training protocol
        self.inputVol = self.inputVolume.get()
        self.fnVol = self._getTmpPath("volume.vol")

        #self.pretrainedModel = self.pretrainedModel.get()
        
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

        args = ' -i ' + self.inputInferenceFn # file containing images for inference
        args += ' -m ' + self.model # file containing the model to be used
        args += ' -o ' + self.fnOutputFile # 
        args += ' -b ' + str(self.batchSize)

        #TODO: IMPORTANT DON'T FORGET GPU usage

        self.runJob(program,args,env=self.getCondaEnv())

    #TODO: explain what is going on in the function
    #TODO: write function definition
    def createOutputStep(self):
        
        outputSet = self._createSetOfParticles(suffix = "_scored")
        outputSet.copyInfo(self.inputInference)
        outMd = xmippLib.MetaData(self.fnOutputFile)
        probabilities = outMd.getColumnValues(xmippLib.MDL_CLASS_PROBABILITY)
        readSetOfParticles(self.fnOutputFile,outputSet,prob = probabilities)

        self._defineOutputs(**{"OutputParticles": outputSet})
        self._defineSourceRelation(self.inputInference, outputSet)
        """
        

        for scoreRow, ptcl in zip(probabilities, outputSet):
            #newPtcl = ptcl.clone()
            #newPtcl.setObjId(count)
            #TODO: Evaluate changing this name into something more coherent with the MDL label
            ptcl._score = Float(scoreRow) #Extended atribute
            #ptcl.write()
            #outputSet.append(newPtcl)

        outputSet.write()
        self._store()
        """
        '''
        "__________"
        outputSet = self._createSetOfParticles(suffix = "_scored")
        outputSet.copyInfo(self.inputInference)
        readSetOfParticles(self.fnOutputFile,outputSet)
        outMd = xmippLib.MetaData(self.fnOutputFile)
        probabilities = outMd.getColumnValues(xmippLib.MDL_CLASS_PROBABILITY)

        idCounter = 1
        for scoreRow, ptcl in zip(probabilities, self.inputInference):
            newPtcl = ptcl.clone()
            newPtcl.setObjId(idCounter)
            newPtcl._score = Float(scoreRow) #Extended atribute
            outputSet.append(newPtcl)
            idCounter += 1

        self._defineOutputs(**{"OutputParticles": outputSet})
        self._defineSourceRelation(self.inputInference, outputSet)
        '''

    #--------------- INFO functions -------------------------
    def _summary(self):
        summary = []
        #TODO: appends to summary with the relevant info

    # ------------- UTILS functions -------------------------

    #TODO: Any change must be also included in training protocol
    #TODO: write function definition    
    def generateResiduals(self, fnSet, fnVol, ref):

        #TODO: properly comment this external code (and inform it is external)

        xDim, _, _, _, _ = xmippLib.MetaDataInfo(fnSet)
        xDimVol = self.inputVolume.get().getXDim()

        img = ImageHandler()
        vol = self._getTmpPath("volume.vol")
        img.convert(fnVol,vol)

        #TODO: Include changes in training protocol once checked
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
        #TODO: investigate CTF for educated comments

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

def _containsAny(row, labels):
    """ Check if the labels (values) in labelsDict
    are present in the row.
    """
    values = labels.values() if isinstance(labels, dict) else labels
    return any(row.containsLabel(l) for l in values)

def readSetOfParticles(filename, partSet, **kwargs):
    readSetOfImages(filename, partSet, rowToParticle, **kwargs)

def readSetOfImages(filename, imgSet, rowToFunc, **kwargs):
    """read from Xmipp image metadata.
        filename: The metadata filename where the image are.
        imgSet: the SetOfParticles that will be populated.
        rowToFunc: this function will be used to convert the row to Object
    """
    imgMd = emlib.MetaData(filename)

    # By default remove disabled items from metadata
    # be careful if you need to preserve the original number of items
    if kwargs.get('removeDisabled', True):
        imgMd.removeDisabled()

    # If the type of alignment is not sent through the kwargs
    # try to deduced from the metadata labels
    if 'alignType' not in kwargs:
        imgRow = rowFromMd(imgMd, imgMd.firstObject())
        if _containsAny(imgRow, ALIGNMENT_DICT):
            if imgRow.containsLabel(emlib.MDL_ANGLE_TILT):
                kwargs['alignType'] = ALIGN_PROJ
            else:
                kwargs['alignType'] = ALIGN_2D
        else:
            kwargs['alignType'] = ALIGN_NONE

    if imgMd.size() > 0:
        for pb, objId in zip(kwargs['prob'],imgMd):
            imgRow = rowFromMd(imgMd, objId)
            img = rowToFunc(imgRow, **kwargs)
            setattr(img, EXTRA_FIELD_CP_SCORE,Float(pb))
            imgSet.append(img)

        imgSet.setHasCTF(img.hasCTF())
        imgSet.setAlignment(kwargs['alignType'])

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