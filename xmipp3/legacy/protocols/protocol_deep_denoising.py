# **************************************************************************
# *
# * Authors:     Javier Mota   (original implementation)
# *              Ruben Sanchez (added U-net and refactoring)
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
# **************************************************************************

import json
from shutil import copyfile
import sys, os
import numpy as np
import re

from pyworkflow import VERSION_2_0
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
from pyworkflow.utils.path import cleanPath

import pwem.emlib.metadata as md
from pwem import emlib

import xmipp3
from xmipp3.convert import writeSetOfParticles, setXmippAttributes, xmippToLocation
from xmipp3.protocols import XmippProtCompareReprojections
from xmipp3.protocols import XmippProtGenerateReprojections
from xmipp3 import XmippProtocol

from pyworkflow import BETA, UPDATED, NEW, PROD

EXEC_MODES = ['Train & Predict', 'Predict']
ITER_TRAIN = 0
ITER_PREDICT = 1

MODEL_TYPES = ["GAN", "U-Net"]
MODEL_TYPE_GAN = 0
MODEL_TYPE_UNET = 1


TRAINING_DATA_MODE = ["ParticlesAndSyntheticNoise", "OnlyParticles"]
TRAINING_DATA_MODE_SYNNOISE = 0
TRAINING_DATA_MODE_PARTS = 1


TRAINING_LOSS = [ "MSE", "PerceptualLoss", "Both"]
TRAINING_LOSS_MSE = 0
TRAINING_LOSS_PERCEPTUAL = 1
TRAINING_LOSS_BOTH = 2

PRED_BATCH_SIZE=2000

class XmippProtDeepDenoising(XmippProtGenerateReprojections, XmippProtocol):

    _label ="deep denoising"
    _lastUpdateVersion = VERSION_2_0
    _conda_env = 'xmipp_DLTK_v0.3'
    _devStatus = BETA

    def __init__(self, **args):
        XmippProtGenerateReprojections.__init__(self, **args)

    def _defineParams(self, form):

        form.addSection('Input')
        form.addHidden(params.GPU_LIST, params.StringParam,
                       expertLevel=cons.LEVEL_ADVANCED,
                       label="Choose GPU IDs", default='0',
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on."
                            " In case to use several GPUs separate with comas:"
                            "0,1,2")

        form.addParam('modelTrainPredMode', params.EnumParam, choices=EXEC_MODES,
                       default=ITER_TRAIN,
                       label='Train or predict model',
                       help='*Train*: Train the model using noisy particles '
                            'or their projections in an initial volume'
                            '*Predict*: The particles are denoised with a '
                            'pretrained model')

        form.addParam('continueTraining', params.BooleanParam, default = False,
                      condition='modelTrainPredMode==%d'%ITER_TRAIN,
                      label='Continue training? (or train from scratch)', help='Setting "yes" '
                      'you can continue training from pretrained model '
                      ' or your previous executions. If you choose'
                      '"no" the model will be trained from scratch. yes option is experimental')

        form.addParam('customModelOverPretrain', params.BooleanParam, default = False,
                      condition='modelTrainPredMode==%d or continueTraining'%ITER_PREDICT,
                      label='Use your own model (or use pretrained)', help='Setting "yes" '
                      'you can choose your own model trained. If you choose'
                      '"no" a general model pretrained will be assign')

        form.addParam('ownModel', params.PointerParam,
                      pointerClass=self.getClassName(),
                      condition='customModelOverPretrain==True and (modelTrainPredMode==%d or continueTraining)'%ITER_PREDICT,
                      label='Set your model',
                      help='Choose the protocol where your model is trained')

        form.addParam('modelType', params.EnumParam,
                      choices=MODEL_TYPES,
                      default=MODEL_TYPE_UNET,
                      condition='modelTrainPredMode==%d'%ITER_TRAIN,
                      label='Select model type',
                      help='If you set to *%s*, GAN will be employed '
                           'employed. If you set to *%s* U-Net will be used instead'
                           % tuple(MODEL_TYPES))

        form.addParam('inputProjections', params.PointerParam, allowsNull=True,
                      pointerClass='SetOfParticles', important=False,
                      label='Input projections to train (mandatory)/compare (optional)',
                      help='use the protocol generate reprojections to generate the '
                      'projections. If compare reprojections protocol output is used as '
                      '"Input noisy particles", this field is ignored')

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticles', important=True,
                      label='Input noisy particles to denoise', help='Input noisy '
                      'particles from the protocol generate reprojections if '
                      'you are training or from any other protocol if you are '
                      'predicting. If compare reprojections protocol output is used as '
                      '"Input noisy particles", "Input projections to train" is ignored')

        form.addParam('emptyParticles', params.PointerParam, expertLevel=cons.LEVEL_ADVANCED,
                      pointerClass='SetOfParticles',  allowsNull=True,
                      label='Input "empty" particles (optional)', help='Input "empty" '
                      'particles to learn how to deal with pure noise')

        form.addParam('imageSize', params.IntParam, allowsNull=True, expertLevel=cons.LEVEL_ADVANCED,
                      condition='modelTrainPredMode==%d and not continueTraining'%ITER_TRAIN,
                      label='Scale images to (px)',
                      default=-1, help='Scale particles to desired size to improve training'
                                        'The recommended particle size is 128 px. The size must be even.'
                                         'Do not use loss=perceptualLoss or loss=Both if  96< size <150.')

        form.addSection(label='Training')

        form.addParam('nEpochs', params.FloatParam,
                      condition='modelTrainPredMode==%d'%ITER_TRAIN,
                      label="Number of epochs", default=25.0,
                      help='Number of epochs for neural network training. GAN requires much '
                           'more epochs (>100) to obtain succesfull results')

        form.addParam('learningRate', params.FloatParam,
                      condition='modelTrainPredMode==%d'%ITER_TRAIN,
                      label="Learning rate", default=1e-4,
                      help='Learning rate for neural network training')


        form.addParam('modelDepth', params.IntParam, default=4,
                       condition='modelTrainPredMode==%d'%ITER_TRAIN, expertLevel=cons.LEVEL_ADVANCED,
                       label='Model depth',
                       help='Indicate the model depth. For 128-64 px images, 4 is the recommend value. '
                            ' larger images may require bigger models')

        form.addParam('regularizationStrength', params.FloatParam, default=1e-5,
                       condition='modelTrainPredMode==%d'%ITER_TRAIN, expertLevel=cons.LEVEL_ADVANCED,
                       label='Regularization strength',
                       help='Indicate the regularization strength. Make it bigger if sufferening overfitting'
                            ' and smaller if suffering underfitting')

        form.addParam('trainingSetType', params.EnumParam, choices=TRAINING_DATA_MODE,
                       condition='modelTrainPredMode==%d'%ITER_TRAIN,
                       default=TRAINING_DATA_MODE_SYNNOISE, expertLevel=cons.LEVEL_ADVANCED,
                       label='Select how to generate training set',
                       help='*ParticlesAndSyntheticNoise*: Train using particles and synthetic noise\n'
                            'or\n*OnlyParticles*: using only particles\n'
                            'or\n*Both*: Train using both strategies')

        form.addParam('trainingLoss', params.EnumParam, choices=TRAINING_LOSS,
                       condition='modelTrainPredMode==%d'%ITER_TRAIN,
                       default=TRAINING_LOSS_BOTH, expertLevel=cons.LEVEL_ADVANCED,
                       label='Select loss for training',
                       help='*MSE*: Train using mean squered error or\n*PerceptualLoss*: '
                            'Train using DeepConsensus perceptual loss\n or\n*Both*: Train '
                            'using both DeepConsensus perceptual loss and mean squered error\n')


        form.addParam('numberOfDiscVsGenUpdates', params.IntParam, default=5,
                       condition='modelType==%d and modelTrainPredMode==%d'%(MODEL_TYPE_GAN, ITER_TRAIN),
                       expertLevel=cons.LEVEL_ADVANCED,
                       label='D/G trainig ratio',
                       help='Indicate the number of times the discriminator is trained for each '
                            'generator training step. If discriminator loss is going to 0, make it '
                            'smaller, whereas if the discriminator is not training, make it bigger')

        form.addParam('loss_logWeight', params.FloatParam, default=3, expertLevel=cons.LEVEL_ADVANCED,
                       condition='modelType==%d and modelTrainPredMode==%d'%(MODEL_TYPE_GAN, ITER_TRAIN),
                       label='D/G loss ratio',
                       help='Indicate the 10^lossRatio times that the generator loss is stronger than '
                            ' the discriminator loss. If discriminator loss is going to 0, make it '
                            'smaller, whereas if the generator is not training, make it bigger')

        form.addParallelSection(threads=2, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
	
        deps = []
        ret= self._insertFunctionStep('preprocessData', prerequisites=[])
        deps+= [ret]
        if self.modelTrainPredMode.get() == ITER_TRAIN:
            ret= self._insertFunctionStep('trainModel', prerequisites=deps)
            deps+= [ret]
        ret= self._insertFunctionStep('predictModel', prerequisites=deps)
        deps+= [ret]
        ret= self._insertFunctionStep('createOutputStep', prerequisites=deps)
        deps+= [ret]

    def _validate(self):
        errorMsg = []
        if not self.checkIfInputIsCompareReprojection()  and self.modelTrainPredMode.get()==ITER_TRAIN and self.inputProjections.get() is None:
          errorMsg.append("Error, in training mode, both particles and projections must be provided")
        if self.imageSize.get() is None and self.modelTrainPredMode.get()==ITER_TRAIN:
          errorMsg.append("Error, in training mode, image size must be provided")      
        return errorMsg

    def _getResizedSize(self):
        resizedSize= self.imageSize.get()
        if self.modelTrainPredMode.get()==ITER_PREDICT:
          if self.customModelOverPretrain.get()== True:
            resizedSize= self.ownModel.get()._getResizedSize()
          else:
            resizedSize= 128
        resizedSize= resizedSize if resizedSize>0 else 128
        return resizedSize

    def getStackOrResize(self, setOfParticles, mdFnameIn, stackFnameOut):

        if self._getResizedSize()== self.inputParticles.get().getDimensions()[0]:
            mdFname= re.sub(r"\.stk$", r".xmd", stackFnameOut)
            writeSetOfParticles(setOfParticles, mdFname )
            self.runJob("xmipp_image_convert", "-i %s -o %s " % (mdFname, stackFnameOut))
        else:
            writeSetOfParticles(setOfParticles, mdFnameIn)
            self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (mdFnameIn, stackFnameOut,
                                                                            self._getResizedSize()))
    def copyStackAndResize(self, stackIn, stackOut):
        copyfile(stackIn, stackOut + ".tmp")
        if self._getResizedSize() != self.inputParticles.get().getDimensions()[0]:
          self.runJob("xmipp_image_resize",
                      "-i %s -o %s --fourier %d" % (stackOut + ".tmp", stackOut, self._getResizedSize()))
          os.remove(stackOut + ".tmp")

    def checkIfInputIsCompareReprojection(self):
        return isinstance( self.inputParticles.getObjValue(), XmippProtCompareReprojections)

    def preprocessData(self):

        particlesFname = self._getExtraPath('noisyParticles.xmd')
        fnNewParticles = self._getExtraPath('resizedParticles.stk')
        projectionsFname = self._getExtraPath('projections.xmd')
        fnNewProjections = self._getExtraPath('resizedProjections.stk')

        if self.checkIfInputIsCompareReprojection():
          for part in self.inputParticles.get():
            projections_stk, particles_stk= part._xmipp_imageRef._filename.get(), part._xmipp_image._filename.get()
            print(projections_stk, particles_stk)
            break
          self.copyStackAndResize(particles_stk, fnNewParticles)
          self.copyStackAndResize(projections_stk, fnNewProjections)
        else:
          self.getStackOrResize(self.inputParticles.get(), particlesFname, fnNewParticles)

          if not self.inputProjections.get() is None:

              self.getStackOrResize(self.inputProjections.get(), projectionsFname, fnNewProjections)

        if not self.emptyParticles.get() is None:
            emptyPartsFname = self._getExtraPath('emptyParts.xmd')
            fnNewEmptyParts = self._getExtraPath('resizedEmptyParts.stk')
            self.getStackOrResize(self.emptyParticles.get(), emptyPartsFname, fnNewEmptyParts)


    def trainModel(self):

        modelFname = self._getPath('ModelTrained.h5')
        builder_args= {"modelType":MODEL_TYPES[self.modelType.get()], "boxSize":self._getResizedSize(),
                       "saveModelFname":modelFname,
                       "modelDepth": self.modelDepth.get(), "gpuList":self.gpuList.get(),
                       "generatorLoss": TRAINING_LOSS[self.trainingLoss.get()],
                       "trainingDataMode": TRAINING_DATA_MODE[self.trainingSetType.get()],
                       "regularizationStrength": self.regularizationStrength.get() }
        if self.modelType.get() == MODEL_TYPE_GAN:
          builder_args["training_DG_ratio"]= self.numberOfDiscVsGenUpdates.get()
          builder_args["loss_logWeight"]= self.loss_logWeight.get()

        dataPathParticles= self._getExtraPath('resizedParticles.xmd')
        dataPathProjections= self._getExtraPath('resizedProjections.xmd')
        dataPathEmpty= self._getExtraPath('resizedEmptyParts.xmd')

        argsDict={}
        argsDict["builder"]=builder_args
        argsDict["running"]={"lr":self.learningRate.get(), "nEpochs":self.nEpochs.get()}

        jsonFname= self._getExtraPath("training_conf.json")
        with open(jsonFname, "w") as f:
          json.dump(argsDict, f)

        args= " --mode training -i %s -p %s -c %s "%(dataPathParticles, dataPathProjections, jsonFname)
        if os.path.isfile(dataPathEmpty):
          args+=" --empty_particles %s"%dataPathEmpty

        self.runJob("xmipp_deep_denoising", args, numberOfMpi=1,
                    env=self.getCondaEnv())


    def _getModelFname(self):
        if self.modelTrainPredMode.get() == ITER_PREDICT:
          if self.customModelOverPretrain == True:
              modelFname = self.ownModel.get()._getPath('ModelTrained.h5')
          else:
              modelFname = self.getModel('deepDenoising', 'ModelTrained.h5')
        else:
          modelFname = self._getPath('ModelTrained.h5')
          if self.continueTraining.get():
            if self.customModelOverPretrain == True:
              modelFnameInit = self.ownModel.get()._getPath('ModelTrained.h5')
            else:
              modelFnameInit = self.getModel('deepDenoising', 'ModelTrained.h5')

            copyfile(modelFnameInit, modelFname)
        return modelFname

    def predictModel(self):

      builder_args = {"boxSize": self._getResizedSize(), "saveModelFname": self._getModelFname(),
                      "modelDepth": -1, "gpuList": self.gpuList.get(),"modelType":MODEL_TYPES[self.modelType.get()],
                      "generatorLoss": TRAINING_LOSS[self.trainingLoss.get()], "batchSize": PRED_BATCH_SIZE}
      argsDict={}
      argsDict["builder"] = builder_args

      jsonFname = self._getExtraPath("predict_conf.json")
      with open(jsonFname, "w") as f:
        json.dump(argsDict, f)

      inputParticlesMdName = self._getExtraPath('resizedParticles.xmd')
      dataPathProjections = self._getExtraPath('resizedProjections.xmd')
      outputParticlesMdName = self._getExtraPath('particlesDenoised.xmd')

      args = " --mode denoising -i %s -c %s -o %s"% (inputParticlesMdName, jsonFname, outputParticlesMdName)

      if os.path.isfile(dataPathProjections):
        args += " -p %s" % dataPathProjections
      elif self.checkIfInputIsCompareReprojection():
        args += " -p %s" % self._getExtraPath('resizedProjections.stk')

      self.runJob("xmipp_deep_denoising", args, numberOfMpi=1,
                  env=self.getCondaEnv())

    def createOutputStep(self):
        imgSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(imgSet)
        Ts = imgSet.getSamplingRate()
        xdim = imgSet.getDimensions()[0]
        outputSet.setSamplingRate((Ts*xdim)/self._getResizedSize())
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath('inputParticles.xmd'))
        imgFname = self._getExtraPath('particlesDenoised.xmd')
        outputSet.copyItems(imgSet, updateItemCallback=self._processRow,
                            itemDataIterator=md.iterRows(imgFname, sortByLabel=md.MDL_ITEM_ID) )

        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

        cleanPath(self._getExtraPath('resizedParticles.xmd'))
        cleanPath(self._getExtraPath('noisyParticles.xmd'))
        cleanPath(self._getExtraPath('projections.xmd'))
        if os.path.exists(self._getExtraPath('resizedProjections.xmd')):
            cleanPath(self._getExtraPath('resizedProjections.xmd'))
        if os.path.exists(self._getExtraPath('resizedEmptyParts.xmd')):
            cleanPath(self._getExtraPath('resizedEmptyParts.xmd'))
        # extra cleaning    
        if os.path.exists(self._getExtraPath('training_conf.json')):
            cleanPath(self._getExtraPath('training_conf.json'))
        if os.path.exists(self._getExtraPath('predict_conf.json')):
            cleanPath(self._getExtraPath('predict_conf.json'))            
        if os.path.exists(self._getExtraPath('batchImages/')):
            #cleanPath(self._getExtraPath('batchImages/*.png'))
            cleanPath(self._getExtraPath('batchImages'))

    def _processRow(self, particle, row):
        particle.setLocation(xmippToLocation(row.getValue(emlib.MDL_IMAGE)))
        if self.inputProjections.get() is not None:
            mdToAdd= (md.MDL_IMAGE_ORIGINAL, md.MDL_IMAGE_REF, md.MDL_CORR_DENOISED_PROJECTION, md.MDL_CORR_DENOISED_NOISY)
        else:
            mdToAdd= (md.MDL_IMAGE_ORIGINAL, md.MDL_CORR_DENOISED_NOISY )

        setXmippAttributes(particle, row, *mdToAdd)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Particles denoised")
        return summary

    def _validate(self):
        assertModel = (self.modelTrainPredMode.get()==ITER_PREDICT and
                       not self.customModelOverPretrain.get())
        errors = self.validateDLtoolkit(assertModel=assertModel,
                                        model=('deepDenoising', 'ModelTrained.h5'),
                                        errorMsg="Required with 'Predict' mode when "
                                                 "no custom model is provided.")

        return errors
