# **************************************************************************
# *
# * Authors:     COS Sorzano, Erney Ramirez and Adrian Sansinena
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

from pyworkflow import VERSION_3_0
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        IntParam, BooleanParam, GPU_LIST, EnumParam)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils.path import moveFile, cleanPattern
from pyworkflow.utils import Message
from pwem.protocols import ProtAlign2D
from pwem.emlib.metadata import iterRows, getFirstRow
from copy import deepcopy
from pwem.objects import String
import pwem.emlib.metadata as md
from xmipp3.convert import createItemMatrix, setXmippAttributes, readSetOfParticles, writeSetOfParticles
from pwem import ALIGN_PROJ
from pwem import emlib
from pwem.emlib.image import ImageHandler
import os
import sys
import numpy as np
import math
from shutil import copy
from os import remove
from os.path import exists, join
import xmipp3
from xmipp_base import XmippScript

class XmippProtDeepCenterAssignmentPredictBase(ProtAlign2D, xmipp3.XmippProtocol):
    """Predict the center particles using deep learning."""
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'

    def __init__(self, **args):
        ProtAlign2D.__init__(self, **args)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addParallelSection(threads=1, mpi=1)

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")

        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputImageSet', PointerParam, label="Input Image set",
                      pointerClass='SetOfParticles',
                      help='The set of particles to predict shift')

        form.addParam('trainModels', BooleanParam, label="Train models",
                      pointerClass='SetOfParticles', default=False, #condition='False',
                      help='Choose if you want to train a model using a centered set of particles')

        form.addParam('numModels', IntParam,
                      label="Number of models", default=5,
                      help="The maximum number of model available in xmipp is 5.")

        form.addParam('inputTrainSet', PointerParam, label="Input Image set",
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignment2D or hasAlignmentProj',
                      help='The set of particles to predict shift')

        form.addSection(label='Consensus')

        form.addParam('tolerance', IntParam,
                      label="Tolerance in pixels", default=3,
                      help="Max difference between predictions and their mean value.")

        form.addParam('maxModels', IntParam,
                      label="Maximum number of models dropped per particle", default=0,
                      help="If more models are dropped, the particle is discarded.")

        form.addSection(label='Training parameters')

        form.addParam('numEpochs_cen', IntParam,
                      label="Number of epochs",
                      default=25, expertLevel=LEVEL_ADVANCED,
                      help="Number of epochs for training.")

        form.addParam('batchSize', IntParam,
                      label="Batch size for training",
                      default=32, expertLevel=LEVEL_ADVANCED,
                      help="Batch size for training.")

        form.addParam('learningRate', FloatParam,
                      label="Learning rate",
                      default=0.001, expertLevel=LEVEL_ADVANCED,
                      help="Learning rate for training.")

        form.addParam('sigma', FloatParam,
                      label="Image shifting",
                      default=10, expertLevel=LEVEL_ADVANCED,
                      help="A measure of the number of pixels that particles can be shifted in each direction from the center.")

        form.addParam('patience', IntParam,
                      label="Patience",
                      default=5, expertLevel=LEVEL_ADVANCED,
                      help="Training will be stopped if the number of epochs without improvement is greater than "
                           "patience.")

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.predictImgsFn = self._getExtraPath('predict_input_imgs.xmd')
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')
        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("predict", numGPU[0], self.predictImgsFn)
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        self.Xdim = 128
        writeSetOfParticles(self.inputImageSet.get(), self.predictImgsFn)
        self.runJob("xmipp_image_resize",
                    "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                    (self.predictImgsFn,
                     self._getExtraPath('trainingResized.stk'),
                     self._getExtraPath('trainingResized.xmd'),
                     self.Xdim), numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())

    def predict(self, predictImgsFn, gpuId, mode="", inputModel="", trainedModel=True):
        if mode != "center" or trainedModel:
            self.model = inputModel
            args = "%s %s %s %s %s %d %d %d" % (
                self._getExtraPath("trainingResized.xmd"), gpuId, self._getPath(), predictImgsFn,
                self.model._getExtraPath(), self.numModels.get(), self.tolerance.get(),
                self.maxModels.get())
        else:
            args = "%s %s %s %s %s %d %d %d" % (
                self._getExtraPath("trainingResized.xmd"), gpuId, self._getPath(), predictImgsFn,
                XmippScript.getModel("deep_center"), self.numModels.get(), self.tolerance.get(),
                self.maxModels.get())
        self.runJob(f"xmipp_deep_{mode}_predict", args, numberOfMpi=1, env=self.getCondaEnv())
        remove(self._getExtraPath("trainingResized.xmd"))
        remove(self._getExtraPath("trainingResized.stk"))

    def createOutputStep(self):
        imgFname = self._getPath('predict_results.xmd')
        outputSet = self._createSetOfParticles()
        readSetOfParticles(imgFname, outputSet)
        outputSet.copyInfo(self.inputImageSet.get())
        outputSet.setAlignmentProj()
        self._defineOutputs(outputParticles=outputSet)
        self._store(outputSet)
        self._defineSourceRelation(self.inputImageSet.get(), outputSet)

    # --------------------------- INFO functions --------------------------------
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We learned a model to center particles from %i input images (%s)." \
                           % (self.inputSet.get().getSize(), self.getObjectTag('inputSet')))
        return methods

class XmippProtDeepCenterPredict(XmippProtDeepCenterAssignmentPredictBase):
    """Predict the center particles using deep learning.""" 
    _label = 'deep center predict'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        _cond_trainModelsTrue = String('trainModels==True')
        # Calling specific parent form
        super()._defineParams(form)
        # Adding extra conditions

        trainModels = form.getParam('trainModels')
        trainModels.condition = String('True')

        numModels = form.getParam('numModels')
        numModels.condition = _cond_trainModelsTrue

        inputTrainSet = form.getParam('inputTrainSet')
        inputTrainSet.condition = _cond_trainModelsTrue

        numEpochs_cen = form.getParam('numEpochs_cen')
        numEpochs_cen.condition = _cond_trainModelsTrue

        batchSize = form.getParam('batchSize')
        batchSize.condition = _cond_trainModelsTrue

        learningRate = form.getParam('learningRate')
        learningRate.condition = _cond_trainModelsTrue

        sigma = form.getParam('sigma')
        sigma.condition = _cond_trainModelsTrue

        patience = form.getParam('patience')
        patience.condition = _cond_trainModelsTrue






    # --------------------------- STEPS functions ---------------------------------------------------
    def predict(self, predictImgsFn, gpuId, mode="center", inputModel="", trainedModel=True):
        boolUseTrainedModel = not self.trainModels.get()
        XmippProtDeepCenterAssignmentPredictBase().predict(gpuId, predictImgsFn, mode=mode,
                                                           inputModel=self._getExtraPath("modelCenter"),
                                                           trainedModel=boolUseTrainedModel)

    def _insertAllSteps(self):
        if self.trainModels.get():
            self.trainImgsFn = self._getExtraPath('train_input_imgs.xmd')
            if self.useQueueForSteps() or self.useQueue():
                myStr = os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                myStr = self.gpuList.get()
                os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
            numGPU = myStr.split(',')

            self._insertFunctionStep("convertTrainStep")
            self._insertFunctionStep("train", numGPU[0])

        XmippProtDeepCenterAssignmentPredictBase()._insertAllSteps()

    def convertTrainStep(self):
        self.Xdim = 128
        writeSetOfParticles(self.inputTrainSet.get(), self.trainImgsFn)
        self.runJob("xmipp_image_resize",
                    "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                    (self.trainImgsFn,
                     self._getExtraPath('trainingResized.stk'),
                     self._getExtraPath('trainingResized.xmd'),
                     self.Xdim), numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())

    def train(self, gpuId):

        args = "%s %s %f %d %d %s %d %f %d" % (
            self._getExtraPath("trainingResized.xmd"), self._getExtraPath("modelCenter"), self.sigma.get(),
            self.numEpochs_cen, self.batchSize.get(), gpuId, self.numModels.get(), self.learningRate.get(),
            self.patience.get())
        print(args)
        self.runJob("xmipp_deep_center", args, numberOfMpi=1, env=self.getCondaEnv())

        remove(self._getExtraPath("trainingResized.xmd"))
        remove(self._getExtraPath("trainingResized.stk"))

class XmippProtDeepGlobalAssignmentPredict(XmippProtDeepCenterAssignmentPredictBase):
    """Predict the center particles using deep learning."""
    _label = 'deep global assignment predict'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        _cond_trainModelsTrue = String('trainModels==True')
        # Calling specific parent form
        XmippProtDeepCenterAssignmentPredictBase._defineParams(self, form)
        # Adding extra conditions

        #trainModels = form.getParam('trainModels')
        #trainModels.condition = String('True')
#
        #numModels = form.getParam('numModels')
        #numModels.condition = _cond_trainModelsTrue
#
        #inputTrainSet = form.getParam('inputTrainSet')
        #inputTrainSet.condition = _cond_trainModelsTrue
#
        #numEpochs_cen = form.getParam('numEpochs_cen')
        #numEpochs_cen.condition = _cond_trainModelsTrue
#
        #batchSize = form.getParam('batchSize')
        #batchSize.condition = _cond_trainModelsTrue
#
        #learningRate = form.getParam('learningRate')
        #learningRate.condition = _cond_trainModelsTrue
#
        #sigma = form.getParam('sigma')
        #sigma.condition = _cond_trainModelsTrue
#
        #patience = form.getParam('patience')
        #patience.condition = _cond_trainModelsTrue



    # --------------------------- STEPS functions ---------------------------------------------------
    def predict(self, predictImgsFn, gpuId, mode="global_assignment", inputModel="", trainedModel=True):
        XmippProtDeepCenterAssignmentPredictBase.predict(self, predictImgsFn, gpuId, mode=mode,
                                                           inputModel=self.inputModel.get())
