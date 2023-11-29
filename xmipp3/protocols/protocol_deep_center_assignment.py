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
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        IntParam, BooleanParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils import Message
from pwem.protocols import ProtAlign2D
from pwem.objects import String
from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import os
from os import remove
import xmipp3
from xmipp_base import XmippScript
from pyworkflow import BETA, UPDATED, NEW, PROD

class XmippProtDeepCenterAssignmentPredictBase(ProtAlign2D, xmipp3.XmippProtocol):
    """Super protocol of Deep Center and Deep Global Assignment."""
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'
    _trainingResizedFileName = 'trainingResized'

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
                      help='The set of particles to predict')

        form.addParam('trainModels', BooleanParam, label="Train models",
                      pointerClass='SetOfParticles', default=False,
                      help='Choose if you want to train a model using a centered set of particles')
        
        trainingGroup = form.addGroup('Training parameters', condition='trainModels==True')

        trainingGroup.addParam('numModels', IntParam,
                      label="Number of models", default=5,
                      help="The maximum number of model available in xmipp is 5.")

        trainingGroup.addParam('inputTrainSet', PointerParam, label="Input train set",
                      pointerClass='SetOfParticles', allowsNull=True,
                      pointerCondition='hasAlignment2D or hasAlignmentProj',
                      help='The set of particles to train the models')

        trainingGroup.addParam('numEpochs', IntParam,
                      label="Number of epochs",
                      default=25,
                      expertLevel=LEVEL_ADVANCED,
                      help="Number of epochs for training.")

        trainingGroup.addParam('batchSize', IntParam,
                      label="Batch size for training",
                      default=32,
                      expertLevel=LEVEL_ADVANCED,
                      help="Batch size for training.")

        trainingGroup.addParam('learningRate', FloatParam,
                      label="Learning rate",
                      default=0.001,
                      expertLevel=LEVEL_ADVANCED,
                      help="Learning rate for training.")

        trainingGroup.addParam('sigma', FloatParam,
                      label="Image shifting",
                      default=5,
                      expertLevel=LEVEL_ADVANCED,
                      help="A measure of the number of pixels that particles can be shifted in each direction from the center.")

        trainingGroup.addParam('patience', IntParam,
                      label="Patience",
                      default=5,
                      expertLevel=LEVEL_ADVANCED,
                      help="Training will be stopped if the number of epochs without improvement is greater than "
                           "patience.")

        form.addSection(label='Consensus')

        form.addParam('tolerance', IntParam,
                      label="Tolerance in pixels", default=3,
                      help="Max difference between predictions and their mean value.")

        form.addParam('maxModels', IntParam,
                      label="Maximum number of models dropped per particle", default=0,
                      help="If more models are dropped, the particle is discarded.")

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.predictImgsFn = self._getExtraPath('predict_input_imgs.xmd')
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')
        self._insertFunctionStep("convertStep", self.inputImageSet.get())
        self._insertFunctionStep("predict", numGPU[0], self.predictImgsFn)
        self._insertFunctionStep('createOutputStep')
    
    def insertTrainSteps(self):
        self.trainImgsFn = self._getExtraPath('train_input_imgs.xmd')
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')

        self._insertFunctionStep("convertTrainStep")
        self._insertFunctionStep("train", numGPU[0])

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self, inputSet):
        self.Xdim = 128
        writeSetOfParticles(inputSet, self.predictImgsFn)
        self.runJob("xmipp_image_resize",
                    "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                    (self.predictImgsFn,
                     self._getExtraPath(f'{self._trainingResizedFileName}.stk'),
                     self._getExtraPath(f'{self._trainingResizedFileName}.xmd'),
                     self.Xdim), numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())
    
    def convertTrainStep(self):
        self.convertStep(self.inputTrainSet.get())
    
    def train(self, gpuId, mode="", orderSymmetry=None):
        args = "%s %s %f %d %d %s %d %f %d %d" % (
            self._getExtraPath(f"{self._trainingResizedFileName}.xmd"), self._getExtraPath("model"), self.sigma.get(),
            self.numEpochs, self.batchSize.get(), gpuId, self.numModels.get(), self.learningRate.get(),
            self.patience.get(), 0)
        if orderSymmetry:
            args += " " + str(orderSymmetry)
        self.runJob(f"xmipp_deep_{mode}", args, numberOfMpi=1, env=self.getCondaEnv())

        remove(self._getExtraPath(f"{self._trainingResizedFileName}.xmd"))
        remove(self._getExtraPath(f"{self._trainingResizedFileName}.stk"))

    def predict(self, predictImgsFn, gpuId, mode="", inputModel="", trainedModel=True, orderSymmetry=None):
        if mode != "center" or trainedModel:
            args = "%s %s %s %s %s %d %d %d" % (
                self._getExtraPath(f"{self._trainingResizedFileName}.xmd"), gpuId, self._getPath(), predictImgsFn,
                inputModel, self.numModels.get(), self.tolerance.get(),
                self.maxModels.get())
            if orderSymmetry:
                args += " " + str(orderSymmetry)
        else:
            args = "%s %s %s %s %s %d %d %d" % (
                self._getExtraPath(f"{self._trainingResizedFileName}.xmd"), gpuId, self._getPath(), predictImgsFn,
                os.path.join(XmippScript.getModel("deep_center"), 'modelCenter'), self.numModels.get(), self.tolerance.get(),
                self.maxModels.get())
        self.runJob(f"xmipp_deep_{mode}_predict", args, numberOfMpi=1, env=self.getCondaEnv())
        remove(self._getExtraPath(f"{self._trainingResizedFileName}.xmd"))
        remove(self._getExtraPath(f"{self._trainingResizedFileName}.stk"))

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

class XmippProtDeepCenter(XmippProtDeepCenterAssignmentPredictBase):
    """Predict the center particles using deep learning.""" 
    _label = 'deep center'
    _devStatus = BETA
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        if self.trainModels.get():
            self.insertTrainSteps()
        super()._insertAllSteps()

    # --------------------------- STEPS functions ---------------------------------------------------
    def train(self, gpuId, mode="", orderSymmetry=None):
        super().train(gpuId, mode="center")

    def predict(self, predictImgsFn, gpuId, mode="", inputModel="", trainedModel=True, orderSymmetry=None):
        super().predict(gpuId, predictImgsFn, mode="center",
                                                           inputModel=self._getExtraPath("model"),
                                                           trainedModel=self.trainModels.get())

class XmippProtDeepGlobalAssignment(XmippProtDeepCenterAssignmentPredictBase):
    """Predict Euler Angles using deep learning."""
    _label = 'deep global assignment'
    _devStatus = BETA

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        # Calling parent form
        super()._defineParams(form)

        # Adding extra conditions
        trainModels = form.getParam('trainModels')
        trainModels.condition = String('False')

        # Always show training parameters group in this protocol
        trainingGroup = form.getParam('Training_parameters')
        trainingGroup.condition = String('True')

        section = form.getSection(label=Message.LABEL_INPUT)
        section.addParam('orderSymmetry', IntParam,
                      label="Order of symmetry", default=1,
                      help="Order of the group of the molecule.")
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.insertTrainSteps()
        super()._insertAllSteps()

    # --------------------------- STEPS functions ---------------------------------------------------
    def train(self, gpuId, mode="", orderSymmetry=None):
        super().train(gpuId, mode="global_assignment", orderSymmetry=self.orderSymmetry.get())

    def predict(self, predictImgsFn, gpuId, mode="", inputModel="", trainedModel=True, orderSymmetry=None):
        super().predict(gpuId, predictImgsFn, mode="global_assignment", inputModel=self._getExtraPath("model"), orderSymmetry=self.orderSymmetry.get())
