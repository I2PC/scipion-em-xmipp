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

class XmippProtDeepCenter(ProtAlign2D, xmipp3.XmippProtocol):
    """Center a set of particles with a neural network."""
    _label = 'deep center'
    _devStatus = BETA
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'

    def __init__(self, **args):
        ProtAlign2D.__init__(self, **args)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addParallelSection(threads=1, mpi=4)

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")

        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputImageSet', PointerParam, label="Input Image set",
                      pointerClass='SetOfParticles',
                      help='The set of particles to center')
        form.addParam('sigma', FloatParam,
                      label="Shift sigma",
                      default=5,
                      expertLevel=LEVEL_ADVANCED,
                      help="Sigma for the training of the shift")
        form.addParam('Xdim', IntParam,
                      label="Image size",
                      default=128,
                      expertLevel=LEVEL_ADVANCED,
                      help="Image size during the processing")

        form.addSection('Training parameters')

        form.addParam('numModels', IntParam,
                      label="Number of models", default=5,
                      help="The maximum number of model available in xmipp is 5.")

        form.addParam('trainSetSize', IntParam, label="Train set size", default=5000,
                      help='How many particles from the training')

        form.addParam('numEpochs', IntParam,
                      label="Number of epochs",
                      default=10,
                      expertLevel=LEVEL_ADVANCED,
                      help="Number of epochs for training.")

        form.addParam('batchSize', IntParam,
                      label="Batch size for training",
                      default=32,
                      expertLevel=LEVEL_ADVANCED,
                      help="Batch size for training.")

        form.addParam('learningRate', FloatParam,
                      label="Learning rate",
                      default=0.001,
                      expertLevel=LEVEL_ADVANCED,
                      help="Learning rate for training.")

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')

        self._insertFunctionStep("train", numGPU[0])
        self._insertFunctionStep("predict", numGPU[0])
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions ---------------------------------------------------
    def train(self, gpuId):
        fnTrain = self._getTmpPath("trainingImages")
        writeSetOfParticles(self.inputImageSet.get(), fnTrain+".xmd")
        self.runJob("xmipp_metadata_utilities","-i %s.xmd --operate random_subset %d"%\
                    (fnTrain,self.trainSetSize), numberOfMpi=1)
        self.runJob("xmipp_image_resize",
                    "-i %s.xmd -o %s.stk --save_metadata_stack %s.xmd --fourier %d" %
                    (fnTrain, fnTrain, fnTrain, self.Xdim),
                    numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())
        args = "%s %s %f %d %d %s %d %f" %\
               (fnTrain+".xmd", self._getExtraPath("model"), self.sigma,
                self.numEpochs, self.batchSize, gpuId, self.numModels, self.learningRate)
        self.runJob(f"xmipp_deep_center", args, numberOfMpi=1, env=self.getCondaEnv())

    def predict(self, gpuId):
        fnPredict = self._getExtraPath("predictImages")
        fnPredictResized = self._getTmpPath("predictImages")
        writeSetOfParticles(self.inputImageSet.get(), fnPredict+".xmd")
        self.runJob("xmipp_image_resize",
                    "-i %s.xmd -o %s.stk --save_metadata_stack %s.xmd --fourier %d" %
                    (fnPredict, fnPredictResized, fnPredictResized, self.Xdim),
                    numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())
        args = "%s %s %s %s" % (
            fnPredict+".xmd", gpuId, fnPredictResized+".xmd", self._getExtraPath("model"))
        self.runJob("xmipp_deep_center_predict", args, numberOfMpi=1, env=self.getCondaEnv())

    def createOutputStep(self):
        fnPredict = self._getExtraPath("predictImages.xmd")
        outputSet = self._createSetOfParticles()
        readSetOfParticles(fnPredict, outputSet)
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
