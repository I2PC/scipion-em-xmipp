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
from pyworkflow.utils.path import createLink
from pwem.protocols import ProtAlign2D
from pwem.objects import String
from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import os
from os import remove
import xmipp3
from xmipp_base import XmippScript
from pyworkflow import BETA, UPDATED, NEW, PROD

class XmippProtDeepCenter(ProtAlign2D, xmipp3.XmippProtocol):
    """Center a set of particles in 2D using a neural network. The particles remain the same, but their alignment
       includes an approximate shift to place them in the center. This protocol performs the training of the model. """
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'
    _label = 'deep center'
    _devStatus = UPDATED

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

        form.addParam('inputParticles', PointerParam, label="Input images",
                      pointerClass='SetOfParticles',
                      help='The set does not need to be centered or have alignment parameters')

        form.addParam('sigma', FloatParam,
                      label="Shift sigma",
                      default=5,
                      help="In pixels. This is used to generate artificially shifted particles.")

        form.addParam('precision', FloatParam,
                      label="Precision",
                      default=0.5,
                      help="In pixels.")

        form.addSection(label="Training")

        form.addParam('trainSetSize', IntParam, label="Train set size", default=5000,
                      help='How many particles to use for training. Set to -1 for all of them')

        form.addParam('numEpochs', IntParam,
                      label="Number of epochs",
                      default=100,
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
        self.fnImgs = self._getTmpPath('imgs.xmd')
        self.fnImgsTrain = self._getTmpPath('imgsTrain.xmd')
        self._insertFunctionStep("convertInputStep", self.inputParticles.get())
        self._insertFunctionStep("train")
        self._insertFunctionStep("predict" )
        self._insertFunctionStep("createOutputStep")

    def getGpusList(self, separator):
        strGpus = ""
        for elem in self._stepsExecutor.getGpuList():
            strGpus = strGpus + str(elem) + separator
        return strGpus[:-1]

    def setGpu(self, oneGPU=False):
        if oneGPU:
            gpus = self.getGpusList(",")[0]
        else:
            gpus = self.getGpusList(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.info(f'Visible GPUS: {gpus}')
        return gpus

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self, inputSet):
        writeSetOfParticles(inputSet, self.fnImgs)
        if self.trainSetSize.get()>0:
            self.runJob("xmipp_metadata_utilities","-i %s --operate random_subset %d -o %s"%\
                        (self.fnImgs,self.trainSetSize, self.fnImgsTrain), numberOfMpi=1)
        else:
            createLink(self.fnImgs, self.fnImgsTrain)

    def train(self):
        gpuId = self.setGpu(oneGPU=True)
        args = "-i %s --omodel %s --sigma %f --maxEpochs %d --batchSize %d --gpu %s --learningRate %f --precision %f"%\
                (self.fnImgsTrain, self._getExtraPath("model.h5"), self.sigma, self.numEpochs, self.batchSize, gpuId,
                 self.learningRate, self.precision)
        self.runJob(f"xmipp_deep_center", args, numberOfMpi=1, env=self.getCondaEnv())

    def predict(self):
        gpuId = self.setGpu(oneGPU=True)
        fnModel = self._getExtraPath("model.h5")
        args = "-i %s --gpu %s --model %s -o %s" % (self.fnImgs, gpuId, fnModel, self._getExtraPath('particles.xmd'))
        self.runJob("xmipp_deep_center_predict", args, numberOfMpi=1, env=self.getCondaEnv())

    def createOutputStep(self):
        fnPredict = self._getExtraPath("particles.xmd")
        outputSet = self._createSetOfParticles()
        readSetOfParticles(fnPredict, outputSet)
        outputSet.copyInfo(self.inputParticles.get())
        outputSet.setAlignment2D()
        self._defineOutputs(outputParticles=outputSet)
        self._store(outputSet)
        self._defineSourceRelation(self.inputParticles.get(), outputSet)


