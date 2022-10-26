# **************************************************************************
# *
# * Authors:     COS Sorzano and Erney Ramirez
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
                                        IntParam, BooleanParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils.path import moveFile, cleanPattern
from pwem.protocols import ProtAlign2D
from pwem.emlib.metadata import iterRows, getFirstRow
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

class XmippProtDeepCenterPredict(ProtAlign2D, xmipp3.XmippProtocol):
    """Predict the center particles using deep learning.""" 
    _label = 'deep center predict'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v0.3'

    def __init__(self, **args):
        ProtAlign2D.__init__(self, **args)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")
        form.addSection(label='Input')
        form.addParam('inputImageSet', PointerParam, label="Input Image set",
                      pointerClass='SetOfParticles',
                      help='The set of particles to predict shift')
        form.addParam('inputModel', PointerParam, label="model trained",
                      pointerClass='XmippProtDeepCenter',
                      help='The set of particles to predict shift')
        
        form.addParam('Xdim', IntParam, label="Size of the images for training", default=128)

        form.addParallelSection(threads=8, mpi=1)

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
        self._insertFunctionStep("predict", numGPU[0])

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputImageSet.get(), self.predictImgsFn)
        self.runJob("xmipp_image_resize",
                    "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                    (self.predictImgsFn,
                        self._getExtraPath('trainingResized.stk'),
                        self._getExtraPath('trainingResized.xmd'),
                        self.Xdim), numberOfMpi=self.numberOfThreads.get()*self.numberOfMpi.get())
        self.model = self.modelTrain.get()
        
        self.model = self.inputModel.get()._getExtraPath()

    def predict(self, gpuId):
        args = "%s %s Shift 5 %d %d %s" % (
                self._getExtraPath("trainingResized.xmd"), self._getExtraPath("modelShift.h5"), self.numEpochs,
                self.batchSize.get(), gpuId)
        self.runJob("xmipp_deep_center", args, numberOfMpi=1, env=self.getCondaEnv())

        args = "%s %s Psi 120 %d %d %s" % (
            self._getExtraPath("trainingResized.xmd"), self._getExtraPath("modelPsi.h5"), self.numEpochs,
            self.batchSize.get(), gpuId)
        self.runJob("xmipp_deep_center", args, numberOfMpi=1, env=self.getCondaEnv())

        args = "%s %s Rot 0 %d %d %s" % (
            self._getExtraPath("trainingResized.xmd"), self._getExtraPath("modelRot.h5"), self.numEpochs,
            self.batchSize.get(), gpuId)
        self.runJob("xmipp_deep_center", args, numberOfMpi=1, env=self.getCondaEnv())

        args = "%s %s Tilt 0 %d %d %s" % (
            self._getExtraPath("trainingResized.xmd"), self._getExtraPath("modelTilt.h5"), self.numEpochs,
            self.batchSize.get(), gpuId)
        self.runJob("xmipp_deep_center", args, numberOfMpi=1, env=self.getCondaEnv())

        remove(self._getExtraPath("trainingResized.xmd"))
        remove(self._getExtraPath("trainingResized.stk"))

    # --------------------------- INFO functions --------------------------------
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We learned a model to center particles from %i input images (%s)." \
                % (self.inputSet.get().getSize(), self.getObjectTag('inputSet')))
        return methods
