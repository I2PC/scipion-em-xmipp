# **************************************************************************
# *
# * Authors:     COS Sorzano and Adrian Sansinena
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
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam, EnumParam,
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

class XmippProtDeepCenter(ProtAlign2D, xmipp3.XmippProtocol):
    """Learns a model to center particles using deep learning. Particles must be previously centered with respect
       to a volume, and they must have 3D alignment information. """
    _label = 'deep center'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v0.3'
    _cond_modelPretrainTrue = 'modelPretrain==True'
    _cond_modelPretrainFalse = 'modelPretrain==False'


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
        form.addParam('inputTrainSet', PointerParam, label="Input training set",
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignment2D or hasAlignmentProj',
                      help='The set of particles previously aligned to be used as training set')

        form.addParam('Xdim', IntParam, label="Size of the images for training", default=128)

        form.addParam('modelPretrain', BooleanParam, default=False,
                      label='Choose if you want to use a pretrained model',
                      help='Set "yes" if you want to use a previously trained model. '
                           'If you choose "no" new models will be trained.')

        form.addParam('pretrainedModels', PointerParam,
                      pointerClass='XmippProtDeepCenter',
                      condition=self._cond_modelPretrainTrue,
                      label='Pretrained model',
                      help='Select the pretrained model. ')

        form.addSection(label='Training parameters')
        form.addParam('numCenModels', IntParam,
                      label="Number of models for particle centering", default=5,
                      help="Choose number of models you want to train. More than 1 is recommended only if next step "
                           "is inference.")

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

        form.addParallelSection(threads=1, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.trainImgsFn = self._getExtraPath('train_input_imgs.xmd')
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')

        self._insertFunctionStep("convertStep")
        self._insertFunctionStep("train", numGPU[0])

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputTrainSet.get(), self.trainImgsFn)
        self.runJob("xmipp_image_resize",
                    "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                    (self.trainImgsFn,
                        self._getExtraPath('trainingResized.stk'),
                        self._getExtraPath('trainingResized.xmd'),
                        self.Xdim), numberOfMpi=self.numberOfThreads.get()*self.numberOfMpi.get())

    def train(self, gpuId):

        sig = self.sigma.get()

        self.pretrained = 'no'
        self.pathToModel = 'none'
        if self.modelPretrain:
            self.model = self.pretrainedModels.get()
            self.pretrained = 'yes'
            self.pathToModel = self.model._getExtraPath("modelCenter0.h5")

        args = "%s %s %f %d %d %s %d %f %d %s %s" % (
            self._getExtraPath("trainingResized.xmd"), self._getExtraPath("modelCenter"), sig,
            self.numEpochs_cen, self.batchSize.get(), gpuId, self.numCenModels.get(), self.learningRate.get(),
            self.patience.get(), self.pretrained, self.pathToModel)
        print(args)
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
