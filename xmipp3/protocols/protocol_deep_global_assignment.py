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


class XmippProtDeepGlobalAssignment(ProtAlign2D, xmipp3.XmippProtocol):
    """Learns a model to assign angles to particles using deep learning. Particles must be previously centered with respect
       to a volume, and they must have 3D alignment information. """
    _label = 'deep global assignment'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'

    def __init__(self, **args):
        ProtAlign2D.__init__(self, **args)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        _cond_modelAngPretrainTrue = 'modelAngPretrain==True'
        _cond_modelAngPretrainFalse = 'modelAngPretrain==False'
        _cond_modelCenPretrainTrue = 'modelCenPretrain==True'
        _cond_modelCenPretrainFalse = 'modelCenPretrain==False'
        trainingOption = 'trainingOption=='
        _center_training = 0
        _angular_training = 1
        _center_angular_training = 2

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

        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')

        form.addSection(label='Training parameters')

        form.addParam('trainingOption', EnumParam, display=EnumParam.DISPLAY_COMBO,
                      default=_center_angular_training,
                      choices=['Center assignment', 'Angular assignment', 'Center and angular assignment'], label="Training option: ",
                      help='Select if you want to train a model to center particles, perform angular assignment or '
                           'both.')

        form.addParam('modelAngPretrain', BooleanParam, default=False,
                      label='Use a pretrained model for global assignment',
                      condition=f'not {trainingOption}{_center_training}',
                      help='Set "yes" if you want to use a previously trained model. '
                           'If you choose "no" new models will be trained.')

        form.addParam('modelCenPretrain', BooleanParam, default=False,
                      label='Use a pretrained model to center particles',
                      condition=f'not {trainingOption}{_angular_training}',
                      help='Set "yes" if you want to use a previously trained model. '
                           'If you choose "no" new models will be trained.')

        form.addParam('pretrainedAngModels', PointerParam,
                      pointerClass='XmippProtDeepGlobalAssignment',
                      condition=f'{_cond_modelAngPretrainTrue} and (not {trainingOption}{_center_training})',
                      label='Pretrained global assignment model',
                      help='Select a pretrained model that performs global assignment to re-train it. ')

        form.addParam('pretrainedCenModels', PointerParam,
                      pointerClass='XmippProtDeepGlobalAssignment',
                      condition=f'{_cond_modelCenPretrainTrue} and (not {trainingOption}{_angular_training})',
                      label='Pretrained center model',
                      help='Select a pretrained model that centers particles to re-train it. ')

        form.addParam('numAngModels', IntParam,
                      label="Number of models for angular assignment", default=5,
                      condition=f'not {trainingOption}{_center_training}',
                      help="Choose number of models you want to train. More than 1 is recommended only if next step "
                           "is inference.")

        form.addParam('numCenModels', IntParam,
                      label="Number of models to center particles", default=5,
                      condition=f'not {trainingOption}{_angular_training}',
                      help="Choose number of models you want to train. More than 1 is recommended only if next step "
                           "is inference.")

        form.addParam('numAngEpochs', IntParam,
                      label="Number of epochs for global assignment",
                      default=25, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_center_training}',
                      help="Number of epochs for training.")

        form.addParam('numCenEpochs', IntParam,
                      label="Number of epochs to center particles",
                      default=25, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_angular_training}',
                      help="Number of epochs for training.")

        form.addParam('batchSizeAng', IntParam,
                      label="Batch size for global assignment",
                      default=32, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_center_training}',
                      help="Batch size for training.")

        form.addParam('batchSizeCen', IntParam,
                      label="Batch size to center particles",
                      default=32, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_angular_training}',
                      help="Batch size for training.")

        form.addParam('learningRateAng', FloatParam,
                      label="Learning rate for global assignment",
                      default=0.001, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_center_training}',
                      help="Learning rate for training.")

        form.addParam('learningRateCen', FloatParam,
                      label="Learning rate to cneter particles",
                      default=0.001, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_angular_training}',
                      help="Learning rate for training.")

        form.addParam('sigmaAng', FloatParam,
                      label="Image shifting during global assignment training",
                      default=3, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_center_training}',
                      help="Perform Data Augmentation. Maximum number of pixels that particles can be shifted in each "
                           "direction from the center.")

        form.addParam('sigmaCen', FloatParam,
                      label="Image shifting during center training",
                      default=15, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_angular_training}',
                      help="Perform Data Augmentation. Maximum number of pixels that particles can be shifted in each "
                           "direction from the center.")

        form.addParam('patienceAng', IntParam,
                      label="Training patience for global assignment",
                      default=10, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_center_training}',
                      help="Training will be stopped if the number of epochs without improvement is greater than "
                           "patience.")

        form.addParam('patienceCen', IntParam,
                      label="Training patience to center particles",
                      default=10, expertLevel=LEVEL_ADVANCED,
                      condition=f'not {trainingOption}{_angular_training}',
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
                     self.Xdim), numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())

    def train(self, gpuId):

        sig = self.sigma.get()

        self.pretrained = 'no'
        self.pathToModel = 'none'
        #if self.modelPretrain:
        #    self.model = self.pretrainedModels.get()
        #    self.pretrained = 'yes'
        #    self.pathToModel = self.model._getExtraPath("modelAngular0.h5")

        args = "%s %s %f %d %d %s %d %f %d %s %s %s" % (
            self._getExtraPath("trainingResized.xmd"), self._getExtraPath("modelAngular"), sig,
            self.numEpochs_ang, self.batchSize.get(), gpuId, self.numAngModels.get(), self.learningRate.get(),
            self.patience.get(), self.pretrained, self.pathToModel, self.symmetryGroup.get())
        print(args)
        self.runJob("xmipp_deep_global_assignment", args, numberOfMpi=1, env=self.getCondaEnv())

        remove(self._getExtraPath("trainingResized.xmd"))
        remove(self._getExtraPath("trainingResized.stk"))

    # --------------------------- INFO functions --------------------------------
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We learned a model to center particles from %i input images (%s)." \
                           % (self.inputSet.get().getSize(), self.getObjectTag('inputSet')))
        return methods
