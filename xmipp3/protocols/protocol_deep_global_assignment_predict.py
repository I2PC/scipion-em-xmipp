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


class XmippProtDeepGlobalAssignmentPredict(ProtAlign2D, xmipp3.XmippProtocol):
    """Predict the center particles using deep learning."""
    _label = 'deep global assignment predict'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'

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

        form.addParam('Xdim', IntParam, label="Size of the images for training", default=128)

        form.addParam('inputModel', PointerParam, label="Model trained",
                      pointerClass='XmippProtDeepGlobalAssignment',
                      help='The model to predict angles')

        form.addParam('symmetry', IntParam, label="Order of Symmetry", default=1,
                      help='Order of symmetry by the symmetry axis of the molecule')

        form.addSection(label='Consensus')

        form.addParam('numAngModels', IntParam,
                      label="Number of models trained", default=5)

        form.addParam('tolerance', IntParam,
                      label="Angular tolerance (º)", default=15,
                      help="Max angular difference between predictions and their mean value.")

        form.addParam('maxModels', IntParam,
                      label="Maximum number of models dropped per particle", default=0,
                      help="If more models are dropped, the particle is discarded.")

        form.addParallelSection(threads=1, mpi=1)

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
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputImageSet.get(), self.predictImgsFn)
        self.runJob("xmipp_image_resize",
                    "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                    (self.predictImgsFn,
                     self._getExtraPath('trainingResized.stk'),
                     self._getExtraPath('trainingResized.xmd'),
                     self.Xdim), numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())

    def predict(self, gpuId):
        self.modelAng = self.inputModel.get()
        args = "%s %s %s %s %s %d %d %d %d" % (
            self._getExtraPath("trainingResized.xmd"), gpuId, self._getPath(), self.predictImgsFn,
            self.modelAng._getExtraPath("modelAngular"), self.numAngModels.get(), self.tolerance.get(), self.maxModels.get(), self.symmetry.get())

        self.runJob("xmipp_deep_global_assignment_predict", args, numberOfMpi=1, env=self.getCondaEnv())

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
