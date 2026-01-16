# **************************************************************************
# *
# * Authors:     COS Sorzano
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
from pyworkflow.protocol.params import (PointerParam, StringParam, EnumParam, FileParam, IntParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils import Message
from pwem.protocols import ProtAlign2D
from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import os
import xmipp3
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippProtDeepCenterPredict(ProtAlign2D, xmipp3.XmippProtocol):
    """Center a set of particles in 2D using a neural network. The particles remain the same, but their alignment
       includes an approximate shift to place them in the center. This protocol only predicts, it does not train. """
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'
    _label = 'deep center predict'
    _devStatus = BETA

    PRETRAINED = 0
    PREVIOUS = 1
    LOCALFILE = 2

    def __init__(self, **args):
        ProtAlign2D.__init__(self, **args)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addParallelSection(threads=1, mpi=16)

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

        form.addParam('modelSource', EnumParam, label="Alignment source", default=self.PRETRAINED,
                      choices=["Pretrained model", "Previous protocol", "Local file"],
                      help="Source for the neural network")

        form.addParam('protocolPointer', PointerParam, label="Protocol", condition="modelSource==1",
                      pointerClass="XmippProtDeepCenter", allowsNull=True)

        form.addParam('modelFile', FileParam, label="Model", condition="modelSource==2",
                      help="Provide a local .h5 file")
        
        form.addParam('modelXdim', IntParam, label="Image size of model", condition="modelSource==2",
                      default = 64, help="Image size on which the model was trained")

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.fnImgs = self._getExtraPath('imgs.xmd')
        self._insertFunctionStep("convertInputStep", self.inputParticles.get())
        self._insertFunctionStep("predict")
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
        Xdim = self.inputParticles.get().getDimensions()[0]
        self.scaleFactor = 1.0
        if self.modelSource == self.PRETRAINED and Xdim!=64:
            self.scaleFactor = float(Xdim)/64.0
        elif self.modelSource == self.LOCALFILE and Xdim!=self.modelXdim.get():
            self.scaleFactor = float(Xdim)/self.modelXdim.get()
        epsilon = 1e-6  # Un margen de tolerancia pequeño
        if abs(self.scaleFactor - 1.0) > epsilon:
            fnTmp = self._getTmpPath("imgs.stk")
            self.runJob("xmipp_image_resize", "-i %s -o %s --fourier 64" % (self.fnImgs, fnTmp))
            self.fnImgs = self._getTmpPath("imgs.xmd")

    def predict(self):
        if self.modelSource==self.PRETRAINED:
            fnModel = self.getModel('deepCenter', 'deepCenterModel.h5') 
        elif self.getRunMode()==self.PREVIOUS:
            fnModel = self.protocolPointer.get()._getExtraPath("model.h5")
        else:
            fnModel = self.modelFile.get()
        gpuId = self.setGpu(oneGPU=True)
        args = "-i %s --gpu %s --model %s -o %s --scale %f" % (self.fnImgs, gpuId, fnModel,
                                                               self.fnImgs, self.scaleFactor)
        self.runJob("xmipp_deep_center_predict", args, numberOfMpi=1, env=self.getCondaEnv())
        epsilon = 1e-6
        if abs(self.scaleFactor - 1.0) > epsilon:
            fnShifts = self._getTmpPath("shifts.xmd")
            self.runJob("xmipp_metadata_utilities", '-i %s --operate keep_column "itemId shiftX shiftY psi" -o %s'%\
                        (self.fnImgs,fnShifts), numberOfMpi=1)
            self.fnImgs=self._getExtraPath("imgs.xmd")
            self.runJob("xmipp_metadata_utilities", '-i %s --set join %s itemId -o %s'%\
                        (fnShifts, self.fnImgs, self.fnImgs), numberOfMpi=1)

    def createOutputStep(self):
        fnPredict = self.fnImgs
        outputSet = self._createSetOfParticles()
        readSetOfParticles(fnPredict, outputSet)
        outputSet.copyInfo(self.inputParticles.get())
        outputSet.setAlignment2D()
        self._defineOutputs(outputParticles=outputSet)
        self._store(outputSet)
        self._defineSourceRelation(self.inputParticles.get(), outputSet)


    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are not errors. If some errors are found, a list with
        the error messages will be returned.
        """
        error=self.validateDLtoolkit()
        return error
