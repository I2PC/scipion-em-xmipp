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
from pyworkflow.protocol.params import (PointerParam, EnumParam, FileParam, StringParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils import Message
from pwem.protocols import ProtRefine3D
from pwem.objects import String
from pwem.emlib.image import ImageHandler
from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import glob
import math
import os
import xmipp3
import xmippLib
from xmipp_base import XmippScript
from pyworkflow import BETA, UPDATED, NEW, PROD

class XmippProtDeepAlign2Predict(ProtRefine3D, xmipp3.XmippProtocol):
    """Apply neural network models to align particles"""
    _label = 'deep global predict'
    _devStatus = BETA
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'

    def __init__(self, **args):
        ProtRefine3D.__init__(self, **args)

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

        form.addParam('inputParticles', PointerParam, label="Input particles",
                      pointerClass='SetOfParticles', allowsNull=False)

        form.addParam('modelSource', EnumParam, label='Model source', choices=['Protocol', 'Directory'], default=0)
        form.addParam('modelProtocol', PointerParam, label='Protocol', condition='modelSource==0',
                      pointerClass='XmippProtDeepAlign2')
        form.addParam('modelDir', FileParam, label='Directory', condition='modelSource==1',
                      help='If the directory is chosen, it should contain model*.h5 and info.txt')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.predictImgsFn = self._getExtraPath('predict_input_imgs.xmd')
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')
        self._insertFunctionStep("getInfo")
        self._insertFunctionStep("prepareData")
        self._insertFunctionStep("predict", numGPU[0])
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ---------------------------------------------------
    def getInfo(self):
        if self.modelSource==0:
            self.fnModelDir = self.modelProtocol.get()._getExtraPath()
        else:
            self.fnModelDir = self.modelDir.get()

        fhInfo = open(os.path.join(self.fnModelDir,"info.txt"))
        self.Xdim = int(fhInfo.readline().split('=')[1])
        self.volDiameter = int(fhInfo.readline().split('=')[1])
        self.symmetry = fhInfo.readline().split('=')[1].strip()
        fhInfo.close()

    def prepareData(self):
        fnImgs = self._getTmpPath("images")
        fnImgsResized = self._getTmpPath("imagesResized")
        writeSetOfParticles(self.inputParticles.get(), fnImgs+".xmd")
        cropSize = int(self.volDiameter/self.inputParticles.get().getSamplingRate())
        self.runJob("xmipp_transform_window", "-i %s.xmd -o %s.mrcs --save_metadata_stack %s.xmd --keep_input_columns --size %d"%(
                    fnImgs,fnImgsResized,fnImgsResized, cropSize),
                    numberOfMpi=1)
        self.runJob("xmipp_image_resize", "-i %s.mrcs --fourier %d"%(fnImgsResized, self.Xdim),
                    numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())
        XdimOrig, _, _, _, _ = xmippLib.MetaDataInfo(fnImgs+".xmd")
        K=float(self.Xdim)/float(cropSize)
        self.runJob('xmipp_metadata_utilities', '-i %s.xmd --operate modify_values "shiftX=%f*shiftX"' %
                    (fnImgsResized, K), numberOfMpi=1)
        self.runJob('xmipp_metadata_utilities', '-i %s.xmd --operate modify_values "shiftY=%f*shiftY"' %
                    (fnImgsResized, K), numberOfMpi=1)

    def predict(self, gpuId):
        fnImgs = self._getTmpPath("images.xmd")
        fnImgsResized = self._getTmpPath("imagesResized.xmd")
        args = "%s %s %s %s %s %s" % (fnImgs, fnImgsResized, gpuId, self.fnModelDir, self.symmetry,
                                      self._getExtraPath('particles.xmd'))
        self.runJob("xmipp_deep_global_assignment_predict", args, numberOfMpi=1, env=self.getCondaEnv())

    def createOutputStep(self):
        fnPredict = self._getExtraPath("particles.xmd")
        outputSet = self._createSetOfParticles()
        readSetOfParticles(fnPredict, outputSet)
        outputSet.copyInfo(self.inputParticles.get())
        outputSet.setAlignmentProj()
        self._defineOutputs(outputParticles=outputSet)
        self._store(outputSet)
        self._defineSourceRelation(self.inputParticles.get(), outputSet)

    # --------------------------- INFO functions --------------------------------
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We learned a model to align particles from %i input images (%s)." \
                           % (self.inputSet.get().getSize(), self.getObjectTag('inputSet')))
        return methods

    def _validate(self):
        errors = []
        if self.modelSource==1:
            if len(glob.glob(os.path.join(self.modelDir.get(),"model*.h5")))==0:
                errors.append('The given directory does not have network model*.h5')
            if len(glob.glob(os.path.join(self.modelDir.get(), "info.txt"))) == 0:
                errors.append('The given directory does not have info.txt')
        return errors