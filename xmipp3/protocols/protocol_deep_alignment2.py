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
from pwem.protocols import ProtRefine3D
from pwem.objects import String
from pwem.emlib.image import ImageHandler
from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import math
import os
import xmipp3
from xmipp_base import XmippScript
from pyworkflow import BETA, UPDATED, NEW, PROD

class XmippProtDeepAlign2(ProtRefine3D, xmipp3.XmippProtocol):
    """Learn neural network models to align particles"""
    _label = 'deep global training'
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

        form.addParam('inputVol', PointerParam, label="Reference volume",
                      pointerClass='Volume', allowsNull=False)

        form.addParam('volDiameter', IntParam, label="Protein diameter (A)",
                      help="The diameter should be relatively tight")

        form.addParam('symmetry', StringParam,
                      label="Symmetry", default="c1",
                      help="Symmetry of the molecule")

        form.addParam('angSample', FloatParam, label="Angular sampling", default=5,
                      help="Angular sampling of the projection sphere in degrees")

        form.addSection(label="Training")

        form.addParam('Xdim', IntParam,
                      label="Image size", default=32,
                      help="Image size during the training")

        form.addParam('numModels', IntParam,
                      label="Number of models", default=5,
                      help="Multiple models will be trained independently")

        form.addParam('trainSetSize', IntParam, label="Train set size", default=100000,
                      help='How many particles should be used in the training')

        form.addParam('batchSize', IntParam,
                      label="Batch size for training",
                      default=8,
                      expertLevel=LEVEL_ADVANCED,
                      help="Batch size for training.")

        form.addParam('learningRate', FloatParam,
                      label="Learning rate",
                      default=0.001,
                      expertLevel=LEVEL_ADVANCED,
                      help="Learning rate for training.")

        form.addParam('maxShift', FloatParam,
                      label="Max. Shift (px)",
                      default=3,
                      expertLevel=LEVEL_ADVANCED)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.predictImgsFn = self._getExtraPath('predict_input_imgs.xmd')
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')
        self._insertFunctionStep("prepareData")
        self._insertFunctionStep("train", numGPU[0])

    # --------------------------- STEPS functions ---------------------------------------------------
    def prepareData(self):
        fnVol = self._getTmpPath("reference.mrc")
        ImageHandler().convert(self.inputVol.get(), fnVol)
        self.runJob("xmipp_transform_window", "-i %s --size %d"%(fnVol,
                    int(self.volDiameter.get()/self.inputVol.get().getSamplingRate())),
                    numberOfMpi=1)
        self.runJob("xmipp_image_resize", "-i %s --dim %d"%(fnVol, self.Xdim),
                    numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())

        paramContent = """# XMIPP_STAR_1 *
data_block1
_dimensions2D   '%d %d'
_projRotRange    '0 360 %d'
_projRotRandomness   even 
_projTiltRange    '0 180 %d'
_projTiltRandomness   even 
_projPsiRange    '0 0 1'
_projPsiRandomness   even 
_noiseCoord '0 0'
        """ % (self.Xdim, self.Xdim, math.ceil(360/self.angSample.get()), math.ceil(180/self.angSample.get()))
        fnParam = self._getTmpPath("projectionParameters.xmd")
        fhParam = open(fnParam, 'w')
        fhParam.write(paramContent)
        fhParam.close()

        self.runJob("xmipp_phantom_project",
                    "-i %s -o %s --method fourier 2 0.5 --params %s --sym %s" %
                    (fnVol, self._getExtraPath("reference.mrcs"), fnParam, self.symmetry), numberOfMpi=1)

        fhInfo = open(self._getExtraPath("info.txt"),'w')
        fhInfo.write("Xdim=%d\n"%self.Xdim)
        fhInfo.write("Diameter=%d\n"%self.volDiameter)
        fhInfo.write("Symmetry=%s\n"%self.symmetry)
        fhInfo.close()

    def train(self, gpuId):
        fnReference = self._getExtraPath("reference.xmd")
        args = "%s %s %f %d %d %s %d %f %s" % (
            fnReference, self._getExtraPath("model"), self.maxShift,
            self.trainSetSize, self.batchSize, gpuId, self.numModels, self.learningRate, self.symmetry)
        self.runJob(f"xmipp_deep_global_assignment", args, numberOfMpi=1, env=self.getCondaEnv())

    # --------------------------- INFO functions --------------------------------
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We learned a model to align particles from %i input images (%s)." \
                           % (self.inputSet.get().getSize(), self.getObjectTag('inputSet')))
        return methods
