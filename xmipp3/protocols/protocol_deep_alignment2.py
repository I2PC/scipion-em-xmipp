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
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam, EnumParam,
                                        IntParam, BooleanParam, FileParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils import Message
from pyworkflow.utils.path import cleanPattern, createLink
from pwem.protocols import ProtRefine3D
from pwem.objects import String, Integer
from pwem.emlib.image import ImageHandler
from pwem.emlib.metadata import getFirstRow
from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import math
import os
import xmipp3
import xmippLib
from pyworkflow import BETA, UPDATED, NEW, PROD

class XmippProtDeepAlign2Base(ProtRefine3D, xmipp3.XmippProtocol):
    _devStatus = BETA
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v1.0'

    def __init__(self, **args):
        ProtRefine3D.__init__(self, **args)

    def getCropSize(self):
        if isinstance(self.volDiameter,Integer):
            cropSize = int(self.volDiameter.get() / self.inputParticles.get().getSamplingRate())
        else:
            cropSize = int(self.volDiameter / self.inputParticles.get().getSamplingRate())
        return cropSize

    def prepareImages(self, gpuId):
        fnImgs = self._getTmpPath("images")
        writeSetOfParticles(self.inputParticles.get(), fnImgs + ".xmd")

        if self.correctCTF:
            fnImgsCorrected = self._getTmpPath("imagesCorrected")
            args = "-i %s.xmd -o %s.mrcs --save_metadata_stack %s.xmd --keep_input_columns" % (
                fnImgs, fnImgsCorrected, fnImgsCorrected)
            args += " --sampling_rate %f --correct_envelope" % self.inputParticles.get().getSamplingRate()
            if self.inputParticles.get().isPhaseFlipped():
                args += " --phase_flipped"
            self.runJob("xmipp_ctf_correct_wiener2d", args,
                        numberOfMpi=min(self.numberOfThreads.get() * self.numberOfMpi.get(), 24))
            fnImgs = fnImgsCorrected

        cropSize = self.getCropSize()
        fnImgsCropped= self._getTmpPath("imagesCropped")
        # Only for cropping
        self.runJob("xmipp_transform_crop_resize_gpu",
                    "-i %s.xmd --oroot %s --cropSize %d --finalSize %d --gpu %s" % (
                        fnImgs, fnImgsCropped, cropSize, cropSize, gpuId), numberOfMpi=1, env=self.getCondaEnv())

        # Downsampling in Fourier, which is much more precise
        fnImgsResized=self._getTmpPath("imagesCroppedResized")
        self.runJob("xmipp_image_resize",
                    "-i %s.xmd -o %s.mrcs --fourier %d --save_metadata_stack --keep_input_columns"%\
                        (fnImgsCropped, fnImgsResized, self.Xdim),
                    numberOfMpi=min(self.numberOfThreads.get() * self.numberOfMpi.get(), 24))

        if self.correctCTF:
            cleanPattern(fnImgsCorrected + "*")


class XmippProtDeepAlign2(XmippProtDeepAlign2Base):
    """Learn neural network models to align particles.

    If the loss function starts diverging or does not converge to a high precision,
    consider lowering the learning rate"""
    _label = 'deep global training'

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
                      pointerClass='SetOfParticles', allowsNull=True, pointerCondition='hasAlignmentProj')
        form.addParam('correctCTF', BooleanParam, label='Correct CTF', default=True,
                      help='Correct the CTF through a Wiener filter')

        form.addParam('volDiameter', IntParam, label="Protein diameter (A)",
                      help="The diameter should be relatively tight")

        form.addSection(label="Training")

        form.addParam('Xdim', IntParam,
                      label="Image size", default=32,
                      help="Image size during the training")

        form.addParam('numModels', IntParam,
                      label="Number of models", default=5,
                      help="Multiple models will be trained independently")

        form.addParam('maxEpochsShift', IntParam, label="Max. Epochs Shift", default=300, expertLevel=LEVEL_ADVANCED,
                      help='Number of epochs for the shift models')

        form.addParam('maxEpochsAngle', IntParam, label="Max. Epochs Angle", default=300, expertLevel=LEVEL_ADVANCED,
                      help='Number of epochs for the angular models')

        form.addParam('NImgs', IntParam,
                      label="Number of images for training",
                      default=1000,
                      expertLevel=LEVEL_ADVANCED,
                      help="Each model is trained on a random subset of images of this size. If the input set is "
                           "smaller, then each model is trained on all images.")

        form.addParam('batchSize', IntParam,
                      label="Batch size for training",
                      default=8,
                      expertLevel=LEVEL_ADVANCED,
                      help="Batch size for training.")

        form.addParam('learningRate', FloatParam,
                      label="Learning rate",
                      default=0.0001,
                      expertLevel=LEVEL_ADVANCED,
                      help="Learning rate for training.")

        form.addParam('shiftPrecision', FloatParam,
                      label="Shift precision (px)",
                      default=0.5)

        form.addParam('anglePrecision', FloatParam,
                      label="Angular precision (deg)",
                      default=3)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.predictImgsFn = self._getExtraPath('predict_input_imgs.xmd')
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()
        numGPU = myStr.split(',')
        self._insertFunctionStep("prepareData", numGPU[0])
        self._insertFunctionStep("train", numGPU[0])

    # --------------------------- STEPS functions ---------------------------------------------------
    def prepareData(self, gpuId):
        self.prepareImages(gpuId)

        fhInfo = open(self._getExtraPath("info.txt"),'w')
        fhInfo.write("Xdim=%d\n"%self.Xdim)
        fhInfo.write("Diameter=%d\n"%self.volDiameter)
        fhInfo.close()

    def train(self, gpuId):
        fnReference = self._getTmpPath("imagesCropped.xmd")
        args = "-i %s --oroot %s --maxEpochs %d --batchSize %d --gpu %s --Nmodels %d --learningRate %f "\
               "--precision %f --Nimgs %d --mode shift" % (
            fnReference, self._getExtraPath("modelShift"), self.maxEpochsShift, 128, gpuId, self.numModels, 0.0001,
            self.shiftPrecision, self.NImgs)
        self.runJob("xmipp_deep_global_assignment", args, numberOfMpi=1, env=self.getCondaEnv())

        args = "-i %s --gpu %s --modelDir %s --mode shift" %\
               (fnReference, gpuId, self._getExtraPath("modelShift"))
        self.runJob("xmipp_deep_global_assignment_predict", args, numberOfMpi=1, env=self.getCondaEnv())
        md = xmippLib.MetaData(fnReference)
        shiftX = md.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = md.getColumnValues(xmippLib.MDL_SHIFT_Y)

        K = float(self.Xdim) / float(self.getCropSize())
        fnReference = self._getTmpPath("imagesCroppedResized.xmd")
        mdResized = xmippLib.MetaData(fnReference)
        mdResized.setColumnValues(xmippLib.MDL_SHIFT_X, [K*x for x in shiftX])
        mdResized.setColumnValues(xmippLib.MDL_SHIFT_Y, [K*y for y in shiftY])
        mdResized.write(fnReference)

        args = "-i %s --oroot %s --maxEpochs %d --batchSize %d --gpu %s --Nmodels %d --learningRate %f " \
               "--precision %f --Nimgs %d --mode angles --numThreads %d" % (
                   fnReference, self._getExtraPath("modelAngles"), self.maxEpochsAngle, self.batchSize, gpuId,
                   self.numModels, self.learningRate, self.anglePrecision, self.NImgs,
                   self.numberOfThreads.get() * self.numberOfMpi.get())
        self.runJob("xmipp_deep_global_assignment", args, numberOfMpi=1, env=self.getCondaEnv())

    # --------------------------- INFO functions --------------------------------
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We learned a model to align particles from %i input images (%s)." \
                           % (self.inputSet.get().getSize(), self.getObjectTag('inputSet')))
        return methods

class XmippProtDeepAlign2Predict(XmippProtDeepAlign2Base):
    """Apply neural network models to align particles"""
    _label = 'deep global predict'

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
        form.addParam('correctCTF', BooleanParam, label='Correct CTF', default=True,
                      help='Correct the CTF through a Wiener filter')
        form.addParam('symmetry', StringParam, label="Symmetry", default="c1")

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
        self._insertFunctionStep("prepareImages", numGPU[0])
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
        fhInfo.close()

    def predict(self, gpuId):
        K = float(self.Xdim) / float(self.getCropSize())

        fnImages = self._getTmpPath("imagesCropped.xmd")
        args = "-i %s --gpu %s --modelDir %s --mode shift" %\
               (fnImages, gpuId, os.path.join(self.fnModelDir,"modelShift"))
        self.runJob("xmipp_deep_global_assignment_predict", args, numberOfMpi=1, env=self.getCondaEnv())
        md = xmippLib.MetaData(fnImages)
        shiftX = md.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = md.getColumnValues(xmippLib.MDL_SHIFT_Y)

        fnImgsResized = self._getTmpPath("imagesCroppedResized.xmd")
        mdResized = xmippLib.MetaData(fnImgsResized)
        mdResized.setColumnValues(xmippLib.MDL_SHIFT_X, [K*x for x in shiftX])
        mdResized.setColumnValues(xmippLib.MDL_SHIFT_Y, [K*y for y in shiftY])
        mdResized.write(fnImgsResized)

        args = "-i %s --gpu %s --modelDir %s --mode angles --sym %s" %\
               (fnImgsResized, gpuId, os.path.join(self.fnModelDir,"modelAngles"), self.symmetry)
        self.runJob("xmipp_deep_global_assignment_predict", args, numberOfMpi=1, env=self.getCondaEnv())
        mdResized = xmippLib.MetaData(fnImgsResized)
        rot = mdResized.getColumnValues(xmippLib.MDL_ANGLE_ROT)
        tilt = mdResized.getColumnValues(xmippLib.MDL_ANGLE_TILT)
        psi = mdResized.getColumnValues(xmippLib.MDL_ANGLE_PSI)

        fnImages = self._getTmpPath("images.xmd")
        md = xmippLib.MetaData(fnImages)
        md.setColumnValues(xmippLib.MDL_SHIFT_X, shiftX)
        md.setColumnValues(xmippLib.MDL_SHIFT_Y, shiftY)
        md.setColumnValues(xmippLib.MDL_ANGLE_ROT, rot)
        md.setColumnValues(xmippLib.MDL_ANGLE_TILT, tilt)
        md.setColumnValues(xmippLib.MDL_ANGLE_PSI, psi)
        md.write(self._getExtraPath('particles.xmd'))

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
            if len(glob.glob(os.path.join(self.modelDir.get(),"model*.tf")))==0:
                errors.append('The given directory does not have network model*.tf')
            if len(glob.glob(os.path.join(self.modelDir.get(), "info.txt"))) == 0:
                errors.append('The given directory does not have info.txt')
        return errors