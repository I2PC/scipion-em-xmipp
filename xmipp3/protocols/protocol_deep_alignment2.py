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
from pyworkflow.utils.path import cleanPattern
from pwem.protocols import ProtRefine3D
from pwem.objects import String
from pwem.emlib.image import ImageHandler
from pwem.emlib.metadata import getFirstRow
from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import math
import os
import xmipp3
import xmippLib
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

        form.addParam('inputType', EnumParam, label="Reference", choices=['Volume', 'Particles'], default=0,
                      help='If volume, projections are generated from this volume. If particles, they must have '
                           'a previous and reliable angular assignment')

        form.addParam('inputVol', PointerParam, label="Reference volume", condition='inputType==0',
                      pointerClass='Volume', allowsNull=True)

        form.addParam('angSample', FloatParam, label="Angular sampling", default=5, condition='inputType==0',
                      help="Angular sampling of the projection sphere in degrees")

        form.addParam('SNR', FloatParam, label="SNR", default=-1, condition='inputType==0',
                      help="SNR of the simulated images. -1 for no noise")

        form.addParam('inputParticles', PointerParam, label="Input particles", condition='inputType==1',
                      pointerClass='SetOfParticles', allowsNull=True, pointerCondition='hasAlignmentProj')
        form.addParam('correctCTF', BooleanParam, label='Correct CTF', condition='inputType==1', default=True,
                      help='Correct the CTF through a Wiener filter')

        form.addParam('volDiameter', IntParam, label="Protein diameter (A)",
                      help="The diameter should be relatively tight")

        form.addParam('symmetry', StringParam,
                      label="Symmetry", default="c1",
                      help="Symmetry of the molecule")


        form.addSection(label="Training")

        form.addParam('Xdim', IntParam,
                      label="Image size", default=32,
                      help="Image size during the training")

        form.addParam('numModels', IntParam,
                      label="Number of models", default=5,
                      help="Multiple models will be trained independently")

        form.addParam('trainSetSize', IntParam, label="Train set size", default=100000,
                      help='How many particles should be used in the training')

        form.addParam('modelSize', EnumParam, label="Model size", choices=['Small', 'Medium', 'Large'], default=0,
                      help='Model size (1M, ..., 19M) parameters')

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
        if self.inputType.get()==0:
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
        else:
            # TODO> no esta hecho
            pass

        fhInfo = open(self._getExtraPath("info.txt"),'w')
        fhInfo.write("Xdim=%d\n"%self.Xdim)
        fhInfo.write("Diameter=%d\n"%self.volDiameter)
        fhInfo.write("Symmetry=%s\n"%self.symmetry)
        fhInfo.close()

    def train(self, gpuId):
        fnReference = self._getExtraPath("reference.xmd")
        args = "%s %s %f %d %d %s %d %f %s %f %d" % (
            fnReference, self._getExtraPath("model"), self.maxShift,
            self.trainSetSize, self.batchSize, gpuId, self.numModels, self.learningRate, self.symmetry,
            self.SNR, self.modelSize.get())
        self.runJob(f"xmipp_deep_global_assignment", args, numberOfMpi=1, env=self.getCondaEnv())

    # --------------------------- INFO functions --------------------------------
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We learned a model to align particles from %i input images (%s)." \
                           % (self.inputSet.get().getSize(), self.getObjectTag('inputSet')))
        return methods

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
        form.addParam('correctCTF', BooleanParam, label='Correct CTF', default=True,
                      help='Correct the CTF through a Wiener filter')

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
        writeSetOfParticles(self.inputParticles.get(), fnImgs+".xmd")

        if self.correctCTF:
            fnImgsCorrected=self._getTmpPath("imagesCorrected")
            args = "-i %s.xmd -o %s.mrcs --save_metadata_stack %s.xmd --keep_input_columns" % (
                fnImgs, fnImgsCorrected, fnImgsCorrected)
            args += " --sampling_rate %f --correct_envelope" % self.inputParticles.get().getSamplingRate()
            if self.inputParticles.get().isPhaseFlipped():
                args += " --phase_flipped"
            self.runJob("xmipp_ctf_correct_wiener2d", args,
                        numberOfMpi=min(self.numberOfThreads.get() * self.numberOfMpi.get(), 24))
            fnImgs=fnImgsCorrected

        fnImgsResized = self._getTmpPath("imagesResized")
        cropSize = int(self.volDiameter/self.inputParticles.get().getSamplingRate())
        self.runJob("xmipp_transform_window", "-i %s.xmd -o %s.mrcs --save_metadata_stack %s.xmd --keep_input_columns --size %d"%(
                    fnImgs,fnImgsResized,fnImgsResized, cropSize),
                    numberOfMpi=1)
        self.runJob("xmipp_image_resize", "-i %s.mrcs --fourier %d"%(fnImgsResized, self.Xdim),
                    numberOfMpi=self.numberOfThreads.get() * self.numberOfMpi.get())
        row = getFirstRow(fnImgs+".xmd")
        hasShift = row.containsLabel(xmippLib.MDL_SHIFT_X)
        if hasShift:
            XdimOrig, _, _, _, _ = xmippLib.MetaDataInfo(fnImgs+".xmd")
            K=float(self.Xdim)/float(cropSize)
            self.runJob('xmipp_metadata_utilities', '-i %s.xmd --operate modify_values "shiftX=%f*shiftX"' %
                        (fnImgsResized, K), numberOfMpi=1)
            self.runJob('xmipp_metadata_utilities', '-i %s.xmd --operate modify_values "shiftY=%f*shiftY"' %
                        (fnImgsResized, K), numberOfMpi=1)
        if self.correctCTF:
            cleanPattern(fnImgsCorrected+"*")

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