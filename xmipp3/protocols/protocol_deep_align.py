# **************************************************************************
# *
# * Authors:     Amaya Jimenez
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
from pwem.protocols import ProtRefine3D
from xmipp3.base import writeInfoField, readInfoField
from pwem.emlib.metadata import iterRows, getFirstRow
import pwem.emlib.metadata as md
from xmipp3.convert import createItemMatrix, setXmippAttributes, readSetOfParticles
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

class XmippProtDeepAlign(ProtRefine3D, xmipp3.XmippProtocol):
    """Performs an angular assignment using deep learning"""
    _label = 'deep align'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v0.3'
    _ih = ImageHandler()
    _cond_modelPretrainTrue = 'modelPretrain==True'
    _cond_modelPretrainFalse = 'modelPretrain==False'
    _fnCorrectedParticlesStk = 'corrected_particles.stk'    
    _fnCorrectedParticlesXmd = 'corrected_particles.xmd'
    _fnVolumeVol = 'volume.vol'
    _fnConeCenterDoc = 'coneCenters.doc'
    _tempNumXmd = '%d.xmd'


    def __init__(self, **args):
        ProtRefine3D.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images",
                      pointerClass='SetOfParticles')
        form.addParam('inputVolume', PointerParam, label="Volume",
                      pointerClass='Volume')

        form.addParam('modelPretrain', BooleanParam, default=False,
                      label='Choose your if you want to use pretrained models',
                      help='Setting "yes" you can choose previously trained models. '
                           'If you choose "no" new models will be trained.')
        form.addParam('pretrainedModels', PointerParam,
                      pointerClass=self.getClassName(),
                      condition=self._cond_modelPretrainTrue,
                      label='Set pretrained models',
                      help='Choose the protocol where your models were trained. '
                           'Be careful with using proper models for your new prediction.')

        form.addParam('inputTrainSet', PointerParam, label="Input training set",
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      help='The set of particles previously aligned to be used as training set',
                      condition=self._cond_modelPretrainFalse)
        form.addParam('targetResolution', FloatParam, label="Target resolution",
                      default=3.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        form.addParam('numEpochs', IntParam,
                      label="Number of epochs for training",
                      default=10,
                      help="Number of epochs for training.",
                      condition=self._cond_modelPretrainFalse)
        form.addParam('batchSize', IntParam,
                      label="Batch size for training",
                      default=128,
                      help="Batch size for training.",
                      condition=self._cond_modelPretrainFalse)
        form.addParam('spanConesTilt', FloatParam,
                      label="Distance between region centers",
                      default=30,
                      help="Distance in degrees between region centers.",
                      condition=self._cond_modelPretrainFalse)
        form.addParam('numConesSelected', IntParam,
                      label="Number of selected regions per image",
                      default=2,
                      help="Number of selected regions per image.")
        form.addParam('applyCTF', BooleanParam, default=False,
                      label='Correct CTF',
                      help='Setting "yes" a wiener filter will be applied to correct the ctf in the input particles. ')
        form.addParam('gpuAlign', BooleanParam, label="Use GPU alignment", default=True,
                      help='Use GPU alignment algorithm to determine the final 3D alignment parameters')
        form.addParam('myMPI', IntParam, label="Xmipp MPIs", default=8,
                      help='Number of MPI to run the Xmipp programs to prepare the input images sets.')

        form.addParallelSection(threads=8, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):

        deps = []
        deps2 = []

        self.lastIter = 0
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        self.trainImgsFn = self._getExtraPath('train_input_imgs.xmd')

        firstStepId = self._insertFunctionStep("convertStep")
        firstStepId = self._insertFunctionStep("computeTrainingSet", 'projections', prerequisites=[firstStepId])

        # Trainig steps
        firstStepId = self._insertFunctionStep("prepareImagesForTraining", prerequisites=[firstStepId])

        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuList.get()

        numGPU = myStr.split(',')
        for idx, gpuId in enumerate(numGPU):
            stepId = self._insertFunctionStep("trainNClassifiers2ClassesStep", idx, gpuId, len(numGPU),
                                              prerequisites=[firstStepId])
            deps.append(stepId)

        # Predict step
        predictStepId = self._insertFunctionStep("predictStep", numGPU[0], prerequisites=deps)

        # Correlation step
        if self.gpuAlign:
            for idx, gpuId in enumerate(numGPU):
                stepId = self._insertFunctionStep("correlationCudaStep", idx, str(idx), len(numGPU),
                                                  prerequisites=[predictStepId])
                deps2.append(stepId)
        else:
            stepId = self._insertFunctionStep("correlationSignificantStep", prerequisites=[predictStepId])
            deps2.append(stepId)

        stepId = self._insertFunctionStep("createOutputMetadataStep", prerequisites=deps2)

        self._insertFunctionStep("createOutputStep", prerequisites=[stepId])

    # --------------------------- STEPS functions ---------------------------------------------------
    
    def _getProjectionsExp(self, num):
        return self._getExtraPath('projectionsExp%d.xmd' % num)

    def _getConePrediction(self):
        return self._getExtraPath('conePrediction.txt')

    def _getOutCone(self, num):
        return 'outCone%d.xmd' % num

    def _resize(self, Xdim, fnCorrected, prefix, fnTarget):
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (fnCorrected,
                            self._getExtraPath(prefix + '.stk'),
                            self._getExtraPath(prefix + '.xmd'),
                            self.newXdim), numberOfMpi=self.myMPI.get())
            moveFile(self._getExtraPath(prefix + '.xmd'), fnTarget)

    def _correctWiener(self, hasCTF, fn, Ts):
        if hasCTF and self.applyCTF.get():
            fnCorrectedStk = self._getExtraPath(self._fnCorrectedParticlesStk)
            fnCorrected = self._getExtraPath(self._fnCorrectedParticlesXmd)
            args = "-i %s -o %s --save_metadata_stack %s --keep_input_columns" % (
                fn, fnCorrectedStk, fnCorrected)
            args += " --sampling_rate %f --correct_envelope" % Ts
            if self.inputSet.get().isPhaseFlipped():
                args += " --phase_flipped"
            self.runJob("xmipp_ctf_correct_wiener2d",
                        args, numberOfMpi=self.myMPI.get())

    def _removeCorrectedParticles(self):
        if exists(self._getExtraPath(self._fnCorrectedParticlesStk)):
            remove(self._getExtraPath(self._fnCorrectedParticlesStk))
            remove(self._getExtraPath(self._fnCorrectedParticlesXmd))

    def convertStep(self):
        if self.modelPretrain.get() is True:
            fnPreProtocol = self.pretrainedModels.get()._getExtraPath()
            preXDim = readInfoField(fnPreProtocol, "size", emlib.MDL_XSIZE)
            self.inputTrainSet = self.pretrainedModels.get().inputTrainSet

        from ..convert import writeSetOfParticles
        inputParticles = self.inputSet.get()
        writeSetOfParticles(inputParticles, self.imgsFn)

        Ts = inputParticles.getSamplingRate()
        row = getFirstRow(self.imgsFn)
        hasCTF = row.containsLabel(emlib.MDL_CTF_DEFOCUSU) or emlib.containsLabel(
            emlib.MDL_CTF_MODEL)

        fnCorrected = self.imgsFn

        self._correctWiener(hasCTF, self.imgsFn, Ts)

        Xdim = inputParticles.getXDim()
        newTs = self.targetResolution.get() * 1.0 / 3.0
        newTs = max(Ts, newTs)

        self.newXdim = int(preXDim) if self.modelPretrain.get() else int(
            float(Xdim * Ts / newTs))

        self.firstMaxShift = int(round(self.newXdim / 10))
        writeInfoField(self._getExtraPath(), "sampling",
                       emlib.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", emlib.MDL_XSIZE,
                       self.newXdim)

        self._resize(Xdim, fnCorrected, 'scaled_particles', self.imgsFn)

        self._removeCorrectedParticles()

        fnVol = self._getTmpPath(self._fnVolumeVol)
        self._ih.convert(self.inputVolume.get(), fnVol)
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --fourier %d" % (fnVol, self.newXdim),
                        numberOfMpi=self.myMPI.get())

        inputTrain = self.inputTrainSet.get()
        writeSetOfParticles(inputTrain, self.trainImgsFn)
        row = getFirstRow(self.trainImgsFn)
        hasCTF = row.containsLabel(emlib.MDL_CTF_DEFOCUSU) or emlib.containsLabel(
            emlib.MDL_CTF_MODEL)

        fnCorrected = self.trainImgsFn

        self._correctWiener(hasCTF, self.trainImgsFn, Ts)

        self._resize(Xdim, fnCorrected,
                     'scaled_train_particles', self.trainImgsFn)

        self._removeCorrectedParticles()

    def generateConeCenters(self, fn):
        fnVol = self._getTmpPath(self._fnVolumeVol)
        fnCenters = self._getExtraPath(fn + ".stk")
        fnCentersMd = self._getExtraPath(fn + ".doc")
        # if self.spanConesTilt.get()>30:
        #     sampling_rate = 30
        # else:
        #     sampling_rate = self.spanConesTilt.get()

        self.runJob("xmipp_angular_project_library",
                    "-i %s -o %s --sym c1 --sampling_rate %d"
                    % (fnVol, fnCenters, self.spanConesTilt.get()),
                    numberOfMpi=self.myMPI.get())
        mdExp = emlib.MetaData(fnCentersMd)
        return mdExp.size()

    def angularDistance(self, rot, tilt, mdCones):
        # Angular distance between particle and region center
        dist = []
        for row in iterRows(mdCones):
            rotCenterCone = row.getValue(emlib.MDL_ANGLE_ROT)
            tiltCenterCone = row.getValue(emlib.MDL_ANGLE_TILT)
            if rotCenterCone < 0:
                rotCenterCone = rotCenterCone + 360
            if tiltCenterCone < 0:
                tiltCenterCone = tiltCenterCone + 360
            srot = math.sin(math.radians(rot))
            crot = math.cos(math.radians(rot))
            stilt = math.sin(math.radians(tilt))
            ctilt = math.cos(math.radians(tilt))
            srotCone = math.sin(math.radians(rotCenterCone))
            crotCone = math.cos(math.radians(rotCenterCone))
            stiltCone = math.sin(math.radians(tiltCenterCone))
            ctiltCone = math.cos(math.radians(tiltCenterCone))
            aux = (stilt * crot * stiltCone * crotCone) + (
                    stilt * srot * stiltCone * srotCone) + (
                          ctilt * ctiltCone)
            auxA = aux
            if aux < -1:
                auxA = -1
            elif aux > 1:
                auxA = 1
            auxAcos = math.degrees(math.acos(auxA))
            dist.append(auxAcos)
        minDist = min(dist)
        finalCone = dist.index(minDist) + 1
        return finalCone

    def computeTrainingSet(self, nameTrain):

        totalCones = self.generateConeCenters('coneCenters')
        self.numCones = totalCones
        fnCentersMd = self._getExtraPath(self._fnConeCenterDoc)
        mdCones = emlib.MetaData(fnCentersMd)

        auxList = []

        mdTrain = emlib.MetaData(self.trainImgsFn)
        mdList = []
        for i in range(totalCones):
            mdList.append(emlib.MetaData())
        for row in iterRows(mdTrain):
            rot = row.getValue(emlib.MDL_ANGLE_ROT)
            tilt = row.getValue(emlib.MDL_ANGLE_TILT)
            flip = row.getValue(emlib.MDL_FLIP)
            if flip:
                tilt = tilt + 180
            if rot < 0:
                rot = rot + 360
            if tilt < 0:
                tilt = tilt + 360
            numCone = self.angularDistance(rot, tilt, mdCones)
            mdCone = mdList[numCone - 1]
            auxList.append(numCone - 1)
            row.addToMd(mdCone)
        for i in range(totalCones):
            fnTrain = self._getExtraPath(nameTrain + self._tempNumXmd % (i + 1))
            mdList[i].write(fnTrain)


    def prepareImagesForTraining(self):

        fnCentersMd = self._getExtraPath(self._fnConeCenterDoc)
        mdCones = emlib.MetaData(fnCentersMd)
        span = self.spanConesTilt.get()
        counterCones = 0
        for row in iterRows(mdCones):
            rotCenter = row.getValue(emlib.MDL_ANGLE_ROT)
            tiltCenter = row.getValue(emlib.MDL_ANGLE_TILT)
            if rotCenter < 0:
                rotCenter = rotCenter + 360
            if tiltCenter < 0:
                tiltCenter = tiltCenter + 360
            iniRot = rotCenter - span
            endRot = rotCenter + span
            iniTilt = tiltCenter - span
            endTilt = tiltCenter + span
            if iniRot < 0:
                iniRot = iniRot + 360
            if iniTilt < 0:
                iniTilt = iniTilt + 360
            if endRot < 0:
                endRot = endRot + 360
            if endTilt < 0:
                endTilt = endTilt + 360
            mdProj = emlib.MetaData(self._getExtraPath('projections%d.xmd' % (counterCones + 1)))
            sizeProj = mdProj.size()
            if sizeProj > 0:
                lastLabel = counterCones + 1
                self.projectStep(300, iniRot, endRot, iniTilt, endTilt,
                                 'projectionsCudaCorr', counterCones + 1)
                if self.modelPretrain.get() is False:
                    self.generateExpImagesStep(10000, 'projections',
                                                   'projectionsExp',
                                                   counterCones + 1)
            else:
                remove(self._getExtraPath('projections%d.xmd' % (counterCones + 1)))
            counterCones = counterCones + 1

        if self.modelPretrain.get() is False:
            fnToFilter = self._getProjectionsExp(lastLabel)
            self.runJob("xmipp_transform_filter", " -i %s --fourier low_pass %f" %
                        (fnToFilter, 0.15), numberOfMpi=self.myMPI.get())

    def projectStep(self, numProj, iniRot, endRot, iniTilt, endTilt, fn, idx):

        newXdim = readInfoField(self._getExtraPath(), "size",
                                emlib.MDL_XSIZE)
        fnVol = self._getTmpPath(self._fnVolumeVol)

        uniformProjectionsStr = """
# XMIPP_STAR_1 *
data_block1
_dimensions2D   '%d %d'
_projRotRange    '%d %d %d'
_projRotRandomness   random
_projRotNoise   '0'
_projTiltRange    '%d %d 1'
_projTiltRandomness   random
_projTiltNoise   '0'
_projPsiRange    '0 0 1'
_projPsiRandomness   random
_projPsiNoise   '0'
_noisePixelLevel   '0'
_noiseCoord   '0'
""" % (newXdim, newXdim, iniRot, endRot, numProj, iniTilt, endTilt)
        fnParams = self._getExtraPath("uniformProjections%d.xmd" % idx)
        fh = open(fnParams, "w")
        fh.write(uniformProjectionsStr)
        fh.close()

        fnProjs = self._getExtraPath(fn + "%d.stk" % idx)
        self.runJob("xmipp_phantom_project",
                    "-i %s -o %s --method fourier 1 0.5 "
                    "--params %s" % (fnVol, fnProjs, fnParams), numberOfMpi=1)

        fnProjsXmd=fnProjs[:-3]+'xmd'
        self.runJob("xmipp_metadata_utilities",
                    "-i %s --fill ref lineal 1 1 " % (fnProjsXmd), numberOfMpi=1)

        cleanPattern(self._getExtraPath('uniformProjections*'))

    def generateExpImagesStep(self, Nimgs, nameProj, nameExp, label):
        fnProj = self._getExtraPath(nameProj + self._tempNumXmd % label)
        fnExp = self._getExtraPath(nameExp + self._tempNumXmd % label)
        fnLabels = self._getExtraPath('labels.txt')
        mdIn = emlib.MetaData(fnProj)
        mdExp = emlib.MetaData()
        NimgsMd = mdIn.size()
        Nrepeats = int(Nimgs / NimgsMd)
        # if Nrepeats<10:
        #     Nrepeats=10
        print("Nrepeats", Nrepeats)
        if (label == 1 and exists(fnLabels)):
            remove(fnLabels)
        fileLabels = open(fnLabels, "a")
        self._processRows(label, fnExp, mdIn, mdExp, Nrepeats, fileLabels)
        mdExp.write(fnExp)
        fileLabels.close()
        if (label - 1) > 0:
            labelPrev = -1
            for n in range(1, label):
                if exists(self._getExtraPath(nameExp + self._tempNumXmd % (label - n))):
                    labelPrev = label - n
                    break
            if labelPrev != -1:
                lastFnExp = self._getExtraPath(
                    nameExp + self._tempNumXmd % (labelPrev))
                self.runJob("xmipp_metadata_utilities",
                            " -i %s --set union %s -o %s " %
                            (lastFnExp, fnExp, fnExp), numberOfMpi=1)
        remove(fnProj)

    def _processRows(self, label, fnExp, mdIn, mdExp, Nrepeats, fileLabels):
        newXdim = readInfoField(self._getExtraPath(), "size",
                                emlib.MDL_XSIZE)
        maxPsi = 180
        maxShift = round(newXdim / 10)
        for row in iterRows(mdIn):
            fnImg = row.getValue(emlib.MDL_IMAGE)
            myRow = row
            I = emlib.Image(fnImg)
            Xdim, Ydim, _, _ = I.getDimensions()
            Xdim2 = Xdim / 2
            Ydim2 = Ydim / 2
            if Nrepeats == 0:
                myRow.addToMd(mdExp)
                idx += 1
                fileLabels.write(str(label - 1) + '\n')
            else:
                for i in range(Nrepeats):
                    psiDeg = np.random.uniform(-maxPsi, maxPsi)
                    psi = psiDeg * math.pi / 180.0
                    deltaX = np.random.uniform(-maxShift, maxShift)
                    deltaY = np.random.uniform(-maxShift, maxShift)
                    c = math.cos(psi)
                    s = math.sin(psi)
                    M = np.float32([[c, s, (1 - c) * Xdim2 - s * Ydim2 + deltaX],
                                    [-s, c, s * Xdim2 + (1 - c) * Ydim2 + deltaY]])
                    newFn = ('%06d@' % idx) + fnExp[:-3] + 'stk'
                    self._ih.applyTransform(
                        fnImg, newFn, M, (Ydim, Xdim), doWrap=True)

                    myRow.setValue(emlib.MDL_IMAGE, newFn)
                    myRow.setValue(emlib.MDL_ANGLE_PSI, psiDeg)
                    myRow.setValue(emlib.MDL_SHIFT_X, deltaX)
                    myRow.setValue(emlib.MDL_SHIFT_Y, deltaY)
                    myRow.addToMd(mdExp)
                    idx += 1
                    fileLabels.write(str(label - 1) + '\n')

    def trainNClassifiers2ClassesStep(self, thIdx, gpuId, totalGpu):

        mdNumCones = emlib.MetaData(self._getExtraPath(self._fnConeCenterDoc))
        self.numCones = mdNumCones.size()

        for i in range(self.numCones):

            idx = i + 1

            if (idx % totalGpu) != thIdx:
                continue

            modelFn = 'modelCone%d' % idx
            if self.modelPretrain.get() is True:
                if exists(self.pretrainedModels.get()._getExtraPath(modelFn + '.h5')):
                    copy(self.pretrainedModels.get()._getExtraPath(modelFn + '.h5'),
                         self._getExtraPath(modelFn + '.h5'))

            expCheck = self._getProjectionsExp(idx)
            if exists(expCheck) and not exists(self._getExtraPath(modelFn + '.h5')):
                self._coneStep(gpuId, idx, modelFn)

    def _coneStep(self, gpuId, idx, modelFn):
        fnLabels = self._getExtraPath('labels.txt')
        fileLabels = open(fnLabels, "r")
        expSet = self._getProjectionsExp(self.numCones)
        if not exists(expSet):
            for n in range(1, self.numCones):
                if exists(self._getProjectionsExp(self.numCones - n)):
                    expSet = self._getProjectionsExp(self.numCones - n)
                    break
        newFnLabels = self._getExtraPath('labels%d.txt' % idx)
        newFileLabels = open(newFnLabels, "w")
        lines = fileLabels.readlines()
        for line in lines:
            if line == str(idx - 1) + '\n':
                newFileLabels.write('1\n')
            else:
                newFileLabels.write('0\n')
        newFileLabels.close()
        fileLabels.close()

        newXdim = readInfoField(self._getExtraPath(), "size",
                                        emlib.MDL_XSIZE)
        fnLabels = self._getExtraPath('labels%d.txt' % idx)

        print("Training region ", idx, " in GPU ", gpuId)
        sys.stdout.flush()

        try:
            args = "%s %s %s %s %d %d %d %d " % (
                    expSet, fnLabels, self._getExtraPath(),
                    modelFn+'_aux', self.numEpochs, newXdim, 2, self.batchSize.get())
                    #args += " %(GPU)s"
            args += " %s " % (gpuId)
                    #args += " %s " %(int(idx % totalGpu))
            self.runJob("xmipp_cone_deepalign", args, numberOfMpi=1, env=self.getCondaEnv())
        except Exception as e:
            raise Exception(
                        "ERROR: Please, if you are suffering memory problems, "
                        "check the target resolution to work with lower dimensions.")

        moveFile(self._getExtraPath(modelFn + '_aux.h5'), self._getExtraPath(modelFn + '.h5'))

    def predictStep(self, gpuId):

        if not exists(self._getConePrediction()):
            # if self.useQueueForSteps() or self.useQueue():
            #     myStr = os.environ["CUDA_VISIBLE_DEVICES"]
            # else:
            #     myStr = self.gpuList.get()
            # numGPU = myStr.split(',')
            # numGPU = numGPU[0]
            # print("Predict", myStr, numGPU)
            # sys.stdout.flush()

            mdNumCones = emlib.MetaData(self._getExtraPath(self._fnConeCenterDoc))
            self.numCones = mdNumCones.size()

            imgsOutStk = self._getExtraPath('images_out_filtered.stk')
            imgsOutXmd = self._getExtraPath('images_out_filtered.xmd')
            self.runJob("xmipp_transform_filter", " -i %s -o %s "
                                              "--save_metadata_stack %s "
                                              "--keep_input_columns "
                                              "--fourier low_pass %f " %
                    (self.imgsFn, imgsOutStk, imgsOutXmd, 0.15), numberOfMpi=self.myMPI.get())

            numMax = int(self.numConesSelected)
            newXdim = readInfoField(self._getExtraPath(), "size",
                                emlib.MDL_XSIZE)
            args = "%s %s %d %d %d " % (imgsOutXmd, self._getExtraPath(), newXdim, self.numCones, numMax)
            #args += " %(GPU)s"
            args += " %s "%(gpuId)
            self.runJob("xmipp_cone_deepalign_predict", args, numberOfMpi=1, env=self.getCondaEnv())


    def correlationCudaStep(self, thIdx, gpuId, totalGpu):

        mdNumCones = emlib.MetaData(self._getExtraPath(self._fnConeCenterDoc))
        self.numCones = mdNumCones.size()

        # Cuda Correlation step - creating the metadata
        predCones = np.loadtxt(self._getConePrediction())
        mdConeList = []
        numMax = int(self.numConesSelected)
        for i in range(self.numCones):
            mdConeList.append(emlib.MetaData())
        mdIn = emlib.MetaData(self.imgsFn)

        for i in range(self.numCones):

            idx = i + 1

            if (idx % totalGpu) != thIdx:
                continue

            #modelFn = 'modelCone%d_aux' % idx
            #f = open(join(self._getExtraPath(), modelFn+'.txt'),'r')
            #mae = float(f.readline())
            #f.close()

            modelFn = 'modelCone%d' % idx

            positions = []
            for n in range(numMax):
                posAux = np.where(predCones[:, (n * 2) + 1] == (i + 1))
                positions = positions + (np.ndarray.tolist(posAux[0]))

            if len(positions) > 0 and exists(self._getExtraPath(modelFn + '.h5')):
                print("Classifying cone ", idx, "in GPU ", gpuId)

                for pos in positions:
                    id = pos + 1
                    row = md.Row()
                    row.readFromMd(mdIn, id)
                    row.addToMd(mdConeList[i])
                fnExpCone = self._getExtraPath('metadataCone%d.xmd' % (i + 1))
                mdConeList[i].write(fnExpCone)

                fnProjCone = self._getExtraPath('projectionsCudaCorr%d.xmd' % (i + 1))

                self.runJob("xmipp_metadata_utilities", "-i %s --fill ref lineal 1 1 " % (fnProjCone), numberOfMpi=1)

                fnOutCone = self._getOutCone(i + 1)

                if not exists(self._getExtraPath(fnOutCone)):
                    params = '  -i %s' % fnExpCone
                    params += ' -r  %s' % fnProjCone
                    params += ' -o  %s' % self._getExtraPath(fnOutCone)
                    params += ' --dev %s '%(gpuId)
                    self.runJob("xmipp_cuda_align_significant", params, numberOfMpi=1)


    def correlationSignificantStep(self):

        mdNumCones = emlib.MetaData(self._getExtraPath(self._fnConeCenterDoc))
        self.numCones = mdNumCones.size()

        # Cuda Correlation step - creating the metadata
        predCones = np.loadtxt(self._getConePrediction())
        mdConeList = []
        numMax = int(self.numConesSelected)
        for i in range(self.numCones):
            mdConeList.append(emlib.MetaData())
        mdIn = emlib.MetaData(self.imgsFn)

        for i in range(self.numCones):

            print("Classifying cone ", i + 1)
            positions = []
            for n in range(numMax):
                posAux = np.where(predCones[:, (n * 2) + 1] == (i + 1))
                positions = positions + (np.ndarray.tolist(posAux[0]))

            if len(positions) > 0:
                for pos in positions:
                    id = pos + 1
                    row = md.Row()
                    row.readFromMd(mdIn, id)
                    row.addToMd(mdConeList[i])
                fnExpCone = self._getExtraPath('metadataCone%d.xmd' % (i + 1))
                mdConeList[i].write(fnExpCone)

                fnProjCone = self._getExtraPath('projectionsCudaCorr%d.xmd' % (i + 1))
                fnOutCone = self._getOutCone(i + 1)

                if not exists(self._getExtraPath(fnOutCone)):
                    # Correlation step - calling significant program
                    args = '-i %s --initgallery %s --odir %s --dontReconstruct --useForValidation %d ' \
                           '--dontCheckMirrors --maxShift 30' % (fnExpCone, fnProjCone, self._getExtraPath(), 1)
                    self.runJob('xmipp_reconstruct_significant', args,
                                    numberOfMpi=self.myMPI.get())
                    copy(self._getExtraPath('images_significant_iter001_00.xmd'), self._getExtraPath(fnOutCone))
                    remove(self._getExtraPath('angles_iter001_00.xmd'))
                    remove(self._getExtraPath('images_significant_iter001_00.xmd'))


    def createOutputMetadataStep(self):

        mdNumCones = emlib.MetaData(self._getExtraPath(self._fnConeCenterDoc))
        self.numCones = mdNumCones.size()
        numMax = int(self.numConesSelected)
        mdIn = emlib.MetaData(self.imgsFn)
        allInFns = mdIn.getColumnValues(emlib.MDL_IMAGE)

        coneFns = []
        coneCCs = []
        mdCones = []
        shiftX = []
        shiftY = []
        fnFinal = self._getExtraPath('outConesParticles.xmd')
        for i in range(self.numCones):

            fnOutCone = self._getOutCone(i + 1)

            if exists(self._getExtraPath(fnOutCone)):             
                self._updateCone(numMax, coneFns, coneCCs, mdCones, shiftX, shiftY, fnFinal, i, fnOutCone)
            elif numMax > 1:
                    mdCones.append(None)
                    coneFns.append([])
                    coneCCs.append([])
                    shiftX.append([])
                    shiftY.append([])

        if numMax > 1:
            self._writeMDFinal(allInFns, coneFns, coneCCs, mdCones, shiftX, shiftY, fnFinal)
        else:
            mdCones = emlib.MetaData(fnFinal)
            mdCones.removeObjects(emlib.MDValueGT(emlib.MDL_SHIFT_X, 30.))
            mdCones.removeObjects(emlib.MDValueGT(emlib.MDL_SHIFT_Y, 30.))
            mdCones.removeObjects(emlib.MDValueLT(emlib.MDL_SHIFT_X, -30.))
            mdCones.removeObjects(emlib.MDValueLT(emlib.MDL_SHIFT_Y, -30.))
            mdCones.write(fnFinal)

    def _updateCone(self, numMax, coneFns, coneCCs, mdCones, shiftX, shiftY, fnFinal, i, fnOutCone):
        if numMax == 1:
            if not exists(fnFinal):
                copy(self._getExtraPath(fnOutCone), fnFinal)
            else:
                params = ' -i %s --set union %s -o %s' % (fnFinal, self._getExtraPath(fnOutCone),
                                                                  fnFinal)
                self.runJob("xmipp_metadata_utilities", params,
                                    numberOfMpi=1)
        else:
            mdCones.append(emlib.MetaData(self._getExtraPath(fnOutCone)))
            coneFns.append(mdCones[i].getColumnValues(emlib.MDL_IMAGE))
            shiftX.append(mdCones[i].getColumnValues(emlib.MDL_SHIFT_X))
            shiftY.append(mdCones[i].getColumnValues(emlib.MDL_SHIFT_Y))
            coneCCs.append(mdCones[i].getColumnValues(emlib.MDL_MAXCC))

    def _writeMDFinal(self, allInFns, coneFns, coneCCs, mdCones, shiftX, shiftY, fnFinal):
        mdFinal = emlib.MetaData()
        row = md.Row()
        for myFn in allInFns:
            myCCs = []
            myCones = []
            myPos = []
            for n in range(self.numCones):
                if myFn in coneFns[n]:
                    pos = coneFns[n].index(myFn)
                    myPos.append(pos)
                    if abs(shiftX[n][pos])<30 and abs(shiftY[n][pos])<30:
                        myCCs.append(coneCCs[n][pos])
                    else:
                        myCCs.append(0)
                    myCones.append(n + 1)
            if len(myPos) > 0:
                if max(myCCs)==0:
                    continue
                coneMax = myCones[myCCs.index(max(myCCs))]
                objId = myPos[myCCs.index(max(myCCs))] + 1
                row.readFromMd(mdCones[coneMax - 1], objId)
                row.addToMd(mdFinal)
        mdFinal.write(fnFinal)

    def createOutputStep(self):

        cleanPattern(self._getExtraPath('*.stk'))
        cleanPattern(self._getExtraPath('projectionsCudaCorr*'))

        inputParticles = self.inputSet.get()
        fnOutputParticles = self._getExtraPath('outConesParticles.xmd')

        outputSetOfParticles = self._createSetOfParticles()
        outputSetOfParticles.copyInfo(inputParticles)
        outputSetOfParticles.setAlignmentProj()

        Xdim = inputParticles.getXDim()
        newXdim = readInfoField(self._getExtraPath(), "size",
                                emlib.MDL_XSIZE)
        Ts = readInfoField(self._getExtraPath(), "sampling",
                           emlib.MDL_SAMPLINGRATE)
        if newXdim != Xdim:
            self.scaleFactor = Ts / inputParticles.getSamplingRate()
            self.iterMd = md.iterRows(fnOutputParticles, emlib.MDL_ITEM_ID)
            self.lastRow = next(self.iterMd)
            outputSetOfParticles.copyItems(inputParticles,
                                           updateItemCallback=self._updateItem)
        else:
            readSetOfParticles(fnOutputParticles, outputSetOfParticles)
        self._defineOutputs(outputParticles=outputSetOfParticles)

    def _updateItem(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(
                emlib.MDL_ITEM_ID):
            count += 1
            if count:
                self._createItemMatrix(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
        particle._appendItem = count > 0

    def _createItemMatrix(self, particle, row):

        row.setValue(emlib.MDL_SHIFT_X,
                     row.getValue(emlib.MDL_SHIFT_X) * self.scaleFactor)
        row.setValue(emlib.MDL_SHIFT_Y,
                     row.getValue(emlib.MDL_SHIFT_Y) * self.scaleFactor)
        setXmippAttributes(particle, row, emlib.MDL_SHIFT_X,
                           emlib.MDL_SHIFT_Y,
                           emlib.MDL_ANGLE_ROT, emlib.MDL_ANGLE_TILT,
                           emlib.MDL_ANGLE_PSI)
        createItemMatrix(particle, row, align=ALIGN_PROJ)

    # --------------------------- INFO functions --------------------------------
    def _summary(self):
        summary = []
        summary.append("Images evaluated: %i" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        return summary

    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We classify %i input images %s regarding to volume %s." \
                % (self.inputSet.get().getSize(), self.getObjectTag('inputSet'),
                   self.getObjectTag('inputVolume')))
        return methods

    def _validate(self):
        errors = []
        if self.numberOfMpi>1:
            errors.append("You must select Threads to make the parallelization in Scipion level. "
                          "To parallelize the Xmipp program use MPIs in the form.")
        if self.spanConesTilt.get()>30:
            errors.append("The distance between region centers should be lower than 31 degress.")
        return errors


