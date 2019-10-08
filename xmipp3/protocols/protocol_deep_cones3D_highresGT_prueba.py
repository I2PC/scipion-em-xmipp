# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano
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


from pyworkflow import VERSION_1_2
from pyworkflow.protocol.params import PointerParam, StringParam, FloatParam, \
    IntParam, BooleanParam, GPU_LIST, STEPS_PARALLEL
from pyworkflow.utils.path import moveFile, cleanPattern
from pyworkflow.em.protocol import ProtRefine3D
from xmipp3.utils import writeInfoField, readInfoField
from pyworkflow.em.metadata.utils import iterRows
import pyworkflow.em.metadata as md
from xmipp3.convert import createItemMatrix, setXmippAttributes, rowToAlignment, \
    readSetOfParticles, geometryFromMatrix
import pyworkflow.em as em
from pyworkflow.protocol.constants import LEVEL_ADVANCED
import xmippLib
import os
import sys
import numpy as np
import math
import cv2
from shutil import copy
from os import remove
from os.path import exists

class XmippProtDeepCones3DGT_2(ProtRefine3D):
    """Performs a fast and approximate angular assignment that can be further refined
    with Xmipp highres local refinement"""
    _label = 'deep cones3D highres GT 2'
    _lastUpdateVersion = VERSION_1_2

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

        form.addParam('modelPretrain', BooleanParam, default=False,
                      label='Choose your if you want to use pretrained models',
                      help='Setting "yes" you can choose previously trained models. '
                           'If you choose "no" new models will be trained.')
        form.addParam('pretrainedModels', PointerParam,
                      pointerClass=self.getClassName(),
                      condition='modelPretrain==True',
                      label='Set pretrained models',
                      help='Choose the protocol where your models were trained. '
                           'Be careful with using proper models for your new prediction.')

        form.addParam('inputVolume', PointerParam, label="Volume",
                      pointerClass='Volume',
                      condition='modelPretrain==False')
        form.addParam('inputTrainSet', PointerParam, label="Input training set",
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      help='The set of particles previously aligned to be used as training set',
                      condition='modelPretrain==False')
        form.addParam('targetResolution', FloatParam, label="Target resolution",
                      default=3.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='See http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry for a description of the symmetry groups format'
                           'If no symmetry is present, give c1')
        form.addParam('numEpochs', IntParam,
                      label="Number of epochs for training",
                      default=10,
                      help="Number of epochs for training.",
                      condition='modelPretrain==False')
        form.addParam('spanConesTilt', FloatParam,
                      label="Distance between cone centers",
                      default=30,
                      help="Distance in degrees between cone centers.",
                      condition='modelPretrain==False')
        form.addParam('numConesSelected', IntParam,
                      label="Number of selected cones per image",
                      default=1,
                      help="Number of selected cones per image.")
        form.addParam('gpuAlign', BooleanParam, label="Use GPU alignment", default=True,
                      help='Use GPU alignment algorithm to determine the final 3D alignment parameters')
        form.addParam('myMPI', IntParam, label="XMipp MPIs", default=8,
                      help='Number of MPI to run the Xmipp protocols.')

        form.addParallelSection(threads=8, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):

        deps = []
        deps2 = []

        self.lastIter = 0
        self.batchSize = 128  # 1024
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        self.trainImgsFn = self._getExtraPath('train_input_imgs.xmd')

        firstStepId = self._insertFunctionStep("convertStep")
        firstStepId = self._insertFunctionStep("computeTrainingSet", 'projections', prerequisites=[firstStepId])

        # Trainig steps
        firstStepId = self._insertFunctionStep("prepareImagesForTraining", prerequisites=[firstStepId])


        #myStr = self.gpuList.get()
        #import os
        #print('os.environ["CUDA_VISIBLE_DEVICES"]',os.environ["CUDA_VISIBLE_DEVICES"])
        #myStr = " "
        if self.useQueueForSteps() or self.useQueue():
            myStr = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            myStr = self.gpuList.get()

        print("AAAAA", myStr)
        numGPU = myStr.split(',')
        print("GPUUUUU", myStr, numGPU)
        for idx, gpuId in enumerate(numGPU):
            print("Bucle GPUUUUUUU", idx, gpuId)
            stepId = self._insertFunctionStep("trainNClassifiers2ClassesStep", idx, gpuId, len(numGPU), prerequisites=[firstStepId])
            deps.append(stepId)

        # self._insertFunctionStep("trainOneClassifierNClassesStep")
        # Predict step
        predictStepId = self._insertFunctionStep("predictStep", numGPU[0], prerequisites=deps)

        # Correlation step
        if self.gpuAlign:
            for idx, gpuId in enumerate(numGPU):
                stepId = self._insertFunctionStep("correlationCudaStep", idx, gpuId, len(numGPU), prerequisites=[predictStepId])
                deps2.append(stepId)
        else:
            stepId = self._insertFunctionStep("correlationSignificantStep", prerequisites=[predictStepId])
            deps2.append(stepId)

        stepId = self._insertFunctionStep("createOutputMetadataStep", prerequisites=deps2)

        self._insertFunctionStep("createOutputStep", prerequisites=[stepId])

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):

        if self.modelPretrain.get() is True:
            fnPreProtocol = self.pretrainedModels.get()._getExtraPath()
            preXDim = readInfoField(fnPreProtocol, "size", xmippLib.MDL_XSIZE)

        from ..convert import writeSetOfParticles
        inputParticles = self.inputSet.get()
        writeSetOfParticles(inputParticles, self.imgsFn)
        Xdim = inputParticles.getXDim()
        Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0 / 3.0
        newTs = max(Ts, newTs)
        if self.modelPretrain.get() is False:
            self.newXdim = long(Xdim * Ts / newTs)
        else:
            self.newXdim = preXDim
        self.firstMaxShift = round(self.newXdim / 10)
        writeInfoField(self._getExtraPath(), "sampling",
                       xmippLib.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE,
                       self.newXdim)
        writeInfoField(self._getExtraPath(), "shift", xmippLib.MDL_SHIFT_X,
                       self.firstMaxShift)
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (self.imgsFn,
                         self._getExtraPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         self.newXdim), numberOfMpi=self.myMPI.get())
            moveFile(self._getExtraPath('scaled_particles.xmd'), self.imgsFn)

        from pyworkflow.em.convert import ImageHandler
        ih = ImageHandler()
        fnVol = self._getTmpPath("volume.vol")
        ih.convert(self.inputVolume.get(), fnVol)
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --fourier %d" % (fnVol, self.newXdim),
                        numberOfMpi=self.myMPI.get())

        #if self.modelPretrain.get() is False:
        inputTrain = self.inputTrainSet.get()
        writeSetOfParticles(inputTrain, self.trainImgsFn)
        Xdim = inputTrain.getXDim()
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (self.trainImgsFn,
                         self._getExtraPath('scaled_train_particles.stk'),
                         self._getExtraPath('scaled_train_particles.xmd'),
                         self.newXdim), numberOfMpi=self.myMPI.get())
            moveFile(self._getExtraPath('scaled_train_particles.xmd'),
                     self.trainImgsFn)

    def generateConeCenters(self, fn):
        fnVol = self._getTmpPath("volume.vol")
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
        mdExp = xmippLib.MetaData(fnCentersMd)
        return mdExp.size()

    def angularDistance(self, rot, tilt, mdCones):
        # AJ aqui calcular la distancia angular entre particula y centro de cada cono
        dist = []
        for row in iterRows(mdCones):
            rotCenterCone = row.getValue(xmippLib.MDL_ANGLE_ROT)
            tiltCenterCone = row.getValue(xmippLib.MDL_ANGLE_TILT)
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
            # print("angular distance:", rotCenterCone, tiltCenterCone, rot, tilt, aux, auxA, auxAcos)
            # print("aux:", (stilt * crot * stiltCone * crotCone), (stilt * srot * stiltCone * srotCone), (ctilt * ctiltCone))
        minDist = min(dist)
        finalCone = dist.index(minDist) + 1
        # print("angular distance final:", minDist, finalCone)

        return finalCone

    def computeTrainingSet(self, nameTrain):

        totalCones = self.generateConeCenters('coneCenters')
        self.numCones = totalCones
        fnCentersMd = self._getExtraPath("coneCenters.doc")
        mdCones = xmippLib.MetaData(fnCentersMd)

        auxList = []

        mdTrain = xmippLib.MetaData(self.trainImgsFn)
        mdList = []
        for i in range(totalCones):
            mdList.append(xmippLib.MetaData())
        for row in iterRows(mdTrain):
            rot = row.getValue(xmippLib.MDL_ANGLE_ROT)
            tilt = row.getValue(xmippLib.MDL_ANGLE_TILT)
            flip = row.getValue(xmippLib.MDL_FLIP)
            #AJ FLIP
            if flip:
                tilt=tilt+180

            if rot < 0:
                rot = rot + 360
            if tilt < 0:
                tilt = tilt + 360
            numCone = self.angularDistance(rot, tilt, mdCones)
            mdCone = mdList[numCone - 1]
            auxList.append(numCone - 1)
            row.addToMd(mdCone)
        print("SELECTED CONES", auxList)
        for i in range(totalCones):
            fnTrain = self._getExtraPath(nameTrain + "%d.xmd" % (i + 1))
            mdList[i].write(fnTrain)


        # #################################
        # mdOut = xmippLib.MetaData()
        # for i in auxList:
        #     for row in iterRows(mdCones):
        #         idCone = row.getValue(xmippLib.MDL_REF)
        #         if (i) == idCone:
        #             row.addToMd(mdOut)
        #             break
        # # mdPart = xmippLib.MetaData(self.trainImgsFn)
        # # for row in iterRows(mdPart):
        # #     row.addToMd(mdOut)
        # mdOut.write(self._getExtraPath('VamosAVer.xmd'))
        #
        # outputSetOfParticles = self._createSetOfParticles()
        # outputSetOfParticles.copyInfo(self.inputSet.get())
        # outputSetOfParticles.setAlignmentProj()
        # readSetOfParticles(self._getExtraPath('VamosAVer.xmd'), outputSetOfParticles)
        # self._defineOutputs(outputParticles=outputSetOfParticles)
        # aaaaaaaaaaaaaaaaa

    def prepareImagesForTraining(self):

        fnCentersMd = self._getExtraPath("coneCenters.doc")
        mdCones = xmippLib.MetaData(fnCentersMd)
        span = self.spanConesTilt.get()
        counterCones = 0
        for row in iterRows(mdCones):
            rotCenter = row.getValue(xmippLib.MDL_ANGLE_ROT)
            tiltCenter = row.getValue(xmippLib.MDL_ANGLE_TILT)
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
            mdProj = xmippLib.MetaData(self._getExtraPath('projections%d.xmd' % (counterCones + 1)))
            sizeProj = mdProj.size()
            if sizeProj > 0:
                lastLabel = counterCones + 1
                self.projectStep(300, iniRot, endRot, iniTilt, endTilt,
                                 'projectionsCudaCorr', counterCones + 1)
                if self.modelPretrain.get() is False:
                    self.generateExpImagesStep(10000, 'projections',
                                                   'projectionsExp',
                                                   counterCones + 1, False)
                    # AJ posiblemente con alrededor de 8000 podria valer...
            else: #AJ to check en general y con modelPretrain
                remove(self._getExtraPath('projections%d.xmd' % (counterCones + 1)))
            counterCones = counterCones + 1

        if self.modelPretrain.get() is False:
            fnToFilter = self._getExtraPath('projectionsExp%d.xmd' % (lastLabel))
            self.runJob("xmipp_transform_filter", " -i %s --fourier low_pass %f" %
                        (fnToFilter, 0.15), numberOfMpi=self.myMPI.get())

    def projectStep(self, numProj, iniRot, endRot, iniTilt, endTilt, fn, idx):

        newXdim = readInfoField(self._getExtraPath(), "size",
                                xmippLib.MDL_XSIZE)
        fnVol = self._getTmpPath("volume.vol")

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

        cleanPattern(self._getExtraPath('uniformProjections*'))

    def generateExpImagesStep(self, Nimgs, nameProj, nameExp, label, boolNoise):

        newXdim = readInfoField(self._getExtraPath(), "size",
                                xmippLib.MDL_XSIZE)
        fnProj = self._getExtraPath(nameProj + "%d.xmd" % label)
        fnExp = self._getExtraPath(nameExp + "%d.xmd" % label)
        fnLabels = self._getExtraPath('labels.txt')
        mdIn = xmippLib.MetaData(fnProj)
        mdExp = xmippLib.MetaData()
        newImage = xmippLib.Image()
        maxPsi = 180
        maxShift = round(newXdim / 10)
        idx = 1
        NimgsMd = mdIn.size()
        Nrepeats = int(Nimgs / NimgsMd)
        # if Nrepeats<10:
        #     Nrepeats=10
        print("Nrepeats", Nrepeats)
        if (label == 1 and exists(fnLabels)):
            remove(fnLabels)
        fileLabels = open(fnLabels, "a")
        for row in iterRows(mdIn):
            fnImg = row.getValue(xmippLib.MDL_IMAGE)
            myRow = row
            I = xmippLib.Image(fnImg)
            Xdim, Ydim, _, _ = I.getDimensions()
            Xdim2 = Xdim / 2
            Ydim2 = Ydim / 2
            if Nrepeats==0:
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
                    newImg = cv2.warpAffine(I.getData(), M, (Xdim, Ydim),
                                            borderMode=cv2.BORDER_REFLECT_101)
                    # AAAAJJJJJJ cuidado con el borderMode del warpAffine
                    if boolNoise:
                        newImg = newImg + np.random.normal(0.0, 5.0, [Xdim,
                                                                      Xdim])  # AJ 2.0 antes
                    newFn = ('%06d@' % idx) + fnExp[:-3] + 'stk'
                    newImage.setData(newImg)
                    newImage.write(newFn)
                    myRow.setValue(xmippLib.MDL_IMAGE, newFn)
                    myRow.setValue(xmippLib.MDL_ANGLE_PSI, psiDeg)
                    myRow.setValue(xmippLib.MDL_SHIFT_X, deltaX)
                    myRow.setValue(xmippLib.MDL_SHIFT_Y, deltaY)
                    myRow.addToMd(mdExp)
                    idx += 1
                    fileLabels.write(str(label - 1) + '\n')
        mdExp.write(fnExp)
        fileLabels.close()
        if (label - 1) > 0:
            labelPrev = -1
            for n in range(1, label):
                if exists(self._getExtraPath(nameExp + "%d.xmd" % (label - n))):
                    labelPrev = label - n
                    break
            if labelPrev is not -1:
                lastFnExp = self._getExtraPath(nameExp + "%d.xmd" % (labelPrev))
                self.runJob("xmipp_metadata_utilities",
                            " -i %s --set union %s -o %s " %
                            (lastFnExp, fnExp, fnExp), numberOfMpi=1)
        remove(fnProj)

    def trainNClassifiers2ClassesStep(self, thIdx, gpuId, totalGpu):

        mdNumCones = xmippLib.MetaData(self._getExtraPath("coneCenters.doc"))
        self.numCones = mdNumCones.size()

        for i in range(self.numCones):

            idx = i + 1

            if (idx % totalGpu) != thIdx:
                continue

            print("TRAINING GPU:", thIdx, gpuId, idx, totalGpu)
            sys.stdout.flush()

            modelFn = 'modelCone%d' % idx
            if self.modelPretrain.get() is True:
                if exists(self.pretrainedModels.get()._getExtraPath(modelFn + '.h5')):
                    copy(self.pretrainedModels.get()._getExtraPath(modelFn + '.h5'),
                         self._getExtraPath(modelFn + '.h5'))

            expCheck = self._getExtraPath('projectionsExp%d.xmd' % idx)
            if exists(expCheck):
                if not exists(self._getExtraPath(modelFn + '.h5')):
                    # AJ esto puede ser peligroso con modelos a medias de entrenamiento
                    fnLabels = self._getExtraPath('labels.txt')
                    fileLabels = open(fnLabels, "r")
                    expSet = self._getExtraPath('projectionsExp%d.xmd' % self.numCones)
                    if not exists(expSet):
                        for n in range(1, self.numCones):
                            if exists(self._getExtraPath('projectionsExp%d.xmd' % (self.numCones - n))):
                                expSet = self._getExtraPath('projectionsExp%d.xmd' % (self.numCones - n))
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
                                            xmippLib.MDL_XSIZE)
                    fnLabels = self._getExtraPath('labels%d.txt' % idx)

                    try:
                        args = "%s %s %s %s %d %d %d %d " % (
                        expSet, fnLabels, self._getExtraPath(),
                        modelFn, self.numEpochs, newXdim, 2, self.batchSize)
                        #args += " %(GPU)s"
                        #args += " %s " % (gpuId)
                        args += " %s " %(int(idx % totalGpu))
                        print("2 ARGS", args)
                        self.runJob("xmipp_cone_deepalign", args, numberOfMpi=1)
                    except Exception as e:
                        raise Exception(
                            "ERROR: Please, if you are having memory problems, "
                            "check the target resolution to work with lower dimensions.")

                # remove(expSet)

    def predictStep(self, gpuId):

        # mdNumCones = xmippLib.MetaData(self._getExtraPath("coneCenters.doc"))
        # self.numCones = mdNumCones.size()

        if not exists(self._getExtraPath('conePrediction.txt')):
            # if self.useQueueForSteps() or self.useQueue():
            #     myStr = os.environ["CUDA_VISIBLE_DEVICES"]
            # else:
            #     myStr = self.gpuList.get()
            # numGPU = myStr.split(',')
            # numGPU = numGPU[0]
            # print("Predict", myStr, numGPU)
            # sys.stdout.flush()

            mdNumCones = xmippLib.MetaData(self._getExtraPath("coneCenters.doc"))
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
                                xmippLib.MDL_XSIZE)
            args = "%s %s %d %d %d " % (imgsOutXmd, self._getExtraPath(), newXdim, self.numCones, numMax)
            args += " %(GPU)s"
            #AJ dejar que se envie el trabajo de prediccion a todas las GPUs disponibles??
            #args += " %s "%(gpuId)
            self.runJob("xmipp_cone_deepalign_predict", args, numberOfMpi=1)


    def correlationCudaStep(self, thIdx, gpuId, totalGpu):

        mdNumCones = xmippLib.MetaData(self._getExtraPath("coneCenters.doc"))
        self.numCones = mdNumCones.size()

        # Cuda Correlation step - creating the metadata
        predCones = np.loadtxt(self._getExtraPath('conePrediction.txt'))
        mdConeList = []
        numMax = int(self.numConesSelected)
        for i in range(self.numCones):
            mdConeList.append(xmippLib.MetaData())
        mdIn = xmippLib.MetaData(self.imgsFn)
        #allInFns = mdIn.getColumnValues(xmippLib.MDL_IMAGE)

        for i in range(self.numCones):

            idx = i + 1

            if (idx % totalGpu) != thIdx:
                continue

            print("CORRELATION GPU:", thIdx, gpuId, idx, totalGpu)
            sys.stdout.flush()

            print("Classifying cone ", i + 1)
            positions = []
            for n in range(numMax):
                posAux = np.where(predCones[:, (n * 2) + 1] == (i + 1))
                positions = positions + (np.ndarray.tolist(posAux[0]))
                # print(posAux, positions, len(positions))

            if len(positions) > 0:
                for pos in positions:
                    # print(pos)
                    # imageName = allInFns[pos]
                    # cone = (i+1)
                    id = pos + 1  # int(predCones[pos,0])
                    # print(imageName, cone, id)
                    row = md.Row()
                    row.readFromMd(mdIn, id)
                    row.addToMd(mdConeList[i])
                fnExpCone = self._getExtraPath('metadataCone%d.xmd' % (i + 1))
                mdConeList[i].write(fnExpCone)

                fnProjCone = self._getExtraPath('projectionsCudaCorr%d.xmd' % (i + 1))
                fnOutCone = 'outCone%d.xmd' % (i + 1)

                if not exists(self._getExtraPath(fnOutCone)):
                    # Correlation step - calling cuda program
                    params = ' -i_ref %s -i_exp %s -o %s --odir %s --keep_best 1 ' \
                                 '--maxShift 10 ' % (fnProjCone, fnExpCone, fnOutCone,
                                 self._getExtraPath())
                    #params += ' --device %(GPU)s'
                    params += ' --device %d' %(int(idx % totalGpu))
                    self.runJob("xmipp_cuda_correlation", params, numberOfMpi=1)



    def correlationSignificantStep(self):

        mdNumCones = xmippLib.MetaData(self._getExtraPath("coneCenters.doc"))
        self.numCones = mdNumCones.size()

        # Cuda Correlation step - creating the metadata
        predCones = np.loadtxt(self._getExtraPath('conePrediction.txt'))
        mdConeList = []
        numMax = int(self.numConesSelected)
        for i in range(self.numCones):
            mdConeList.append(xmippLib.MetaData())
        mdIn = xmippLib.MetaData(self.imgsFn)
        #allInFns = mdIn.getColumnValues(xmippLib.MDL_IMAGE)

        for i in range(self.numCones):

            print("Classifying cone ", i + 1)
            positions = []
            for n in range(numMax):
                posAux = np.where(predCones[:, (n * 2) + 1] == (i + 1))
                positions = positions + (np.ndarray.tolist(posAux[0]))
                # print(posAux, positions, len(positions))

            if len(positions) > 0:
                for pos in positions:
                    # print(pos)
                    # imageName = allInFns[pos]
                    # cone = (i+1)
                    id = pos + 1  # int(predCones[pos,0])
                    # print(imageName, cone, id)
                    row = md.Row()
                    row.readFromMd(mdIn, id)
                    row.addToMd(mdConeList[i])
                fnExpCone = self._getExtraPath('metadataCone%d.xmd' % (i + 1))
                mdConeList[i].write(fnExpCone)

                fnProjCone = self._getExtraPath('projectionsCudaCorr%d.xmd' % (i + 1))
                fnOutCone = 'outCone%d.xmd' % (i + 1)

                if not exists(self._getExtraPath(fnOutCone)):
                    # Correlation step - calling significant program
                    args = '-i %s --initgallery %s --odir %s --dontReconstruct --useForValidation %d --dontCheckMirrors ' % \
                               (fnExpCone, fnProjCone, self._getExtraPath(), 1)
                    self.runJob('xmipp_reconstruct_significant', args,
                                    numberOfMpi=self.myMPI.get())
                    copy(self._getExtraPath('images_significant_iter001_00.xmd'), self._getExtraPath(fnOutCone))
                    remove(self._getExtraPath('angles_iter001_00.xmd'))
                    remove(self._getExtraPath('images_significant_iter001_00.xmd'))


    def createOutputMetadataStep(self):

        mdNumCones = xmippLib.MetaData(self._getExtraPath("coneCenters.doc"))
        self.numCones = mdNumCones.size()
        numMax = int(self.numConesSelected)
        mdIn = xmippLib.MetaData(self.imgsFn)
        allInFns = mdIn.getColumnValues(xmippLib.MDL_IMAGE)

        coneFns = []
        coneCCs = []
        mdCones = []
        fnFinal = self._getExtraPath('outConesParticles.xmd')
        for i in range(self.numCones):

            fnOutCone = 'outCone%d.xmd' % (i + 1)

            if exists(self._getExtraPath(fnOutCone)):             

                if numMax == 1:
                    if not exists(fnFinal):
                        copy(self._getExtraPath(fnOutCone), fnFinal)
                    else:
                        params = ' -i %s --set union %s -o %s' % (fnFinal,
                                                                  self._getExtraPath(
                                                                      fnOutCone),
                                                                  fnFinal)
                        self.runJob("xmipp_metadata_utilities", params,
                                    numberOfMpi=1)
                else:
                    mdCones.append(
                        xmippLib.MetaData(self._getExtraPath(fnOutCone)))
                    coneFns.append(
                        mdCones[i].getColumnValues(xmippLib.MDL_IMAGE))
                    coneCCs.append(
                        mdCones[i].getColumnValues(xmippLib.MDL_MAXCC))

            else:
                if numMax > 1:
                    mdCones.append(None)
                    coneFns.append([])
                    coneCCs.append([])

        if numMax > 1:
            mdFinal = xmippLib.MetaData()
            row = md.Row()
            for myFn in allInFns:
                myCCs = []
                myCones = []
                myPos = []
                for n in range(self.numCones):
                    if myFn in coneFns[n]:
                        pos = coneFns[n].index(myFn)
                        myPos.append(pos)
                        myCCs.append(coneCCs[n][pos])
                        myCones.append(n + 1)
                if len(myPos) > 0:
                    coneMax = myCones[myCCs.index(max(myCCs))]
                    objId = myPos[myCCs.index(max(myCCs))] + 1
                    row.readFromMd(mdCones[coneMax - 1], objId)
                    row.addToMd(mdFinal)
            mdFinal.write(fnFinal)

    def createOutputStep(self):

        #cleanPattern(self._getExtraPath('metadataCone*'))
        #AJ new (to check)
        #cleanPattern(self._getExtraPath('projectionsCudaCorr*'))

        inputParticles = self.inputSet.get()
        fnOutputParticles = self._getExtraPath('outConesParticles.xmd')

        outputSetOfParticles = self._createSetOfParticles()
        outputSetOfParticles.copyInfo(inputParticles)
        outputSetOfParticles.setAlignmentProj()

        Xdim = inputParticles.getXDim()
        newXdim = readInfoField(self._getExtraPath(), "size",
                                xmippLib.MDL_XSIZE)
        Ts = readInfoField(self._getExtraPath(), "sampling",
                           xmippLib.MDL_SAMPLINGRATE)
        if newXdim != Xdim:

            # Option 1
            # self.runJob("xmipp_image_resize",
            #             "-i %s -o %s --save_metadata_stack %s --fourier %d" %
            #             (fnOutputParticles,
            #              self._getExtraPath('outConesParticlesScaled.stk'),
            #              self._getExtraPath('outConesParticlesScaled.xmd'),
            #              Xdim))
            # fnOutputParticles = self._getExtraPath('outConesParticlesScaled.xmd')
            # readSetOfParticles(fnOutputParticles, outputSetOfParticles)

            # Option 2, evitando el resize
            self.scaleFactor = Ts / inputParticles.getSamplingRate()
            self.iterMd = md.iterRows(fnOutputParticles, xmippLib.MDL_ITEM_ID)
            self.lastRow = next(self.iterMd)
            outputSetOfParticles.copyItems(inputParticles,
                                           updateItemCallback=self._updateItem)
        else:
            readSetOfParticles(fnOutputParticles, outputSetOfParticles)
        self._defineOutputs(outputParticles=outputSetOfParticles)

    def _updateItem(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(
                xmippLib.MDL_ITEM_ID):
            count += 1
            if count:
                self._createItemMatrix(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
        particle._appendItem = count > 0

    def _createItemMatrix(self, particle, row):

        row.setValue(xmippLib.MDL_SHIFT_X,
                     row.getValue(xmippLib.MDL_SHIFT_X) * self.scaleFactor)
        row.setValue(xmippLib.MDL_SHIFT_Y,
                     row.getValue(xmippLib.MDL_SHIFT_Y) * self.scaleFactor)
        setXmippAttributes(particle, row, xmippLib.MDL_SHIFT_X,
                           xmippLib.MDL_SHIFT_Y,
                           xmippLib.MDL_ANGLE_ROT, xmippLib.MDL_ANGLE_TILT,
                           xmippLib.MDL_ANGLE_PSI)
        createItemMatrix(particle, row, align=em.ALIGN_PROJ)

    # --------------------------- INFO functions --------------------------------
    def _summary(self):
        summary = []
        summary.append("Images evaluated: %i" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        return summary

    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append(
                "We evaluated %i input images %s regarding to volume %s." \
                % (self.inputSet.get().getSize(), self.getObjectTag('inputSet'),
                   self.getObjectTag('inputVolume')))
        return methods

    def _validate(self):
        errors = []
        if self.numberOfMpi>1:
            errors.append("You must select Threads to make the parallelization in Scipion level. "
                          "To parallelize the Xmipp program use MPIs in the form.")
        return errors


