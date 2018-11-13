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

from os import remove
from os.path import exists
from pyworkflow import VERSION_1_2
from pyworkflow.protocol.params import PointerParam, StringParam, FloatParam, IntParam
from pyworkflow.utils.path import moveFile, cleanPattern
from pyworkflow.em.protocol import ProtRefine3D
from shutil import copy
from xmipp3.convert import readSetOfParticles
import xmippLib
from xmipp3.utils import writeInfoField, readInfoField
import numpy as np
import math
from pyworkflow.em.metadata.utils import iterRows
import cv2
import pyworkflow.em.metadata as md

        
class XmippProtDeepCones3D(ProtRefine3D):
    """Performs a fast and approximate angular assignment that can be further refined
    with Xmipp highres local refinement"""
    _label = 'deep cones3D'
    _lastUpdateVersion = VERSION_1_2
    
    def __init__(self, **args):
        ProtRefine3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images", pointerClass='SetOfParticles')
        form.addParam('inputVolume', PointerParam, label="Volume", pointerClass='Volume')
        # form.addParam('mask', PointerParam, label="Mask", pointerClass='VolumeMask', allowsNull=True,
        #               help='The mask values must be between 0 (remove these pixels) and 1 (let them pass). Smooth masks are recommended.')
        form.addParam('targetResolution', FloatParam, label="Target resolution", default=3.0,
            help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                 "2/3 of the Fourier spectrum.")
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group', 
                      help='See http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry for a description of the symmetry groups format'
                        'If no symmetry is present, give c1')
        form.addParam('numEpochs', IntParam, label="Number of epochs for training",
                      default=10,
                      help="Number of epochs for training.")
        form.addParam('numConesTilt', IntParam, label="Number of cones per 180 degrees",
                      default=3,
                      help="Number of cones per 180 degrees.")
        form.addParallelSection(threads=0, mpi=8)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.lastIter = 0
        self.batchSize = 1024
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        
        self._insertFunctionStep("convertStep")
        # Trainig steps
        numConesTilt = int(self.numConesTilt)
        numConesRot = numConesTilt*2
        self.numCones = numConesTilt * numConesRot
        stepRot = int(360/numConesRot)
        stepTilt = int(180 / numConesTilt)
        iniRot = 0
        endRot = stepRot
        counterCones=0
        for i in range(numConesRot):
            iniTilt = 0
            endTilt = stepTilt
            for j in range(numConesTilt):
                self._insertFunctionStep("projectStep", 3000, iniRot, endRot, iniTilt, endTilt, 'projections', counterCones+1)
                self._insertFunctionStep("generateExpImagesStep", 1, 'projections', 'projectionsExp', counterCones+1, True)
                iniTilt +=stepTilt
                endTilt +=stepTilt
                counterCones+=1
            iniRot +=stepRot
            endRot +=stepRot

        for i in range(self.numCones):
            self._insertFunctionStep("trainNClassifiers2ClassesStep", i+1)

        #self._insertFunctionStep("trainOneClassifierNClassesStep")
        # Predict step
        self._insertFunctionStep("predictStep")

        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        from ..convert import writeSetOfParticles
        inputParticles = self.inputSet.get()
        writeSetOfParticles(inputParticles, self.imgsFn)
        Xdim = inputParticles.getXDim()
        Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0/3.0
        newTs = max(Ts, newTs)
        self.newXdim = long(Xdim * Ts / newTs)
        self.firstMaxShift = round(self.newXdim / 10)
        writeInfoField(self._getExtraPath(), "sampling", xmippLib.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE, self.newXdim)
        writeInfoField(self._getExtraPath(), "shift", xmippLib.MDL_SHIFT_X, self.firstMaxShift)
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (self.imgsFn,
                         self._getExtraPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         self.newXdim))
            moveFile(self._getExtraPath('scaled_particles.xmd'), self.imgsFn)

        from pyworkflow.em.convert import ImageHandler
        ih = ImageHandler()
        fnVol = self._getTmpPath("volume.vol")
        ih.convert(self.inputVolume.get(), fnVol)
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize","-i %s --dim %d"%(fnVol,self.newXdim),numberOfMpi=1)

    def projectStep(self, numProj, iniRot, endRot, iniTilt, endTilt, fn, idx):

        newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
        fnVol = self._getTmpPath("volume.vol")

        uniformProjectionsStr ="""
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
"""%(newXdim, newXdim, iniRot, endRot, numProj, iniTilt, endTilt)
        fnParams = self._getExtraPath("uniformProjections%d.xmd"%idx)
        fh = open(fnParams,"w")
        fh.write(uniformProjectionsStr)
        fh.close()

        fnProjs = self._getExtraPath(fn+"%d.stk"%idx)
        self.runJob("xmipp_phantom_project","-i %s -o %s --method fourier 1 0.5 --params %s"%(fnVol,fnProjs,fnParams),numberOfMpi=1)
        cleanPattern(self._getExtraPath('uniformProjections'))

    def generateExpImagesStep(self, Nrepeats, nameProj, nameExp, label, boolNoise):

        newXdim = readInfoField(self._getExtraPath(), "size",xmippLib.MDL_XSIZE)
        fnProj = self._getExtraPath(nameProj+"%d.xmd"%label)
        fnExp = self._getExtraPath(nameExp+"%d.xmd"%label)
        fnLabels = self._getExtraPath('labels.txt')
        fileLabels=open(fnLabels,"a")
        mdIn = xmippLib.MetaData(fnProj)
        mdExp = xmippLib.MetaData()
        newImage = xmippLib.Image()
        maxPsi = 180
        maxShift = round(newXdim / 10)
        idx=1
        for row in iterRows(mdIn):
            objId = row.getObjId()
            fnImg = mdIn.getValue(xmippLib.MDL_IMAGE, objId)
            myRow = row
            I = xmippLib.Image(fnImg)
            Xdim, Ydim, _, _ = I.getDimensions()
            Xdim2 = Xdim / 2
            Ydim2 = Ydim / 2

            for i in range(Nrepeats):
                psiDeg = np.random.uniform(-maxPsi, maxPsi)
                psi = psiDeg * math.pi / 180.0
                deltaX = np.random.uniform(-maxShift, maxShift)
                deltaY = np.random.uniform(-maxShift, maxShift)
                c = math.cos(psi)
                s = math.sin(psi)
                M = np.float32([[c, s, (1 - c) * Xdim2 - s * Ydim2 + deltaX],
                                [-s, c, s * Xdim2 + (1 - c) * Ydim2 + deltaY]])
                newImg = cv2.warpAffine(I.getData(), M, (Xdim, Ydim))
                if boolNoise:
                    newImg = newImg + np.random.normal(0.0, 10.0, [Xdim, Xdim])
                newFn = ('%06d@'%idx)+fnExp[:-3]+'stk'
                newImage.setData(newImg)
                newImage.write(newFn)
                myRow.setValue(xmippLib.MDL_IMAGE, newFn)
                myRow.setValue(xmippLib.MDL_ANGLE_PSI, psiDeg)
                myRow.setValue(xmippLib.MDL_SHIFT_X, deltaX)
                myRow.setValue(xmippLib.MDL_SHIFT_Y, deltaY)
                myRow.addToMd(mdExp)
                idx+=1
                fileLabels.write(str(label-1)+'\n')
        mdExp.write(fnExp)
        fileLabels.close()
        if (label-1)>0:
            lastFnExp = self._getExtraPath(nameExp+"%d.xmd"%(label-1))
            self.runJob("xmipp_metadata_utilities", " -i %s --set union %s -o %s " %
                        (lastFnExp, fnExp, fnExp), numberOfMpi=1)

    def trainNClassifiers2ClassesStep(self, idx):

        fnLabels = self._getExtraPath('labels.txt')
        fileLabels = open(fnLabels, "r")
        newFnLabels = self._getExtraPath('labels%d.txt' %idx)
        newFileLabels = open(newFnLabels, "w")
        lines = fileLabels.readlines()
        for line in lines:
            if line == str(idx-1)+'\n':
                newFileLabels.write('1\n')
            else:
                newFileLabels.write('0\n')
        newFileLabels.close()
        fileLabels.close()

        newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
        expSet = self._getExtraPath('projectionsExp%d.xmd'%self.numCones)
        fnLabels = self._getExtraPath('labels%d.txt'%idx)
        modelFn = 'modelCone%d'%idx

        self.runJob("xmipp_cone_deepalign", "%s %s %s %s %d %d %d %d" %
                    (expSet, fnLabels, self._getExtraPath(),
                     modelFn, self.numEpochs, newXdim, 2, self.batchSize), numberOfMpi=1)
        #copy(self._getExtraPath('pruebaYpred.txt'), self._getExtraPath('prediction%d.txt'%idx))

    # def trainOneClassifierNClassesStep(self):
    #
    #     newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
    #     expSet = self._getExtraPath('projectionsExp%d.xmd'%self.numCones)
    #     fnLabels = self._getExtraPath('labels.txt')
    #     modelFn = 'modelCone'
    #
    #     self.runJob("xmipp_cone_deepalign", "%s %s %s %s %d %d %d" %
    #                 (expSet, fnLabels, self._getExtraPath(),
    #                  modelFn, 30, newXdim, self.numCones), numberOfMpi=1)


    def predictStep(self):
        newXdim = readInfoField(self._getExtraPath(), "size",xmippLib.MDL_XSIZE)
        self.runJob("xmipp_cone_deepalign_predict", "%s %s %d %d" %
                    (self.imgsFn, self._getExtraPath(), newXdim, self.numCones), numberOfMpi=1)

        #Cuda Correlation step - creating the metadata
        predCones = np.loadtxt(self._getExtraPath('conePrediction.txt'))
        mdConeList=[]
        for i in range(self.numCones):
            mdConeList.append(xmippLib.MetaData())
        mdIn = xmippLib.MetaData(self.imgsFn)
        allInFns = mdIn.getColumnValues(xmippLib.MDL_IMAGE)
        fnFinal = self._getExtraPath('outConesParticles.xmd')
        for i in range(self.numCones):
            print("Classifying cone ", i+1)
            positions = np.where(predCones[:,1]==(i+1))
            positions=positions[0]
            #print(positions, len(positions))

            if len(positions)>0:
                for pos in positions:
                    #print(pos)
                    imageName = allInFns[pos]
                    cone = (i+1)
                    #id = int(predCones[pos,0])
                    #print(imageName, cone, id)
                    row = md.Row()
                    row.setValue(md.MDL_IMAGE, imageName)
                    row.setValue(md.MDL_REF, cone)
                    row.addToMd(mdConeList[i])
                fnExpCone = self._getExtraPath('metadataCone%d.xmd' % (i + 1))
                mdConeList[i].write(fnExpCone)

                fnProjCone = self._getExtraPath('projections%d.xmd'%(i+1))
                fnOutCone = 'outCone%d.xmd'%(i+1)
                #fnOutClassesCone = 'outClassesCone%d.xmd'%(i+1)
                #Cuda Correlation step - calling cuda program
                params = ' -i_ref %s -i_exp %s -o %s --odir %s --keep_best 1 ' \
                         '--maxShift 10 '%(fnProjCone, fnExpCone, fnOutCone,
                                           self._getExtraPath())
                self.runJob("xmipp_cuda_correlation", params, numberOfMpi=1)

                if not exists(fnFinal):
                    copy(self._getExtraPath(fnOutCone), fnFinal)
                else:
                    params = ' -i %s --set union %s -o %s'%(fnFinal,
                                                            self._getExtraPath(fnOutCone),
                                                            fnFinal)
                    self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)
                remove(self._getExtraPath(fnOutCone))


    def createOutputStep(self):
        cleanPattern(self._getExtraPath('metadataCone'))

        inputParticles = self.inputSet.get()
        fnOutputParticles = self._getExtraPath('outConesParticles.xmd')
        outputSetOfParticles = self._createSetOfParticles()
        outputSetOfParticles.copyInfo(inputParticles)
        readSetOfParticles(fnOutputParticles, outputSetOfParticles)
        self._defineOutputs(outputParticles=outputSetOfParticles)

    #--------------------------- INFO functions --------------------------------
    def _summary(self):
        summary = []
        summary.append("Images evaluated: %i" % self.inputSet.get().getSize())
        summary.append("Volume: %s" % self.inputVolume.getNameId())
        return summary
    
    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We evaluated %i input images %s regarding to volume %s."\
                           %(self.inputSet.get().getSize(), self.getObjectTag('inputSet'), self.getObjectTag('inputVolume')) )
        return methods

