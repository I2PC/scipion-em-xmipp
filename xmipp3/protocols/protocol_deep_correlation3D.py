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

import os
from pyworkflow import VERSION_1_2
from pyworkflow.protocol.params import PointerParam, StringParam, FloatParam, IntParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils.path import moveFile
from pyworkflow.em.protocol import ProtRefine3D
from shutil import copy
from xmipp3.convert import readSetOfParticles
import xmippLib
from xmipp3.utils import writeInfoField, readInfoField
import numpy as np
import cv2
import math
from pyworkflow.em.metadata.utils import iterRows

        
class XmippProtDeepCorrelation3D(ProtRefine3D):
    """Performs a fast and approximate angular assignment that can be further refined
    with Xmipp highres local refinement"""
    _label = 'deep correlation3D'
    _lastUpdateVersion = VERSION_1_2
    
    def __init__(self, **args):
        ProtRefine3D.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet', PointerParam, label="Input images", pointerClass='SetOfParticles')
        form.addParam('inputVolume', PointerParam, label="Volume", pointerClass='Volume')
        form.addParam('targetResolution', FloatParam, label="Target resolution", default=8.0,
            help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                 "2/3 of the Fourier spectrum.")
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group', 
                      help='See http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry for a description of the symmetry groups format'
                        'If no symmetry is present, give c1')
        form.addParallelSection(threads=0, mpi=8)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.lastIter = 0
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        
        self._insertFunctionStep("convertStep")
        # Trainig steps
        self._insertFunctionStep("generateProjImagesStep", 1000, 'projections')
        self._insertFunctionStep("generateExpImagesStep", 30, 'projections', 'projectionsExp', True)
        self._insertFunctionStep("calculateCorrMatrixStep")
        self._insertFunctionStep("trainStep")
        # Predict step
        #self._insertFunctionStep("predictStep")

        #self._insertFunctionStep("createOutputStep")

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

    def generateProjImagesStep(self, numProj, fn):

        newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
        fnVol = self._getTmpPath("volume.vol")

        uniformProjectionsStr ="""
# XMIPP_STAR_1 *
data_block1
_dimensions2D   '%d %d'
_projRotRange    '0 360 %d'
_projRotRandomness   random 
_projRotNoise   '0'
_projTiltRange    '0 180 1'
_projTiltRandomness   random 
_projTiltNoise   '0'
_projPsiRange    '0 0 1'
_projPsiRandomness   random 
_projPsiNoise   '0'
_noisePixelLevel   '0'
_noiseCoord   '0'
"""%(newXdim, newXdim, numProj)
        fnParams = self._getExtraPath("uniformProjections.xmd")
        fh = open(fnParams,"w")
        fh.write(uniformProjectionsStr)
        fh.close()

        fnProjs = self._getExtraPath(fn+".stk")
        self.runJob("xmipp_phantom_project","-i %s -o %s --method fourier 1 0.5 --params %s"%(fnVol,fnProjs,fnParams),numberOfMpi=1)

    def generateExpImagesStep(self, Nrepeats, nameProj, nameExp, boolNoise):

        fnProj = self._getExtraPath(nameProj+".xmd")
        fnExp = self._getExtraPath(nameExp+".xmd")
        mdIn = xmippLib.MetaData(fnProj)
        mdExp = xmippLib.MetaData()
        newImage = xmippLib.Image()
        maxPsi = 180
        maxShift = round(self.newXdim / 10)
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
        mdExp.write(fnExp)


    def calculateCorrMatrixStep(self):

        refSet = self._getExtraPath('projections.xmd')
        expSet = self._getExtraPath('projectionsExp.xmd')
        args = '-i_ref %s -i_exp %s -o %s --odir %s --keep_best %d ' \
               '--maxShift %d ' %(refSet, expSet,
                self._getExtraPath('outputGl2d.xmd'), self._getExtraPath(), 1, 10)
        self.runJob("xmipp_cuda_correlation", args, numberOfMpi=1)

    def trainStep(self):

        newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
        refSet = self._getExtraPath('projections.xmd')
        expSet = self._getExtraPath('projectionsExp.xmd')
        fnCorrMatrix = self._getExtraPath('example.txt')
        modelFn = 'modelCorr'

        self.runJob("xmipp_correlation_deepalign", "%s %s %s %s %s %d %d" %
                    (refSet, expSet, fnCorrMatrix, self._getExtraPath(),
                     modelFn, 10, newXdim), numberOfMpi=1)


    def predictStep(self):
        newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
        fnProjs = self._getExtraPath("projections.xmd")
        outMdFn = self._getExtraPath('outputParticles.xmd')
        self.runJob("xmipp_correlation_deepalign_predict", " %s %s %s %s %d" %
                    (fnProjs, self.imgsFn, outMdFn, self._getExtraPath(), newXdim), numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputSet.get()
        fnDeformedParticles = self._getExtraPath('outputParticles.xmd')
        outputSetOfParticles = self._createSetOfParticles()
        outputSetOfParticles.copyInfo(inputParticles)
        readSetOfParticles(fnDeformedParticles, outputSetOfParticles)
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

