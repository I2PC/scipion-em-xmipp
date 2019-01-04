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

        
class XmippProtDeepAlignment3D(ProtRefine3D):
    """Performs a fast and approximate angular assignment that can be further refined
    with Xmipp highres local refinement"""
    _label = 'deep alignment3D'
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
        form.addParam('numIterMin', IntParam, label="Min iterations",
                      default=3,
                      help="The minimum number of iterations in the training process.")
        form.addParam('numEpochs', IntParam, label="Number of epochs in psi and shift training",
                      default=3,
                      help="Number of epochs in training for psi and shift paremeters.")
        form.addParam('numEpochsRot', IntParam, label="Number of epochs in rot and tilt training",
                      default=8,
                      help="Number of epochs in training for rot and tilt paremeters.")
        form.addParam('minShiftError', FloatParam, label="Min shift error",
                      default=1.0,
                      help="In pixels, the minimum error allowed in the training of shifts.")
        form.addParam('minPsiError', FloatParam, label="Min psi error",
                      default=5.0,
                      help="In degrees, the minimum error allowed in the training of psi.")
        form.addParam('minReduction', IntParam, label="Min train reduction",
                      default=10,
                      help="In percentage, the minimum change required in the training process.")
        form.addParallelSection(threads=0, mpi=8)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.lastIter = 0
        self.stdNoise = 2.0
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        
        self._insertFunctionStep("convertStep")
        # Trainig steps
        self._insertFunctionStep("projectStep", 10000, 'projections')
        self._insertFunctionStep("trainStep")
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

    def projectStep(self, numProj, fn):

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
        self.runJob("xmipp_phantom_project","-i %s -o %s --method fourier 1 0.5 "
                    "--params %s --sym %s "%(fnVol,fnProjs,fnParams, self.symmetryGroup.get()),numberOfMpi=1)

    def trainStep(self):

        maxShiftPrev = readInfoField(self._getExtraPath(), "shift", xmippLib.MDL_SHIFT_X)
        maxPsiPrev = 180
        keepTraining = True
        i=0
        while keepTraining:
            self.shiftAlign(i)
            self.psiAlign(i)
            #self.shiftAlign(i)

            #Checking if we need to continue the training
            maxShift = np.loadtxt(os.path.join(self._getExtraPath(), 'shift_iter%06d.txt' % i))
            prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(), 'psi_iter%06d.txt' % i))
            maxPsi = np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))

            shiftReduction = float(maxShiftPrev-maxShift)*100/float(maxShiftPrev)
            psiReduction = float(maxPsiPrev - maxPsi) * 100 / float(maxPsiPrev)

            print("Conditions to train: ", i, maxShiftPrev, maxShift, maxPsiPrev, maxPsi, shiftReduction, psiReduction)

            if i>=self.numIterMin and maxShift<self.minShiftError and \
                    maxPsi<self.minPsiError and \
                    shiftReduction<self.minReduction and \
                    psiReduction<self.minReduction:
                keepTraining=False
                self.lastIter=i
            if i>=self.numIterMin and (shiftReduction<0 or psiReduction<0):
                keepTraining = False
                self.lastIter=i-1

            maxShiftPrev = maxShift
            maxPsiPrev = maxPsi
            i+=1

        self.rotTiltAlign(self.lastIter)

        maxShift = np.loadtxt(os.path.join(self._getExtraPath(), 'shift_iter%06d.txt' % self.lastIter))
        prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(), 'psi_iter%06d.txt' % self.lastIter))
        maxPsi = np.rad2deg(np.arctan2(prevErrorPsi, 1-prevErrorPsi))
        prevErrorRot = np.loadtxt(os.path.join(self._getExtraPath(), 'rot_iter%06d.txt' % (self.lastIter+1)))
        maxRot = np.rad2deg(np.arctan2(prevErrorRot, 1-prevErrorRot))
        prevErrorTilt = np.loadtxt(os.path.join(self._getExtraPath(), 'tilt_iter%06d.txt' % (self.lastIter+1)))
        maxTilt = np.rad2deg(np.arctan2(prevErrorTilt, 1-prevErrorTilt))
        writeInfoField(self._getExtraPath(), "iter", xmippLib.MDL_REF, self.lastIter)
        print("FINAL ERROR VALUES")
        print("shift ", float(maxShift), " psi ", maxPsi, " rot ", maxRot, " tilt ", maxTilt)


    def psiAlign(self, i):
        if i==0:
            maxShift = 3*np.loadtxt(os.path.join(self._getExtraPath(),'shift_iter%06d.txt'%i)) #round(self.newXdim/10)
            maxPsi = 180
        else:
            maxShift = 3*np.loadtxt(os.path.join(self._getExtraPath(),'shift_iter%06d.txt'%(i))) #i-1
            prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(),'psi_iter%06d.txt'%(i-1)))
            maxPsi = 3*np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))

        mode = 'psi'
        modelFn = mode+'_iter%06d'%i
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s %d %d"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn, self.numEpochs, self.stdNoise),numberOfMpi=1)

    def shiftAlign(self, i):
        if i == 0:
            maxShift = readInfoField(self._getExtraPath(), "shift", xmippLib.MDL_SHIFT_X)
            #prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(),'psi_iter%06d.txt'%i))
            maxPsi = 180 #3*np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))
        else:
            maxShift = 3*np.loadtxt(os.path.join(self._getExtraPath(),'shift_iter%06d.txt'%(i-1)))
            prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(),'psi_iter%06d.txt'%(i-1))) #i
            maxPsi = 3*np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))

        mode = 'shift'
        modelFn = mode+'_iter%06d'%i
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s %d %d"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn, self.numEpochs, self.stdNoise),numberOfMpi=1)

    def rotTiltAlign(self, i):

        print ('shift_iter%06d.txt'%i)
        maxShift = 2*np.loadtxt(os.path.join(self._getExtraPath(),'shift_iter%06d.txt'%i))
        prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(),'psi_iter%06d.txt'%i))
        maxPsi = 2*np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))

        mode = 'rot'
        modelFn = mode+'_iter%06d'%(i+1)
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s %d %d"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn, self.numEpochsRot, self.stdNoise),numberOfMpi=1)

        mode = 'tilt'
        modelFn = mode+'_iter%06d'%(i+1)
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s %d %d"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn, self.numEpochsRot, self.stdNoise),numberOfMpi=1)

    def predictStep(self):
        lastIter = readInfoField(self._getExtraPath(), "iter", xmippLib.MDL_REF)
        firstMaxShift = readInfoField(self._getExtraPath(), "shift", xmippLib.MDL_SHIFT_X)
        outMdFn = self._getExtraPath('outputParticles.xmd')
        copy(self.imgsFn, outMdFn)
        self.runJob("xmipp_angular_deepalign_predict", "%s %f %f %s %d" %
                    (outMdFn, firstMaxShift, 180, self._getExtraPath(),
                     lastIter+1), numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputSet.get()
        fnDeformedParticles = self._getExtraPath('outputParticles.xmd')
        self.runJob("xmipp_metadata_utilities",
                    " -i %s --fill weight constant 1.0" %
                    (fnDeformedParticles), numberOfMpi=1)
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

