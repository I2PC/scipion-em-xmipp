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

from math import floor
import os

from pyworkflow import VERSION_1_2
from pyworkflow.protocol.params import PointerParam, StringParam, FloatParam, BooleanParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.em.constants import ALIGN_PROJ
from pyworkflow.utils.path import cleanPath, moveFile
from pyworkflow.em.protocol import ProtRefine3D
import pyworkflow.em.metadata as md
from pyworkflow.em.data import SetOfClasses2D, SetOfAverages
from shutil import copy
from xmipp3.convert import readSetOfParticles

import xmippLib
from xmipp3.convert import setXmippAttributes, xmippToLocation, rowToAlignment
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
        form.addParallelSection(threads=0, mpi=8)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.numIter = 3
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        
        self._insertFunctionStep("convertStep")
        # Trainig steps
        self._insertFunctionStep("projectStep", 10000, 'projections')
        self._insertFunctionStep("alignStep")
        # # Predicting steps
        self._insertFunctionStep("projectStep", 100, 'projectionsTest')
        self._insertFunctionStep("predictStep")

        self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        from ..convert import writeSetOfClasses2D, writeSetOfParticles
        inputParticles = self.inputSet.get()
        writeSetOfParticles(inputParticles, self.imgsFn)
        Xdim = inputParticles.getXDim()
        Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0/3.0
        newTs = max(Ts, newTs)
        self.newXdim = long(Xdim * Ts / newTs)
        writeInfoField(self._getExtraPath(), "sampling", xmippLib.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE, self.newXdim)
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
        self.runJob("xmipp_phantom_project","-i %s -o %s --method fourier 1 0.5 --params %s"%(fnVol,fnProjs,fnParams),numberOfMpi=1)

    def alignStep(self):

        for i in range(self.numIter):
            self.shiftAlign(i)
            self.psiAlign(i)
            #self.shiftAlign(i)

        self.rotTiltAlign(self.numIter)

        maxShift = np.loadtxt(os.path.join(self._getExtraPath(), 'shift_iter%06d.txt' % (self.numIter - 1)))
        prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(), 'psi_iter%06d.txt' % (self.numIter - 1)))
        maxPsi = np.rad2deg(np.arctan2(prevErrorPsi, 1-prevErrorPsi))
        prevErrorRot = np.loadtxt(os.path.join(self._getExtraPath(), 'rot_iter%06d.txt' % (self.numIter)))
        maxRot = np.rad2deg(np.arctan2(prevErrorRot, 1-prevErrorRot))
        prevErrorTilt = np.loadtxt(os.path.join(self._getExtraPath(), 'tilt_iter%06d.txt' % (self.numIter)))
        maxTilt = np.rad2deg(np.arctan2(prevErrorTilt, 1-prevErrorTilt))
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
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn),numberOfMpi=1)

    def shiftAlign(self, i):
        if i == 0:
            maxShift = round(self.newXdim/10) # *** To form
            #prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(),'psi_iter%06d.txt'%i))
            maxPsi = 180 #3*np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))
        else:
            maxShift = 3*np.loadtxt(os.path.join(self._getExtraPath(),'shift_iter%06d.txt'%(i-1)))
            prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(),'psi_iter%06d.txt'%(i-1))) #i
            maxPsi = 3*np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))

        mode = 'shift'
        modelFn = mode+'_iter%06d'%i
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn),numberOfMpi=1)

    def rotTiltAlign(self, i):

        maxShift = 2*np.loadtxt(os.path.join(self._getExtraPath(),'shift_iter%06d.txt'%(i-1)))
        prevErrorPsi = np.loadtxt(os.path.join(self._getExtraPath(),'psi_iter%06d.txt'%(i-1)))
        maxPsi = 2*np.rad2deg(np.arctan2(prevErrorPsi, 1 - prevErrorPsi))

        mode = 'rot'
        modelFn = mode+'_iter%06d'%i
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn),numberOfMpi=1)

        mode = 'tilt'
        modelFn = mode+'_iter%06d'%i
        self.runJob("xmipp_angular_deepalign","%s %f %f %s %s %s"%
                    (self._getExtraPath("projections.xmd"),maxShift,maxPsi,mode,self._getExtraPath(),modelFn),numberOfMpi=1)

    def predictStep(self):
        newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
        outMdFn = self._getExtraPath('outputParticles.xmd')
        copy(self.imgsFn, outMdFn)
        self.runJob("xmipp_angular_deepalign_predict", "%s %f %f %s %d" %
                    (outMdFn,
                     round(newXdim/10), 180, self._getExtraPath(),
                     self.numIter), numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputSet.get()
        fnDeformedParticles = self._getExtraPath('outputParticles.xmd')
        outputSetOfParticles = self._createSetOfParticles()
        outputSetOfParticles.copyInfo(inputParticles)
        readSetOfParticles(fnDeformedParticles, outputSetOfParticles)
        self._defineOutputs(outputParticles=outputSetOfParticles)

        #--------------------------- INFO functions --------------------------------------------
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
            methods.append("The residuals were evaluated according to their mean, variance and covariance structure [Cherian2013].")
        return methods

