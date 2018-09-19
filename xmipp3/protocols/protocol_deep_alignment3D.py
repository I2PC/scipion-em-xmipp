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

import xmippLib
from xmipp3.convert import setXmippAttributes, xmippToLocation, rowToAlignment
from xmipp3.utils import writeInfoField, readInfoField

        
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
        # Convert input images if necessary
        self.imgsFn = self._getExtraPath('input_imgs.xmd')
        
        self._insertFunctionStep("convertStep")
        #self._insertFunctionStep("align")
        #self._insertFunctionStep("createOutputStep")

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        from ..convert import writeSetOfClasses2D, writeSetOfParticles
        inputParticles = self.inputSet.get()
        writeSetOfParticles(inputParticles, self.imgsFn)
        Xdim = inputParticles.getXDim()
        Ts = inputParticles.getSamplingRate()
        newTs = self.targetResolution.get() * 1.0/3.0
        newTs = max(Ts, newTs)
        newXdim = long(Xdim * Ts / newTs)
        writeInfoField(self._getExtraPath(), "sampling", xmippLib.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE, newXdim)
        if newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (self.imgsFn,
                         self._getExtraPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         newXdim))
            moveFile()

        from pyworkflow.em.convert import ImageHandler
        ih = ImageHandler()
        fnVol = self._getTmpPath("volume.vol")
        ih.convert(self.inputVolume.get(), fnVol)
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim != newXdim:
            self.runJob("xmipp_image_resize","-i %s --dim %d"%(fnVol,newXdim),numberOfMpi=1)

        uniformProjectionsStr ="""
# XMIPP_STAR_1 *
data_block1
_dimensions2D   '%d %d'
_projRotRange    '0 360 5000'
_projRotRandomness   random 
_projRotNoise   '0'
_projTiltRange    '0 180 1'
_projTiltRandomness   random 
_projTiltNoise   '0'
_projPsiRange    '0 360 1'
_projPsiRandomness   random 
_projPsiNoise   '0'
_noisePixelLevel   '0'
_noiseCoord   '0'
"""%(newXdim,newXdim)
        fnParams = self._getExtraPath("uniformProjections.xmd")
        fh = open(fnParams,"w")
        fh.write(uniformProjectionsStr)
        fh.close()

        fnProjs = self._getExtraPath("projections.stk")
        self.runJob("xmipp_phantom_project","-i %s -o %s --method fourier 1 0.5 --params %s"%(fnVol,fnProjs,fnParams),numberOfMpi=1)
    
    def createOutputStep(self):
        fnImgs = self._getExtraPath('images.stk')
        if os.path.exists(fnImgs):
            cleanPath(fnImgs)

        outputSet = self._createSetOfParticles()
        imgSet = self.inputSet.get()
        imgFn = self._getExtraPath("anglesCont.xmd")
        self.newAssignmentPerformed = os.path.exists(self._getExtraPath("angles.xmd"))
        self.samplingRate = self.inputSet.get().getSamplingRate()
        if isinstance(imgSet, SetOfClasses2D):
            outputSet = self._createSetOfClasses2D(imgSet)
            outputSet.copyInfo(imgSet.getImages())
        elif isinstance(imgSet, SetOfAverages):
            outputSet = self._createSetOfAverages()
            outputSet.copyInfo(imgSet)
        else:
            outputSet = self._createSetOfParticles()
            outputSet.copyInfo(imgSet)
            if not self.newAssignmentPerformed:
                outputSet.setAlignmentProj()
        outputSet.copyItems(imgSet,
                            updateItemCallback=self._processRow,
                            itemDataIterator=md.iterRows(imgFn, sortByLabel=md.MDL_ITEM_ID))
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(self.inputSet, outputSet)

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

