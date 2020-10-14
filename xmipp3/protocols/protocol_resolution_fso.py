# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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

import numpy as np

from pwem.emlib.image import ImageHandler
from pwem.objects import FSC
from pwem.protocols import ProtAnalysis3D
from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import STEPS_PARALLEL
import pyworkflow.protocol.params as params
import pwem.emlib.metadata as md

from xmipp3.convert import locationToXmipp, writeSetOfParticles

from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pwem.objects import Volume, SetOfParticles



OUTPUT_3DFSC = '3dFSC.mrc'
OUTPUT_SPHERE = 'fsc/sphere.mrc'
OUTPUT_DIRECTIONAL_FILTER = 'fsc/filteredMap.mrc'


class XmippProtFSO(ProtAnalysis3D):
    """    
    Given two half maps the protocol estimates Fourier Shell Occupancy to determine the global anisotropy of the map
    """
    _label = 'resolution fso'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('half1', PointerParam, pointerClass='Volume',
                      label="Half Map 1", important=True,
                      help='Select one map for determining the '
		      'directional FSC resolution.')

        form.addParam('half2', PointerParam, pointerClass='Volume',
                      label="Half Map 2", important=True,
                      help='Select the second map for determining the '
                      'directional FSC resolution.')
        
        form.addParam('mask', PointerParam, pointerClass='VolumeMask', 
                      allowsNull=True, 
                      label="Mask", 
                      help='The mask determines which points are specimen'
                      ' and which are not')

        form.addParam('bestAngle', BooleanParam, default=True, 
                      label="Find Best Cone angle",
                      help='The algorithm will try to determine by cross validation'
			   'the best cone angle to then determine the directional FSC'
			   'Note that the cone angle is the angle between the axis '
			   'of the cone and the generatrix')

        form.addParam('coneAngle', FloatParam, default=20.0, 
		      condition = 'not bestAngle ',
                      label="Cone Angle",
                      help='Angle between the axis of the cone and the generatrix')
        
        form.addParallelSection(threads = 4, mpi = 0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {OUTPUT_3DFSC: self._getExtraPath("3dFSC.mrc"),
                 OUTPUT_SPHERE: self._getExtraPath("fsc/sphere.mrc"),
		 OUTPUT_DIRECTIONAL_FILTER: self._getExtraPath("fsc/filteredMap.mrc"),
                 }
        self._updateFilenamesDict(myDict)


    def _insertAllSteps(self):
        self._createFilenameTemplates() 
        # Convert input into xmipp Metadata format
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('resolutionDirectionalFSCStep')
        self._insertFunctionStep('createOutputStep')

    def mrc_convert(self, fileName, outputFileName):
        """Check if the extension is .mrc, if not then uses xmipp to convert it
        """
        ext = getExt(fileName)
        if (ext !='.mrc'):
            params = ' -i %s' % fileName
            params += ' -o %s' % outputFileName
            self.runJob('xmipp_image_convert', params)
            #outputFileName = outputFileName # +':mrc'
            return outputFileName
        else:
            return fileName

    def convertInputStep(self):
        """ Read the input volume.
        """
        self.vol1Fn = self.mrc_convert(self.half1.get().getFileName(),
                                  self._getTmpPath('half1.mrc'))
        self.vol2Fn = self.mrc_convert(self.half2.get().getFileName(),
                                  self._getTmpPath('half2.mrc'))
        self.maskFn = self.mrc_convert(self.mask.get().getFileName(),
                                  self._getExtraPath('mask.mrc'))

    def resolutionDirectionalFSCStep(self):
        import os
        fndir = self._getExtraPath("fsc") 

        os.mkdir(fndir) 	

        params = ' --half1 %s' % self.vol1Fn
        params += ' --half2 %s' % self.vol2Fn

        if self.mask.hasValue():
            params += ' --mask %s' % self.maskFn

        params += ' --sampling %f' % self.half1.get().getSamplingRate()
        if self.bestAngle.get() is False:
            params += ' --anglecone %f' % self.coneAngle.get()
        params += ' --threedfsc %s' % self._getExtraPath("3dFSC.mrc")
        params += ' --fscfolder %s' % (fndir+'/')
        params += ' --anisotropy %s' % self._getExtraPath("fso.xmd")
        
        self.runJob('xmipp_resolution_fso', params)
      
     
    def createOutputStep(self):
        volume=Volume()
        volume.setFileName(self._getExtraPath("3dFSC.mrc"))

        volume.setSamplingRate(self.half1.get().getSamplingRate())
        self._defineOutputs(fsc3D=volume)
        self._defineSourceRelation(self.half1, volume)



    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ')
        return messages
    
    def _summary(self):
        summary = []
        summary.append(" ")
        return summary

    def _citations(self):
        return ['Vilas2020']
