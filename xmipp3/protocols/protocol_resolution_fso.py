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


from pyworkflow import VERSION_2_0
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pwem.objects import Volume
from pwem.protocols import ProtAnalysis3D

OUTPUT_3DFSC = '3dFSC.mrc'
OUTPUT_DIRECTIONAL_FILTER = 'filteredMap.mrc'
OUTPUT_DIRECTIONAL_DISTRIBUTION = 'Resolution_Distribution.xmd'


class XmippProtFSO(ProtAnalysis3D):
    """    
    Given two half maps the protocol estimates Fourier Shell Occupancy to determine the global anisotropy of the map
    """
    _label = 'resolution fso'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.min_res_init = Float()
        self.max_res_init = Float()
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('halfVolumesFile', BooleanParam, default=True,
                      label="Are the half volumes stored with the input volume?",
                      help='Usually, the half volumes are stored as properties of '
                      'the input volume. If this is not the case, set this to '
                      'False and specify the two halves you want to use.')

        form.addParam('inputHalves', PointerParam, pointerClass='Volume',
                      label="Input Half Maps",
                      condition = 'halfVolumesFile',
                      help='Select a half maps for determining its '
                      ' resolution anisotropy and resolution.')

        form.addParam('half1', PointerParam, pointerClass='Volume',
                      condition = "not halfVolumesFile",
                      label="Half Map 1", important=True,
                      help='Select one map for determining the '
		      'directional FSC resolution.')

        form.addParam('half2', PointerParam, pointerClass='Volume',
                      condition = "not halfVolumesFile",
                      label="Half Map 2", important=True,
                      help='Select the second map for determining the '
                      'directional FSC resolution.')
        
        form.addParam('mask', PointerParam, pointerClass='VolumeMask', 
                      allowsNull=True, 
                      label="Mask", 
                      help='The mask determines which points are specimen'
                      ' and which are not')

        form.addParam('coneAngle', FloatParam, default=17.0,
                      expertLevel=LEVEL_ADVANCED, 
                      label="Cone Angle",
                      help='Angle between the axis of the cone and the generatrix. '
                           'An angle of 17 degrees is the best angle (see publication
                           'Vilas 2021) to measuare directional FSCs')
        
        form.addParam('estimate3DFSC', BooleanParam, default=True, 
                      label="Estimate 3DFSC and directional filtered map",
                      help='Set to estimate the 3DFSCD map, and applyting the '
                           ' 3DFSC as anisotropic filter to obtain a directional'
                           'filtered map.')

        form.addParam('threshold', FloatParam, expertLevel=LEVEL_ADVANCED, 
                      default=0.143, 
                      label="FSC Threshold",
                      help='Threshold for the fsc. By default the standard 0.143. '
                           'Other common thresholds are 0.5 and 0.3.')
        
        form.addParallelSection(threads = 4, mpi = 0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {OUTPUT_3DFSC: self._getExtraPath("3dFSC.mrc"),
                  OUTPUT_DIRECTIONAL_FILTER: self._getExtraPath("filteredMap.mrc"),
                  }
        self._updateFilenamesDict(myDict)


    def _insertAllSteps(self):
        self._createFilenameTemplates() 
        # Convert input into xmipp Metadata format
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('FSOestimationStep')
        self._insertFunctionStep('createOutputStep')

    def mrc_convert(self, fileName, outputFileName):
        """Check if the extension is .mrc, if not then uses xmipp to convert it
        """
        ext = getExt(fileName)
        if (ext != '.mrc') and (ext != '.map'):
            params = ' -i "%s"' % fileName
            params += ' -o "%s"' % outputFileName
            self.runJob('xmipp_image_convert', params)
            return outputFileName
        else:
            return fileName+':mrc'

    def convertInputStep(self):
        """ Read the input volume.
        """

        if self.halfVolumesFile:
            self.vol1Fn, self.vol2Fn = self.inputHalves.get().getHalfMaps().split(',')
        else:
            self.vol1Fn = self.mrc_convert(self.half1.get().getFileName(),
                                  self._getTmpPath('half1.mrc'))
            self.vol2Fn = self.mrc_convert(self.half2.get().getFileName(), 
                                  self._getTmpPath('half2.mrc'))
        if (self.mask.hasValue()):
            self.maskFn = self.mrc_convert(self.mask.get().getFileName(), 
                                  self._getExtraPath('mask.mrc'))

    def FSOestimationStep(self):
        import os
        fndir = self._getExtraPath("fsc") 

        os.mkdir(fndir) 	

        params = ' --half1 "%s"' % self.vol1Fn
        params += ' --half2 "%s"' % self.vol2Fn
        params += ' -o %s' % self._getExtraPath()
        if (self.halfVolumesFile):
            params += ' --sampling %f' % self.inputHalves.get().getSamplingRate()
        else:
            params += ' --sampling %f' % self.half1.get().getSamplingRate()

        if self.mask.hasValue():
            params += ' --mask "%s"' % self.maskFn

        params += ' --anglecone %f' % self.coneAngle.get()

        if self.estimate3DFSC.get():
            params += ' --threedfsc_filter'
        
        params += ' --threshold %s' % self.threshold.get()
        params += ' --threads %s' % self.numberOfThreads.get()
        
        self.runJob('xmipp_resolution_fso', params)
      
     
    def createOutputStep(self):
        volume=Volume()
        volume.setFileName(self._getExtraPath("3dFSC.mrc"))

        volume.setSamplingRate(self.half1.get().getSamplingRate())
        self._defineOutputs(fsc3D=volume)
        self._defineSourceRelation(self.half1, volume)

        volume.setFileName(self._getExtraPath("filteredMap.mrc"))

        volume.setSamplingRate(self.half1.get().getSamplingRate())
        self._defineOutputs(directionalFilteredMap=volume)
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
        return ['Vilas2021']
