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

from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, StringParam, 
                                        BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pyworkflow.em.protocol.protocol_3d import ProtAnalysis3D
from pyworkflow.object import Float
from pyworkflow.em import ImageHandler
from pyworkflow.utils import getExt
from pyworkflow.em.data import Volume
import pyworkflow.em.metadata as md


OUTPUT_RESOLUTION_FILE = 'resolutionMap'
FN_FILTERED_MAP = 'filteredMap'


class XmippProtResLocalFilter(ProtAnalysis3D):
    """    
    Given a map the protocol assigns local resolutions to each voxel of the map.
    """
    _label = 'local MonoTomo'
    _lastUpdateVersion = VERSION_1_1
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.min_res_init = Float() 
        self.max_res_init = Float()
       
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input volume/tomogram", important=True,
                      help='Select a volume or tomogram to be locally filtered '
                      'by local resolution values.')

        form.addParam('resVol', PointerParam, pointerClass='Volume',
                      label="Resolution volume", important=True,
                      help='Select a local resolution map to filter locally'
                      ' by its values.')
        
        form.addParallelSection(threads = 4, mpi = 0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
                 FN_MEAN_VOL: self._getExtraPath('mean_volume.vol'),
                 OUTPUT_MASK_FILE: self._getExtraPath("output_Mask.vol"),
                 OUTPUT_RESOLUTION_FILE_CHIMERA: self._getExtraPath(CHIMERA_RESOLUTION_VOL),
                 FN_FILTERED_MAP: self._getExtraPath('filteredMap.vol'),
                 OUTPUT_RESOLUTION_FILE: self._getExtraPath('mgresolution.vol'),
                 METADATA_MASK_FILE: self._getExtraPath('mask_data.xmd'),
                 FN_METADATA_HISTOGRAM: self._getExtraPath('hist.xmd'),
                 BINARY_MASK: self._getExtraPath('binarized_mask.vol'),
                 FN_GAUSSIAN_MAP: self._getExtraPath('gaussianfilted.vol'),
                                  }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format
        self._createFilenameTemplates() 
        self._insertFunctionStep('convertInputStep', )
        self._insertFunctionStep('localFilterStep')
        self._insertFunctionStep('createOutputStep')


    def convertInputStep(self):
        """ Read the input volume.
        """

        self.volFn = self.inputVolume.get().getFileName()
        self.resFn = self.inputVolume2.get().getFileName()
        
        extVol1 = getExt(self.vol1Fn)
        extVol2 = getExt(self.vol2Fn)
        if (extVol1 == '.mrc') or (extVol1 == '.map'):
            self.vol1Fn = self.vol1Fn + ':mrc'
        if (extVol2 == '.mrc') or (extVol2 == '.map'):
            self.vol2Fn = self.vol2Fn + ':mrc'
        

    def localFilterStep(self):
        # Number of frequencies
        max_ = self.maxRes.get()

        if self.stepSize.hasValue():
            freq_step = self.stepSize.get()
        else:
            freq_step = 0.25
  
        xdim = self.maskRadius()

        params = ' --vol %s' % self.volFn
        params += ' --resVol %s' % self.resFn
        params += ' --sampling_rate %f' % self.inputVolume.get().getSamplingRate()
        params += ' -o %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
        params += ' --filteredMap %s' % self._getFileName(FN_FILTERED_MAP)
        params += ' --threads %f' % 
       
        
            addParamsLine("  --vol <vol_file=\"\">               : Volume");
    addParamsLine("  --resvol <vol_file=\"\">                : Resolution map");
    addParamsLine("  -o <output=\"MGresolution.vol\">    : Local resolution volume (in Angstroms)");
    addParamsLine("  --filteredMap <output=\"filteredMap.vol\">    : Local resolution volume filtered (in Angstroms)");
    addParamsLine("  [--sampling_rate <s=1>]               : Sampling rate (A/px)");
    addParamsLine("  [--step <s=0.25>]                   : The resolution is computed at a number of frequencies between minimum and");
    addParamsLine("  [--significance <s=0.95>]           : The level of confidence for the hypothesis test.");
    addParamsLine("  [--threads <s=4>]                   : Number of threads");
        
          

        self.runJob('xmipp_resolution_localfilter', params)


    def createOutputStep(self):
        volume=Volume()
        volume.setFileName(self._getFileName(OUTPUT_RESOLUTION_FILE))

        volume.setSamplingRate(self.inputVolume.get().getSamplingRate())
        self._defineOutputs(resolution_Volume=volume)
        self._defineSourceRelation(self.inputVolume, volume)
        
        volume=Volume()
        volume.setFileName(self._getFileName(FN_FILTERED_MAP))
        volume.setSamplingRate(self.inputVolume.get().getSamplingRate())
        self._defineOutputs(outputVolume_Filtered=volume)
        self._defineSourceRelation(self.inputVolume, volume)




    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + MONORES_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        summary.append("Highest resolution %.2f Å,   "
                       "Lowest resolution %.2f Å. \n" % (self.min_res_init,
                                                         self.max_res_init))
        return summary

    def _citations(self):
        return ['Vilas2018']

