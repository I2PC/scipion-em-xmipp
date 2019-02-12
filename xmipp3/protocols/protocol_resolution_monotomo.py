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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, StringParam, 
                                        BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pyworkflow.em.protocol.protocol_3d import ProtAnalysis3D
from pyworkflow.object import Float
from pyworkflow.em import ImageHandler
from pyworkflow.utils import getExt
from pyworkflow.em.data import Volume
import pyworkflow.em.metadata as md


CHIMERA_RESOLUTION_VOL = 'MG_Chimera_resolution.vol'
MONORES_METHOD_URL = 'http://github.com/I2PC/scipion/wiki/XmippProtMonoRes'
OUTPUT_RESOLUTION_FILE = 'resolutionMap'
FN_FILTERED_MAP = 'filteredMap'
OUTPUT_RESOLUTION_FILE_CHIMERA = 'outputChimera'
OUTPUT_MASK_FILE = 'outputmask'
FN_MEAN_VOL = 'meanvol'
METADATA_MASK_FILE = 'metadataresolutions'
FN_METADATA_HISTOGRAM = 'mdhist'
BINARY_MASK = 'binarymask'
FN_GAUSSIAN_MAP = 'gaussianfilter'


class XmippProtMonoTomo(ProtAnalysis3D):
    """    
    Given a map the protocol assigns local resolutions to each voxel of the map.
    """
    _label = 'local MonoTomo'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.min_res_init = Float() 
        self.max_res_init = Float()
       
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Odd tomogram", important=True,
                      help='Select a volume for determining its '
                      'local resolution.')

        form.addParam('inputVolume2', PointerParam, pointerClass='Volume',
                      label="Even Tomogram", important=True,
                      help='Select a second volume for determining a '
                      'local resolution.')
        
        form.addParam('useMask', BooleanParam, default=False,
                      label="Use mask?", 
                      help='The mask determines which points are specimen'
                      ' and which are not.')
        
        form.addParam('Mask', PointerParam, pointerClass='VolumeMask', 
                      condition='useMask', allowsNull=True, 
                      label="Binary Mask", 
                      help='The mask determines which points are specimen'
                      ' and which are not')

        group = form.addGroup('Extra parameters')

        line = group.addLine('Resolution Range (Å)',
                            help="If the user knows the range of resolutions or"
                                " only a range of frequencies needs to be analysed." 
                                "If Low is empty MonoRes will try to estimate the range. "
                                "it should be better if a range is provided")
        
        group.addParam('significance', FloatParam, default=0.95, 
                       expertLevel=LEVEL_ADVANCED,
                      label="Significance",
                      help='Relution is computed using hipothesis tests, '
                      'this value determines the significance of that test')
        
        line.addParam('minRes', FloatParam, default=0, label='High')
        line.addParam('maxRes', FloatParam, allowsNull=True, label='Low')
        line.addParam('stepSize', FloatParam, allowsNull=True,
                      expertLevel=LEVEL_ADVANCED, label='Step')

        
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
        self._insertFunctionStep('resolutionMonogenicSignalStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep("createHistrogram")

    def convertInputStep(self):
        """ Read the input volume.
        """

        self.vol1Fn = self.inputVolume.get().getFileName()
        self.vol2Fn = self.inputVolume2.get().getFileName()
        extVol1 = getExt(self.vol1Fn)
        extVol2 = getExt(self.vol2Fn)
        if (extVol1 == '.mrc') or (extVol1 == '.map'):
            self.vol1Fn = self.vol1Fn + ':mrc'
        if (extVol2 == '.mrc') or (extVol2 == '.map'):
            self.vol2Fn = self.vol2Fn + ':mrc'
        
        if self.useMask.get() is True:
            if (not self.Mask.hasValue()):
                self.ifNomask(self.vol1Fn)
            else:
                self.maskFn = self.Mask.get().getFileName()
                
            extMask = getExt(self.maskFn)
            
            if (extMask == '.mrc') or (extMask == '.map'):
                self.maskFn = self.maskFn + ':mrc'
                
            if self.Mask.hasValue():
                params = ' -i %s' % self.maskFn
                params += ' -o %s' % self._getFileName(BINARY_MASK)
                params += ' --select below %f' % 0.5# Mask threshold = 0.5 self.maskthreshold.get()
                params += ' --substitute binarize'
                 
                self.runJob('xmipp_transform_threshold', params)
        

    def ifNomask(self, fnVol):
        xdim, _ydim, _zdim = self.inputVolume.get().getDim()
        params = ' -i %s' % fnVol
        params += ' -o %s' % self._getFileName(FN_GAUSSIAN_MAP)
        setsize = 0.02*xdim
        params += ' --fourier real_gaussian %f' % (setsize)
     
        self.runJob('xmipp_transform_filter', params)
        img = ImageHandler().read(self._getFileName(FN_GAUSSIAN_MAP))
        imgData = img.getData()
        max_val = np.amax(imgData)*0.05
         
        params = ' -i %s' % self._getFileName(FN_GAUSSIAN_MAP)
        params += ' --select below %f' % max_val
        params += ' --substitute binarize'
        params += ' -o %s' % self._getFileName(BINARY_MASK)
     
        self.runJob('xmipp_transform_threshold', params)
        
        self.maskFn = self._getFileName(BINARY_MASK)


    def maskRadius(self):

        xdim, _ydim, _zdim = self.inputVolume.get().getDim()
        xdim = xdim*0.5

        return xdim
    

    def resolutionMonogenicSignalStep(self):
        # Number of frequencies
        max_ = self.maxRes.get()

        if self.stepSize.hasValue():
            freq_step = self.stepSize.get()
        else:
            freq_step = 0.25
  
        xdim = self.maskRadius()

        params = ' --vol %s' % self.vol1Fn
        params += ' --vol2 %s' % self.vol2Fn
        params += ' --meanVol %s' % self._getFileName(FN_MEAN_VOL)
        
        if self.useMask.get() is True:
            params += ' --mask %s' % self._getFileName(BINARY_MASK)
        
        params += ' --sampling_rate %f' % self.inputVolume.get().getSamplingRate()

        params += ' --minRes %f' % self.minRes.get()
        params += ' --maxRes %f' % max_
            
        params += ' --step %f' % freq_step
        params += ' --mask_out %s' % self._getFileName(OUTPUT_MASK_FILE)
        params += ' -o %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
        params += ' --filteredMap %s' % self._getFileName(FN_FILTERED_MAP)
        params += ' --chimera_volume %s' % self._getFileName(
                                                    OUTPUT_RESOLUTION_FILE_CHIMERA)
        params += ' --significance %f' % self.significance.get()
        params += ' --md_outputdata %s' % self._getFileName(METADATA_MASK_FILE)  

        self.runJob('xmipp_resolution_monotomo', params)



    def createHistrogram(self):

        M = float(self.max_res_init)
        m = float(self.min_res_init)
        range_res = round((M - m))

        params = ' -i %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
        params += ' --mask binary_file %s' % self._getFileName(OUTPUT_MASK_FILE)
        params += ' --steps %f' % (range_res)
        params += ' --range %f %f' % (self.min_res_init, self.max_res_init)
        params += ' -o %s' % self._getFileName(FN_METADATA_HISTOGRAM)

        self.runJob('xmipp_image_histogram', params)
        
        
    def readMetaDataOutput(self):
        mData = md.MetaData(self._getFileName(METADATA_MASK_FILE))
        NvoxelsOriginalMask = float(mData.getValue(md.MDL_COUNT, mData.firstObject()))
        NvoxelsOutputMask = float(mData.getValue(md.MDL_COUNT2, mData.firstObject()))
        nvox = int(round(
                ((NvoxelsOriginalMask-NvoxelsOutputMask)/NvoxelsOriginalMask)*100))
        return nvox

    def getMinMax(self, imageFile):
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        min_res = round(np.amin(imgData) * 100) / 100
        max_res = round(np.amax(imgData) * 100) / 100
        return min_res, max_res

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
            
            
        #Setting the min max for the summary
        imageFile = self._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA)
        min_, max_ = self.getMinMax(imageFile)
        self.min_res_init.set(round(min_*100)/100)
        self.max_res_init.set(round(max_*100)/100)
        self._store(self.min_res_init)
        self._store(self.max_res_init)



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

