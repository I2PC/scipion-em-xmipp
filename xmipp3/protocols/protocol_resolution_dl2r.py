# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
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


DL2R_METHOD_URL = 'http://github.com/I2PC/scipion/wiki/XmippProtDl2r'
MODEL_DEEP_LEARNING = '/home/erney/git/xmipp-bundle/src/xmipp/applications/scripts/dl2r_resolution/model_w13.h5'
BINARY_MASK = 'binaryMask.vol' 
RESIZE_VOL = 'originalVolume.vol'
OPERATE_VOL = 'operateVolume.vol'
CHIMERA_RESOLUTION_VOL = 'dl2r_resolution.vol'
OUTPUT_RESOLUTION_FILE = 'resolutionMap'
#FN_FILTERED_MAP = 'filteredMap'
OUTPUT_RESOLUTION_FILE_CHIMERA = 'outputChimera'
METADATA_MASK_FILE = 'metadataresolutions'
FN_METADATA_HISTOGRAM = 'mdhist'


class XmippProtDl2r(ProtAnalysis3D):
    """    
    Given a map the protocol assigns local resolutions to each voxel of the map.
    """
    _label = 'deep learning Resolution DL2R'
    _lastUpdateVersion = VERSION_1_1
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.min_res_init = Float() 
        self.max_res_init = Float()
       
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')


        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True,
                      help='Select a volume for determining its '
                      'local resolution.')

        form.addParam('Mask', PointerParam, pointerClass='VolumeMask', 
                      allowsNull=True,
                      label="Mask", 
                      help='The mask determines which points are specimen'
                      ' and which are not')

#         group = form.addGroup('Extra parameters')
#         group.addParam('symmetry', StringParam, default='c1',
#                       label="Symmetry",
#                       help='Symmetry group. By default = c1.'
#                       'See [[http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry][Symmetry]]'
#                       'for a description of the symmetry groups format,' 
#                       'If no symmetry is present, give c1.')

#         group.addParam('filterInput', BooleanParam, default=False, 
#                       label="Filter input volume with local resolution?",
#                       help='The input map is locally filtered at'
#                       'the local resolution map.')
        
        #form.addParallelSection(threads = 1, mpi = 0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
                 BINARY_MASK: self._getExtraPath('binaryMask.vol'),
                 RESIZE_VOL: self._getExtraPath('originalVolume.vol'),
                 OUTPUT_RESOLUTION_FILE_CHIMERA: self._getExtraPath(CHIMERA_RESOLUTION_VOL),
                 OPERATE_VOL: self._getTmpPath('operateVolume.vol'),                 
                 OUTPUT_RESOLUTION_FILE: self._getExtraPath('dl2r_resolution.vol'),
                 FN_METADATA_HISTOGRAM: self._getExtraPath('hist.xmd')
                 }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format
        self._createFilenameTemplates() 
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('resizeStep')                
        self._insertFunctionStep('resolutionDL2RStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep("createHistrogram")

    def convertInputStep(self):
        """ Read the input volume.
        """      
        self.micsFn = self._getPath()
                
        self.volFn = self.inputVolume.get().getFileName()
        self.maskFn = self.Mask.get().getFileName()
           
        extVol = getExt(self.volFn)
        if (extVol == '.mrc') or (extVol == '.map'):
            self.volFn = self.volFn + ':mrc'              

        extMask = getExt(self.maskFn)
        if (extMask == '.mrc') or (extMask == '.map'):
            self.maskFn = self.maskFn + ':mrc'  
            
    def resizeStep(self):

        samplingFactor = float(self.inputVolume.get().getSamplingRate())/1.0
        fourierValue = float(self.inputVolume.get().getSamplingRate())/(2*1.0)   
             
        if self.inputVolume.get().getSamplingRate() > 1.0:
            #mask with sampling=1.0
            paramsResizeMask = ' -i %s' % self.maskFn
            paramsResizeMask += ' -o %s' % self._getFileName(BINARY_MASK)        
            paramsResizeMask += ' --factor %s' % samplingFactor
            self.runJob('xmipp_image_resize', paramsResizeMask ) 
            #Original volume with sampling=1.0
            paramsResizeVol = ' -i %s' % self.volFn
            paramsResizeVol += ' -o %s' % self._getFileName(RESIZE_VOL)        
            paramsResizeVol += ' --factor %s' % samplingFactor
            self.runJob('xmipp_image_resize', paramsResizeVol )             
        else:
            #mask with sampling=1.0
            paramsFilterMask = ' -i %s' % self.maskFn
            paramsFilterMask += ' -o %s' % self._getFileName(BINARY_MASK)
            paramsFilterMask += ' --fourier low_pass %s' % fourierValue            
            paramsResizeMask = ' -i %s' % self._getFileName(BINARY_MASK)  
            paramsResizeMask += ' -o %s' % self._getFileName(BINARY_MASK)           
            paramsResizeMask += ' --factor %s' % samplingFactor            
            self.runJob('xmipp_transform_filter', paramsFilterMask )
            self.runJob('xmipp_image_resize', paramsResizeMask )   
            #Original volume with sampling=1.0
            paramsFilterVol = ' -i %s' % self.volFn
            paramsFilterVol += ' -o %s' % self._getFileName(RESIZE_VOL)
            paramsFilterVol += ' --fourier low_pass %s' % fourierValue            
            paramsResizeVol = ' -i %s' % self._getFileName(RESIZE_VOL)  
            paramsResizeVol += ' -o %s' % self._getFileName(RESIZE_VOL)           
            paramsResizeVol += ' --factor %s' % samplingFactor            
            self.runJob('xmipp_transform_filter', paramsFilterVol )
            self.runJob('xmipp_image_resize', paramsResizeVol )                                
        
        params = ' -i %s' % self._getFileName(BINARY_MASK)
        params += ' -o %s' % self._getFileName(BINARY_MASK)
        params += ' --select below %f' % 0.2
        params += ' --substitute binarize'
             
        self.runJob('xmipp_transform_threshold', params )                       
            

    def resolutionDL2RStep(self):

        params  = ' -dl %s' % MODEL_DEEP_LEARNING
        params += ' -i  %s' % self._getFileName(RESIZE_VOL)
        params += ' -m  %s' % self._getFileName(BINARY_MASK)
        params += ' -s  %f' % 1.0            
        params += ' -o  %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
        #params += ' --chimera_volume %s' % self._getFileName(
        #                                           OUTPUT_RESOLUTION_FILE_CHIMERA)
        #params += ' --sym %s' % 'c1'#self.symmetry.get()
        #params += ' --significance %f' % self.significance.get()
        #params += ' --md_outputdata %s' % self._getFileName(METADATA_MASK_FILE)  
        #if self.filterInput.get():
        #    params += ' --filtered_volume %s' % self._getFileName(FN_FILTERED_MAP)
        #else:
        #    params += ' --filtered_volume %s' % ''

        self.runJob("xmipp_dl2r_resolution", params)
        
#         paramsOperate = ' -i  %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)   
#         paramsOperate += ' --mult  %s' % self._getFileName(BINARY_MASK)
#         paramsOperate += ' -o  %s' % self._getFileName(OPERATE_VOL)                     
#         self.runJob("xmipp_image_operate", paramsOperate)

    def createHistrogram(self):

        M = float(self.max_res_init)
        m = float(self.min_res_init)
#         M = 12.5
#         m = 2.5
        range_res = round((M - m)*4.0)

        params = ' -i %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
        params += ' --mask binary_file %s' % self._getFileName(BINARY_MASK)
        params += ' --steps %f' % (range_res)
        params += ' --range %f %f' % (self.min_res_init, self.max_res_init)
        params += ' -o %s' % self._getFileName(FN_METADATA_HISTOGRAM)

        self.runJob('xmipp_image_histogram', params)
        
        
#     def readMetaDataOutput(self):
#         mData = md.MetaData(self._getFileName(METADATA_MASK_FILE))
#         NvoxelsOriginalMask = float(mData.getValue(md.MDL_COUNT, mData.firstObject()))
#         NvoxelsOutputMask = float(mData.getValue(md.MDL_COUNT2, mData.firstObject()))
#         nvox = int(round(
#                 ((NvoxelsOriginalMask-NvoxelsOutputMask)/NvoxelsOriginalMask)*100))
#         return nvox

    def getMinMax(self, imageFile):
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        imgData = imgData[imgData!=0]
        min_res = round(np.amin(imgData) * 100) / 100
        max_res = round(np.amax(imgData) * 100) / 100
        return min_res, max_res

    def createOutputStep(self):
        volume=Volume()
        volume.setFileName(self._getFileName(OUTPUT_RESOLUTION_FILE))

        volume.setSamplingRate(1.0)
        self._defineOutputs(resolution_Volume=volume)
        self._defineTransformRelation(self.inputVolume, volume)
            
            
        #Setting the min max for the summary
#        imageFile = self._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA)
        imageFile = self._getFileName(OUTPUT_RESOLUTION_FILE)
        min_, max_ = self.getMinMax(imageFile)
        self.min_res_init.set(round(min_*100)/100)
        self.max_res_init.set(round(max_*100)/100)
        self._store(self.min_res_init)
        self._store(self.max_res_init)

#         if self.filterInput.get():
#             print 'Saving filtered map'
#             volume.setFileName(self._getFileName(FN_FILTERED_MAP))
# 
#             volume.setSamplingRate(self.inputVolume.get().getSamplingRate())
#             self._defineOutputs(outputVolume_Filtered=volume)
#             self._defineSourceRelation(self.inputVolume, volume)
            

                
    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + DL2L_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        summary.append("Highest resolution %.2f Å,   "
                       "Lowest resolution %.2f Å. \n" % (self.min_res_init,
                                                         self.max_res_init))
        return summary

    def _citations(self):
        return ['Ramirez-Aportela2018']

