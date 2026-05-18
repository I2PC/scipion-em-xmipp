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

import os
import numpy as np
from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, EnumParam, 
                                        StringParam, GPU_LIST)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.object import Float
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import getExt
from pwem.objects import Volume
from xmipp3.base import XmippProtocol
from pwem.emlib import Image


def updateEnviron(gpuNum):
    """ Create the needed environment for TensorFlow programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)

DEEPRES_METHOD_URL = 'https://github.com/I2PC/scipion/wiki/XmippProtDeepRes'
RESIZE_MASK = 'binaryMask.vol' 
MASK_DILATE = 'Mask_dilate.vol'  
RESIZE_VOL = 'originalVolume.vol'
OPERATE_VOL = 'operateVolume.vol'
#CHIMERA_RESOLUTION_VOL = 'deepRes_resolution.vol'
OUTPUT_RESOLUTION_FILE = 'resolutionMap'
OUTPUT_ORIGINAL_SIZE =  'deepRes_resolution_originalSize.vol'
OUTPUT_RESOLUTION_FILE_CHIMERA = 'chimera_resolution.vol'
METADATA_MASK_FILE = 'metadataresolutions'
FN_METADATA_HISTOGRAM = 'mdhist'


class XmippProtDeepRes(ProtAnalysis3D, XmippProtocol):
    """    
    A deep-learning-based method designed to estimate local resolution in
    cryo-electron microscopy maps. It analyzes 3D structural features directly
    from the map data to assign local resolution values to beach voxel of a
    3D map.. Using models trained on simulated and experimental data, DeepRes
    provides spatially resolved estimates that reflect variations in structural
    clarity across the map. The method captures subtle improvements from
    post-processing steps, offering a refined view of resolution distribution
    that supports better interpretation and validation of cryo-EM structures.

    AI Generated

    ## Overview

    The Local deepRes protocol estimates the local resolution of a cryo-EM map
    using a deep-learning model.

    Local resolution describes how map quality varies across different regions of
    a 3D reconstruction. Some parts of a structure may be highly resolved, while
    others may be more flexible, less ordered, or supported by fewer particles.
    Instead of assigning a single global resolution to the whole map, this protocol
    produces a 3D resolution map, where each voxel inside the mask is assigned an
    estimated local resolution value.

    The method uses trained DeepRes neural-network models. The user selects the
    expected resolution range, and the protocol applies the corresponding model to
    the masked input volume.

    The main output is a local-resolution volume at the original sampling rate of
    the input map.

    ## Inputs and General Workflow

    The protocol requires:

    - an input volume;
    - a binary mask;
    - an expected resolution range.

    The mask is first dilated slightly and multiplied by the input volume so that
    the analysis focuses on the specimen region. The volume and mask are then
    resized to the sampling rate expected by the selected DeepRes model.

    The protocol runs the deep-learning local-resolution estimator and produces a
    resolution map. This map is then resized back to the original dimensions and
    sampling rate of the input volume.

    Finally, the protocol computes basic resolution statistics and creates a
    histogram of local-resolution values inside the mask.

    ## Input Volume

    The **Input Volume** parameter defines the map whose local resolution will be
    estimated.

    This volume should be a 3D cryo-EM reconstruction with a correct sampling rate.
    The sampling rate is important because the protocol uses it when resizing the
    map and interpreting resolution values in angstroms.

    The input map should correspond to the structure covered by the mask. Strong
    artifacts, incorrect sharpening, extreme filtering, or poor masking may affect
    the local-resolution estimates.

    ## Mask

    The **Mask** parameter is required.

    The mask defines which voxels correspond to the specimen and which voxels are
    background. It should be a binary mask, with the specimen region included
    inside the mask and the background outside it.

    The protocol dilates the mask by one voxel before multiplying it by the input
    volume. This helps include boundary regions of the specimen during the
    analysis.

    The quality of the mask is important. If the mask is too tight, real density
    may be excluded and local-resolution estimates near the boundary may be
    distorted. If it is too loose, background noise may influence the analysis.

    ## Expected Resolution Range

    The **Expected resolutions range** parameter selects which trained DeepRes
    model is used.

    There are two options:

    **2.5 Å - 13.0 Å** uses the model intended for lower- to medium-resolution
    maps. In this mode, the protocol works internally at a sampling rate of 1.0 Å.

    **1.5 Å - 6.0 Å** uses the model intended for higher-resolution maps. In this
    mode, the protocol works internally at a sampling rate of 0.5 Å.

    The user should choose the range that best matches the expected resolution of
    the map. Choosing an inappropriate range may reduce the reliability of the
    local-resolution estimate.

    ## Resizing and Internal Sampling

    Before running DeepRes, the protocol resizes the input volume and mask to the
    sampling rate expected by the selected model.

    For the 2.5–13.0 Å range, the working sampling is 1.0 Å.

    For the 1.5–6.0 Å range, the working sampling is 0.5 Å.

    If the input sampling is coarser than the required working sampling, the
    protocol resizes the data. If the input sampling is finer, it applies
    low-pass filtering before resizing to avoid aliasing.

    After DeepRes finishes, the output resolution map is resized back to the
    original sampling rate and dimensions of the input volume.

    ## Thresholding the Mask

    After resizing, the protocol thresholds the mask to make it binary again.

    Values below 0.15 are selected and substituted using a binarization operation.
    This step ensures that the mask used by DeepRes clearly separates specimen and
    background after interpolation and resizing.

    A properly binarized mask helps the neural network focus on the relevant map
    region.

    ## DeepRes Resolution Estimation

    The core step runs the Xmipp DeepRes local-resolution estimator.

    The protocol provides the selected trained model, the resized volume, the
    resized binary mask, the input sampling rate, and the output file name.

    The result is a 3D volume in which voxel values represent estimated local
    resolution in angstroms.

    The calculation uses GPU-compatible TensorFlow execution, with GPU memory
    growth enabled to reduce memory-allocation problems.

    ## Output Resolution Volume

    The main output is **resolution_Volume**.

    This output is the DeepRes local-resolution map resized back to the original
    sampling rate of the input volume. It is registered as a Scipion Volume and
    linked to the input volume.

    Voxel values inside the mask represent estimated local resolution in angstroms.
    Lower values correspond to better local resolution, and higher values
    correspond to worse local resolution.

    The output can be visualized as a volume map or used in downstream protocols
    that analyze local-resolution variation.

    ## Chimera Visualization Volume

    The protocol also creates an auxiliary Chimera-oriented resolution volume.

    In this auxiliary map, background voxels with value zero are replaced by the
    median local-resolution value. This can make visualization in Chimera or
    ChimeraX easier by avoiding empty or zero-valued regions that may distort the
    color scale.

    This file is mainly a visualization aid. The main scientific output remains
    the local-resolution volume.

    ## Histogram and Resolution Statistics

    After producing the resolution map, the protocol computes basic statistics
    inside the nonzero region of the DeepRes output.

    The summary reports:

    - the median resolution;
    - the highest resolution;
    - the lowest resolution.

    Here, “highest resolution” corresponds to the smallest numerical resolution
    value in angstroms, while “lowest resolution” corresponds to the largest
    numerical value.

    The protocol also creates a histogram metadata file describing the distribution
    of local-resolution values inside the mask.

    These statistics are useful for summarizing the spatial resolution variation
    of the map.

    ## Interpreting the Local-Resolution Map

    The local-resolution map should be interpreted as a spatial estimate of map
    quality.

    Regions with lower angstrom values are estimated to contain better-resolved
    features. Regions with higher values are estimated to be less resolved, which
    may reflect flexibility, weaker density, preferred orientation effects,
    compositional heterogeneity, masking problems, or lower local signal.

    The map should be interpreted together with the original density, half-map FSC,
    global resolution, local map features, and biological knowledge.

    Local resolution is not a direct atomic certainty measure. It is an estimate of
    local map quality and should not be overinterpreted at single-voxel level.

    ## Practical Recommendations

    Use a mask that includes the full molecular density but avoids unnecessary
    background.

    Choose the expected resolution range according to the approximate global
    resolution of the map. Use the 2.5–13.0 Å range for medium/lower-resolution
    maps and the 1.5–6.0 Å range for high-resolution maps.

    Inspect the output resolution map together with the original cryo-EM density.
    Check whether well-ordered regions show better resolution than flexible or
    peripheral regions.

    Use the median, minimum, and maximum values as summary descriptors, but inspect
    the spatial distribution rather than relying only on global statistics.

    Be cautious near mask boundaries, flexible domains, low-occupancy regions, or
    areas with strong artifacts.

    If the output appears inconsistent with the visual quality of the map, review
    the mask, selected resolution range, input sampling rate, and map
    preprocessing.

    ## Final Perspective

    Local deepRes is a deep-learning-based local-resolution estimation protocol.

    For biological users, its main value is that it provides a spatial map of
    resolution variation across the reconstruction. This can help identify rigid
    well-resolved cores, flexible domains, weaker peripheral regions, and areas
    where interpretation should be more cautious.

    The output should be used as part of a broader validation workflow, together
    with FSC analysis, map inspection, local model fit, and biological
    interpretation.
    """
    _label = 'local deepRes'
    _lastUpdateVersion = VERSION_2_0
    _conda_env = 'xmipp_DLTK_v0.3'
    
    #RESOLUTION RANGE
    LOW_RESOL = 0
    HIGH_RESOL = 1
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.min_res_init = Float() 
        self.max_res_init = Float()
        self.median_res_init = Float()       
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        
        form.addHidden(GPU_LIST, StringParam, default='0',
                       label="Choose GPU ID",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")               

        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True, 
                      help='Select a volume for determining its '
                      'local resolution.')

        form.addParam('Mask', PointerParam, pointerClass='VolumeMask', 
                      important=True,
                      label="Mask", 
                      help='Binary mask. The mask determines which points are specimen'
                      ' and which are not')
        
        form.addParam('range', EnumParam, choices=[u'2.5Å - 13.0Å', u'1.5Å - 6.0Å'],
                      label="Expected resolutions range", default=self.LOW_RESOL,
                      display=EnumParam.DISPLAY_HLIST, 
                      help='The program uses a trained network to determine' 
                      ' resolutions between 2.5Å-13.0Å  or' 
                      ' resolutions between 1.5Å-6.0Å')         


    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
                 MASK_DILATE: self._getTmpPath('Mask_dilate.vol'),  
                 OPERATE_VOL: self._getTmpPath('operateVolume.vol'),                             
                 RESIZE_MASK: self._getTmpPath('binaryMask.vol'),                
                 RESIZE_VOL: self._getExtraPath('originalVolume.vol'),
                 OUTPUT_RESOLUTION_FILE_CHIMERA: self._getExtraPath('chimera_resolution.vol'),
#                 OUTPUT_RESOLUTION_FILE_CHIMERA: self._getExtraPath(CHIMERA_RESOLUTION_VOL),                                 
                 OUTPUT_RESOLUTION_FILE: self._getExtraPath('deepRes_resolution.vol'),
                 OUTPUT_ORIGINAL_SIZE: self._getExtraPath('deepRes_resolution_originalSize.vol'),
                 FN_METADATA_HISTOGRAM: self._getExtraPath('hist.xmd')
                 }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format

        if not self.useQueueForSteps() and not self.useQueue():
            updateEnviron(self.gpuList.get())
            
        self._createFilenameTemplates() 
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('transformStep')          
        self._insertFunctionStep('resizeInputStep')                        
        self._insertFunctionStep('resolutionStep')
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
            
         
        
    def transformStep(self):   
        params = ' -i %s' %  self.maskFn                        
        params += ' -o %s' % self._getFileName(MASK_DILATE)  
        params += '  --binaryOperation dilation --size 1 '    
        
        self.runJob('xmipp_transform_morphology', params ) 
             
        params2 = ' -i %s' % self.volFn      
        params2 += ' --mult %s' % self._getFileName(MASK_DILATE)                        
        params2 += ' -o %s' % self._getFileName(OPERATE_VOL)     
        
        self.runJob('xmipp_image_operate', params2 )  
                  
            
            
    def resizeInputStep(self):

        if self.range == self.LOW_RESOL:
            sampling_new = 1.0
        else:
            sampling_new = 0.5               
            
        samplingFactor = float(self.inputVolume.get().getSamplingRate())/sampling_new
        fourierValue = float(self.inputVolume.get().getSamplingRate())/(2*sampling_new)   
             
        if self.inputVolume.get().getSamplingRate() > sampling_new:
            #mask with sampling=1.0
            paramsResizeMask = ' -i %s' % self.maskFn
            paramsResizeMask += ' -o %s' % self._getFileName(RESIZE_MASK)        
            paramsResizeMask += ' --factor %s' % samplingFactor
            self.runJob('xmipp_image_resize', paramsResizeMask ) 
            #Original volume with sampling=1.0
            paramsResizeVol = ' -i %s' % self._getFileName(OPERATE_VOL)
            paramsResizeVol += ' -o %s' % self._getFileName(RESIZE_VOL)        
            paramsResizeVol += ' --factor %s' % samplingFactor
            self.runJob('xmipp_image_resize', paramsResizeVol )             
        else:
            #mask with sampling=1.0
            paramsFilterMask = ' -i %s' % self.maskFn
            paramsFilterMask += ' -o %s' % self._getFileName(RESIZE_MASK)
            paramsFilterMask += ' --fourier low_pass %s' % fourierValue            
            paramsResizeMask = ' -i %s' % self._getFileName(RESIZE_MASK)  
            paramsResizeMask += ' -o %s' % self._getFileName(RESIZE_MASK)           
            paramsResizeMask += ' --factor %s' % samplingFactor            
            self.runJob('xmipp_transform_filter', paramsFilterMask )
            self.runJob('xmipp_image_resize', paramsResizeMask )   
            #Original volume with sampling=1.0
            paramsFilterVol = ' -i %s' % self._getFileName(OPERATE_VOL)
            paramsFilterVol += ' -o %s' % self._getFileName(RESIZE_VOL)
            paramsFilterVol += ' --fourier low_pass %s' % fourierValue            
            paramsResizeVol = ' -i %s' % self._getFileName(RESIZE_VOL)  
            paramsResizeVol += ' -o %s' % self._getFileName(RESIZE_VOL)           
            paramsResizeVol += ' --factor %s' % samplingFactor            
            self.runJob('xmipp_transform_filter', paramsFilterVol )
            self.runJob('xmipp_image_resize', paramsResizeVol )                                
        
        params = ' -i %s' % self._getFileName(RESIZE_MASK)
        params += ' -o %s' % self._getFileName(RESIZE_MASK)
        params += ' --select below %f' % 0.15
        params += ' --substitute binarize'
             
        self.runJob('xmipp_transform_threshold', params )              
                 

    def resolutionStep(self):

        if self.range == self.LOW_RESOL:
#             sampling_new = 1.0
            MODEL_DEEP_1=self.getModel("deepRes", "model_w13.h5")
            params  = ' -dl %s' % MODEL_DEEP_1
        else:
#             sampling_new = 0.5
            MODEL_DEEP_2=self.getModel("deepRes", "model_w7.h5")
            params  = ' -dl %s' % MODEL_DEEP_2               
        params += ' -i  %s' % self._getFileName(RESIZE_VOL)
        params += ' -m  %s' % self._getFileName(RESIZE_MASK)
        params += ' -s  %f' % self.inputVolume.get().getSamplingRate()            
        params += ' -o  %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self.runJob("xmipp_deepRes_resolution", params, numberOfMpi=1,
                    env=self.getCondaEnv())
        

    def createHistrogram(self):

        M = float(self.max_res_init)
        m = float(self.min_res_init)
#         M = 12.5
#         m = 2.5
        range_res = round((M - m)*4.0)

        params = ' -i %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
        params += ' --mask binary_file %s' % self._getFileName(RESIZE_MASK)
        params += ' --steps %f' % (range_res)
        params += ' --range %f %f' % (self.min_res_init, self.max_res_init)
        params += ' -o %s' % self._getFileName(FN_METADATA_HISTOGRAM)

        self.runJob('xmipp_image_histogram', params)
        

    def getMinMax(self, imageFile):
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        imgData = imgData[imgData!=0]
        min_res = round(np.amin(imgData) * 100) / 100
        max_res = round(np.amax(imgData) * 100) / 100
        median_res= round(np.median(imgData) * 100) / 100 
        return min_res, max_res, median_res
    
    def createChimeraOutput(self, vol, value):
        Vx = Image(vol)
        V = Vx.getData()
        Zdim, Ydim, Xdim = V.shape
#        Vout = V
            
        for z in range(0,Zdim):
            for y in range(0,Ydim):
                for x in range(0,Xdim):
                    if V[z,y,x]==0:
                        V[z,y,x]=value    
        Vx.setData(V)                   
#        Vx.write(Vout)     
        return Vx
                                                        
    def createOutputStep(self):
        if self.range == self.LOW_RESOL:
            sampling_new = 1.0
        else:
            sampling_new = 0.5
            
        #convert size of output volume
        
        samplingFactor = sampling_new/float(self.inputVolume.get().getSamplingRate())
        fourierValue = sampling_new/(2*float(self.inputVolume.get().getSamplingRate()))
             
        if sampling_new > self.inputVolume.get().getSamplingRate():
            paramsResizeVol = ' -i %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
            paramsResizeVol += ' -o %s' % self._getFileName(OUTPUT_ORIGINAL_SIZE)        
            paramsResizeVol += ' --factor %s' % samplingFactor
            self.runJob('xmipp_image_resize', paramsResizeVol )             
        else:   
            paramsFilterVol = ' -i %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
            paramsFilterVol += ' -o %s' % self._getFileName(OUTPUT_ORIGINAL_SIZE)
            paramsFilterVol += ' --fourier low_pass %s' % fourierValue            
            paramsResizeVol = ' -i %s' % self._getFileName(OUTPUT_ORIGINAL_SIZE)  
            paramsResizeVol += ' -o %s' % self._getFileName(OUTPUT_ORIGINAL_SIZE)           
            paramsResizeVol += ' --factor %s' % samplingFactor            
            self.runJob('xmipp_transform_filter', paramsFilterVol )
            self.runJob('xmipp_image_resize', paramsResizeVol )
            
            
        volume=Volume()
        volume.setFileName(self._getFileName(OUTPUT_ORIGINAL_SIZE))

        volume.setSamplingRate(self.inputVolume.get().getSamplingRate())
        self._defineOutputs(resolution_Volume=volume)
        self._defineTransformRelation(self.inputVolume, volume)
            
        #Setting the min max and median for the summary
        imageFile = self._getFileName(OUTPUT_RESOLUTION_FILE)
        min_, max_, median_ = self.getMinMax(imageFile)
        self.min_res_init.set(round(min_*100)/100)
        self.max_res_init.set(round(max_*100)/100)
        self.median_res_init.set(round(median_*100)/100)        
        self._store(self.min_res_init)
        self._store(self.max_res_init)
        self._store(self.median_res_init)  
        
        #create Resolution Map to visialize in Chimera
        #vol_chimera=Volume()         
        vol_chimera = self.createChimeraOutput(
                   self._getFileName(OUTPUT_RESOLUTION_FILE),self.median_res_init) 
#        self.createChimeraOutput(self._getFileName(OUTPUT_RESOLUTION_FILE),
#                                 self.median_res_init, 
#                                 self._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA))
        vol_chimera.write(self._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA))    
#        vol_chimera.setFileName(self._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA))          
#        self.vol_chimera.setSamplingRate(sampling_new)  
#         self._defineOutputs(resolution_Volume=vol_chimera)
#         self._defineTransformRelation(self.inputVolume, vol_chimera)             
                
    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + DEEPRES_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        summary.append("Median resolution %.2f Å." % (self.median_res_init))        
        summary.append("Highest resolution %.2f Å,   "
                       "Lowest resolution %.2f Å. \n" % (self.min_res_init,
                                                         self.max_res_init))        
        return summary

    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are not errors. If some errors are found, a list with
        the error messages will be returned.
        """
        error=self.validateDLtoolkit(model="deepRes")
        return error
    
    def _citations(self):
        return ['Ramirez-Aportela et al., IUCrJ, 2019']

