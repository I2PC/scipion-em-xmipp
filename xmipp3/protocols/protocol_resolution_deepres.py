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
import xmipp3

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
OUTPUT_ORIGINAL_SIZE = 'deepRes_resolution_originalSize.vol'
OUTPUT_RESOLUTION_FILE_CHIMERA = 'chimera_resolution.vol'
METADATA_MASK_FILE = 'metadataresolutions'
FN_METADATA_HISTOGRAM = 'mdhist'


class XmippProtDeepRes(ProtAnalysis3D, xmipp3.XmippProtocol):
    """    
    Given a map the protocol assigns local resolutions to each voxel of the map.
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
                 MASK_DILATE: self._getTmpPath('Mask_dilate.mrc'),
                 OPERATE_VOL: self._getTmpPath('operateVolume.mrc'),
                 RESIZE_MASK: self._getTmpPath('binaryMask.mrc'),
                 RESIZE_VOL: self._getExtraPath('originalVolume.mrc'),
                 OUTPUT_RESOLUTION_FILE_CHIMERA: self._getExtraPath('chimera_resolution.vol'),
                 OUTPUT_RESOLUTION_FILE: self._getExtraPath('deepRes_resolution.mrc'),
                 OUTPUT_ORIGINAL_SIZE: self._getExtraPath('deepRes_resolution_originalSize.vol'),
                 FN_METADATA_HISTOGRAM: self._getExtraPath('hist.xmd')
                 }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format
        
        updateEnviron(self.gpuList.get())    
            
        self._createFilenameTemplates() 
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('resolutionStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep("createHistrogram")

    def convertInputStep(self):
        """ Read the input volume.
        """      
        volFn = self.inputVolume.get().getFileName()
        maskFn = self.Mask.get().getFileName()

        def _isMrc(fn):
            return getExt(fn) in ['.mrc', '.map']

        def _resize(i, o, factor):
            self.runJob('xmipp_image_resize',
                        f' -i {i} -o {o} --factor {factor}')

        def _filter(i, o, factor):
            self.runJob('xmipp_transform_filter',
                        f' -i {i} -o {o} --fourier low_pass {factor}')

        if _isMrc(volFn):
            volFn += ':mrc'

        if _isMrc(maskFn):
            maskFn += ':mrc'

        # Transform
        maskDilateFn = self._getFileName(MASK_DILATE)
        params = f' -i {maskFn} -o {maskDilateFn}'
        params += '  --binaryOperation dilation --size 1 '
        self.runJob('xmipp_transform_morphology', params)
             
        params = f' -i {volFn} --mult {maskDilateFn}'
        params += ' -o %s' % self._getFileName(OPERATE_VOL)
        self.runJob('xmipp_image_operate', params)

        # Resize maps
        if self.range == self.LOW_RESOL:
            sampling_new = 1.0
        else:
            sampling_new = 0.5               
        volSampling = self.inputVolume.get().getSamplingRate()
        samplingFactor = volSampling / sampling_new
        fourierValue = samplingFactor / 2

        maskResizeFn = self._getFileName(RESIZE_MASK)
        volOpFn = self._getFileName(OPERATE_VOL)
        volResizeFn = self._getFileName(RESIZE_VOL)

        if volSampling > sampling_new:
            # mask with sampling = 1.0
            _resize(maskFn, maskResizeFn, samplingFactor)
            # Original volume with sampling = 1.0
            _resize(volOpFn, volResizeFn, samplingFactor)
        else:
            # Mask with sampling=1.0
            _filter(maskFn, maskResizeFn, fourierValue)
            _resize(maskResizeFn, maskResizeFn, samplingFactor)
            # Original volume with sampling=1.0
            _filter(volOpFn, volResizeFn, fourierValue)
            _resize(volResizeFn, volResizeFn, samplingFactor)

        params = f' -i {maskResizeFn} -o {maskResizeFn}'
        params += ' --select below %f' % 0.15
        params += ' --substitute binarize'
        self.runJob('xmipp_transform_threshold', params)
                 
    def resolutionStep(self):
        if self.range == self.LOW_RESOL:
            # sampling_new = 1.0
            model = self.getModel("deepRes", "model_w13.h5")
        else:
            # sampling_new = 0.5
            model = self.getModel("deepRes", "model_w7.h5")
        params = ' -dl %s' % model
        params += ' -i  %s' % self._getFileName(RESIZE_VOL)
        params += ' -m  %s' % self._getFileName(RESIZE_MASK)
        params += ' -s  %f' % self.inputVolume.get().getSamplingRate()            
        params += ' -o  %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        if deepRes := os.environ.get('DEEPRES', None):
            self.runJob(deepRes, params, numberOfMpi=1)
        else:
            self.runJob("xmipp_deepRes_resolution", params, numberOfMpi=1,
                        env=self.getCondaEnv())
        
    def createHistrogram(self):
        M = float(self.max_res_init)
        m = float(self.min_res_init)
        range_res = round((M - m) * 4.0)
        params = ' -i %s' % self._getFileName(OUTPUT_RESOLUTION_FILE)
        params += ' --mask binary_file %s' % self._getFileName(RESIZE_MASK)
        params += ' --steps %f' % range_res
        params += ' --range %f %f' % (self.min_res_init, self.max_res_init)
        params += ' -o %s' % self._getFileName(FN_METADATA_HISTOGRAM)
        self.runJob('xmipp_image_histogram', params)
        
    def getMinMax(self, imageFile):
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        imgData = imgData[imgData!=0]
        min_res = round(np.amin(imgData) * 100) / 100
        max_res = round(np.amax(imgData) * 100) / 100
        median_res = round(np.median(imgData) * 100) / 100
        return min_res, max_res, median_res
    
    def createChimeraOutput(self, vol, value):
        Vx = xmipp3.Image(vol)
        V = Vx.getData()
        Zdim, Ydim, Xdim = V.shape
        for z in range(0,Zdim):
            for y in range(0,Ydim):
                for x in range(0,Xdim):
                    if V[z,y,x]==0:
                        V[z,y,x]=value    
        Vx.setData(V)                   
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

        volume = Volume()
        volume.setFileName(self._getFileName(OUTPUT_ORIGINAL_SIZE))

        volume.setSamplingRate(self.inputVolume.get().getSamplingRate())
        self._defineOutputs(resolution_Volume=volume)
        self._defineTransformRelation(self.inputVolume, volume)
            
        # Setting the min max and median for the summary
        imageFile = self._getFileName(OUTPUT_RESOLUTION_FILE)
        min_, max_, median_ = self.getMinMax(imageFile)
        self.min_res_init.set(round(min_*100)/100)
        self.max_res_init.set(round(max_*100)/100)
        self.median_res_init.set(round(median_*100)/100)        
        self._store(self.min_res_init)
        self._store(self.max_res_init)
        self._store(self.median_res_init)  
        vol_chimera = self.createChimeraOutput(imageFile, self.median_res_init)
        vol_chimera.write(self._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA))

                
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

