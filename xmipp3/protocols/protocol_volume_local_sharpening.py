# -*- coding: utf-8 -*-
# **************************************************************************
# *
#* Authors:    Erney Ramirez-Aportela,                                    eramirez@cnb.csic.es
# *             Jose Luis Vilas,                                           jlvilas@cnb.csic.es
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
from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, FloatParam, IntParam,
                                        LEVEL_ADVANCED)
from pwem.protocols import ProtAnalysis3D
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import getExt
from pwem.objects import Volume
import numpy as np
import os
import pwem.emlib.metadata as md
from pwem.emlib.metadata import (MDL_COST, MDL_ITER, MDL_SCALE)
from ntpath import dirname
from os.path import exists

LOCALDEBLUR_METHOD_URL='https://github.com/I2PC/scipion/wiki/XmippProtLocSharp' 

class XmippProtLocSharp(ProtAnalysis3D):
    """    
    Calculates sharpened maps based on a given resolution map. The sharpening process enhances high-resolution features by boosting contrast where the local resolution allows to do so.
    """
    _label = 'localdeblur sharpening'
    _lastUpdateVersion = VERSION_1_1
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')      
        
        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Map", important=True,
                      help='Select a volume for sharpening.')

        form.addParam('resolutionVolume', PointerParam, pointerClass='Volume',
                      label="Resolution Map", important=True,
                      help='Select a local resolution map.'
                      ' LocalDeblur has been specially designed to work with'
                      ' resolution maps obtained with MonoRes,' 
                      ' however resolution map from ResMap and BlocRes are also accepted.')
                
        form.addParam('const', FloatParam, default=1, 
                      expertLevel=LEVEL_ADVANCED,
                      label="lambda",
                      help='Regularization Param.' 
                        'The method determines this parameter automatically.'
                        ' This parameter is directly related to the convergence.'
                        ' Increasing it would accelerate the convergence,' 
                        ' however it presents the risk of falling into local minima.')
        form.addParam('K', FloatParam, default=0.025, 
                      expertLevel=LEVEL_ADVANCED,
                      label="K",
                      help='K = 0.025 works well for all tested cases.'
                      ' K should be in the 0.01-0.05 range.'
                      ' For maps with FSC resolution lower than 6Å,'
                      ' K = 0.01 can be a good alternative.')
        form.addParam('Niter', IntParam, default=1,
                      expertLevel=LEVEL_ADVANCED,
                      label="No. Iterations",
                      help='If this number is larger than 1, for each iteration, the previous map is sharpened and the '
                           'local resolution reestimated for a new round. Set to a fixed number or -1 for automatic '
                           'determination')
        
        form.addParallelSection(threads = 4, mpi = 0)
        
    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict= {           
                'BINARY_MASK': self._getTmpPath('binaryMask.mrc'),
                'OUTPUT_RESOLUTION_FILE': self._getTmpPath('monoresResolutionMap.mrc'),                
                'METADATA_PARAMS_SHARPENING': self._getTmpPath('params.xmd'),                                                                                 
                }
        self._updateFilenamesDict(myDict) 
    

    def _insertAllSteps(self):
        self.iteration = 0
        self._createFilenameTemplates() 
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('checkBackgroundStep')
        self._insertFunctionStep('createMaskStep')      
        self._insertFunctionStep('sharpeningAndMonoResStep')       
        self._insertFunctionStep('createOutputStep')


    def convertInputStep(self):
        """ Read the input volume.
        """
        self.volFn = self.inputVolume.get().getFileName()
        self.resFn = self.resolutionVolume.get().getFileName()      
        extVol = getExt(self.volFn)
        extRes = getExt(self.resFn)        
        if (extVol == '.mrc') or (extVol == '.map'):
            self.volFn = self.volFn + ':mrc'
        if (extRes == '.mrc') or (extRes == '.map'):
            self.resFn = self.resFn + ':mrc'     

    def checkBackgroundStep(self): 
        
        initRes=self.resolutionVolume.get().getFileName() 

        img = ImageHandler().read(initRes)
        imgData = img.getData()
        max_value = np.amax(imgData)
        min_value = np.amin(imgData) 
        # print("minvalue %s  y maxvalue  %s"  % (min_value, max_value))
        
        if (min_value > 0.01):
            params = ' -i %s' % self.resFn  
            params += ' -o %s' % self.resFn    
            params += ' --select above %f' % (max_value-1)   
            params += ' --substitute value 0'
            
            self.runJob('xmipp_transform_threshold', params)  
                             
    def createMaskStep(self):
        
        params = ' -i %s' % self.resFn
        params += ' -o %s' % self._getFileName('BINARY_MASK')
        params += ' --select below %f' % 0.1
        params += ' --substitute binarize'
             
        self.runJob('xmipp_transform_threshold', params)
        
    
    def MonoResStep(self, iter):
        sampling = self.inputVolume.get().getSamplingRate()
        
        if (iter == 1):
            resFile = self.resolutionVolume.get().getFileName()
        else:
            resFile = self._getFileName('OUTPUT_RESOLUTION_FILE')
            
        pathres=dirname(resFile)
                       
        img = ImageHandler().read(resFile)
        imgData = img.getData()
        max_res = np.amax(imgData)
        
        significance = 0.95

        mtd = md.MetaData()
        if exists(os.path.join(pathres,'mask_data.xmd')):    
            mtd.read(os.path.join(pathres,'mask_data.xmd')) 
            radius = mtd.getValue(MDL_SCALE,1)
        else:
            xdim, _ydim, _zdim = self.inputVolume.get().getDim()
            radius = xdim*0.5 
        
        params = ' --vol %s' % self._getExtraPath('sharpenedMap_'+str(iter)+'.mrc')
        params += ' --mask %s' % self._getFileName('BINARY_MASK')
        params += ' --sampling_rate %f' % sampling
        params += ' --minRes %f' % (2*sampling)
        params += ' --maxRes %f' % max_res           
        params += ' --step %f' % 0.25
        params += ' -o %s' % self._getTmpPath()
        params += ' --significance %f' % significance
        params += ' --threads %i' % self.numberOfThreads.get()

        self.runJob('xmipp_resolution_monogenic_signal', params)
        
        
    def sharpenStep(self, iter):   
        sampling = self.inputVolume.get().getSamplingRate()   
        #params = ' --vol %s' % self.volFn
        if (iter == 1):
            params = ' --resolution_map %s' % self.resFn
        else:
            params = ' --resolution_map %s' % self._getFileName(
                                                   'OUTPUT_RESOLUTION_FILE')
            
        params += ' --sampling %f' % sampling
        params += ' -n %i' %  self.numberOfThreads.get()   
        params += ' -k %f' %  self.K
        params += ' --md %s' % self._getFileName('METADATA_PARAMS_SHARPENING')  
        if (iter == 1 and self.const!=1):
             params += ' -l %f' % self.const
         
        if (iter == 1):
            invol = ' --vol %s' % self.volFn
        else:
            invol = ' --vol %s' % self._getExtraPath('sharpenedMap_'+str(iter-1)+'.mrc')
            
        params +=invol    
            
        self.runJob("xmipp_volume_local_sharpening  -o %s"
                    %(self._getExtraPath('sharpenedMap_'+str(iter)+'.mrc')), params)
        self.runJob("xmipp_image_header -i %s -s %f"
                    %(self._getExtraPath('sharpenedMap_'+str(iter)+'.mrc'), sampling), "")


    def sharpeningAndMonoResStep(self):
        last_Niters=-1
        last_lambda_sharpening = 1e38
        nextIter = True
        maxNiters=self.Niter.get()

        while nextIter is True:
            self.iteration = self.iteration + 1
            # print("iteration")
            print('\n====================\n'
                  'Iteration  %s'  % (self.iteration))
            self.sharpenStep(self.iteration)           
            mtd = md.MetaData()
            mtd.read(self._getFileName('METADATA_PARAMS_SHARPENING'))
            
            lambda_sharpening = mtd.getValue(MDL_COST,1)
            Niters = mtd.getValue(MDL_ITER,1)
            
            if (maxNiters<0 and (abs(lambda_sharpening - last_lambda_sharpening)<= 0.2)) or \
               (maxNiters>0 and self.iteration==maxNiters):
                nextIter = False   
                break

            last_Niters = Niters
            last_lambda_sharpening = lambda_sharpening
            
            self.MonoResStep(self.iteration)
            
            imageFile = self._getFileName('OUTPUT_RESOLUTION_FILE')

            img = ImageHandler().read(imageFile)
            imgData = img.getData()
            max_res = np.amax(imgData)
            min_res = 2*self.inputVolume.get().getSamplingRate()
        
            if maxNiters<0 and (max_res-min_res<0.75):
                nextIter = False

        os.rename(self._getExtraPath('sharpenedMap_' + str(self.iteration) + '.mrc'),
                  self._getExtraPath('sharpenedMap_last.mrc'))
        
        resFile = self.resolutionVolume.get().getFileName()        
        pathres=dirname(resFile)
        if  not exists(self._getFileName('OUTPUT_RESOLUTION_FILE')):     

            print('\n====================\n' 
                  ' WARNING---This is not the ideal case because resolution map has been imported.'
                  ' The ideal case is to calculate it previously' 
                  ' in the same project using MonoRes.'
                  '\n====================\n')
  
             
    def createOutputStep(self):
        if self.iteration>1:
            volumesSet = self._createSetOfVolumes()
            volumesSet.setSamplingRate(self.inputVolume.get().getSamplingRate())
            for i in range(self.iteration):
                vol = Volume()
                vol.setOrigin(self.inputVolume.get().getOrigin(True))
                if (self.iteration > (i + 1)):
                    vol.setLocation(i, self._getExtraPath('sharpenedMap_%d.mrc' % (i + 1)))
                    vol.setObjComment("Sharpened Map, \n Epoch %d" % (i + 1))
                else:
                    vol.setLocation(i, self._getExtraPath('sharpenedMap_last.mrc'))
                    vol.setObjComment("Sharpened Map, \n Epoch last")
                volumesSet.append(vol)
                self._defineOutputs(outputVolumes=volumesSet)
                self._defineSourceRelation(self.inputVolume, volumesSet)
        else:
            vol = Volume()
            vol.setSamplingRate(self.inputVolume.get().getSamplingRate())
            vol.setOrigin(self.inputVolume.get().getOrigin(True))
            vol.setLocation(self._getExtraPath('sharpenedMap_last.mrc'))
            vol.setObjComment("Sharpened Map, \n Epoch last")
            self._defineOutputs(outputVolume=vol)
            self._defineSourceRelation(self.inputVolume, vol)

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'sharpened_map'):
            messages.append(
                'Information about the method/article in ' + LOCALDEBLUR_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []
        summary.append("LocalDeblur Map")
        return summary

    def _citations(self):
        return ['Ramirez-Aportela 2018']

