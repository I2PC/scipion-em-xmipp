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
    Calculates sharpened maps based on a given resolution map. The sharpening
    process enhances high-resolution features by boosting contrast where the
    local resolution allows to do so.

    AI Generated

    ## Overview

    The LocalDeblur Sharpening protocol performs local sharpening of a cryo-EM map
    using an associated local-resolution map.

    In cryo-EM, different regions of the same reconstruction may have different
    local resolution. A single global sharpening factor may therefore be too weak
    for well-resolved regions and too strong for poorly resolved regions.
    LocalDeblur addresses this by using a local-resolution map to guide the
    sharpening process spatially.

    The protocol enhances map features according to the local resolution estimated
    at each region. It can run for one or several iterations. When several
    iterations are used, the protocol sharpens the map, re-estimates local
    resolution using MonoRes, and then sharpens again using the updated resolution
    map.

    The main output is a sharpened volume. If several iterations are performed,
    the protocol outputs a set of sharpened volumes, one per iteration.

    ## Inputs and General Workflow

    The protocol requires:

    - an input cryo-EM map;
    - a local-resolution map.

    The input map and local-resolution map are converted to the format expected by
    Xmipp. The protocol checks and prepares the background of the resolution map,
    then creates a binary mask from it. The mask defines the region where local
    resolution is meaningful.

    The protocol then runs the local sharpening program using the input map,
    resolution map, sampling rate, regularization parameters, and number of
    threads. If iterative sharpening is requested, it alternates between
    sharpening and MonoRes local-resolution estimation.

    At the end, the last sharpened map is written as `sharpenedMap_last.mrc` and
    registered as the output.

    ## Input Map

    The **Input Map** parameter defines the cryo-EM volume to be sharpened.

    This map should be a reconstruction that the user wants to enhance for
    visualization, interpretation, or model building. The sampling rate of this
    volume is used throughout the calculation.

    The input map should correspond to the same coordinate frame and box as the
    resolution map. If the map and resolution map are not aligned or do not have
    compatible dimensions, the local sharpening will not be meaningful.

    ## Resolution Map

    The **Resolution Map** parameter provides the local-resolution information used
    to guide sharpening.

    LocalDeblur was specially designed to work with resolution maps obtained from
    MonoRes, but the protocol help also indicates that resolution maps from ResMap
    and BlocRes are accepted.

    The resolution map should contain local-resolution values in angstroms. Regions
    outside the molecule should ideally be zero or clearly distinguishable from the
    valid resolution region.

    The quality of the sharpening depends strongly on the quality of this
    resolution map. If the resolution map is noisy, poorly masked, or inconsistent
    with the input map, the sharpening may be unreliable.

    ## Background Check

    Before sharpening, the protocol checks the background values of the resolution
    map.

    If the minimum value of the resolution map is larger than a small threshold, it
    assumes that the background may not be properly set to zero. In that case, it
    sets the largest background-like values to zero using a thresholding operation.

    This step helps prepare imported resolution maps whose background convention
    may not be ideal for LocalDeblur.

    ## Binary Mask Creation

    The protocol creates a binary mask from the resolution map.

    It thresholds the resolution map so that voxels below 0.1 are converted into a
    binary mask. This mask is used by the MonoRes re-estimation step when iterative
    sharpening is performed.

    The mask should represent the region where the local-resolution values are
    valid. If the resolution map background is not correct, the mask may be wrong,
    and the sharpening may be affected.

    ## Lambda

    The **lambda** parameter is a regularization parameter.

    The method normally determines this parameter automatically. The form help
    indicates that lambda is directly related to convergence. Increasing it may
    accelerate convergence, but it also increases the risk of falling into local
    minima.

    The default value is 1. In the protocol, the user-provided lambda value is
    passed explicitly only in the first iteration when it differs from 1.

    Most users should keep the default value unless they have a specific reason to
    control convergence behavior.

    ## K Parameter

    The **K** parameter controls part of the sharpening behavior.

    The help text indicates that **K = 0.025** works well for all tested cases and
    that K should usually remain in the range **0.01–0.05**.

    For maps with FSC resolution worse than 6 Å, the help suggests that **K =
    0.01** can be a good alternative.

    This is an advanced parameter. Users should normally start with the default and
    adjust it only if the sharpening appears too weak, too strong, or unstable.

    ## Number of Iterations

    The **No. Iterations** parameter controls how many sharpening rounds are
    performed.

    If the value is **1**, the protocol performs a single local-sharpening step
    using the input resolution map.

    If the value is larger than 1, the protocol performs several rounds. In each
    round, the previously sharpened map is used as input, and MonoRes is run to
    estimate a new local-resolution map for the next sharpening round.

    If the value is **-1**, the protocol determines the stopping point
    automatically. It stops when the sharpening convergence criterion stabilizes or
    when the updated local-resolution range becomes sufficiently narrow.

    ## Iterative MonoRes Re-estimation

    When more than one iteration is requested, the protocol re-estimates local
    resolution after each sharpening step.

    The MonoRes calculation uses:

    - the current sharpened map;
    - the binary mask created from the resolution map;
    - the input map sampling rate;
    - a minimum resolution of twice the sampling rate;
    - a maximum resolution derived from the current resolution map;
    - a step of 0.25 Å;
    - significance of 0.95.

    This creates an updated local-resolution map that guides the next sharpening
    iteration.

    ## Automatic Stopping

    When the number of iterations is set to **-1**, the protocol stops
    automatically according to convergence criteria.

    It monitors the sharpening lambda/cost value stored in the metadata file. If
    the value changes only slightly between iterations, the protocol stops.

    It can also stop when the range of local-resolution values becomes smaller
    than a predefined threshold.

    Automatic stopping is useful when the user does not know how many sharpening
    iterations are appropriate, but the result should still be inspected carefully.

    ## Metadata Parameters

    During sharpening, the protocol writes a metadata file named `params.xmd`.

    This file stores parameters reported by the local sharpening program, including
    values used to evaluate convergence and the number of internal iterations.

    These metadata are mainly useful for advanced users who want to inspect the
    behavior of the sharpening process.

    ## Output Volume

    If only one sharpening iteration is performed, the main output is
    **outputVolume**.

    This output is the final sharpened map, written as:

    `sharpenedMap_last.mrc`

    The output volume keeps the sampling rate and origin of the input volume.

    It is annotated as the last sharpening epoch.

    ## Output Volumes

    If more than one sharpening iteration is performed, the protocol creates
    **outputVolumes**, a set containing the sharpened maps from the successive
    iterations.

    Intermediate maps are named according to their iteration, while the final map
    is stored as `sharpenedMap_last.mrc`.

    This output set allows the user to compare sharpening rounds and choose the map
    that provides the best balance between enhanced detail and noise amplification.

    ## Imported Resolution Maps

    If the protocol cannot find the internally updated MonoRes resolution map, it
    prints a warning indicating that the resolution map was probably imported.

    The warning notes that the ideal case is to calculate the resolution map
    previously in the same project using MonoRes.

    This is not necessarily an error, but it reminds the user that resolution-map
    compatibility is important for robust local sharpening.

    ## Interpreting the Sharpened Map

    The sharpened output should be interpreted as an enhanced version of the input
    map.

    LocalDeblur can improve the visibility of high-resolution features where the
    local-resolution map indicates that such features are supported. However,
    sharpening can also amplify noise or create misleading visual contrast if the
    local-resolution map is inaccurate or if parameters are too aggressive.

    Poorly resolved regions should not be overinterpreted simply because the map
    appears sharper. The sharpened map should always be compared with the original
    map, half maps, FSC information, and local-resolution estimates.

    ## Practical Recommendations

    Use a local-resolution map that corresponds exactly to the input map.

    Prefer MonoRes resolution maps when available, since LocalDeblur was designed
    for them.

    Start with the default K and lambda values.

    Use one iteration first. If the result is conservative or if iterative
    refinement is desired, try multiple iterations and compare the outputs.

    Use automatic iteration only when you are prepared to inspect the sequence of
    sharpened maps.

    For maps worse than about 6 Å global FSC resolution, consider a smaller K value
    such as 0.01, as suggested by the protocol help.

    Inspect the sharpened map together with the original map and validation data.
    Avoid interpreting isolated sharpened features without support from the
    experimental data.

    ## Final Perspective

    LocalDeblur Sharpening is a local map-enhancement protocol guided by a
    local-resolution map.

    For biological users, its main value is that it sharpens different regions of
    a cryo-EM map according to their estimated local resolution, instead of
    applying a single global sharpening factor.

    The protocol can improve interpretability and model-building guidance, but it
    should be used as part of a validation-aware workflow. The final map is an
    enhanced representation of the reconstruction, not a replacement for the
    original map or for independent validation.
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

