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
"""
Protocol to perform 3D Initial Volume with pca projection
"""

from os.path import join, exists, split
import os
import numpy as np

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, IntParam, GPU_LIST)
from pyworkflow.utils.path import (moveFile)
from pwem.protocols import ProtRefine3D
from pwem.objects import SetOfVolumes, Volume
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pwem.emlib.image import ImageHandler
import xmipp3
import xmippLib

from pwem import emlib
from xmipp3.convert import  writeSetOfParticles, readSetOfParticles, xmippToLocation

def updateEnviron(gpuNum):
    """ Create the needed environment for TensorFlow programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)


class XmippProtReconstructInitVolPca(ProtRefine3D, xmipp3.XmippProtocol):
    """This is a 3D initial volume"""
    _label = 'initial volume pca'
    _lastUpdateVersion = VERSION_2_0
    _conda_env = 'xmipp_pyTorch'
    
    def __init__(self, **args):
        ProtRefine3D.__init__(self, **args)
    #     self.resolutionHalf = Float()
    

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        
        form.addHidden(GPU_LIST, StringParam, default='0',
                       label="Choose GPU ID",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")


        form.addSection(label='Input')

        form.addParam('inputParticles', PointerParam, label="Experimental Images", important=True,
                      pointerClass='SetOfParticles, SetOfClasses2D, SetOfAverages',
                      help='Select a set of images at full resolution')
        form.addParam('inputVolumes', PointerParam, label="Reference volumes", allowsNull=True,
                      pointerClass='Volume', expertLevel=LEVEL_ADVANCED,
                      help='Reference volume')
        form.addParam('classes', IntParam, default=10,
                      label='Number of classes',
                      help='The number of classes for a multi reference refinement')
        form.addParam('particleRadius', FloatParam, default=-1,
                     label='Radius of particle (px)', 
                     help='This is the radius (in pixels) of the spherical mask covering the particle in the input images')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        form.addParam('classify', BooleanParam, default=False, condition='classes > 1',
                      label='Do you want clasify?',
                      help='If you set to *Yes*, the particles are divided between volume')
       
                  
        form.addSection(label='Pca training')
        
        form.addParam('resolution',FloatParam, label="max resolution", default=12,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef' ,FloatParam, label="% variance", default=0.75, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of variance to determine the number of coefficients to be considers (between 0-1).'
                      ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
      


        form.addParallelSection(threads=1, mpi=4)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        
        updateEnviron(self.gpuList.get()) 
        
        self.imgsOrigXmd = self._getExtraPath('images_original.xmd')
        self.imgsFnXmd = self._getTmpPath('images.xmd')
        self.imgsFn = self._getTmpPath('images.mrcs')
        self.refsFn = self._getTmpPath('references.mrcs')
        self.refsFnXmd = self._getTmpPath('references.xmd')
        self.sampling = self.inputParticles.get().getSamplingRate()
        self.size = self.inputParticles.get().getDimensions()[0]
        self.iterations = 20
        self.classes = self.classes.get()
        
 
        if self.particleRadius.get() == -1:
            radius = int(self.size/2)
        else:
            radius = int(self.particleRadius.get())

        
        self._insertFunctionStep('convertInputStep', self.inputParticles.get(), self.imgsOrigXmd, self.imgsFn)
        self._insertFunctionStep("pcaTraining", self.imgsFn, self.resolution.get())
        for cl in range (self.classes):
            refVol = self._getTmpPath('randomVol_class%s.mrc'%cl)+ ':mrc'
            if self.inputVolumes.get() is None:
                self._insertFunctionStep('initRandomVol', self.imgsOrigXmd, self._getTmpPath('random_class%s.xmd'%(cl)), refVol)
            else:
                img=ImageHandler()
                vol=self.inputVolumes.get()
                img.convert(vol, refVol)
        
        
        for iter in range(self.iterations):
                

            if iter < 10:
                angleGallery, angle, shift, maxShift = 12, 8, 3, 12
                # angleGallery, angle, shift, maxShift = 12, 8, 4, 20
            elif iter < 14:
                angleGallery, angle, shift, maxShift = 8, 6, 3, 12
                # angleGallery, angle, shift, maxShift = 8, 6, 3, 12
            elif iter < 17:
                angleGallery, angle, shift, maxShift = 6, 5, 3, 12
                # angleGallery, angle, shift, maxShift = 6, 5, 3, 9
            elif iter < 20:
                angleGallery, angle, shift, maxShift = 5, 5, 3, 12
                # angleGallery, angle, shift, maxShift = 5, 5, 2, 6
                
                
            
            # angleGallery, angle, shift, maxShift = self._parameters(iter)
            # print(angleGallery, angle, shift, maxShift)
                

            if self.classify and iter > 3:
                saveClass = True
            else:
                saveClass = False
            # saveClass = False
        
            applyShift = False
            # applyShift = True
                          
            refVol = [None] * self.classes  
            refIm = [None] * self.classes
            outXmd = [None] * self.classes 
            outVol = [None] * self.classes
            select = [None] * self.classes
            if iter > 0:
                # refVol = outVol
                # inputXmd = outXmd
                for cl in range(self.classes):
                    if saveClass == True and iter > 4 :
                        inputXmd = self._getExtraPath('output_iter%s_classes.xmd'%(iter))
                    else: 
                        inputXmd = self._getExtraPath('output_iter%s_class%s.xmd'%(iter, cl))
                        
                    refVol[cl] = self._getExtraPath('volume_iter%s_class%s.mrc'%(iter, cl)+ ':mrc')
                    outXmd[cl] = self._getExtraPath('output_iter%s_class%s.xmd'%(iter+1, cl))
                    outVol[cl] = self._getExtraPath('volume_iter%s_class%s.mrc'%(iter+1, cl)+ ':mrc')
                    refIm[cl] = self._getTmpPath('references_class%s'%(cl))
                    select[cl] = self._getExtraPath('output_select_iter%s_class%s.xmd'%(iter+1, cl))
            else:
                # applyShift = False
                inputXmd = self.imgsFnXmd
                for cl in range(self.classes):
                    refVol[cl] = self._getTmpPath('randomVol_class%s.mrc'%cl)+ ':mrc' 
                    outXmd[cl] = self._getExtraPath('output_iter%s_class%s.xmd'%(iter+1, cl))
                    outVol[cl] = self._getExtraPath('volume_iter%s_class%s.mrc'%(iter+1, cl)+ ':mrc')
                    refIm[cl] = self._getTmpPath('references_class%s'%(cl))
                    select[cl] = self._getExtraPath('output_select_iter%s_class%s.xmd'%(iter+1, cl))
            
                
            for cl in range(self.classes):
                self._insertFunctionStep("createGallery", angleGallery, refVol[cl], refIm[cl])
            
            self._insertFunctionStep("globalAlign", inputXmd, refIm[cl], outXmd[0], angle, shift, maxShift, applyShift, saveClass, radius, iter)   
            
            for cl in range(self.classes):
                # self._insertFunctionStep("reconstructVolume", outXmd[cl], outVol[cl], iter, self.resolution.get())

                if saveClass or iter < 5:
                    self._insertFunctionStep("reconstructVolume", outXmd[cl], outVol[cl], iter, self.resolution.get())
                else:
                    self._insertFunctionStep("reconstructVolume", select[cl], outVol[cl], iter, self.resolution.get())
                    

        # self._insertFunctionStep("createOutput", iter)
    

    #--------------------------- STEPS functions ---------------------------------------------------        
    def convertInputStep(self, input, outputOrig, outputConvert):
        writeSetOfParticles(input, outputOrig)
        
        args = ' -i  %s -o %s --save_metadata_stack '%(outputOrig, outputConvert)
        self.runJob("xmipp_image_convert",args, numberOfMpi=1) 
        #positivity
        # self._positivity(outputConvert)
        # blur
        # self._applyBlurring(outputConvert)
                
   
    def createGallery(self, angle, refVol, refIm):
        #--perturb 0.01
        args = ' -i  %s --sym %s --sampling_rate %s  -o %s -v 0'% \
                (refVol, self.symmetryGroup.get(), angle, refIm+'.mrcs')
        self.runJob("xmipp_angular_project_library", args, numberOfMpi=1)
        moveFile(refIm+'.doc', refIm+'.xmd')
        
        
    def initRandomVol(self, inputXMD, outputXMD, outputVOL):
        args = ' -s %s  -o %s --random_angles' %(inputXMD, outputXMD)
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_global_align_preprocess", args, numberOfMpi=1, env=env)
        
        program = 'xmipp_cuda_reconstruct_fourier'    
        args = '-i %s -o %s --sym c1  --max_resolution %s  --sampling %s --thr %s --device 0 -gpusPerNode 1 -threadsPerGPU 4 -v 0' %\
        (outputXMD, outputVOL, 0.5, self.sampling, self.numberOfMpi.get()) 
        self.runJob(program, args, numberOfMpi=self.numberOfMpi.get())
        
        self._applyCicularMask(outputVOL, 0)
        
    
    def pcaTraining(self, inputIm, resolutionTrain):
        args = ' -i %s  -s %s -hr %s -lr 530 -p %s -o %s/train_pca  --batchPCA'% \
                (inputIm, self.sampling, resolutionTrain, self.coef.get(), self._getExtraPath())

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_global_align_train", args, numberOfMpi=1, env=env)
        
        
    def globalAlign(self, inputXmd, refIm, outputXmd, angle, shift, MaxShift, applyShift, saveClass, rad, iter):
        args = ' -i %s -r %s -a %s -amax 180 -sh %s -msh %s -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt  -o %s -stExp %s  -stRef %s  -s %s -radius %s -nCl %s'% \
                (self.imgsFn, refIm+'.mrcs', angle, shift, MaxShift,\
                 self._getExtraPath(), self._getExtraPath(), outputXmd, inputXmd, refIm+'.xmd', self.sampling, rad, self.classes)
        if applyShift:
            args += ' --apply_shifts ' 
        if saveClass:
            args += ' --save_class '

        env=self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_initVol_pca", args, numberOfMpi=1, env=env)
        # self.runJob("xmipp_global_align", args, numberOfMpi=1, env=env)

        #Select class and create xmds
        if saveClass:
            self._extract_class_xmd(iter)
        elif iter > 4:    
            self._extract_select_xmd(iter)
        
    def reconstructVolume(self, input, output, iter, resol):
        
        program = 'xmipp_cuda_reconstruct_fourier'    
        args = '-i %s -o %s --sym %s  --max_resolution %s  --sampling %s --thr %s --device 0 -gpusPerNode 1 -threadsPerGPU 4 -v 0' %\
        (input, output, self.symmetryGroup.get(), 0.5, self.sampling, self.numberOfMpi.get()) 
        self.runJob(program, args, numberOfMpi=self.numberOfMpi.get())
        #filter
        # self._filterVolume(output, output, self.resolution.get())
        self._filterVolume(output, output, resol)
        #automatic mask
        if iter < 7:
            self._applyMaskThreshold(output)
        #positivity
        self._positivity(output)
        #gaussian mask
        # self._applyGaussianMask(output)
        #mask
        self._applyCicularMask(output, iter)      
    

    def createOutput(self, iter):      
        #output particle
        
        #modify column name to insert original mrcs name
        args = ' -i %s --operate  rename_column "image image1" '%self._getExtraPath('align_iter%s.xmd'%(iter+1))   
        self.runJob('xmipp_metadata_utilities', args, numberOfMpi=1)
        args = ' -i %s --set join %s itemId ' %(self._getExtraPath('align_iter%s.xmd'%(iter+1)), self.imgsOrigXmd)   
        self.runJob('xmipp_metadata_utilities', args, numberOfMpi=1)
        
        imgInitial = self.inputParticles.get()
        imgSet = self._getExtraPath('align_iter%s.xmd'%(iter+1))
        outSet = self._createSetOfParticles()    
        outSet.copyInfo(imgInitial)         
        EXTRA_LABELS = [
            #emlib.MDL_COST
        ]
         # Fill
        readSetOfParticles(
            imgSet,
            outSet,
            extraLabels=EXTRA_LABELS
        )
        outSet.setSamplingRate(self.inputParticles.get().getSamplingRate())       
        self._defineOutputs(outputParticles=outSet)
        self._defineSourceRelation(self.inputParticles, outSet)
        
        if self.createVolume or self.mode == self.REFINE:
            volume=Volume()
            volume.setFileName(self._getExtraPath('output_iter%s_avg_filt.mrc'%(iter+1)))
            volume.setSamplingRate(self.inputParticles.get().getSamplingRate())
            self._defineOutputs(outputVolume=volume)
            self._defineTransformRelation(self.inputParticles.get(), volume)
        
    
    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []    
        return errors
    
    def _warnings(self):
        warnings = []
        return warnings
    
    def _summary(self):
        summary = []
        summary.append("Symmetry: %s" % self.symmetryGroup.get())  
        return summary
    

    #--------------------------- UTILS functions --------------------------------------------
    
        
    def _reconstructAvg(self, half1, half2, output):
        args = ' -i %s --plus %s -o %s -v 0' %(half1, half2, output)
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)
        args = ' -i %s --mult 0.5' %output
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)
        
    def _computeFSC(self, input, ref):
        args = ' -i %s --ref %s --sampling_rate %s -o %s -v 0' %(input, ref, self.sampling, self._getExtraPath('fsc.xmd'))
        self.runJob('xmipp_resolution_fsc', args, numberOfMpi=1)
        
    def _computeResolution(self, mdFsc, threshold):
        resolution = 2 * self.sampling
        # Iterate until the FSC is under the threshold
        for objId in mdFsc:
            fsc = mdFsc.getValue(emlib.MDL_RESOLUTION_FRC, objId)
            if fsc < threshold:
                resolution = mdFsc.getValue(emlib.MDL_RESOLUTION_FREQREAL, objId)
                break
            
        return resolution
    
    def _filterVolume(self, input, output, resolution):
        # res = self.sampling / resolution
        # raisedw = self.sampling / 100
        args = ' -i %s -o %s --fourier low_pass %s --sampling %s -v 0'%(input, output, resolution, self.sampling)
        # args = ' -i %s -o %s --fourier low_pass %s %s -v 0'%(input, output, res, raisedw)
        self.runJob('xmipp_transform_filter', args, numberOfMpi=1)
        
    def _positivity(self, input):
        program = 'xmipp_transform_threshold'
        args = '-i %s --select below 0 --substitute value 0'%input
        self.runJob(program,args,numberOfMpi=1)
        
    def _applyCicularMask(self, input, iter):
        if iter < 5:
            radius = 48
        else:
            radius = 62
        program = 'xmipp_transform_mask'
        args = '-i %s --mask circular -%s'%(input, radius)
        self.runJob(program,args,numberOfMpi=1)
        
    def _applyGaussianMask(self, input):
        program = 'xmipp_transform_mask'
        args = '-i %s --mask gaussian -64'%input
        self.runJob(program,args,numberOfMpi=1)
        
    def _applyBlurring(self, input):
        program = 'xmipp_transform_filter'
        args = '-i %s --fourier real_gaussian 1'%(input)
        self.runJob(program,args,numberOfMpi=1)
        
    def _applyMaskThreshold(self, input):
        program ='xmipp_volume_segment'
        args = '-i %s -o %s --method otsu'%(input, self._getTmpPath('mask.mrc'))
        self.runJob(program,args,numberOfMpi=1)
        
        program ='xmipp_transform_morphology'
        args = '-i %s --binaryOperation removeSmall 50'%(self._getTmpPath('mask.mrc'))
        self.runJob(program,args,numberOfMpi=1)
        
        program ='xmipp_transform_morphology'
        args = '-i %s --binaryOperation keepBiggest'%(self._getTmpPath('mask.mrc'))
        self.runJob(program,args,numberOfMpi=1)
        
        program ='xmipp_image_operate'
        args = '-i %s  --mult %s '%(input, self._getTmpPath('mask.mrc:mrc'))
        self.runJob(program,args,numberOfMpi=1)
        
        
    def _extract_class_xmd(self, iter):
        
        generic_xmd = self._getExtraPath('output_iter%s_classes.xmd'%(iter+1))
        
        for cl in range(self.classes):
            
            out = self._getExtraPath('output_iter%s_class%s.xmd'%(iter+1, cl))
            
            program ='xmipp_metadata_utilities'
            args = '-i %s --query select "class==%s" -o %s '%(generic_xmd, cl, out)
            self.runJob(program,args,numberOfMpi=1)
            
            
    def _extract_select_xmd(self, iter):
                
        for cl in range(self.classes):
            
            generic_xmd = self._getExtraPath('output_iter%s_class%s.xmd'%(iter+1, cl))
            out = self._getExtraPath('output_select_iter%s_class%s.xmd'%(iter+1, cl))
            
            program ='xmipp_metadata_utilities'
            args = '-i %s --query select "sel==1" -o %s '%(generic_xmd,  out)
            self.runJob(program,args,numberOfMpi=1)
            
        
    def _parameters(self, iter):
        
        if iter < 5:
            angleGallery, angle, shift, maxShift = 12, 8, 4, 20
        elif iter < 10:
            angleGallery, angle, shift, maxShift = 8, 6, 3, 12
        elif iter < 15:
            angleGallery, angle, shift, maxShift = 6, 5, 3, 9
        elif iter < 20:
            angleGallery, angle, shift, maxShift = 5, 5, 2, 6
        # elif iter < 10:
        #     angleGallery, angle, shift, maxShift = 4, 4, 2, 8
            
        return(angleGallery, angle, shift, maxShift)
        


        
        
    