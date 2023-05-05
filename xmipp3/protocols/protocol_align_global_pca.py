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
Protocol to perform global alignment
"""

from os.path import join, exists, split
import os

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, IntParam, EnumParam,
                                        NumericListParam, USE_GPU, GPU_LIST)
from pyworkflow.utils.path import (cleanPath, makePath, copyFile, moveFile,
                                   createLink, cleanPattern)
from pwem.protocols import ProtRefine3D
from pwem.objects import SetOfVolumes, Volume
from pwem.emlib.metadata import getFirstRow, getSize
from pyworkflow.object import Float
from pyworkflow.utils.utils import getFloatListFromValues
from pwem.emlib.image import ImageHandler
import pwem.emlib.metadata as md
import xmipp3

from pwem import emlib
from xmipp3.convert import  writeSetOfParticles, readSetOfParticles, xmippToLocation

def updateEnviron(gpuNum):
    """ Create the needed environment for TensorFlow programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)


class XmippProtAlignGlobalPca(ProtRefine3D, xmipp3.XmippProtocol):
    """This is a 3D global refinement protocol"""
    _label = 'alignPca'
    _lastUpdateVersion = VERSION_2_0
    # _conda_env = 'flexutils-tensorflow'
    _conda_env = 'xmipp_pyTorch'
    
    # def __init__(self, **args):
    #     ProtRefine3D.__init__(self, **args)
    #     self.resolutionHalf = Float()
    

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")


        form.addSection(label='Input')

        form.addParam('inputParticles', PointerParam, label="Experimental Images", important=True,
                      pointerClass='SetOfParticles', allowsNull=True,
                      help='Select a set of images at full resolution')
        # form.addParam('inputReferences', PointerParam, label="References Images", important=True,
        #               pointerClass='SetOfParticles', allowsNull=True,
        #               help='Select a set of references at full resolution')
        form.addParam('inputVolume', PointerParam, label="Initial volumes", important=True,
                      pointerClass='Volume', allowsNull=True,
                      help='Select a initial volume . ')
        form.addParam('corectCtf', BooleanParam, default=True,
                      label='Correct CTF?',
                      help='If you set to *Yes*, the CTF of the experimental particles will be corrected')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        form.addParam('createVolume', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
              label='Create output volume?',
              help='If you set to *Yes*, the final volume is created')
        form.addParam('angleGallery',FloatParam, label="angle for references", default=5, expertLevel=LEVEL_ADVANCED,
                      help='Distance in degrees between sampling points for generate gallery of references images')

        
                  
        form.addSection(label='Pca training')
        
        form.addParam('resolution',FloatParam, label="max resolution", default=10,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef' ,FloatParam, label="% variance", default=0.75, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of variance to determine the number of coefficients to be considers (between 0-1).'
                      ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('numPart' ,IntParam, label="number of particles for PCA", default=20000, 
                      expertLevel=LEVEL_ADVANCED,
                      help='number of particles to be consider for PCA training')
        
        
        form.addSection(label='Global alignment')
        form.addParam('applyShift', BooleanParam, default=False,
                      label='Consider previous alignment?',
                      help='If you set to *Yes*, the particles will be centered acording to previous alignment')
        form.addParam('numberOfIterations', IntParam, default=1, label='Number of iterations')
        form.addParam('angle' ,IntParam, label="initial angular sampling", default=8, 
                      help='Angular sampling for particles alignment')
        form.addParam('shift' ,IntParam, label="initial shift sampling", default=4, 
                      help='Sampling rate in the alignment')
        form.addParam('MaxShift', IntParam, label="Max. shift", default=20, expertLevel=LEVEL_ADVANCED,
                      help='Maximum shift for translational search')
        # form.addParam('MaxShift', IntParam, label="Max. shift (%)", default=10, expertLevel=LEVEL_ADVANCED,
        #               help='Maximum shift as a percentage of the image size')
        


        form.addParallelSection(threads=1, mpi=16)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        
        updateEnviron(self.gpuList.get())
        
        if self.corectCtf:
            self.imgsFnXmd = self._getExtraPath('images.xmd')
        else:
            self.imgsFnXmd = self._getExtraPath('images_original.xmd')
        self.imgsFn = self._getExtraPath('images.mrcs')
        self.refsFn = self._getExtraPath('references.mrcs')
        self.refsFnXmd = self._getExtraPath('references.xmd')
        self.sampling = self.inputParticles.get().getSamplingRate()
        size =self.inputParticles.get().getDimensions()[0]
        # self.MaxShift = int(size * self.MaxShift.get() / 100)
        self.MaxShift = self.MaxShift.get()
        iterations = self.numberOfIterations.get()


        self._insertFunctionStep('convertInputStep', self.inputParticles.get(), self.imgsFnXmd)
        self._insertFunctionStep("createGallery", self.angleGallery.get())
        self._insertFunctionStep("pcaTraining")
        
        for iter in range(iterations):

            if iter == 0:
                angle, MaxAngle, shift, maxShift = self.angle.get(), 180, self.shift.get(), self.MaxShift
                inputXmd = self.imgsFnXmd
                outputXmd = self._getExtraPath('align_iter%s.xmd'%(iter+1))
                applyShift = self.applyShift 
            if iter == 1:
               angle, MaxAngle, shift, maxShift = self.angle.get()/2, 180, 2, 6  
               inputXmd = self._getExtraPath('align_iter%s.xmd'%iter)
               outputXmd = self._getExtraPath('align_iter%s.xmd'%(iter+1))
               applyShift = True
            if iter == 2:
               angle, MaxAngle, shift, maxShift = self.angle.get()/4, 180, 1, 3  
               inputXmd = self._getExtraPath('align_iter%s.xmd'%iter)
               outputXmd = self._getExtraPath('align_iter%s.xmd'%(iter+1))
               applyShift = True
               
               if self.angleGallery.get() > 3:
                   self._insertFunctionStep("createGallery", 4)
            if iter == 3:
               angle, MaxAngle, shift, maxShift = self.angle.get()/8, 180, 0.5, 1.5  
               inputXmd = self._getExtraPath('align_iter%s.xmd'%iter)
               outputXmd = self._getExtraPath('align_iter%s.xmd'%(iter+1))
               applyShift = True
           
            self._insertFunctionStep("globalAlign", inputXmd, outputXmd, angle, MaxAngle, shift, maxShift, applyShift)   
        
        if self.createVolume:    
            self._insertFunctionStep("reconstructVolume", iter)    
        self._insertFunctionStep("createOutput", iter)
    

    #--------------------------- STEPS functions ---------------------------------------------------        
    def convertInputStep(self, inputFn, outputFn):
        writeSetOfParticles(inputFn, outputFn)  

        if self.corectCtf:  
            args = ' -i  %s -o %s --sampling_rate %s '%(self.imgsFnXmd, self.imgsFn, self.sampling)
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get()) 
        else:
            args = ' -i  %s -o %s '%(self.imgsFnXmd, self.imgsFn)
            self.runJob("xmipp_image_convert",args, numberOfMpi=1)           
    
    def createGallery(self, angle):
        refVol = self.inputVolume.get().getFileName()
        args = ' -i  %s --sym %s --sampling_rate %s  -o %s '% \
                (refVol, self.symmetryGroup.get(), angle, self.refsFn)
        self.runJob("xmipp_angular_project_library", args)
        moveFile( self._getExtraPath('references.doc'), self.refsFnXmd)
    
    
    def pcaTraining(self):
        args = ' -i %s -n 1 -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca  --batchPCA'% \
                (self.imgsFn, self.sampling, self.resolution.get(), self.coef.get(), self.numPart.get(), self._getExtraPath())
        # program = self.getProgram("train_pca.py")
        # program = self.getProgram("xmipp_global_align_train")
        # self.runJob(program, args, numberOfMpi=1)
        self.runJob("xmipp_global_align_train", args, numberOfMpi=1,
            env=self.getCondaEnv())
        
        
    def globalAlign(self, inputXmd, outputXmd, angle, MaxAngle, shift, MaxShift, applyShift):
        args = ' -i %s -r %s -a %s -amax %s -sh %s -msh %s -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt  -o %s -stExp %s  -stRef %s  -s %s '% \
                (self.imgsFn, self.refsFn, angle, MaxAngle, shift, MaxShift,\
                 self._getExtraPath(), self._getExtraPath(), outputXmd, inputXmd, self.refsFnXmd, self.sampling)
        if applyShift:
            args += ' --apply_shifts ' 
        # program = self.getProgram("align_images.py")  
        # program = self.getProgram("xmipp_global_align")       
        # self.runJob(program, args, numberOfMpi=1)
        self.runJob("xmipp_global_align", args, numberOfMpi=1,
                    env=self.getCondaEnv())


    def reconstructVolume(self, iter):
        args = '-i %s -n 2 --oroot %s '%(self._getExtraPath('align_iter%s.xmd'%(iter+1)), self._getExtraPath('split'))
        self.runJob("xmipp_metadata_split", args, numberOfMpi=1)
        self._reconstructHalf(self._getExtraPath('split000001.xmd'), self._getExtraPath('split000001_vol.mrc'))
        self._reconstructHalf(self._getExtraPath('split000002.xmd'), self._getExtraPath('split000002_vol.mrc'))
        # self._reconstruct(self._getExtraPath('align_iter%s.xmd'%(iter+1)), self._getExtraPath('output_vol.mrc'))
        self._reconstructAvg(self._getExtraPath('split000001_vol.mrc'), self._getExtraPath('split000002_vol.mrc'), self._getExtraPath('output_avg.mrc'))
        self._computeFSC(self._getExtraPath('split000001_vol.mrc'), self._getExtraPath('split000002_vol.mrc'))
        mdFsc =  emlib.MetaData(self._getExtraPath('fsc.xmd'))
        thr = 0.143
        self.resolutionHalf = self._computeResolution(mdFsc, thr)
        print('la resolucion es %s' %self.resolutionHalf)
        self._filterVolume(self._getExtraPath('output_avg.mrc'), self._getExtraPath('output_avg_filt.mrc'), self.resolutionHalf)

    def createOutput(self, iter):      
        #output particle
        imgSet = self._getExtraPath('align_iter%s.xmd'%(iter+1))
        partSet = self._createSetOfParticles()             
        EXTRA_LABELS = [
            #emlib.MDL_COST
        ]
         # Fill
        readSetOfParticles(
            imgSet,
            partSet,
            extraLabels=EXTRA_LABELS
        )
        partSet.setSamplingRate(self.inputParticles.get().getSamplingRate())       
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(self.inputParticles, partSet)
        
        #output volume
        if self.createVolume: 
            # self.resolutionHalf = self.resolutionHalf.set(self.resolutionHalf)
            # self._store(self.resolutionHalf)
            volume=Volume()
            volume.setFileName(self._getExtraPath('output_avg_filt.mrc'))
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
        # if self.createVolume:
        #     summary.append("Resolution: %s" % self.resolutionHalf) 
        return summary
    

    #--------------------------- UTILS functions --------------------------------------------
    def getTensorflowActivation(self):
        return "conda activate flexutils-tensorflow"
    
    def getProgram(self, program):
        cmd = '%s %s && ' % (xmipp3.Plugin.getCondaActivationCmd(), self.getTensorflowActivation())
        return cmd + ' %(program)s ' % locals()
    
    def _updateLocation(self, item, row):
        index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
        item.setLocation(index, filename)
    
    def _reconstructHalf(self, input, output):
        gpuList = self.getGpuList()
        program = 'xmipp_cuda_reconstruct_fourier'    
        args = '-i %s -o %s --sym %s  --max_resolution 0.5  --sampling %s --thr %s --device 0 -gpusPerNode 1 -threadsPerGPU 4' %\
        (input, output, self.symmetryGroup.get(), self.sampling, self.numberOfMpi.get()) 
        self.runJob(program, args, numberOfMpi=self.numberOfMpi.get())
        
    def _reconstructAvg(self, half1, half2, output):
        args = ' -i %s --plus %s -o %s' %(half1, half2, output)
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)
        args = ' -i %s --mult 0.5' %output
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)
        
    def _computeFSC(self, input, ref):
        args = ' -i %s --ref %s --sampling_rate %s -o %s' %(input, ref, self.sampling, self._getExtraPath('fsc.xmd'))
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
        args = ' -i %s -o %s --fourier low_pass %s --sampling %s '%(input, output, resolution, self.sampling)
        self.runJob('xmipp_transform_filter', args, numberOfMpi=1)
        
        
        
    