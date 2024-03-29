# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
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

from functools import reduce
import math
import random

from xmipp3.constants import CUDA_ALIGN_SIGNIFICANT

try:
    from itertools import izip
except ImportError:
    izip = zip
from os.path import join, exists
import os

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, IntParam, EnumParam, NumericListParam,
                                        USE_GPU, GPU_LIST)
from pyworkflow.utils.path import cleanPath, copyFile, moveFile, makePath, createLink
from pwem.protocols import ProtRefine3D
from pwem.objects import Volume, SetOfVolumes
from pwem.emlib.image import ImageHandler


from pwem import emlib
from xmipp3.base import isXmippCudaPresent
from xmipp3.convert import (writeSetOfParticles, getImageLocation)


class XmippProtReconstructSwarm(ProtRefine3D):
    """This is a 3D refinement protocol whose main input is a set of volumes and a set of particles.
       The set of particles has to be at full size (the finer sampling rate available), but
       the rest of inputs (reference volume and masks) can be at any downsampling factor.
       The protocol scales the input images and volumes to a size that depends on the target resolution.

       The input set of volumes is considered to be a swarm of volumes and they try to optimize
       the correlation between the volumes and the set of particles. This is an stochastic maximization
       and only a fraction of the particles are used to update the volumes and evaluate them.
    """
    _label = 'swarm consensus'
    _version = VERSION_2_0

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
        
        form.addParam('inputParticles', PointerParam, label="Full-size Images", important=True, 
                      pointerClass='SetOfParticles', allowsNull=True,
                      help='Select a set of images at full resolution')
        form.addParam('inputVolumes', PointerParam, label="Initial volumes", important=True,
                      pointerClass='SetOfVolumes',
                      help='Select a set of volumes with 2 volumes or a single volume')
        form.addParam('particleRadius', IntParam, default=-1, label='Radius of particle (px)',
                     help='This is the radius (in pixels) of the spherical mask covering the particle in the input images')       

        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group', 
                      help='See http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/Symmetry for a description of the symmetry groups format'
                        'If no symmetry is present, give c1')
        
        form.addParam('nextMask', PointerParam, label="Mask", pointerClass='VolumeMask', allowsNull=True,
                      help='The mask values must be between 0 (remove these pixels) and 1 (let them pass). Smooth masks are recommended.')

        form.addParam('numberOfIterations', IntParam, default=15, label='Number of iterations', expertLevel=LEVEL_ADVANCED)
        form.addParam('targetResolution', FloatParam, label="Max. Target Resolution", default="12", expertLevel=LEVEL_ADVANCED,
                      help="In Angstroms.")
        form.addParam('minAngle', FloatParam, label="Min. Angle", default="10", expertLevel=LEVEL_ADVANCED,
                      help="The angular search is limited by this parametr (in degrees).")
        form.addParam('NimgTrain', IntParam, default=500, label='# Images to update', expertLevel=LEVEL_ADVANCED)
        form.addParam('NimgTest', IntParam, default=100, label='# Images to evaluate', expertLevel=LEVEL_ADVANCED)
        
        form.addParallelSection(threads=1, mpi=8)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self.imgsFn=self._getExtraPath('images.xmd')
        self._insertFunctionStep('convertInputStep', self.inputParticles.getObjId())
        self._insertFunctionStep('evaluateIndividuals',0)
        for self.iteration in range(1,self.numberOfIterations.get()+1):
            self._insertFunctionStep('reconstructNewVolumes',self.iteration)
            self._insertFunctionStep('postProcessing',self.iteration)
            self._insertFunctionStep('evaluateIndividuals',self.iteration)
            if self.iteration>1:
                self._insertFunctionStep('updateVolumes',self.iteration)
        self._insertFunctionStep('calculateAverage',self.numberOfIterations.get()+1)
        self._insertFunctionStep('cleanVolume',self._getExtraPath("volumeAvg.vol"))
        self._insertFunctionStep("createOutput")
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def readInfoField(self,fnDir,block,label):
        mdInfo = emlib.MetaData("%s@%s"%(block,join(fnDir,"info.xmd")))
        return mdInfo.getValue(label,mdInfo.firstObject())

    def writeInfoField(self,fnDir,block,label, value):
        mdInfo = emlib.MetaData()
        objId=mdInfo.addObject()
        mdInfo.setValue(label,value,objId)
        mdInfo.write("%s@%s"%(block, join(fnDir,"info.xmd")),emlib.MD_APPEND)

    def convertInputVolume(self, imgHandler, obj, fnIn, fnOut, TsCurrent, newXdim):
        self.runJob('xmipp_image_resize',"-i %s -o %s --factor %f"%(fnIn,fnOut,obj.getSamplingRate()/TsCurrent),numberOfMpi=1)
        objXdim, _, _, _ =imgHandler.getDimensions((1,fnOut))
        if newXdim!=objXdim:
            self.runJob('xmipp_transform_window',"-i %s --size %d"%(fnOut,newXdim),numberOfMpi=1)

    def convertInputStep(self, inputParticlesId):
        fnDir=self._getExtraPath()
        writeSetOfParticles(self.inputParticles.get(),self.imgsFn)

        # Choose the target sampling rate        
        TsOrig=self.inputParticles.get().getSamplingRate()
        TsCurrent=max(TsOrig,self.targetResolution.get()/3)
        Xdim=self.inputParticles.get().getDimensions()[0]
        newXdim=int(round(Xdim*TsOrig/TsCurrent))
        if newXdim<40:
            newXdim=int(40)
            TsCurrent=Xdim*(TsOrig/newXdim)
        print("Preparing images to sampling rate=",TsCurrent)
        self.writeInfoField(fnDir,"size",emlib.MDL_XSIZE,newXdim)
        self.writeInfoField(fnDir,"sampling",emlib.MDL_SAMPLINGRATE,TsCurrent)
        
        # Prepare particles
        fnNewParticles=join(fnDir,"images.stk")
        if newXdim!=Xdim:
            self.runJob("xmipp_image_resize","-i %s -o %s --fourier %d"%(self.imgsFn,fnNewParticles,newXdim),
                        numberOfMpi=self.numberOfMpi.get()*self.numberOfThreads.get())
        else:
            self.runJob("xmipp_image_convert","-i %s -o %s --save_metadata_stack %s"%(self.imgsFn,fnNewParticles,join(fnDir,"images.xmd")),
                        numberOfMpi=1)
        R=self.particleRadius.get()
        if R<=0:
            R=self.inputParticles.get().getDimensions()[0]/2
        R=min(round(R*TsOrig/TsCurrent*1.1),newXdim/2)
        self.runJob("xmipp_transform_mask","-i %s --mask circular -%d"%(fnNewParticles,R),numberOfMpi=self.numberOfMpi.get()*self.numberOfThreads.get())
        
        # Prepare mask
        imgHandler=ImageHandler()
        if self.nextMask.hasValue():
            self.convertInputVolume(imgHandler, self.nextMask.get(), getImageLocation(self.nextMask.get()), join(fnDir,"mask.vol"), TsCurrent, newXdim)
        
        # Prepare references
        i=0
        for vol in self.inputVolumes.get():
            fnVol=join(fnDir,"volume%03d.vol"%i)
            self.convertInputVolume(imgHandler, vol, getImageLocation(vol), fnVol, TsCurrent, newXdim)
            self.runJob("xmipp_image_operate","-i %s --mult 0 -o %s"%(fnVol,join(fnDir,"volume%03d_speed.vol"%i)),numberOfMpi=1)
            i+=1
        emlib.MetaData().write("best@"+self._getExtraPath("swarm.xmd")) # Empty write to guarantee this block is the first one
        emlib.MetaData().write("bestByVolume@"+self._getExtraPath("swarm.xmd"),emlib.MD_APPEND) # Empty write to guarantee this block is the second one
    
    def evaluateIndividuals(self,iteration):
        fnDir = self._getExtraPath()
        newXdim = self.readInfoField(fnDir,"size",emlib.MDL_XSIZE)
        angleStep = max(math.atan2(1,newXdim/2),self.minAngle.get())
        TsOrig=self.inputParticles.get().getSamplingRate()
        TsCurrent = self.readInfoField(fnDir,"sampling",emlib.MDL_SAMPLINGRATE)
        fnMask = self._getExtraPath("mask.vol")
        fnImages = join(fnDir, "images.xmd")

        maxShift = round(0.1 * newXdim)
        R = self.particleRadius.get()
        if R <= 0:
            R = self.inputParticles.get().getDimensions()[0] / 2
        R = R * TsOrig / TsCurrent

        bestWeightVol={}
        bestIterVol={}
        if iteration>0:
            mdPrevious = emlib.MetaData("bestByVolume@"+self._getExtraPath("swarm.xmd"))
            for objId in mdPrevious:
                idx = int(mdPrevious.getValue(emlib.MDL_IDX,objId))
                bestWeightVol[idx]=mdPrevious.getValue(emlib.MDL_WEIGHT,objId)
                bestIterVol[idx]=mdPrevious.getValue(emlib.MDL_ITER,objId)

        # Global alignment
        md=emlib.MetaData()
        if iteration>1:
            mdBest = emlib.MetaData("best@"+self._getExtraPath("swarm.xmd"))
            objId = mdBest.firstObject()
            bestWeight = mdBest.getValue(emlib.MDL_WEIGHT,objId)
        else:
            bestWeight = -1e38
        for i in range(self.inputVolumes.get().getSize()):
            # Filter and mask volumes
            fnVol = self._getExtraPath("volume%03d.vol"%i)
            self.runJob("xmipp_transform_filter","-i %s --fourier low_pass %f --sampling %f"%(fnVol,self.targetResolution,TsCurrent),numberOfMpi=1)
            if exists(fnMask):
                self.runJob("xmipp_image_operate","-i %s --mult %s"%(fnVol,fnMask),numberOfMpi=1)
            
            # Prepare subset of experimental images
            fnTest =  join(fnDir,"imgs_%03d.xmd"%i)
            self.runJob("xmipp_metadata_utilities","-i %s --operate random_subset %d -o %s"%(fnImages,self.NimgTest,fnTest),numberOfMpi=1)
            
            # Generate projections
            fnGallery=join(fnDir,"gallery%02d.stk"%i)
            fnGalleryMd=join(fnDir,"gallery%02d.doc"%i)
#             args="-i %s -o %s --sampling_rate %f --perturb %f --sym %s"%\
#                  (fnVol,fnGallery,angleStep,math.sin(angleStep*math.pi/180.0)/4,self.symmetryGroup)
            args="-i %s -o %s --sampling_rate %f --sym %s"%\
                 (fnVol,fnGallery,angleStep,self.symmetryGroup)
            args+=" --compute_neighbors --angular_distance -1 --experimental_images %s"%fnTest
            self.runJob("xmipp_angular_project_library",args,numberOfMpi=self.numberOfMpi.get()*self.numberOfThreads.get())
            
            # Assign angles
            fnAngles = join(fnDir, "angles_iter001_00.xmd")
            if not self.useGpu.get():
                args = '-i %s --initgallery %s --maxShift %d --odir %s --dontReconstruct --useForValidation 1' % \
                       (fnTest, fnGalleryMd, maxShift, fnDir)
                self.runJob('xmipp_reconstruct_significant', args,
                            numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
            else:
                count=0
                GpuListCuda=''
                if self.useQueueForSteps() or self.useQueue():
                    GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                    GpuList = GpuList.split(",")
                    for elem in GpuList:
                        GpuListCuda = GpuListCuda+str(count)+' '
                        count+=1
                else:
                    GpuList = ' '.join([str(elem) for elem in self.getGpuList()])
                    GpuListAux = ''
                    for elem in self.getGpuList():
                        GpuListCuda = GpuListCuda+str(count)+' '
                        GpuListAux = GpuListAux+str(elem)+','
                        count+=1
                    os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux

                args = '-i %s -r %s -o %s --keepBestN 1 --dev %s ' % (
                fnTest, fnGalleryMd, fnAngles, GpuListCuda)
                self.runJob(CUDA_ALIGN_SIGNIFICANT, args, numberOfMpi=1)

            # Evaluate
            fnAngles = join(fnDir, "angles_iter001_00.xmd")
            if exists(fnAngles):
                # Significant may decide not to write it if it is not significant
                mdAngles = emlib.MetaData(fnAngles)
                weight = mdAngles.getColumnValues(emlib.MDL_MAXCC)
                avgWeight = reduce(lambda x, y: x + y, weight) / len(weight)
                print("Average weight for "+fnVol+" = "+str(avgWeight))
                objId = md.addObject()
                md.setValue(emlib.MDL_IDX,int(i),objId)
                md.setValue(emlib.MDL_IMAGE,fnVol,objId)
                md.setValue(emlib.MDL_WEIGHT,avgWeight,objId)
                md.setValue(emlib.MDL_ITER,iteration,objId)
                
                # Is global best
                if avgWeight>bestWeight:
                    bestWeight = avgWeight
                    if iteration==0:
                        # None of the input volumes can be the best volume in the first iteration since their gray values may significantly
                        # differ from those in the projections
                        self.runJob("xmipp_image_operate","-i %s --mult 0 -o %s"%(fnVol,self._getExtraPath("volumeBest.vol")),numberOfMpi=1)
                    else:
                        copyFile(fnVol, self._getExtraPath("volumeBest.vol"))
                    mdBest=emlib.MetaData()
                    objId=mdBest.addObject()
                    mdBest.setValue(emlib.MDL_IMAGE,fnVol,objId)
                    mdBest.setValue(emlib.MDL_WEIGHT,bestWeight,objId)
                    mdBest.setValue(emlib.MDL_ITER,iteration,objId)
                    mdBest.write("best@"+self._getExtraPath("swarm.xmd"),emlib.MD_APPEND)
                
                # Is local best
                if iteration==0:
                    self.runJob("xmipp_image_operate","-i %s --mult 0 -o %s"%(fnVol, self._getExtraPath("volume%03d_best.vol"%i)),numberOfMpi=1)
                elif avgWeight>bestWeightVol[i] or iteration==1:
                    bestWeightVol[i]=avgWeight
                    bestIterVol[i]=iteration
                    copyFile(fnVol,self._getExtraPath("volume%03d_best.vol"%i))

            # Clean
            cleanPath(fnTest)
            self.runJob("rm -f",fnDir+"/*iter00?_00.xmd",numberOfMpi=1)
            self.runJob("rm -f",fnDir+"/gallery*",numberOfMpi=1)
        md.write("evaluations_%03d@"%iteration+self._getExtraPath("swarm.xmd"),emlib.MD_APPEND)
        
        # Update best by volume
        if iteration==0:
            md.write("bestByVolume@"+self._getExtraPath("swarm.xmd"),emlib.MD_APPEND)
        else:
            md.clear()
            for i in range(self.inputVolumes.get().getSize()):
                objId = md.addObject()
                md.setValue(emlib.MDL_IDX,int(i),objId)
                md.setValue(emlib.MDL_IMAGE,self._getExtraPath("volume%03d_best.vol"%i),objId)
                md.setValue(emlib.MDL_WEIGHT,bestWeightVol[i],objId)
                md.setValue(emlib.MDL_ITER,bestIterVol[i],objId)
            md.write("bestByVolume@"+self._getExtraPath("swarm.xmd"),emlib.MD_APPEND)
        
    def createOutput(self):
        fnDir = self._getExtraPath()
        Ts=self.readInfoField(fnDir,"sampling",emlib.MDL_SAMPLINGRATE)

        # Remove files that will not be used any longer
        cleanPath(join(fnDir,"images.stk"))
        cleanPath(join(fnDir,"images.xmd"))

        # Final average
        TsOrig=self.inputParticles.get().getSamplingRate()
        XdimOrig=self.inputParticles.get().getDimensions()[0]
        fnAvg = self._getExtraPath("volumeAvg.vol")
        fnAvgMrc = self._getExtraPath("volumeAvg.mrc")
        self.runJob("xmipp_image_resize","-i %s --dim %d"%(fnAvg,XdimOrig),numberOfMpi=1)
        self.runJob("xmipp_image_convert","-i %s -o %s -t vol"%(fnAvg,fnAvgMrc),numberOfMpi=1)
        cleanPath(fnAvg)
        self.runJob("xmipp_image_header","-i %s --sampling_rate %f"%(fnAvgMrc,TsOrig),numberOfMpi=1)
        volume=Volume()
        volume.setFileName(fnAvgMrc)
        volume.setSamplingRate(TsOrig)
        self._defineOutputs(outputVolume=volume)
        self._defineSourceRelation(self.inputParticles.get(),volume)
        self._defineSourceRelation(self.inputVolumes.get(),volume)
        
        # Swarm of volumes
        volSet = self._createSetOfVolumes()
        volSet.setSamplingRate(Ts)
        for i in range(self.inputVolumes.get().getSize()):
            fnVol = self._getExtraPath("volume%03d_best.vol"%i)
            fnMrc = self._getExtraPath("volume%03d_best.mrc"%i)
            self.runJob("xmipp_image_convert", "-i %s -o %s" % (fnVol, fnMrc), numberOfMpi=1)
            self.runJob("xmipp_image_header", "-i %s --sampling_rate %f" % (fnMrc, TsOrig), numberOfMpi=1)
            cleanPath(fnVol)

            vol=Volume()
            vol.setFileName(fnMrc)
            vol.setSamplingRate(Ts)
            volSet.append(vol)
        self._defineOutputs(outputVolumes=volSet)
        self._defineSourceRelation(self.inputParticles.get(),volSet)
        self._defineSourceRelation(self.inputVolumes.get(),volSet)
            
    def reconstructNewVolumes(self,iteration):
        fnDir = self._getExtraPath()
        newXdim = self.readInfoField(fnDir,"size",emlib.MDL_XSIZE)
        angleStep = max(math.atan2(1,newXdim/2),self.minAngle.get())
        TsOrig=self.inputParticles.get().getSamplingRate()
        TsCurrent = self.readInfoField(fnDir,"sampling",emlib.MDL_SAMPLINGRATE)
        fnImages = join(fnDir,"images.xmd")
        
        maxShift=round(0.1*newXdim)
        R=self.particleRadius.get()
        if R<=0:
            R=self.inputParticles.get().getDimensions()[0]/2
        R=R*TsOrig/TsCurrent

        # Global alignment
        for i in range(self.inputVolumes.get().getSize()):
            fnVol = self._getExtraPath("volume%03d.vol"%i)

            # Prepare subset of experimental images
            fnTrain =  join(fnDir,"imgs_%03d.xmd"%i)
            self.runJob("xmipp_metadata_utilities","-i %s --operate random_subset %d -o %s"%(fnImages,self.NimgTrain,fnTrain),numberOfMpi=1)
            
            # Generate projections
            fnGallery=join(fnDir,"gallery%02d.stk"%i)
            fnGalleryMd=join(fnDir,"gallery%02d.doc"%i)
#             args="-i %s -o %s --sampling_rate %f --perturb %f --sym %s"%\
#                  (fnVol,fnGallery,angleStep,math.sin(angleStep*math.pi/180.0)/4,self.symmetryGroup)
            args="-i %s -o %s --sampling_rate %f --sym %s"%\
                 (fnVol,fnGallery,angleStep,self.symmetryGroup)
            args+=" --compute_neighbors --angular_distance -1 --experimental_images %s"%fnTrain
            self.runJob("xmipp_angular_project_library",args,numberOfMpi=self.numberOfMpi.get()*self.numberOfThreads.get())
            
            # Assign angles
            fnAngles = join(fnDir, "angles_iter001_00.xmd")
            if not self.useGpu.get():
                args = '-i %s --initgallery %s --maxShift %d --odir %s --dontReconstruct --useForValidation 1' % \
                       (fnTrain, fnGalleryMd, maxShift, fnDir)
                self.runJob('xmipp_reconstruct_significant', args,
                            numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
            else:
                count=0
                GpuListCuda=''
                if self.useQueueForSteps() or self.useQueue():
                    GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                    GpuList = GpuList.split(",")
                    for elem in GpuList:
                        GpuListCuda = GpuListCuda+str(count)+' '
                        count+=1
                else:
                    GpuList = ' '.join([str(elem) for elem in self.getGpuList()])
                    GpuListAux = ''
                    for elem in self.getGpuList():
                        GpuListCuda = GpuListCuda+str(count)+' '
                        GpuListAux = GpuListAux+str(elem)+','
                        count+=1
                    os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux

                args = '-i %s -r %s -o %s --keepBestN 1 --dev %s ' % (
                fnTrain, fnGalleryMd, fnAngles, GpuListCuda)
                self.runJob(CUDA_ALIGN_SIGNIFICANT, args, numberOfMpi=1)

            # Reconstruct
            if exists(fnAngles):
                # Significant may decide not to write it if no image is significant
                args = "-i %s -o %s --sym %s --weight --fast" % (
                fnAngles, fnVol, self.symmetryGroup)
                if self.useGpu.get():
                    #AJ to make it work with and without queue system
                    if self.numberOfMpi.get()>1:
                        N_GPUs = len((self.gpuList.get()).split(','))
                        args += ' -gpusPerNode %d' % N_GPUs
                        args += ' -threadsPerGPU %d' % max(self.numberOfThreads.get(),4)
                    count=0
                    GpuListCuda=''
                    if self.useQueueForSteps() or self.useQueue():
                        GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                        GpuList = GpuList.split(",")
                        for elem in GpuList:
                            GpuListCuda = GpuListCuda+str(count)+' '
                            count+=1
                    else:
                        GpuListAux = ''
                        for elem in self.getGpuList():
                            GpuListCuda = GpuListCuda+str(count)+' '
                            GpuListAux = GpuListAux+str(elem)+','
                            count+=1
                        os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux
                    args += " --thr %s" % self.numberOfThreads.get()
                    if self.numberOfMpi.get()==1:
                        args += " --device %s" % GpuListCuda
                    if self.numberOfMpi.get()>1:
                        self.runJob('xmipp_cuda_reconstruct_fourier', args, numberOfMpi=len((self.gpuList.get()).split(','))+1)
                    else:
                        self.runJob('xmipp_cuda_reconstruct_fourier', args)
                else:
                    self.runJob('xmipp_reconstruct_fourier_accel', args)
                args = "-i %s --mask circular %f" % (fnVol, -R)
                self.runJob("xmipp_transform_mask", args, numberOfMpi=1)
                args = "-i %s --select below 0 --substitute value 0" % fnVol
                self.runJob("xmipp_transform_threshold", args, numberOfMpi=1)

            # Clean
            cleanPath(fnTrain)
            self.runJob("rm -f",fnDir+"/*iter00?_00.xmd",numberOfMpi=1)
            self.runJob("rm -f",fnDir+"/gallery*",numberOfMpi=1)
    
    def cleanVolume(self,fnVol):
        # Generate mask if available
        if self.nextMask.hasValue():
            fnMask=self._getExtraPath("mask.vol")
        else:
            fnMask=""

        fnRootRestored=self._getExtraPath("volumeRestored")
        args='--i1 %s --i2 %s --oroot %s --denoising 1'%(fnVol,fnVol,fnRootRestored)
        if fnMask!="":
            args+=" --mask binary_file %s"%fnMask
        self.runJob('xmipp_volume_halves_restoration',args,numberOfMpi=1)
        moveFile("%s_restored1.vol"%fnRootRestored,fnVol)
        cleanPath("%s_restored2.vol"%fnRootRestored)
 
        args='--i1 %s --i2 %s --oroot %s --filterBank 0.01'%(fnVol,fnVol,fnRootRestored)
        if fnMask!="":
            args+=" --mask binary_file %s"%fnMask
        self.runJob('xmipp_volume_halves_restoration',args,numberOfMpi=1)
        moveFile("%s_restored1.vol"%fnRootRestored,fnVol)
        cleanPath("%s_restored2.vol"%fnRootRestored)
        cleanPath("%s_filterBank.vol"%fnRootRestored)
    
    def postProcessing(self, iteration):
        # Calculate average
        self.calculateAverage(iteration)

        # Align volumes
        fnAvg = self._getExtraPath("volumeAvg.vol")
        for i in range(self.inputVolumes.get().getSize()):
            fnVol = self._getExtraPath("volume%03d.vol"%i)
            self.runJob('xmipp_volume_align','--i1 %s --i2 %s --local --apply'%(fnAvg,fnVol),numberOfMpi=1)
            if iteration>=2:
                fnVol = self._getExtraPath("volume%03d_best.vol"%i)
                self.runJob('xmipp_volume_align','--i1 %s --i2 %s --local --apply'%(fnAvg,fnVol),numberOfMpi=1)

        # Remove untrusted background voxels
        for i in range(self.inputVolumes.get().getSize()):
            fnVol = self._getExtraPath("volume%03d.vol"%i)
            self.cleanVolume(fnVol)

    def updateVolumes(self,iteration):
        fnBest = self._getExtraPath("volumeBest.vol")
        fnInternal = self._getExtraPath("internalBest.vol")
        fnExternal = self._getExtraPath("externalBest.vol")
        for i in range(self.inputVolumes.get().getSize()):
            fnVol = self._getExtraPath("volume%03d.vol"%i)
            fnVolBest = self._getExtraPath("volume%03d_best.vol"%i)
            fnSpeed = self._getExtraPath("volume%03d_speed.vol"%i)
            
            u1 = random.uniform(0, 1)
            u2 = random.uniform(0, 1)
            
            self.runJob("xmipp_image_operate","-i %s --minus %s -o %s"%(fnVolBest,fnVol,fnInternal),numberOfMpi=1)
            self.runJob("xmipp_image_operate","-i %s --minus %s -o %s"%(fnBest,fnVol,fnExternal),numberOfMpi=1)
            self.runJob("xmipp_image_operate","-i %s --mult %f"%(fnInternal,2*u1),numberOfMpi=1)
            self.runJob("xmipp_image_operate","-i %s --mult %f"%(fnExternal,2*u2),numberOfMpi=1)
            self.runJob("xmipp_image_operate","-i %s --plus %s"%(fnSpeed,fnInternal),numberOfMpi=1)
            self.runJob("xmipp_image_operate","-i %s --plus %s"%(fnSpeed,fnExternal),numberOfMpi=1)
            self.runJob("xmipp_image_operate","-i %s --plus %s"%(fnVol,fnSpeed),numberOfMpi=1)
            
        cleanPath(fnInternal)
        cleanPath(fnExternal)

    def calculateAverage(self,iteration):
        fnAvg = self._getExtraPath("volumeAvg.vol")
        N=0
        for i in range(self.inputVolumes.get().getSize()):
            if iteration<=2:
                fnVol = self._getExtraPath("volume%03d.vol"%i)
                if i==0:
                    copyFile(fnVol,fnAvg)
                else:
                    self.runJob("xmipp_image_operate","-i %s --plus %s"%(fnAvg,fnVol),numberOfMpi=1)
            else:
                fnVol = self._getExtraPath("volume%03d_best.vol"%i)
                self.runJob("xmipp_image_operate","-i %s --plus %s"%(fnAvg,fnVol),numberOfMpi=1)
            N+=1
        if iteration>0:
            self.runJob("xmipp_image_operate","-i %s --plus %s"%(fnAvg,self._getExtraPath("volumeBest.vol")),numberOfMpi=1)
            N+=1
        self.runJob("xmipp_image_operate","-i %s --divide %f"%(fnAvg,N),numberOfMpi=1)

    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Symmetry: %s" % self.symmetryGroup.get())
        summary.append("Number of iterations: "+str(self.numberOfIterations))
        return summary
    
    def _methods(self):
        strline = ''
        if hasattr(self, 'outputVolume') or True:
            strline += 'We processed %d particles from %s ' % (self.inputParticles.get().getSize(), 
                                                                self.getObjectTag('inputParticles'))
            strline += 'using %s as starting swarm and Xmipp swarm procedure. ' % (self.getObjectTag('inputVolumes'))
            if self.symmetryGroup!="c1":
                strline+="We imposed %s symmetry. "%self.symmetryGroup
            strline += "We performed %d iterations of "%self.numberOfIterations.get()
        return [strline]

    def _validate(self):
        errors = []
        if self.useGpu and not isXmippCudaPresent():
            errors.append("You have asked to use GPU, but I cannot find the Xmipp GPU programs")
        return errors