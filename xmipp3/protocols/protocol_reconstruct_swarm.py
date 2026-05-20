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

       AI Generated

        ## Overview

        The Swarm Consensus protocol refines a set of initial volumes against a set of
        particles using a swarm-like optimization strategy.

        Instead of starting from a single reference volume, the protocol starts from a
        set of candidate volumes. These volumes are treated as a swarm. During the
        iterations, each volume is evaluated according to how well its projections
        match subsets of the experimental particles. Volumes that perform well are
        kept as local or global best solutions, and the other volumes are progressively
        updated toward better-supported regions of the solution space.

        This strategy is useful when the user has several possible initial volumes and
        wants to combine them into a consensus refinement. It can help explore
        alternative starting hypotheses and reduce dependence on a single initial
        reference.

        The protocol produces a final averaged consensus volume and a set of the best
        volumes found for each member of the swarm.

        ## Inputs and General Workflow

        The protocol requires:

        - a set of full-size particles;
        - a set of initial volumes;
        - optionally, a mask.

        The particles are converted to Xmipp metadata format and internally resampled
        according to the target resolution. The input volumes are also rescaled to the
        same working sampling rate and box size. If a mask is provided, it is prepared
        at the same scale.

        The protocol then evaluates the initial volumes. In each iteration, it uses
        random subsets of particles to update the volumes, post-processes the swarm,
        evaluates each volume on another subset of particles, and updates the local and
        global best volumes.

        At the end, it computes an average consensus volume, cleans it, restores the
        original box size and sampling rate, and creates the final outputs.

        ## Full-Size Images

        The **Full-size Images** parameter defines the particle set used to evaluate
        and update the swarm.

        These particles should be provided at the finest available sampling rate. The
        protocol may internally resample them to a coarser working sampling rate based
        on the target resolution, but the original particle sampling is used again for
        the final output scale.

        The particle set should be reasonably clean and should correspond to the
        structure represented by the input volumes. Strong heterogeneity, many
        contaminants, or very poor particles can make the swarm evaluation unstable.

        ## Initial Volumes

        The **Initial volumes** parameter provides the starting swarm.

        This input must be a set of volumes. Each volume acts as one member of the
        swarm and is refined and evaluated during the protocol.

        The volumes may represent alternative initial models, different reconstructions,
        different classes, or different hypotheses about the structure. They should
        all be plausible enough to generate projections that can be compared with the
        particles.

        If the initial volumes are very poor, unrelated to the particles, or mutually
        incompatible, the swarm may not converge to a meaningful consensus.

        ## Particle Radius

        The **Radius of particle** parameter defines the radius, in pixels, of the
        spherical mask covering the particle in the input images.

        If the value is -1 or otherwise not positive, the protocol uses approximately
        half the particle box size.

        This radius is used when masking the resampled particle stack and during
        alignment and reconstruction. It should include the particle density while
        excluding unnecessary background.

        For elongated or non-spherical particles, the radius should still be chosen to
        cover the full particle.

        ## Symmetry Group

        The **Symmetry group** parameter defines the symmetry used when generating
        projection galleries, assigning particle orientations, and reconstructing
        volumes.

        For asymmetric particles, use **c1**. If the structure has known symmetry, the
        corresponding Xmipp symmetry group can be specified.

        Symmetry should only be imposed when it is biologically justified. Incorrect
        symmetry may make different swarm volumes converge toward an artificial or
        misleading structure.

        ## Mask

        The **Mask** parameter allows the user to provide a volume mask.

        The mask is rescaled to the working sampling rate and box size. It is then used
        during post-processing and cleaning operations to focus the analysis on the
        molecular region.

        Mask values should range from 0 to 1. Smooth masks are recommended. A mask that
        is too tight may remove real density, while a mask that is too loose may allow
        background noise to influence the swarm.

        ## Number of Iterations

        The **Number of iterations** parameter controls how many swarm update cycles are
        performed.

        Each iteration updates the candidate volumes, post-processes them, evaluates
        their agreement with the particles, and updates the best-volume records.

        More iterations allow more time for the swarm to move toward better-supported
        solutions, but increase computation time. Too few iterations may stop the
        optimization before the volumes have stabilized.

        The default value is intended to provide a practical refinement schedule.

        ## Target Resolution

        The **Max. Target Resolution** parameter defines the target resolution used for
        the working representation, in angstroms.

        The protocol chooses a working sampling rate based on this target resolution.
        The goal is to focus the optimization on low- to medium-resolution information
        appropriate for comparing initial volumes with particles.

        This is important because swarm consensus is intended to refine global
        structural hypotheses, not to produce high-resolution detail directly.

        A very aggressive target resolution may make the comparisons noisy or unstable,
        whereas a conservative value focuses on robust shape information.

        ## Minimum Angle

        The **Min. Angle** parameter defines a lower bound on the angular sampling used
        during projection matching.

        The protocol estimates an angular step from the working box size, but it does
        not allow the search to become finer than this minimum value.

        A smaller minimum angle allows a denser angular search but increases
        computation. A larger value is faster but may reduce angular accuracy.

        This parameter is advanced and should usually be left at its default unless the
        user has a specific reason to change the angular search density.

        ## Images Used to Update the Swarm

        The **# Images to update** parameter defines how many particles are randomly
        selected to update each volume during a training step.

        For each volume, the protocol selects a random subset of particles, generates
        projections of the current volume, assigns orientations to the selected
        particles, and reconstructs an updated volume from those assignments.

        Using a subset makes the optimization stochastic. This can help the swarm
        explore the solution space and reduces computation compared with using all
        particles at every update.

        A larger value gives more stable updates but increases computation. A smaller
        value is faster but noisier.

        ## Images Used to Evaluate the Swarm

        The **# Images to evaluate** parameter defines how many particles are randomly
        selected to score each volume during evaluation.

        The evaluation subset is used to estimate how well a given volume explains the
        particles. The average correlation from the assigned particles is used as a
        score.

        Using a separate evaluation subset helps compare swarm members without using
        the full dataset each time.

        A larger evaluation subset gives more stable scores. A smaller subset is
        faster but may make the ranking of volumes noisier.

        ## Volume Evaluation

        During evaluation, each volume is low-pass filtered to the target resolution
        and optionally masked. The protocol then selects a random subset of particles,
        generates a projection gallery from the volume, assigns particle orientations,
        and computes an average correlation score.

        The best volume globally is stored as the current global best. In addition,
        each swarm member keeps its own best version across iterations.

        These best-volume records guide later swarm updates and are used to produce
        the final output set of volumes.

        ## Volume Update Strategy

        After the first iterations, each volume is updated using a swarm-like rule.

        The update depends on two directions:

        - the difference between the current volume and its own best previous version;
        - the difference between the current volume and the global best volume.

        Random coefficients control how strongly each volume moves in these
        directions. This gives the protocol a stochastic optimization behavior similar
        to particle-swarm optimization.

        Conceptually, each volume is encouraged to improve based both on its own
        history and on the best solution found by the whole swarm.

        ## Post-Processing During Iterations

        After reconstructing new volumes, the protocol computes an average volume,
        locally aligns individual volumes to that average, and applies cleaning steps.

        The cleaning uses volume restoration tools to remove untrusted background
        voxels and reduce noise-like components. If a mask is provided, it is used to
        focus these operations on the relevant region.

        This post-processing helps keep the swarm volumes comparable and reduces
        spurious differences caused by noise or background artifacts.

        ## Final Average Volume

        At the end of the protocol, an average consensus volume is computed.

        This average combines the best volumes found by the swarm and the global best
        volume. The final average is then cleaned, resized back to the original
        particle box size, converted to MRC format, and assigned the original particle
        sampling rate.

        This final averaged map is the main consensus result of the protocol.

        ## Output Volume

        The main output is **outputVolume**.

        This is the final average consensus volume. It is registered in Scipion with
        the original sampling rate of the input particles.

        This output should be interpreted as a consensus volume derived from the swarm
        of initial volumes. It can be used as a refined starting model or as an
        intermediate result for further refinement.

        ## Output Volumes

        The protocol also produces **outputVolumes**, a set containing the best volume
        found for each member of the swarm.

        These volumes are useful for inspecting whether the swarm converged to similar
        solutions or whether different starting volumes retained distinct structural
        features.

        If the best volumes are very similar, this supports the stability of the
        consensus. If they remain very different, the dataset may contain
        heterogeneity, the initial volumes may represent different hypotheses, or the
        optimization may not have converged.

        ## GPU Execution

        The protocol can use GPU acceleration for the projection-assignment and
        Fourier-reconstruction steps.

        GPU execution is enabled by default. If GPU execution is requested but Xmipp
        GPU programs are not available, the protocol reports a validation error.

        GPU acceleration is recommended because the protocol repeatedly generates
        projection galleries, assigns particles, and reconstructs volumes for several
        swarm members.

        ## Practical Recommendations

        Use this protocol when you have several plausible initial volumes and want to
        obtain a consensus refinement from them.

        Provide a reasonably clean particle set. Strong contaminants or severe
        heterogeneity can make the swarm scores unreliable.

        Use initial volumes that are related to the same structure. Completely
        unrelated volumes may prevent meaningful convergence.

        Use a mask when you want the evaluation and cleaning to focus on the molecular
        region.

        Keep the target resolution conservative for initial consensus refinement. The
        goal is to compare global structure reliably, not to refine high-resolution
        detail.

        Increase the number of particles used for evaluation if volume scores appear
        unstable.

        Inspect both the final consensus volume and the output best-volume set. The
        agreement or disagreement among best volumes is an important diagnostic.

        Use the final output as a starting point for a more standard high-resolution
        refinement protocol.

        ## Final Perspective

        Swarm Consensus is a multi-reference refinement protocol. It starts from a set
        of candidate volumes and uses stochastic particle subsets to evaluate, update,
        and combine them.

        For biological users, its main value is that it can reduce dependence on a
        single starting model. By allowing several volumes to compete and move toward
        better-supported solutions, the protocol can help identify a consensus
        structural hypothesis.

        The output should be treated as a refined starting model or consensus
        intermediate, not as a final high-resolution reconstruction. Its reliability
        should be assessed by visual inspection, comparison of the best swarm volumes,
        and subsequent refinement behavior.
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
                      help='See https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry for a description of the symmetry groups format'
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
        self._insertFunctionStep(self.convertInputStep, self.inputParticles.getObjId())
        self._insertFunctionStep(self.evaluateIndividuals,0)
        for self.iteration in range(1,self.numberOfIterations.get()+1):
            self._insertFunctionStep(self.reconstructNewVolumes,self.iteration)
            self._insertFunctionStep(self.postProcessing,self.iteration)
            self._insertFunctionStep(self.evaluateIndividuals,self.iteration)
            if self.iteration>1:
                self._insertFunctionStep(self.updateVolumes,self.iteration)
        self._insertFunctionStep(self.calculateAverage,self.numberOfIterations.get()+1)
        self._insertFunctionStep(self.cleanVolume,self._getExtraPath("volumeAvg.vol"))
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
