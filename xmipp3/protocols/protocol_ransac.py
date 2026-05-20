# **************************************************************************
# *
# * Authors:     Javier Vargas and Adrian Quintana (jvargas@cnb.csic.es aquintana@cnb.csic.es)
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
from math import floor

from pyworkflow.object import Float
from pyworkflow.utils.path import cleanPath, copyFile
from pyworkflow.protocol.params import (PointerParam, FloatParam, BooleanParam,
                                        IntParam, StringParam,
                                        STEPS_PARALLEL, LEVEL_ADVANCED, USE_GPU, GPU_LIST)

from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtInitialVolume
from pwem.objects import SetOfClasses2D

from pwem import emlib
from xmipp3.convert import (writeSetOfClasses2D, readSetOfVolumes,
                            writeSetOfParticles)
from xmipp3.base import isMdEmpty, isXmippCudaPresent
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippProtRansac(ProtInitialVolume):
    """ 
    Computes an initial 3d model from a set of projections/classes 
    using RANSAC algorithm.
    
    This method is based on an initial non-lineal dimensionality
    reduction approach which allows to select representative small 
    sets of class average images capturing the most of the structural 
    information of the particle under study. These reduced sets are 
    then used to generate volumes from random orientation assignments. 
    The best volume is determined from these guesses using a random 
    sample consensus (RANSAC) approach.

    AI Generated

    ## Overview

    The RANSAC protocol generates initial 3D volumes from a set of 2D class
    averages or projection averages.

    Initial model generation is a critical step in single-particle cryo-EM. Before
    a high-quality 3D refinement can be performed, the workflow usually needs a
    starting volume that has approximately the correct global shape and orientation
    distribution. The RANSAC protocol addresses this problem by generating many
    candidate initial volumes from small random subsets of the input averages and
    then selecting the best candidates according to how well they explain the full
    set of input images.

    The method follows a random sample consensus strategy. Each RANSAC iteration
    uses a small subset of 2D averages to propose a tentative 3D volume. The
    candidate volume is then projected, and the projections are compared with the
    input averages. Volumes supported by many well-correlated images are considered
    better candidates. The best volumes are then refined by projection matching.

    The output is a set of proposed initial volumes, each annotated with scoring
    information.

    ## Inputs and General Workflow

    The main input is a set of 2D class averages or projection averages.

    The protocol first converts the input images into Xmipp metadata format. It
    then low-pass filters and resizes the input averages to a working size suitable
    for initial-model search. This makes the procedure faster and focuses the
    search on low- to medium-resolution structural information.

    The protocol then runs many independent RANSAC iterations. In each iteration,
    it selects a small subset of input averages, assigns or estimates orientations,
    reconstructs a tentative volume, projects that volume, and compares the
    projections with the input averages.

    After all iterations, the protocol evaluates the candidate volumes using an
    inlier criterion based on projection correlation. The best volumes are selected
    and refined through several projection-matching iterations. Finally, the
    refined volumes are resized back to the original box size and written as output
    volumes.

    ## Input Averages

    The **Input averages** parameter should point to a set of 2D classes or
    averages.

    If the input is a SetOfClasses2D, the classes should have representative
    images. These representative images are the class averages used by the
    protocol. If the input is a SetOfAverages or similar particle-average set, the
    images are used directly.

    The quality of the input averages is very important. RANSAC initial-model
    generation works best when the averages show clear structural features and
    represent a broad range of particle views.

    If the input averages are noisy, dominated by contaminants, affected by strong
    preferred orientation, or contain many inconsistent particle populations, the
    candidate volumes may be poor or ambiguous.

    ## Symmetry Group

    The **Symmetry group** parameter defines the symmetry assumed during projection
    generation, reconstruction, and volume evaluation.

    For asymmetric particles, use **c1**. If the particle has known symmetry, the
    appropriate Xmipp symmetry group should be provided.

    Correct symmetry can help generate better initial volumes by enforcing
    equivalent views and reducing ambiguity. However, using an incorrect symmetry
    can produce misleading models by averaging non-equivalent features.

    Users should only impose symmetry when it is biologically justified.

    ## Angular Sampling Rate

    The **Angular sampling rate** parameter defines how finely projection
    directions are explored, in degrees.

    A smaller angular sampling value means a denser angular search. This can
    improve orientation assignment and candidate-volume evaluation, but increases
    computation time.

    A larger value is faster but may miss important orientation differences and
    reduce the quality of the initial model.

    The default value is a practical compromise for many datasets. Advanced users
    may adjust it depending on particle size, symmetry, and expected angular
    complexity.

    ## Number of RANSAC Iterations

    The **Number of RANSAC iterations** parameter controls how many candidate
    volumes are generated and tested.

    Each iteration proposes a volume from a different random subset or sampling of
    the input averages. Increasing the number of iterations increases the chance of
    finding a good initial model, especially when the input set contains outliers
    or multiple particle populations.

    However, more iterations require more computation. The default is designed to
    provide broad exploration while remaining practical for typical workflows.

    If the output volumes are unstable or poor, increasing the number of RANSAC
    iterations may help.

    ## Dimensionality Reduction

    The **Perform dimensionality reduction** option changes how representative
    input averages are selected during RANSAC.

    When enabled, the protocol uses Local Tangent Space Alignment, or LTSA, to
    organize the input averages in a reduced-dimensional space. It then samples
    representative images from this space using a grid.

    This can help select a small set of averages that captures the diversity of
    the dataset, rather than choosing purely random images.

    Dimensionality reduction is an advanced option. It requires that the number of
    input classes be large enough relative to the grid size. If there are too few
    classes, the protocol reports a validation error.

    ## Number of Grids per Dimension

    When dimensionality reduction is enabled, the **Number of grids per dimension**
    parameter controls how the reduced space is sampled.

    For example, a value of 3 creates a 3 by 3 grid, giving up to 9 regions from
    which representative classes can be sampled.

    A larger grid gives more detailed sampling of the reduced space, but requires
    more input classes. A smaller grid is more conservative.

    This parameter should be chosen according to the number of available classes
    and the diversity of the input averages.

    ## Number of Random Samples

    When dimensionality reduction is disabled, the **Number of random samples**
    parameter defines how many input averages are selected in each RANSAC
    iteration.

    These randomly selected averages are assigned random orientations, or
    orientations constrained by an optional initial volume, and are used to
    reconstruct a candidate volume.

    If too few samples are used, the candidate volumes may be unstable or
    insufficiently constrained. If too many are used, each iteration becomes more
    expensive and less exploratory.

    The default is intended to generate many quick candidate volumes from small
    subsets.

    ## Initial Volume

    The **Initial volume** parameter allows the user to provide a very rough volume
    to constrain the angular search.

    This is optional. The protocol can run without an initial volume.

    A rough initial volume may be useful when the specimen has a known overall
    shape, such as a cylinder for a filament-like object. In that case, the initial
    volume can help assign more plausible tilt angles, while still allowing the
    rotational angle to be uncertain.

    The initial volume should be used carefully. If it is too specific or
    incorrect, it may bias the search toward a wrong structure. It is best used as
    a broad geometrical constraint rather than as a detailed reference.

    ## Maximum Frequency of the Initial Volume

    The **Max frequency of the initial volume** parameter controls the resolution
    used during the initial-model search, expressed in angstroms.

    The protocol low-pass filters and resizes the input averages according to this
    parameter. The goal is to focus the RANSAC search on robust low-resolution
    features rather than noisy high-resolution details.

    A lower-resolution search is usually appropriate for initial model generation.
    The purpose is to obtain the correct global shape, not to recover fine
    features.

    ## Inliers Threshold

    The **Inliers threshold** defines the correlation value used to decide whether
    an input average supports a candidate volume.

    After a candidate volume is generated, the protocol projects it and compares
    those projections with the input averages. Averages with correlation above the
    threshold are considered inliers.

    Candidate volumes with more and better inliers receive higher scores.

    If the threshold is too high, few or no candidate volumes may be considered
    valid. If this happens, the user should lower the threshold.

    If the threshold is too low, poor candidate volumes may appear to have many
    inliers, reducing the selectivity of the method.

    ## Number of Best Volumes to Refine

    The **Number of best volumes to refine** parameter controls how many of the
    best RANSAC candidates are selected for refinement.

    After all RANSAC iterations are evaluated, the protocol ranks the candidates by
    their inlier support. The best candidates are copied into refinement branches.

    Keeping several candidates is useful because initial-model generation can be
    ambiguous. Different candidates may correspond to alternative orientations,
    different conformations, or different local optima.

    The output contains the refined versions of these selected candidates.

    ## Number of Iterations to Refine the Volumes

    The **Number of iterations to refine the volumes** parameter defines how many
    projection-matching refinement cycles are applied to each selected candidate.

    In each cycle, the current volume is projected, input averages are assigned to
    projections, and a new volume is reconstructed from the updated assignments.

    More iterations can improve the consistency of the candidate volume, but also
    increase computation time. Too many iterations are not always beneficial if the
    initial candidate is poor or if the input averages are inconsistent.

    ## Use All Images to Refine

    The **Use all images to refine** option controls which averages are used during
    the refinement of selected RANSAC volumes.

    If disabled, only inlier images are used. This keeps refinement focused on the
    images that strongly support the candidate volume.

    If enabled, all input averages are used for refinement. This may be useful when
    the input set is clean and the user wants each candidate to be refined using
    all available information.

    For heterogeneous or noisy input sets, using only inliers may be safer.

    ## GPU Execution

    The protocol can use GPU acceleration for reconstruction steps when available.

    GPU execution is enabled through the hidden GPU parameters. If GPU use is
    requested but the required Xmipp CUDA programs are not available, the protocol
    reports a validation error.

    GPU acceleration can substantially reduce computation time, especially when
    many RANSAC iterations and refinement steps are requested.

    ## Output Volumes

    The main output is **outputVolumes**, a set of refined candidate initial
    volumes.

    Each output volume is written in MRC format, resized to the original input box
    size, and assigned the sampling rate of the input averages.

    The output volumes are annotated with Xmipp scoring values, including:

    - total volume score;
    - thresholded volume score;
    - mean correlation score;
    - minimum correlation score.

    These scores help the user compare candidate volumes, but they should not be
    used blindly. Visual inspection and downstream refinement behavior are also
    important.

    ## Interpreting the Candidate Volumes

    The RANSAC output volumes should be interpreted as possible initial models.

    They are not final reconstructions. Their role is to provide plausible starting
    points for later 3D refinement.

    A good candidate should have a reasonable global shape, be compatible with the
    2D class averages, and produce stable behavior in subsequent refinement. A poor
    candidate may show distorted shape, missing regions, strong artifacts, or
    features inconsistent with the class averages.

    When several output volumes are produced, users should inspect them and choose
    the most plausible candidates for downstream refinement.

    ## Practical Recommendations

    Use clean and representative 2D class averages as input. Remove obvious junk
    classes before running RANSAC.

    Use the correct symmetry group only when symmetry is known. Do not impose
    symmetry merely to make the output look cleaner.

    Start with the default number of RANSAC iterations and best volumes. Increase
    the number of iterations if the results are unstable or if the dataset is
    difficult.

    If no valid maps are found, lower the inliers threshold.

    Use dimensionality reduction only when there are enough input classes and when
    you want representative sampling of class variability.

    Consider providing a very rough initial volume only when there is a strong
    reason to constrain the angular search. Avoid using a detailed reference that
    could bias the initial model.

    Inspect all output volumes visually before selecting one for refinement.

    ## Final Perspective

    RANSAC is an initial-volume generation protocol based on repeated random
    hypotheses and consensus scoring.

    For biological users, its value is that it can propose several possible 3D
    starting models from 2D averages without requiring tilted-pair data. By testing
    many candidate volumes and retaining those best supported by the input
    averages, the protocol reduces dependence on any single random initialization.

    The resulting volumes should be treated as starting hypotheses. Their quality
    must be assessed by visual inspection, agreement with the 2D averages, scoring
    information, and performance in subsequent 3D refinement.
     """
    _label = 'ransac'
    _devStatus = PROD

    def __init__(self, **args):
        ProtInitialVolume.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    #--------------------------- DEFINE param functions --------------------------------------------        
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
         
        form.addParam('inputSet', PointerParam, label="Input averages", important=True, 
                      pointerClass='SetOfClasses2D, SetOfAverages',# pointerCondition='hasRepresentatives',
                      help='Select the input images from the project.'
                           'It should be a SetOfClasses2D object')  
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',  
                      help="See http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry"
                           " for a description of the symmetry groups format in Xmipp.\n"
                           "If no symmetry is present, use _c1_.")
        form.addParam('angularSampling', FloatParam, default=5, expertLevel=LEVEL_ADVANCED,
                      label='Angular sampling rate',
                      help='In degrees.'
                      ' This sampling defines how fine the projection gallery from the volume is explored.')
        form.addParam('nRansac', IntParam, default="400", expertLevel=LEVEL_ADVANCED,
                      label="Number of RANSAC iterations", 
                      help='Number of initial volumes to test by RANSAC')
        
        form.addParam('dimRed', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Perform dimensionality reduction', 
                      help='The dimensionality reduction is performed using the Local Tangent Space'
                      'Alignment. See http://www.stat.missouri.edu/~ys873/research/LTSA11.pdf')
        form.addParam('numGrids', IntParam, default=3, condition='dimRed', expertLevel=LEVEL_ADVANCED,
                      label='Number of grids per dimension',
                      help='Number of squares to sample the classes')
        form.addParam('numSamples', IntParam, default=8, condition='not dimRed', expertLevel=LEVEL_ADVANCED,
                      label='Number of random samples',
                      help='Number of squares to sample the classes')
        
        form.addParam('corrThresh', FloatParam, default=0.77, expertLevel=LEVEL_ADVANCED,
                      label='Inliers threshold',
                      help='Correlation value threshold to determine if an experimental projection is an inlier or outlier.')        
        form.addParam('numVolumes', IntParam, default=10, expertLevel=LEVEL_ADVANCED,
                      label='Number of best volumes to refine',
                      help='Number of best volumes to refine using projection matching approach and the input classes')
        form.addParam('numIter', IntParam, default=10, expertLevel=LEVEL_ADVANCED,
                      label='Number of iterations to refine the volumes',
                      help='Number of iterations to refine the best volumes using projection matching approach and the input classes')
        form.addParam('initialVolume', PointerParam, label="Initial volume",  expertLevel=LEVEL_ADVANCED,
                      pointerClass='Volume', allowsNull=True,
                      help='You may provide a very rough initial volume as a way to constraint the angular search.'
                            'For instance, when reconstructing a fiber, you may provide a cylinder so that side views'
                            'are assigned to the correct tilt angle, although the rotational angle may be completely wrong')           
                
        form.addParam('maxFreq', IntParam, default=12, expertLevel=LEVEL_ADVANCED,
                      label='Max frequency of the initial volume',
                      help=' Max frequency of the initial volume in Angstroms')
        
        form.addParam('useAll', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Use all images to refine', 
                      help=' When refining a RANSAC volume, use all images to refine it instead of only inliers')
        
        form.addParallelSection(threads=8, mpi=1)
            
         
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):
        # Insert some initialization steps
        initialStepId = self._insertInitialSteps()
        
        deps = [] # Store all steps ids, final step createOutput depends on all of them    
        for n in range(self.nRansac.get()):
            # CTF estimation with Xmipp
            stepId = self._insertFunctionStep(self.ransacIterationStep, n,
                                    prerequisites=[initialStepId], needsGPU=self.usesGpu()) # Make estimation steps indepent between them
            deps.append(stepId)
        
        # Look for threshold, evaluate volumes and get the best
        self._insertFunctionStep(self.getCorrThreshStep, prerequisites=deps, needsGPU=False) # Make estimation steps indepent between them)
        self._insertFunctionStep(self.evaluateVolumesStep, needsGPU=False)
        bestVolumesStepId = self._insertFunctionStep(self.getBestVolumesStep, needsGPU=False)
        
        deps = [] # Store all steps ids, final step createOutput depends on all of them
        # Refine the best volumes
        for n in range(self.numVolumes.get()):
            fnBase='proposedVolume%05d' % n
            fnRoot=self._getPath(fnBase)
                    
            for it in range(self.numIter.get()):    
                if it==0:
                    self._insertFunctionStep(self.reconstructStep,fnRoot, needsGPU=self.usesGpu(),prerequisites=[bestVolumesStepId])
                else:
                    self._insertFunctionStep(self.reconstructStep,fnRoot, needsGPU=self.usesGpu())
                self._insertFunctionStep(self.projMatchStep,fnBase, needsGPU=False)
            
            stepId =  self._insertFunctionStep(self.resizeStep,fnRoot,self.Xdim, needsGPU=False)
            
            deps.append(stepId)
        
        # Score each of the final volumes
        self._insertFunctionStep(self.scoreFinalVolumes,
                                 prerequisites=deps, needsGPU=False) # Make estimation steps indepent between them
        
        self._insertFunctionStep(self.createOutputStep,needsGPU=False)
    
    def _insertInitialSteps(self):
        # Convert the input classes to a metadata ready for xmipp
        self.imgsFn = self._getExtraPath('input_classes.xmd')
        self._insertFunctionStep(self.convertInputStep, self.imgsFn, needsGPU=False)
        
        inputSet = self.inputSet.get()
        self.Xdim = inputSet.getDimensions()[0]
        
        fnOutputReducedClass = self._getExtraPath("reducedClasses.xmd")
        fnOutputReducedClassNoExt = os.path.splitext(fnOutputReducedClass)[0]
    
        # Low pass filter and resize        
        maxFreq = self.maxFreq.get()
        ts = inputSet.getSamplingRate()
        K = 0.25 * (maxFreq / ts)
        if K < 1:
            K = 1
        self.Xdim2 = self.Xdim / K
        if self.Xdim2 < 32:
            self.Xdim2 = 32
            K = self.Xdim / self.Xdim2
            
        freq = ts / maxFreq
        ts = K * ts

        self._insertRunJobStep("xmipp_transform_filter",
                               "-i %s -o %s --fourier low_pass %f --oroot %s"
                                %(self.imgsFn,fnOutputReducedClass,freq,fnOutputReducedClassNoExt),
                               needsGPU=False)
        lastId = self._insertRunJobStep("xmipp_image_resize",
                                        "-i %s --fourier %d -o %s"
                                        %(fnOutputReducedClass,self.Xdim2,fnOutputReducedClassNoExt),
                                        needsGPU=False)

        # Generate projection gallery from the initial volume
        if self.initialVolume.hasValue():
            lastId = self._insertFunctionStep(self.projectInitialVolume, needsGPU=False)
            
        return lastId

    #--------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, classesFn):
        inputSet = self.inputSet.get()
        
        if isinstance(inputSet, SetOfClasses2D):
            writeSetOfClasses2D(inputSet, classesFn)
        else:
            writeSetOfParticles(inputSet, classesFn)
        
    def ransacIterationStep(self, n):
    
        fnOutputReducedClass = self._getExtraPath("reducedClasses.xmd")  
        fnBase = "ransac%05d"%n
        fnRoot = self._getTmpPath(fnBase)
        
    
        if self.dimRed:
            # Get a random sample of images
            self.runJob("xmipp_transform_dimred","-i %s --randomSample %s.xmd  %d -m LTSA "%(fnOutputReducedClass,fnRoot,self.numGrids.get()))
        else:        
            self.runJob("xmipp_metadata_utilities","-i %s -o %s.xmd  --operate random_subset %d --mode overwrite "%(fnOutputReducedClass,fnRoot,self.numSamples.get()))
            self.runJob("xmipp_metadata_utilities","-i %s.xmd --fill angleRot rand_uniform -180 180 "%(fnRoot))
            self.runJob("xmipp_metadata_utilities","-i %s.xmd --fill angleTilt rand_uniform 0 180 "%(fnRoot))
            self.runJob("xmipp_metadata_utilities","-i %s.xmd --fill anglePsi  rand_uniform 0 360 "%(fnRoot)) 
    
        # If there is an initial volume, assign angles        
        if self.initialVolume.hasValue():
            fnGallery=self._getTmpPath('gallery_InitialVolume.stk')
            self.runJob("xmipp_angular_projection_matching", "-i %s.xmd -o %s.xmd --ref %s --Ri 0 --Ro %s --max_shift %s --append"\
                   %(fnRoot,fnRoot,fnGallery,str(self.Xdim/2),str(self.Xdim/20)))
    
        # Reconstruct with the small sample
        self.reconstructStep(fnRoot)
        
        fnVol = fnRoot+'.vol'
        
        # Generate projections from this reconstruction
        fnGallery=self._getTmpPath('gallery_'+fnBase+'.stk')
        self.runJob("xmipp_angular_project_library", "-i %s -o %s --sampling_rate %f --sym %s --method fourier 1 0.25 bspline --compute_neighbors --angular_distance -1 --experimental_images %s --max_tilt_angle 90"\
                    %(fnVol,fnGallery,self.angularSampling.get(),self.symmetryGroup.get(),fnOutputReducedClass))
            
        # Assign angles to the rest of images
        fnAngles=self._getTmpPath('angles_'+fnBase+'.xmd')
        self.runJob("xmipp_angular_projection_matching", "-i %s -o %s --ref %s --Ri 0 --Ro %s --max_shift %s --append"\
                              %(fnOutputReducedClass,fnAngles,fnGallery,str(self.Xdim/2),str(self.Xdim/20)))
       
        # Delete intermediate files 
        cleanPath(fnGallery)
        cleanPath(self._getTmpPath('gallery_'+fnBase+'_sampling.xmd'))
        cleanPath(self._getTmpPath('gallery_'+fnBase+'.doc'))
        cleanPath(fnVol)
        cleanPath(self._getTmpPath(fnBase+'.xmd'))
    
    def reconstructStep(self, fnRoot):
        from pwem.emlib.metadata import getSize
        if os.path.exists(fnRoot+".xmd"):
            Nimages=getSize(fnRoot+".xmd")
            if Nimages>0:
                if self.useGpu.get():
                    # protocol will run several reconstructions at once, so execute each reconstruction separately
                    args = "-i %s.xmd -o %s.vol --sym %s --thr %s --fast"\
                           % (fnRoot, fnRoot, self.symmetryGroup.get(), self.numberOfThreads.get())

                    #AJ to make it work with and without queue system
                    if self.numberOfMpi.get()>1:
                        N_GPUs = len((self.gpuList.get()).split(','))
                        args += ' -gpusPerNode %d' % N_GPUs
                        args += ' -threadsPerGPU %d' % max(self.numberOfThreads.get(),4)

                    gpuList = list(map(str, self._stepsExecutor.getGpuList()))
                    gpuListArg = " ".join(gpuList)
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpuList)
                    self.info("GPUs used in CUDA_VISIBLE_DEVICES: %s" % gpuListArg)

                    if self.numberOfMpi.get() == 1:
                        args += " --device " + gpuListArg
                    if self.numberOfMpi.get() > 1:
                        self.runJob('xmipp_cuda_reconstruct_fourier', args, numberOfMpi=len(gpuList)+1)
                    else:
                        self.runJob('xmipp_cuda_reconstruct_fourier', args)
                else:
                    self.runJob("xmipp_reconstruct_fourier_accel","-i %s.xmd -o %s.vol --sym %s "
                                %(fnRoot,fnRoot,self.symmetryGroup.get()))
                self.runJob("xmipp_transform_mask","-i %s.vol --mask circular -%d "%(fnRoot,self.Xdim2/2))
        else:
            print(fnRoot+".xmd is empty. The corresponding volume is not generated.")
    
    def resizeStep(self,fnRoot,Xdim):
        if os.path.exists(fnRoot+".vol"):
            self.runJob("xmipp_image_resize","-i %s.vol -o %s.vol --dim %d %d" %(fnRoot,fnRoot,Xdim,Xdim))
            self.runJob("xmipp_image_convert","-i %s.vol -o %s.mrc -t vol" %(fnRoot,fnRoot))
            cleanPath("%s.vol"%fnRoot)
            self.runJob("xmipp_image_header", "-i %s.mrc --sampling_rate %f" %\
                        (fnRoot, self.inputSet.get().getSamplingRate()))
     
    def getCorrThreshStep(self):
        corrVector = []
        fnCorr=self._getExtraPath("correlations.xmd")               
        mdCorr= emlib.MetaData()
    
        for n in range(self.nRansac.get()):
            fnRoot="ransac%05d"%n
            fnAngles=self._getTmpPath("angles_"+fnRoot+".xmd")
            md = emlib.MetaData(fnAngles)
            
            for objId in md:
                corr = md.getValue(emlib.MDL_MAXCC, objId)
                corrVector.append(corr)
                objIdCorr = mdCorr.addObject()
                mdCorr.setValue(emlib.MDL_MAXCC,float(corr),objIdCorr)
    
        mdCorr.write("correlations@"+fnCorr,emlib.MD_APPEND)
        mdCorr= emlib.MetaData()
        sortedCorrVector = sorted(corrVector)
        indx = int(floor(self.corrThresh.get()*(len(sortedCorrVector)-1)))    
        
        #With the line below commented the percentil is not used for the threshold and is used the value introduced in the form
        #CorrThresh = sortedCorrVector[indx]#
            
        objId = mdCorr.addObject()
        mdCorr.setValue(emlib.MDL_WEIGHT,self.corrThresh.get(),objId)
        mdCorr.write("corrThreshold@"+fnCorr,emlib.MD_APPEND)
        print("Correlation threshold: "+str(self.corrThresh.get()))
    
    
    def evaluateVolumesStep(self):
        fnCorr=self._getExtraPath("correlations.xmd")
        fnCorr = 'corrThreshold@'+fnCorr
        mdCorr= emlib.MetaData(fnCorr)
        objId = mdCorr.firstObject()    
        CorrThresh = mdCorr.getValue(emlib.MDL_WEIGHT,objId)
        numValid=0
        for n in range(self.nRansac.get()):        
            fnRoot="ransac%05d"%n              
            fnAngles=self._getTmpPath("angles_"+fnRoot+".xmd")    
            md = emlib.MetaData(fnAngles)
            numInliers=0
            for objId in md:
                corr = md.getValue(emlib.MDL_MAXCC, objId)
               
                if (corr >= CorrThresh) :
                    numInliers = numInliers+corr
    
            md= emlib.MetaData()
            objId = md.addObject()
            md.setValue(emlib.MDL_WEIGHT,float(numInliers),objId)
            if numInliers>0:
                numValid+=1
            md.write("inliers@"+fnAngles,emlib.MD_APPEND)
        if numValid==0:
            raise ValueError("There are no valid map. Consider lowering the threshold to have more inliers. "\
                             "Current threshold %f"%CorrThresh)
    
    def getBestVolumesStep(self):
        volumes = []
        inliers = []
        
        for n in range(self.nRansac.get()):
            fnAngles = self._getTmpPath("angles_ransac%05d"%n+".xmd")
            md=emlib.MetaData("inliers@"+fnAngles)
            numInliers=md.getValue(emlib.MDL_WEIGHT,md.firstObject())
            volumes.append(fnAngles)
            inliers.append(numInliers)
        
        index = sorted(range(inliers.__len__()), key=lambda k: inliers[k])
        fnBestAngles = ''
        threshold=self.getCCThreshold()
     
        i = self.nRansac.get()-1
        indx = 0
        while i >= 0 and indx < self.numVolumes:
            fnBestAngles = volumes[index[i]]
            fnBestAnglesOut = self._getPath("proposedVolume%05d"%indx+".xmd")
            copyFile(fnBestAngles, fnBestAnglesOut)
            self._log.info("Best volume %d = %s" % (indx, fnBestAngles))
            if not self.useAll:
                self.runJob("xmipp_metadata_utilities","-i %s -o %s --query select \"maxCC>%f \" --mode append" %(fnBestAnglesOut,fnBestAnglesOut,threshold))
                if not isMdEmpty(fnBestAnglesOut):
                    indx += 1
            else:
                indx += 1
            i -= 1
            
        # Remove unnecessary files
        for n in range(self.nRansac.get()):
            fnAngles = self._getTmpPath("angles_ransac%05d"%n+".xmd")
            cleanPath(fnAngles)
             
    def projMatchStep(self,fnBase):
        fnRoot=self._getPath(fnBase)
        if os.path.exists(fnRoot+".vol"):
            fnGallery=self._getTmpPath('gallery_'+fnBase+'.stk')
            fnOutputReducedClass = self._getExtraPath("reducedClasses.xmd") 
            
            AngularSampling=max(self.angularSampling.get()/2.0,7.5);
            self.runJob("xmipp_angular_project_library", "-i %s.vol -o %s --sampling_rate %f --sym %s --method fourier 1 0.25 bspline --compute_neighbors --angular_distance -1 --experimental_images %s"\
                                  %(fnRoot,fnGallery,float(AngularSampling),self.symmetryGroup.get(),fnOutputReducedClass))
        
            self.runJob("xmipp_angular_projection_matching", "-i %s.xmd -o %s.xmd --ref %s --Ri 0 --Ro %s --max_shift %s --append"\
                   %(fnRoot,fnRoot,fnGallery,str(self.Xdim/2),str(self.Xdim/20)))
                    
            cleanPath(self._getTmpPath('gallery_'+fnBase+'_sampling.xmd'))
            cleanPath(self._getTmpPath('gallery_'+fnBase+'.doc'))
            cleanPath(self._getTmpPath('gallery_'+fnBase+'.stk'))
                 
    def scoreFinalVolumes(self):
        threshold=self.getCCThreshold()
        mdOut=emlib.MetaData()
        for n in range(self.numVolumes.get()):
            fnRoot=self._getPath('proposedVolume%05d'%n)
            fnAssignment=fnRoot+".xmd"
            if os.path.exists(fnAssignment):
                self.runJob("xmipp_metadata_utilities","-i %s --fill weight constant 1"%fnAssignment)
                MDassignment=emlib.MetaData(fnAssignment)
                sum=0
                thresholdedSum=0
                N=0
                minCC=2
                for id in MDassignment:
                    cc=MDassignment.getValue(emlib.MDL_MAXCC,id)
                    sum+=cc
                    thresholdedSum+=cc-threshold
                    if cc<minCC:
                        minCC=cc
                    N+=1
                if N>0:
                    avg=sum/N
                else:
                    avg=0.0
                id=mdOut.addObject()
                mdOut.setValue(emlib.MDL_IMAGE,fnRoot+".mrc",id)
                mdOut.setValue(emlib.MDL_VOLUME_SCORE_SUM,float(sum),id)
                mdOut.setValue(emlib.MDL_VOLUME_SCORE_SUM_TH,float(thresholdedSum),id)
                mdOut.setValue(emlib.MDL_VOLUME_SCORE_MEAN,float(avg),id)
                mdOut.setValue(emlib.MDL_VOLUME_SCORE_MIN,float(minCC),id)
        mdOut.write(self._getPath("proposedVolumes.xmd"))


    def projectInitialVolume(self):
        fnOutputInitVolume=self._getTmpPath("initialVolume.vol")
        img = ImageHandler()
        img.convert(self.initialVolume.get(), fnOutputInitVolume)
        self.runJob("xmipp_image_resize","-i %s --dim %d %d"%(fnOutputInitVolume,self.Xdim2,self.Xdim2))
        fnGallery=self._getTmpPath('gallery_InitialVolume.stk')
        fnOutputReducedClass = self._getExtraPath("reducedClasses.xmd") 
        self.runJob("xmipp_angular_project_library", "-i %s -o %s --sampling_rate %f --sym %s --method fourier 1 0.25 bspline --compute_neighbors --angular_distance -1 --experimental_images %s"\
                              %(fnOutputInitVolume,fnGallery,self.angularSampling.get(),self.symmetryGroup.get(),fnOutputReducedClass))
   
    def _postprocessVolume(self, vol, row):
        self._counter += 1
        vol.setObjComment('ransac volume %02d' % self._counter)
        vol._xmipp_volScoreSum   = Float(row.getValue(emlib.MDL_VOLUME_SCORE_SUM))
        vol._xmipp_volScoreSumTh = Float(row.getValue(emlib.MDL_VOLUME_SCORE_SUM_TH))
        vol._xmipp_volScoreMean = Float(row.getValue(emlib.MDL_VOLUME_SCORE_MEAN))
        vol._xmipp_volScoreMin = Float(row.getValue(emlib.MDL_VOLUME_SCORE_MIN))
        
    def createOutputStep(self):
        inputSet = self.inputSet.get()
        fn = self._getPath('proposedVolumes.xmd')
#        md = emlib.MetaData(fn)
#        md.addItemId()
#        md.write(fn)
        
        volumesSet = self._createSetOfVolumes()
        volumesSet.setSamplingRate(inputSet.getSamplingRate())
        self._counter = 0
        readSetOfVolumes(fn, volumesSet, postprocessImageRow=self._postprocessVolume)
        
        # Set a meaningful comment
#         for vol in volumesSet:
#             vol.setObjComment('ransac volume %02d' % vol.getObjId())
#             volumesSet.update(vol)
        
        self._defineOutputs(outputVolumes=volumesSet)
        self._defineSourceRelation(self.inputSet, volumesSet)
        self._storeSummaryInfo(self.numVolumes.get())
    
    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        inputSet = self.inputSet.get()
        if isinstance(inputSet, SetOfClasses2D):
            if not self.inputSet.get().hasRepresentatives():
                errors.append("The input classes should have representatives.")
                
                
        if self.dimRed:
            nGrids = self.numGrids.get()
            if (nGrids * nGrids) > inputSet.getSize():
                errors.append('Dimensionaly reduction could not be applied')
                errors.append('if the number of classes is less than the number')
                errors.append('of grids squared. \n')
                errors.append('Consider either provide more classes or')
                errors.append('disable dimensionality reduction')

        if self.useGpu and not isXmippCudaPresent():
            errors.append("You have asked to use GPU, but I cannot find Xmipp GPU programs in the path")

        return errors
    
    def _summary(self):
        summary = []
        if hasattr(self, 'outputVolumes'):                        
            summary.append("RANSAC iterations: %d" % self.nRansac)
            summary.append("Number of volumes to refine: %d" % self.numVolumes.get())
            summary.append("Number of refinement interations: %d" % self.numIter.get())
            if self.summaryVar.hasValue():
                summary.append(self.summaryVar.get())
        else:
            summary.append("Output volumes not ready yet.")

        return summary
        
    def _methods(self):
        strline=''
        if hasattr(self, 'outputVolumes'):
            strline+='We obtained %d initial volume(s) from %s set of images and symmetry %s.'%\
                           (self.numVolumes.get(),self.getObjectTag('inputSet'),self.symmetryGroup.get())
            strline+='The number of RANSAC volumes generated are %d.From this set, we selected the best %d volume(s) that was refined by %d projection matching iteration(s)'%\
                           (self.nRansac.get(),self.numVolumes.get(),self.numIter.get())
            strline+= ' [Vargas2014]'
            if self.dimRed:
                strline+='We performed dimensionality reduction using a %d x %d grid'%\
                            (self.numGrids.get(),self.numGrids.get())
        return [strline]               
    #--------------------------- UTILS functions --------------------------------------------        
    def getCCThreshold(self):
        fnCorr = self._getExtraPath("correlations.xmd")               
        mdCorr = emlib.MetaData("corrThreshold@"+fnCorr)
        return mdCorr.getValue(emlib.MDL_WEIGHT, mdCorr.firstObject())
    
    def _storeSummaryInfo(self, numVolumes):
        """ Store some information when the protocol finishes. """
        msg1 = ''
        msg2 = ''
        
        for n in range(numVolumes):
            fnBase = 'proposedVolume%05d' % n
            fnRoot = self._getPath(fnBase + ".xmd")
                               
            if os.path.isfile(fnRoot):
                md = emlib.MetaData(fnRoot)
                size = md.size()
                if (size < 5):
                    msg1 = "Num of inliers for model %d too small and equal to %d \n" % (n, size)
                    msg1 += "Decrease the value of Inlier Threshold parameter and run again \n"
                 
        fnRoot = self._getTmpPath("ransac00000.xmd")
        if os.path.isfile(fnRoot):
            md = emlib.MetaData(fnRoot)
            size = md.size()
            if (size < 5):
                msg2 = "Num of random samples too small and equal to %d.\n" % size
                msg2 += "If the option Dimensionality reduction is on, increase the number of grids per dimension (In this case we recommend to put Dimensionality reduction off).\n"
                msg2 += "If the option Dimensionality reduction is off, increase the number of random samples.\n"
                
        msg = msg1 + msg2
        self.summaryVar.set(msg)

    def _citations(self):
        return ['Vargas2014']
