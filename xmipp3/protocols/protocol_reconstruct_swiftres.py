# ***************************************************************************
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

from pwem.protocols import ProtRefine3D
from pwem.objects import Volume, FSC, SetOfVolumes, Class3D
from pwem import emlib

from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        MultiPointerParam,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )
from pyworkflow.utils.path import (cleanPath, makePath, copyFile, moveFile,
                                   createLink, cleanPattern)

import xmipp3
from xmipp3.convert import writeSetOfParticles, readSetOfParticles

from typing import Iterable
import math

class XmippProtReconstructSwiftres(ProtRefine3D, xmipp3.XmippProtocol):
    _label = 'swiftres'
    _conda_env = 'xmipp_torch'
        
    def __init__(self, **kwargs):
        ProtRefine3D.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form: Form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label='Particles', important=True,
                      pointerClass='SetOfParticles',
                      help='Input particle set')
        form.addParam('considerInputAlignment', BooleanParam, label='Consider previous alignment',
                      default=True,
                      help='Consider the alignment of input particles')
        form.addParam('inputVolumes', MultiPointerParam, label='Initial volumes', important=True,
                      pointerClass='Volume', minNumObjects=1,
                      help='Provide a volume for each class of interest')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        
        form.addSection(label='Global refinement')
        form.addParam('numberOfIterations', IntParam, label='Number of iterations', default=3)
        form.addParam('initialResolution', FloatParam, label="Initial resolution (A)", default=10.0,
                      help='Image comparison resolution limit at the first iteration of the refinement')
        form.addParam('maximumResolution', FloatParam, label="Maximum resolution (A)", default=8.0,
                      help='Image comparison resolution limit of the refinement')
        form.addParam('nextResolutionCriterion', FloatParam, label="FSC criterion", default=0.5, 
                      expertLevel=LEVEL_ADVANCED,
                      help='The resolution of the reconstruction is defined as the inverse of the frequency at which '\
                      'the FSC drops below this value. Typical values are 0.143 and 0.5' )
        form.addParam('initialMaxPsi', FloatParam, label='Maximum psi (deg)', default=180.0,
                      expertLevel=LEVEL_ADVANCED,
                      help='Maximum psi parameter of the particles')
        form.addParam('initialMaxShift', FloatParam, label='Maximum shift (px)', default=16.0,
                      help='Maximum shift of the particle in pixels')
        form.addParam('reconstructPercentage', FloatParam, label='Reconstruct percentage (%)', default=50,
                      help='Percentage of best particles used for reconstruction')

        form.addSection(label='Compute')
        form.addParam('databaseRecipe', StringParam, label='Database recipe', 
                      default='OPQ48_192,IVF32768,PQ48', expertLevel=LEVEL_ADVANCED,
                      help='FAISS database structure. Please refer to '
                      'https://github.com/facebookresearch/faiss/wiki/The-index-factory')
        form.addParam('databaseTrainingSetSize', IntParam, label='Database training set size', 
                      default=int(2e6),
                      help='Number of data-augmented particles to used when training the database')
        form.addParam('databaseMaximumSize', IntParam, label='Database size limit', 
                      default=int(10e6),
                      help='Maximum number of elements that can be stored in the database '
                      'before performing an alignment and flush')
        form.addParam('batchSize', IntParam, label='Batch size', 
                      default=1024,
                      help='Batch size used when processing')

        form.addParallelSection(threads=1, mpi=8)
    
    #--------------------------- INFO functions --------------------------------------------

    
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):
        convertInputStepId = self._insertFunctionStep('convertInputStep', prerequisites=[])
        correctCtfStepId = self._insertFunctionStep('correctCtfStep', prerequisites=[convertInputStepId])
        
        lastIds = [correctCtfStepId]
        for i in range(self._getIterationCount()):
            lastIds = self._insertIterationSteps(i, False, prerequisites=lastIds)
        
        self._insertFunctionStep('createOutputStep', prerequisites=lastIds)
 
    def _insertIterationSteps(self, iteration: int, local: bool, prerequisites):
        setupIterationStepId = self._insertFunctionStep('setupIterationStep', iteration, local, prerequisites=prerequisites)
        ctfGroupStepId = self._insertFunctionStep('ctfGroupStep', iteration, prerequisites=[setupIterationStepId])
        projectIds = self._insertProjectSteps(iteration, prerequisites=[setupIterationStepId])
        alignIds = self._insertAlignmentSteps(iteration, local, prerequisites=projectIds + [ctfGroupStepId])
        compareAnglesStepId = self._insertFunctionStep('compareAnglesStep', iteration, prerequisites=alignIds)

        ids = []
        for cls in range(self._getClassCount()):
            reconstructIds = self._insertReconstructSteps(iteration, cls, prerequisites=alignIds)
            postProcessIds = self._insertPostProcessSteps(iteration, cls, prerequisites=reconstructIds)
            ids += postProcessIds
        
        return ids + [compareAnglesStepId]
        
    def _insertProjectSteps(self, iteration: int, prerequisites):
        # Project all volumes
        projectStepIds = []
        for cls in range(self._getClassCount()):
            projectStepIds.append(self._insertFunctionStep('projectVolumeStep', iteration, cls, prerequisites=prerequisites))
        
        # Merge galleries
        mergeGalleriesStepId = self._insertFunctionStep('mergeGalleriesStep', iteration, prerequisites=projectStepIds)
        
        return [mergeGalleriesStepId]
        
    def _insertAlignmentSteps(self, iteration: int, local: bool, prerequisites):
        trainDatabaseStepId = self._insertFunctionStep('trainDatabaseStep', iteration, prerequisites=prerequisites)
        alignStepId = self._insertFunctionStep('alignStep', iteration, local, prerequisites=[trainDatabaseStepId])
        intersectInputStepId = self._insertFunctionStep('intersectInputStep', iteration, prerequisites=[alignStepId])
        
        return [intersectInputStepId]
 
    def _insertReconstructSteps(self, iteration: int, cls: int, prerequisites):
        selectAlignmentStepId = self._insertFunctionStep('selectAlignmentStep', iteration, cls, prerequisites=prerequisites)
        compareReprojectionStepId = self._insertFunctionStep('compareReprojectionStep', iteration, cls, prerequisites=[selectAlignmentStepId])
        computeWeightsStepId = self._insertFunctionStep('computeWeightsStep', iteration, cls, prerequisites=[compareReprojectionStepId])
        #filterByWeightsStepId = self._insertFunctionStep('filterByWeightsStep', iteration, cls, prerequisites=[computeWeightsStepId])
        splitStepId = self._insertFunctionStep('splitStep', iteration, cls, prerequisites=[computeWeightsStepId])
        reconstructStepId1 = self._insertFunctionStep('reconstructStep', iteration, cls, 1, prerequisites=[splitStepId])
        reconstructStepId2 = self._insertFunctionStep('reconstructStep', iteration, cls, 2, prerequisites=[splitStepId])
        computeFscStepId = self._insertFunctionStep('computeFscStep', iteration, cls, prerequisites=[reconstructStepId1, reconstructStepId2])
        averageVolumeStepId = self._insertFunctionStep('averageVolumeStep', iteration, cls, prerequisites=[reconstructStepId1, reconstructStepId2])
        return [computeFscStepId, averageVolumeStepId]
    
    def _insertPostProcessSteps(self, iteration: int, cls: int, prerequisites):
        filterVolumeStepId = self._insertFunctionStep('filterVolumeStep', iteration, cls, prerequisites=prerequisites)
        return [filterVolumeStepId]
    
    #--------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(), 
                            self._getInputParticleMdFilename())
    
    def correctCtfStep(self):
        particles = self.inputParticles.get()

        args = []
        args += ['-i', self._getInputParticleMdFilename()]
        args += ['-o', self._getWienerParticleStackFilename()]
        args += ['--save_metadata_stack', self._getWienerParticleMdFilename()]
        args += ['--keep_input_columns']
        args += ['--sampling_rate', self._getSamplingRate()]
        args += ['--pad', '2']
        args += ['--wc', '-1.0']
        if particles.isPhaseFlipped():
            args +=  ['--phase_flipped']

        self.runJob('xmipp_ctf_correct_wiener2d', args)
        
    def setupIterationStep(self, iteration: int, local: bool):
        makePath(self._getIterationPath(iteration))
        
        for cls in range(self._getClassCount()):
            makePath(self._getClassPath(iteration, cls))
            
        if iteration > 0:
            # Fill missing particles from the previous alignment with
            # The input particles. In order to use union, columns must
            # match, so join columns prior to doing the union
            args = []
            args += ['-i', self._getWienerParticleMdFilename()]
            args += ['-o', self._getIterationInputParticleMdFilename(iteration)]
            args += ['--set', 'join', self._getAlignmentMdFilename(iteration-1), 'itemId']
            self._runMdUtils(args)
            
            args = []
            args += ['-i', self._getAlignmentMdFilename(iteration-1)]
            args += ['-o', self._getIterationInputParticleMdFilename(iteration)]
            args += ['--set', 'union', self._getIterationInputParticleMdFilename(iteration), 'itemId']
            self._runMdUtils(args)
            
            resolution = self._computeIterationResolution(iteration-1)
            
        else:
            # For the first iteration, simply use the input particles.
            createLink(
                self._getWienerParticleMdFilename(),
                self._getIterationInputParticleMdFilename(iteration)
            )
            
            resolution = float(self.initialResolution)
        
        frequency = self._getSamplingRate() / resolution
        maxPsi = self._getIterationMaxPsi(iteration)
        maxShift = self._getIterationMaxShift(iteration)
        shiftStep = self._computeShiftStep(frequency)
        angleStep = self._computeAngleStep(frequency, 160) #TODO
        maxResolution = max(resolution, float(self.maximumResolution))
        maxFrequency = self._getSamplingRate() / maxResolution
        
        # Write to metadata
        md = emlib.MetaData()
        id = md.addObject()
        md.setValue(emlib.MDL_RESOLUTION_FREQ, maxResolution, id)
        md.setValue(emlib.MDL_RESOLUTION_FREQREAL, maxFrequency, id)
        md.setValue(emlib.MDL_ANGLE_PSI, maxPsi, id)
        md.setValue(emlib.MDL_SHIFT_X, maxShift, id)
        md.setValue(emlib.MDL_SHIFT_Y, maxShift, id)
        md.setValue(emlib.MDL_SHIFT_DIFF, shiftStep, id)
        md.setValue(emlib.MDL_ANGLE_DIFF, angleStep, id)
        md.write(self._getIterationParametersFilename(iteration))
        
    def ctfGroupStep(self, iteration: int):
        md = emlib.MetaData(self._getIterationParametersFilename(iteration))
        resolution = md.getValue(emlib.MDL_RESOLUTION_FREQ, 1)
        particles = self.inputParticles.get()
        oroot = self._getCtfGroupOutputRoot(iteration)

        args = []
        args += ['--ctfdat', self._getInputParticleMdFilename()]
        args += ['-o', oroot]
        args += ['--sampling_rate', self._getSamplingRate()]
        args += ['--pad', 2]
        args += ['--error', 1.0] # TODO make param
        args += ['--resol', resolution]
        if particles.isPhaseFlipped():
            args +=  ['--phase_flipped']
        
        self.runJob('xmipp_ctf_group', args, numberOfMpi=1)

        # Rename files and removed unused ones
        cleanPath(oroot + 'Info.xmd')
        cleanPath(oroot + '_split.doc')
        moveFile(oroot + '_images.sel', self._getCtfGroupMdFilename(iteration))
        moveFile(oroot + '_ctf.mrcs', self._getCtfGroupAveragesFilename(iteration))
        
    def projectVolumeStep(self, iteration: int, cls: int):
        md = emlib.MetaData(self._getIterationParametersFilename(iteration))
        angleStep = md.getValue(emlib.MDL_ANGLE_DIFF, 1)
        perturb = math.sin(math.radians(angleStep)) / 4

        args = []
        args += ['-i', self._getIterationInputVolumeFilename(iteration, cls)]
        args += ['-o', self._getClassGalleryStackFilename(iteration, cls)]
        args += ['--sampling_rate', angleStep]
        args += ['--perturb', perturb]
        args += ['--sym', self.symmetryGroup]
        
        if False: # TODO speak with coss
            args += ['--compute_neighbors']
            args += ['--angular_distance', -1]    
            args += ['--experimental_images', self._getIterationInputParticleMdFilename(iteration)]

        self.runJob('xmipp_angular_project_library', args)
    
        args = []
        args += ['-i', self._getClassGalleryMdFilename(iteration, cls)]
        args += ['--fill', 'ref3d', 'constant', cls+1]
        self._runMdUtils(args)
    
    def mergeGalleriesStep(self, iteration):
        self._mergeMetadata(
            map(lambda cls : self._getClassGalleryMdFilename(iteration, cls), range(self._getClassCount())),
            self._getGalleryMdFilename(iteration)
        )

        # Reindex
        args = []
        args += ['-i', self._getGalleryMdFilename(iteration)]
        args += ['--fill', 'ref', 'lineal', 1, 1]
        self._runMdUtils(args)
    
    def trainDatabaseStep(self, iteration: int):
        trainingSize = int(self.databaseTrainingSetSize)
        recipe = self.databaseRecipe

        md = emlib.MetaData(self._getIterationParametersFilename(iteration))
        imageSize = 160 # TODO determine
        maxFrequency = md.getValue(emlib.MDL_RESOLUTION_FREQREAL, 1)
        maxPsi = md.getValue(emlib.MDL_ANGLE_PSI, 1)
        maxShiftPx = md.getValue(emlib.MDL_SHIFT_X, 1)
        maxShift = maxShiftPx / imageSize

        args = []
        args += ['-i', self._getGalleryMdFilename(iteration)]
        args += ['-o', self._getTrainingIndexFilename(iteration)]
        args += ['--recipe', recipe]
        #args += ['--weights', self._getWeightsFilename(iteration)]
        args += ['--max_shift', maxShift]
        args += ['--max_psi', maxPsi]
        args += ['--max_frequency', maxFrequency]
        args += ['--method', 'fourier']
        args += ['--training', trainingSize]
        args += ['--batch', self.batchSize]
        args += ['--scratch', self._getTrainingScratchFilename()]
        if self.useGpu:
            args += ['--device', 'cuda:0'] # TODO select
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_train_database', args, numberOfMpi=1, env=env)
    
    def alignStep(self, iteration: int, local: bool):
        md = emlib.MetaData(self._getIterationParametersFilename(iteration))
        imageSize = 160 # TODO determine
        maxFrequency = md.getValue(emlib.MDL_RESOLUTION_FREQREAL, 1)
        maxPsi = md.getValue(emlib.MDL_ANGLE_PSI, 1)
        maxShiftPx = md.getValue(emlib.MDL_SHIFT_X, 1)
        maxShift = maxShiftPx / imageSize
        nShift = round((2*maxShiftPx) / md.getValue(emlib.MDL_SHIFT_DIFF, 1)) + 1
        nRotations = round(360 / md.getValue(emlib.MDL_ANGLE_DIFF, 1))

        # Perform the alignment
        args = []
        args += ['-i', self._getIterationInputParticleMdFilename(iteration)]
        args += ['-o', self._getAlignmentMdFilename(iteration)]
        args += ['-r', self._getGalleryMdFilename(iteration)]
        args += ['--index', self._getTrainingIndexFilename(iteration)]
        #args += ['--weights', self._getWeightsFilename(iteration)]
        args += ['--max_shift', maxShift]
        args += ['--max_psi', maxPsi]
        args += ['--rotations', nRotations]
        args += ['--shifts', nShift]
        args += ['--max_frequency', maxFrequency]
        args += ['--method', 'fourier']
        args += ['--dropna']
        args += ['--batch', self.batchSize]
        args += ['--max_size', self.databaseMaximumSize]
        if self.useGpu:
            args += ['--device', 'cuda:0'] # TODO select
        if local:
            args += ['--local_shift', '--local_psi']
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_query_database', args, numberOfMpi=1, env=env)

    def intersectInputStep(self, iteration: int):
        args = []
        args += ['-i', self._getIterationInputParticleMdFilename(iteration)]
        args += ['-o', self._getInputIntersectionMdFilename(iteration)]
        args += ['--set', 'intersection', self._getAlignmentMdFilename(iteration), 'itemId']
        self._runMdUtils(args)

    def compareAnglesStep(self, iteration: int):
        oroot = self._getAngleDiffOutputRoot(iteration)
        
        args = []
        args += ['--ang1', self._getAlignmentMdFilename(iteration)]
        args += ['--ang2', self._getInputIntersectionMdFilename(iteration)]
        args += ['--oroot', oroot]
        args += ['--sym', self.symmetryGroup]
        self.runJob('xmipp_angular_distance', args, numberOfMpi=1)
        
        # Rename files to have xmd extension
        moveFile(oroot+'_vec_diff_hist.txt', self._getAngleDiffVecDiffHistogramMdFilename(iteration))
        moveFile(oroot+'_shift_diff_hist.txt', self._getAngleDiffShiftDiffHistogramMdFilename(iteration))
        
    def selectAlignmentStep(self, iteration: int, cls: int):
        args = []
        args += ['-i', self._getAlignmentMdFilename(iteration)]
        args += ['-o', self._getReconstructionMdFilename(iteration, cls)]
        args += ['--query', 'select', 'ref3d==%d' % (cls+1)]
        self._runMdUtils(args)
    
    def compareReprojectionStep(self, iteration: int, cls: int):
        args = []
        args += ['-i', self._getReconstructionMdFilename(iteration, cls)]
        args += ['--ref', self._getIterationInputVolumeFilename(iteration, cls)]
        args += ['--ignoreCTF'] # As we're using wiener corrected images
        args += ['--doNotWriteStack'] # Do not undo shifts
        self.runJob('xmipp_angular_continuous_assign2', args, numberOfMpi=self.numberOfMpi.get())
    
    def computeWeightsStep(self, iteration: int, cls: int):
        args = []
        args += ['-i', self._getReconstructionMdFilename(iteration, cls)]
        args += ['--fill', 'weight', 'constant', 0.0]
        self._runMdUtils(args)
        
        args = []
        args += ['-i', self._getReconstructionMdFilename(iteration, cls)]
        args += ['--operate', 'modify_values', 'weight=corrIdx*corrWeight*corrMask']
        self._runMdUtils(args)

    def filterByWeightsStep(self, iteration: int, cls: int):
        args = []
        args += ['-i', self._getReconstructionMdFilename(iteration, cls)]
        args += ['-o', self._getFilteredReconstructionMdFilename(iteration, cls)]
        args += ['--operate', 'percentile', 'weight', 'weight']
        self._runMdUtils(args)

        args = []
        args += ['-i', self._getFilteredReconstructionMdFilename(iteration, cls)]
        args += ['-o', self._getFilteredReconstructionMdFilename(iteration, cls)]
        args += ['--query', 'select', 'weight>=%f' % self._getReconstructPercentile()]
        self._runMdUtils(args)

    def splitStep(self, iteration: int, cls: int):
        args = []
        args += ['-i', self._getReconstructionMdFilename(iteration, cls)]
        args += ['-n', 2]
        
        self.runJob('xmipp_metadata_split', args, numberOfMpi=1)
    
    def reconstructStep(self, iteration: int, cls: int, half: int):
        args = []
        args += ['-i', self._getReconstructionHalfMdFilename(iteration, cls, half)]
        args += ['-o', self._getHalfVolumeFilename(iteration, cls, half)]
        args += ['--sym', self.symmetryGroup.get()]
        args += ['--weight'] # TODO determine if used
    
        # Determine the execution parameters
        numberOfMpi = self.numberOfMpi.get()
        if self.useGpu.get():
            reconstructProgram = 'xmipp_cuda_reconstruct_fourier'

            gpuList = self.getGpuList()
            if self.numberOfMpi.get() > 1:
                numberOfGpus = len(gpuList)
                numberOfMpi = numberOfGpus + 1
                args += ['-gpusPerNode', numberOfGpus]
                args += ['-threadsPerGPU', max(self.numberOfThreads.get(), 4)]
            else:
                args += ['--device', ','.join(gpuList)]
                
            """
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
            """
                
            args += ['--thr', self.numberOfThreads.get()]
        
        else:
            reconstructProgram = 'xmipp_reconstruct_fourier_accel'
        
        # Run
        self.runJob(reconstructProgram, args, numberOfMpi=numberOfMpi)
        
    def computeFscStep(self, iteration: int, cls: int):
        args = []
        args += ['--ref', self._getHalfVolumeFilename(iteration, cls, 1)]
        args += ['-i', self._getHalfVolumeFilename(iteration, cls, 2)]
        args += ['-o', self._getFscFilename(iteration, cls)]
        args += ['--sampling_rate', self._getSamplingRate()]
        
        self.runJob('xmipp_resolution_fsc', args, numberOfMpi=1)
    
    def averageVolumeStep(self, iteration: int, cls: int):
        args = []
        args += ['-i', self._getHalfVolumeFilename(iteration, cls, 1)]
        args += ['--plus', self._getHalfVolumeFilename(iteration, cls, 2)]
        args += ['-o', self._getAverageVolumeFilename(iteration, cls)]
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)

        args = []
        args += ['-i', self._getAverageVolumeFilename(iteration, cls)]
        args += ['--mult', '0.5']
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)
    
    def filterVolumeStep(self, iteration: int, cls: int):
        mdFsc = emlib.MetaData(self._getFscFilename(iteration, cls))
        resolution = self._computeResolution(mdFsc, self._getSamplingRate(), 0.5)

        args = []
        args += ['-i', self._getAverageVolumeFilename(iteration, cls)]
        args += ['-o', self._getFilteredVolumeFilename(iteration, cls)]
        args += ['--fourier', 'low_pass', resolution]
        args += ['--sampling', self._getSamplingRate()]

        self.runJob('xmipp_transform_filter', args, numberOfMpi=1)
    
    def mergeAlignmentsStep(self, iteration: int):
        self._mergeMetadata(
            map(lambda cls : self._getReconstructionMdFilename(iteration, cls), range(self._getClassCount())),
            self._getAlignmentMdFilename(iteration)
        )
    
    def createOutputStep(self):
        lastIteration = self._getIterationCount() - 1

        # Rename the wiener filtered image column
        args = []
        args += ['-i', self._getAlignmentMdFilename(lastIteration)]
        args += ['-o', self._getOutputParticlesMdFilename()]
        args += ['--operate', 'rename_column', 'image image1']
        self._runMdUtils(args)
        
        # Add the input image column
        args = []
        args += ['-i', self._getOutputParticlesMdFilename()]
        args += ['--set', 'join', self._getInputParticleMdFilename(), 'itemId']
        self._runMdUtils(args)
        
        # Link last iteration
        for cls in range(self._getClassCount()):
            for i in range(1, 3):
                createLink(
                    self._getHalfVolumeFilename(lastIteration, cls, i), 
                    self._getOutputHalfVolumeFilename(cls, i)
                )
                
            createLink(
                self._getFilteredVolumeFilename(lastIteration, cls),
                self._getOutputVolumeFilename(cls)
            )
            createLink(
                self._getFscFilename(lastIteration, cls), 
                self._getOutputFscFilename(cls)
            )
        
        # Create output objects
        volumes = self._createOutputVolumes()
        self._createOutputClasses3D(volumes)
        self._createOutputFscs()
        
    
    #--------------------------- UTILS functions --------------------------------------------        
    def _getIterationCount(self) -> int:
        return int(self.numberOfIterations)
        
    def _getClassCount(self) -> int:
        return len(self.inputVolumes)
    
    def _getReconstructPercentile(self) -> float:
        return 1.0 - (float(self.reconstructPercentage) / 100.0)
    
    def _getSamplingRate(self) -> float:
        return float(self.inputParticles.get().getSamplingRate())
    

    def _getIterationPath(self, iteration: int, *paths):
        return self._getExtraPath('iteration_%04d' % iteration, *paths)
    
    def _getClassPath(self, iteration: int, cls: int, *paths):
        return self._getIterationPath(iteration, 'class_%06d' % cls, *paths)
    
    def _getInputParticleMdFilename(self):
        return self._getExtraPath('input_particles.xmd')
    
    def _getWienerParticleMdFilename(self):
        return self._getExtraPath('input_particles_wiener.xmd')
    
    def _getWienerParticleStackFilename(self):
        return self._getExtraPath('input_particles_wiener.mrcs')
    
    def _getInputVolumeFilename(self, cls: int):
        return self.inputVolumes[cls].get().getFileName()
    
    def _getIterationParametersFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'params.xmd')
    
    def _getIterationInputVolumeFilename(self, iteration: int, cls: int):
        if iteration > 0:
            return self._getFilteredVolumeFilename(iteration-1, cls)
        else:
            return self._getInputVolumeFilename(cls)

    def _getIterationInputParticleMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'input_particles.xmd')
    
    def _getCtfGroupOutputRoot(self, iteration: int):
        return self._getIterationPath(iteration, 'ctf_group')
    
    def _getCtfGroupMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'ctf_groups.xmd')
    
    def _getCtfGroupAveragesFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'ctfs.mrcs')

    def _getClassGalleryMdFilename(self, iteration: int, cls: int):
        return self._getClassPath(iteration, cls, 'gallery.doc')
    
    def _getClassGalleryStackFilename(self, iteration: int, cls: int):
        return self._getClassPath(iteration, cls, 'gallery.mrcs')
    
    def _getGalleryMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'gallery.xmd')
    
    def _getWeightsFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'weights.mrc')
    
    def _getTrainingIndexFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'database.idx')
    
    def _getAlignmentMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'aligned.xmd')
    
    def _getInputIntersectionMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'input_intersection.xmd')
    
    def _getAngleDiffOutputRoot(self, iteration: int):
        return self._getIterationPath(iteration, 'angles')

    def _getAngleDiffMdFilename(self, iteration: int):
        return self._getAngleDiffOutputRoot(iteration) + '.xmd'

    def _getAngleDiffVecDiffHistogramMdFilename(self, iteration: int):
        return self._getAngleDiffOutputRoot(iteration) + '_vec_diff_hist.xmd'

    def _getAngleDiffShiftDiffHistogramMdFilename(self, iteration: int):
        return self._getAngleDiffOutputRoot(iteration) + '_shift_diff_hist.xmd'

    def _getReconstructionMdFilename(self, iteration: int, cls: int):
        return self._getClassPath(iteration, cls, 'aligned.xmd')
    
    def _getFilteredReconstructionMdFilename(self, iteration: int, cls: int):
        return self._getClassPath(iteration, cls, 'well_aligned.xmd')
    
    def _getReconstructionHalfMdFilename(self, iteration: int, cls: int, half: int):
        return self._getClassPath(iteration, cls, 'aligned%06d.xmd' % half)

    def _getHalfVolumeFilename(self, iteration: int, cls: int, half: int):
        return self._getClassPath(iteration, cls, 'volume_half%01d.mrc' % half)
    
    def _getAverageVolumeFilename(self, iteration: int, cls: int):
        return self._getClassPath(iteration, cls, 'volume_avg.mrc')
    
    def _getFscFilename(self, iteration: int, cls: int):
        return self._getClassPath(iteration, cls, 'fsc.xmd')
    
    def _getFilteredVolumeFilename(self, iteration: int, cls: int):
        return self._getClassPath(iteration, cls, 'volume_filtered.mrc')
    
    def _getOutputParticlesMdFilename(self):
        return self._getExtraPath('output_particles.xmd')
    
    def _getOutputVolumeFilename(self, cls: int):
        return self._getExtraPath('output_volume%06d.mrc' % cls)

    def _getOutputHalfVolumeFilename(self, cls: int, half: int):
        return self._getExtraPath('output_volume%06d_half%01d.mrc' % (cls, half))
    
    def _getOutputFscFilename(self, cls: int):
        return self._getExtraPath('output_fsc%06d.xmd' % cls)
    
    def _getTrainingScratchFilename(self):
        return self._getTmpPath('scratch.bin')
    
    def _computeResolution(self, mdFsc, Ts, threshold):
        resolution = 2 * Ts

        # Iterate until the FSC is under the threshold
        for objId in mdFsc:
            fsc = mdFsc.getValue(emlib.MDL_RESOLUTION_FRC, objId)
            if fsc < threshold:
                resolution = mdFsc.getValue(emlib.MDL_RESOLUTION_FREQREAL, objId)
                break
            
        return resolution
    
    def _computeIterationResolution(self, iteration: int) -> float:
        res = 0.0
        threshold = float(self.nextResolutionCriterion)
        for cls in range(self._getClassCount()):
            mdFsc = emlib.MetaData(self._getFscFilename(iteration, cls))
            sampling = self._getSamplingRate()
            res += self._computeResolution(mdFsc, sampling, threshold)
        
        res /= self._getClassCount()

        return res
    
    def _computeAngleStep(self, maxFrequency: float, size: int) -> float:
        # At the alignment resolution limit, determine the 
        # angle between to neighboring Fourier coefficients
        c = maxFrequency*size  # Cos: radius
        s = 1.0 # Sin: 1 coefficient
        angle = math.atan2(s, c)
        
        # The angular error is at most half of the sampling
        # Therefore use the double angular sampling
        angle *= 2
        
        return math.degrees(angle)
    
    def _computeShiftStep(self, digital_freq: float, eps: float = 0.5) -> float:
        # Assuming that in Nyquist (0.5) we are able to
        # detect a shift of eps pixels
        return (0.5 / digital_freq) * eps
    
    def _getIterationMaxPsi(self, iteration: int) -> float:
        #return float(self.initialMaxPsi) / math.pow(2.0, iteration)
        return float(self.initialMaxPsi)

    def _getIterationMaxShift(self, iteration: int) -> float:
        #return float(self.initialMaxShift) / math.pow(2.0, iteration)
        return float(self.initialMaxShift)
    
    def _createOutputClasses3D(self, volumes: SetOfVolumes):
        particles = self._createSetOfParticles()

        EXTRA_LABELS = [
            #emlib.MDL_COST,
            #emlib.MDL_WEIGHT,
            #emlib.MDL_CORRELATION_IDX,
            #emlib.MDL_CORRELATION_MASK,
            #emlib.MDL_CORRELATION_WEIGHT,
            #emlib.MDL_IMED
        ]
        
        # Fill
        readSetOfParticles(
            self._getOutputParticlesMdFilename(), 
            particles,
            extraLabels=EXTRA_LABELS
        )
        particles.setSamplingRate(self._getSamplingRate())
        self._insertChild('outputParticles', particles)
        
        def updateClass(cls: Class3D):
            clsId = cls.getObjId()
            representative = volumes[clsId]
            cls.setRepresentative(representative)
            
        classes3d = self._createSetOfClasses3D(particles)
        classes3d.classifyItems(updateClassCallback=updateClass)
        
        # Define the output
        self._defineOutputs(outputClasses=classes3d)
        self._defineSourceRelation(self.inputParticles, classes3d)
        self._defineSourceRelation(self.inputVolumes, classes3d)
        
        return classes3d
    
    def _createOutputVolumes(self):
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(self._getSamplingRate())
        
        for cls in range(self._getClassCount()):
            volume=Volume(objId=cls+1)
            
            # Fill
            volume.setFileName(self._getOutputVolumeFilename(cls))
            volume.setHalfMaps([
                self._getOutputHalfVolumeFilename(cls, 1),
                self._getOutputHalfVolumeFilename(cls, 2),
            ])
            
            volumes.append(volume)
        
        # Define the output
        self._defineOutputs(outputVolumes=volumes)
        self._defineSourceRelation(self.inputParticles, volumes)
        self._defineSourceRelation(self.inputVolumes, volumes)
        
        return volumes
    
    def _createOutputFscs(self):
        fscs = self._createSetOfFSCs()
        
        for cls in range(self._getClassCount()):
            fsc = FSC(objId=cls+1)
            
            # Load from metadata
            fsc.loadFromMd(
                self._getOutputFscFilename(cls),
                emlib.MDL_RESOLUTION_FREQ,
                emlib.MDL_RESOLUTION_FRC
            )
            
            fscs.append(fsc)
        
        # Define the output
        self._defineOutputs(outputFSC=fscs)
        self._defineSourceRelation(self.inputParticles, fscs)
        self._defineSourceRelation(self.inputVolumes, fscs)
        
        return fscs
    
    def _mergeMetadata(self, src: Iterable[str], dst: str):
        it = iter(src)
        
        # Copy the first alignment
        copyFile(next(it), dst)
        
        # Merge subsequent alignments
        for s in it:
            args = []
            args += ['-i', dst]
            args += ['--set', 'union_all', s] 
            self._runMdUtils(args)
    
    def _runMdUtils(self, args):
        self.runJob('xmipp_metadata_utilities', args, numberOfMpi=1)
