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
from pwem.objects import Volume, FSC
from pwem import emlib

from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )
from pyworkflow.utils.path import (cleanPath, makePath, copyFile, moveFile,
                                   createLink, cleanPattern)

import xmipp3
from xmipp3.convert import writeSetOfParticles, readSetOfParticles

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
        form.addParam('inputParticles', PointerParam, label="Particles", important=True,
                      pointerClass='SetOfParticles')
        form.addParam('inputVolume', PointerParam, label="Initial volumes", important=True,
                      pointerClass='Volume')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        
        form.addSection(label='Refinement')
        form.addParam('numberOfIterations', IntParam, label='Number of iterations', default=3)
        form.addParam('initialResolution', FloatParam, label="Initial resolution (A)", default=10.0)
        form.addParam('nextResolutionCriterion', FloatParam, label="FSC criterion", default=0.5, expertLevel=LEVEL_ADVANCED,
                      help='The resolution of the reconstruction is defined as the inverse of the frequency at which '\
                      'the FSC drops below this value. Typical values are 0.143 and 0.5' )
        form.addParam('angularSampling', FloatParam, label="Angular sampling (º)", default=5.0)
        form.addParam('shiftCount', IntParam, label="Shifts", default=9)
        form.addParam('maxShift', FloatParam, label="Maximum shift (%)", default=10.0)

        form.addSection(label='Compute')
        form.addParam('databaseRecipe', StringParam, label='Database recipe', 
                      default='OPQ48_192,IVF32768,PQ48' )
        form.addParam('databaseTrainingSetSize', IntParam, label='Database training set size', 
                      default=int(2e6) )

        form.addParallelSection(threads=1, mpi=8)
    
    #--------------------------- INFO functions --------------------------------------------

    
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):
        convertInputStepId = self._insertFunctionStep('convertInputStep', prerequisites=[])
        correctCtfStepId = self._insertFunctionStep('correctCtfStep', prerequisites=[convertInputStepId])
        
        lastIds = [correctCtfStepId]
        for i in range(int(self.numberOfIterations)):
            lastIds = self._insertIterationSteps(i, prerequisites=lastIds)
        
        self._insertFunctionStep('createOutputStep', prerequisites=lastIds)
 
    def _insertIterationSteps(self, iteration: int, prerequisites):
        setupIterationStepId = self._insertFunctionStep('setupIterationStep', iteration, prerequisites=prerequisites)
        alignIds = self._insertAlignmentSteps(iteration, prerequisites=[setupIterationStepId])
        reconstructIds = self._insertReconstructSteps(iteration, prerequisites=alignIds)
        postProcessIds = self._insertPostProcessSteps(iteration, prerequisites=reconstructIds)
        return postProcessIds
        
    def _insertAlignmentSteps(self, iteration: int, prerequisites):
        projectVolumeStepId = self._insertFunctionStep('projectVolumeStep', iteration, prerequisites=prerequisites)
        trainDatabaseStepId = self._insertFunctionStep('trainDatabaseStep', iteration, prerequisites=[projectVolumeStepId])
        alignStepId = self._insertFunctionStep('alignStep', iteration, prerequisites=[trainDatabaseStepId])
        compareReprojectionStepId = self._insertFunctionStep('compareReprojectionStep', iteration, prerequisites=[alignStepId])
        return [compareReprojectionStepId]
 
    def _insertReconstructSteps(self, iteration: int, prerequisites):
        splitStepId = self._insertFunctionStep('splitStep', iteration, prerequisites=prerequisites)
        reconstructStepId1 = self._insertFunctionStep('reconstructStep', iteration, 1, prerequisites=[splitStepId])
        reconstructStepId2 = self._insertFunctionStep('reconstructStep', iteration, 2, prerequisites=[splitStepId])
        computeFscStepId = self._insertFunctionStep('computeFscStep', iteration, prerequisites=[reconstructStepId1, reconstructStepId2])
        averageVolumeStepId = self._insertFunctionStep('averageVolumeStep', iteration, prerequisites=[reconstructStepId1, reconstructStepId2])
        return [computeFscStepId, averageVolumeStepId]
    
    def _insertPostProcessSteps(self, iteration: int, prerequisites):
        """
        filterVolumeStepId = self._insertFunctionStep('filterVolumeStep', iteration, prerequisites=prerequisites)
        return [filterVolumeStepId]
        """
        return prerequisites
    
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
        if (particles.isPhaseFlipped()):
            args +=  ['--phase_flipped']

        self.runJob('xmipp_ctf_correct_wiener2d', args)
    
    def setupIterationStep(self, iteration: int):
        makePath(self._getIterationPath(iteration))
    
    def projectVolumeStep(self, iteration: int):
        args = []
        args += ['-i', self._getIterationInputVolumeFilename(iteration)]
        args += ['-o', self._getGalleryStackFilename(iteration)]
        args += ['--sampling_rate', self.angularSampling]
        args += ['--sym', self.symmetryGroup]
        
        self.runJob('xmipp_angular_project_library', args)
    
    def trainDatabaseStep(self, iteration: int):
        trainingSize = int(self.databaseTrainingSetSize)
        recipe = self.databaseRecipe

        args = []
        args += ['-i', self._getGalleryMdFilename(iteration)]
        args += ['-o', self._getTrainingIndexFilename(iteration)]
        args += ['--recipe', recipe]
        #args += ['--weights', self._getWeightsFilename(iteration)]
        args += ['--max_shift', self._getMaxShift()]
        args += ['--max_frequency', self._getIterationDigitalFrequencyLimit(iteration)]
        args += ['--method', 'fourier']
        args += ['--training', trainingSize]
        args += ['--scratch', self._getTrainingScratchFilename()]
        if self.useGpu:
            args += ['--gpu', 0] # TODO select
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_train_database', args, numberOfMpi=1, env=env)
    
    def alignStep(self, iteration: int):
        batchSize = 1024
        nRotations = round(360 / float(self.angularSampling))
        nShift = self.shiftCount
        
        args = []
        args += ['-i', self._getWienerParticleMdFilename()]
        args += ['-o', self._getAlignmentMdFilename(iteration)]
        args += ['-r', self._getGalleryMdFilename(iteration)]
        args += ['--index', self._getTrainingIndexFilename(iteration)]
        #args += ['--weights', self._getWeightsFilename(iteration)]
        args += ['--max_shift', self._getMaxShift()]
        args += ['--rotations', nRotations]
        args += ['--shifts', nShift]
        args += ['--max_frequency', self._getIterationDigitalFrequencyLimit(iteration)]
        args += ['--method', 'fourier']
        args += ['--dropna']
        args += ['--batch', batchSize]
        if self.useGpu:
            args += ['--gpu', 0] # TODO select
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_query_database', args, numberOfMpi=1, env=env)
    
    def compareReprojectionStep(self, iteration: int):
        args = []
        args += ['-i', self._getAlignmentMdFilename(iteration)]
        args += ['--ref', self._getIterationInputVolumeFilename(iteration)]
        args += ['--ignoreCTF'] # As we're using wiener corrected images
        args += ['--doNotWriteStack'] # Do not undo shifts
        self.runJob('xmipp_angular_continuous_assign2', args, numberOfMpi=self.numberOfMpi.get())
    
        args = []
        args += ['-i', self._getAlignmentMdFilename(iteration)]
        args += ['--operate', 'rename_column', 'weightContinuous2 weight']
        self._runMdUtils(args)
    
    def splitStep(self, iteration: int):
        args = []
        args += ['-i', self._getAlignmentMdFilename(iteration)]
        args += ['-n', 2]
        
        self.runJob('xmipp_metadata_split', args, numberOfMpi=1)
    
    def reconstructStep(self, iteration: int, half: int):
        args = []
        args += ['-i', self._getAlignmentHalfMdFilename(iteration, half)]
        args += ['-o', self._getHalfVolumeFilename(iteration, half)]
        args += ['--sym', self.symmetryGroup.get()]
        args += ['--weight']
    
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
        
        
    def computeFscStep(self, iteration: int):
        args = []
        args += ['--ref', self._getHalfVolumeFilename(iteration, 1)]
        args += ['-i', self._getHalfVolumeFilename(iteration, 2)]
        args += ['-o', self._getFscFilename(iteration)]
        args += ['--sampling_rate', self._getSamplingRate()]
        
        self.runJob('xmipp_resolution_fsc', args, numberOfMpi=1)
    
    def averageVolumeStep(self, iteration: int):
        args = []
        args += ['-i', self._getHalfVolumeFilename(iteration, 1)]
        args += ['--plus', self._getHalfVolumeFilename(iteration, 2)]
        args += ['-o', self._getAverageVolumeFilename(iteration)]
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)

        args = []
        args += ['-i', self._getAverageVolumeFilename(iteration)]
        args += ['--mult', '0.5']
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)
    
    def filterVolumeStep(self, iteration: int):
        mdFsc = emlib.MetaData(self._getFscFilename(iteration))
        resolution = self._computeResolution(mdFsc, self._getSamplingRate(), 0.5)

        args = []
        args += ['-i', self._getAverageVolumeFilename(iteration)]
        args += ['-o', self._getFilteredVolumeFilename(iteration)]
        args += ['--fourier', 'low_pass', resolution]
        args += ['--sampling', self._getSamplingRate()]

        self.runJob('xmipp_transform_filter', args, numberOfMpi=1)
    
    def createOutputStep(self):
        lastIteration = int(self.numberOfIterations) - 1

        # Keep only the image and id from the input particle set
        args = []
        args += ['-i', self._getAlignmentMdFilename(lastIteration)]
        args += ['-o', self._getOutputParticlesMdFilename()]
        args += ['--operate', 'rename_column', 'image image1']
        self._runMdUtils(args)
        
        # Add the rest from the last alignment
        args = []
        args += ['-i', self._getOutputParticlesMdFilename()]
        args += ['--set', 'join', self._getInputParticleMdFilename(), 'itemId']
        self._runMdUtils(args)
        
        # Link last iteration
        for i in range(1, 3):
            createLink(
                self._getHalfVolumeFilename(lastIteration, i), 
                self._getOutputHalfVolumeFilename(i)
            )
        createLink(
            self._getAverageVolumeFilename(lastIteration), # TODO replace with post-processed volume
            self._getOutputVolumeFilename()
        )
        createLink(
            self._getFscFilename(lastIteration), 
            self._getOutputFscFilename()
        )
        
        # Create output objects
        self._createOutputParticleSet()
        self._createOutputVolume()
        self._createOutputFsc()
        
    
    #--------------------------- UTILS functions --------------------------------------------        
    def _getMaxShift(self) -> float:
        return float(self.maxShift) / 100.0
    
    def _getSamplingRate(self) -> float:
        return float(self.inputParticles.get().getSamplingRate())
    
    def _getIterationResolutionLimit(self, iteration: int) -> float:
        if iteration > 0:
            mdFsc = emlib.MetaData(self._getFscFilename(iteration-1))
            sampling = self._getSamplingRate()
            threshold = float(self.nextResolutionCriterion)
            return self._computeResolution(mdFsc, sampling, threshold)
        else:
            return float(self.initialResolution)
    
    def _getIterationDigitalFrequencyLimit(self, iteration: int) -> float:
        return self._getSamplingRate() / self._getIterationResolutionLimit(iteration)
    
    def _getIterationPath(self, iteration: int, *paths):
        return self._getExtraPath('iteration_%04d' % iteration, *paths)
    
    def _getInputParticleMdFilename(self):
        return self._getExtraPath('input_particles.xmd')
    
    def _getWienerParticleMdFilename(self):
        return self._getExtraPath('input_particles_wiener.xmd')
    
    def _getWienerParticleStackFilename(self):
        return self._getExtraPath('input_particles_wiener.mrcs')
    
    def _getInputVolumeFilename(self):
        return self.inputVolume.get().getFileName()
    
    def _getIterationInputVolumeFilename(self, iteration: int):
        if iteration > 0:
            return self._getAverageVolumeFilename(iteration-1) # TODO replace with post processed volume
        else:
            return self._getInputVolumeFilename()
    
    def _getGalleryMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'gallery.doc')
    
    def _getGalleryStackFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'gallery.mrcs')
    
    def _getWeightsFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'weights.mrc')
    
    def _getTrainingIndexFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'database.idx')
    
    def _getAlignmentMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'aligned.xmd')
    
    def _getAlignmentHalfMdFilename(self, iteration: int, half: int):
        return self._getIterationPath(iteration, 'aligned%06d.xmd' % half)

    def _getHalfVolumeFilename(self, iteration: int, half: int):
        return self._getIterationPath(iteration, 'volume_half%01d.mrc' % half)
    
    def _getAverageVolumeFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'volume_avg.mrc')
    
    def _getFscFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'fsc.xmd')
    
    def _getFilteredVolumeFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'volume_filtered.mrc')
    
    def _getOutputParticlesMdFilename(self):
        return self._getExtraPath('output_particles.xmd')
    
    def _getOutputVolumeFilename(self):
        return self._getExtraPath('output_volume.mrc')

    def _getOutputHalfVolumeFilename(self, half: int):
        return self._getExtraPath('output_volume_half%01d.mrc' % half)
    
    def _getOutputFscFilename(self):
        return self._getExtraPath('output_fsc.xmd')
    
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
    
    def _createOutputVolume(self):
        volume=Volume()
        
        # Fill
        volume.setFileName(self._getOutputVolumeFilename())
        volume.setSamplingRate(self._getSamplingRate())
        volume.setHalfMaps([
            self._getOutputHalfVolumeFilename(1),
            self._getOutputHalfVolumeFilename(2),
        ])
        
        # Define the output
        self._defineOutputs(outputVolume=volume)
        self._defineSourceRelation(self.inputParticles.get(), volume)
        
        return volume
    
    def _createOutputParticleSet(self):
        particleSet = self._createSetOfParticles()
        
        """
        EXTRA_LABELS = [
            emlib.MDL_COST,
            emlib.MDL_WEIGHT,
            emlib.MDL_CORRELATION_IDX,
            emlib.MDL_CORRELATION_MASK,
            emlib.MDL_CORRELATION_WEIGHT,
            emlib.MDL_IMED
        ]
        """
        
        # Fill
        readSetOfParticles(
            self._getOutputParticlesMdFilename(), 
            particleSet 
            #extraLabels=EXTRA_LABELS
        )
        particleSet.setSamplingRate(self._getSamplingRate())
        
        # Define the output
        self._defineOutputs(outputParticles=particleSet)
        self._defineSourceRelation(self.inputParticles.get(), particleSet)
        
        return particleSet
    
    def _createOutputFsc(self):
        fsc = FSC()
        
        # Load from metadata
        fsc.loadFromMd(
            self._getOutputFscFilename(),
            emlib.MDL_RESOLUTION_FREQ,
            emlib.MDL_RESOLUTION_FRC
        )
        
        # Define the output
        self._defineOutputs(outputFSC=fsc)
        self._defineSourceRelation(self.inputParticles.get(), fsc)
        
        return fsc
    
    def _runMdUtils(self, args):
        self.runJob('xmipp_metadata_utilities', args, numberOfMpi=1)
        
    