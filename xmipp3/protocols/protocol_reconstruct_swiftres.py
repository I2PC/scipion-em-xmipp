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
        form.addParam('resolutionLimit', FloatParam, label="Resolution limit (A)", default=10.0)
        form.addParam('angularSampling', FloatParam, label="Angular sampling (º)", default=5.0)
        form.addParam('shiftCount', IntParam, label="Shifts", default=9)
        form.addParam('maxShift', FloatParam, label="Maximum shift (%)", default=10.0)

        form.addParallelSection(threads=1, mpi=8)
    
    #--------------------------- INFO functions --------------------------------------------

    
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):
        convertInputStepId = self._insertFunctionStep('convertInputStep', prerequisites=[])
        correctCtfStepId = self._insertFunctionStep('correctCtfStep', prerequisites=[convertInputStepId])
        projectVolumeStepId = self._insertFunctionStep('projectVolumeStep', prerequisites=[convertInputStepId])
        trainDatabaseStepId = self._insertFunctionStep('trainDatabaseStep', prerequisites=[projectVolumeStepId])
        alignStepId = self._insertFunctionStep('alignStep', prerequisites=[correctCtfStepId, trainDatabaseStepId])
        reconstructId = self._insertReconstructSteps(prerequisites=[alignStepId])
        createOutputStepId = self._insertFunctionStep('createOutputStep', prerequisites=reconstructId)
 
    def _insertReconstructSteps(self, prerequisites):
        splitStepId = self._insertFunctionStep('splitStep', prerequisites=prerequisites)
        reconstructStepId1 = self._insertFunctionStep('reconstructStep', 1, prerequisites=[splitStepId])
        reconstructStepId2 = self._insertFunctionStep('reconstructStep', 2, prerequisites=[splitStepId])
        computeFscStepId = self._insertFunctionStep('computeFscStep', prerequisites=[reconstructStepId1, reconstructStepId2])
        averageVolumeStepId = self._insertFunctionStep('averageVolumeStep', prerequisites=[reconstructStepId1, reconstructStepId2])
        return [computeFscStepId, averageVolumeStepId]
    
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
    
    def projectVolumeStep(self):
        args = []
        args += ['-i', self._getInputVolumeFilename()]
        args += ['-o', self._getGalleryStackFilename()]
        args += ['--sampling_rate', self.angularSampling]
        args += ['--sym', self.symmetryGroup]
        
        self.runJob('xmipp_angular_project_library', args)
    
    def trainDatabaseStep(self):
        expectedSize = int(2e6) # TODO determine form gallery
        trainingSize = int(2e6) # TODO idem

        args = []
        args += ['-i', self._getGalleryMdFilename()]
        args += ['-o', self._getTrainingIndexFilename()]
        #args += ['--weights', self._getWeightsFilename()]
        args += ['--max_shift', self._getMaxShift()]
        args += ['--max_frequency', self._getDigitalFrequencyLimit()]
        args += ['--method', 'fourier']
        args += ['--size', expectedSize]
        args += ['--training', trainingSize]
        if self.useGpu:
            args += ['--gpu', 0] # TODO select
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_train_database', args, numberOfMpi=1, env=env)
    
    def alignStep(self):
        batchSize = 1024
        nRotations = round(360 / float(self.angularSampling))
        nShift = self.shiftCount
        
        args = []
        args += ['-i', self._getWienerParticleMdFilename()]
        args += ['-o', self._getAlignmentMdFilename()]
        args += ['-r', self._getGalleryMdFilename()]
        args += ['--index', self._getTrainingIndexFilename()]
        #args += ['--weights', self._getWeightsFilename()]
        args += ['--max_shift', self._getMaxShift()]
        args += ['--rotations', nRotations]
        args += ['--shifts', nShift]
        args += ['--max_frequency', self._getDigitalFrequencyLimit()]
        args += ['--method', 'fourier']
        args += ['--dropna']
        args += ['--batch', batchSize]
        if self.useGpu:
            args += ['--gpu', 0] # TODO select
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_query_database', args, numberOfMpi=1, env=env)
    
    def splitStep(self):
        args = []
        args += ['-i', self._getAlignmentMdFilename()]
        args += ['-n', 2]
        
        self.runJob('xmipp_metadata_split', args, numberOfMpi=1)
    
    def reconstructStep(self, i: int):
        args = []
        args += ['-i', self._getAlignmentHalfMdFilename(i)]
        args += ['-o', self._getHalfVolumeFilename(i)]
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
        
        
    def computeFscStep(self):
        args = []
        args += ['--ref', self._getHalfVolumeFilename(1)]
        args += ['-i', self._getHalfVolumeFilename(2)]
        args += ['-o', self._getFscFilename()]
        args += ['--sampling_rate', self._getSamplingRate()]
        
        self.runJob('xmipp_resolution_fsc', args, numberOfMpi=1)
    
    def averageVolumeStep(self):
        args = []
        args += ['-i', self._getHalfVolumeFilename(1)]
        args += ['--plus', self._getHalfVolumeFilename(2)]
        args += ['-o', self._getAverageVolumeFilename()]
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)

        args = []
        args += ['-i', self._getAverageVolumeFilename()]
        args += ['--mult', '0.5']
        self.runJob('xmipp_image_operate', args, numberOfMpi=1)
    
    def createOutputStep(self):
        self._createOutputParticleSet()
        self._createOutputVolume()
        self._createOutputFsc()
        
    
    #--------------------------- UTILS functions --------------------------------------------        
    def _getDigitalFrequencyLimit(self):
        return self.inputParticles.get().getSamplingRate() / float(self.resolutionLimit)
    
    def _getMaxShift(self):
        return float(self.maxShift) / 100.0
    
    def _getSamplingRate(self):
        return self.inputParticles.get().getSamplingRate()
    
    def _getInputParticleMdFilename(self):
        return self._getExtraPath('input_particles.xmd')
    
    def _getWienerParticleMdFilename(self):
        return self._getExtraPath('input_particles_wiener.xmd')
    
    def _getWienerParticleStackFilename(self):
        return self._getExtraPath('input_particles_wiener.mrcs')
    
    def _getInputVolumeFilename(self):
        return self.inputVolume.get().getFileName()
    
    def _getGalleryMdFilename(self):
        return self._getExtraPath('gallery.doc')
    
    def _getGalleryStackFilename(self):
        return self._getExtraPath('gallery.mrcs')
    
    def _getWeightsFilename(self):
        return self._getExtraPath('weights.mrc')
    
    def _getTrainingIndexFilename(self):
        return self._getExtraPath('database.idx')
    
    def _getAlignmentMdFilename(self):
        return self._getExtraPath('aligned.xmd')
    
    def _getAlignmentHalfMdFilename(self, i: int):
        return self._getExtraPath('aligned%06d.xmd' % i)

    def _getHalfVolumeFilename(self, i: int):
        return self._getExtraPath('volume%01d.mrc' % i)
    
    def _getAverageVolumeFilename(self):
        return self._getExtraPath('average_volume.mrc')
    
    def _getFscFilename(self):
        return self._getExtraPath('fsc.xmd')
    
    def _createOutputVolume(self):
        volume=Volume()
        
        # Fill
        volume.setFileName(self._getAverageVolumeFilename())
        volume.setSamplingRate(self._getSamplingRate())
        volume.setHalfMaps([
            self._getHalfVolumeFilename(1),
            self._getHalfVolumeFilename(2)
        ])
        
        # Define the output
        self._defineOutputs(outputVolume=volume)
        self._defineSourceRelation(self.inputParticles.get(), volume)
        
        return volume
    
    def _createOutputParticleSet(self):
        particleSet = self._createSetOfParticles()
        
        # TODO replace wiener corrected images
        
        # Fill
        readSetOfParticles(self._getAlignmentMdFilename(), particleSet)
        particleSet.setSamplingRate(self._getSamplingRate())
        
        # Define the output
        self._defineOutputs(outputParticles=particleSet)
        self._defineSourceRelation(self.inputParticles.get(), particleSet)
        
        return particleSet
    
    def _createOutputFsc(self):
        fsc = FSC()
        
        # Load from metadata
        fsc.loadFromMd(
            self._getFscFilename(),
            emlib.MDL_RESOLUTION_FREQ,
            emlib.MDL_RESOLUTION_FRC
        )
        
        # Define the output
        self._defineOutputs(outputFSC=fsc)
        self._defineSourceRelation(self.inputParticles.get(), fsc)
        
        return fsc
        