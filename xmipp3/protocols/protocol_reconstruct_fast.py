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
from pwem.objects import Volume, Transform, SetOfVolumes

from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )
from pyworkflow.utils.path import (cleanPath, makePath, copyFile, moveFile,
                                   createLink, cleanPattern)

import xmipp3
from xmipp3.convert import writeSetOfParticles

class XmippProtReconstructFast(ProtRefine3D, xmipp3.XmippProtocol):
    _label = 'fast reconstruct'
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
        form.addParam('correctCtf', BooleanParam, label="Correct CTF", default=True)
        
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
        reconstructIds = self._insertReconstructSteps(prerequisites=[alignStepId])
        createOutputStepId = self._insertFunctionStep('createOutputStep', prerequisites=reconstructIds)
 
    def _insertReconstructSteps(self, prerequisites):
        splitStepId = self._insertFunctionStep('splitStep', prerequisites=prerequisites)
        reconstructStepId1 = self._insertFunctionStep('reconstructStep', i=1, prerequisites=[splitStepId])
        reconstructStepId2 = self._insertFunctionStep('reconstructStep', i=2, prerequisites=[splitStepId])
        return [reconstructStepId1, reconstructStepId2]
 
    #--------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(), 
                            self._getInputParticleMdFilename())
    
    def correctCtfStep(self):
        particles = self.inputParticles.get()

        if self.correctCtf:
            # Wiener filtering is required
            args = []
            args += ['-i', self._getInputParticleMdFilename()]
            args += ['-o', self._getWienerParticleStackFilename()]
            args += ['--save_metadata_stack', self._getWienerParticleMdFilename()]
            args += ['--keep_input_columns']
            args += ['--sampling_rate', particles.getSamplingRate()]
            
            if (particles.isPhaseFlipped()):
                args +=  ['--phase_flipped']

            self.runJob('xmipp_ctf_correct_wiener2d', args)
        
        else:
            #  Wiener filtering is not required, link the input md
            createLink(
                self._getInputParticleMdFilename(),
                self._getWienerParticleMdFilename()
            )
    
    def projectVolumeStep(self):
        args = []
        args += ['-i', self._getInputVolumeFilename()]
        args += ['-o', self._getGalleryStackFilename()]
        args += ['--sampling_rate', self.angularSampling]
        args += ['--sym', self.symmetryGroup]
        
        self.runJob('xmipp_angular_project_library', args)
    
    def trainDatabaseStep(self):
        expectedSize = int(2e6) # TODO determine form gallery
        trainingSize = int(4e6) # TODO idem

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
        
        self.runJob('xmipp_train_database', args, numberOfMpi=1, env=self.getCondaEnv())
    
    def alignStep(self):
        nRotations = round(360 / self.angularSampling)
        nShift = self.shiftCount
        
        args = []
        args += ['-i', self._getWienerParticleMdFilename()]
        args += ['-o', self._getAlignmentMdFilename()]
        args += ['--index', self._getTrainingIndexFilename()]
        #args += ['--weights', self._getWeightsFilename()]
        args += ['--max_shift', self._getMaxShift()]
        args += ['--rotations', nRotations]
        args += ['--shifts', nShift]
        args += ['--max_frequency', self._getDigitalFrequencyLimit()]
        args += ['--method', 'fourier']
        args += ['--dropna']
        if self.useGpu:
            args += ['--gpu', 0] # TODO select
        
        self.runJob('xmipp_query_database', args, numberOfMpi=1, env=self.getCondaEnv())
    
    def splitStep(self):
        args = []
        args += ['-i', self._getAlignmentMdFilename()]
        args += ['-n', 2]
        
        self.runJob('xmipp_metadata_split', args, numberOfMpi=1)
    
    def reconstructStep(self, i: int):
        args []
        args += ['-i', self._getAlignmentHalfMdFilename(i)]
        args += ['-o', self._getHalfVolumeFilename(i)]
        args += ['--sym', self.symmetryGroup.get()]
        args += ['--weight']
    
        self.runJob("xmipp_reconstruct_fourier_accel", args)
    
    def createOutputStep(self):
        pass
    
    #--------------------------- UTILS functions --------------------------------------------        
    def _getDigitalFrequencyLimit(self):
        return self.inputParticles.get().getSamplingRate() / float(self.resolutionLimit)
    
    def _getMaxShift(self):
        return float(self.maxShift) / 100.0
    
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