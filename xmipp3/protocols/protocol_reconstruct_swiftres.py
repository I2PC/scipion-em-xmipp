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
from pwem.objects import (Volume, FSC, SetOfVolumes, Class3D, 
                          SetOfParticles, SetOfClasses3D, Particle,
                          Pointer )
from pwem.convert import transformations
from pwem import emlib

from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        MultiPointerParam,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )
from pyworkflow.utils.path import (cleanPath, makePath, copyFile, moveFile,
                                   createLink, cleanPattern)

import xmipp3
from xmipp3.convert import writeSetOfParticles, rowToParticle

from typing import Iterable, Sequence, Optional
import math

import numpy as np
from scipy import stats
import itertools
import os

class XmippProtReconstructSwiftres(ProtRefine3D, xmipp3.XmippProtocol):
    OUTPUT_CLASSES_NAME = 'classes'
    OUTPUT_VOLUMES_NAME = 'volumes'
    
    _label = 'swiftres'
    _conda_env = 'xmipp_swiftalign'
    _possibleOutputs = {
        OUTPUT_CLASSES_NAME: SetOfClasses3D,
        OUTPUT_VOLUMES_NAME: SetOfVolumes
    }
        
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
        form.addParam('initialResolution', FloatParam, label="Initial resolution (A)", default=10.0,
                      help='Image comparison resolution limit at the first iteration of the refinement')
        form.addParam('mask', PointerParam, label="Mask", pointerClass='VolumeMask', allowsNull=True,
                      help='The mask values must be between 0 (remove these pixels) and 1 (let them pass). Smooth masks are recommended.')
        
        form.addSection(label='CTF')
        form.addParam('considerInputCtf', BooleanParam, label='Consider CTF',
                      default=True,
                      help='Consider the CTF of the particles')

        form.addSection(label='Global refinement')
        form.addParam('reconstructLast', BooleanParam, label='Reconstruct last volume', default=True)
        form.addParam('numberOfIterations', IntParam, label='Number of iterations', default=3)
        form.addParam('numberOfLocalIterations', IntParam, label='Number of local iterations', default=1)
        form.addParam('numberOfAlignmentRepetitions', IntParam, label='Number of repetitions', default=2)
        form.addParam('maximumResolution', FloatParam, label="Maximum alignment resolution (A)", default=8.0,
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
        form.addParam('useAutomaticStep', BooleanParam, label='Use automatic step', default=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Automatically determine the step used when exploring the projection landscape')
        stepGroup = form.addGroup('Steps', condition='not useAutomaticStep', expertLevel=LEVEL_ADVANCED)
        stepGroup.addParam('angleStep', FloatParam, label='Angle step (deg)', default=5.0)
        stepGroup.addParam('shiftStep', FloatParam, label='Shift step (px)', default=2.0)
        form.addParam('reconstructPercentage', FloatParam, label='Reconstruct percentage (%)', default=50,
                      help='Percentage of best particles used for reconstruction')
        form.addParam('numberOfMatches', IntParam, label='Number of matches', default=1,
                      help='Number of reference matches for each particle')

        form.addSection(label='Compute')
        form.addParam('databaseRecipe', StringParam, label='Database recipe', 
                      default='OPQ48_192,IVF32768,PQ48',
                      help='FAISS database structure. Please refer to '
                      'https://github.com/facebookresearch/faiss/wiki/The-index-factory')
        form.addParam('useFloat16', BooleanParam, label='Use float16', default=False, 
                      help='When enabled, FAISS will be prompted to use half precision floating point '
                      'numbers. This may improve performance and/or memory footprint at some '
                      'accuracy cost. Only supported for GPUs')
        form.addParam('usePrecomputed', BooleanParam, label='Precompute centroid distances', default=False, 
                      help='When using PQ encoding precompute pairwise distances between centroids')
        form.addParam('databaseTrainingSetSize', IntParam, label='Database training set size', 
                      default=int(2e6),
                      help='Number of data-augmented particles to used when training the database')
        form.addParam('databaseMaximumSize', IntParam, label='Database size limit', 
                      default=int(10e6),
                      help='Maximum number of elements that can be stored in the database '
                      'before performing an alignment and flush')
        form.addParam('batchSize', IntParam, label='Batch size', 
                      default=8192,
                      help='It is recommended to use powers of 2. Using numbers around 8192 works well')
        form.addParam('copyParticles', BooleanParam, label='Copy particles to scratch', default=False, 
                      help='Copy input particles to scratch directory. Note that if input file format is '
                      'incompatible the particles will be converted into scratch anyway')

        form.addParallelSection(threads=1, mpi=8)
    
    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        particles = self.inputParticles.get()

        if self.considerInputCtf and not particles.hasCTF():
            errors.append('Input must have CTF information for being able to consider it')

        return errors
    
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):
        convertInputStepId = self._insertFunctionStep('convertInputStep', prerequisites=[])
        
        if self.considerInputCtf and self.reconstructLast:
            correctCtfStepId = self._insertFunctionStep('correctCtfStep', prerequisites=[convertInputStepId])
            lastIds = [correctCtfStepId]
        else:
            lastIds = [convertInputStepId]
            
        for i in range(self._getIterationCount()):
            lastIds = self._insertIterationSteps(i, prerequisites=lastIds)
        
        self._insertFunctionStep('createOutputStep', prerequisites=lastIds)
 
    def _insertIterationSteps(self, iteration: int, prerequisites):
        setupIterationStepId = self._insertFunctionStep('setupIterationStep', iteration, prerequisites=prerequisites)
        ctfGroupStepIds = []
        if self.considerInputCtf:
            ctfGroupStepIds.append(self._insertFunctionStep('ctfGroupStep', iteration, prerequisites=[setupIterationStepId]))
            
        projectIds = self._insertProjectSteps(iteration, prerequisites=[setupIterationStepId])
        alignIds = self._insertAlignmentSteps(iteration, prerequisites=projectIds + ctfGroupStepIds)
        compareAnglesStepId = self._insertFunctionStep('compareAnglesStep', iteration, prerequisites=alignIds)

        ids = []
        if self.reconstructLast or iteration < (self._getIterationCount() - 1):        
            for cls in range(self._getClassCount()):
                reconstructIds = self._insertReconstructSteps(iteration, cls, prerequisites=alignIds)
                postProcessIds = self._insertPostProcessSteps(iteration, cls, prerequisites=reconstructIds)
                ids += postProcessIds
        
        return ids + [compareAnglesStepId]
        
    def _insertProjectSteps(self, iteration: int, prerequisites):
        mergeStepIds = []
            
        for repetition in range(self._getAlignmentRepetitionCount()):
            projectStepIds = []

            # Project all classes
            for cls in range(self._getClassCount()):
                projectStepIds.append(self._insertFunctionStep('projectVolumeStep', iteration, cls, repetition, prerequisites=prerequisites))

            # Merge galleries of classes
            mergeStepIds.append(self._insertFunctionStep('mergeGalleriesStep', iteration, repetition, prerequisites=projectStepIds))
        
        #mergeStepIds.append(self._insertFunctionStep('computeNoiseModelStep', iteration, prerequisites=prerequisites))
        
        return mergeStepIds
        
    def _insertAlignmentSteps(self, iteration: int, prerequisites):
        ensembleTrainingSetStepId = self._insertFunctionStep('ensembleTrainingSetStep', iteration, prerequisites=prerequisites)
        trainDatabaseStepId = self._insertFunctionStep('trainDatabaseStep', iteration, prerequisites=[ensembleTrainingSetStepId])
        
        alignStepIds = []
        for repetition in range(self._getAlignmentRepetitionCount()):
            alignStepId = trainDatabaseStepId
            for local in range(self._getLocalIterationCount()):
                alignStepId = self._insertFunctionStep('alignStep', iteration, repetition, local, prerequisites=[alignStepId])
            alignStepIds.append(alignStepId)
            
        alignmentConsensusStepId = self._insertFunctionStep('alignmentConsensusStep', iteration, prerequisites=alignStepIds)
        intersectInputStepId = self._insertFunctionStep('intersectInputStep', iteration, prerequisites=[alignmentConsensusStepId])
        
        return [intersectInputStepId]
 
    def _insertReconstructSteps(self, iteration: int, cls: int, prerequisites):
        selectAlignmentStepId = self._insertFunctionStep('selectAlignmentStep', iteration, cls, prerequisites=prerequisites)
        #compareReprojectionStepId = self._insertFunctionStep('compareReprojectionStep', iteration, cls, prerequisites=[selectAlignmentStepId])
        #computeWeightsStepId = self._insertFunctionStep('computeWeightsStep', iteration, cls, prerequisites=[compareReprojectionStepId])
        #filterByWeightsStepId = self._insertFunctionStep('filterByWeightsStep', iteration, cls, prerequisites=[computeWeightsStepId])
        splitStepId = self._insertFunctionStep('splitStep', iteration, cls, prerequisites=[selectAlignmentStepId])
        #splitStepId = self._insertFunctionStep('splitStep', iteration, cls, prerequisites=[computeWeightsStepId])
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
        particles: SetOfParticles = self.inputParticles.get()
        
        writeSetOfParticles(particles, 
                            self._getInputParticleMdFilename())

        def is_mrc(path: str) -> bool:
            _, ext = os.path.splitext(path)
            return ext == '.mrc' or ext == '.mrcs'
        
        # Convert to MRC if necessary
        if self.copyParticles or not all(map(is_mrc, particles.getFiles())):
            args = []
            args += ['-i', self._getInputParticleMdFilename()]
            args += ['-o', self._getInputParticleStackFilename()]
            args += ['--save_metadata_stack', self._getInputParticleMdFilename()]
            args += ['--keep_input_columns']
            args += ['--track_origin']

            self.runJob('xmipp_image_convert', args, numberOfMpi=1)
    
    def correctCtfStep(self):
        particles: SetOfParticles = self.inputParticles.get()
        acquisition = particles.getAcquisition()
        
        # Perform a CTF correction using Wiener Filtering
        args = []
        args += ['-i', self._getInputParticleMdFilename()]
        args += ['-o', self._getWienerParticleMdFilename()]
        args += ['--pixel_size', self._getSamplingRate()]
        args += ['--spherical_aberration', acquisition.getSphericalAberration()]
        args += ['--voltage', acquisition.getVoltage()]
        if particles.isPhaseFlipped():
            args +=  ['--phase_flipped']

        args += ['--batch', self.batchSize]
        if self.useGpu:
            args += ['--device'] + self._getDeviceList()

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_swiftalign_wiener_2d', args, numberOfMpi=1, env=env)
        
        # Append the wiener corrected images to the second image label of the input
        inputMd = emlib.MetaData(self._getInputParticleMdFilename())
        wienerMd = emlib.MetaData(self._getWienerParticleMdFilename())
        wienerImageFns = wienerMd.getColumnValues(emlib.MDL_IMAGE)
        inputMd.setColumnValues(emlib.MDL_IMAGE1, wienerImageFns)
        inputMd.write(self._getInputParticleMdFilename())

    def setupIterationStep(self, iteration: int):
        makePath(self._getIterationPath(iteration))
        
        for cls in range(self._getClassCount()):
            makePath(self._getClassPath(iteration, cls))
            
        if iteration > 0:
            # Fill missing particles from the previous alignment with
            # The input particles. In order to use union, columns must
            # match, so join columns prior to doing the union
            args = []
            args += ['-i', self._getInputParticleMdFilename()]
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
                self._getInputParticleMdFilename(),
                self._getIterationInputParticleMdFilename(iteration)
            )
            
            resolution = float(self.initialResolution)
        
        imageSize = self._getImageSize()
        frequency = self._getSamplingRate() / resolution
        maxPsi = self._getIterationMaxPsi(iteration)
        maxShift = self._getIterationMaxShift(iteration)
        shiftStep = self._computeShiftStep(frequency) if self.useAutomaticStep else float(self.shiftStep)
        angleStep = self._computeAngleStep(frequency, imageSize) if self.useAutomaticStep else float(self.angleStep)
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
        args += ['-o', oroot + ':mrc']
        args += ['--sampling_rate', self._getSamplingRate()]
        args += ['--pad', 1]
        args += ['--error', 0.5] # TODO make param
        args += ['--resol', resolution]
        if particles.isPhaseFlipped():
            args +=  ['--phase_flipped']
        
        self.runJob('xmipp_ctf_group', args, numberOfMpi=1)

        # Convert group info
        groups = emlib.MetaData('groups@' + oroot + 'Info.xmd')
        representatives = [str(i+1) + '@' + self._getCtfGroupAveragesFilename(iteration) for i in range(groups.size())]
        groups.setColumnValues(emlib.MDL_IMAGE, representatives)
        groups.write(self._getCtfGroupInfoMdFilename(iteration))

        # Rename files and removed unused ones
        cleanPath(oroot + 'Info.xmd')
        cleanPath(oroot + '_split.doc')
        moveFile(oroot + '_images.sel', self._getCtfGroupMdFilename(iteration))
        moveFile(oroot + '_ctf.mrc', self._getCtfGroupAveragesFilename(iteration))
        
    def projectVolumeStep(self, iteration: int, cls: int, repetition: int):
        md = emlib.MetaData(self._getIterationParametersFilename(iteration))
        angleStep = md.getValue(emlib.MDL_ANGLE_DIFF, 1)
        perturb = math.sin(math.radians(angleStep)) / 4

        args = []
        args += ['-i', self._getIterationInputVolumeFilename(iteration, cls)]
        args += ['-o', self._getClassGalleryStackFilename(iteration, cls, repetition)]
        args += ['--sampling_rate', angleStep]
        args += ['--perturb', perturb]
        args += ['--sym', self.symmetryGroup]
        
        self.runJob('xmipp_angular_project_library', args)
    
        args = []
        args += ['-i', self._getClassGalleryMdFilename(iteration, cls, repetition)]
        args += ['--fill', 'ref3d', 'constant', cls+1]
        self._runMdUtils(args)
    
    def mergeGalleriesStep(self, iteration: int, repetition: int):
        self._mergeMetadata(
            map(lambda cls : self._getClassGalleryMdFilename(iteration, cls, repetition), range(self._getClassCount())),
            self._getGalleryMdFilename(iteration, repetition)
        )

        # Reindex
        args = []
        args += ['-i', self._getGalleryMdFilename(iteration, repetition)]
        args += ['--fill', 'ref', 'lineal', 1, 1]
        self._runMdUtils(args)
        
    def computeNoiseModelStep(self, iteration: int):
        args = []
        args += ['-i', self._getIterationInputParticleMdFilename(iteration)]
        args += ['-r', self._getIterationInputVolumeFilename(iteration, cls=0)] # TODO for multiple classes
        args += ['--oroot', self._getIterationPath(iteration, '')]
        args += ['--padding', 2.0]
        args += ['--max_resolution', 1.0] # TODO
        self.runJob('xmipp_reconstruct_noise_psd', args, numberOfMpi=1)
        
        args = []
        args += ['-i', self._getNoiseModelFilename(iteration)]
        args += ['-o', self._getWeightsFilename(iteration)]
        args += ['--pow', -1]
        self.runJob('xmipp_image_operate', args)
        
    def ensembleTrainingSetStep(self, iteration: int):
        self._mergeMetadata(
            map(lambda i : self._getGalleryMdFilename(iteration, i), range(self._getAlignmentRepetitionCount())),
            self._getTrainingMdFilename(iteration)
        )
        
    def trainDatabaseStep(self, iteration: int):
        trainingSize = int(self.databaseTrainingSetSize)
        recipe = self.databaseRecipe

        md = emlib.MetaData(self._getIterationParametersFilename(iteration))
        imageSize = self._getImageSize()
        maxFrequency = md.getValue(emlib.MDL_RESOLUTION_FREQREAL, 1)
        maxPsi = md.getValue(emlib.MDL_ANGLE_PSI, 1)
        maxShiftPx = md.getValue(emlib.MDL_SHIFT_X, 1)
        maxShift = maxShiftPx / imageSize
    
        args = []
        args += ['-i', self._getTrainingMdFilename(iteration)]
        args += ['-o', self._getTrainingIndexFilename(iteration)]
        args += ['--recipe', recipe]
        #args += ['--weights', self._getWeightsFilename(iteration)]
        args += ['--max_shift', maxShift] #FIXME fails with != 180
        args += ['--max_psi', maxPsi]
        args += ['--max_frequency', maxFrequency]
        args += ['--training', trainingSize]
        args += ['--batch', self.batchSize]
        args += ['--scratch', self._getTrainingScratchFilename()]
        if self.useGpu:
            args += ['--device'] + self._getDeviceList()
        if self.useFloat16:
            args += ['--fp16']
        if self.usePrecomputed:
            args += ['--use_precomputed']
        if self.considerInputCtf:
            args += ['--ctf', self._getCtfGroupInfoMdFilename(iteration)]
            
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_swiftalign_train', args, numberOfMpi=1, env=env)
    
    def alignStep(self, iteration: int, repetition: int, local: int):
        md = emlib.MetaData(self._getIterationParametersFilename(iteration))
        imageSize = self._getImageSize()
        maxFrequency = md.getValue(emlib.MDL_RESOLUTION_FREQREAL, 1)
        maxPsi = md.getValue(emlib.MDL_ANGLE_PSI, 1)
        maxShiftPx = md.getValue(emlib.MDL_SHIFT_X, 1)
        maxShift = maxShiftPx / imageSize
        nShift = round((2*maxShiftPx) / md.getValue(emlib.MDL_SHIFT_DIFF, 1)) + 1
        nRotations = round(360 / md.getValue(emlib.MDL_ANGLE_DIFF, 1))
        
        if local > 0:
            inputMdFilename = self._getAlignmentRepetitionMdFilename(iteration, repetition) 
        else: 
            inputMdFilename = self._getIterationInputParticleMdFilename(iteration)
        
        localFactor = math.pow(2, -local)
        maxPsi *= localFactor
        maxShift *= localFactor
    
        # Perform the alignment
        args = []
        args += ['-i', inputMdFilename]
        args += ['-o', self._getAlignmentRepetitionMdFilename(iteration, repetition)]
        args += ['-r', self._getGalleryMdFilename(iteration, repetition)]
        args += ['--index', self._getTrainingIndexFilename(iteration)]
        #args += ['--weights', self._getWeightsFilename(iteration)]
        args += ['--max_shift', maxShift]
        args += ['--max_psi', maxPsi]
        args += ['--rotations', nRotations]
        args += ['--shifts', nShift]
        args += ['--max_frequency', maxFrequency]
        args += ['--dropna']
        args += ['--batch', self.batchSize]
        args += ['--max_size', self.databaseMaximumSize]
        args += ['-k', self.numberOfMatches]
        args += ['--reference_labels', 'angleRot', 'angleTilt', 'ref3d', 'imageRef']
        if self.useGpu:
            args += ['--device'] + self._getDeviceList()
        if local > 0:
            args += ['--local']
        if self.usePrecomputed:
            args += ['--use_precomputed']
        if self.considerInputCtf:
            args += ['--ctf', self._getCtfGroupInfoMdFilename(iteration)]
        
        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it
        self.runJob('xmipp_swiftalign_query', args, numberOfMpi=1, env=env)

    def alignmentConsensusStep(self, iteration: int):
        paramMd = emlib.MetaData(self._getIterationParametersFilename(iteration))
        angleStep = paramMd.getValue(emlib.MDL_ANGLE_DIFF, 1)
        shiftStep = paramMd.getValue(emlib.MDL_SHIFT_DIFF, 1)

        # Ensure that both alignments have the same particles
        alignmentRepetitionMds = []
        for i in range(self._getAlignmentRepetitionCount()):
            filename = self._getAlignmentRepetitionMdFilename(iteration, i)
            md = emlib.MetaData(filename)
            alignmentRepetitionMds.append(md)

        self._keepCommonMetadataElements(
            alignmentRepetitionMds,
            emlib.MDL_ITEM_ID
        )
        
        for i, md in enumerate(alignmentRepetitionMds):
            filename = self._getAlignmentRepetitionMdFilename(iteration, i)
            md.write(filename)
            
        # Run angular distance to bring particles to the closest as possible
        alignmentConsensusMds = [emlib.MetaData(self._getAlignmentRepetitionMdFilename(iteration, 0))]
        for i in range(1, self._getAlignmentRepetitionCount()):
            self._runAngularDistance(
                self._getAlignmentRepetitionMdFilename(iteration, 0),
                self._getAlignmentRepetitionMdFilename(iteration, i),
                self._getAlignmentConsensusOutputRoot(iteration, i),
                extrargs=['--set', 2]
            )
            
            args = []
            args += ['-i', self._getAlignmentConsensusMdFilename(iteration, i)]
            args += ['--set', 'join', self._getAlignmentRepetitionMdFilename(iteration, i), 'itemId']
            self._runMdUtils(args)
            
            alignmentConsensusMds.append(emlib.MetaData(self._getAlignmentRepetitionMdFilename(iteration, i)))
            
        consensusMd = self._computeAlignmentConsensus(
            alignmentConsensusMds,
            angleStep,
            shiftStep,
            np.inf 
        )
        
        consensusMd.write(self._getAlignmentMdFilename(iteration))

    def intersectInputStep(self, iteration: int):
        args = []
        args += ['-i', self._getIterationInputParticleMdFilename(iteration)]
        args += ['-o', self._getInputIntersectionMdFilename(iteration)]
        args += ['--set', 'intersection', self._getAlignmentMdFilename(iteration), 'itemId']
        self._runMdUtils(args)
        
        # Add the missing columns
        args = []
        args += ['-i', self._getAlignmentMdFilename(iteration)]
        args += ['--set', 'intersection', self._getAlignmentMdFilename(iteration), 'itemId']
        self._runMdUtils(args)

    def compareAnglesStep(self, iteration: int):
        self._runAngularDistance(
            ang1=self._getAlignmentMdFilename(iteration),
            ang2=self._getInputIntersectionMdFilename(iteration),
            oroot=self._getAngleDiffOutputRoot(iteration)
        )
        
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
        if self.considerInputCtf:
            args = []
            args += ['-i', self._getReconstructionMdFilename(iteration, cls)]
            args += ['--operate', 'modify_values', 'image=image1']
            self._runMdUtils(args)

        args = []
        args += ['-i', self._getReconstructionMdFilename(iteration, cls)]
        args += ['-n', 2]
        
        self.runJob('xmipp_metadata_split', args, numberOfMpi=1)
    
    def reconstructStep(self, iteration: int, cls: int, half: int):
        args = []
        args += ['-i', self._getReconstructionHalfMdFilename(iteration, cls, half)]
        args += ['-o', self._getHalfVolumeFilename(iteration, cls, half)]
        args += ['--sym', self.symmetryGroup.get()]
    
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
        
        # Create output volumes if necessary
        if self.reconstructLast:
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
            self._createOutputFscs()
        else:
            # Create a dict mimicking the base-1 ids of SetOfVolumes
            volumes = dict(enumerate(map(Pointer.get, self.inputVolumes), start=1))
    
        # Link last iteration
        if os.path.exists(self._getInputParticleStackFilename()):
            # Use original images
            alignmentMd = emlib.MetaData(self._getAlignmentMdFilename(lastIteration))
            alignmentMd.copyColumn(emlib.MDL_IMAGE, emlib.MDL_IMAGE_ORIGINAL)
            alignmentMd.write(self._getOutputParticlesMdFilename())

        else:
            # Link
            createLink(
                self._getAlignmentMdFilename(lastIteration),
                self._getOutputParticlesMdFilename()
            )
        
        # Create output particles
        self._createOutputClasses3D(volumes)
        

    #--------------------------- UTILS functions --------------------------------------------        
    def _getIterationCount(self) -> int:
        return int(self.numberOfIterations)

    def _getCompletedIterationCount(self) -> int:
        return self._getIterationCount() # TODO
    
    def _getAlignmentRepetitionCount(self) -> int:
        return int(self.numberOfAlignmentRepetitions)
        
    def _getLocalIterationCount(self) -> int:
        return int(self.numberOfLocalIterations)
        
    def _getClassCount(self) -> int:
        return len(self.inputVolumes)
    
    def _getReconstructPercentile(self) -> float:
        return 1.0 - (float(self.reconstructPercentage) / 100.0)
    
    def _getSamplingRate(self) -> float:
        return float(self.inputParticles.get().getSamplingRate())
    
    def _getImageSize(self) -> int:
        return int(self.inputParticles.get().getXDim())

    def _getIterationPath(self, iteration: int, *paths):
        return self._getExtraPath('iteration_%04d' % iteration, *paths)
    
    def _getClassPath(self, iteration: int, cls: int, *paths):
        return self._getIterationPath(iteration, 'class_%06d' % cls, *paths)
    
    def _getInputParticleMdFilename(self):
        return self._getExtraPath('input_particles.xmd')
    
    def _getInputParticleStackFilename(self):
        return self._getTmpPath('input_particles.mrcs')
    
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
    
    def _getCtfGroupInfoMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'ctf_group_info.xmd')

    def _getCtfGroupMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'ctf_groups.xmd')
    
    def _getCtfGroupAveragesFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'ctf_group_representatives.mrcs')

    def _getClassGalleryMdFilename(self, iteration: int, cls: int, repetition: int):
        return self._getClassPath(iteration, cls, 'gallery%05d.doc' % repetition)
    
    def _getClassGalleryStackFilename(self, iteration: int, cls: int, repetition: int):
        return self._getClassPath(iteration, cls, 'gallery%05d.mrcs' % repetition)
    
    def _getGalleryMdFilename(self, iteration: int, repetition: int):
        return self._getIterationPath(iteration, 'gallery%05d.xmd' % repetition)

    def _getTrainingMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'training.xmd')
    
    def _getNoiseModelFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'avgNoisePsd.mrc')
    
    def _getWeightsFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'weights.mrc')
    
    def _getTrainingIndexFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'database.idx')

    def _getAlignmentRepetitionMdFilename(self, iteration: int, repetition: int):
        return self._getIterationPath(iteration, 'aligned%05d.xmd' % repetition)
    
    def _getAlignmentMdFilename(self, iteration: int):
        return self._getIterationPath(iteration, 'aligned.xmd')
    
    def _getAlignmentConsensusOutputRoot(self, iteration: int, repetition: int):
        return self._getIterationPath(iteration, 'alignment_consensus%05d' % repetition)

    def _getAlignmentConsensusMdFilename(self, iteration: int, repetition: int):
        return self._getAlignmentConsensusOutputRoot(iteration, repetition) + '.xmd'

    def _getAlignmentConsensusVecDiffHistogramMdFilename(self, iteration: int, repetition: int):
        return self._getAlignmentConsensusOutputRoot(iteration, repetition) + '_vec_diff_hist.xmd'

    def _getAlignmentConsensusShiftDiffHistogramMdFilename(self, iteration: int, repetition: int):
        return self._getAlignmentConsensusOutputRoot(iteration, repetition) + '_shift_diff_hist.xmd'
    
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
    
    def _getDeviceList(self):
        gpus = self.getGpuList()
        return list(map('cuda:{:d}'.format, gpus))
    
    def _quaternionAverage(self, quaternions: np.ndarray) -> np.ndarray:
        s = np.matmul(quaternions.T, quaternions)
        s /= len(quaternions)
        eigenValues, eigenVectors = np.linalg.eig(s)
        return np.real(eigenVectors[:,np.argmax(eigenValues)])
    
    def _quaternionDistance(self,
                            quaternions: np.ndarray,
                            quaternion: np.ndarray ) -> np.ndarray:
        dots = np.dot(quaternions, quaternion)
        return np.arccos(2*(dots**2)-1)
    
    def _computeAlignmentConsensus( self, 
                                    mds: Sequence[emlib.MetaData],
                                    maxAngleDiff: float,
                                    maxShiftDiff: float,
                                    maxCost: float ) -> emlib.MetaData:
        if len(mds) > 1:
            resultMd = emlib.MetaData()
            
            maxAngleDiff = np.deg2rad(maxAngleDiff)
            
            quaternions = np.empty((len(mds), 4))
            shifts = np.empty((len(mds), 2))
            ref3ds = np.empty(len(mds), dtype=int)
            costs = np.empty(len(mds))
            row = emlib.metadata.Row()
            
            for objIds in zip(*mds):
                # Obtain the alignment data
                for i, (objId, md) in enumerate(zip(objIds, mds)):
                    quaternions[i,:] = transformations.quaternion_from_euler(
                        -np.deg2rad(md.getValue(emlib.MDL_ANGLE_ROT, objId)),
                        -np.deg2rad(md.getValue(emlib.MDL_ANGLE_TILT, objId)),
                        -np.deg2rad(md.getValue(emlib.MDL_ANGLE_PSI, objId)),
                        axes='szyz'
                    )
                    shifts[i,:] = ( md.getValue(emlib.MDL_SHIFT_X, objId),
                                    md.getValue(emlib.MDL_SHIFT_Y, objId) )
                    ref3ds[i] = md.getValue(emlib.MDL_REF3D, objId)
                    costs[i] = md.getValue(emlib.MDL_COST, objId)
                    
                # Check if more than a half agree
                quaternion = self._quaternionAverage(quaternions)
                angleDiff = self._quaternionDistance(quaternions, quaternion)
                if np.count_nonzero(angleDiff <= maxAngleDiff) <= len(angleDiff) / 2:
                    continue

                shift = np.mean(shifts, axis=0)
                shiftDiff = np.linalg.norm(shifts-shift, axis=1)
                if np.count_nonzero(shiftDiff <= maxShiftDiff) <= len(shiftDiff) / 2:
                    continue
                
                ref3d = int(stats.mode(ref3ds).mode)
                if np.count_nonzero(ref3ds == ref3d) <= len(ref3ds) / 2:
                    continue
                
                cost = np.median(costs)
                if cost > maxCost:
                    continue
                
                # All checks succeeded. Elaborate output
                row.readFromMd(mds[0], objIds[0])
                
                rot, tilt, psi = transformations.euler_from_quaternion(quaternion, axes='szyz')
                row.setValue(emlib.MDL_ANGLE_ROT, -np.rad2deg(rot))
                row.setValue(emlib.MDL_ANGLE_TILT, -np.rad2deg(tilt))
                row.setValue(emlib.MDL_ANGLE_PSI, -np.rad2deg(psi))
                row.setValue(emlib.MDL_SHIFT_X, shift[0])
                row.setValue(emlib.MDL_SHIFT_Y, shift[1])
                row.setValue(emlib.MDL_ANGLE_DIFF, np.rad2deg(np.mean(angleDiff)))
                row.setValue(emlib.MDL_SHIFT_DIFF, np.mean(shiftDiff))
                row.setValue(emlib.MDL_REF3D, ref3d)
                row.setValue(emlib.MDL_COST, cost)
                
                assert(len(mds) > 1)
                row.setValue(emlib.MDL_ANGLE_PSI+1, mds[0].getValue(emlib.MDL_ANGLE_PSI, objIds[0]))
                row.setValue(emlib.MDL_ANGLE_PSI+2, mds[1].getValue(emlib.MDL_ANGLE_PSI, objIds[1]))
                row.setValue(emlib.MDL_ANGLE_ROT+1, mds[0].getValue(emlib.MDL_ANGLE_ROT, objIds[0]))
                row.setValue(emlib.MDL_ANGLE_ROT+2, mds[1].getValue(emlib.MDL_ANGLE_ROT, objIds[1]))
                row.setValue(emlib.MDL_ANGLE_TILT+1, mds[0].getValue(emlib.MDL_ANGLE_TILT, objIds[0]))
                row.setValue(emlib.MDL_ANGLE_TILT+2, mds[1].getValue(emlib.MDL_ANGLE_TILT, objIds[1]))
                row.setValue(emlib.MDL_SHIFT_X+1, mds[0].getValue(emlib.MDL_SHIFT_X, objIds[0]))
                row.setValue(emlib.MDL_SHIFT_X+2, mds[1].getValue(emlib.MDL_SHIFT_X, objIds[1]))
                row.setValue(emlib.MDL_SHIFT_Y+1, mds[0].getValue(emlib.MDL_SHIFT_Y, objIds[0]))
                row.setValue(emlib.MDL_SHIFT_Y+2, mds[1].getValue(emlib.MDL_SHIFT_Y, objIds[1]))
                
                row.writeToMd(resultMd, resultMd.addObject())

            return resultMd
        
        elif len(mds) == 1:
            return mds[0]
            
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
        s = 1.0 / (maxFrequency*size)
        angle = math.asin(s)
        
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
        EXTRA_LABELS = [
            #emlib.MDL_COST,
            #emlib.MDL_WEIGHT,
            #emlib.MDL_CORRELATION_IDX,
            #emlib.MDL_CORRELATION_MASK,
            #emlib.MDL_CORRELATION_WEIGHT,
            #emlib.MDL_IMED
        ]
        
        def updateItem(item: Particle, row: emlib.metadata.Row):
            if row is not None:
                particle: Particle = rowToParticle(row, extraLabels=EXTRA_LABELS)
                item.copy(particle)
            else:
                item._appendItem = False
                
        def updateClass(cls: Class3D):
            clsId = cls.getObjId()
            representative = volumes[clsId]
            cls.setRepresentative(representative)
            
        particlesMd = emlib.MetaData(self._getOutputParticlesMdFilename())
        classes3d = self._createSetOfClasses3D(self.inputParticles)
        classes3d.classifyItems(
            updateItemCallback=updateItem,
            updateClassCallback=updateClass,
            itemDataIterator=itertools.chain(emlib.metadata.iterRows(particlesMd), itertools.repeat(None))
        )
        
        # Set the alignment for the output particle set
        outputParticles: SetOfParticles = classes3d.getImages()
        outputParticles.setAlignmentProj()
        
        # Define the output
        self._defineOutputs(**{self.OUTPUT_CLASSES_NAME: classes3d})
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
        
    def _keepCommonMetadataElements(self, 
                                    mds: Sequence[emlib.MetaData],
                                    label: int = emlib.MDL_ITEM_ID ):
        # Intersect the first md with the rest of mds
        commonMd = mds[0] # Use first item to keep track of common items
        for md in mds[1:]:
            commonMd.intersection(md, label)
            
        # Intersect all of the mds with the first one
        for md in mds[1:]:
            md.intersection(commonMd, label)
            
    def _runMdUtils(self, args):
        self.runJob('xmipp_metadata_utilities', args, numberOfMpi=1)

    def _runAngularDistance(self, 
                            ang1: str, 
                            ang2: str, 
                            oroot: str, 
                            sym: Optional[str] = None,
                            extrargs = []):
        args = []
        args += ['--ang1', ang1]
        args += ['--ang2', ang2]
        args += ['--oroot', oroot]
        args += ['--sym', sym or self.symmetryGroup]
        args += extrargs
        self.runJob('xmipp_angular_distance', args, numberOfMpi=1)
        
        # Rename files to have xmd extension
        moveFile(oroot+'_vec_diff_hist.txt', oroot+'_vec_diff_hist.xmd')
        moveFile(oroot+'_shift_diff_hist.txt', oroot+'_shift_diff_hist.xmd')