# **************************************************************************
# *
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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


from pwem import emlib
from pwem.protocols import ProtClassify3D
from pwem.objects import Particle, VolumeMask, SetOfParticles, Integer

from pyworkflow import BETA
from pyworkflow.utils import makePath, createLink, moveFile, cleanPath
from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        GE, GT, Range,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )

import xmipp3
from xmipp3.convert import writeSetOfParticles, setXmippAttributes
import xmippLib

import os.path
import pickle
import itertools
import collections
import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.mixture
import sklearn.model_selection

class XmippProtHetAnalysis(ProtClassify3D, xmipp3.XmippProtocol):
    OUTPUT_PARTICLES_NAME = 'Particles'
    
    _label = 'heterogeneity analysis'
    _devStatus = BETA
    _possibleOutputs = {
        OUTPUT_PARTICLES_NAME: SetOfParticles
    }

    def __init__(self, *args, **kwargs):
        ProtClassify3D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions ------------------------
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
                      pointerClass=SetOfParticles, pointerCondition='hasAlignmentProj',
                      help='Input particle set')
        form.addParam('inputMask', PointerParam, label='Mask', important=True,
                      pointerClass=VolumeMask, allowsNull=True,
                      help='Volume mask used for focussing the classification')
        form.addParam('symmetryGroup', StringParam, label='Symmetry group', default='c1')
        form.addParam('resize', IntParam, label='Resize', default=0,
                      validators=[GE(0)])

        form.addSection(label='Classification')
        form.addParam('principalComponents', IntParam, label='Principal components',
                      default=6, validators=[GT(0)],
                      help='Number of principal components used for directional classification')
        form.addParam('secondaryPrincipalComponents', IntParam, label='Secondary principal components',
                      default=96, validators=[GT(0)],
                      help='Number of principal components used in decomposition.')
        form.addParam('outputPrincipalComponents', IntParam, label='Output principal components',
                      default=4, validators=[GT(0)],
                      help='Number of principal components represented in the output.')

        form.addSection(label='Analysis')
        form.addParam('maxClasses', IntParam, label='Maximum number of classes',
                      default=16, validators=[GT(0)],
                      help='Maximum number of classes considered in the analysis step')
        
        form.addSection(label='CTF')
        form.addParam('considerInputCtf', BooleanParam, label='Consider CTF',
                      default=True,
                      help='Consider the CTF of the particles')

        form.addSection(label='Angular sampling')
        form.addParam('angularSampling', FloatParam, label='Angular sampling',
                      default=7.5, validators=[Range(0,180)],
                      help='Angular sampling interval in degrees')
        form.addParam('angularDistance', FloatParam, label='Angular distance',
                      default=7.5, validators=[Range(0,180)],
                      help='Maximum angular distance in degrees')
        form.addParam('checkMirrors', BooleanParam, label='Check mirrors',
                      default=True)
        
        form.addParallelSection(threads=0, mpi=8)

        form.addSection(label='Compute')
        form.addParam('batchSize', IntParam, label='Batch size', 
                      default=1024,
                      help='It is recommended to use powers of 2. Using numbers around 1024 works well')
        form.addParam('copyParticles', BooleanParam, label='Copy particles to scratch', default=False, 
                      help='Copy input particles to scratch directory. Note that if input file format is '
                      'incompatible the particles will be converted into scratch anyway')
        
    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        
        if self.considerInputCtf:
            self._insertFunctionStep('correctCtfStep')

        self._insertFunctionStep('projectMaskStep')
        self._insertFunctionStep('angularNeighborhoodStep')
        self._insertFunctionStep('classifyDirectionsStep')
        self._insertFunctionStep('buildGraphStep')
        self._insertFunctionStep('basisSynchronizationStep')
        self._insertFunctionStep('correctBasisStep')
        self._insertFunctionStep('combineDirectionsStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep('analysisStep')

    def _summary(self):
        summary = []
        
        recommendedNumberOfClasses = getattr(self, 'recommendedNumberOfClasses', None)
        if recommendedNumberOfClasses is not None:
            summary.append("Recommended number of classes: %d." % recommendedNumberOfClasses)
        else:
            summary.append("Recommended number of classes not ready")

        return summary

    # --------------------------- STEPS functions -------------------------------
    def convertInputStep(self):
        inputParticles: SetOfParticles = self.inputParticles.get()
        inputMask: VolumeMask = self.inputMask.get()
        
        writeSetOfParticles(inputParticles, 
                            self._getInputParticleMdFilename())

        def is_mrc(path: str) -> bool:
            _, ext = os.path.splitext(path)
            return ext == '.mrc' or ext == '.mrcs'
        
        # Convert particles to MRC and resize if necessary
        if self.copyParticles or self.resize > 0 or not all(map(is_mrc, inputParticles.getFiles())):
            args = []
            args += ['-i', self._getInputParticleMdFilename()]
            args += ['-o', self._getInputParticleStackFilename()]
            args += ['--save_metadata_stack', self._getInputParticleMdFilename()]
            args += ['--keep_input_columns']
            args += ['--track_origin']
            
            if self.resize > 0:
                args += ['--fourier', self.resize]
                self.runJob('xmipp_image_resize', args)
            else:
                self.runJob('xmipp_image_convert', args, numberOfMpi=1)
    
        # Convert or create mask
        if inputMask is None:
            # Create a spherical mask
            dim: int = self.resize.get() if self.resize > 0 else inputParticles.getXDim()
            emlib.createEmptyFile(self._getInputMaskFilename(), dim, dim, dim)

            args = []
            args += ['-i', self._getInputMaskFilename()]
            args += ['--create_mask', self._getInputMaskFilename()]
            args += ['--mask', 'circular', -dim/2]
            self.runJob('xmipp_transform_mask', args, numberOfMpi=1)
            
        else:
            # Convert particles to MRC and resize if necessary
            if self.resize > 0 or not is_mrc(inputMask.getFileName()):
                args = []
                args += ['-i', inputMask.getFileName()]
                args += ['-o', self._getInputMaskFilename()]
                
                if self.resize > 0:
                    args += ['--fourier', self.resize]
                    self.runJob('xmipp_image_resize', args, numberOfMpi=1)
                else:
                    self.runJob('xmipp_image_convert', args, numberOfMpi=1)
                    
            else:
                createLink(inputMask.getFileName(), self._getInputMaskFilename())
    
    def correctCtfStep(self):
        particles: SetOfParticles = self.inputParticles.get()
        
        # Perform a CTF correction using Wiener Filtering
        args = []
        args += ['-i', self._getInputParticleMdFilename()]
        args += ['-o', self._getWienerParticleMdFilename()]
        args += ['--sampling_rate', self._getSamplingRate()]
        if particles.isPhaseFlipped():
            args +=  ['--phase_flipped']

        self.runJob('xmipp_ctf_correct_wiener2d', args)

    def projectMaskStep(self):
        args = []
        args += ['-i', self._getInputMaskFilename()]
        args += ['-o', self._getMaskGalleryStackFilename()]
        args += ['--sampling_rate', self._getAngularSampling()]
        args += ['--sym', self._getSymmetryGroup()]
        #args += ['--method', 'fourier', 1, 1, 'bspline'] 
        args += ['--method', 'real_space'] 
        if self.checkMirrors:
            args += ['--max_tilt_angle', 90.01] #HACK: .01 to include 90

        # Create a gallery of projections of the input volume
        # with the given angular sampling
        self.runJob("xmipp_angular_project_library", args)
        
        # Binarize the mask
        args = []
        args += ['-i', self._getMaskGalleryMdFilename()]
        args += ['--ge', 1]
        self.runJob("xmipp_image_operate", args)
        
    def angularNeighborhoodStep(self):
        args = []
        args += ['--i1', self._getWienerParticleMdFilename()]
        args += ['--i2', self._getMaskGalleryMdFilename()]
        args += ['-o', self._getNeighborsMdFilename()]
        args += ['--dist', self._getAngularDistance()]
        args += ['--sym', self._getSymmetryGroup()]
        if self.checkMirrors:
            args += ['--check_mirrors']

        # Compute several groups of the experimental images into
        # different angular neighbourhoods
        self.runJob("xmipp_angular_neighbourhood", args, numberOfMpi=1)
        
        # Create directory structure
        maskGalleryMd = emlib.MetaData(self._getMaskGalleryMdFilename())
        maskRow = emlib.metadata.Row()
        directionalMd = emlib.MetaData()
        directionRow = emlib.metadata.Row()
        particles = emlib.MetaData()
        itemIds = set()
        for block in emlib.getBlocksInMetaDataFile(self._getNeighborsMdFilename()):
            directionId = int(block.split("_")[1])

            particles.readBlock(self._getNeighborsMdFilename(), block)
            if particles.size() == 0:
                print('Direction %d has no particles. Skipping' % directionId)
                continue

            # Write particles
            makePath(self._getDirectionPath(directionId))
            particles.write(self._getDirectionParticlesMdFilename(directionId))
            
            # Update particle ids
            itemIds.update(particles.getColumnValues(emlib.MDL_ITEM_ID))
            
            # Read information from the mask metadata
            maskRow.readFromMd(maskGalleryMd, directionId)
            maskFilename = maskRow.getValue(emlib.MDL_IMAGE)

            # Ensemble the output row
            directionRow.copyFromRow(maskRow)
            directionRow.setValue(emlib.MDL_MASK, maskFilename)
            directionRow.setValue(emlib.MDL_IMAGE, self._getDirectionalAverageImageFilename(directionId))
            directionRow.setValue(emlib.MDL_IMAGE_RESIDUAL, self._getDirectionalEigenImagesFilename(directionId))
            directionRow.setValue(emlib.MDL_SELFILE, self._getDirectionalClassificationMdFilename(directionId))
            directionRow.setValue(emlib.MDL_COUNT, particles.size())
            directionRow.addToMd(directionalMd)

        if len(itemIds) != len(self._getInputParticles()):
            raise RuntimeError('Not all particles were grouped. Please try '
                               'increasing "Angular distance" or decreasing '
                               '"Angular sampling"')

        directionalMd.write(self._getDirectionalMdFilename())
            
    def classifyDirectionsStep(self):
        k = self._getPrincipalComponentsCount()
        
        env = self.getCondaEnv(_conda_env='xmipp_pyTorch')
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it

        directionalMd = emlib.MetaData(self._getDirectionalMdFilename())
        for directionId in directionalMd:
            # Read information from the mask metadata
            maskFilename = directionalMd.getValue(emlib.MDL_MASK, directionId)
            rot = directionalMd.getValue(emlib.MDL_ANGLE_ROT, directionId)
            tilt = directionalMd.getValue(emlib.MDL_ANGLE_TILT, directionId)
            psi = directionalMd.getValue(emlib.MDL_ANGLE_PSI, directionId)
                
            # Perform the classification
            args = []
            args += ['-i', self._getDirectionParticlesMdFilename(directionId)]
            args += ['-o', self._getDirectionPath(directionId, '')]
            args += ['--mask', maskFilename]
            args += ['--align_to', rot, tilt, psi]
            args += ['-k', k]
            args += ['--batch', self.batchSize]
            if self.useGpu:
                args += ['--device'] + self._getDeviceList()

            self.runJob('xmipp_swiftalign_aligned_2d_classification', args, numberOfMpi=1, env=env)
  
    def buildGraphStep(self):
        # Build graph intersecting direction pairs and calculate the similarity
        # between projection values
        k = self._getPrincipalComponentsCount()
        symList = xmippLib.SymList()
        symList.readSymmetryFile(self._getSymmetryGroup())
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        md0 = emlib.MetaData()
        md1 = emlib.MetaData()
        objIds = list(directionMd)
        nDirections = len(objIds)
        crossCorrelations = scipy.sparse.lil_matrix((nDirections*k, )*2)
        weights = scipy.sparse.lil_matrix(crossCorrelations.shape)
        for (idx0, directionId0), (idx1, directionId1) in itertools.combinations(enumerate(objIds), r=2):
            # Obtain the projection angles
            rot0 = directionMd.getValue(emlib.MDL_ANGLE_ROT, directionId0)
            rot1 = directionMd.getValue(emlib.MDL_ANGLE_ROT, directionId1)
            tilt0 = directionMd.getValue(emlib.MDL_ANGLE_TILT, directionId0)
            tilt1 = directionMd.getValue(emlib.MDL_ANGLE_TILT, directionId1)
            
            # Only consider direction pairs that are close
            angularDistance = symList.computeDistanceAngles(
                rot0, tilt0, 0.0, 
                rot1, tilt1, 0.0,
                True, self.checkMirrors.get(), False
            )
            if angularDistance <= 2*self.angularDistance.get():
                # Obtain the intersection of the particles belonging to
                # both directions
                md0.read(directionMd.getValue(emlib.MDL_SELFILE, directionId0))
                md1.read(directionMd.getValue(emlib.MDL_SELFILE, directionId1))
                
                md0.intersection(md1, emlib.MDL_ITEM_ID)
                if md0.size() > 0:
                    md1.intersection(md0, emlib.MDL_ITEM_ID)

                    # Get their likelihood values
                    projections0 = np.array(md0.getColumnValues(emlib.MDL_DIMRED)).T
                    projections1 = np.array(md1.getColumnValues(emlib.MDL_DIMRED)).T

                    # Normalize projections
                    norm0 = np.linalg.norm(projections0, axis=-1, keepdims=True)
                    norm1 = np.linalg.norm(projections1, axis=-1, keepdims=True)
                    projections0 /= norm0
                    projections1 /= norm1
                    
                    # Compute the cross correlation and weight matrix
                    crossCorrelation = projections0 @ projections1.T
                    weight = norm0 @ norm1.T

                    # Write them in symmetric positions
                    start0 = idx0*k
                    end0 = start0+k
                    start1 = idx1*k
                    end1 = start1+k
                    crossCorrelations[start0:end0, start1:end1] = crossCorrelation
                    crossCorrelations[start1:end1, start0:end0] = crossCorrelation.T
                    weights[start0:end0, start1:end1] = weight
                    weights[start1:end1, start0:end0] = weight.T
                    
        # Normalize weights
        weights /= abs(weights).max()
        
        # Store the matrices
        self._writeCrossCorrelations(crossCorrelations.tocsr())
        self._writeWeights(weights.tocsr())

    def basisSynchronizationStep(self):
        k = self._getPrincipalComponentsCount()
        k2 = self._getSecondaryPrincipalComponentsCount()
        
        cmd = 'xmipp_synchronize_transform'
        args = []
        args += ['-i', self._getCrossCorrelationsFilename()]
        args += ['-w', self._getWeightsFilename()]
        args += ['-o', self._getBasesFilename()]
        args += ['-k', k]
        args += ['-k2', k2]
        args += ['-p', self._getPairwiseFilename()]
        args += ['--triangular_upper']
        args += ['-v']

        env = self.getCondaEnv(_conda_env='xmipp_graph')
        self.runJob(cmd, args, env=env, numberOfMpi=1)
   
    def correctBasisStep(self):
        ko = self._getOutputPrincipalComponentsCount()
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        classificationMd = emlib.MetaData()
        bases = self._readBases()
        inverseBases = np.linalg.pinv(bases)

        for i, directionId in enumerate(directionMd):
            classificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId))
            
            projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
            correctedProjections = projections @ inverseBases[i,-ko:].T
            
            classificationMd.setColumnValues(emlib.MDL_DIMRED, correctedProjections.tolist())
            classificationMd.write(self._getCorrectedDirectionalClassificationMdFilename(directionId))
   
    def combineDirectionsStep(self):   
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        directionalClassificationMd = emlib.MetaData()

        # Accumulate all PCA projection values
        projections = collections.Counter()
        sigmas = collections.Counter()
        for directionId in directionMd:
            # Read the classification of this direction
            #directionalClassificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId)) # TODO
            directionalClassificationMd.read(self._getCorrectedDirectionalClassificationMdFilename(directionId))
            
            values = np.array(directionalClassificationMd.getColumnValues(emlib.MDL_DIMRED))
            sigma = np.std(values, axis=0)
            
            # Increment the result likelihood value
            for objId in directionalClassificationMd:
                itemId = directionalClassificationMd.getValue(emlib.MDL_ITEM_ID, objId)
                projection = np.array(directionalClassificationMd.getValue(emlib.MDL_DIMRED, objId))
                
                projections[itemId] += projection
                sigmas[itemId] += sigma

        # Write PCA projection values to the output metadata
        result = emlib.MetaData(self._getWienerParticleMdFilename())
        for objId in result:
            itemId = result.getValue(emlib.MDL_ITEM_ID, objId)
            projection = projections[itemId] / sigmas[itemId]
            result.setValue(emlib.MDL_DIMRED, projection.tolist(), objId)
                
        # Store
        result.write(self._getClassificationMdFilename())
        
    def createOutputStep(self):
        classification = emlib.MetaData(self._getClassificationMdFilename())
        
        def updateItem(item: Particle, row: emlib.metadata.Row):
            setXmippAttributes(item, row, emlib.MDL_DIMRED)
        
        particles: SetOfParticles = self._createSetOfParticles()
        particles.copyInfo(self._getInputParticles())
        particles.copyItems(
            self._getInputParticles(),
            updateItemCallback=updateItem,
            itemDataIterator=emlib.metadata.iterRows(classification, sortByLabel=emlib.MDL_ITEM_ID)
        )
        
        self._defineOutputs(**{self.OUTPUT_PARTICLES_NAME: particles})

    def analysisStep(self):
        classificationMd = emlib.MetaData(self._getClassificationMdFilename())
        projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
        maxClasses = self.maxClasses.get()

        search = sklearn.model_selection.GridSearchCV(
            sklearn.mixture.GaussianMixture(), 
            param_grid={ "n_components": range(1, maxClasses+1) }, 
            scoring=lambda estimator, x : -estimator.bic(x)
        )
        search.fit(projections)
    
        result = search.cv_results_
        index = np.argmax(result['mean_test_score'])
        nClasses = result['param_n_components'][index]

        self.recommendedNumberOfClasses = Integer(nClasses)
        self._store()
        self._writeGmmAnalysis(result)
        
    # --------------------------- UTILS functions -------------------------------
    def _getDeviceList(self):
        gpus = self.getGpuList()
        return list(map('cuda:{:d}'.format, gpus))
    
    def _getInputParticles(self) -> SetOfParticles:
        return self.inputParticles.get()

    def _getReferenceVolumeFilename(self) -> str:
        return self._getReferenceVolume().getFileName()
    
    def _getSamplingRate(self):
        return float(self.inputParticles.get().getSamplingRate())
    
    def _getSymmetryGroup(self):
        return self.symmetryGroup.get()
    
    def _getAngularSampling(self):
        return self.angularSampling.get()
    
    def _getAngularDistance(self):
        return self.angularDistance.get()
    
    def _getPrincipalComponentsCount(self) -> int:
        return self.principalComponents.get()
    
    def _getSecondaryPrincipalComponentsCount(self) -> int:
        return self.secondaryPrincipalComponents.get()
    
    def _getOutputPrincipalComponentsCount(self) -> int:
        return self.outputPrincipalComponents.get()

    def _getInputMaskFilename(self):
        return self._getTmpPath('mask.mrc')
    
    def _getInputParticleMdFilename(self):
        return self._getPath('input_particles.xmd')
    
    def _getInputParticleStackFilename(self):
        return self._getTmpPath('input_particles.mrc')

    def _getWienerParticleMdFilename(self):
        if self.considerInputCtf:
            return self._getExtraPath('particles_wiener.xmd')
        else:
            return self._getInputParticleMdFilename()

    def _getWienerParticleStackFilename(self):
        return self._getTmpPath('particles_wiener.mrc')

    def _getClassificationMdFilename(self):
        return self._getExtraPath('classification.xmd')

    def _getMaskGalleryMdFilename(self):
        return self._getExtraPath('mask_gallery.doc')

    def _getMaskGalleryAnglesMdFilename(self):
        return self._getExtraPath('mask_gallery_angles.doc')

    def _getMaskGalleryStackFilename(self):
        return self._getExtraPath('mask_gallery.mrcs')

    def _getNeighborsMdFilename(self):
        return self._getExtraPath('neighbors.xmd')
    
    def _getDirectionPath(self, direction_id: int, *paths):
        return self._getExtraPath('direction_%06d' % direction_id, *paths)
    
    def _getDirectionParticlesMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'particles.xmd')

    def _getDirectionalEigenImagesFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'eigen_images.mrc')
    
    def _getDirectionalAverageImageFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'average.mrc')

    def _getDirectionalClassificationMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'classification.xmd')

    def _getCorrectedDirectionalClassificationMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'corrected_classification.xmd')
    
    def _getDirectionalMdFilename(self):
        return self._getExtraPath('directional.xmd')

    def _getCrossCorrelationsFilename(self):
        return self._getExtraPath('cross_correlations.npz')

    def _getWeightsFilename(self):
        return self._getExtraPath('weights.npz')
    
    def _getBasesFilename(self):
        return self._getExtraPath('bases.npy')
    
    def _getPairwiseFilename(self):
        return self._getExtraPath('pairwise.npy')

    def _getGmmAnalysisFilename(self):
        return self._getExtraPath('analysis.pkl')

    def _writeCrossCorrelations(self, correlations: scipy.sparse.csr_matrix) -> str:
        path = self._getCrossCorrelationsFilename()
        scipy.sparse.save_npz(path, correlations)
        return path
    
    def _readCrossCorrelations(self) -> scipy.sparse.csr_matrix:
        return scipy.sparse.load_npz(self._getCrossCorrelationsFilename())
    
    def _writeWeights(self, weights: scipy.sparse.csr_matrix) -> str:
        path = self._getWeightsFilename()
        scipy.sparse.save_npz(path, weights)
        return path
    
    def _readWeights(self) -> scipy.sparse.csr_matrix:
        return scipy.sparse.load_npz(self._getWeightsFilename())

    def _readBases(self):
        return np.load(self._getBasesFilename())

    def _writeGmmAnalysis(self, analysis):
        with open(self._getGmmAnalysisFilename(), 'wb') as f:
            pickle.dump(analysis, f)
        
    def _readGmmAnalysis(self):
        with open(self._getGmmAnalysisFilename(), 'rb') as f:
            return pickle.load(f)
    