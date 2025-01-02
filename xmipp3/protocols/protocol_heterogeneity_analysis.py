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
from pwem.objects import (Particle, VolumeMask, SetOfParticles, 
                          Volume, SetOfVolumes, Integer)

from pyworkflow import BETA
from pyworkflow.utils import makePath, createLink, moveFile, cleanPath
from pyworkflow.protocol.params import (Form, PointerParam, EnumParam,
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        GE, GT, Range,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )
import sklearn.decomposition

import xmipp3
from xmipp3.convert import (writeSetOfParticles, setXmippAttributes, 
                            locationToXmipp )
import xmippLib

import os.path
import itertools
import collections
import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.decomposition

class XmippProtHetAnalysis(ProtClassify3D, xmipp3.XmippProtocol):
    OUTPUT_PARTICLES_NAME = 'Particles'
    OUTPUT_EIGENVOLUMES_NAME = 'Eigenvolumes'
    
    _label = 'heterogeneity analysis'
    _devStatus = BETA
    _possibleOutputs = {
        OUTPUT_PARTICLES_NAME: SetOfParticles,
        OUTPUT_EIGENVOLUMES_NAME: SetOfVolumes
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

        form.addSection(label='Analysis')
        form.addParam('principalComponents', IntParam, label='Principal components',
                      default=6, validators=[GT(0)],
                      help='Number of principal components used for directional classification')
        form.addParam('outputPrincipalComponents', IntParam, label='Output principal components',
                      default=0, validators=[GE(0)], expertLevel=LEVEL_ADVANCED,
                      help='Number of principal components represented in the output.')
        form.addParam('optimizationMethod', EnumParam, label='Optimization method',
                      choices=['SDP', 'Burer-Monteiro'], expertLevel=LEVEL_ADVANCED,
                      default=0 )
        
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
        self._insertFunctionStep('diagonalizeStep')
        self._insertFunctionStep('correctEigenImagesStep')
        self._insertFunctionStep('reconstructEigenVolumesStep')
        self._insertFunctionStep('createOutputStep')

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
        EPS = 1e-3
        args = []
        args += ['-i', self._getInputMaskFilename()]
        args += ['-o', self._getMaskGalleryStackFilename()]
        args += ['--sampling_rate', self._getAngularSampling()]
        args += ['--sym', self._getSymmetryGroup()]
        #args += ['--method', 'fourier', 1, 1, 'bspline'] 
        args += ['--method', 'real_space'] 
        if self.checkMirrors:
            args += ['--max_tilt_angle', 90.0+EPS]

        # Create a gallery of projections of the input volume
        # with the given angular sampling
        self.runJob("xmipp_angular_project_library", args)
         
        # Remove redundant points
        if self.checkMirrors:
            r = np.sin(EPS)
            z = np.cos(EPS)
            x = r*z
            y = r*r
            direction = np.array((x,y,z))
            
            projectionMd = emlib.MetaData(self._getMaskGalleryMdFilename())
            newProjectionMd = emlib.MetaData()
            for row in emlib.metadata.iterRows(projectionMd):
                x = row.getValue(emlib.MDL_X)
                y = row.getValue(emlib.MDL_Y)
                z = row.getValue(emlib.MDL_Z)
                
                if np.dot(direction, (x, y, z)) >= 0:
                    row.addToMd(newProjectionMd)
                    
            newProjectionMd.write(self._getMaskGalleryMdFilename())
            
        # Binarize the mask
        args = []
        args += ['-i', self._getMaskGalleryMdFilename()]
        args += ['-o', self._getBinarizedMaskGalleryStackFilename()]
        args += ['--save_metadata_stack', self._getMaskGalleryMdFilename()]
        args += ['--keep_input_columns']
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
            particles.readBlock(self._getNeighborsMdFilename(), block)
            if particles.size() == 0:
                print('Direction %s has no particles. Skipping' % block)
                continue

            # Write particles
            directionId = directionalMd.addObject()
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
            directionRow.writeToMd(directionalMd, directionId)

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
        adjacency = scipy.sparse.lil_matrix((nDirections, )*2)
        similarities = scipy.sparse.lil_matrix((nDirections*k, )*2)
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
                n = md0.size()
                if n > 0:
                    md1.intersection(md0, emlib.MDL_ITEM_ID)
                    adjacency[idx0,idx1] = adjacency[idx1,idx0] = n
                    
                    # Get their PCA projection values
                    projections0 = np.array(md0.getColumnValues(emlib.MDL_DIMRED)).T
                    projections1 = np.array(md1.getColumnValues(emlib.MDL_DIMRED)).T
                    
                    projections0 -= projections0.mean(axis=1, keepdims=True)
                    projections1 -= projections1.mean(axis=1, keepdims=True)
                    
                    # Compute similarity
                    similarity = projections0 @ projections1.T

                    # Write them in symmetric positions
                    start0 = idx0*k
                    end0 = start0+k
                    start1 = idx1*k
                    end1 = start1+k
                    similarities[start0:end0, start1:end1] = similarity
                    similarities[start1:end1, start0:end0] = similarity.T
                    
        # Normalize similarities
        similarities /= abs(similarities).max()
        
        # Store the matrices
        self._writeAdjacencyGraph(adjacency.tocsr())
        self._writeCrossCorrelations(similarities.tocsr())

    def basisSynchronizationStep(self):
        k = self._getPrincipalComponentsCount()
        p = self._getOutputPrincipalComponentsCount()
        
        cmd = 'xmipp_synchronize_transform'
        args = []
        args += ['-i', self._getCrossCorrelationsFilename()]
        args += ['-o', self._getBasesFilename()]
        args += ['-k', k]
        args += ['--adjacency', self._getAdjacencyGraphFilename()]
        args += ['--pairwise', self._getPairwiseFilename()]
        args += ['--eigenvalues', self._getEigenvaluesFilename()]
        #args += ['--verbose']
        args += ['--triangular_upper']

        if p > 0:
            args += ['-p', p]
        else:
            args += ['--auto_trim']

        if self.optimizationMethod == 0:
            args += ['--method', 'sdp']
        elif self.optimizationMethod == 1:
            args += ['--method', 'burer-monteiro']

        env = self.getCondaEnv(_conda_env='xmipp_graph')
        self.runJob(cmd, args, env=env, numberOfMpi=1)
   
    def correctBasisStep(self):
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        correctedDirectionMd = emlib.MetaData()
        classificationMd = emlib.MetaData()
        bases = self._readBases()
        inverseBases = bases.transpose(0, 2, 1)

        for i, directionRow in enumerate(emlib.metadata.iterRows(directionMd)):
            directionId = directionRow.getObjId()
            inverseBasis = inverseBases[i]
            correctedClassificationFilename = self._getCorrectedDirectionalClassificationMdFilename(directionId)
            correctedEigenImageFilename = self._getCorrectedDirectionalEigenImagesFilename(directionId)
            
            # Apply basis to projection values
            classificationMd.read(directionRow.getValue(emlib.MDL_SELFILE))
            projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
            correctedProjections = projections @ inverseBasis.T
            classificationMd.setColumnValues(emlib.MDL_DIMRED, correctedProjections.tolist())
            classificationMd.write(correctedClassificationFilename)
            
            # Add to the corrected direction metadata
            directionRow.setValue(emlib.MDL_SELFILE, correctedClassificationFilename)
            directionRow.setValue(emlib.MDL_IMAGE_RESIDUAL, correctedEigenImageFilename)
            directionRow.addToMd(correctedDirectionMd)
            
        correctedDirectionMd.write(self._getCorrectedDirectionalMdFilename())
            
    def combineDirectionsStep(self):   
        symList = xmippLib.SymList()
        symList.readSymmetryFile(self._getSymmetryGroup())
        maxAngularDistance = self._getAngularDistance()
        directionMd = emlib.MetaData(self._getCorrectedDirectionalMdFilename())
        directionalClassificationMd = emlib.MetaData()

        # Accumulate all PCA projection values
        projections = collections.Counter()
        weights = collections.Counter()
        amplitudes = 0
        for directionId in directionMd:
            directionRot = directionMd.getValue(emlib.MDL_ANGLE_ROT, directionId)
            directionTilt = directionMd.getValue(emlib.MDL_ANGLE_TILT, directionId)

            # Read the classification of this direction
            directionalClassificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId))
            values = np.array(directionalClassificationMd.getColumnValues(emlib.MDL_DIMRED))
            var = np.var(values, axis=0)
            stddev = np.sqrt(var)
            amplitudes += var
            
            # Increment the result likelihood value
            for objId in directionalClassificationMd:
                itemId = directionalClassificationMd.getValue(emlib.MDL_ITEM_ID, objId)
                rot = directionalClassificationMd.getValue(emlib.MDL_ANGLE_ROT, objId)
                tilt = directionalClassificationMd.getValue(emlib.MDL_ANGLE_TILT, objId)
                projection = np.array(directionalClassificationMd.getValue(emlib.MDL_DIMRED, objId))
                
                angularDistance = symList.computeDistanceAngles(
                    directionRot, directionTilt, 0.0, 
                    rot, tilt, 0.0,
                    True, self.checkMirrors.get(), False
                )
                weight = (maxAngularDistance-angularDistance) / maxAngularDistance
                assert weight >= 0
                
                projections[itemId] += weight * (projection / stddev)
                weights[itemId] += weight

        amplitudes /= directionMd.size()
        amplitudes = np.sqrt(amplitudes)

        # Write PCA projection values to the output metadata
        result = emlib.MetaData(self._getWienerParticleMdFilename())
        for objId in result:
            itemId = result.getValue(emlib.MDL_ITEM_ID, objId)
            projection = amplitudes*(projections[itemId] / weights[itemId])
            result.setValue(emlib.MDL_DIMRED, projection.tolist(), objId)
                
        # Store
        result.write(self._getClassificationMdFilename())
        self.outputPrincipalComponentCount = Integer(len(projection))
        self._store()
        
    def diagonalizeStep(self):
        pca = sklearn.decomposition.PCA()
        
        classificationFilename = self._getClassificationMdFilename()
        classificationMd = emlib.MetaData(classificationFilename)
        projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
        projections = pca.fit_transform(projections)
        classificationMd.setColumnValues(emlib.MDL_DIMRED, projections.tolist())
        classificationMd.write(classificationFilename)
        
        bases = self._readBases()
        bases = bases @ pca.components_.T
        self._writeBases(bases)

        directionMd = emlib.MetaData(self._getCorrectedDirectionalMdFilename())
        for directionId in directionMd:
            classificationFilename = directionMd.getValue(emlib.MDL_SELFILE, directionId)
            classificationMd = emlib.MetaData(classificationFilename)
            projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
            projections = pca.transform(projections)
            classificationMd.setColumnValues(emlib.MDL_DIMRED, projections.tolist())
            classificationMd.write(classificationFilename)
     
    def correctEigenImagesStep(self):
        bases = self._readBases()
        inverseBases = bases.transpose(0, 2, 1)

        directionMd = emlib.MetaData(self._getCorrectedDirectionalMdFilename())
        eigenimageHandler = emlib.Image()
        for i, directionId in enumerate(directionMd):
            inverseBasis = inverseBases[i]
            eigenimageHandler.read(self._getDirectionalEigenImagesFilename(directionId))
            eigenimages = eigenimageHandler.getData()
            eigenvectors = eigenimages.reshape(inverseBasis.shape[1], -1)
            eigenvectors = inverseBasis @ eigenvectors
            eigenimages = eigenvectors.reshape(inverseBasis.shape[0], 1, *eigenimages.shape[-2:])
            eigenimageHandler.setData(eigenimages)
            eigenimageHandler.write(self._getCorrectedDirectionalEigenImagesFilename(directionId))
        
    def reconstructEigenVolumesStep(self):
        ko = self.outputPrincipalComponentCount.get()
        
        correctedDirectionMd = emlib.MetaData()
        eigenImagesMd = emlib.MetaData()
        for directionId in range(1, 1+ko):
            eigenImagesMd.clear()
            correctedDirectionMd.read(self._getCorrectedDirectionalMdFilename())
            for row in emlib.metadata.iterRows(correctedDirectionMd):
                pcaImages = row.getValue(emlib.MDL_IMAGE_RESIDUAL)
                pcaImage = locationToXmipp(directionId, pcaImages)
                row.setValue(emlib.MDL_IMAGE, pcaImage)
                row.removeLabel(emlib.MDL_MASK)
                row.removeLabel(emlib.MDL_SELFILE)
                row.removeLabel(emlib.MDL_IMAGE_RESIDUAL)
                row.addToMd(eigenImagesMd)
                
            eigenImagesMd.write(self._getEigenImageMdFilename(directionId))
            
            program = 'xmipp_reconstruct_fourier'
            args = []
            args += ['-i', self._getEigenImageMdFilename(directionId)]
            args += ['-o', self._getEigenVolumeFilename(directionId)]
            args += ['--sym', self._getSymmetryGroup()]
            
            self.runJob(program, args)
        
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
        
        ko = self.outputPrincipalComponentCount.get()
        eigenVolumes: SetOfVolumes = self._createSetOfVolumes()
        eigenVolumes.setSamplingRate(self._getInputParticles().getSamplingRate())
        for directionId in range(1, 1+ko):
            eigenVolume = Volume(
                objId=directionId, 
                location=self._getEigenVolumeFilename(directionId)
            )
            eigenVolumes.append(eigenVolume)
        
        self._defineOutputs(**{self.OUTPUT_PARTICLES_NAME: particles})
        self._defineOutputs(**{self.OUTPUT_EIGENVOLUMES_NAME: eigenVolumes})

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
    
    def _getBinarizedMaskGalleryStackFilename(self):
        return self._getExtraPath('mask_gallery_binarized.mrcs')

    def _getNeighborsMdFilename(self):
        return self._getExtraPath('neighbors.xmd')
    
    def _getDirectionPath(self, direction_id: int, *paths):
        return self._getExtraPath('direction_%06d' % direction_id, *paths)
    
    def _getDirectionParticlesMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'particles.xmd')

    def _getDirectionalEigenImagesFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'eigen_images.mrc')

    def _getCorrectedDirectionalEigenImagesFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'corrected_eigen_images.mrc')
    
    def _getDirectionalAverageImageFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'average.mrc')

    def _getDirectionalClassificationMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'classification.xmd')

    def _getCorrectedDirectionalClassificationMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'corrected_classification.xmd')
    
    def _getEigenImageMdFilename(self, directionId: int):
        return self._getExtraPath('directional_eigen_image_%05d.xmd' % directionId)
    
    def _getEigenVolumeFilename(self, directionId: int):
        return self._getExtraPath('eigen_volume_%05d.mrc' % directionId)
    
    def _getDirectionalMdFilename(self):
        return self._getExtraPath('directional.xmd')

    def _getCorrectedDirectionalMdFilename(self):
        return self._getExtraPath('corrected_directional.xmd')

    def _getAdjacencyGraphFilename(self):
        return self._getExtraPath('adjacency.npz')
    
    def _getCrossCorrelationsFilename(self):
        return self._getExtraPath('cross_correlations.npz')

    def _getWeightsFilename(self):
        return self._getExtraPath('weights.npz')
    
    def _getBasesFilename(self):
        return self._getExtraPath('bases.npy')
    
    def _getEigenvaluesFilename(self):
        return self._getExtraPath('eigenvalues.npy')

    def _getPairwiseFilename(self):
        return self._getExtraPath('pairwise.npy')

    def _writeAdjacencyGraph(self, adjacency: scipy.sparse.csr_matrix) -> str:
        path = self._getAdjacencyGraphFilename()
        scipy.sparse.save_npz(path, adjacency)
        return path
    
    def _readAdjacencyGraph(self) -> scipy.sparse.csr_matrix:
        return scipy.sparse.load_npz(self._getAdjacencyGraphFilename())
    
    def _writeCrossCorrelations(self, correlations: scipy.sparse.csr_matrix) -> str:
        path = self._getCrossCorrelationsFilename()
        scipy.sparse.save_npz(path, correlations)
        return path
    
    def _readCrossCorrelations(self) -> scipy.sparse.csr_matrix:
        return scipy.sparse.load_npz(self._getCrossCorrelationsFilename())

    def _readBases(self):
        return np.load(self._getBasesFilename())

    def _writeBases(self, bases):
        return np.save(self._getBasesFilename(), bases)

    def _readEigenvalues(self):
        return np.load(self._getEigenvaluesFilename())
    
    def _readPairwise(self):
        return np.load(self._getPairwiseFilename())
