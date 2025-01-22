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

from typing import Dict, List

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

import xmipp3
from xmipp3.convert import (writeSetOfParticles, setXmippAttributes, 
                            locationToXmipp )
import xmippLib

import os.path
import itertools
import numpy as np
import pickle as pkl
import scipy.sparse
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
                      default=1 )
        form.addParam('averagingIterations', IntParam, label='Averaging iterations',
                      expertLevel=LEVEL_ADVANCED, default=32 )
        
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
        self._insertFunctionStep('combineDirectionsStep')
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
        self.directionCount = Integer(directionalMd.size())
        self._store()
            
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
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        particlesMd = emlib.MetaData(self._getInputParticleMdFilename())
        itemIdToIndex = dict(zip(particlesMd.getColumnValues(emlib.MDL_ITEM_ID), itertools.count()))
        nComponents = self._getPrincipalComponentsCount()
        nDirections = directionMd.size()
        nImages = particlesMd.size()
        
        data = scipy.sparse.lil_array((nImages, nDirections*nComponents))
        directionalClassificationMd = emlib.MetaData()
        for j, directionId in enumerate(directionMd):
            directionalClassificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId))
            itemIds = directionalClassificationMd.getColumnValues(emlib.MDL_ITEM_ID)
            values = np.array(directionalClassificationMd.getColumnValues(emlib.MDL_DIMRED))
            start = j*nComponents
            end = start + nComponents
            
            for itemId, value in zip(itemIds, values):
                i = itemIdToIndex[itemId]
                data[i, start:end] = value
                
        data = data.tocsc()
        similarities = data.T @ data
        
        for j in range(nDirections):
            start = j*nComponents
            end = start + nComponents
            similarities[start:end, start:end] = 0
        similarities.eliminate_zeros()

        similarities /= abs(similarities).max()
        self._writeCrossCorrelations(similarities.tocsr())
             
    def basisSynchronizationStep(self):
        n = self._getDirectionCount()
        k = self._getPrincipalComponentsCount()
        p = self._getOutputPrincipalComponentsCount()
        
        cmd = 'xmipp_synchronize_transform'
        args = []
        args += ['-i', self._getCrossCorrelationsFilename()]
        args += ['-o', self._getBasesFilename()]
        args += ['-n', n]
        args += ['-k', k]
        args += ['--pairwise', self._getPairwiseFilename()]
        args += ['--eigenvalues', self._getEigenvaluesFilename()]
        args += ['--auto_trim', 0.99]
        #args += ['--verbose']
        #args += ['--group', 'O']

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

        for i, directionRow in enumerate(emlib.metadata.iterRows(directionMd)):
            directionId = directionRow.getObjId()
            basis = bases[i]
            correctedClassificationFilename = self._getCorrectedDirectionalClassificationMdFilename(directionId)
            correctedEigenImageFilename = self._getCorrectedDirectionalEigenImagesFilename(directionId)
            
            # Apply basis to projection values
            classificationMd.read(directionRow.getValue(emlib.MDL_SELFILE))
            projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
            correctedProjections = projections @ basis # (basis.T @ projections.T).T
            classificationMd.setColumnValues(emlib.MDL_DIMRED, correctedProjections.tolist())
            classificationMd.write(correctedClassificationFilename)
            
            # Add to the corrected direction metadata
            directionRow.setValue(emlib.MDL_SELFILE, correctedClassificationFilename)
            directionRow.setValue(emlib.MDL_IMAGE_RESIDUAL, correctedEigenImageFilename)
            directionRow.addToMd(correctedDirectionMd)
            
        correctedDirectionMd.write(self._getCorrectedDirectionalMdFilename())
            
    def combineDirectionsStep(self):   
        bases = self._readBases()
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        particlesMd = emlib.MetaData(self._getInputParticleMdFilename())
        itemIdToIndex = dict(zip(particlesMd.getColumnValues(emlib.MDL_ITEM_ID), itertools.count()))
        data = self._readDirectionalValues(directionMd, itemIdToIndex, bases)
        nComponents = self._getOutputPrincipalComponentsCount()
        m, _, p = bases.shape
        
        gains = np.ones((m, p))
        noise2 = np.ones((m, p))
        for _ in range(self.averagingIterations.get()):
            averages = self._computeAverages(data, gains, noise2)
            gains = self._computeGains(data, averages, noise2)
            noise2 = self._computeNoise(data, averages, gains)
        averages = self._computeAverages(data, gains, noise2)

        if nComponents == 0:
            pca = sklearn.decomposition.PCA(n_components='mle')
        else:
            pca = sklearn.decomposition.PCA(n_components=nComponents)
        averages = pca.fit_transform(averages)
        
        # Write PCA projection values to the output metadata
        for objId in particlesMd:
            itemId = particlesMd.getValue(emlib.MDL_ITEM_ID, objId)
            i = itemIdToIndex[itemId]
            average = averages[i]
            particlesMd.setValue(emlib.MDL_DIMRED, average.tolist(), objId)

        self._writeGains(gains)
        self._writeAveragingNoise(noise2)
        self._writeDiagonalization(pca)
        particlesMd.write(self._getClassificationMdFilename())
        self.outputPrincipalComponentCount = Integer(len(average))
        self._store()
        
    def reconstructEigenVolumesStep(self):
        bases = self._readBases()
        pca = self._readDiagonalization()
        nComponents = self.outputPrincipalComponentCount.get()

        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        eigenimageHandler = emlib.Image()
        for i, directionId in enumerate(directionMd):
            transform = pca.components_ @ bases[i].T
            
            eigenimageHandler.read(self._getDirectionalEigenImagesFilename(directionId))
            eigenimages = eigenimageHandler.getData()
            eigenvectors = eigenimages.reshape(transform.shape[1], -1)
            eigenvectors = transform @ eigenvectors
            eigenimages = eigenvectors.reshape(transform.shape[0], 1, *eigenimages.shape[-2:])
            eigenimageHandler.setData(eigenimages)
            eigenimageHandler.write(self._getCorrectedDirectionalEigenImagesFilename(directionId))
        
        eigenImagesMd = emlib.MetaData()
        row = emlib.metadata.Row()
        for i in range(1, 1+nComponents):
            eigenImagesMd.clear()
            for directionId in directionMd:
                row.readFromMd(directionMd, directionId)
                pcaImagesFilename = self._getCorrectedDirectionalEigenImagesFilename(directionId)
                pcaImageFilename = locationToXmipp(i, pcaImagesFilename)
                row.setValue(emlib.MDL_IMAGE, pcaImageFilename)
                row.removeLabel(emlib.MDL_MASK)
                row.removeLabel(emlib.MDL_SELFILE)
                row.removeLabel(emlib.MDL_IMAGE_RESIDUAL)
                row.addToMd(eigenImagesMd)
                
            eigenImagesMd.write(self._getEigenImageMdFilename(i))
            
            program = 'xmipp_reconstruct_fourier'
            args = []
            args += ['-i', self._getEigenImageMdFilename(i)]
            args += ['-o', self._getEigenVolumeFilename(i)]
            args += ['--sym', self._getSymmetryGroup()]
            self.runJob(program, args)
            
            program = 'xmipp_image_header'
            args = []
            args += ['-i',  self._getEigenVolumeFilename(i)]
            args += ['--sampling_rate', self._getSamplingRate()]
            self.runJob(program, args, numberOfMpi=1)
        
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
        
        nComponents = self.outputPrincipalComponentCount.get()
        eigenVolumes: SetOfVolumes = self._createSetOfVolumes()
        eigenVolumes.setSamplingRate(self._getInputParticles().getSamplingRate())
        for i in range(1, 1+nComponents):
            eigenVolume = Volume(
                objId=i, 
                location=self._getEigenVolumeFilename(i)
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

    def _getDirectionCount(self) -> int:
        return self.directionCount.get()
    
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

    def _getGainsFilename(self):
        return self._getExtraPath('gains.npy')

    def _getAveragingNoiseFilename(self):
        return self._getExtraPath('noise.npy')

    def _getPcaFilename(self):
        return self._getExtraPath('pca.pkl')

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

    def _readGains(self):
        return np.load(self._getGainsFilename())

    def _writeGains(self, gains):
        return np.save(self._getGainsFilename(), gains)

    def _readAveragingNoise(self):
        return np.load(self._getAveragingNoiseFilename())

    def _writeAveragingNoise(self, noise):
        return np.save(self._getAveragingNoiseFilename(), noise)

    def _readDiagonalization(self) -> sklearn.decomposition.PCA:
        with open(self._getPcaFilename(), 'rb') as f:
            return pkl.load(f)
        
    def _writeDiagonalization(self, pca: sklearn.decomposition.PCA):
        with open(self._getPcaFilename(), 'wb') as f:
            return pkl.dump(pca, f)

    def _readDirectionalValues(self, 
                               directionMd: emlib.MetaData,
                               itemIdToIndex: Dict[int, int],
                               bases: np.ndarray) -> List[scipy.sparse.csr_matrix]:
        m, _, p = bases.shape
        n = len(itemIdToIndex)
        result = [scipy.sparse.lil_matrix((n, m)) for _ in range(p)]
                
        directionalClassificationMd = emlib.MetaData()
        for j, directionId in enumerate(directionMd):
            directionalClassificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId))
            basis = bases[j]
            itemIds = directionalClassificationMd.getColumnValues(emlib.MDL_ITEM_ID)
            values = np.array(directionalClassificationMd.getColumnValues(emlib.MDL_DIMRED))
            values = values @ basis # (basis.T @ values.T).T
            
            for itemId, value in zip(itemIds, values):
                i = itemIdToIndex[itemId]
                for k in range(p):
                    result[k][i, j] = value[k]
                    
        return [item.tocsc() for item in result]

    def _computeAverages(self, 
                         data: List[scipy.sparse.csc_array],
                         gains: np.ndarray, 
                         noise2: np.ndarray ) -> np.ndarray:
        p = len(data)
        n, m = data[0].shape

        numerator = np.zeros((n, p))
        denominator = np.zeros((n, p))
        for k in range(p):
            plane = data[k]
            for j in range(m):
                gain = gains[j,k]
                sigma2 = noise2[j,k]
                start = plane.indptr[j]
                end = plane.indptr[j+1]
                indices = plane.indices[start:end]
                values = plane.data[start:end]
                
                numerator[indices,k] += gain / sigma2 * values
                denominator[indices,k] += np.square(gain) / sigma2

        return numerator / denominator
    
    def _computeGains(self, 
                      data: List[scipy.sparse.csc_array],
                      averages: np.ndarray,
                      noise2: np.ndarray ) -> np.ndarray:
        p = len(data)
        n, m = data[0].shape
        
        result = np.empty((m, p))
        correlations = np.empty(m)
        counts = np.empty(m, dtype=np.int64)
        for k in range(p):
            plane = data[k]
            power = np.dot(averages[:,k], averages[:,k]) / n
            sigma2 = noise2[:,k]
            for j in range(m):
                start = plane.indptr[j]
                end = plane.indptr[j+1]
                indices = plane.indices[start:end]
                y = plane.data[start:end]
                x = averages[indices,k]
                
                correlations[j] = np.dot(x, y) / len(x)
                counts[j] = len(x)

            l = (np.sum(correlations) - m*power) / np.sum(sigma2/counts)
            result[:,k] = (correlations - l*sigma2/counts) / power
            
        return result
    
    def _computeNoise(self, 
                      data: List[scipy.sparse.csc_array],
                      averages: np.ndarray,
                      gains: np.ndarray) -> np.ndarray:
        p = len(data)
        n, m = data[0].shape
        
        result = np.empty((m, p))
        for k in range(p):
            plane = data[k]
            for j in range(m):
                gain = gains[j,k]
                start = plane.indptr[j]
                end = plane.indptr[j+1]
                indices = plane.indices[start:end]
                y = plane.data[start:end]
                x = averages[indices,k]
                
                error = gain*x - y
                result[j,k] = np.dot(error, error) / len(error)

        return result
