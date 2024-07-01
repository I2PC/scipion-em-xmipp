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


import pwem
from pwem import emlib
import pwem.convert
from pwem.protocols import ProtClassify3D
from pwem.objects import (VolumeMask, Class3D, 
                          SetOfParticles, SetOfClasses3D, Particle,
                          Volume, SetOfVolumes )

from pyworkflow import BETA
from pyworkflow.utils import makePath, createLink, moveFile, cleanPath
from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        GE, GT, Range,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )

import xmipp3
from xmipp3.convert import writeSetOfParticles
import xmippLib

import os.path
import pickle
import itertools
import collections
import random
import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.mixture

import matplotlib.pyplot as plt

class XmippProtSplitVolume(ProtClassify3D, xmipp3.XmippProtocol):
    OUTPUT_CLASSES_NAME = 'classes'
    OUTPUT_VOLUMES_NAME = 'volumes'
    
    _label = 'split volume'
    _devStatus = BETA
    _possibleOutputs = {
        OUTPUT_CLASSES_NAME: SetOfClasses3D,
        OUTPUT_VOLUMES_NAME: SetOfVolumes
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
                      default=2, validators=[GT(0)],
                      help='Number of principal components used for classification')
        form.addParam('classCount', IntParam, label='Class count',
                      default=2, validators=[GT(0)],
                      help='Number of classes to be reconstructed')
        
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
        self._insertFunctionStep('analysisStep')
        self._insertFunctionStep('classifyStep')
    
        for i in range(self.classCount.get()):
            self._insertFunctionStep('reconstructStep', i+1)

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
        args = []
        args += ['-i', self._getInputMaskFilename()]
        args += ['-o', self._getMaskGalleryStackFilename()]
        args += ['--sampling_rate', self._getAngularSampling()]
        args += ['--angular_distance', self._getAngularDistance()]
        args += ['--sym', self._getSymmetryGroup()]
        #args += ['--method', 'fourier', 1, 1, 'bspline'] 
        args += ['--method', 'real_space'] 
        args += ['--compute_neighbors']
        args += ['--experimental_images', self._getInputParticleMdFilename()]
        if self.checkMirrors:
            args += ['--max_tilt_angle', 90]

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
        
    def classifyDirectionsStep(self):
        maskGalleryMd = emlib.MetaData(self._getMaskGalleryMdFilename())
        particles = emlib.MetaData()
        directionalMd = emlib.MetaData()
        maskRow = emlib.metadata.Row()
        directionRow = emlib.metadata.Row()
        k: int = self.principalComponents.get()
        
        env = self.getCondaEnv(_conda_env='xmipp_pyTorch')
        env['LD_LIBRARY_PATH'] = '' # Torch does not like it

        for block in emlib.getBlocksInMetaDataFile(self._getNeighborsMdFilename()):
            directionId = int(block.split("_")[1])
            
            # Read the particles
            particles.readBlock(self._getNeighborsMdFilename(), block)
            if particles.size() == 0:
                print('Direction %d has no particles. Skipping' % directionId)
                continue
            
            # Create a working directory for the direction and copy
            # particles to it
            makePath(self._getDirectionPath(directionId))
            particles.write(self._getDirectionParticlesMdFilename(directionId))
            
            # Read information from the mask metadata
            maskRow.readFromMd(maskGalleryMd, directionId)
            maskFilename = maskRow.getValue(emlib.MDL_IMAGE)
            rot = maskRow.getValue(emlib.MDL_ANGLE_ROT)
            tilt = maskRow.getValue(emlib.MDL_ANGLE_TILT)
            psi = maskRow.getValue(emlib.MDL_ANGLE_PSI)
                
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
            
            directionalClassificationMdFilename = self._getDirectionalClassificationMdFilename(directionId)
            
            # Ensemble the output row
            directionRow.copyFromRow(maskRow)
            directionRow.setValue(emlib.MDL_MASK, maskFilename)
            directionRow.setValue(emlib.MDL_IMAGE, self._getDirectionalAverageImageFilename(directionId))
            directionRow.setValue(emlib.MDL_IMAGE_RESIDUAL, self._getDirectionalEigenImagesFilename(directionId))
            directionRow.setValue(emlib.MDL_SELFILE, directionalClassificationMdFilename)
            directionRow.setValue(emlib.MDL_COUNT, particles.size())
            directionRow.addToMd(directionalMd)

        directionalMd.write(self._getDirectionalMdFilename())
    
    def combineDirectionsStep(self):
        result = emlib.MetaData(self._getWienerParticleMdFilename())
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        classificationMd = emlib.MetaData()
        itemIds = result.getColumnValues(emlib.MDL_ITEM_ID)

        nImages = len(result)
        nDirections = len(directionMd)
        nFeatures = nDirections*k
        k: int = self.principalComponents.get()
        
        projections = scipy.sparse.lil_matrix(nImages, nFeatures)
        for i, directionId in enumerate(directionMd):
            start = i*k
            end = start+k
            classificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId))
            for objId in classificationMd:
                index = itemIds.index(classificationMd.getValue(emlib.MDL_ITEM_ID, objId))
                projection = classificationMd.getValue(emlib.MDL_DIMRED, objId)
                projections[index, start:end] = projection

        projections = projections.tocsr()
        u, s, vh = scipy.sparse.linalg.svds(projections, k=k, return_singular_vectors='vh')
        projections2 = projections @ vh.T
        
        result.setColumnValues(emlib.MLD_DIMRED, projections2.tolist())
        result.write(self._getClassificationMdFilename())
    
    def buildGraphStep(self):
        # Build graph intersecting direction pairs and calculate the similarity
        # between projection values
        k: int = self.principalComponents.get()
        symList = xmippLib.SymList()
        symList.readSymmetryFile(self._getSymmetryGroup())
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        md0 = emlib.MetaData()
        md1 = emlib.MetaData()
        objIds = list(directionMd)
        nDirections = len(objIds)
        cross_correlations = scipy.sparse.lil_matrix((nDirections*k, )*2)
        weights = scipy.sparse.lil_matrix((nDirections*k, )*2)
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
                    cross_correlation = projections0 @ projections1.T
                    weight = norm0 @ norm1.T

                    # Write them in symmetric positions
                    start0 = idx0*k
                    end0 = start0+k
                    start1 = idx1*k
                    end1 = start1+k
                    cross_correlations[start0:end0, start1:end1] = cross_correlation
                    cross_correlations[start1:end1, start0:end0] = cross_correlation.T
                    weights[start0:end0, start1:end1] = weight
                    weights[start1:end1, start0:end0] = weight.T
                    
                    #fig, (ax0, ax1) = plt.subplots(1, 2)
                    #plt.colorbar(ax0.imshow(cross_correlation))
                    #plt.colorbar(ax1.imshow(weight))
                    #plt.show()
                    
        # Normalize weights
        weights /= abs(weights).max()
        
        # Store the matrices
        self._writeCrossCorrelations(cross_correlations.tocsr())
        self._writeWeights(weights.tocsr())

    def basisSynchronizationStep(self):
        k: int = self.principalComponents.get()
        
        cmd = 'xmipp_synchronize_transform'
        args = []
        args += ['-i', self._getCrossCorrelationsFilename()]
        args += ['-w', self._getWeightsFilename()]
        args += ['-o', self._getBasesFilename()]
        args += ['-k', k]
        args += ['--triangular_upper']
        args += ['--orthogonal']
        args += ['-v']

        env = self.getCondaEnv(_conda_env='xmipp_graph')
        self.runJob(cmd, args, env=env, numberOfMpi=1)
   
    def correctBasisStep(self):
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        classificationMd = emlib.MetaData()
        bases = self._readBases()
        #ih = pwem.convert.ImageHandler()
        for i, directionId in enumerate(directionMd):
            classificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId))
            #eigenImages = ih.read(self._getDirectionalEigenImagesFilename(directionId))
            #eigenVectors = eigenImages.reshape(eigenImages.shape[0], -1)
            
            projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED)).T
            basis = bases[i]
            
            correctedProjections = basis.T @ projections
            
            classificationMd.setColumnValues(emlib.MDL_DIMRED, correctedProjections.T.tolist())
            
            classificationMd.write(self._getCorrectedDirectionalClassificationMdFilename(directionId))
   
    def combineDirectionsStep(self):   
        k: int = self.principalComponents.get()
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        directionalClassificationMd = emlib.MetaData()

        # Accumulate all PCA projection values
        counts = collections.Counter()
        projections = collections.Counter()
        for directionId in directionMd:
            # Read the classification of this direction
            #directionalClassificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, directionId)) # TODO
            directionalClassificationMd.read(self._getCorrectedDirectionalClassificationMdFilename(directionId))
            
            # Increment the result likelihood value
            for objId in directionalClassificationMd:
                itemId = directionalClassificationMd.getValue(emlib.MDL_ITEM_ID, objId)
                projection = np.array(directionalClassificationMd.getValue(emlib.MDL_DIMRED, objId))
                
                counts[itemId] += 1
                projections[itemId] += projection

        # Write PCA projection values to the output metadata
        result = emlib.MetaData(self._getWienerParticleMdFilename())
        for objId in result:
            itemId = result.getValue(emlib.MDL_ITEM_ID, objId)

            projection = projections.get(itemId, None)
            if projection is not None:
                projection = projection / counts.get(itemId, 1)
                result.setValue(emlib.MDL_DIMRED, projection.tolist(), objId)
            else:
                result.setValue(emlib.MDL_DIMRED, [0.0]*k, objId)
                
        # Store
        result.write(self._getClassificationMdFilename())
        
    def analysisStep(self):
        classificationMd = emlib.MetaData(self._getClassificationMdFilename())
        projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))

        maxClasses = 16
        aics = np.empty(maxClasses)
        bics = np.empty(maxClasses)
        
        for i in range(maxClasses):
            gmm = sklearn.mixture.GaussianMixture(n_components=i+1)
            gmm.fit(projections)
            aics[i] = gmm.aic(projections)
            bics[i] = gmm.bic(projections)
            
        self._writeAicArray(aics)
        self._writeBicArray(bics)
    
    def classifyStep(self):
        classificationMd = emlib.MetaData(self._getClassificationMdFilename())
        projections = np.array(classificationMd.getColumnValues(emlib.MDL_DIMRED))
        
        gmm = sklearn.mixture.GaussianMixture(n_components=self.classCount.get())
        gmm.fit(projections)
        ref3d = gmm.predict(projections) + 1
        
        classificationMd.setColumnValues(emlib.MDL_REF3D, ref3d.tolist())
        classificationMd.write(self._getClassificationMdFilename())
    
    def reconstructStep(self, cls):
        # Select particles
        classification = emlib.MetaData(self._getClassificationMdFilename())
        classification.removeObjects(emlib.MDValueNE(emlib.MDL_REF3D, cls))
        classification.write(self._getClassMdFilename(cls))
        
        args = []
        args += ['-i', self._getClassMdFilename(cls)]
        args += ['-o', self._getClassVolumeFilename(cls)]
        args += ['--sym', self._getSymmetryGroup()]
    
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
        
    def createOutputStep(self):
        classificationMd = emlib.MetaData(self._getClassificationMdFilename())
        
        # Create volumes
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(self._getSamplingRate())
        for i in range(self.classCount.get()):
            objId = i+1
            volumeFilename = self._getClassVolumeFilename(objId)
            if self.resize > 0:
                resizedVolumeFilename = self._getOutputVolumeFilename(objId)

                args = []
                args += ['--fourier', self._getInputParticles().getXDim()]
                args += ['-i', volumeFilename]
                args += ['-o', resizedVolumeFilename] 
                self.runJob('xmipp_image_resize', args, numberOfMpi=1)

                volumeFilename = resizedVolumeFilename
            
            volume = Volume(objId=objId)
            volume.setFileName(volumeFilename)
            volumes.append(volume)
        
        # Classify particles
        def updateItem(item: Particle, row: emlib.metadata.Row):
            assert(item.getObjId() == row.getValue(emlib.MDL_ITEM_ID))
            item.setClassId(row.getValue(emlib.MDL_REF3D))
            
        def updateClass(cls: Class3D):
            clsId = cls.getObjId()
            representative = volumes[clsId].clone()
            cls.setRepresentative(representative)
            cls.setAlignmentProj()
            
        classes3d: SetOfClasses3D = self._createSetOfClasses3D(self.inputParticles)
        classes3d.classifyItems(
            updateItemCallback=updateItem,
            updateClassCallback=updateClass,
            itemDataIterator=emlib.metadata.iterRows(classificationMd)
        )
        
        # Define the output
        self._defineOutputs(**{self.OUTPUT_CLASSES_NAME: classes3d})
        self._defineOutputs(**{self.OUTPUT_VOLUMES_NAME: volumes})
        self._defineSourceRelation(self.inputParticles, classes3d)
        self._defineSourceRelation(self.inputParticles, volumes)


        
    # --------------------------- UTILS functions -------------------------------
    def _getDeviceList(self):
        gpus = self.getGpuList()
        return list(map('cuda:{:d}'.format, gpus))
    
    def _getInputParticles(self) -> SetOfParticles:
        return self.inputParticles.get()

    def _getReferenceVolume(self) -> Volume:
        return self.referenceVolume.get()

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
    
    def _getInputMaskFilename(self):
        return self._getTmpPath('mask.mrc')
    
    def _getInputParticleMdFilename(self):
        return self._getPath('input_particles.xmd')
    
    def _getInputParticleStackFilename(self):
        return self._getTmpPath('input_particles.mrc')
   
    def _getOutputVolumeFilename(self, cls: int):
        return self._getExtraPath('class_%02d.mrc' % cls)

    def _getWienerParticleMdFilename(self):
        if self.considerInputCtf:
            return self._getExtraPath('particles_wiener.xmd')
        else:
            return self._getInputParticleMdFilename()

    def _getWienerParticleStackFilename(self):
        return self._getTmpPath('particles_wiener.mrc')

    def _getClassificationMdFilename(self):
        return self._getExtraPath('classification.xmd')
    
    def _getClassMdFilename(self, cls: int):
        return self._getExtraPath('class_%02d.xmd' % cls)

    def _getClassVolumeFilename(self, cls: int):
        return self._getExtraPath('class_%02d.mrc' % cls)

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

    def _getDirectionalClassesStackFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'classes.mrcs')
    
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
    
    def _getAicArrayFilename(self):
        return self._getExtraPath('aic.npy')
    
    def _getBicArrayFilename(self):
        return self._getExtraPath('bic.npy')
    
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
    
    def _writeAicArray(self, aics):
        np.save(self._getAicArrayFilename(), aics)
        
    def _readAicArray(self):
        return np.load(self._getAicArrayFilename())
    
    def _writeBicArray(self, bics):
        np.save(self._getBicArrayFilename(), bics)
        
    def _readBicArray(self):
        return np.load(self._getBicArrayFilename())
    