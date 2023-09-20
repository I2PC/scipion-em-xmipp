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
from pwem.protocols import ProtAnalysis3D
from pwem.objects import (VolumeMask, Class3D, 
                          SetOfParticles, SetOfClasses3D, Particle,
                          Volume, SetOfVolumes )
from pwem.constants import ALIGN_PROJ

from pyworkflow import BETA
from pyworkflow.utils import makePath
from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        MultiPointerParam, EnumParam,
                                        GT, GE, Range,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )

import xmipp3
from xmipp3.convert import writeSetOfParticles, setXmippAttribute
import xmippLib

import os.path
import pickle
import itertools
import numpy as np
import scipy.sparse

class XmippProtAlignedSolidAngles(ProtAnalysis3D, xmipp3.XmippProtocol):
    OUTPUT_CLASSES_NAME = 'classes'
    OUTPUT_VOLUMES_NAME = 'volumes'
    OUTPUT_EXTRA_LABELS = [
        emlib.MDL_SCORE_BY_PCA_RESIDUAL,
    ]
    
    _label = 'aligned solid angles'
    _conda_env = 'xmipp_swiftalign'
    _devStatus = BETA
    _possibleOutputs = {
        OUTPUT_CLASSES_NAME: SetOfClasses3D,
        OUTPUT_VOLUMES_NAME: SetOfVolumes
    }

    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

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
                      pointerClass=VolumeMask,
                      help='Volume mask used for focussing the classification')
        form.addParam('symmetryGroup', StringParam, label='Symmetry group', default='c1')
        form.addParam('resize', IntParam, label='Resize', default=0,
                      validators=[GE(0)])

        form.addSection(label='CTF')
        form.addParam('considerInputCtf', BooleanParam, label='Consider CTF',
                      default=True,
                      help='Consider the CTF of the particles')

        form.addSection(label='Angular neighborhood')
        form.addParam('angularSampling', FloatParam, label='Angular sampling',
                      default=5.0, validators=[Range(0,180)],
                      help='Angular sampling interval in degrees')
        form.addParam('angularDistance', FloatParam, label='Angular distance',
                      default=10.0, validators=[Range(0,180)],
                      help='Maximum angular distance in degrees')
        form.addParam('checkMirrors', BooleanParam, label='Check mirrors',
                      default=False)
        
        form.addParallelSection(threads=0, mpi=8)

        form.addSection(label='Compute')
        form.addParam('batchSize', IntParam, label='Batch size', 
                      default=1024,
                      help='It is recommended to use powers of 2. Using numbers around 8192 works well')
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
        self._insertFunctionStep('classifyStep')
        self._insertFunctionStep('buildGraphStep')
        self._insertFunctionStep('graphOptimizationStep')
        self._insertFunctionStep('classificationStep')
        
        for i in range(2):
            self._insertFunctionStep('reconstructStep', i+1)
        
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions -------------------------------
    def convertInputStep(self):
        particles: SetOfParticles = self.inputParticles.get()
        
        writeSetOfParticles(particles, 
                            self._getInputParticleMdFilename())

        def is_mrc(path: str) -> bool:
            _, ext = os.path.splitext(path)
            return ext == '.mrc' or ext == '.mrcs'
        
        # Convert to MRC if necessary
        if self.copyParticles or self.resize or not all(map(is_mrc, particles.getFiles())):
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
        args += ['--i1', self._getInputParticleMdFilename()]
        args += ['--i2', self._getMaskGalleryMdFilename()]
        args += ['-o', self._getNeighborsMdFilename()]
        args += ['--dist', self._getAngularDistance()]
        args += ['--sym', self._getSymmetryGroup()]
        if self.checkMirrors:
            args += ['--check_mirrors']

        # Compute several groups of the experimental images into
        # different angular neighbourhoods
        self.runJob("xmipp_angular_neighbourhood", args, numberOfMpi=1)
        
    def classifyStep(self):
        maskGalleryMd = emlib.MetaData(self._getMaskGalleryMdFilename())
        particles = emlib.MetaData()
        directionalMd = emlib.MetaData()
        maskRow = emlib.metadata.Row()
        directionRow = emlib.metadata.Row()
        
        for block in emlib.getBlocksInMetaDataFile(self._getNeighborsMdFilename()):
            directionId = int(block.split("_")[1])
            
            # Read information from the metadata
            maskRow.readFromMd(maskGalleryMd, directionId)
            maskFilename = maskRow.getValue(emlib.MDL_IMAGE)
            rot = maskRow.getValue(emlib.MDL_ANGLE_ROT)
            tilt = maskRow.getValue(emlib.MDL_ANGLE_TILT)
            psi = maskRow.getValue(emlib.MDL_ANGLE_PSI)
                
            # Create the working directory for the direction            
            makePath(self._getDirectionPath(directionId))
            
            # Copy the particles
            particles.readBlock(self._getNeighborsMdFilename(), block)
            particles.write(self._getDirectionParticlesMdFilename(directionId))
            
            # Perform the classification
            args = []
            args += ['-i', self._getDirectionParticlesMdFilename(directionId)]
            args += ['-o', self._getDirectionPath(directionId, '')]
            args += ['--mask', maskFilename]
            args += ['--align_to', rot, tilt, psi]
            args += ['--batch', self.batchSize]
            if self.useGpu:
                args += ['--device'] + self._getDeviceList()

            env = self.getCondaEnv()
            env['LD_LIBRARY_PATH'] = '' # Torch does not like it
            self.runJob('xmipp_swiftalign_aligned_2d_classification', args, numberOfMpi=1, env=env)
            
            # Ensemble the output row
            directionRow.copyFromRow(maskRow)
            directionRow.setValue(emlib.MDL_MASK, maskFilename)
            directionRow.setValue(emlib.MDL_IMAGE, self._getDirectionalAverageImageFilename(directionId))
            directionRow.setValue(emlib.MDL_IMAGE_RESIDUAL, self._getDirectionalEigenImageFilename(directionId))
            directionRow.setValue(emlib.MDL_SELFILE, self._getDirectionalClassificationMdFilename(directionId))
            directionRow.setValue(emlib.MDL_CLASS_COUNT, particles.size())
            directionRow.addToMd(directionalMd)

        directionalMd.write(self._getDirectionalMdFilename())
            
    def buildGraphStep(self):
        # Build graph intersecting direction pairs and calculate the similarity
        # between projection values
        symList = xmippLib.SymList()
        symList.readSymmetryFile(self._getSymmetryGroup())
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        md0 = emlib.MetaData()
        md1 = emlib.MetaData()
        nDirections = directionMd.size()
        adjacency = np.zeros((nDirections, )*2)
        for idx0, idx1 in itertools.combinations(range(nDirections), r=2):
            # Get de direction ids of the indices
            directionId0 = idx0 + directionMd.firstObject()
            directionId1 = idx1 + directionMd.firstObject()
            
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
                md1.intersection(md0, emlib.MDL_ITEM_ID)
                
                # Get their PCA projection values
                projection0 = md0.getColumnValues(emlib.MDL_SCORE_BY_PCA_RESIDUAL)
                projection1 = md1.getColumnValues(emlib.MDL_SCORE_BY_PCA_RESIDUAL)
                
                # Compute the similarity as the dot product of their
                # PCA projections
                similarity = np.dot(projection0, projection1)
                
                if similarity:
                    # Write the negative similarity in symmetric positions of 
                    # the adjacency matrix, as the Ising Model's graph analogy
                    # has weights equal to the negative interaction
                    adjacency[idx0, idx1] = adjacency[idx1, idx0] = -similarity
        
        # Convert the adjacency matrix to sparse form
        graph = scipy.sparse.csr_matrix(adjacency)
        graph /= abs(graph).max() # Normalize to avoid numerical inestability

        # Check if there is a single component
        nComponents = scipy.sparse.csgraph.connected_components(
            graph, 
            directed=False, 
            return_labels=False
        )
        if nComponents != 1:
            raise RuntimeError('The ensenbled graph has multiple components. '
                               'Try incrementing the maximum distance parameter')

        # Store the graph
        scipy.sparse.save_npz(self._getGraphFilename(), graph)
        
    def graphOptimizationStep(self):
        # Perform a max cut of the graph
        args = []
        args += ['-i', self._getGraphFilename()]
        args += ['-o', self._getGraphCutFilename()]
        args += ['--sdp']
        #env = self.getCondaEnv(_conda_env='scipion3')
        env = self.getCondaEnv() # TODO
        self.runJob('xmipp_graph_max_cut', args, env=env, numberOfMpi=1)
        
        # Load the cut
        with open(self._getGraphCutFilename(), 'rb') as f:
            _, (v0, v1) = pickle.load(f)
        
        # Invert the least amount of directions
        invert_list = min(v0, v1, key=len)
        
        # Invert the projection values of the chosen
        # directions
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        directionalClassificationMd = emlib.MetaData()
        for i in invert_list:
            objId = int(directionMd.firstObject() + i)
            directionalClassificationMdFilename = directionMd.getValue(
                emlib.MDL_SELFILE, objId
            )
            
            directionalClassificationMd.read(directionalClassificationMdFilename)
            directionalClassificationMd.operate('scoreByPcaResidual=-scoreByPcaResidual')
            directionalClassificationMd.write(directionalClassificationMdFilename)
            # TODO invert the eigenimage
    
    def classificationStep(self):
        # Accumulate all the projection values for a given particle
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        directionalClassificationMd = emlib.MetaData()
        projections = {}
        for objId in directionMd:
            # Read the classification of this direction
            directionalClassificationMd.read(directionMd.getValue(emlib.MDL_SELFILE, objId))
            
            # Increment the result PCA projection value
            for objId2 in directionalClassificationMd:
                itemId = directionalClassificationMd.getValue(emlib.MDL_ITEM_ID, objId2)
                projection = directionalClassificationMd.getValue(emlib.MDL_SCORE_BY_PCA_RESIDUAL, objId2)
                
                if itemId in projections:
                    projections[itemId] += projection
                else:
                    projections[itemId] = projection
        
        # Write projections to the output metadata
        result = emlib.MetaData(self._getInputParticleMdFilename())
        for objId in result:
            itemId = result.getValue(emlib.MDL_ITEM_ID, objId)
            projection = projections.get(itemId, 0.0)
            result.setValue(emlib.MDL_SCORE_BY_PCA_RESIDUAL, projection, objId)
        
        # Classify items according to their projection sign
        result.addLabel(emlib.MDL_REF3D)
        result.operate('ref3d=(scoreByPcaResidual>0)+1')
        
        # Store
        result.write(self._getOutputMetadataFilename())
    
    def reconstructStep(self, cls):
        # Select particles
        classification = emlib.MetaData(self._getOutputMetadataFilename())
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
        classificationMd = emlib.MetaData(self._getOutputMetadataFilename())

        # Create volumes
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(self._getSamplingRate())
        for i in range(2):
            clsId = i + 1
            volume = Volume(objId=clsId)
            volume.setFileName(self._getClassVolumeFilename(clsId))
            volumes.append(volume)
        
        # Classify particles
        def updateItem(item: Particle, row: emlib.metadata.Row):
            assert(item.getObjId() == row.getValue(emlib.MDL_ITEM_ID))
            item.setClassId(row.getValue(emlib.MDL_REF3D))
            setXmippAttribute(item, emlib.MDL_SCORE_BY_PCA_RESIDUAL, row.getValue(emlib.MDL_SCORE_BY_PCA_RESIDUAL))

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
    
    def _getSamplingRate(self):
        return float(self.inputParticles.get().getSamplingRate())
    
    def _getSymmetryGroup(self):
        return self.symmetryGroup.get()
    
    def _getAngularSampling(self):
        return self.angularSampling.get()
    
    def _getAngularDistance(self):
        return self.angularDistance.get()
    
    def _getInputMaskFilename(self):
        return self.inputMask.get().getFileName()
    
    def _getInputParticleMdFilename(self):
        return self._getPath('input_particles.xmd')
    
    def _getInputParticleStackFilename(self):
        return self._getTmpPath('input_particles.mrcs')

    def _getWienerParticleMdFilename(self):
        return self._getExtraPath('particles_wiener.xmd')

    def _getWienerParticleStackFilename(self):
        return self._getExtraPath('particles_wiener.mrcs')

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
    
    def _getDirectionalEigenImageFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'eigen_image.mrc')
    
    def _getDirectionalAverageImageFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'average.mrc')

    def _getDirectionalClassificationMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'classification.xmd')
    
    def _getDirectionalMdFilename(self):
        return self._getExtraPath('directional.xmd')

    def _getGraphFilename(self):
        return self._getExtraPath('graph.npz')
    
    def _getGraphCutFilename(self):
        return self._getExtraPath('graph_cut.pkl')
    
    def _getClassMdFilename(self, cls: int):
        return self._getExtraPath('class_%02d.xmd' % cls)

    def _getClassVolumeFilename(self, cls: int):
        return self._getExtraPath('class_%02d.mrc' % cls)

    def _getOutputMetadataFilename(self):
        return self._getPath('output_particles.xmd')
    