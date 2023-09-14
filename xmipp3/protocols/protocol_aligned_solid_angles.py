# **************************************************************************
# *
# * Authors:     C.O.S. Sorzano (coss@cnb.csic.es)
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
                          Pointer, SetOfFSCs )

from pyworkflow import BETA
from pyworkflow.utils import makePath
from pyworkflow.protocol.params import (Form, PointerParam, 
                                        FloatParam, IntParam,
                                        StringParam, BooleanParam,
                                        MultiPointerParam, EnumParam,
                                        GT, GE, Range,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST )

import xmipp3
from xmipp3.convert import writeSetOfParticles, rowToParticle
import xmippLib

import os.path
import pickle
import itertools
import numpy as np
import scipy.sparse

class XmippProtAlignedSolidAngles(ProtAnalysis3D, xmipp3.XmippProtocol):
    OUTPUT_CLASSES_NAME = 'classes'
    OUTPUT_EXTRA_LABELS = [
        emlib.MDL_SCORE_BY_PCA_RESIDUAL,
    ]
    
    _label = 'aligned solid angles'
    _conda_env = 'xmipp_swiftalign'
    _devStatus = BETA
    _possibleOutputs = {
        OUTPUT_CLASSES_NAME: SetOfClasses3D
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
                      pointerClass=SetOfParticles,
                      help='Input particle set')
        form.addParam('inputMask', PointerParam, label='Mask', important=True,
                      pointerClass=VolumeMask,
                      help='Volume mask used for focussing the classification')
        form.addParam('symmetryGroup', StringParam, label='Symmetry group', default='c1')
        form.addParam('resize', IntParam, label='Resize', default=0,
                      validators=[GT(0)])

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
        self._insertFunctionStep('isingModelOptimizationStep')
        self._insertFunctionStep('classificationStep')
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
            
            if self.resize:
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
        directionalClassesMd = emlib.MetaData()
        eigenImagesMd = emlib.MetaData()
        particles = emlib.MetaData()
        maskRow = emlib.metadata.Row()
        eigenImageRow = emlib.metadata.Row()
        directionalClassRow = emlib.metadata.Row()
        
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
            
            # Write class information
            eigenImageRow.copyFromRow(maskRow)
            eigenImageRow.setValue(emlib.MDL_IMAGE, self._getDirectionalEigenImageFilename(directionId))
            eigenImageRow.addToMd(eigenImagesMd)

            directionalClassRow.copyFromRow(maskRow)
            for classId in range(2):
                directionalClassRow.setValue(emlib.MDL_REF2, classId)
                directionalClassRow.addToMd(directionalClassesMd)
                
        eigenImagesMd.write(self._getDirectionalEigenImagesMdFilename())
        directionalClassesMd.write(self._getDirectionalClassesMdFilename())
            
    def buildGraphStep(self):
        blocks = emlib.getBlocksInMetaDataFile(self._getNeighborsMdFilename())
        directionIds = list(map(lambda block : int(block.split("_")[1]), blocks ))
        
        symList = xmippLib.SymList(self._getSymmetryGroup())
        directionMd = emlib.MetaData(self._getMaskGalleryAnglesMdFilename())
        md0 = emlib.MetaData()
        md1 = emlib.MetaData()
        
        adjacency = np.zeros((len(blocks), )*2)
        for idx0, idx1 in itertools.combinations(range(len(blocks)), r=2):
            # Get de direction ids of the indices
            directionId0 = directionIds[idx0]
            directionId1 = directionIds[idx1]
            
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
            if angularDistance < self.angularDistance:
                # Obtain the intersection of the particles belonging to
                # both directions
                md0.read(self._getDirectionalClassificationMdFilename(directionId0))
                md1.read(self._getDirectionalClassificationMdFilename(directionId1))
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
        
        # Save the graph
        graph = scipy.sparse.csr_matrix(adjacency)
        scipy.sparse.save_npz(self._getGraphFilename(), graph)
        
    def isingModelOptimizationStep(self):
        # Perform a max cut of the graph
        args = []
        args += ['-i', self._getGraphFilename()]
        args += ['-o', self._getGraphCutFilename()]
        #args += ['--sdp']
        #env = self.getCondaEnv(_conda_env='scipion3')
        env = self.getCondaEnv() # TODO
        self.runJob('xmipp_graph_max_cut', args, env=env, numberOfMpi=1)
        
        # Load the cut
        with open(self._getGraphCutFilename(), 'rb') as f:
            _, (v0, v1) = pickle.load(f)
        
        # Invert the least amount of directions
        invert_list = min(v0, v1, key=len)
        print(invert_list)
        
        blocks = emlib.getBlocksInMetaDataFile(self._getNeighborsMdFilename())
        classificationMd = emlib.MetaData()
        for i in invert_list:
            # Get the direction id
            block = blocks[i]
            directionId = int(block.split("_")[1])
            
            # Read the classificationMetaData
            classificationMdFilename = self._getDirectionalClassificationMdFilename(directionId)
            classificationMd.read(classificationMdFilename)
            
            # Invert the PCA projection values
            classificationMd.operate('scoreByPcaResidual=-scoreByPcaResidual')
            
            # Overwrite
            classificationMd.write(classificationMdFilename)
    
    def classificationStep(self):
        result = emlib.MetaData(self._getInputParticleMdFilename())
        
        directionalClassificationMd = emlib.MetaData()
        result.addLabel(emlib.MDL_SCORE_BY_PCA_RESIDUAL)
        blocks = emlib.getBlocksInMetaDataFile(self._getNeighborsMdFilename())
        for block in blocks:
            # Get the direction id
            directionId = int(block.split("_")[1])
            
            # Read the directional classfication
            directionalClassificationMd.read(self._getDirectionalClassificationMdFilename(directionId))
            for objId in directionalClassificationMd:
                itemId = directionalClassificationMd.getValue(emlib.MDL_ITEM_ID, objId)
                projection = directionalClassificationMd.getValue(emlib.MDL_SCORE_BY_PCA_RESIDUAL, objId)
                
                if result.getValue(emlib.MDL_ITEM_ID, itemId) != itemId:
                    raise NotImplementedError('Non contiguous itemId-s are not supported yet')
                
                # Increment the projection value
                projection += result.getValue(emlib.MDL_SCORE_BY_PCA_RESIDUAL, itemId)

                # Overwrite
                result.setValue(emlib.MDL_SCORE_BY_PCA_RESIDUAL, projection, itemId)
        
        
        # Classify items according to their projection ign
        result.addLabel(emlib.MDL_REF3D)
        result.operate('ref3d=(scoreByPcaResidual>0)+1')
        
        # Replace with the original image label
        if result.containsLabel(emlib.MDL_IMAGE_ORIGINAL):
            #FIXME revert shift if scaled
            result.removeLabel(emlib.MDL_IMAGE)
            result.renameColumn(emlib.MDL_IMAGE_ORIGINAL, emlib.MDL_IMAGE)
        
        # Store
        result.write(self._getOutputMetadataFilename())
    
    def createOutputStep(self):
        classificationMd = emlib.MetaData(self._getOutputMetadataFilename())

        def updateItem(item: Particle, row: emlib.metadata.Row):
            particle: Particle = rowToParticle(row, extraLabels=self.OUTPUT_EXTRA_LABELS)
            item.copy(particle)
                
        def updateClass(cls: Class3D):
            cls.setAlignmentProj()
            
        classes3d: SetOfClasses3D = self._createSetOfClasses3D(self.inputParticles)
        classes3d.classifyItems(
            updateItemCallback=updateItem,
            updateClassCallback=updateClass,
            itemDataIterator=emlib.metadata.iterRows(classificationMd)
        )
        
        # Define the output
        self._defineOutputs(**{self.OUTPUT_CLASSES_NAME: classes3d})
        self._defineSourceRelation(self.inputParticles, classes3d)


        
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
    
    def _getDirectionalClassificationMdFilename(self, direction_id: int):
        return self._getDirectionPath(direction_id, 'classification.xmd')
    
    def _getDirectionalEigenImagesMdFilename(self):
        return self._getExtraPath('eigen_images.xmd')
    
    def _getDirectionalClassesMdFilename(self):
        return self._getExtraPath('directional_classes.xmd')
    
    def _getGraphFilename(self):
        return self._getExtraPath('graph.npz')
    
    def _getGraphCutFilename(self):
        return self._getExtraPath('graph_cut.pkl')
    
    def _getOutputMetadataFilename(self):
        return self._getPath('output_particles.xmd')
    