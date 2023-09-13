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
from xmipp3.convert import readSetOfParticles, writeSetOfParticles, rowToParticle

import os.path
import math
import itertools
import numpy as np

import scipy.sparse

class XmippProtAlignedSolidAngles(ProtAnalysis3D, xmipp3.XmippProtocol):
    _label = 'aligned solid angles'
    _conda_env = 'xmipp_swiftalign'
    _devStatus = BETA

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
        form.addParam('pcaQuantile', FloatParam, label='PCA Quantile',
                      default=0.1, validators=[Range(0,0.5)],
                      help='PCA Quantile used for obtaining class averages')
        
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
        args += ['--max_tilt_angle', 90]
        args += ['--experimental_images', self._getInputParticleMdFilename()]

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
            args += ['-q', self.pcaQuantile]
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
        
        directionMd = emlib.MetaData(self._getMaskGalleryAnglesMdFilename())
        md0 = emlib.MetaData()
        md1 = emlib.MetaData()
        
        minimumDirectionDot = math.cos(math.radians(2*float(self.angularDistance)))
        
        adjacency = np.zeros((len(blocks), )*2)
        for idx0, idx1 in itertools.combinations(range(len(blocks)), r=2):
            # Get de direction ids of the indices
            directionId0 = directionIds[idx0]
            directionId1 = directionIds[idx1]
            
            # Get the direction vectors
            direction0 = [
                directionMd.getValue(emlib.MDL_X, directionId0),
                directionMd.getValue(emlib.MDL_Y, directionId0),
                directionMd.getValue(emlib.MDL_Z, directionId0)
            ]
            direction1 = [
                directionMd.getValue(emlib.MDL_X, directionId1),
                directionMd.getValue(emlib.MDL_Y, directionId1),
                directionMd.getValue(emlib.MDL_Z, directionId1)
            ]
            
            # Compute the dot product of the angle difference for this pair
            directionDot = np.dot(direction0, direction1)
            
            # Only consider if nearby angles
            if directionDot > minimumDirectionDot:
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
                
                # Write the similarity in symmetric positions of the adjacency
                # matrix
                adjacency[idx0, idx1] = similarity
                adjacency[idx1, idx0] = similarity
        
        # Save the graph
        graph = scipy.sparse.csr_matrix(adjacency)
        scipy.sparse.save_npz(self._getGraphFilename(), graph)
        
    def optimizeRelationsStep(self):
        pass
        
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