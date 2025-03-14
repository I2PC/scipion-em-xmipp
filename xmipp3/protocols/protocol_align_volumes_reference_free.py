# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
# * Authors:     Oier Lauzirika (olauzirika@cnb.csic.es)
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

from typing import List
import itertools
import numpy as np
import scipy

from pyworkflow import VERSION_2_0
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)

from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.objects import Volume, SetOfVolumes
from pwem.protocols import ProtAnalysis3D
import pwem.emlib.image as emlib

class XmippProtAlignVolumesReferenceFree(ProtAnalysis3D):
    """    
    This protocol aligns volumes using matrix synchronization 
    """
    _label = 'align volumes reference free'
    _lastUpdateVersion = BETA
    _devStatus = PROD

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', PointerParam, pointerClass=SetOfVolumes, 
                      label='Input volumes')
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.pairwiseAlignStep)
        self._insertFunctionStep(self.synchronizationStep)
        self._insertFunctionStep(self.createOutputStep)
    
    #------------------------------- STEPS functions ---------------------------
    def pairwiseAlignStep(self):
        volumes = self._getInputVolumes()
        rotations = np.empty((3*len(volumes), 3*len(volumes)))
        shifts = np.empty((3, len(volumes), len(volumes)))
        
        program = 'xmipp_volume_align'
        for index0, index1 in itertools.combinations(range(len(volumes)), r=2):
            volume0 = volumes[index0]
            volume1 = volumes[index1]
            matrixFilename = self._getPairTransformMatrixFilename(index0, index1)
            
            args = []
            args += ['--i1', emlib.ImageHandler.locationToXmipp(volume0.getLocation())]
            args += ['--i2', emlib.ImageHandler.locationToXmipp(volume1.getLocation())]
            args += ['--frm'] # TODO input parameter
            args += ['--copyGeo', matrixFilename ]
            self.runJob(program, args)
            matrix = np.loadtxt(matrixFilename).reshape(4, 4)
            
            start0 = 3*index0
            end0 = start0 + 3
            start1 = 3*index1
            end1 = start1 + 3
            rotations[start0:end0, start1:end1] = matrix[:3,:3]
            rotations[start1:end1, start0:end0] = matrix[:3,:3].T
            
            shifts[:,index0,index1] = matrix[:3, 3]
            shifts[:,index1,index0] = -matrix[:3, 3]
            
        for index in range(len(volumes)):
            start = 3*index
            end = start + 3
            rotations[start:end,start:end] = np.eye(3)
            shifts[:,index,index] = 0
            
        np.save(self._getPairwiseRotationMatrixFilename(), rotations)
        np.save(self._getPairwiseShiftMatrixFilename(), shifts)
        
    def synchronizationStep(self):
        volumes = self._getInputVolumes()
        rotations = np.load(self._getPairwiseRotationMatrixFilename())
        shifts = np.load(self._getPairwiseShiftMatrixFilename())
        n = len(volumes)
        
        np.save(self._getSynchronizedRotationMatrixFilename(), self._decomposeRotations(rotations, n))
        np.save(self._getSynchronizedShiftsMatrixFilename(), self._decomposeShifts(rotations, n, np.array([64, 64, 64])))
        
    def createOutputStep(self):
        volumes = self._getInputVolumes()
        rotations = np.load(self._getSynchronizedRotationMatrixFilename())
        
        program = 'xmipp_transform_geometry'
        for i, volume, rotation in zip(itertools.count(), volumes, rotations):
            inputVolumeFilename = emlib.ImageHandler.locationToXmipp(volume.getLocation())
            outputVolumeFilename = self._getTransoformedVolumeFilename(i)
            transform = np.zeros((4, 4))
            transform[:3, :3] = rotation.T
            transform[3, 3] = 1
            
            args = []
            args += ['-i', inputVolumeFilename]
            args += ['-o', outputVolumeFilename]
            args += ['--matrix']
            args += [','.join(map(str, transform.flatten()))]
            self.runJob(program, args)
            
            

    # ------------------------------ INFO functions ----------------------------
    def _methods(self):
        messages = []
        return messages

    def _validate(self):
        errors = []
        return errors

    def _summary(self):
        summary = []
        return summary


    #--------------------------- UTILS functions -------------------------------
    def _getInputVolumes(self) -> List[Volume]:
        result = []
        
        for volume in self.inputVolumes.get():
            result.append(volume.clone())
        
        return result

    def _getPairTransformMatrixFilename(self, i: int, j: int) -> str:
        return self._getExtraPath('transform_%06d_%06d.txt' % (i, j))
    
    def _getPairwiseRotationMatrixFilename(self) -> str:
        return self._getExtraPath('pairwise_rotations.npy')
    
    def _getPairwiseShiftMatrixFilename(self) -> str:
        return self._getExtraPath('pairwise_rotations.npy')
    
    def _getSynchronizedRotationMatrixFilename(self) -> str:
        return self._getExtraPath('synchronized_rotations.npy')
    
    def _getSynchronizedShiftsMatrixFilename(self) -> str:
        return self._getExtraPath('synchronized_shifts.npy')
    
    def _getTransoformedVolumeFilename(self, i: int) -> str:
        return self._getExtraPath('transformed_volume_%06d.mrc' % i)
    
    def _decomposeRotations(self, pairwise: np.ndarray, n: int):
        w, v = np.linalg.eigh(pairwise)
        w = w[-3:]
        v = v[:,-3:]
        v *= np.sqrt(w)
        return v.reshape(n, 3, 3)
    
    def _decomposeShifts(self, pairwise: np.ndarray, n: int, dimensions: np.ndarray):
        angles = (pairwise / dimensions[:,None,None]) * 2*np.pi
        s = np.sin(angles)
        c = np.cos(angles)
        
        matrix = np.empty((3, 2*n, 2*n))
        matrix[:, 0::2, 0::2] = c
        matrix[:, 0::2, 1::2] = -s
        matrix[:, 1::2, 0::2] = s
        matrix[:, 1::2, 1::2] = c
        
        w, v = np.linalg.eigh(matrix)
        w = w[:,-2:]
        v = v[:,:,-2:]
        v *= np.sqrt(w[:,None,:])
        
        shifts = np.arctan2(v[:,1::2,0], v[:,0::2,0])
        return shifts