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
import multiprocessing.dummy as mp
import numpy as np
import scipy.sparse

from pyworkflow.protocol.params import (MultiPointerParam, FloatParam, GE)

from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.objects import Volume, SetOfVolumes
from pwem.protocols import ProtAnalysis3D
import pwem.emlib.image as emlib

import xmippLib

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
        form.addParam('inputVolumes', MultiPointerParam, label='Input volumes',
                      pointerClass=[SetOfVolumes, Volume], minNumObjects=1 )
        form.addParam('maxShift', FloatParam, label='Maximum shift (px)',
                      validators=[GE(0)], default=16 )
        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.pairwiseAlignStep)
        self._insertFunctionStep(self.synchronizeStep)
        self._insertFunctionStep(self.applyTransformationsStep)
        self._insertFunctionStep(self.averageVolumesStep)
        self._insertFunctionStep(self.createOutputStep)
    
    #------------------------------- STEPS functions ---------------------------
    def pairwiseAlignStep(self):
        volumes = self._getInputVolumes()
        n = len(volumes)
        
        pool = mp.Pool(processes=int(self.numberOfThreads))
        program = 'xmipp_volume_align'
        futures = []
        for index0, index1 in itertools.combinations(range(n), r=2):
            volume0 = volumes[index0]
            volume1 = volumes[index1]
            matrixFilename = self._getPairTransformMatrixFilename(index0, index1)
            
            args = []
            args += ['--i1', emlib.ImageHandler.locationToXmipp(volume0.getLocation())]
            args += ['--i2', emlib.ImageHandler.locationToXmipp(volume1.getLocation())]
            args += ['--frm', 0.25, self.maxShift]
            args += ['--copyGeo', matrixFilename]
            futures.append(pool.apply_async(self.runJob, args=(program, args)))

        for future in futures:
            future.wait()
        
        rotations = np.empty((3*n, 3*n))
        shifts = np.empty((3, n, n))
        for index0, index1 in itertools.combinations(range(n), r=2):
            matrixFilename = self._getPairTransformMatrixFilename(index0, index1)
            matrix = np.loadtxt(matrixFilename).reshape(4, 4)
            
            start0 = 3*index0
            end0 = start0 + 3
            start1 = 3*index1
            end1 = start1 + 3

            rotation = matrix[:3,:3]
            shift = matrix[:3, 3]
            
            rotations[start0:end0, start1:end1] = rotation
            rotations[start1:end1, start0:end0] = rotation.T
            
            shifts[:,index0,index1] = -(rotation.T @ shift)
            shifts[:,index1,index0] = shift
            
        for index in range(n):
            start = 3*index
            end = start + 3
            rotations[start:end,start:end] = np.eye(3)
            
        np.save(self._getPairwiseRotationMatrixFilename(), rotations)
        np.save(self._getPairwiseShiftMatrixFilename(), shifts)
        
    def synchronizeStep(self):
        n = self._getVolumeCount()
        pairwiseRotations = np.load(self._getPairwiseRotationMatrixFilename())
        pairwiseShifts = np.load(self._getPairwiseShiftMatrixFilename())
        
        rotations, w = self._decomposeRotations(pairwiseRotations, n)
        shifts, err = self._decomposeShifts(pairwiseShifts, rotations)

        print('Rotation consistency: %f' % (w.sum() / (3*n)))
        print('Shift consistency: %f' % err)
        
        transforms = np.empty((n, 4, 4))
        transforms[:,:3,:3] = rotations
        transforms[:,:3,3] = shifts
        transforms[:,3,:3] = 0
        transforms[:,3,3] = 1
        
        np.save(self._getTransformationMatrixFilename(), transforms)
        
    def applyTransformationsStep(self):
        volumes = self._getInputVolumes()
        transforms = np.load(self._getTransformationMatrixFilename())
        transforms = np.linalg.inv(transforms)
        
        program = 'xmipp_transform_geometry'
        for i, volume, transform in zip(itertools.count(), volumes, transforms):
            inputVolumeFilename = emlib.ImageHandler.locationToXmipp(volume.getLocation())
            outputVolumeFilename = self._getTransoformedVolumeFilename(i)
            
            args = []
            args += ['-i', inputVolumeFilename]
            args += ['-o', outputVolumeFilename]
            args += ['--matrix']
            args += [','.join(map(str, transform.flatten()))]
            self.runJob(program, args)
            
    def averageVolumesStep(self):
        n = self._getVolumeCount()
        
        volume = xmippLib.Image()
        average = 0
        for i in range(n):
            volume.read(self._getTransoformedVolumeFilename(i))
            average += volume.getData()
        average /= n
        
        volume.setData(average)
        volume.write(self._getAverageFilename())
        
        program = 'xmipp_image_header'
        args = []
        args += ['-i', self._getAverageFilename()]
        args += ['--sampling_rate', self._getSamplingRate()]
        self.runJob(program, args)
                    
    def createOutputStep(self):
        samplingRate = self._getSamplingRate()
        volumes = self._createSetOfVolumes()
        volumes.setSamplingRate(samplingRate)
        n = self._getVolumeCount()
        
        for i in range(n):
            volume = Volume(location=self._getTransoformedVolumeFilename(i))
            volumes.append(volume)
        
        average = Volume(location=self._getAverageFilename())
        average.setSamplingRate(samplingRate)
        
        self._defineOutputs(volumes=volumes, average=average)
            
    # ------------------------------ INFO functions ----------------------------
    def _methods(self):
        messages = []
        return messages

    def _validate(self):
        errors = []
        # TODO validate that all volumes have the same sampling rate
        return errors

    def _summary(self):
        summary = []
        return summary

    #--------------------------- UTILS functions -------------------------------
    def _getInputVolumes(self) -> List[Volume]:
        result = []
        
        objId = 1
        for pointer in self.inputVolumes:
            obj = pointer.get()
            if isinstance(obj, Volume):
                volume: Volume = obj.clone()
                volume.setObjId(objId)
                result.append(volume)
                objId += 1
            elif isinstance(obj, SetOfVolumes):
                volumes: SetOfVolumes = obj
                for volume in volumes:
                    volume = volume.clone()
                    volume.setObjId(objId)
                    result.append(volume)
                    objId += 1
        
        return result

    def _getSamplingRate(self) -> float:
        return self.inputVolumes[0].get().getSamplingRate()
            
    def _getVolumeCount(self) -> int:
        count = 0
        
        for pointer in self.inputVolumes:
            obj = pointer.get()
            if isinstance(obj, Volume):
                count += 1
            elif isinstance(obj, SetOfVolumes):
                volumes: SetOfVolumes = obj
                count += len(volumes)
        
        return count
    
    def _getPairTransformMatrixFilename(self, index0: int, index1: int) -> str:
        return self._getTmpPath('matrix_%06d_%06d.txt' % (index0, index1))
    
    def _getPairwiseRotationMatrixFilename(self) -> str:
        return self._getExtraPath('pairwise_rotations.npy')
    
    def _getPairwiseShiftMatrixFilename(self) -> str:
        return self._getExtraPath('pairwise_shifts.npy')
    
    def _getTransformationMatrixFilename(self) -> str:
        return self._getExtraPath('transforms.npy')
    
    def _getRotatedVolumeFilename(self, i: int) -> str:
        return self._getExtraPath('rotated_volume_%06d.mrc' % i)
    
    def _getTransoformedVolumeFilename(self, i: int) -> str:
        return self._getExtraPath('transformed_volume_%06d.mrc' % i)
    
    def _getAverageFilename(self):
        return self._getExtraPath('average.mrc')
    
    def _decomposeRotations(self, pairwise: np.ndarray, n: int):
        w, v = np.linalg.eigh(pairwise)
        w = w[-3:]
        v = v[:,-3:]
        v *= np.sqrt(w)
        
        matrices = v.reshape(n, 3, 3)
        u, _, vh = np.linalg.svd(matrices, full_matrices=True)
        matrices = u @ vh
        
        return matrices, w
    
    def _decomposeShifts(self, pairwise: np.ndarray, rotations: np.ndarray):
        n = len(rotations)
        nCols = 3*n
        nRows = 3*n*(n-1) // 2

        desing = scipy.sparse.lil_array((nRows, nCols))
        y = np.empty(nRows)
        for i, (index0, index1) in enumerate(itertools.combinations(range(n), r=2)):
            startRow = 3*i
            endRow = startRow + 3
            start0 = 3*index0
            end0 = start0 + 3
            start1 = 3*index1
            end1 = start1 + 3
            
            desing[startRow:endRow,start0:end0] = -rotations[index1] @ rotations[index0].T
            desing[startRow:endRow,start1:end1] = np.eye(3)
            y[startRow:endRow] = pairwise[:,index0,index1]
        desing = desing.tocoo()
        
        result = scipy.sparse.linalg.lsqr(desing, y)
        x = result[0]
        err = result[3]
        shifts = x.reshape((n, 3))
        
        return shifts, err
    