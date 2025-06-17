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

from pyworkflow.protocol.params import (MultiPointerParam, FloatParam, GE, Range)

from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.objects import Volume, SetOfVolumes
from pwem.protocols import ProtAnalysis3D
import pwem.emlib as emlib

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
        form.addParam('maxFreq', FloatParam, label='Maximum frequency',
                      validators=[Range(0, 1.0)], default=0.25 )
        form.addParam('maxTilt', FloatParam, label='Maximium tilt angle (deg)',
                      validators=[Range(0, 90.0)], default=60.0 )
        form.addParallelSection(mpi=8)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.makePairsStep)
        self._insertFunctionStep(self.pairwiseAlignStep)
        self._insertFunctionStep(self.synchronizeStep)
        self._insertFunctionStep(self.applyTransformationsStep)
        self._insertFunctionStep(self.averageVolumesStep)
        self._insertFunctionStep(self.createOutputStep)
    
    #------------------------------- STEPS functions ---------------------------
    def makePairsStep(self):
        volumes = self._getInputVolumes()
        n = len(volumes)
        
        program = 'xmipp_transform_filter'
        args = []
        args += ['-i', emlib.image.ImageHandler.locationToXmipp(volumes[0].getLocation())]
        args += ['--fourier', 'cone', self.maxTilt.get()]
        args += ['--save', self._getMultiplicityMaskFilename()]
        self.runJob(program, args, numberOfMpi=1)
        
        md = emlib.metadata.MetaData()
        for index0, index1 in itertools.combinations(range(n), r=2):
            objId = md.addObject()
            volume0 = volumes[index0]
            volume1 = volumes[index1]
            md.setValue(emlib.metadata.MDL_IMAGE_REF, emlib.image.ImageHandler.locationToXmipp(volume0.getLocation()), objId)
            md.setValue(emlib.metadata.MDL_IMAGE, emlib.image.ImageHandler.locationToXmipp(volume1.getLocation()), objId)
            md.setValue(emlib.metadata.MDL_MASK, self._getMultiplicityMaskFilename(), objId)
            md.setValue(emlib.metadata.MDL_COMMENT, '%06d,%06d' % (index0, index1), objId)
        md.write(self._getExtraPath('pairs.xmd'))
        
    def pairwiseAlignStep(self):
        volumes = self._getInputVolumes()
        n = len(volumes)
        
        program = 'xmipp_tomo_align_subtomo_pairs'
        args = []
        args += ['-i', self._getPairMetadataFilename()]
        args += ['-o', self._getAlignedPairMetadataFilename()]
        args += ['--maxShift', self.maxShift.get()]
        args += ['--maxFreq', self.maxFreq.get()]
        args += ['--keep_input_columns']
        self.runJob(program, args)
        
        rotations = np.empty((3*n, 3*n))
        shifts = np.empty((3, n, n))
        md = emlib.metadata.MetaData(self._getAlignedPairMetadataFilename())
        for objId in md:
            comment: str = md.getValue(emlib.metadata.MDL_COMMENT, objId)
            rot = md.getValue(emlib.metadata.MDL_ANGLE_ROT, objId)
            tilt = md.getValue(emlib.metadata.MDL_ANGLE_TILT, objId)
            psi = md.getValue(emlib.metadata.MDL_ANGLE_PSI, objId)
            x = md.getValue(emlib.metadata.MDL_SHIFT_X, objId)
            y = md.getValue(emlib.metadata.MDL_SHIFT_Y, objId)
            z = md.getValue(emlib.metadata.MDL_SHIFT_Z, objId)
            
            index0, index1 = comment.split(',')
            index0 = int(index0)
            index1 = int(index1)
            
            start0 = 3*index0
            end0 = start0 + 3
            start1 = 3*index1
            end1 = start1 + 3

            rotation = xmippLib.Euler_angles2matrix(rot, tilt, psi)
            shift = np.array([x, y, z])
            
            rotations[start0:end0, start1:end1] = rotation
            rotations[start1:end1, start0:end0] = rotation.T
            
            shifts[:,index0,index1] = -(rotation.T @ shift)
            shifts[:,index1,index0] = shift
            
        for index in range(n):
            start = 3*index
            end = start + 3
            rotations[start:end,start:end] = np.eye(3)
            shifts[:,index,index] = 0
            
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
            inputVolumeFilename = emlib.image.ImageHandler.locationToXmipp(volume.getLocation())
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
        self.runJob(program, args, numberOfMpi=1)
                    
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
    
    def _getPairMetadataFilename(self):
        return self._getExtraPath('pairs.xmd')
    
    def _getAlignedPairMetadataFilename(self):
        return self._getExtraPath('aligned_pairs.xmd')

    def _getMultiplicityMaskFilename(self):
        return self._getExtraPath('multiplicity.mrc')
    
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
    