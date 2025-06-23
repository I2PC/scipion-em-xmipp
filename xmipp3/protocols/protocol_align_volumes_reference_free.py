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

from typing import List, Tuple
import math
import numpy as np

from pyworkflow.protocol.params import (MultiPointerParam, FloatParam, EnumParam,
                                        IntParam, GE, GT, Range)

from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.objects import Volume, SetOfVolumes
from pwem.protocols import ProtAnalysis3D
import pwem.emlib as emlib

from xmipp3.convert import readSetOfVolumes
import xmippLib

class XmippProtAlignVolumesReferenceFree(ProtAnalysis3D):
    """    
    This protocol aligns volumes using matrix synchronization 
    """
    _label = 'align volumes reference free'
    _lastUpdateVersion = BETA
    _devStatus = PROD

    CHOICE_NONE = 'None'
    CHOICE_WEDGE = 'Wedge'
    CHOICE_CONE = 'Cone'
    CHOICES = [
        CHOICE_NONE,
        CHOICE_WEDGE,
        CHOICE_CONE
    ]
    
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
        form.addParam('missingRegion', EnumParam, label='Missing region',
                      choices=self.CHOICES)
        form.addParam('connectivity', IntParam, label='Conectivity',
                      validators=[GT(0)])
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
        k = self.connectivity.get()
        
        program = 'xmipp_transform_filter'
        args = []
        args += ['-i', emlib.image.ImageHandler.locationToXmipp(volumes[0].getLocation())]
        args += ['--fourier']
        choice = self.CHOICES[self.missingRegion.get()]
        if choice == self.CHOICE_WEDGE:
            args += ['wedge', -self.maxTilt.get(), self.maxTilt.get()]
        elif choice == self.CHOICE_CONE:
            args += ['cone', self.maxTilt.get()]
        else:
            raise ValueError('Invalid choice')
        args += ['--save', self._getMultiplicityMaskFilename()]
        self.runJob(program, args, numberOfMpi=1)

        pairs = self._generatePairs(n, k)
        md = emlib.metadata.MetaData()
        for index0, index1 in pairs:
            objId = md.addObject()
            volume0: Volume = volumes[index0]
            location0 = emlib.image.ImageHandler.locationToXmipp(volume0.getLocation())
            md.setValue(emlib.metadata.MDL_REF, index0, objId)
            md.setValue(emlib.metadata.MDL_IMAGE_REF, location0, objId)

            volume1: Volume = volumes[index1]
            location1 = emlib.image.ImageHandler.locationToXmipp(volume1.getLocation())
            md.setValue(emlib.metadata.MDL_IDX, index1, objId)
            md.setValue(emlib.metadata.MDL_IMAGE, location1, objId)

            md.setValue(emlib.metadata.MDL_MASK, self._getMultiplicityMaskFilename(), objId)
        md.write(self._getExtraPath('pairs.xmd'))
        
    def pairwiseAlignStep(self):
        program = 'xmipp_tomo_align_subtomo_pairs'
        args = []
        args += ['-i', self._getPairMetadataFilename()]
        args += ['-o', self._getAlignedPairMetadataFilename()]
        args += ['--maxShift', self.maxShift.get()]
        args += ['--maxFreq', self.maxFreq.get()]
        args += ['--keep_input_columns']
        self.runJob(program, args)
        
    def synchronizeStep(self):
        volumes = self._getInputVolumes()

        program = 'xmipp_synchronize_transform'
        args = []
        args += ['-i', self._getAlignedPairMetadataFilename()]
        args += ['-o', self._getSynchronizedMetadataFilename()]
        args += ['--error', self._getErrorMetadataFilename()]
        self.runJob(program, args, numberOfMpi=1)
        
        md = emlib.metadata.MetaData(self._getSynchronizedMetadataFilename())
        for objId in md:
            itemId = md.getValue(emlib.metadata.MDL_ITEM_ID, objId)
            volume: Volume = volumes[itemId]
            location = emlib.image.ImageHandler.locationToXmipp(volume.getLocation())
            md.setValue(emlib.metadata.MDL_IMAGE, location, objId)
        md.write(self._getSynchronizedMetadataFilename())
        
    def applyTransformationsStep(self):
        program = 'xmipp_transform_geometry'
        args = []
        args += ['-i', self._getSynchronizedMetadataFilename()]
        args += ['-o', self._getAlignedMetadataFilename()]
        args += ['--oroot', self._getAlignedVolumeRoot()]
        args += ['--apply_transform', '--inverse']
        self.runJob(program, args, numberOfMpi=1)
            
    def averageVolumesStep(self):
        md = emlib.metadata.MetaData(self._getAlignedMetadataFilename())
        
        volume = xmippLib.Image()
        average = 0
        for objId in md:
            volume.read(md.getValue(emlib.metadata.MDL_IMAGE, objId))
            average += volume.getData()
        average /= md.size()
        
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
        readSetOfVolumes(self._getAlignedMetadataFilename(), volumes)
        
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
    
    def _generatePairs(self, n: int, k) -> List[Tuple[int, int]]:
        n_2 = n // 2
        e = np.linspace(0, 1, k)
        jumps = np.unique(np.round(n_2 ** e).astype(np.int64))
            
        pairs = set()
        for i in range(n):
            for jump in jumps:
                first = i
                second = (i+int(jump)) % n
                pairs.add((min(first, second), max(first, second)))

        return list(pairs)
        
    def _getPairMetadataFilename(self):
        return self._getExtraPath('pairs.xmd')
    
    def _getAlignedPairMetadataFilename(self):
        return self._getExtraPath('aligned_pairs.xmd')

    def _getSynchronizedMetadataFilename(self):
        return self._getExtraPath('synchronized.xmd')

    def _getErrorMetadataFilename(self):
        return self._getExtraPath('error.xmd')

    def _getMultiplicityMaskFilename(self):
        return self._getExtraPath('multiplicity.mrc')
    
    def _getAlignedMetadataFilename(self):
        return self._getExtraPath('aligned.xmd')
    
    def _getAlignedVolumeRoot(self) -> str:
        return self._getExtraPath('aligned')
    
    def _getAverageFilename(self):
        return self._getExtraPath('average.mrc')
    