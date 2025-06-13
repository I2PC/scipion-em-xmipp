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

from pyworkflow.protocol.params import (PointerParam, FloatParam, GE)

from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.objects import Volume, SetOfVolumes
from pwem.protocols import Prot3D
import pwem.emlib.image as emlib

class XmippProtVolumeMissingCone(Prot3D):
    """    
    This protocol adds a missing cone to a set of volumes. 
    """
    _label = 'add missing cone to volumes'
    _lastUpdateVersion = BETA
    _devStatus = PROD

    def __init__(self, **args):
        Prot3D.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', PointerParam, label='Input volumes',
                      pointerClass=[SetOfVolumes, Volume], minNumObjects=1 )
        form.addParam('angle', FloatParam, label='Max tilt angle (deg)',
                      validators=[GE(0)], default=60.0 )

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.createOutputStep)
    
    #------------------------------- STEPS functions ---------------------------
    def createOutputStep(self):
        result = SetOfVolumes.create(self._getPath())
        result.copyInfo(self.inputVolumes.get())
        
        for volume in self._iterInputVolumes():
            inputFn = emlib.ImageHandler.locationToXmipp(volume.getLocation())
            outputFn = self._getOutputFilename(volume.getObjId())
            
            program = 'xmipp_transform_filter'
            args = []
            args += ['-i', inputFn]
            args += ['-o', outputFn]
            args += ['--fourier', 'cone', self.angle.get()]
            self.runJob(program, args)
            
            program = 'xmipp_image_header'
            args = []
            args += ['-i', outputFn]
            args += ['--sampling_rate', volume.getSamplingRate()]
            self.runJob(program, args)
                    
            volume.setLocation(outputFn)
            result.append(volume)

        self._defineOutputs(volumes=result)
        self._defineSourceRelation(self.inputVolumes, result)
            
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
    def _iterInputVolumes(self):
        volumes = self.inputVolumes.get()
        if isinstance(volumes, SetOfVolumes):
            for volume in volumes:
                yield volume
        
        elif isinstance(volumes, Volume):
            yield volumes
            
        else:
            raise RuntimeError('Unknown input type')

    def _getOutputFilename(self, objId: int) -> str:
        return self._getExtraPath('volume%06d.mrc' % objId)

    