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

from typing import Optional

import pyworkflow.protocol.params as params
from pyworkflow.utils.properties import Message

from pwem.objects import (Particle, Coordinate, Micrograph, CTFModel,
                          SetOfParticles, SetOfMicrographs)
from pwem.protocols import EMProtocol
from pyworkflow import BETA, UPDATED, NEW, PROD

import math

class XmippProtApplyTiltToCtf(EMProtocol):
    """ Applies a local deviation correction to the particle’s contrast transfer function (CTF) estimation based on the tilt angle of the micrograph. This adjustment improves reconstruction quality, especially for tilted samples."""
    _devStatus = PROD

    _label = 'apply tilt to ctf'
    _tilt_axes = ['X', 'Y']
    _tilt_signs = ['Increasing', 'Decreasing']

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam, label=Message.LABEL_INPUT_PART,
                      pointerClass=SetOfParticles, pointerCondition='hasCTF and hasCoordinates',
                      important=True,
                      help='Select the particles that you want to apply the'
                           'local CTF correction.')
        form.addParam('inputMicrographs', params.PointerParam, label=Message.LABEL_INPUT_MIC,
                      pointerClass=SetOfMicrographs,
                      important=True,
                      help='The micrographs from which particles were extracted.')
        form.addParam('tiltAxis', params.EnumParam, label='Tilt axis', 
                      choices=self._tilt_axes, default=1,
                      help='The tilt axis. In tomography this is Y by convention.')
        form.addParam('tiltAngle', params.FloatParam, label='Tilt angle',
                      default=0, validators=[params.Range(0, 90)],
                      help='The angle at which the acquisition is tilted. '
                      'In degrees.')
        form.addParam('tiltSign', params.EnumParam, label='Tilt sign',
                      choices=self._tilt_signs, default=0,
                      help='Wether defocus increases or decreases in terms '
                      'of the selected tilt axis.')
        
    #--------------------------- INSERT steps functions --------------------------------------------
    
    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')        

    #--------------------------- STEPS functions --------------------------------------------        
    def createOutputStep(self):
        inputParticles: SetOfParticles = self.inputParticles.get()
        outputParticles: SetOfParticles = self._createSetOfParticles()
        
        TILT_INDICES = [1, 0]
        SIGNS = [+1, -1]
        sign = SIGNS[self.tiltSign.get()]
        self.tiltIndex = TILT_INDICES[self.tiltAxis.get()]
        self.sineFactor = sign*math.sin(math.radians(self.tiltAngle.get()))
        
        outputParticles.copyInfo(inputParticles)
        outputParticles.copyItems(inputParticles,
                                  updateItemCallback=self._updateItem )
     
        self._defineOutputs(outputParticles=outputParticles)
        self._defineSourceRelation(self.inputParticles, outputParticles)
    
    #--------------------------- UTILS functions --------------------------------------------
    def _updateItem(self, particle: Particle, _):
        # Obtain necessary objects
        coordinate: Coordinate = particle.getCoordinate()
        micrograph: Optional[Micrograph] = coordinate.getMicrograph()
        if micrograph is None:
            micrographId = coordinate.getMicId()
            micrograph = self.inputMicrographs.get()[micrographId]
        
        # Compute the CTF offset
        dimensions = micrograph.getXDim(), micrograph.getYDim()
        position = coordinate.getPosition()
        r = position[self.tiltIndex] - (dimensions[self.tiltIndex] / 2)
        r *= micrograph.getSamplingRate() # Convert to angstroms.
        dy = self.sineFactor*r
        
        # Write to output
        ctf: CTFModel = particle.getCTF()
        ctf.setDefocusU(ctf.getDefocusU() + dy)
        ctf.setDefocusV(ctf.getDefocusV() + dy)
