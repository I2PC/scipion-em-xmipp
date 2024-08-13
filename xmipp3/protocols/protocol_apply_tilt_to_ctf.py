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

import pyworkflow.protocol.params as params
from pyworkflow.utils.properties import Message

from pwem.objects import (Particle, Coordinate, Micrograph, CTFModel,
                          SetOfParticles)
from pwem.protocols import EMProtocol

import math

class XmippProtApplyTiltToCtf(EMProtocol):
    """ Apply a local deviation to the CTF based on the micrograph's tilt
    angle"""
    _label = 'apply tilt to ctf'
    _tilt_axes = ['X', 'Y']

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam, label=Message.LABEL_INPUT_PART,
                      pointerClass=SetOfParticles, pointerCondition='hasCTF and hasCoordinates',
                      important=True,
                      help='Select the particles that you want to apply the'
                           'local CTF correction.')
        form.addParam('tiltAxis', params.EnumParam, label='Tilt axis', 
                      choices=self._tilt_axes, default=1,
                      help='The tilt axis. In tomography this is Y by convention.')
        form.addParam('tiltAngle', params.FloatParam, label='Tilt angle',
                      default=0, validators=[params.Range(-90, 90)],
                      help='The angle at which the acquisition is tilted. '
                      'In degrees. Mind the sign of the angle.')
        
    #--------------------------- INSERT steps functions --------------------------------------------
    
    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')        

    #--------------------------- STEPS functions --------------------------------------------        
    def createOutputStep(self):
        inputParticles: SetOfParticles = self.inputParticles.get()
        outputParticles: SetOfParticles = self._createSetOfParticles()
        
        TILT_INDICES = [1, 0]
        self.sineFactor = math.sin(math.radians(self.tiltAngle.get()))
        self.tiltIndex = TILT_INDICES[self.tiltAxis.get()]
        
        outputParticles.copyInfo(inputParticles)
        outputParticles.copyItems(inputParticles,
                                  updateItemCallback=self._updateItem )
     
        self._defineOutputs(outputParticles=outputParticles)
        self._defineSourceRelation(self.inputParticles, outputParticles)
    
    #--------------------------- UTILS functions --------------------------------------------
    def _updateItem(self, particle: Particle, _):
        # Compute the CTF offset
        coordinate: Coordinate = particle.getCoordinate()
        micrograph: Micrograph = coordinate.getMicrograph()
        position = coordinate.getPosition()
        r = position[self.tiltIndex]
        r *= micrograph.getSamplingRate() # Convert to angstroms.
        dy = self.sineFactor*r
        
        # Write to output
        ctf: CTFModel = particle.getCTF()
        ctf.setDefocusU(ctf.getDefocusU() + dy)
        ctf.setDefocusV(ctf.getDefocusV() + dy)
