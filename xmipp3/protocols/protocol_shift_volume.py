# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
# *  BCU, Centro Nacional de Biotecnologia, CSIC
# *
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

from pyworkflow.protocol.params import PointerParam, FloatParam, BooleanParam, IntParam
import pyworkflow.object as pwobj
from pwem.protocols import EMProtocol
from pwem.objects import Volume
from pwem.emlib.image import ImageHandler as ih

import numpy as np


class XmippProtShiftVolume(EMProtocol):
    """ Shifts a 3D volume spatially according to user-provided parameters."""

    _label = 'shift volume'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVol', PointerParam, pointerClass='Volume', label="Volume", help='Volume to shift')
        form.addParam('shiftBool', BooleanParam, label='Use the same shifts as for the particles?', default='True',
                      help='Use output shifts of protocol "shift particles" which should be executed previously')
        form.addParam('inputProtocol', PointerParam, pointerClass='XmippProtShiftParticles', allowsNull=True,
                      label="Shift particles protocol", condition='shiftBool')
        form.addParam('useCM', BooleanParam, label='Use center of mass?', default='False',
                      help='Select the position where the particles will be shifted in a volume displayed in a wizard.')
        COND = 'not shiftBool and not useCM'
        form.addParam('x', FloatParam, label="x", condition=COND, allowsNull=True)
        form.addParam('y', FloatParam, label="y", condition=COND, allowsNull=True)
        form.addParam('z', FloatParam, label="z", condition=COND, allowsNull=True)
        form.addParam('boxSizeBool', BooleanParam, label='Use original box size for the shifted volume?',
                      default='True', help='Use input volume box size for the shifted volume.')
        form.addParam('boxSize', IntParam, label='Final box size', condition='not boxSizeBool',
                      help='Box size of the shifted volume.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('shiftStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def shiftStep(self):
        fnVol = self.inputVol.get().getFileName()
        if not self.boxSizeBool:
            box = self.boxSize.get()
            self.runJob('xmipp_transform_window', '-i "%s" -o "%s" --size %d %d %d' %
                        (fnVol, self._getExtraPath("resized_volume.mrc"), box, box, box))
            fnVol = self._getExtraPath("resized_volume.mrc")
        if self.shiftBool:
            shiftprot = self.inputProtocol.get()
            self.shiftx = shiftprot.shiftX.get()
            self.shifty = shiftprot.shiftY.get()
            self.shiftz = shiftprot.shiftZ.get()
        else:
            if not self.useCM:
                self.shiftx = self.x.get()
                self.shifty = self.y.get()
                self.shiftz = self.z.get()
            else:
                if fnVol.endswith('.mrc'):
                    fnVol += ':mrc'
                vol = ih().read(fnVol).getData()
                vol[vol < 0] = 0
                xs = np.linspace(-vol.shape[2] / 2, vol.shape[2] / 2, vol.shape[2])
                ys = np.linspace(-vol.shape[1] / 2, vol.shape[1] / 2, vol.shape[1])
                zs = np.linspace(-vol.shape[0] / 2, vol.shape[0] / 2, vol.shape[0])
                xs, ys, zs = np.meshgrid(xs, ys, zs, indexing='ij')
                totalMass = vol.sum()
                self.shiftx = -(xs * vol).sum() / totalMass
                self.shifty = -(ys * vol).sum() / totalMass
                self.shiftz = -(zs * vol).sum() / totalMass
        program = "xmipp_transform_geometry"
        args = '-i %s -o %s --shift %f %f %f --dont_wrap' % \
               (fnVol, self._getExtraPath("shift_volume.mrc"), self.shiftx, self.shifty, self.shiftz)
        self.runJob(program, args)

    def createOutputStep(self):
        out_vol = Volume()
        in_vol = self.inputVol.get()
        out_vol.setSamplingRate(in_vol.getSamplingRate())
        out_vol.setFileName(self._getExtraPath("shift_volume.mrc"))
        self._defineOutputs(outputVolume=out_vol)
        self._defineOutputs(shiftX=pwobj.Float(self.shiftx),
                            shiftY=pwobj.Float(self.shifty),
                            shiftZ=pwobj.Float(self.shiftz))
        self._defineSourceRelation(in_vol, out_vol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            if self.shiftBool:
                summary.append("Volume shift as particles in %s" % self.inputProtocol.get())
            else:
                summary.append("User defined shift")
        return summary
