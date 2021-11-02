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


class XmippProtShiftVolume(EMProtocol):
    """ This protocol shifts a volume according to the input shifts"""

    _label = 'shift volume'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVol', PointerParam, pointerClass='Volume', label="Volume", help='Volume to shift')
        form.addParam('shiftBool', BooleanParam, label='Use the same shifts as for the particles?', default='True',
                      help='Use output shifts of protocol "shift particles" which should be executed previously')
        form.addParam('xp', FloatParam, allowsPointers=True, label="shift x", condition='shiftBool', allowsNull=True)
        form.addParam('yp', FloatParam, allowsPointers=True, label="shift y", condition='shiftBool', allowsNull=True)
        form.addParam('zp', FloatParam, allowsPointers=True, label="shift z", condition='shiftBool', allowsNull=True)
        form.addParam('x', FloatParam, label="x", condition='not shiftBool', allowsNull=True)
        form.addParam('y', FloatParam, label="y", condition='not shiftBool', allowsNull=True)
        form.addParam('z', FloatParam, label="z", condition='not shiftBool', allowsNull=True)
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
        if not self.boxSizeBool.get():
            box = self.boxSize.get()
            self.runJob('xmipp_transform_window', '-i "%s" -o "%s" --size %d %d %d' %
                        (fnVol, self._getExtraPath("resized_volume.mrc"), box, box, box))
            fnVol = self._getExtraPath("resized_volume.mrc")
        if self.shiftBool:
            shiftx = self.xp.get()
            shifty = self.yp.get()
            shiftz = self.zp.get()
        else:
            shiftx = self.x.get()
            shifty = self.y.get()
            shiftz = self.z.get()
        program = "xmipp_transform_geometry"
        args = '-i %s -o %s --shift %f %f %f --dont_wrap' % \
               (fnVol, self._getExtraPath("shift_volume.mrc"), shiftx, shifty, shiftz)
        self.runJob(program, args)

    def createOutputStep(self):
        out_vol = Volume()
        in_vol = self.inputVol.get()
        out_vol.setSamplingRate(in_vol.getSamplingRate())
        out_vol.setFileName(self._getExtraPath("shift_volume.mrc"))
        self._defineOutputs(outputVolume=out_vol)
        self._defineOutputs(shiftX=pwobj.Float(self.x.get()),
                            shiftY=pwobj.Float(self.y.get()),
                            shiftZ=pwobj.Float(self.z.get()))
        self._defineSourceRelation(in_vol, out_vol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            if self.shiftBool:
                shiftx = self.xp.get()
                shifty = self.yp.get()
                shiftz = self.zp.get()
            else:
                shiftx = self.x.get()
                shifty = self.y.get()
                shiftz = self.z.get()
            summary.append("Shifts:\nx: %s\ny: %s\nz: %s" % (shiftx, shifty, shiftz))
        return summary

    def _methods(self):
        methods = []
        if self.shiftBool:
            shiftx = self.xp.get()
            shifty = self.yp.get()
            shiftz = self.zp.get()
        else:
            shiftx = self.x.get()
            shifty = self.y.get()
            shiftz = self.z.get()
        methods.append("%s volume shifted to x = %d, y = %d, z = %d." % (self.getObjectTag('outputVolume'),
                                                                         shiftx, shifty, shiftz))
        return methods
