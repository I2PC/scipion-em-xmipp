# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
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

from pyworkflow.protocol.params import PointerParam, EnumParam, FloatParam
from pwem.convert.headers import setMRCSamplingRate
from pwem.objects.data import Volume
from pwem.protocols import EMProtocol


class XmippProtRotateVolume(EMProtocol):
    """ Rotates a 3D volume around the x, y, and z axes by specified angles. This protocol allows flexible repositioning of volumes for alignment, comparison, or visualization. """

    _label = 'rotate volume'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vol', PointerParam, pointerClass='Volume', label="Volume ", help='Specify a volume.')
        form.addParam('rotType', EnumParam, choices=['Align with Z', 'rotate'], display=EnumParam.DISPLAY_HLIST,
                      default=1, label='Rotation mode: ', help='Align (x,y,z) with Z axis')
        form.addParam('dirParam', EnumParam, choices=['X', 'Y', 'Z'], default=1, display=EnumParam.DISPLAY_HLIST,
                      label='Axis: ', help='Align (x,y,z) with Z axis')
        form.addParam('deg', FloatParam, label='Degrees: ', default=90, condition='rotType == 1',
                      help='degrees of rotation in selected axis')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('rotateStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def rotateStep(self):
        vol = self.vol.get()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        args = '-i %s -o %s --rotate_volume' % (fnVol, self._getExtraPath('rotated_vol.mrc'))
        rotType = self.rotType.get()
        if rotType == 0:
            args += ' alignZ'
        if rotType == 1:
            args += ' axis %d' % self.deg.get()
        dirParam = self.dirParam.get()
        if dirParam == 0:
            args += ' 1 0 0'
        if dirParam == 1:
            args += ' 0 1 0'
        if dirParam == 2:
            args += ' 0 0 1'
        program = "xmipp_transform_geometry"
        self.runJob(program, args)

    def createOutputStep(self):
        outputVol = Volume()
        inputVol = self.vol.get()
        outputVol.copyInfo(inputVol)
        outputVol.setLocation(self._getExtraPath('rotated_vol.mrc'))
        setMRCSamplingRate(outputVol.getFileName(), inputVol.getSamplingRate())
        self._defineOutputs(outputVolume=outputVol)
        self._defineSourceRelation(inputVol, outputVol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = 'Volume'
        rotType = self.rotType.get()
        if rotType == 0:
            summary += ' aligned with'
        if rotType == 1:
            summary += ' rotated %d degrees around' % self.deg.get()
        dirParam = self.dirParam.get()
        if dirParam == 0:
            summary += ' X axis'
        if dirParam == 1:
            summary += ' Y axis'
        if dirParam == 2:
            summary += ' Z axis'
        return summary
