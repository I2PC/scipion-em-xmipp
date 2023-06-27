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

from pyworkflow.protocol.params import PointerParam

from pwem.convert import headers
from pwem.objects import Volume, Transform
from pwem.protocols import EMProtocol

class XmippProtLocalVolAdj(EMProtocol):
    """ Protocol to adjust locally volume intensity to a reference volume."""
    _label = 'volume local adjustment'

    # --------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vol1', PointerParam, pointerClass='Volume', label="Volume 1 ", help='Specify a volume.')
        form.addParam('vol2', PointerParam, pointerClass='Volume', label="Volume 2 ", help='Specify a volume.')
        form.addParam('mask1', PointerParam, pointerClass='VolumeMask', label="Mask for volume 1",
                      help='Specify a mask for volume 1.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('adjustStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def adjustStep(self):
        vol1 = self.vol1.get().clone()
        fnVol1 = vol1.getFileName()
        vol2 = self.vol2.get().getFileName()
        if fnVol1.endswith('.mrc'):
            fnVol1 += ':mrc'
        if vol2.endswith('.mrc'):
            vol2 += ':mrc'
        program = "xmipp_local_volume_adjust"
        args = '--i1 %s --i2 %s -o %s --mask1 %s' % \
               (fnVol1, vol2, self._getExtraPath("output_volume.mrc"), self.mask1.get().getFileName())
        self.runJob(program, args)

    def createOutputStep(self):
        vol1 = self.vol1.get()
        volume = Volume()
        volume.setSamplingRate(vol1.getSamplingRate())
        if vol1.getFileName().endswith('mrc'):
            origin = Transform()
            ccp4header = headers.Ccp4Header(vol1.getFileName(), readHeader=True)
            shifts = ccp4header.getOrigin()
            origin.setShiftsTuple(shifts)
            volume.setOrigin(origin)
        volume.setFileName(self._getExtraPath("output_volume.mrc"))
        filename = volume.getFileName()
        if filename.endswith('.mrc') or filename.endswith('.map'):
            volume.setFileName(filename + ':mrc')
        self._defineOutputs(outputVolume=volume)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = ["Volume 1: %s\nVolume 2: %s\nInput mask 1: %s" %
                   (self.vol1.get().getFileName(), self.vol2.get().getFileName(), self.mask1.get().getFileName())]
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            methods.append("Volume %s adjusted to volume %s" % (self.vol2.get().getFileName(),
                                                                self.vol1.get().getFileName()))
        return methods

    def _validate(self):
        errors = []
        return errors
