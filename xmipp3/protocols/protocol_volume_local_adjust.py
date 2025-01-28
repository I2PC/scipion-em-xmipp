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

from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam

from pwem.convert import headers
from pwem.objects import Volume, Transform
from pwem.protocols import EMProtocol
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippProtLocalVolAdj(EMProtocol):
    """Protocol to adjust locally volume intensity to a reference volume. Occupancy volume is saved in protocol folder. Based on 
    https://www.sciencedirect.com/science/article/pii/S1047847723000874?via%3Dihub"""
    _label = 'volume local adjustment'
    _possibleOutputs = Volume
    _devStatus = NEW


    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vol1', PointerParam, pointerClass='Volume', label="Reference volume",
                      help='Specify a volume to be used as reference volume.')
        form.addParam('vol2', PointerParam, pointerClass='Volume', label="Input volume",
                      help='Specify a volume which will be adjusted to the reference volume.')
        form.addParam('mask', PointerParam, pointerClass='VolumeMask', label="Mask for reference volume",
                      help='Specify a mask to define region of interest (which is signal in white (1s) and background in '
                           'black (0s))')
        form.addParam('neighborhood', IntParam, label="Neighborhood (A)", default=5,
                      help='Side length (in Angstroms) of a square which will define the region of adjustment')
        form.addParam('subtract', BooleanParam, label="Perform subtraction?", default=False,
                      help='Perform subtraction of reference volume minus input volume in real space')
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
        args = '--i1 %s --i2 %s -o %s --mask %s --neighborhood %d --sampling %s --save %s' % \
               (fnVol1, vol2, self._getExtraPath("output_volume.mrc"), self.mask.get().getFileName(),
                self.neighborhood.get(), vol1.getSamplingRate(), self._getExtraPath())
        if self.subtract.get():
            args += ' --sub'
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
        neighborhood = self.neighborhood.get()
        vol1 = self.vol1.get()
        summary = ["Volume 1: %s\nVolume 2: %s\nInput mask 1: %s\n\nNeighborhood: %d Å (%d px)" %
                   (vol1.getFileName(), self.vol2.get().getFileName(), self.mask.get().getFileName(),
                    neighborhood, round(neighborhood/vol1.getSamplingRate()))]
        if self.subtract.get():
            summary.append("\nSubtraction performed")
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
        if self.vol1.get().getSamplingRate() != self.vol2.get().getSamplingRate():
            errors.append('Input volumes should have same pixel size')
        if self.vol1.get().getSamplingRate() != self.mask.get().getSamplingRate():
            errors.append('\nInput mask and volumes should have same pixel size')
        return errors
