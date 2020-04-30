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

from os.path import join
from shutil import move
from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pwem.objects import Volume
from pwem.protocols import ProtInitialVolume


class XmippProtVolSubtraction(ProtInitialVolume):
    """Subtraction of two volumes, second one could be from a pdb. The volumes should be aligned and equal in size"""

    _label = 'volumes subtraction'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        form.addSection(label='Input')
        form.addParam('vol1', PointerParam, pointerClass='Volume', label="Input volume 1 ", help='Specify a volume.')
        form.addParam('vol2', PointerParam, pointerClass='Volume', label="Input volume 2 ", help='Specify a volume.')
        form.addParam('pdb', BooleanParam, label='Does volume 2 come from a PDB?', default=True)
        form.addParam('masks', BooleanParam, label='Mask volumes?', default=True,
                      help='The masks are not mandatory but highly recommendable.')
        form.addParam('mask1', PointerParam, pointerClass='VolumeMask', label="Mask for volume 1",
                      condition='masks', help='Specify a mask for volume 1.')
        form.addParam('mask2', PointerParam, pointerClass='VolumeMask', label="Mask for volume 2",
                      condition='masks', help='Specify a mask for volume 1.')
        form.addParam('iter', IntParam, label="Number of iterations: ", default=5, expertLevel=LEVEL_ADVANCED)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):

        self._insertFunctionStep('subtractionStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def subtractionStep(self):
        program = "xmipp_volume_subtraction"
        args = '-i1 %s -i2 %s -o %s --iter %s' % (self.vol1.get().getFileName(), self.vol2.get().getFileName(),
                                                  self._getExtraPath("vol_diff.mrc"), self.iter.get())
        if self.pdb:
            args += ' --pdb'
        if self.masks:
            args += ' --mask1 %s --mask2 %s' % (self.mask1.get().getFileName(), self.mask2.get().getFileName())
        self.runJob(program, args)

        if self.masks:
            move('commonmask.mrc', join(self._getExtraPath(), 'common_mask.mrc'))
            move('V1masked.mrc', join(self._getExtraPath(), 'V1_masked.mrc'))
            move('V2masked.mrc', join(self._getExtraPath(), 'V2_masked.mrc'))
        move('V2masked_Amp1.mrc', join(self._getExtraPath(), 'V2_Amp1.mrc'))
        move('V2masked_Amp1_ph2.mrc', join(self._getExtraPath(), 'V2_Amp1_ph2.mrc'))
        move('V2masked_Amp1_ph2_nonneg.mrc', join(self._getExtraPath(), 'V2_Amp1_ph2_nonneg.mrc'))

    def createOutputStep(self):
        volume = Volume()
        volume.setSamplingRate(self.vol1.get().getSamplingRate())
        volume.setFileName(self._getExtraPath("vol_diff.mrc"))
        self._defineOutputs(outputVolume=volume)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            summary.append("Input vol 1: %s" % self.vol1.get().getFileName())
            summary.append("Input vol 2: %s" % self.vol2.get().getFileName())
            if self.masks:
                summary.append("Input mask 1: %s" % self.mask1.get().getFileName())
                summary.append("Input mask 2: %s" % self.mask2.get().getFileName())

        return summary

    def _methods(self):
        if not hasattr(self, 'outputVolume'):
            return ["Output volume not ready yet."]
        else:
            return ["Volume %s subtracted from volume %s." %
                    (self.vol1.get().getFileName(), self.vol2.get().getFileName())]
