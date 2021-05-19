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

from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, EnumParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
import pyworkflow.object as pwobj
from pwem import ALIGN_3D, ALIGN_2D
import pwem.emlib.metadata as md
from pwem.protocols import EMProtocol
from xmipp3.convert import xmippToLocation, writeSetOfParticles


class XmippProtShiftParticles(EMProtocol):
    """ This protocol shifts particles to center them into a point selected in a volume. To do so, it generates new
    shifted images and modify the transformation matrix according to the shift performed."""

    _label = 'shift particles'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, pointerClass='SetOfParticles', label="Particles",
                      help='Select the SetOfParticles with transformation matrix to be shifted.')
        form.addParam('inputVol', PointerParam, pointerClass='Volume', label="Volume", allowsNull=True,
                      help='Volume to select the point (by clicking in the wizard for selecting the new center) that '
                           'will be the new center of the particles.')
        form.addParam('x', IntParam, label="x", help='Use the wizard to select by clicking in the volume the new '
                                                     'center for the shifted particles')
        form.addParam('y', IntParam, label="y")
        form.addParam('z', IntParam, label="z")
        form.addParam('boxSizeBool', BooleanParam, label='Use original box size for the shifted particles?',
                      default='True', help='Use input particles box size for the shifted particles.')
        form.addParam('boxSize', IntParam, label='Final box size', condition='not boxSizeBool',
                      help='Box size for the shifted particles.')
        form.addParam('inv', BooleanParam, label='Inverse', expertLevel=LEVEL_ADVANCED, default='True',
                      help='Use inverse transformation matrix')
        form.addParam('interp', EnumParam, default=0, choices=['Linear', 'Spline'], expertLevel=LEVEL_ADVANCED,
                      display=EnumParam.DISPLAY_HLIST , label='Interpolation',
                      help='Linear: Use bilinear/trilinear interpolation\nSpline: Use spline interpolation')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('shiftStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        """convert input particles into .xmd file """
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath("input_particles.xmd"),
                            alignType=ALIGN_3D)

    def shiftStep(self):
        """call xmipp program to shift the particles"""
        program = "xmipp_transform_geometry"
        centermd = self._getExtraPath("center_particles.xmd")
        if not self.interp.get():
            interp = 'linear'
        else:
            interp = 'spline'
        args = '-i "%s" -o "%s" --shift_to %f %f %f --apply_transform --dont_wrap --interp %s' % \
               (self._getExtraPath("input_particles.xmd"), centermd, self.x.get(), self.y.get(), self.z.get(), interp)
        if self.inv.get():
            args += ' --inverse'
        self.runJob(program, args)

        if not self.boxSizeBool.get():
            box = self.boxSize.get()
            self.runJob('xmipp_transform_window', '-i "%s" -o "%s" --size %d %d %d --save_metadata_stack' %
                        (centermd, self._getExtraPath("crop_particles.stk"), box, box, 1))

    def createOutputStep(self):
        """create output with the new particles"""
        self.ix = 0
        inputParticles = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputParticles)
        if self.boxSizeBool.get():
            outputmd = self._getExtraPath("center_particles.xmd")
        else:
            outputmd = self._getExtraPath("crop_particles.xmd")
        outputSet.copyItems(inputParticles, updateItemCallback=self._updateItem,
                            itemDataIterator=md.iterRows(outputmd))
        self._defineOutputs(outputParticles=outputSet)
        self._defineOutputs(shiftX=pwobj.Float(self.x.get()),
                            shiftY=pwobj.Float(self.y.get()),
                            shiftZ=pwobj.Float(self.z.get()))
        self._defineSourceRelation(inputParticles, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            if not self.interp.get():
                interp = 'linear'
            else:
                interp = 'spline'
            summary.append("%d particles shifted\ninterpolation: %s" % (self.inputParticles.get().getSize(), interp))
            if self.inv.get():
                summary.append("inverse matrix applied")
        return summary

    def _methods(self):
        return ["%d particles shifted to x = %d, y = %d, z = %d." % (self.inputParticles.get().getSize(),
                                                                     self.x.get(), self.y.get(), self.z.get())]

    def _validate(self):
        for part in self.inputParticles.get().iterItems():
            if not part.hasTransform():
                validatemsg = ['Please provide particles which have transformation matrix.']
                return validatemsg

    # --------------------------- UTLIS functions --------------------------------------------
    def _updateItem(self, item, row):
        newFn = row.getValue(md.MDL_IMAGE)
        newLoc = xmippToLocation(newFn)
        item.setLocation(newLoc)
