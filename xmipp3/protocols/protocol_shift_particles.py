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

from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam
from pwem import ALIGN_3D
from pwem.emlib import lib
import pwem.emlib.metadata as md
from pwem.protocols import EMProtocol
from xmipp3.convert import alignmentToRow, ctfModelToRow


class XmippProtShiftParticles(EMProtocol):
    """ This protocol shifts particles to center them into a point selected in a volume."""

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
        form.addParam('y', IntParam, label="y", help='Use the wizard to select by clicking in the volume the new '
                                                     'center for the shifted particles')
        form.addParam('z', IntParam, label="z", help='Use the wizard to select by clicking in the volume the new '
                                                     'center for the shifted particles')
        form.addParam('boxSizeBool', BooleanParam, label='Use original box size for the shifted particles?',
                      default='True', help='Use input particles box size for the shifted particles.')
        form.addParam('boxSize', IntParam, label='Final box size', condition='not boxSizeBool',
                      help='Box size for the shifted particles.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('shiftStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        """convert input particles into .xmd file """
        mdParticles = lib.MetaData()
        for part in self.inputParticles.get():
            id = part.getObjId()
            ix = part.getIndex()
            fn = "%s@%s" % (ix, part.getFileName())
            nRow = md.Row()
            nRow.setValue(lib.MDL_ITEM_ID, int(id))
            nRow.setValue(lib.MDL_IMAGE, fn)
            alignmentToRow(part.getTransform(), nRow, ALIGN_3D)
            if part.hasCTF():
                ctfModelToRow(part.getCTF(), nRow)
            nRow.addToMd(mdParticles)
        mdParticles.write(self._getExtraPath("input_particles.xmd"))

    def shiftStep(self):
        """call xmipp program to shift the particles"""
        if self.boxSizeBool.get():
            boxSize = self.inputParticles.get().getFirstItem().getXDim()
        else:
            boxSize = self.boxSize.get()
        program = "xmipp_shift_particles"
        args = '-i %s --center %f %f %f -o %s --boxSize %d' % \
               (self._getExtraPath("input_particles.xmd"), self.x.get(), self.y.get(), self.z.get(),
                self._getExtraPath("output_particles"), boxSize)
        self.runJob(program, args)

    def createOutputStep(self):
        """create output with the new particles"""
        self.ix = 0
        inputParticles = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputParticles)
        outputSet.copyItems(inputParticles, updateItemCallback=self._updateItem,
                            itemDataIterator=md.iterRows(self._getExtraPath("input_particles.xmd")))
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputParticles, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            summary.append("%d particles shifted to %d %d %d." % (self.inputParticles.get().getSize(),
                                                                  self.x.get(), self.y.get(), self.z.get()))
        return summary

    def _methods(self):
        pass

    def _validate(self):
        for part in self.inputParticles.get().iterItems():
            if not part.hasTransform():
                validatemsg = ['Please provide particles which have transformation matrix.']
                return validatemsg

    # --------------------------- UTLIS functions --------------------------------------------
    def _updateItem(self, item, row):
        newFn = row.getValue(md.MDL_IMAGE)
        self.ix = self.ix + 1
        newFn = newFn.split('@')[1]
        item.setLocation(self.ix, newFn)
        nwshiftx = row.getValue(md.MDL_SHIFT_X)
        nwshifty = row.getValue(md.MDL_SHIFT_Y)
        A = item.getTransform().getMatrix()
        A[0, 3] = nwshiftx
        A[1, 3] = nwshifty
        item.getTransform().setMatrix(A)
