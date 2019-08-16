# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez
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

from pyworkflow.em.convert import ImageHandler
import pyworkflow.em as em
import pyworkflow.em.metadata as md
from pyworkflow.em.protocol import EMProtocol
from pyworkflow.protocol.params import PointerParam

from xmipp3.convert import xmippToLocation, writeSetOfVolumes

class XmippProtApplyTransformSubtomo(EMProtocol):
    """ Apply alignment matrix and produce a new set of subtomograms, with each subtomogram aligned to its reference. """

    _label = 'apply alignment subtomogram'

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input subtomograms')
        form.addParam('inputSubtomograms', PointerParam, pointerClass="SetOfSubTomograms",
                      label='Set of subtomograms', help="Set of subtomograms to be alignment")
        form.addParallelSection(threads=0, mpi=8)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        subtomosFn = self._getPath('input_subtomos.xmd')
        self._insertFunctionStep('convertInputStep', subtomosFn)
        self._insertFunctionStep('applyAlignmentStep', subtomosFn)
        self._insertFunctionStep('createOutputStep')

        # --------------------------- STEPS functions --------------------------------------------

    def convertInputStep(self, outputFn):
        """ Create a metadata with the images and geometrical information. """
        writeSetOfVolumes(self.inputSubtomograms.get(), outputFn, alignType=em.ALIGN_3D)

        return [outputFn]

    def applyAlignmentStep(self, inputFn):
        """ Create a metadata with the images and geometrical information. """
        outputStk = self._getPath('aligned_subtomograms.stk')
        args = '-i %(inputFn)s -o %(outputStk)s --apply_transform ' % locals()
        self.runJob('xmipp_transform_geometry', args)

        return [outputStk]

    def createOutputStep(self):
        subtomograms = self.inputSubtomograms.get()

        # Generate the SetOfAligned subtomograms
        alignedSet = self._createSetOfVolumes()
        alignedSet.copyInfo(subtomograms)

        inputMd = self._getPath('aligned_subtomograms.xmd')
        alignedSet.copyItems(subtomograms,
                             updateItemCallback=self._updateItem,
                             itemDataIterator=md.iterRows(inputMd, sortByLabel=md.MDL_ITEM_ID))
        # Remove alignment
        alignedSet.setAlignment(em.ALIGN_NONE)

        # Define the output average
        avgFile = self._getExtraPath("average.xmp")

        imgh = ImageHandler()
        avgImage = imgh.computeAverage(alignedSet)

        avgImage.write(avgFile)

        avg = em.Volume()
        avg.setLocation(1, avgFile)
        avg.copyInfo(alignedSet)

        self._defineOutputs(outputAverage=avg)
        self._defineSourceRelation(self.inputSubtomograms, avg)

        self._defineOutputs(outputSubtomograms=alignedSet)
        self._defineSourceRelation(self.inputSubtomograms, alignedSet)

        # --------------------------- INFO functions --------------------------------------------

    def _validate(self):
        errors = []
        return errors

    def _summary(self):
        summary = []
        if not hasattr(self, 'outputSubtomograms'):
            summary.append("Output subtomograms not ready yet.")
        else:
            summary.append("Alignment applied to %s subtomograms." % self.inputSubtomograms.get().getSize())
        return summary

    def _methods(self):
        if not hasattr(self, 'outputSubtomograms'):
            return ["Output subtomograms not ready yet."]
        else:
            return ["We applied alignment to %s subtomograms %s and the output produced is %s."
                    % (self.inputSubtomograms.get().getSize(), self.getObjectTag('inputSubtomograms'),
                       self.getObjectTag('outputSubtomograms'))]

        # --------------------------- UTILS functions --------------------------------------------

    def _updateItem(self, item, row):
        """ Implement this function to do some
        update actions over each single item
        that will be stored in the output Set.
        """
        # By default update the item location (index, filename) with the new binary data location
        newFn = row.getValue(md.MDL_IMAGE)
        newLoc = xmippToLocation(newFn)
        item.setLocation(newLoc)
        # Also remove alignment info
        item.setTransform(None)

