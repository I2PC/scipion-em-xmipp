# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
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

from math import floor
import numpy as np
import os

from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam, BooleanParam, IntParam)
from pyworkflow.utils.path import cleanPath
from pwem.protocols import ProtAnalysis3D
from pwem.objects import (Volume, SetOfVolumes, AtomStruct, SetOfAtomStructs, SetOfParticles)
import pwem.emlib.metadata as md
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem import emlib
from xmipp3.convert import setXmippAttributes, xmippToLocation
import xmippLib

        
class XmippProtComputeLikelihood(ProtAnalysis3D):
    """This protocol computes the likelihood of a set of particles with assigned angles when compared to a
       set of maps or atomic models"""

    _label = 'compute likelihood'
    _lastUpdateVersion = VERSION_1_1
    _possibleOutputs = {"reprojections": SetOfParticles}
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self._classesInfo = dict()

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label="Input images", important=True,
                      pointerClass='SetOfParticles', pointerCondition='hasAlignmentProj')
        form.addParam('inputRefs', PointerParam, label="References", important=True,
                      pointerClass='Volume,SetOfVolumes',
                      help='Volume, set of volumes or set of atomic structures to which the set of '\
                           'particles will be compared to')
        form.addParam('particleRadius', IntParam, label="Particle radius (px): ", default=-1)
        form.addParam('resol', FloatParam, label="Filter at resolution: ", default=0, expertLevel=LEVEL_ADVANCED,
                      help='Resolution (A) at which subtraction will be performed, filtering the volume projections.'
                           'Value 0 implies no filtering.')
        form.addParam('optimizeGray', BooleanParam, label="Optimize gray: ", default=True, expertLevel=LEVEL_ADVANCED,
                      help='Optimize the gray value between the map reprojection and the experimental image')
        form.addParam('maxGrayChange', FloatParam, label="Max. gray change: ", default=0.99, expertLevel=LEVEL_ADVANCED,
                      condition='optimizeGray',
                      help='The actual gray value can be at most as small as 1-change or as large as 1+change')
        form.addParam('keepResiduals', BooleanParam, label="Keep residuals?", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Keep residuals rather than deleting them after calculation')
        form.addParam('useVariance', BooleanParam, label="Use variance?", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Whether to divide by the variance instead of the standard deviation of the noise (residuals). '
                           'This enhances all matrix elements and seems to work poorly for simulated data')
        form.addParam('squareSum', BooleanParam, label="Square sum of squares?", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Whether to take the square of the sum of squares. This seems to work better when using the variance.')
        form.addParallelSection(threads=0, mpi=8)
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """ Convert input images if necessary """
        self._insertFunctionStep(self.convertStep)
        self._insertFunctionStep(self.produceAllResiduals)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        from ..convert import writeSetOfParticles
        imgSet = self.inputParticles.get()
        writeSetOfParticles(imgSet, self._getTmpPath("images.xmd"))

    def produceResiduals(self, fnVol, i, mask):
        fnAngles = self._getTmpPath("images.xmd")
        anglesOutFn = self._getTmpPath("anglesCont.stk")
        if self.keepResiduals:
            fnResiduals = self._getExtraPath("residuals%03d.stk" % i)
        else:
            fnResiduals = self._getTmpPath("residuals%03d.stk"%i)

        Ts = self.inputParticles.get().getSamplingRate()
        args = "-i %s -o %s --ref %s --sampling %f --oresiduals %s" % (fnAngles, anglesOutFn, fnVol, Ts, fnResiduals)
        if self.resol.get()>0:
            args+=" --max_resolution %f"%self.resol
        if self.optimizeGray:
            args+=" --optimizeGray --max_gray_scale %f"%self.maxGrayChange
        self.runJob("xmipp_angular_continuous_assign2", args)

        mdResults = md.MetaData(self._getTmpPath("anglesCont.xmd"))
        mdOut = md.MetaData()
        for objId in mdResults:
            itemId = mdResults.getValue(emlib.MDL_ITEM_ID,objId)
            fnResidual = mdResults.getValue(emlib.MDL_IMAGE_RESIDUAL,objId)
            I = xmippLib.Image(fnResidual)
            elements_within_circle = I.getData()[mask]
            std_dev = np.std(elements_within_circle)
            if self.useVariance.get():
                std_dev = std_dev**2
            sum_of_squares = np.sum(elements_within_circle ** 2)
            if self.squareSum.get():
                sum_of_squares = sum_of_squares**2
            LL = -sum_of_squares/std_dev

            newRow = md.Row()
            newRow.setValue(emlib.MDL_ITEM_ID, itemId)
            newRow.setValue(emlib.MDL_LL, float(LL))
            newRow.setValue(emlib.MDL_IMAGE_REF, fnVol)
            if self.keepResiduals:
                newRow.setValue(emlib.MDL_IMAGE_RESIDUAL, fnResidual)
            newRow.addToMd(mdOut)
        mdOut.write(self._getExtraPath("logLikelihood%03d.xmd"%i))

    def produceAllResiduals(self):
        Xdim = self.inputParticles.get().getDimensions()[0]
        Y, X = np.ogrid[:Xdim, :Xdim]
        dist_from_center = np.sqrt((X - Xdim/2) ** 2 + (Y - Xdim/2) ** 2)
        particleRadius = self.particleRadius.get()
        if particleRadius<0:
            particleRadius=Xdim/2
        mask = dist_from_center <= particleRadius

        inputRefs = self.inputRefs.get()
        i=1
        if isinstance(inputRefs, Volume):
            self.produceResiduals(inputRefs.getFileName(), i, mask)
        else:
            for volume in self.inputRefs.get():
                self.produceResiduals(volume.getFileName(), i, mask)
                i += 1

    def appendRows(self, outputSet, fnXmd):
        self.iterMd = md.iterRows(fnXmd, md.MDL_ITEM_ID)
        self.lastRow = next(self.iterMd)
        outputSet.copyItems(self.inputParticles.get(), updateItemCallback=self._processRow)

    def createOutputStep(self):
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(self.inputParticles.get())

        i=1
        if isinstance(self.inputRefs.get(), Volume):
            self.appendRows(outputSet, self._getExtraPath("logLikelihood%03d.xmd"%i))
        else:
            for _ in self.inputRefs.get():
                self.appendRows(outputSet, self._getExtraPath("logLikelihood%03d.xmd" % i))
                i += 1

        self._defineOutputs(reprojections=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

    def _getMdRow(self, mdFile, id):
        """ To get a row. Maybe there is way to request a specific row."""
        for row in md.iterRows(mdFile):
            if row.getValue(md.MDL_ITEM_ID) == id:
                return row

        raise Exception("Missing row %s at %s" % (id, mdFile))

    def _processRow(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_ITEM_ID):
            count += 1
            if count:
                particle.setObjId(None)
                setXmippAttributes(particle, self.lastRow,
                                   emlib.MDL_LL, emlib.MDL_IMAGE_REF)
                if self.keepResiduals:
                    setXmippAttributes(particle, self.lastRow, emlib.MDL_IMAGE_RESIDUAL)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
        particle._appendItem = count > 0
