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

import numpy as np
from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, MultiPointerParam,
                                        FloatParam, BooleanParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume, SetOfParticles
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
        form.addParam('particleRadius', IntParam, label="Particle radius (px): ", default=-1,
                      help='This radius should include the particle but be small enough to leave room to create a ring for estimating noise')
        form.addParam('noiseRadius', IntParam, label="Noise radius (px): ", default=-1,
                      help='This radius should be larger than the particle radius to create a ring for estimating noise')
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

        Xdim = self.inputParticles.get().getDimensions()[0]
        Y, X = np.ogrid[:Xdim, :Xdim]
        dist_from_center = np.sqrt((X - Xdim/2) ** 2 + (Y - Xdim/2) ** 2)
        particleRadius = self.particleRadius.get()
        noiseRadius = self.noiseRadius.get()

        noiseMask = particleRadius < dist_from_center <= noiseRadius

        mdResults = md.MetaData(self._getTmpPath("anglesCont.xmd"))
        mdOut = md.MetaData()
        for objId in mdResults:
            itemId = mdResults.getValue(emlib.MDL_ITEM_ID,objId)
            fnResidual = mdResults.getValue(emlib.MDL_IMAGE_RESIDUAL,objId)
            I = xmippLib.Image(fnResidual)

            elements_within_circle = I.getData()[mask]
            sum_of_squares = np.sum(elements_within_circle ** 2)
            Npix = elements_within_circle.size

            elements_between_circles = I.getData()[noiseMask]
            var = np.var(elements_between_circles)

            LL = -sum_of_squares/(2*var) - Npix/2 * np.log(2*np.pi*var)

            newRow = md.Row()
            newRow.setValue(emlib.MDL_ITEM_ID, itemId)
            newRow.setValue(emlib.MDL_LL, float(LL))
            newRow.setValue(emlib.MDL_IMAGE_REF, fnVol)
            # newRow.setValue(emlib.MDL_RESIDUAL_VARIANCE, var)
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
            i += 1
        else:
            for volume in inputRefs:
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
        for item in self.inputRefs:
            if isinstance(item.get(), Volume):
                self.appendRows(outputSet, self._getExtraPath("logLikelihood%03d.xmd" % i))
                i += 1
            else:
                for _ in item.get():
                    self.appendRows(outputSet, self._getExtraPath("logLikelihood%03d.xmd" % i))
                    i += 1

        self._defineOutputs(reprojections=outputSet)
        self._defineSourceRelation(self.inputParticles, outputSet)

        matrix = np.array([particle._xmipp_logLikelihood.get() for particle in outputSet])
        matrix = matrix.reshape((i-1,-1))
        np.save(self._getExtraPath('matrix.npy'), matrix)

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
