# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
# *             Tomas Majtner (tmajtner@cnb.csic.es)  -- streaming version
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************
"""
Consensus picking protocol
"""

import numpy as np

import pyworkflow.protocol.params as params
from pyworkflow.utils import getFiles, removeBaseExt

from .protocol_particle_pick_consensus import (XmippProtConsensusPicking,
                                               consensusWorker, getReadyMics)

FACTOR_RADIUS = 0.6

class XmippProtPickingRemoveDuplicates(XmippProtConsensusPicking):
    """
    This protocol removes coordinates that are closer than a given threshold.
    The remaining coordinate is the average of the previous ones.
    """

    _label = 'remove duplicates'
    outputName = 'outputCoordinates'
    FN_PREFIX = 'purgedCoords_'

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates", important=True,
                      help='Select the set of coordinates to compare')
        form.addParam('consensusRadius', params.IntParam, default=-1,
                      label="Radius",
                      allowsPointers=True,
                      help="All coordinates within this radius (in pixels) "
                           "are presumed to correspond to the same particle.\n "
                           "If -1 then 0.6 * box size is going to be assigned as radius.")

        # FIXME: It's not using more than one since
        #         self.stepsExecutionMode = STEPS_SERIAL
        # form.addParallelSection(threads=4, mpi=0)

#--------------------------- INSERT steps functions ----------------------------
    def insertNewCoorsSteps(self, mics):
        deps = []
        for micrograph in mics:
            stepId = self._insertFunctionStep("removeDuplicatesStep",
                                              micrograph.getObjId(),
                                              micrograph.getMicName(),
                                              prerequisites=[])
            deps.append(stepId)
        return deps

    def _checkNewInput(self):
        # If continue from an stopped run, don't repeat what is done
        if not self.checkedMics:
            for fn in getFiles(self._getExtraPath()):
                fn = removeBaseExt(fn)
                if fn.startswith(self.FN_PREFIX):
                    self.checkedMics.update([self.getMicId(fn)])
                    self.processedMics.update([self.getMicId(fn)])

        readyMics, self.streamClosed = getReadyMics(self.inputCoordinates.get())

        newMicsIds = readyMics.difference(self.checkedMics)

        if newMicsIds:
            self.checkedMics.update(newMicsIds)

            inMics = self.getMainInput().getMicrographs()
            newMics = [inMics[micId].clone() for micId in newMicsIds]

            fDeps = self.insertNewCoorsSteps(newMics)
            outputStep = self._getFirstJoinStep()
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def getMainInput(self):
        return self.inputCoordinates.get()

    def getConsensusRadius(self):
        if self.consensusRadius.get() == -1:
            consensusRadius = int(self.inputCoordinates.get().getBoxSize()*FACTOR_RADIUS)
        else:
            consensusRadius = self.consensusRadius.get()
        return consensusRadius

    def defineRelations(self, outputSet):
        self._defineTransformRelation(self.getMainInput(), outputSet)

    def removeDuplicatesStep(self, micId, micName):
        print("Removing duplicates for micrograph %d: '%s'"
              % (micId, micName))

        coordArray = np.asarray([x.getPosition() for x in
                                 self.getMainInput().iterCoordinates(micId)],
                                dtype=int)

        consensusWorker([coordArray], 1, self.getConsensusRadius(),
                        self._getTmpPath('%s%s.txt' % (self.FN_PREFIX, micId)))

        self.processedMics.update([micId])

    def _validate(self):
        errors = []
        return errors

    def _summary(self):
        return ["Radius = %d" % self.getConsensusRadius()]
