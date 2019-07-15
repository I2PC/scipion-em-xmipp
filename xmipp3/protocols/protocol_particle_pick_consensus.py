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

import os, time
import sys

from math import sqrt
import numpy as np

from pyworkflow.object import Set, String, Pointer
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtParticlePicking
from pyworkflow.protocol.constants import *
from pyworkflow.em.data import SetOfCoordinates, Coordinate


class XmippProtConsensusPicking(ProtParticlePicking):
    """
    Protocol to estimate the agreement between different particle picking
    algorithms. The protocol takes several Sets of Coordinates calculated
    by different programs and/or different parameter settings. Let's say:
    we consider N independent pickings. Then, a coordinate is considered
    to be a correct particle if M pickers have selected the same particle
    (within a radius in pixels specified in the form).

    If you want to be very strict, then set M=N; that is, a coordinate
    represents a particle if it has been selected by all particles (this
    is the default behaviour). Then you may relax this condition by setting
    M=N-1, N-2, ...

    If you want to be very flexible, set M=1, in this way it suffices that
    1 picker has selected the coordinate to be considered as a particle. Note
    that in this way, the cleaning of the dataset has to be performed by other
    means (screen particles, 2D and 3D classification, ...).
    """

    _label = 'picking consensus'

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = STEPS_SERIAL

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinates', params.MultiPointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates",
                      help='Select the set of coordinates to compare')
        form.addParam('consensusRadius', params.IntParam, default=10,
                      label="Radius",
                      help="All coordinates within this radius (in pixels) "
                           "are presumed to correspond to the same particle")
        form.addParam('consensus', params.IntParam, default=-1,
                      label="Consensus",
                      help="How many times need a particle to be selected to "
                           "be considered as a consensus particle.\n"
                           "*Set to -1* to indicate that it needs to be selected "
                           "by all algorithms: *AND* operation.\n"
                           "*Set to 1* to indicate that it suffices that only "
                           "1 algorithm selects the particle: *OR* operation.")

        # FIXME: It's not using more than one since
        #         self.stepsExecutionMode = STEPS_SERIAL
        # form.addParallelSection(threads=4, mpi=0)

#--------------------------- INSERT steps functions ---------------------------
    def _insertAllSteps(self):
        self.inputMics = 0
        self.processedMics = []
        self.checkedMics = set()
        self.setOfCoords = []
        self.mainInput = self.inputCoordinates[0].get()
        self.sampligRates = []
        coorSteps = self.insertNewCoorsSteps([])
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=coorSteps, wait=True)

    def createOutputStep(self):
        pass

    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all mics
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

    def insertNewCoorsSteps(self, mics):
        deps = []
        for micrograph in mics:
            stepId = self._insertFunctionStep("calculateConsensusStep",
                                              micrograph, prerequisites=[])
            deps.append(stepId)
        return deps

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        streamClosed = []
        readyMics = None
        allMics = set()
        for coordSet in self.inputCoordinates:
            coorSet = SetOfCoordinates(filename=coordSet.get().getFileName())
            coorSet._xmippMd = String()
            coorSet.loadAllProperties()
            streamClosed.append(coorSet.isStreamClosed())
            coorSet.close()
            currentPickMics = {micAgg["_micId"] for micAgg in
                               coordSet.get().aggregate(["MAX"], "_micId", ["_micId"])}
            if not readyMics:  # first time
                readyMics = currentPickMics
            else:  # available mics are those ready for all pickers
                readyMics.intersection_update(currentPickMics)
            allMics = allMics.union(currentPickMics)

        self.streamClosed = all(streamClosed)
        if self.streamClosed:
            # for non streaming and the last iteration of streaming, do all or the rest
            newMicIds = allMics.difference(self.checkedMics)
        else:  # for streaming processing, only go for the ready mics in all pickers
            newMicIds = readyMics.difference(self.checkedMics)
        self.checkedMics.update(newMicIds)

        inMics = self.mainInput.getMicrographs()
        newMics = [inMics[micId].clone() for micId in newMicIds]

        if len(newMics) > 0:
            fDeps = self.insertNewCoorsSteps(newMics)
            self.inputMics = self.inputMics + len(newMics)
            outputStep = self._getFirstJoinStep()
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        self.finished = self.streamClosed and \
                        (self.inputMics == len(self.processedMics))
        streamMode = Set.STREAM_CLOSED if getattr(self, 'finished', False) \
                     else Set.STREAM_OPEN
        if len(self.setOfCoords) > 0:
            outSet = self._loadOutputSet(SetOfCoordinates, 'coordinates.sqlite')
            for item in self.setOfCoords:
                outSet.append(item.clone())
            # outSet.copyItems(self.setOfCoords)
            self.setOfCoords = []
            self._updateOutputSet('consensusCoordinates', outSet, streamMode)
            if self.firstTime:
                for inCorrds in self.inputCoordinates:
                    self._defineTransformRelation(inCorrds, outSet)
            outSet.close()
        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)
        else:
            return

    def _loadOutputSet(self, SetClass, baseName):
        setFile = self._getPath(baseName)

        if os.path.exists(setFile):
            self.firstTime = False
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            self.firstTime = True
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)
            outputSet.setBoxSize(self.mainInput.getBoxSize())

        inMicsPointer = Pointer(self.getMapper().getParent(
                                            self.mainInput.getMicrographs()),
                                            extended='outputMicrographs')
        outputSet.setMicrographs(inMicsPointer)

        return outputSet

    def calculateConsensusStep(self, micrograph):
        print("Consensus calculation for micrograph %d: '%s'"
              % (micrograph.getObjId(), micrograph.getMicName()))

        # Take the sampling rates just once
        if not self.sampligRates:
            for coordinates in self.inputCoordinates:
                micrograph = coordinates.get().getMicrographs()
                self.sampligRates.append(micrograph.getSamplingRate())

        # Get all coordinates for this micrograph
        coords = []
        Ncoords = 0
        n = 0
        for coordinates in self.inputCoordinates:
            coordArray = np.asarray([x.getPosition() for x in
                                     coordinates.get().iterCoordinates(
                                         micrograph.getObjId())], dtype=float)
            coordArray *= float(self.sampligRates[n]) / float(self.sampligRates[0])
            coords.append(np.asarray(coordArray, dtype=int))
            Ncoords += coordArray.shape[0]
            n += 1

        allCoords = np.zeros([Ncoords, 2])
        votes = np.zeros(Ncoords)

        # Add all coordinates in the first method
        N0 = coords[0].shape[0]
        inAllMicrographs = (self.consensus <= 0 or
                            self.consensus >= len(self.inputCoordinates))
        if N0 == 0 and inAllMicrographs:
            self.processedMics.append(micrograph.getObjId())
            return
        elif N0 > 0:
            allCoords[0:N0, :] = coords[0]
            votes[0:N0] = 1

        # Add the rest of coordinates
        Ncurrent = N0
        for n in range(1, len(self.inputCoordinates)):
            for coord in coords[n]:
                if Ncurrent > 0:
                    dist = np.sum((coord - allCoords[0:Ncurrent]) ** 2, axis=1)
                    imin = np.argmin(dist)
                    if sqrt(dist[imin]) < self.consensusRadius:
                        newCoord = (votes[imin] * allCoords[imin,] + coord) / (
                            votes[imin] + 1)
                        allCoords[imin,] = newCoord
                        votes[imin] += 1
                    else:
                        allCoords[Ncurrent, :] = coord
                        votes[Ncurrent] = 1
                        Ncurrent += 1
                else:
                    allCoords[Ncurrent, :] = coord
                    votes[Ncurrent] = 1
                    Ncurrent += 1

        # Select those in the consensus
        if self.consensus <= 0 or self.consensus > len(self.inputCoordinates):
            consensus = len(self.inputCoordinates)
        else:
            consensus = self.consensus.get()
        consensusCoords = allCoords[votes >= consensus, :]
        try:
            jaccardIdx = float(len(consensusCoords)) / (
                float(len(allCoords)) / len(self.inputCoordinates))
            # COSS: Possible problem with concurrent writes
            with open(self._getExtraPath('jaccard.txt'), "a") as fhJaccard:
                fhJaccard.write(
                    "%d %f\n" % (micrograph.getObjId(), jaccardIdx))
        except:
            pass
        # Write the consensus file only if there
        # are some coordinates (size > 0)
        if consensusCoords.size:
            np.savetxt(self._getExtraPath(
                'consensus_%06d.txt' % micrograph.getObjId()), consensusCoords)

            fnTmp = self._getExtraPath(
                'consensus_%06d.txt' % micrograph.getObjId())
            if os.path.exists(fnTmp):
                coords = np.loadtxt(fnTmp)
                if coords.size == 2:  # special case with only one coordinate in consensus
                    coords = [coords]
                for coord in coords:
                    aux = Coordinate()
                    aux.setMicrograph(micrograph)
                    aux.setPosition(coord[0], coord[1])
                    self.setOfCoords.append(aux)
        self.processedMics.append(micrograph.getObjId())

    def _summary(self):
        message = []
        for i, coordinates in enumerate(self.inputCoordinates):
            protocol = self.getMapper().getParent(coordinates.get())
            message.append("Method %d %s" % (i + 1, protocol.getClassLabel()))
        message.append("Radius = %d" % self.consensusRadius)
        message.append("Consensus = %d" % self.consensus)
        return message

    def _methods(self):
        return []
