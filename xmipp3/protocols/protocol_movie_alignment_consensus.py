# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
# *             Daniel Marchán Torres (da.marchan@cnb.csic.es)  -- streaming version
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

import os

from math import sqrt
import numpy as np

from pyworkflow.object import Set, String, Pointer
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAlignMovies
from pyworkflow.protocol.constants import *
from pwem.objects import SetOfMovies, Coordinate
from pyworkflow.utils import getFiles, removeBaseExt, moveFile


PICK_MODE_LARGER = 0
PICK_MODE_EQUAL = 1

class XmippProtConsensusMovieAlignment(ProtAlignMovies):
    """
    Protocol to estimate the agreement between different movie alignment
    algorithms.  ...).
    """

    _label = 'movie alignment consensus'
    outputName = 'consensusAlignments'
    FN_PREFIX = 'consensusAlignments_'

    def __init__(self, **args):
        ProtAlignMovies.__init__(self, **args)
        self.stepsExecutionMode = STEPS_SERIAL
        #self.stepsExecutionMode = STEPS_PARALLEL

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputMovies', params.MultiPointerParam,
                      pointerClass='SetOfMovies',
                      label="Input movies", important=True,
                      help='Select the set of movies to compare')
        form.addParam('consensus', params.IntParam, default=-1,
                      label="Consensus",
                      help="How many times need a particle to be selected to "
                           "be considered as a consensus particle.\n"
                           "*Set to -1* to indicate that it needs to be selected "
                           "by all algorithms: *AND* operation.\n"
                           "*Set to 1* to indicate that it suffices that only "
                           "1 algorithm selects the particle: *OR* operation.")
        form.addParam('mode', params.EnumParam, label='Consensus mode',
                      choices=['>=', '='], default=PICK_MODE_LARGER,
                      expertLevel=LEVEL_ADVANCED,
                      help='If the number of votes to progress to the output '
                           'must be either (=) strictly speaking equals to '
                           'the consensus number or (>=) at least equals.')

        # FIXME: It's not using more than one since
        #         self.stepsExecutionMode = STEPS_SERIAL
        # form.addParallelSection(threads=4, mpi=0)

#--------------------------- INSERT steps functions ---------------------------
    def _insertAllSteps(self):
        self.checkedMovies = set()   # those mics ready to be processed (micId)
        self.processedMovies = set() # those mics already processed (micId)
        self.sampligRates = []
        coorSteps = self.insertNewAlignmentSteps([])
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

    def insertNewAlignmentSteps(self, movies):
        deps = []
        for movie in movies:
            stepId = self._insertFunctionStep("calculateConsensusStep",
                                              movie.getObjId(),
                                              movie.getFileName(),
                                              prerequisites=[])
            deps.append(stepId)
        return deps

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        # If continue from an stopped run, don't repeat what is done
        if not self.checkedMovies:
            for fn in getFiles(self._getExtraPath()):
                fn = removeBaseExt(fn)
                if fn.startswith(self.FN_PREFIX):
                    self.checkedMovies.update([self.getMovieId(fn)])
                    self.processedMovies.update([self.getMovieId(fn)])

        streamClosed = []
        readyMovies = None
        allMovies = set()

        for movieSet in self.inputMovies:
            currentAlignedMovies, isSetClosed = getReadyMovies(movieSet.get())
            streamClosed.append(isSetClosed)
            if not readyMovies:  # first time
                readyMovies = currentAlignedMovies
            else:  # available mics are those ready for all pickers
                readyMovies.intersection_update(currentAlignedMovies)
            allMovies = allMovies.union(currentAlignedMovies)

        self.streamClosed = all(streamClosed)
        if self.streamClosed:
            # for non streaming do all and in the last iteration of streaming do the rest
            newMovieIds = allMovies.difference(self.checkedMovies)
        else:  # for streaming processing, only go for the ready mics in all pickers
            newMovieIds = readyMovies.difference(self.checkedMovies)

        if newMovieIds:
            self.checkedMovies.update(newMovieIds)

            inMovies = self.getMainInput().get()
            newMovies = [inMovies[movieId].clone() for movieId in newMovieIds]

            fDeps = self.insertNewAlignmentSteps(newMovies)
            outputStep = self._getFirstJoinStep()
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        self.finished = self.streamClosed and self.checkedMics == self.processedMics
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        newFiles = getFiles(self._getTmpPath())
        if newFiles or self.finished:  # when finished to close the output set
            outSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite')

            for fnTmp in newFiles:
                coords = np.loadtxt(fnTmp)
                moveFile(fnTmp, self._getExtraPath())
                if coords.size == 2:  # special case with only one coordinate
                    coords = [coords]
                for coord in coords:
                    newCoord = Coordinate()
                    micrographs = self.getMainInput().getMicrographs()
                    newCoord.setMicrograph(micrographs[self.getMicId(fnTmp)])
                    newCoord.setPosition(coord[0], coord[1])
                    outSet.append(newCoord)

            firstTime = not self.hasAttribute(self.outputName)
            self._updateOutputSet(self.outputName, outSet, streamMode)
            if firstTime:
                self.defineRelations(outSet)
            outSet.close()

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

    def defineRelations(self, outputSet):
        for inCorrds in self.inputCoordinates:
            self._defineTransformRelation(inCorrds, outputSet)

    def _loadOutputSet(self, SetClass, baseName):
        setFile = self._getPath(baseName)
        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)
            outputSet.setBoxSize(self.getMainInput().getBoxSize())

        # FIXME: The commented line below should work but it fails in streaming
        #        when extracting particles. The line below it fixs that error..
        # inMicsPointer = self.getMainInput().getMicrographs()
        inMicsPointer = Pointer(self.getMapper().getParent(
                                           self.getMainInput().getMicrographs()),
                                           extended='outputMicrographs')
        outputSet.setMicrographs(inMicsPointer)

        return outputSet

    def calculateConsensusStep(self, micId, micName):

        print("Consensus calculation for micrograph %d: '%s'"
              % (micId, micName))

        # Take the sampling rates just once
        if not self.sampligRates:
            for coordinates in self.inputCoordinates:
                micrograph = coordinates.get().getMicrographs()
                self.sampligRates.append(micrograph.getSamplingRate())

        # Get all coordinates for this micrograph
        coords = []
        for idx, coordinates in enumerate(self.inputCoordinates):
            coordArray = np.asarray([x.getPosition() for x in
                                     coordinates.get().iterCoordinates(micId)],
                                    dtype=float)
            coordArray *= float(self.sampligRates[idx]) / float(self.sampligRates[0])
            coords.append(np.asarray(coordArray, dtype=int))

        consensusWorker(coords, self.consensus.get(), self.consensusRadius.get(),
                        self._getTmpPath('%s%s.txt' % (self.FN_PREFIX, micId)),
                        self._getExtraPath('jaccard.txt'), self.mode.get())

        self.processedMics.update([micId])

    def _validate(self):

        errors = []

        # Only for Scipion 2.0, next versions should have the default
        # PointerList validation and this can be removed
        if len(self.inputCoordinates) == 0:
                errors.append('inputCoordinates cannot be EMPTY.')
        # Consider empty pointers:
        else:
            for pointer in self.inputCoordinates:
                obj = pointer.get()
                if obj is None:
                    errors.append('%s is empty.' % obj)

        return errors

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

    def getMainInput(self):
        return self.inputMovies[0].get()

    @classmethod
    def getMovieId(self, fn):
        return int(removeBaseExt(fn).lstrip(self.FN_PREFIX))


def consensusWorker(coords, consensus, consensusRadius, posFn, jaccFn=None,
                    mode=PICK_MODE_LARGER):
    """ Worker for calculate the consensus of N picking algorithms of
          M_n coordinates each one.

        coords: Array of N numpy arrays of M_n coordinates each one.
        consensus: Minimum number of votes to get a consensus coordinate
        consensusRadius: Tolerance to see two coordinates as the same (in pixels)
        posFn: Where to write the consensus coordinates
        jaccFN: Where to write the Jaccard index per micrograph
    """
    if len(coords) == 1:  # self consensus (remove duplicates)
        N0 = 0
        firstInput = 0
    else:  # in regular consensus all first coords are directly added to allCoords
        N0 = coords[0].shape[0]
        firstInput = 1

    # initializing arrays
    Ninputs = len(coords)
    Ncoords = sum([x.shape[0] for x in coords])
    allCoords = np.zeros([Ncoords, 2])
    votes = np.zeros(Ncoords)

    inAllMicrographs = consensus <= 0 or consensus >= Ninputs

    # if nothing in the first and it should be in all, nothing to do
    if (not all([coords[idx].shape[0] for idx in range(Ninputs)])
            and inAllMicrographs):
        print("Returning from worker: doing AND consensus and, at least, one "
              "picker is empty for this micrograph (%s)." % posFn)
        return

    # Add all the first coordinates to 'allCoords' and 'votes' lists
    if N0 > 0:
        allCoords[0:N0, :] = coords[0]
        votes[0:N0] = 1

    # Add the rest of coordinates to 'allCoords' and 'votes' lists
    Ncurrent = N0
    for n in range(firstInput, Ninputs):
        for coord in coords[n]:
            if Ncurrent > 0:
                dist = np.sum((coord - allCoords[0:Ncurrent]) ** 2, axis=1)
                imin = np.argmin(dist)
                if sqrt(dist[imin]) < consensusRadius:
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
    if consensus <= 0 or consensus > Ninputs:
        consensus = Ninputs
    elif not isinstance(consensus, int):
        consensus = consensus.get()

    if mode==PICK_MODE_LARGER:
        consensusCoords = allCoords[votes >= consensus, :]
    else:
        consensusCoords = allCoords[votes == consensus, :]

    try:
        if jaccFn:
            jaccardIdx = float(len(consensusCoords)) / (
                    float(len(allCoords)) / Ninputs)
            # COSS: Possible problem with concurrent writes
            with open(jaccFn, "a") as fhJaccard:
                fhJaccard.write("%s \t %f\n" % (posFn, jaccardIdx))
    except Exception as exc:
        print("Some error occurred during Jaccard index calculation or "
              "writing it's file. Maybe a concurrence issue:\n%s" % exc)
    # Write the consensus file only if there
    # are some coordinates (size > 0)
    if consensusCoords.size:
        np.savetxt(posFn, consensusCoords)


def getReadyMovies(MovieSet):
    movieSet = SetOfMovies(filename=MovieSet.getFileName())
    movieSet._xmippMd = String()
    movieSet.loadAllProperties()
    setClosed = movieSet.isStreamClosed()
    movieSet.close()
    currentAlignedMovies = {movieAgg["_movieId"] for movieAgg in
                       movieSet.aggregate(["MAX"], "_movieId", ["_movieId"])}
    return currentAlignedMovies, setClosed
