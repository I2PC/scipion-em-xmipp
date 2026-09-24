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
import os
import enum
from math import sqrt
import numpy as np

from pyworkflow.object import Set, String, Pointer
import pyworkflow.protocol.params as params
from pwem.protocols import ProtParticlePicking
from pyworkflow.protocol.constants import *
from pwem.objects import SetOfCoordinates, Coordinate
from pyworkflow.utils import getFiles, removeBaseExt, moveFile
from pyworkflow import UPDATED, PROD


PICK_MODE_LARGER = 0
PICK_MODE_EQUAL = 1


class ProtPickingConsensusOutput(enum.Enum):
    """ Possible outputs for particle picking protocols

    AI Generated

    ## Overview

    The Picking Consensus protocol combines several particle-picking results and
    keeps coordinates that are supported by a user-defined level of agreement.

    Different particle pickers, or the same picker with different parameters, may
    select slightly different coordinates. Some picks may be reliable because
    several methods select the same particle. Other picks may be method-specific
    and may correspond to false positives, contaminants, weak particles, or
    borderline cases.

    This protocol compares multiple coordinate sets micrograph by micrograph. A
    coordinate is considered to represent the same particle as another coordinate
    if the two positions are within a user-defined radius. The protocol then counts
    how many input pickers support each candidate particle and writes a consensus
    coordinate set.

    The protocol can behave strictly, keeping only particles selected by all
    pickers, or flexibly, keeping particles selected by one or more pickers. This
    makes it useful both for cleaning particle sets and for merging complementary
    picking strategies.

    ## Inputs and General Workflow

    The input is a list of coordinate sets.

    Each input coordinate set should correspond to particle picks on the same
    micrographs, although the protocol can also handle streaming situations where
    different coordinate sets become available progressively.

    For each micrograph, the protocol collects the coordinates from all input
    coordinate sets. If the input coordinate sets have different sampling rates,
    their coordinates are rescaled to the sampling rate of the first coordinate
    set.

    The protocol then groups nearby coordinates within the selected consensus
    radius. Each group receives a number of votes corresponding to how many input
    coordinate sets contributed a coordinate to that group.

    Finally, the protocol keeps only the coordinate groups that satisfy the
    selected consensus rule and writes them as a new Scipion coordinate set.

    ## Input Coordinates

    The **Input coordinates** parameter contains the coordinate sets to compare.

    These coordinate sets may come from different particle-picking protocols,
    different picking algorithms, different parameter settings, or manual and
    automatic picking runs.

    The first input coordinate set is used as the main reference for micrograph
    information, box size, and sampling-rate scaling. The output coordinate set is
    linked to the micrographs of this first input.

    For best results, all input coordinate sets should refer to the same
    micrographs or to matching micrograph sets. If the input pickers were run on
    different subsets of micrographs, the protocol processes only the micrographs
    that are available according to its streaming and readiness logic.

    ## Consensus Radius

    The **Radius** parameter defines the distance, in pixels, within which two
    coordinates are considered to correspond to the same particle.

    For example, if the radius is 10 pixels, two picks closer than 10 pixels are
    merged into a single candidate particle. The merged coordinate is updated as a
    weighted average of the contributing positions.

    A small radius is strict. It requires different pickers to place coordinates
    very close to each other. This may reduce false matches but can also fail to
    merge picks that clearly correspond to the same particle but are slightly
    shifted.

    A large radius is more permissive. It can merge picks with larger positional
    differences, but if it is too large it may incorrectly merge nearby distinct
    particles.

    The radius should be chosen according to particle size, picking precision, and
    typical particle separation. It should normally be smaller than the particle
    diameter.

    ## Consensus Number

    The **Consensus** parameter defines how many input coordinate sets must support
    a particle for it to be included in the output.

    If the value is set to **-1**, the protocol requires support from all input
    coordinate sets. This corresponds to an AND operation. It is the strictest
    mode.

    If the value is set to **1**, a coordinate is accepted if it is selected by at
    least one input coordinate set. This corresponds to an OR operation. It is the
    most permissive mode.

    Intermediate values allow more flexible behavior. For example, if four picking
    methods are provided and the consensus value is 3, a particle is accepted if
    three or more pickers agree on it, depending on the selected consensus mode.

    This parameter controls the balance between precision and recall. Strict
    consensus reduces false positives but may discard true particles missed by some
    pickers. Flexible consensus recovers more particles but may retain more false
    positives.

    ## Consensus Mode

    The **Consensus mode** parameter controls how the number of votes is compared
    with the consensus number.

    If the mode is **>=**, a particle is accepted if it has at least the selected
    number of votes. This is the usual and more flexible behavior.

    If the mode is **=**, a particle is accepted only if it has exactly the selected
    number of votes. This is more specialized and can be useful when the user wants
    to isolate particles picked by a specific number of methods.

    For most workflows, the **>=** mode is recommended.

    ## Coordinate Merging

    When coordinates from different pickers fall within the consensus radius, the
    protocol merges them into a single candidate coordinate.

    The merged coordinate is computed progressively as an average of the positions
    that vote for the same particle. This means that the output coordinate may not
    be exactly equal to any individual input coordinate, but instead represents the
    central consensus position.

    This is useful when different pickers identify the same particle with slightly
    different centers.

    ## AND and OR Interpretations

    The protocol can be understood in terms of familiar logical operations.

    With **Consensus = -1**, the protocol behaves like an AND operation: a particle
    is accepted only if all input pickers selected it.

    With **Consensus = 1** and mode **>=**, the protocol behaves like an OR
    operation: a particle is accepted if at least one picker selected it.

    Values between these extremes implement majority-like rules. For example, with
    three input pickers and Consensus = 2, the protocol keeps particles selected by
    at least two pickers.

    These modes allow users to tune how conservative the final coordinate set
    should be.

    ## Jaccard Index File

    During consensus calculation, the protocol writes a Jaccard-like agreement
    value for each processed micrograph.

    This value summarizes the degree of overlap between the input coordinate sets
    for that micrograph. It can help identify micrographs where pickers agree well
    and micrographs where the picking results are inconsistent.

    Low agreement may indicate poor micrograph quality, difficult particle
    appearance, contaminants, low contrast, or strong differences between picking
    methods.

    This file is mainly diagnostic and should be interpreted as a guide to picking
    agreement rather than as an absolute quality measure.

    ## Output Consensus Coordinates

    The main output is **consensusCoordinates**, a new SetOfCoordinates.

    This output contains the coordinates that satisfy the selected consensus rule.
    It is linked to the micrographs of the first input coordinate set and uses the
    box size from that same set.

    The consensus coordinate set can be passed to particle extraction protocols in
    the same way as any other coordinate set.

    Depending on the selected consensus parameters, the output may be smaller,
    similar in size, or larger than the individual input coordinate sets.

    ## Streaming Behavior

    The protocol supports streaming coordinate inputs.

    In streaming mode, the protocol waits until coordinates for a micrograph are
    available from the relevant input coordinate sets before computing consensus
    for that micrograph. When all input streams are closed, it processes any
    remaining available micrographs and closes the output stream.

    This makes the protocol suitable for automated pipelines where particle
    picking results are produced progressively by different pickers.

    ## Practical Recommendations

    Use this protocol when you have several picking results and want to combine
    them in a controlled way.

    For a conservative cleaned set, use Consensus = -1, requiring all pickers to
    agree. This can reduce false positives but may lose true particles.

    For a permissive merged set, use Consensus = 1 with mode >=. This keeps any
    particle selected by at least one picker, but later cleaning by particle
    screening or 2D classification becomes more important.

    For a balanced strategy, use a majority rule, such as requiring 2 out of 3
    pickers or 3 out of 4 pickers.

    Choose the consensus radius carefully. It should reflect expected picking
    uncertainty but should not be so large that nearby particles are merged.

    Inspect the output coordinates visually on representative micrographs. This is
    especially important when combining pickers with very different behavior.

    Use the Jaccard agreement information to identify micrographs where picking
    methods disagree strongly.

    ## Final Perspective

    Picking Consensus is a coordinate-combination and quality-control protocol. It
    does not pick particles directly. Instead, it integrates the results of several
    picking methods or parameter settings and produces a coordinate set based on
    their agreement.

    For biological users, this is useful because no single picker is perfect.
    Consensus can help reduce method-specific false positives, recover robust
    particles, and create more reliable coordinate sets for extraction.

    The protocol is most effective when the input pickers are complementary and
    when the consensus radius and voting threshold are chosen according to the
    particle size and the desired strictness of the workflow.
    """
    consensusCoordinates = SetOfCoordinates


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
    _devStatus = PROD
    _possibleOutputs = ProtPickingConsensusOutput
    outputName = ProtPickingConsensusOutput.consensusCoordinates.name
    FN_PREFIX = 'consensusCoords_'

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = STEPS_SERIAL

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinates', params.MultiPointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates", important=True,
                      help='Select the set of coordinates to compare')
        form.addParam('consensusRadius', params.IntParam, default=10,
                      label="Radius",  allowsPointers=True, allowsNull=True,
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
        self.checkedMics = set()   # those mics ready to be processed (micId)
        self.processedMics = set() # those mics already processed (micId)
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
                                              micrograph.getObjId(),
                                              micrograph.getFileName(),
                                              prerequisites=[])
            deps.append(stepId)
        return deps

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        # If continue from an stopped run, don't repeat what is done
        if not self.checkedMics:
            for fn in getFiles(self._getExtraPath()):
                fn = removeBaseExt(fn)
                if fn.startswith(self.FN_PREFIX):
                    self.checkedMics.update([self.getMicId(fn)])
                    self.processedMics.update([self.getMicId(fn)])

        streamClosed = []
        readyMics = None
        allMics = set()
        for coordSet in self.inputCoordinates:
            currentPickMics, isSetClosed = getReadyMics(coordSet.get())
            streamClosed.append(isSetClosed)
            if not readyMics:  # first time
                readyMics = currentPickMics
            else:  # available mics are those ready for all pickers
                readyMics.intersection_update(currentPickMics)
            allMics = allMics.union(currentPickMics)

        mainPickMics = getReadyMics(self.inputCoordinates[0].get())[0]
        secondaryPickMics = getReadyMics(self.inputCoordinates[1].get())[0]
        if mainPickMics != secondaryPickMics:
            allMics = mainPickMics.intersection(secondaryPickMics)

        self.streamClosed = all(streamClosed)
        if self.streamClosed:
            # for non streaming do all and in the last iteration of streaming do the rest
            newMicIds = allMics.difference(self.checkedMics)
        else:  # for streaming processing, only go for the ready mics in all pickers
            if readyMics is not None:
                newMicIds = readyMics.difference(self.checkedMics)

        if newMicIds:
            self.checkedMics.update(newMicIds)

            inMics = self.getMainInput().getMicrographs()
            newMics = [inMics[micId].clone() for micId in newMicIds]

            fDeps = self.insertNewCoorsSteps(newMics)
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
            outSet = self._loadOutputSet(SetOfCoordinates, 'coordinates.sqlite')

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

        inMicsPointer = self.getMainInput().getMicrographs(asPointer=True)
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
        return self.inputCoordinates[0].get()

    @classmethod
    def getMicId(self, fn):
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


def getReadyMics(coordSet):
    coorSet = SetOfCoordinates(filename=coordSet.getFileName())
    coorSet._xmippMd = String()
    coorSet.loadAllProperties()
    setClosed = coorSet.isStreamClosed()
    coorSet.close()
    currentPickMics = {micAgg["_micId"] for micAgg in
                       coordSet.aggregate(["MAX"], "_micId", ["_micId"])}
    return currentPickMics, setClosed
