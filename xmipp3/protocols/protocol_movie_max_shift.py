# **************************************************************************
# *
# * Authors:    David Maluenda    (dmaluenda@cnb.csic.es)
# *             Daniel Marchan (da.marchan@cnb.csic.es) REFACTOR
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
import os
from os.path import exists
import numpy as np
import copy
import time

from pyworkflow import VERSION_3_0
from pyworkflow import UPDATED, PROD
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Set
from pyworkflow.protocol import ProtStreamingBase
from pyworkflow.protocol.constants import MODE_RESUME
from pyworkflow.protocol.params import PointerParam
from pyworkflow.utils.properties import Message

from pwem.protocols import ProtProcessMovies
from pwem.objects import SetOfMicrographs, SetOfMovies

from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase


OUTPUT_MICS = "outputMicrographs"
OUTPUT_MICS_DISCARDED = "outputMicrographsDiscarded"
OUTPUT_MOVIES = "outputMovies"
OUTPUT_MOVIES_DISCARDED = "outputMoviesDiscarded"
OUTPUT_MICS_DW = "outputMicrographsDoseWeighted"
OUTPUT_MICS_DW_DISCARDED = "outputMicrographsDoseWeightedDiscarded"


class XmippProtMovieMaxShift(XmippStreamingBase, ProtStreamingBase, ProtProcessMovies):
    """
    Protocol to make an automatic rejection of those movies whose
    frames move more than a given threshold.
        Rejection criteria:
            - *by frame*: Rejects movies with drifts between frames
                              bigger than a certain maximum.
            - *by whole movie*: Rejects movies with a total travel
                                         bigger than a certain maximum.
            - *by frame and movie*: Rejects movies if both conditions
                                                above are met.
            - *by frame or movie*: Rejects movies if one of the conditions
                                             above are met.

    AI Generated

    ## Overview

    The Movie Maxshift protocol automatically separates aligned movies according
    to the amount of motion estimated during movie alignment.

    During cryo-EM movie acquisition, particles and ice may move because of
    beam-induced motion, mechanical instabilities, charging, or other acquisition
    effects. Movie-alignment protocols estimate this motion as a sequence of shifts
    between frames. If a movie shows excessive motion, it may produce a blurred or
    unreliable averaged micrograph even after alignment.

    This protocol evaluates the alignment shifts already stored in a set of aligned
    movies. It can reject movies based on large jumps between consecutive frames,
    large total accumulated motion across the whole movie, or a combination of both
    criteria.

    The output consists of accepted and discarded movie sets. When associated
    micrographs are available from the previous movie-alignment protocol, the
    protocol also produces accepted and discarded micrograph sets.

    ## Inputs and General Workflow

    The main input is a set of previously aligned movies.

    The protocol does not perform movie alignment itself. Instead, it reads the
    shift trajectory already associated with each movie. For each movie, it
    computes motion statistics from the X and Y shifts and applies the selected
    rejection rule.

    Movies that satisfy the selected motion-quality criteria are placed in the
    accepted output set. Movies that exceed the selected thresholds are placed in
    the discarded output set.

    If the input aligned movies are associated with output micrographs from the
    previous alignment protocol, those micrographs are also separated into accepted
    and discarded sets using the same movie decisions.

    ## Input Movies

    The **Input movies** parameter must point to a set of aligned movies.

    This is required because the protocol evaluates the shifts estimated during
    movie alignment. If the movies do not contain alignment information, the
    protocol cannot compute frame-to-frame motion or total movie travel.

    Typical inputs are movie sets produced by movie-alignment protocols such as
    FlexAlign, MotionCorr-like workflows, or other Scipion-compatible alignment
    steps.

    ## Rejection Type

    The **Rejection type** parameter defines how the protocol decides whether a
    movie should be discarded.

    There are four possible criteria:

    **By frame** rejects movies if the maximum displacement between two consecutive
    frames is larger than the selected frame-shift threshold.

    **By whole movie** rejects movies if the total accumulated trajectory length
    over the full movie is larger than the selected movie-shift threshold.

    **By frame and movie** rejects movies only if both conditions are met: the movie
    has a large frame-to-frame jump and a large total displacement.

    **By frame or movie** rejects movies if either condition is met. This is the
    most conservative option and is the default behavior.

    The best choice depends on how strict the user wants the quality-control step
    to be. The “or” criterion removes any movie that violates either motion limit,
    whereas the “and” criterion removes only movies that are problematic by both
    measures.

    ## Maximum Frame Shift

    The **Max. frame shift** parameter defines the largest allowed displacement
    between consecutive frames.

    The protocol computes the difference between the X and Y shifts of consecutive
    frames and converts this displacement into angstroms using the movie sampling
    rate. If the largest frame-to-frame displacement exceeds the threshold, the
    movie satisfies the frame-rejection condition.

    Large frame-to-frame jumps may indicate unstable alignment, sudden specimen
    movement, stage instability, charging events, or problematic frames. Such jumps
    can be especially damaging because they may not be well corrected by smooth
    motion models.

    This threshold is used when the rejection type includes the frame criterion.

    ## Maximum Movie Shift

    The **Max. movie shift** parameter defines the largest allowed total travel
    across the movie.

    The protocol computes the distance between consecutive points of the shift
    trajectory and sums these distances along the movie. This gives the total path
    length traveled by the aligned frames, expressed in angstroms.

    A movie may have no single extreme jump but still show a large accumulated
    motion. Such movies may be less reliable because the specimen has moved
    substantially during exposure.

    This threshold is used when the rejection type includes the whole-movie
    criterion.

    ## Frame-to-Frame Motion Versus Total Motion

    The two rejection criteria capture different types of problems.

    Frame-to-frame motion detects sudden jumps. A movie with one abrupt movement
    may be rejected even if its total motion is not very large.

    Total movie motion detects accumulated drift. A movie may move gradually and
    smoothly, but if the accumulated travel is very large, the final correction may
    still be less reliable.

    Using both criteria gives a more complete assessment of movie behavior. The
    default “by frame or movie” option is strict because it discards movies that
    fail either test.

    ## Accepted Movies

    The **outputMovies** set contains the movies that pass the selected motion
    criteria.

    These movies keep their original alignment information. They can be used for
    downstream processing or for generating accepted micrographs when available.

    The accepted set should be understood as the subset whose estimated motion is
    within the user-defined limits.

    ## Discarded Movies

    The **outputMoviesDiscarded** set contains movies that exceed the selected
    motion thresholds.

    A discarded movie is not necessarily completely unusable, but it has motion
    behavior that the user decided to consider risky. Users may inspect discarded
    movies to decide whether the thresholds are too strict or whether the discarded
    movies show real acquisition problems.

    This output is useful for troubleshooting, quality-control reports, and
    understanding how much data is lost because of excessive motion.

    ## Associated Micrograph Outputs

    If the input movies come from a previous protocol that also produced
    micrographs, Movie Maxshift can separate those micrographs into accepted and
    discarded sets using the same movie decisions.

    Depending on the previous protocol, these may be regular averaged micrographs
    or dose-weighted micrographs.

    This is useful because downstream processing often continues from
    micrographs rather than movies. By propagating the movie-level decision to the
    micrographs, the protocol allows the user to continue the workflow only with
    motion-acceptable micrographs.

    If no associated micrograph set is found, the protocol still produces the
    movie outputs, but no micrograph outputs are created.

    ## Streaming Behavior

    The protocol supports streaming input.

    As new aligned movies appear, their shifts are evaluated and the accepted and
    discarded outputs are updated. The output sets remain open while the input
    stream is open and are closed when all movies have been processed.

    This makes the protocol useful for online quality control during data
    acquisition or automated processing.

    ## Practical Recommendations

    Use this protocol after movie alignment, not before. It requires alignment
    shifts stored in the input movies.

    Start with the default rejection type, **by frame or movie**, if you want a
    conservative quality-control filter.

    Use **by frame** when you mainly want to detect abrupt jumps between frames.

    Use **by whole movie** when you mainly want to detect large accumulated drift.

    Use **by frame and movie** when you want a more permissive rule that discards
    only movies with both sudden jumps and large total travel.

    Inspect a subset of discarded movies and their corresponding micrographs. This
    helps determine whether the thresholds are appropriate for the dataset.

    Be careful with very strict thresholds. They may remove useful data,
    especially in datasets with strong but well-corrected beam-induced motion.

    Remember that this protocol evaluates the magnitude of estimated motion. It
    does not directly evaluate CTF quality, particle quality, ice contamination, or
    final reconstruction performance.

    ## Final Perspective

    Movie Maxshift is a movie-alignment quality-control protocol. It uses the
    motion trajectory estimated during alignment to identify movies with excessive
    frame-to-frame jumps or excessive accumulated drift.

    For biological users and facility workflows, this provides a simple and
    interpretable way to remove movies that may compromise downstream processing.
    It is especially useful in automated pipelines, where problematic movies should
    be detected and separated before CTF estimation, particle picking, and
    reconstruction.

    The protocol is most effective when thresholds are chosen in relation to the
    expected motion behavior of the dataset and when accepted/discarded results are
    checked visually during workflow setup.
    """
    _label = 'movie maxshift'
    _devStatus = PROD
    _lastUpdateVersion = VERSION_3_0
    _possibleOutputs = {OUTPUT_MICS: SetOfMicrographs,
                        OUTPUT_MICS_DW: SetOfMicrographs,
                        OUTPUT_MICS_DISCARDED: SetOfMicrographs,
                        OUTPUT_MICS_DW_DISCARDED: SetOfMicrographs,
                        OUTPUT_MOVIES: SetOfMovies,
                        OUTPUT_MOVIES_DISCARDED: SetOfMovies
                        }
    
    REJ_TYPES = ['by frame', 'by whole movie', 'by frame and movie', 
                 'by frame or movie']
    REJ_FRAME = 0
    REJ_MOVIE = 1
    REJ_AND = 2
    REJ_OR = 3

    MOVIE_VISIBILITY_MAX_ATTEMPTS = 3
    MOVIE_VISIBILITY_RETRY_DELAY = 1  # seconds

    def __init__(self, **args):
        ProtProcessMovies.__init__(self, **args)

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputMovies', PointerParam, important=True,
                      label=Message.LABEL_INPUT_MOVS,
                      pointerClass='SetOfMovies', 
                      help='Select a set of previously aligned Movies.')

        form.addParam('rejType', params.EnumParam, choices=self.REJ_TYPES,
                      label='Rejection type', default=self.REJ_OR,
                      help='Rejection criteria:\n'
                           ' - *by frame*: Rejects movies with drifts between '
                                    'frames bigger than a certain maximum.\n'
                           ' - *by whole movie*: Rejects movies with a total '
                                    'travel bigger than a certain maximmum.\n'
                           ' - *by frame and movie*: Rejects movies if both '
                                    'conditions above are met.\n'
                           ' - *by frame or movie*: Rejects movies if one of '
                                    'the conditions above are met.')

        form.addParam('maxFrameShift', params.FloatParam, default=10,
                       label='Max. frame shift (A)',
                       condition='rejType==%s or rejType==%s or rejType==%s'
                                  % (self.REJ_FRAME, self.REJ_AND, self.REJ_OR),
                       help='Maximum drift between consecutive frames '
                            'to evaluate the frame condition.')
        form.addParam('maxMovieShift', params.FloatParam, default=45,
                       label='Max. movie shift (A)',
                       condition='rejType==%s or rejType==%s or rejType==%s'
                                  % (self.REJ_MOVIE, self.REJ_AND, self.REJ_OR),
                       help='Maximum total travel to evaluate the whole movie '
                            'condition.')

        self._defineStreamingParams(form)
        form.addParallelSection(threads=3, mpi=1)

    # --------------------------- INSERT steps functions ------------------------
    def stepsGeneratorStep(self) -> None:
        self.initializeStep()
        self.newDeps = []

        while not getattr(self, 'finished', False):
            self._checkNewInput()
            self._checkNewOutput()

            if getattr(self, 'finished', False):
                break

            self._streamingSleepOnWait()

        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=self.newDeps, needsGPU=False)

    def initializeStep(self):
        inputMovies = self.inputMovies.get()

        self.samplingRate = inputMovies.getSamplingRate()
        self.insertedIds = []
        self._lastInputId = 0
        self.acceptedIds = []
        self.discardedIds = []
        self.isStreamClosed = inputMovies.isStreamClosed()
        self.alreadyLoad = False

    def createOutputStep(self):
        self._closeOutputSet()

    def _loadMicAssociatedInputSet(self):
        """Load the micrograph Set associated with the input movies."""
        inputMicName = getattr(self, 'inputMicName', None)
        if not inputMicName:
            return None

        parentProt = self.getMapper().getParent(self.inputMovies.get())
        if parentProt is None:
            return None

        micSet = getattr(parentProt, inputMicName, None)
        if micSet is None:
            return None

        micSet.loadAllProperties()
        micSet.close()

        return micSet

    def _checkNewInput(self):
        inputSet = self._loadLogicalSet(self.inputMovies)

        try:
            newIds, self._lastInputId = self._discoverIdsAfter(
                inputSet,
                self._lastInputId,
            )

            producerClosed = inputSet.isStreamClosed()

            newIds, terminalConsistent = (
                self._reconcileClosedStreamIds(
                    inputSet,
                    newIds,
                    self.insertedIds,
                    producerClosed,
                )
            )

            self.isStreamClosed = (
                producerClosed
                and terminalConsistent
            )
        finally:
            inputSet.close()

        if (
            getattr(
                self,
                '_originalRunMode',
                self.runMode.get(),
            ) == MODE_RESUME
            and not self.insertedIds
        ):
            doneIds, _, _, _ = self._getAllDoneIds()
            doneIds = list(doneIds)

            skipIds = sorted(
                set(newIds).intersection(doneIds)
            )
            newIds = sorted(
                set(newIds).difference(doneIds)
            )

            self.info(
                "Skipping Mics with ID: %s, seems to be done"
                % skipIds
            )

            self.insertedIds = doneIds

        if newIds:
            fDeps = self._insertNewMoviesSteps(newIds)
            self.newDeps.extend(fDeps)
            self.updateSteps()

    def _insertNewMoviesSteps(self, newIds):
        """ Insert the processMovieStep for a given movie.
        """
        if not self.alreadyLoad or self.inputMics is None:
            # Streaming parents may publish outputMovies before the sibling
            # micrograph output exists. Keep retrying discovery until mics
            # are actually available.
            self.setInputMics()

        # Insert the selection/rejection step process
        deps = []
        stepId = self._insertFunctionStep(self._evaluateMovieAlign, newIds,  needsGPU=False,
                                          prerequisites=[])
        deps.append(stepId)

        for movId in newIds:
            self.insertedIds.append(movId)

        return deps

    def _evaluateMovieAlign(self, movIds):
        """Fill the accepted or rejected lists for the given movies."""
        sampling = self.samplingRate
        inputMovies = self._loadLogicalSet(self.inputMovies)

        try:
            for movieId in movIds:
                movie = None
                for attempt in range(self.MOVIE_VISIBILITY_MAX_ATTEMPTS):
                    if attempt > 0:
                        time.sleep(self.MOVIE_VISIBILITY_RETRY_DELAY)
                        inputMovies.close()
                        inputMovies = self._loadLogicalSet(self.inputMovies)

                    # Set.getItem raises rather than returning None for a
                    # row it cannot find, so check membership first - a
                    # movie just discovered via the id watermark may not
                    # be selectable yet under a PostgreSQL-backed
                    # compatibility bridge.
                    if movieId in inputMovies:
                        movie = inputMovies.getItem("id", movieId).clone()
                        break

                if movie is None:
                    self.error(
                        "Movie with id %d never became visible in the "
                        "input Set after %d attempts; discarding it."
                        % (movieId, self.MOVIE_VISIBILITY_MAX_ATTEMPTS)
                    )
                    self.discardedIds.append(movieId)
                    continue

                try:
                    alignment = movie.getAlignment()
                    # getShifts() returns the absolute shifts from a reference.
                    shiftListX, shiftListY = alignment.getShifts()

                    rejectedByMovie = False
                    rejectedByFrame = False

                    if any(shiftListX) or any(shiftListY):
                        shiftArrayX = np.asarray(shiftListX)
                        shiftArrayY = np.asarray(shiftListY)

                        evalBoth = (
                            self.rejType == self.REJ_AND
                            or self.rejType == self.REJ_OR
                        )

                        if self.rejType == self.REJ_MOVIE or evalBoth:
                            deltaX = np.diff(shiftArrayX)
                            deltaY = np.diff(shiftArrayY)
                            stepDistances = np.sqrt(
                                deltaX ** 2 + deltaY ** 2
                            )
                            totalPath = (
                                np.sum(stepDistances) * sampling
                            )
                            rejectedByMovie = (
                                totalPath > self.maxMovieShift.get()
                            )

                        if self.rejType == self.REJ_FRAME or evalBoth:
                            frameShiftX = np.diff(shiftArrayX)
                            frameShiftY = np.diff(shiftArrayY)
                            frameShifts = np.sqrt(
                                frameShiftX ** 2
                                + frameShiftY ** 2
                            )
                            maxShiftM = (
                                np.max(frameShifts) * sampling
                            )
                            rejectedByFrame = (
                                maxShiftM > self.maxFrameShift.get()
                            )

                        if self.rejType == self.REJ_AND:
                            if rejectedByFrame and rejectedByMovie:
                                self.discardedIds.append(movieId)
                            else:
                                self.acceptedIds.append(movieId)
                        else:
                            if rejectedByFrame or rejectedByMovie:
                                self.discardedIds.append(movieId)
                            else:
                                self.acceptedIds.append(movieId)
                    else:
                        self.acceptedIds.append(movieId)
                except Exception as e:
                    # A single movie with corrupted/missing alignment data
                    # must not fail the whole batch step (and hence the
                    # whole protocol) - discard just this movie and keep
                    # evaluating the rest.
                    self.error(
                        "Movie with id %d failed while evaluating its "
                        "alignment shifts (%s); discarding it."
                        % (movieId, e)
                    )
                    self.discardedIds.append(movieId)
        finally:
            inputMovies.close()

    def _checkNewOutput(self):
        """ Check for already selected Movies and update the output set. """
        # The producer may expose outputMovies before its sibling micrograph
        # output. Keep trying discovery even when no new movie batch arrives.
        if (getattr(self, 'inputMics', None) is None
                and self.inputMovies.get() is not None):
            self.setInputMics()

        # load if first time in order to make dataSets relations
        _, _, doneListAccepted, doneListDiscarded = self._getAllDoneIds()
        # Check for newly done items
        acceptedIds = copy.deepcopy(self.acceptedIds)
        discardedIds = copy.deepcopy(self.discardedIds)

        newDoneAccepted = [movId for movId in acceptedIds
                           if movId not in doneListAccepted]
        newDoneDiscarded = [movId for movId in discardedIds
                            if movId not in doneListDiscarded]

        # Movies may be persisted before their sibling micrographs become
        # visible. Track only the missing sibling IDs and retry those IDs
        # incrementally instead of rescanning the whole micrograph Set.
        backfillAccepted = []
        backfillDiscarded = []

        if (
            getattr(self, 'inputMics', None) is not None
            and self.outMicName
        ):
            if not hasattr(self, '_pendingAcceptedMicIds'):
                acceptedMicIds = self._getKnownPersistedOutputIds(
                    self.outMicName,
                )
                discardedMicIds = self._getKnownPersistedOutputIds(
                    self.outMicName + 'Discarded',
                )

                self._pendingAcceptedMicIds = set(
                    doneListAccepted
                ).difference(acceptedMicIds)
                self._pendingDiscardedMicIds = set(
                    doneListDiscarded
                ).difference(discardedMicIds)

            self._pendingAcceptedMicIds.update(
                newDoneAccepted
            )
            self._pendingDiscardedMicIds.update(
                newDoneDiscarded
            )

            backfillAccepted = sorted(
                self._pendingAcceptedMicIds
            )
            backfillDiscarded = sorted(
                self._pendingDiscardedMicIds
            )

        firstTimeAccepted = len(doneListAccepted) == 0
        firstTimeDiscarded = len(doneListDiscarded) == 0

        allDone = len(doneListAccepted) + len(doneListDiscarded) + \
                  len(newDoneAccepted) + len(newDoneDiscarded)
        maxMicSize = self.inputMovies.get().getSize()
        # We have finished when there is not more input movies
        # (stream closed), the number of processed movies is equal
        # to the number of inputs, and no sibling micrograph is still
        # awaiting backfill - otherwise it would never be retried once
        # finished latches True and polling stops for good.
        self.finished = (
            self.isStreamClosed
            and allDone == maxMicSize
            and not backfillAccepted
            and not backfillDiscarded
        )
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        # Some checks to debug
        self.debug('_checkNewOutput: ')
        self.debug('   newDoneAccepted: %d.' % len(newDoneAccepted))
        self.debug('   newDoneDiscarded: %d,' % len(newDoneDiscarded))

        if (not self.finished
                and not newDoneDiscarded
                and not newDoneAccepted
                and not backfillAccepted
                and not backfillDiscarded):
            # Nothing new to publish, including late sibling micrographs.
            return

        def fillOutput(newDoneList, firstTime, AccOrDisc='Accepted'):
            """ A single function is used to fill the two sets (Accepted/Rej.)
            """
            suffix1 = '' if self.outMicName == OUTPUT_MICS else '_dose-weighted'
            suffix = 'Discarded' if AccOrDisc=='Discarded' else ''

            enable = False if AccOrDisc=='Discarded' else True
                
            movieSet = self._loadOutputSet(SetOfMovies, 'movies%s.sqlite' % suffix)
            micOutputName = self.outMicName + suffix if self.outMicName else None
            firstTimeMics = (
                micOutputName is not None
                and getattr(self, micOutputName, None) is None
            )
            micsSet = self._loadOutputSet(SetOfMicrographs,'micrographs%s%s.sqlite'%(suffix1, suffix))
            print('micrographs%s%s.sqlite'%(suffix1, suffix))

            def tryToAppend(outSet, micOut):
                """ When micrograph is very big, sometimes it's not ready to be read
                Then we will wait for it up to a minute in 6 time-growing tries. 
                Returns True if fails! """
                if micOut is None:
                    print('Mic/Movie is None do not introduce')
                else:
                    micOut.setEnabled(enable)
                    outSet.append(micOut)

            inputMovies = self._loadLogicalSet(self.inputMovies)
            inputMics = (
                self._loadMicAssociatedInputSet()
                if self.inputMics is not None
                else None
            )

            movieOutputName = OUTPUT_MOVIES + suffix
            movieSetIds = self._getKnownPersistedOutputIds(
                movieOutputName,
            )

            micsSetIds = (
                self._getKnownPersistedOutputIds(
                    micOutputName,
                )
                if micOutputName is not None
                else set()
            )

            pendingMicIds = (
                getattr(self, '_pendingDiscardedMicIds', set())
                if AccOrDisc == 'Discarded'
                else getattr(self, '_pendingAcceptedMicIds', set())
            )

            publishedMovieIds = []
            publishedMicIds = []

            for movieId in newDoneList:
                if movieId not in movieSetIds:
                    # Set.getItem raises rather than returning None for a
                    # row it cannot find, so check membership first - a
                    # movie just marked done may not be selectable yet
                    # under a PostgreSQL-backed compatibility bridge.
                    if movieId in inputMovies:
                        movie = inputMovies.getItem(
                            "id",
                            movieId,
                        )
                        tryToAppend(
                            movieSet,
                            movie.clone(),
                        )
                        movieSetIds.add(movieId)
                        publishedMovieIds.append(movieId)

                if inputMics is None or micsSet is None:
                    continue

                if movieId in micsSetIds:
                    pendingMicIds.discard(movieId)
                    continue

                if movieId not in inputMics:
                    pendingMicIds.add(movieId)
                    self.info(
                        "Movie with id %d has not a micrograph "
                        "associated yet" % movieId
                    )
                    continue

                mic = inputMics.getItem(
                    "id",
                    movieId,
                )

                tryToAppend(
                    micsSet,
                    mic.clone(),
                )
                micsSetIds.add(movieId)
                publishedMicIds.append(movieId)
                pendingMicIds.discard(movieId)
            
            if movieSet.getSize() > 0:
                self._updateOutputSet(OUTPUT_MOVIES + suffix, movieSet,
                                      streamMode)
                self._markOutputIdsPersisted(
                    movieOutputName,
                    publishedMovieIds,
                )
                                      
            if self.inputMics is not None and micsSet.getSize() > 0:
                self._updateOutputSet(
                    self.outMicName + suffix,
                    micsSet,
                    streamMode,
                )
                self._markOutputIdsPersisted(
                    micOutputName,
                    publishedMicIds,
                )
            if firstTime and movieSet.getSize() > 0:
                self._defineTransformRelation(self.inputMovies.get(), movieSet)

            if (firstTimeMics
                    and self.inputMics is not None
                    and micsSet is not None
                    and micsSet.getSize() > 0):
                self._defineTransformRelation(self.inputMics, micsSet)
            
            inputMovies.close()
            movieSet.close()
            if self.inputMics is not None:
                micsSet.close()

        # We fill/update the output if there are something new or to close sets
        acceptedToFill = list(dict.fromkeys(newDoneAccepted + backfillAccepted))
        discardedToFill = list(dict.fromkeys(newDoneDiscarded + backfillDiscarded))

        if acceptedToFill:
            fillOutput(acceptedToFill, firstTimeAccepted, AccOrDisc='Accepted')
        # We fill/update the output for the discarded movies
        if discardedToFill:
            fillOutput(discardedToFill, firstTimeDiscarded, AccOrDisc='Discarded')

        # Unlock createOutputStep if finished all jobs
        if self.finished:  
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

        self._store()

    #--------------------------- UTILS functions -------------------------------
    def _getAllDoneIds(self):
        acceptedIds = sorted(
            self._getKnownPersistedOutputIds(
                OUTPUT_MOVIES,
            )
        )
        discardedIds = sorted(
            self._getKnownPersistedOutputIds(
                OUTPUT_MOVIES_DISCARDED,
            )
        )

        doneIds = acceptedIds + discardedIds
        sizeOutput = len(acceptedIds) + len(discardedIds)

        return (
            doneIds,
            sizeOutput,
            acceptedIds,
            discardedIds,
        )

    def _loadOutputSet(self, SetClass, baseName):
        """ Load the output set if it exists or create a new one based on the inputs.
        """
        outputNameByBaseName = {
            'movies.sqlite': OUTPUT_MOVIES,
            'moviesDiscarded.sqlite': OUTPUT_MOVIES_DISCARDED,
            'micrographs.sqlite': OUTPUT_MICS,
            'micrographsDiscarded.sqlite': OUTPUT_MICS_DISCARDED,
            'micrographs_dose-weighted.sqlite': OUTPUT_MICS_DW,
            'micrographs_dose-weightedDiscarded.sqlite': OUTPUT_MICS_DW_DISCARDED,
        }
        outputName = outputNameByBaseName.get(baseName)
        outputSet = getattr(self, outputName, None) if outputName else None

        if outputSet is not None:
            outputSet.enableAppend()
            return outputSet

        if SetClass == SetOfMicrographs:
            if self.inputMics is None:
                # if no mics to do, do nothing and exit
                return None
            inputSet = self.inputMics
        else:
            inputSet = self.inputMovies.get()

        setFile = self._getPath(baseName)
        print(setFile)

        if exists(setFile):
            outputSet = SetClass(filename=setFile)
            if len(outputSet) == 0:
                pwutils.path.cleanPath(setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

        outputSet.copyInfo(inputSet)

        return outputSet

    def setInputMics(self):
        """Resolve sibling micrographs produced with the input movies.

        Legacy movie-alignment protocols expose ``outputMicrographs`` /
        ``outputMicrographsDoseWeighted`` while New MotionCorr exposes
        ``micrographs`` / ``micrographsDW``. Keep the producer attribute
        name separate from the canonical MaxShift output name.
        """
        self.alreadyLoad = True
        parentProt = self.getMapper().getParent(self.inputMovies.get())
        self.inputMicName = None
        self.outMicName = None
        self.inputMics = None

        if parentProt is not None:
            candidates = (
                (OUTPUT_MICS, OUTPUT_MICS),
                ('micrographs', OUTPUT_MICS),
                (OUTPUT_MICS_DW, OUTPUT_MICS_DW),
                ('micrographsDW', OUTPUT_MICS_DW),
            )

            for inputName, outputName in candidates:
                micSet = getattr(parentProt, inputName, None)
                if micSet is not None:
                    self.inputMicName = inputName
                    self.outMicName = outputName
                    self.inputMics = micSet

            # Fallback for other movie-alignment protocols: if the
            # producer exposes exactly one SetOfMicrographs under an
            # unknown name, it is unambiguous and safe to consume it.
            # Do not guess when several micrograph outputs exist.
            if self.inputMics is None:
                genericMics = [
                    (name, output)
                    for name, output in parentProt.iterOutputAttributes()
                    if isinstance(output, SetOfMicrographs)
                ]

                if len(genericMics) == 1:
                    self.inputMicName, self.inputMics = genericMics[0]
                    self.outMicName = OUTPUT_MICS

        if self.inputMics is None:
            self.warning(
                "The input movies have no associated micrographs available yet."
            )

    # ---------------------- INFO functions ------------------------------------
    def _validate(self):
        errors = []
        if not self.inputMovies.get().getFirstItem().hasAlignment():
            errors.append('The _Input Movies_ must come from an alignment '
                          'protocol.')
        return errors

    def _summary(self):
        def getSize(outNameCondition):
            for outName, outObj in self.iterOutputAttributes():
                if outNameCondition(outName):
                    return outObj.getSize()
            return 0

        moviesAcc = getSize(lambda x: not x.endswith('Discarded'))
        outDisc = getSize(lambda x: x.endswith('Discarded'))

        summary = ['Movies processed: %d' % (moviesAcc+outDisc),
                   'Movies rejected: *%d*' % outDisc]

        return summary
