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

from pyworkflow.gui.plotter import Plotter
import numpy as np
from math import ceil
try:
    from itertools import izip
except ImportError:
    izip = zip
from pwem.objects import SetOfMovies, SetOfMicrographs, MovieAlignment, Image
from pyworkflow.object import Set
import pyworkflow.protocol.params as params
from pyworkflow.protocol import STEPS_PARALLEL, Protocol, ProtStreamingBase
from pwem.protocols import ProtAlignMovies
from pyworkflow.protocol.constants import MODE_RESUME
from xmipp3.convert import getScipionObj
from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase
from pwem.constants import ALIGN_NONE
from pyworkflow import BETA, UPDATED, NEW, PROD

ACCEPTED = 'Accepted'
DISCARDED = 'Discarded'


class XmippProtConsensusMovieAlignment(XmippStreamingBase, ProtStreamingBase, ProtAlignMovies, Protocol):
    """
    The protocol compares two sets of aligned movies (reference and secondary)
    to evaluate their alignment consistency. It calculates the correlation
    between shift trajectories, allowing for a minimum correlation threshold to
    be set. Movies with correlations below this threshold can be discarded. The
    protocol can also generate plots showing the trajectories and correlations
    for each movie. This helps in identifying and validating the quality of
    movie alignments based on consensus among different alignment runs.

    AI Generated

    ## Overview

    The Movie Alignment Consensus protocol compares two independent movie-alignment
    results and evaluates whether their estimated frame-shift trajectories are
    consistent.

    Movie alignment is one of the first critical steps in cryo-EM processing. If
    the estimated shifts are wrong or unstable, the final averaged micrograph may
    be blurred, and this can affect CTF estimation, particle picking,
    classification, and reconstruction. One way to assess the reliability of movie
    alignment is to compare two alignment runs produced by different methods,
    different parameters, or different protocols.

    This protocol takes one set of aligned movies as the reference and a second set
    as the comparison. For each movie present in both sets, it compares the global
    alignment trajectories. Movies whose trajectories disagree below a user-defined
    correlation threshold can be discarded.

    The protocol produces accepted and discarded sets of movies and micrographs.
    It can also generate trajectory plots to help users visually inspect the
    agreement between the two alignments.

    ## Inputs and General Workflow

    The protocol requires two input sets of aligned movies:

    - a reference set of aligned movies;
    - a secondary set of aligned movies.

    Both input sets must already contain movie-alignment information. The protocol
    does not align movies itself. Instead, it reads the frame shifts estimated by
    the two previous alignment protocols and compares them.

    For each movie found in both input sets, the protocol extracts the X and Y
    shift trajectories from the two alignments. It then computes several measures
    of agreement, including correlation, root mean square error, and maximum error.

    If the agreement is high enough, the movie is accepted. If the agreement is
    below the selected consensus threshold, the movie is discarded. The reference
    alignment is used for the accepted output movies and micrographs.

    ## Reference Aligned Movies

    The **Reference Aligned Movies** input defines the alignment that will be used
    as the main reference.

    The global shifts from this set are used as the reference trajectories. The
    accepted output movies preserve the alignment from this first input set.

    In practice, this input should usually be the alignment result that the user
    trusts most, or the result from the main movie-alignment protocol used in the
    workflow.

    The protocol also obtains the corresponding output micrographs from the
    protocol that generated this reference movie set, when available.

    ## Secondary Aligned Movies

    The **Secondary Aligned Movies** input provides the alignment trajectories to
    compare against the reference.

    This second set may come from another alignment program, another run of the
    same program with different parameters, or an alternative alignment strategy.

    The purpose of this second input is not to replace the reference alignment, but
    to test whether an independent alignment estimate agrees with it. Agreement
    between two independent runs increases confidence that the estimated motion is
    robust.

    ## Minimum Consensus Shifts Correlation

    The **Minimum consensus shifts correlation** parameter defines the threshold
    used to accept or discard movies.

    For each movie, the protocol compares the X and Y shift trajectories from the
    two alignment sets. It computes a correlation for X and a correlation for Y,
    and then uses the smaller of the two as the global consensus score.

    If this score is greater than or equal to the threshold, the movie is accepted.
    If it is below the threshold, the movie is discarded.

    A value close to 1 is strict and requires very strong agreement between the two
    alignment trajectories. A lower value is more permissive.

    If the value is set to -1, no movies are discarded based on the correlation
    threshold.

    This parameter should be chosen according to how conservative the user wants
    the quality-control step to be. A strict threshold may remove problematic
    movies, but it may also discard usable movies when the two methods produce
    slightly different but still acceptable trajectories.

    ## Minimum Range Shift to Apply Consensus

    The **Minimum range shift to apply consensus** parameter defines when the
    trajectory comparison should be considered meaningful.

    Some movies have very small estimated motion. In such cases, the shift
    trajectory may be almost flat. When shifts are very small, correlation values
    can become unstable or difficult to interpret, because there is little motion
    signal to compare.

    To avoid overinterpreting such cases, the protocol checks the range of the
    shift trajectory in X and Y. If the motion range is below the selected
    threshold, the movie is omitted from the strict consensus calculation and is
    accepted.

    This is useful because two nearly flat trajectories may have a poor numerical
    correlation even though both indicate essentially no significant motion.

    The threshold is expressed in pixels.

    ## Trajectory Comparison

    When the motion range is large enough to make comparison meaningful, the
    protocol compares the two shift trajectories.

    Before computing the final comparison, the secondary trajectory is transformed
    to best match the reference trajectory. This compensates for global differences
    in trajectory placement and focuses the comparison on the shape and consistency
    of the motion path.

    The protocol then computes:

    - correlation in X;
    - correlation in Y;
    - the minimum of the two correlations;
    - root mean square error;
    - maximum error.

    The minimum of the X and Y correlations is used as the main consensus score.
    This conservative choice means that a movie must show good agreement in both
    directions to pass the consensus test.

    ## Global Alignment Trajectory Plot

    The option **Global Alignment Trajectory Plot** makes the protocol generate a
    plot for each movie.

    The plot overlays the reference and secondary shift trajectories. It also
    stores the correlation values associated with the comparison.

    These plots are useful for visual inspection. They allow the user to see
    whether the two alignments describe the same motion path, whether one method
    introduces jumps, or whether the disagreement is limited to a small part of the
    movie.

    Generating plots may increase output size and processing time, especially for
    large datasets. It is most useful when setting up the workflow, debugging
    alignment problems, or preparing quality-control reports.

    ## Accepted Outputs

    The protocol produces accepted output sets when movies pass the consensus
    criteria.

    The accepted outputs include:

    - accepted aligned movies;
    - accepted micrographs.

    The accepted movies preserve the alignment information from the reference
    input. The accepted micrographs also receive additional metadata describing the
    alignment consensus, such as the trajectory correlation and error measures.

    These accepted outputs can be used for downstream processing, such as CTF
    estimation, particle picking, extraction, classification, and reconstruction.

    ## Discarded Outputs

    The protocol can also produce discarded output sets.

    These contain movies and micrographs whose alignment trajectories did not pass
    the consensus threshold. They are useful for inspection and troubleshooting.

    A discarded movie is not necessarily biologically bad in every sense. It means
    that the two alignment runs did not agree sufficiently according to the chosen
    threshold. The cause may be poor movie quality, excessive drift, low signal,
    bad gain correction, unstable alignment parameters, or differences between the
    two alignment methods.

    Users should inspect some discarded examples before deciding whether the
    threshold is appropriate.

    ## Streaming Behavior

    The protocol supports streaming workflows.

    As new aligned movies appear in the two input sets, the protocol checks which
    movie identifiers are present in both sets. It then compares only movies that
    are available in both inputs and updates the accepted and discarded outputs
    progressively.

    The output streams remain open until both input movie streams are closed and
    all common movies have been processed.

    This makes the protocol suitable for online or near-real-time quality control
    during data acquisition or automated processing.

    ## Interpreting the Consensus Score

    The consensus correlation measures whether two alignment trajectories have a
    similar shape.

    High correlation suggests that the two alignment methods agree about the
    direction and evolution of beam-induced motion. This increases confidence in
    the correction.

    Low correlation suggests disagreement. This may indicate that one or both
    alignments are unstable, that the movie has too little signal for reliable
    alignment, or that some preprocessing issue affects one of the runs.

    The root mean square error and maximum error provide complementary information
    about the magnitude of the disagreement. Two trajectories may be correlated but
    still differ by a noticeable amount, or they may have low correlation because
    the motion is very small.

    For this reason, consensus scores should be interpreted together with the
    motion range and, when available, trajectory plots.

    ## Practical Recommendations

    Use this protocol when you have two independent movie-alignment results and
    want to identify movies whose motion correction is robust.

    Choose the most trusted alignment as the reference input, because accepted
    outputs keep the reference alignment.

    Start with the default correlation threshold and inspect both accepted and
    discarded examples. Adjust the threshold if too many reasonable movies are
    discarded or if clearly problematic movies are accepted.

    Keep the minimum range-shift threshold enabled. It prevents movies with almost
    no motion from being unfairly rejected due to unstable correlation estimates.

    Enable trajectory plots when validating a new workflow or comparing alignment
    methods. Once the workflow is established, plots may be disabled to reduce
    output size.

    Remember that this protocol evaluates agreement between two alignment
    trajectories. It does not directly measure final reconstruction quality.
    Consensus is a strong quality-control signal, but it should be interpreted
    together with PSDs, CTF quality, particle classes, and reconstruction results.

    ## Final Perspective

    Movie Alignment Consensus is a quality-control protocol for motion correction.
    It does not perform movie alignment itself. Instead, it asks whether two
    independent alignment estimates agree for each movie.

    For biological users and facility workflows, this is useful because movie
    alignment errors can propagate through the entire cryo-EM pipeline. By keeping
    movies with consistent alignment trajectories and separating those with
    discrepant behavior, the protocol helps produce a cleaner and more reliable
    set of motion-corrected micrographs.

    The protocol is especially valuable when comparing alignment methods, tuning
    movie-alignment parameters, or building automated workflows where problematic
    movies should be detected early.
    """

    _label = 'movie alignment consensus'
    outputName = 'consensusAlignments'
    _devStatus = PROD

    def __init__(self, **args):
        ProtAlignMovies.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    def _defineParams(self, form):
        form.addSection(label='Input Consensus')
        form.addParam('inputMovies1', params.PointerParam, pointerClass='SetOfMovies',
                      label="Reference Aligned Movies", important=True,
                      help='Select the aligned movies to evaluate (this first set will give the global shifts)')

        form.addParam('inputMovies2', params.PointerParam,
                      pointerClass='SetOfMovies',
                      label="Secondary Aligned Movies",
                      help='Shift to be compared with reference alignment')

        form.addParam('minConsCorrelation', params.FloatParam, default=0.75,
                      label='Minimum consensus shifts correlation',
                      help="Minimum value for the consensus correlations between shifts trajectories."
                           "\n If there are noticeable discrepancies between the two estimations below this correlation,"
                           " it will be discarded. If this value is set to -1 no movies will be discarded."
                           "\n Values near 1 will indicate that there are a clear correlation between shifts trajectories.")

        form.addParam('minRangeShift', params.FloatParam, default=3,
                      label='Minimum range shift to apply consensus (px)',
                      help='Minimum value for the range shift detected in axes X and Y in pixels.'
                      "\n It goes from maximum value to minimum value of each axe, creating a square window of a side size"
                      " equal to this value, omitting those movies from the consensus calculation."
                      "\n It is necessary to know when not to use movies whose range shift is so small that it is not comparable.")

        form.addParam('trajectoryPlot', params.BooleanParam, default=False,
                      label='Global Alignment Trajectory Plot',
                      help="This will generate a plot for each movie where the reference and the secondary trajectory"
                           "will be plot in the same graph with its correlation value.")

        form.addParallelSection(threads=4)
        self._defineStreamingParams(form)

# --------------------------- INSERT steps functions -------------------------
    def stepsGeneratorStep(self) -> None:
        self.initializeParams()
        self.newDeps = []

        while not getattr(self, 'finished', False):
            # A failed step makes the executor stop and then join every
            # thread, this generator included: keep polling and the
            # run hangs for good with nothing left to do.
            if self._streamingMustStop():
                break

            self._checkNewInput()
            self._checkNewOutput()

            if getattr(self, 'finished', False):
                break

            self._streamingSleepOnWait()

        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=self.newDeps, needsGPU=False)

    def createOutputStep(self):
        self._closeOutputSet()

    def initializeParams(self):
        self.finished = False
        self.insertedDict = {}
        self.processedDict = []
        self._micsOutputName = self._resolveMicsOutputName()
        if self._micsOutputName is None:
            raise RuntimeError('Could not resolve the micrographs produced by the reference movie alignment.')

        self.stats = {}
        # Decisions computed by worker steps, pending publication into the
        # real output Sets. Snapshotted (never drained) in _checkNewOutput,
        # so a decision is never lost even if a poll's publish step fails.
        self._decidedAccepted = []
        self._decidedDiscarded = []
        self.isStreamClosed = False
        self.samplingRate = self.inputMovies1.get().getSamplingRate()
        self.acquisition = self.inputMovies1.get().getAcquisition()
        self.allMovies1 = {}
        self.allMovies2 = {}

        # Watermark-based discovery state (replaces reconstructing a
        # fresh SetOfMovies from a filename and doing a full Python-side
        # scan of both input Sets on every poll). A movie id seen on one
        # side but not yet matched on the other stays pending and is
        # retried every poll until both sides have it, mirroring
        # XmippProtCTFConsensus's dual-input consensus pattern.
        self._movies1Watermark = 0
        self._movies2Watermark = 0
        self._pendingMovieIds1 = set()
        self._pendingMovieIds2 = set()

        originalRunMode = getattr(self, '_originalRunMode', self.getRunMode())
        if originalRunMode == MODE_RESUME:
            self._restoreStreamingState()

    def _restoreStreamingState(self):
        # The real output Sets are the durable source of truth for what
        # was already processed - no sidecar file needed. Pending
        # decisions (_decidedAccepted/_decidedDiscarded) start empty on
        # every fresh process, so there is nothing stale to prune here:
        # any movie not yet reflected in the outputs is simply not in
        # processedDict, and _checkNewInput will naturally re-schedule it.
        doneAccepted, doneDiscarded = self._getAllDoneIds()
        self.processedDict = sorted(set(doneAccepted) | set(doneDiscarded))

    def _getAllDoneIds(self):
        """ Movie ids already reflected in the real, persisted outputs. """
        acceptedIds = sorted(
            self._getPersistedOutputIds('outputMovies')
        )
        discardedIds = sorted(
            self._getPersistedOutputIds('outputMoviesDiscarded')
        )
        return acceptedIds, discardedIds

    def _isMovieOutputDone(self, movieId):
        acceptedIds, discardedIds = self._getAllDoneIds()
        return movieId in acceptedIds or movieId in discardedIds

    def _checkNewInput(self):
        movieSet1 = self._loadLogicalSet(self.inputMovies1)
        try:
            newIds1, self._movies1Watermark = self._discoverIdsAfter(
                movieSet1, self._movies1Watermark,
            )
            producerClosed1 = movieSet1.isStreamClosed()
            knownIds1 = set(self.processedDict).union(self._pendingMovieIds1)
            newIds1, terminalConsistent1 = self._reconcileClosedStreamIds(
                movieSet1, newIds1, knownIds1, producerClosed1,
                watermarkAttr='_movies1Watermark',
            )
            newMovies1 = self._loadLogicalSetItemsByIds(movieSet1, newIds1)
            self.allMovies1.update({m.getObjId(): m for m in newMovies1})
        finally:
            movieSet1.close()

        movieSet2 = self._loadLogicalSet(self.inputMovies2)
        try:
            newIds2, self._movies2Watermark = self._discoverIdsAfter(
                movieSet2, self._movies2Watermark,
            )
            producerClosed2 = movieSet2.isStreamClosed()
            knownIds2 = set(self.processedDict).union(self._pendingMovieIds2)
            newIds2, terminalConsistent2 = self._reconcileClosedStreamIds(
                movieSet2, newIds2, knownIds2, producerClosed2,
                watermarkAttr='_movies2Watermark',
            )
            newMovies2 = self._loadLogicalSetItemsByIds(movieSet2, newIds2)
            self.allMovies2.update({m.getObjId(): m for m in newMovies2})
        finally:
            movieSet2.close()

        self._pendingMovieIds1.update(newIds1)
        self._pendingMovieIds2.update(newIds2)

        schedulableIds = sorted(
            self._pendingMovieIds1.intersection(self._pendingMovieIds2)
        )
        self._pendingMovieIds1.difference_update(schedulableIds)
        self._pendingMovieIds2.difference_update(schedulableIds)

        self.isStreamClosed = (
            producerClosed1 and producerClosed2
            and terminalConsistent1 and terminalConsistent2
        )

        fDeps = self._insertNewMovieSteps(schedulableIds, self.insertedDict)
        if not fDeps:
            return

        self.newDeps.extend(fDeps)
        self.updateSteps()

    def _insertNewMovieSteps(self, newIDs, insDict):
        deps = []

        for movieID in newIDs:
            if movieID not in insDict:
                stepId = self._insertFunctionStep('alignmentCorrelationMovieStep', movieID, prerequisites=[])
                deps.append(stepId)
                insDict[movieID] = stepId
                self.processedDict.append(movieID)

        return deps

    def alignmentCorrelationMovieStep(self, movieId):
        movie1 = self.allMovies1.get(movieId)
        movie2 = self.allMovies2.get(movieId)

        if getattr(self, '_originalRunMode', self.getRunMode()) == MODE_RESUME and \
           self._isMovieOutputDone(movieId):
            self.info("Skipping movie with ID: %s, output already persisted" % movieId)
            return

        if (movie1 is None) or (movie2 is None):
            self.info('AlignmentCorrelationMovieStep movie1 or movie2 are None')
            return

        alignment1 = movie1.getAlignment()
        alignment2 = movie2.getAlignment()
        shiftX_1, shiftY_1 = alignment1.getShifts()
        shiftX_2, shiftY_2 = alignment2.getShifts()

        rangeShiftX1 = max(shiftX_1) - min(shiftX_1)
        rangeShiftX2 = max(shiftX_2) - min(shiftX_2)
        rangeShiftY1 = max(shiftY_1) - min(shiftY_1)
        rangeShiftY2 = max(shiftY_2) - min(shiftY_2)

        minR = self.minRangeShift.get()

        # Transformation of the shifts to calculate the shifts trajectory correlation
        S1 = np.ones([3, len(shiftX_1)])
        S2 = np.ones([3, len(shiftX_2)])

        S1[0, :] = shiftX_1
        S1[1, :] = shiftY_1
        S2[0, :] = shiftX_2
        S2[1, :] = shiftY_2

        if (rangeShiftX1 <= minR and rangeShiftX2 <= minR) or (rangeShiftY1 <= minR and rangeShiftY2 <= minR):
            self.info('Shift from movie with id %d is within the range shift threshold, so it is omitted from the consensus calculation' % movieId)

            S1_cart = np.array([S1[0, :] / S1[2, :], S1[1, :] / S1[2, :]])
            S2_p_cart = np.array([S2[0, :] / S2[2, :], S2[1, :] / S2[2, :]])
            rmse_cart = np.sqrt((np.square(S1_cart - S2_p_cart)).mean())
            maxe_cart = np.max(S1_cart - S2_p_cart)
            corrX_cart = np.corrcoef(S1_cart[0, :], S2_p_cart[0, :])[0, 1]
            corrY_cart = np.corrcoef(S1_cart[1, :], S2_p_cart[1, :])[0, 1]
            corr_cart = np.min([corrY_cart, corrX_cart])

            self.info('Root Mean Squared Error %f' % rmse_cart)
            self.info('General Corr min(corrX, corrY) %f' % corr_cart)

            with self._lock:
                self._decidedAccepted.append(movieId)

            stats_loc = {'shift_corr': corr_cart, 'shift_corr_X': corrX_cart, 'shift_corr_Y': corrY_cart,
                         'max_error': maxe_cart, 'rmse_error': rmse_cart, 'S1_cart': S1_cart, 'S2_p_cart': S2_p_cart}

            self.stats[movieId] = stats_loc
            self._store()
        else:
            self.info('Shift from movie with id %d surpasses the range shift threshold, so its shift is considered in the consensus calculation' % movieId)

            # Translation through subtraction of the center of mass
            S1c = np.mean(S1, axis=1, keepdims=True)
            S2c = np.mean(S2, axis=1, keepdims=True)

            S11 = S1 - S1c
            S22 = S2 - S2c

            S11[2, :] = np.ones(len(shiftX_1))
            S22[2, :] = np.ones(len(shiftY_1))

            # SVD Decomposition
            H = np.dot(S22, S11.T)
            U, _, VT = np.linalg.svd(H)
            R = np.dot(VT.T, U.T)
            if np.linalg.det(R)<0:
                VT[-1, :] = -VT[-1, :]
                R = np.dot(VT.T, U.T)

            S2_p = np.dot(R, S22) + S1c
            S2_p[2, :] = np.ones(len(shiftY_1))

            S1_cart = np.array([S1[0, :]/S1[2, :], S1[1, :]/S1[2, :]])
            S2_p_cart = np.array([S2_p[0, :] / S2_p[2, :], S2_p[1, :] / S2_p[2, :]])
            rmse_cart = np.sqrt((np.square(S1_cart - S2_p_cart)).mean())
            maxe_cart = np.max(S1_cart - S2_p_cart)
            corrX_cart = np.corrcoef(S1_cart[0, :], S2_p_cart[0, :])[0, 1]
            corrY_cart = np.corrcoef(S1_cart[1, :], S2_p_cart[1, :])[0, 1]
            corr_cart = np.min([corrY_cart, corrX_cart])

            self.info('Root Mean Squared Error %f' % rmse_cart)
            self.info('General Corr min(corrX, corrY) %f' % corr_cart)

            threshold = self.minConsCorrelation.get()
            accepted = threshold == -1 or (np.isfinite(corr_cart) and corr_cart >= threshold)

            if accepted:
                self.info('Movie with id %d has a correlated alignment shift trajectory' % movieId)
                with self._lock:
                    self._decidedAccepted.append(movieId)
            else:
                self.info('Movie with id %d has discrepancy in the alignment with correlation %f' % (movieId, corr_cart))
                with self._lock:
                    self._decidedDiscarded.append(movieId)

            stats_loc = {'shift_corr': corr_cart, 'shift_corr_X': corrX_cart, 'shift_corr_Y': corrY_cart,
                         'max_error': maxe_cart, 'rmse_error': rmse_cart, 'S1_cart': S1_cart, 'S2_p_cart': S2_p_cart}

            self.stats[movieId] = stats_loc
            self._store()

    def _checkNewOutput(self):
        """ Check for already selected movies and update the output set. """
        # The real output Sets are the durable source of truth for what's
        # already published.
        doneListAccepted, doneListDiscarded = self._getAllDoneIds()
        # Snapshot (not drain) the pending worker decisions, so a decision
        # is retried on the next poll if this one fails before publishing.
        with self._lock:
            movieListIdAccepted = list(self._decidedAccepted)
            movieListIdDiscarded = list(self._decidedDiscarded)

        newDoneAccepted = [movieId for movieId in movieListIdAccepted
                           if movieId not in doneListAccepted]
        newDoneDiscarded = [movieId for movieId in movieListIdDiscarded
                            if movieId not in doneListDiscarded]

        maxMovieSize = len(set(self.allMovies1).intersection(set(self.allMovies2)))

        if not newDoneDiscarded and not newDoneAccepted:
            # Nothing decided since last check - done counts are exactly
            # what's already published.
            allDone = len(doneListAccepted) + len(doneListDiscarded)
            self.finished = (self.isStreamClosed and allDone == maxMovieSize)
            return

        def readOrCreateOutputs(doneList, newDone, label=''):
            if len(doneList) > 0 or len(newDone) > 0:
                with self._lock:
                    movSet = self._loadOutputSet(SetOfMovies, 'outputMovies'+label)
                    micSet = self._loadOutputSet(SetOfMicrographs, 'outputMicrographs'+label)
                    label = ACCEPTED if label == '' else DISCARDED
                    publishedIds = self.fillOutput(movSet, micSet, newDone, label)
                    movSet.setSamplingRate(self.samplingRate)
                    micSet.setSamplingRate(self.samplingRate)
                    micSet.setAcquisition(self.acquisition.clone())
                    movSet.setAcquisition(self.acquisition.clone())

                return movSet, micSet, publishedIds
            return None, None, []

        movieSet, micSet, publishedAccepted = readOrCreateOutputs(doneListAccepted, newDoneAccepted)
        movieSetDiscarded, micSetDiscarded, publishedDiscarded = readOrCreateOutputs(doneListDiscarded, newDoneDiscarded, DISCARDED)

        # A movie whose micrograph isn't visible yet (fillOutput skipped
        # it) must not count as done, or this poll could reach
        # allDone == maxMovieSize and finish prematurely while that
        # movie is still waiting to be published on a later poll.
        allDone = (len(doneListAccepted) + len(publishedAccepted)
                   + len(doneListDiscarded) + len(publishedDiscarded))
        self.finished = (self.isStreamClosed and allDone == maxMovieSize)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        def updateOutputsAndClose(movieSet, micSet, label=''):
            if movieSet is None or micSet is None:
                return False

            micsAttrName = 'outputMicrographs'+label
            self._updateOutputSet(micsAttrName, micSet, streamMode)
            self._updateOutputSet('outputMovies'+label, movieSet, streamMode)

            micSet.close()
            movieSet.close()
            return True

        acceptedUpdated = updateOutputsAndClose(movieSet, micSet)
        discardedUpdated = updateOutputsAndClose(movieSetDiscarded, micSetDiscarded, DISCARDED)

        if acceptedUpdated or discardedUpdated:
            self._refreshOutputRelations()

    def _refreshOutputRelations(self):
        relationOutputs = [getattr(self, name, None) for name in ('outputMicrographs', 'outputMicrographsDiscarded')]
        relationOutputs = [output for output in relationOutputs if output is not None]

        if not relationOutputs:
            return

        if self.mapper is not None:
            self.mapper.deleteRelations(self)

        # Movies are considered transformed into the corresponding micrographs.
        # Rebuilding all relations makes Resume safe if a previous run stopped
        # after persisting an output but before its relation was created.
        for micSet in relationOutputs:
            self._defineTransformRelation(self.inputMovies1, micSet)

        if self.mapper is not None:
            self.mapper.commit()

    def fillOutput(self, movieSet, micSet, newDone, label):
        publishedIds = []

        if newDone:
            inputMovieSet = self._loadInputMovieSet()
            inputMicSet = self._loadInputMicrographSet()
            movieIds = set(movieSet.getIdSet()) if movieSet.getSize() else set()
            micIds = set(micSet.getIdSet()) if micSet.getSize() else set()

            for movieId in newDone:
                # Set.__getitem__(int) returns None (not a raise) for a
                # missing row, but indexing straight into .clone() would
                # still crash. The micrograph Set in particular is the
                # output of a different, independently-paced protocol, so
                # a decided movieId may not have a visible micrograph row
                # yet - skip it for now and retry on the next poll rather
                # than crashing the whole protocol.
                if movieId not in inputMovieSet or movieId not in inputMicSet:
                    self.info(
                        "Movie with id %d is not yet visible in the input "
                        "movie/micrograph Set(s); deferring it to the "
                        "next check." % movieId
                    )
                    continue

                movie = inputMovieSet[movieId].clone()
                mic = inputMicSet[movieId].clone()

                movie.setEnabled(self._getEnable(movieId))
                mic.setEnabled(self._getEnable(movieId))
                alignment1 = movie.getAlignment()
                shiftX_1, shiftY_1 = alignment1.getShifts()
                setAttribute(mic, '_alignment_corr', self.stats[movieId]['shift_corr'])
                setAttribute(mic, '_alignment_rmse_error', self.stats[movieId]['rmse_error'])
                setAttribute(mic, '_alignment_max_error', self.stats[movieId]['max_error'])
                alignment = MovieAlignment(xshifts=shiftX_1, yshifts=shiftY_1)
                movie.setAlignment(alignment)

                if self.trajectoryPlot.get():
                    firstFrame, _, _ = self.inputMovies1.get().getFramesRange()
                    self._createAndSaveTrajectoriesPlot(movieId, firstFrame, self.samplingRate)
                    mic.plotCart = Image()
                    mic.plotCart.setFileName(self._getTrajectoriesPlot(movieId))

                if movieId not in movieIds:
                    movieSet.append(movie)
                    movieIds.add(movieId)

                if movieId not in micIds:
                    micSet.append(mic)
                    micIds.add(movieId)

                publishedIds.append(movieId)

            inputMovieSet.close()
            inputMicSet.close()

        return publishedIds

    def _loadOutputSet(self, SetClass, outputName, fixSampling=True):
        suffixByOutputName = {'outputMovies': '', 'outputMicrographs': '', 'outputMoviesDiscarded': 'Discarded', 'outputMicrographsDiscarded': 'Discarded'}
        if outputName not in suffixByOutputName:
            raise ValueError("Unknown MovieAlignmentConsensus output: %s" % outputName)

        outputSet, created = self._loadOrCreateOutputSet(outputName, SetClass, suffixByOutputName[outputName])
        if created:
            inputMovies = self.inputMovies1.get()
            outputSet.copyInfo(inputMovies)
            if fixSampling:
                outputSet.setSamplingRate(inputMovies.getSamplingRate() * self._getBinFactor())
        return outputSet



    def _loadInputMovieSet(self):
        return self._loadLogicalSet(self.inputMovies1)

    def _loadInputMicrographSet(self):
        # The reference micrographs are an output of a *different*
        # protocol (whichever produced inputMovies1), resolved live by
        # attribute name each time - not a Pointer of our own, and never
        # a raw filename - so it stays correct even if that protocol's
        # output Set is re-saved under the compatibility bridge.
        prot1 = self._getReferenceAlignmentProtocol()
        micSet = getattr(prot1, self._micsOutputName)
        micSet.loadAllProperties()
        return micSet

    def _summary(self):
        message = []

        acceptedMoviesSize = (self.outputMovies.getSize()
                        if hasattr(self, "outputMovies") else 0)

        discardedMoviesSize = (self.outputMoviesDiscarded.getSize()
                         if hasattr(self, "outputMoviesDiscarded") else 0)

        acceptedMicrographsSize = (self.outputMicrographs.getSize()
                              if hasattr(self, "outputMicrographs") else 0)

        discardedMicrographsSize = (self.outputMicrographsDiscarded.getSize()
                               if hasattr(self, "outputMicrographsDiscarded") else 0)

        message.append("%d/%d Movies processed (%d accepted and %d discarded)."
                   % (acceptedMoviesSize+discardedMoviesSize,
                      self.inputMovies1.get().getSize(),
                      acceptedMoviesSize, discardedMoviesSize))
        message.append("%d/%d Micrographs processed (%d accepted and %d discarded)."
                       % (acceptedMicrographsSize + discardedMicrographsSize,
                          self.inputMovies1.get().getSize(),
                          acceptedMicrographsSize, discardedMicrographsSize))
        message.append("Values regarding the minimum correlation between sets of movies below %.2f will be discarded."
                       % self.minConsCorrelation.get())
        message.append("Values regarding the range shift of sets of movies below %d pixels won't be considered in the "
                       "consensus calculation." % self.minRangeShift.get())

        return message

    def _validate(self):
        """ The function of this hook is to add some validation before the
        protocol is launched to be executed. It should return a list of
        errors. If the list is empty the protocol can be executed.
        """
        errors = []
        if (self.inputMovies1.get().hasAlignment() == ALIGN_NONE) or \
           (self.inputMovies2.get().hasAlignment() == ALIGN_NONE):
            errors.append("The inputs ( _Input Movies 1_ or _Input Movies 2_ must be aligned before")

        errors.extend(self._validateParallelProcessing())
        return errors

    def _validateParallelProcessing(self):
        # pyworkflow's executor always reserves one thread out of
        # numberOfThreads for its own bookkeeping, and one more of the
        # remaining slots is permanently held by the streaming generator
        # step for the whole run - at least 3 threads are needed to leave
        # a worker slot free for alignmentCorrelationMovieStep.
        if self.numberOfThreads.get() < 3:
            return ['Please assign at least 3 threads: one is reserved by '
                    'the executor for its own bookkeeping and another is '
                    'permanently held by the streaming generator.']
        return []


    # ------------------------------------ Utils functions ------------------------------------
    def _getReferenceAlignmentProtocol(self):
        prot1 = self.inputMovies1.getObjValue()
        if isinstance(prot1, Protocol):
            return prot1

        movieSet = self.inputMovies1.get()
        parentId = movieSet.getObjParentId() if movieSet is not None else None
        project = self.getProject()

        if parentId is None or project is None:
            return None

        try:
            return project.getProtocol(parentId)
        except Exception as error:
            self.debug("Could not resolve reference movie alignment protocol: %s" % error)
            return None

    def _resolveMicsOutputName(self):
        prot1 = self._getReferenceAlignmentProtocol()
        if prot1 is None:
            return None

        for outputName in ('outputMicrographs', 'outputMicrographsDoseWeighted'):
            if getattr(prot1, outputName, None) is not None:
                return outputName

        return None

    def _createAndSaveTrajectoriesPlot(self, movieId, first, pixSize):
        """ Write to a text file the items that have been done. """
        stats = self.stats[movieId]
        fn = self._getExtraPath('global_trajectories_%d' %movieId+'_plot_cart.png')
        shift_X1 = stats['S1_cart'][0, :]
        shift_Y1 = stats['S1_cart'][1, :]
        shift_X2 = stats['S2_p_cart'][0, :]
        shift_Y2 = stats['S2_p_cart'][1, :]
        # ---------------- PLOT -----------------------
        sumMeanX1 = []
        sumMeanY1= []
        sumMeanX2 = []
        sumMeanY2 = []

        def px_to_ang(px):
            y1, y2 = px.get_ylim()
            x1, x2 = px.get_xlim()
            ax_ang2.set_ylim(y1 * pixSize, y2 * pixSize)
            ax_ang.set_xlim(x1 * pixSize, x2 * pixSize)
            ax_ang.figure.canvas.draw()
            ax_ang2.figure.canvas.draw()

        figureSize = (6, 4)
        plotter = Plotter(*figureSize)
        figure = plotter.getFigure()
        ax_px = figure.add_subplot(111)
        ax_px.grid()

        ax_px.set_xlabel('Shift x (px)')
        ax_px.set_ylabel('Shift y (px)')

        ax_px.set_xlabel('Shift x (px) (CorrX:%.3f)' % stats['shift_corr_X'])
        ax_px.set_ylabel('Shift y (px) (CorrX:%.3f)' % stats['shift_corr_Y'])

        ax_ang = ax_px.twiny()
        ax_ang.set_xlabel('Shift x (A)')
        ax_ang2 = ax_px.twinx()
        ax_ang2.set_ylabel('Shift y (A)')

        i = first
        # The output and log files list the shifts relative to the first frame.
        # ROB unit seems to be pixels since sampling rate is only asked
        # by the program if dose filtering is required
        skipLabels = ceil(len(shift_X1) / 10.0)
        labelTick = 1

        for x1, y1, x2, y2 in zip(shift_X1, shift_Y1, shift_X2, shift_Y2):
            sumMeanX1.append(x1)
            sumMeanY1.append(y1)
            sumMeanX2.append(x2)
            sumMeanY2.append(y2)

            if labelTick == 1:
                ax_px.text(x1 - 0.02, y1 + 0.02, str(i))
                labelTick = skipLabels
            else:
                labelTick -= 1
            i += 1

        # automatically update lim of ax_ang when lim of ax_px changes.
        ax_px.callbacks.connect("ylim_changed", px_to_ang)
        ax_px.callbacks.connect("xlim_changed", px_to_ang)

        ax_px.plot(sumMeanX1, sumMeanY1, color='b', label='reference shifts')
        ax_px.plot(sumMeanX2, sumMeanY2, color='r', label='target shifts')
        ax_px.plot(sumMeanX1, sumMeanY1, 'yo')
        ax_px.plot(sumMeanX1[0], sumMeanY1[0], 'ro', markersize=10, linewidth=0.5)
        ax_px.set_title('Global frame alignment')

        ax_px.legend()
        plotter.tightLayout()
        plotter.savefig(fn)
        plotter.close()

    def _getTrajectoriesPlot(self, movieId):
        """ Write to a text file the items that have been done. """
        return self._getExtraPath('global_trajectories_%d' %movieId+'_plot_cart.png')

    def _getEnable(self, movieId):
        # Preserves the pre-existing contract: True when the worker
        # decided this movie was accepted, None otherwise (callers only
        # ever query this for a movie already known to be decided one
        # way or the other).
        with self._lock:
            if movieId in self._decidedAccepted:
                return True
        return None

def setAttribute(obj, label, value):
    if value is None:
        return
    setattr(obj, label, getScipionObj(value))
