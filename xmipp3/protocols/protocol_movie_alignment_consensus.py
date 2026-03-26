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
Consensus alignment protocol
"""

import os
from datetime import datetime
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
from pyworkflow.protocol import STEPS_PARALLEL, Protocol
import pyworkflow.utils as pwutils
from pwem.protocols import ProtAlignMovies
from pyworkflow.protocol.constants import (STATUS_NEW)
from xmipp3.convert import getScipionObj
from pwem.constants import ALIGN_NONE
from pyworkflow import BETA, UPDATED, NEW, PROD

ACCEPTED = 'Accepted'
DISCARDED = 'Discarded'


class XmippProtConsensusMovieAlignment(ProtAlignMovies, Protocol):
    """
    The protocol compares two sets of aligned movies (reference and secondary) to evaluate their alignment consistency.
    It calculates the correlation between shift trajectories, allowing for a minimum correlation threshold to be set.
    Movies with correlations below this threshold can be discarded. The protocol can also generate plots showing
    the trajectories and correlations for each movie. This helps in identifying and validating the quality of movie alignments
    based on consensus among different alignment runs.
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

# --------------------------- INSERT steps functions -------------------------
    def _insertAllSteps(self):
        self.initializeParams()
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=[], wait=True)

    def createOutputStep(self):
        self._closeOutputSet()

    def initializeParams(self):
        self.finished = False
        self.insertedDict = {}
        self.processedDict = []
        self.movieFn1 = self.inputMovies1.get().getFileName()
        self.movieFn2 = self.inputMovies2.get().getFileName()
        self.micsFn = self._getMicsPath()
        self.stats = {}
        self.isStreamClosed = self.inputMovies1.get().isStreamClosed() and \
                              self.inputMovies2.get().isStreamClosed()
        self.samplingRate = self.inputMovies1.get().getSamplingRate()
        self.acquisition = self.inputMovies1.get().getAcquisition()
        self.allMovies1 = {movie.getObjId(): movie.clone() for movie
                           in self._loadInputMovieSet(self.movieFn1).iterItems()}
        self.allMovies2 = {movie.getObjId(): movie.clone() for movie
                           in self._loadInputMovieSet(self.movieFn2).iterItems()}
        pwutils.makePath(self._getExtraPath('DONE'))

    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all movies
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        # Check if there are new movies to process from the input set
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = max(datetime.fromtimestamp(os.path.getmtime(self.movieFn1)),
                    datetime.fromtimestamp(os.path.getmtime(self.movieFn2)))

        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input movies.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and self.processedDict: # If this is empty it is due to a static "continue" action or it is the first round
            return None

        movieSet1 = self._loadInputMovieSet(self.movieFn1)
        movieSet2 = self._loadInputMovieSet(self.movieFn2)

        movieDict1 = {movie.getObjId(): movie.clone() for movie in movieSet1.iterItems()}
        movieDict2 = {movie.getObjId(): movie.clone() for movie in movieSet2.iterItems()}

        newIds1 = [idMovie for idMovie in movieDict1.keys() if idMovie not in self.processedDict]
        self.allMovies1.update(movieDict1)

        newIds2 = [idMovie for idMovie in movieDict2.keys() if idMovie not in self.processedDict]
        self.allMovies2.update(movieDict2)

        self.lastCheck = datetime.now()
        self.isStreamClosed = movieSet1.isStreamClosed() and \
                              movieSet2.isStreamClosed()

        movieSet1.close()
        movieSet2.close()

        outputStep = self._getFirstJoinStep()

        if len(set(self.allMovies1)) > len(set(self.processedDict)) and \
           len(set(self.allMovies2)) > len(set(self.processedDict)):

            fDeps = self._insertNewMovieSteps(newIds1, newIds2, self.insertedDict)

            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)

            self.updateSteps()

    def _insertNewMovieSteps(self, movies1Dict, movies2Dict, insDict):
        deps = []

        newIDs = list(set(movies1Dict).intersection(set(movies2Dict)))

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
        doneFn = self._getMovieDone(movieId)

        if self.isContinued() and self._isMovieDone(movieId):
            self.info("Skipping movie with ID: %s, seems to be done" % movieId)
            return

        # Clean old finished files
        pwutils.cleanPath(doneFn)

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

            fn = self._getMovieSelecFileAccepted()
            with open(fn, 'a') as f:
                f.write('%d T\n' % movieId)

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

            if corr_cart >= self.minConsCorrelation.get():
                self.info('Movie with id %d has a correlated alignment shift trajectory' % movieId)
                fn = self._getMovieSelecFileAccepted()
                with open(fn, 'a') as f:
                    f.write('%d T\n' % movieId)
            elif corr_cart < self.minConsCorrelation.get():
                self.info('Movie with id %d has discrepancy in the alignment with correlation %f' % (movieId, corr_cart))
                fn = self._getMovieSelecFileDiscarded()
                with open(fn, 'a') as f:
                    f.write('%d F\n' % movieId)

            stats_loc = {'shift_corr': corr_cart, 'shift_corr_X': corrX_cart, 'shift_corr_Y': corrY_cart,
                         'max_error': maxe_cart, 'rmse_error': rmse_cart, 'S1_cart': S1_cart, 'S2_p_cart': S2_p_cart}

            self.stats[movieId] = stats_loc
            self._store()
        # Mark this movie as finished
        open(doneFn, 'w').close()

    def _checkNewOutput(self):
        """ Check for already selected movies and update the output set. """
        # Load previously done items (from text file)
        doneListDiscarded = self._readCertainDoneList(DISCARDED)
        doneListAccepted = self._readCertainDoneList(ACCEPTED)
        # Check for newly done items
        movieListIdAccepted = self._readtMovieId(True)
        movieListIdDiscarded = self._readtMovieId(False)

        newDoneAccepted = [movieId for movieId in movieListIdAccepted
                           if movieId not in doneListAccepted]
        newDoneDiscarded = [movieId for movieId in movieListIdDiscarded
                            if movieId not in doneListDiscarded]

        firstTimeAccepted = len(doneListAccepted) == 0
        firstTimeDiscarded = len(doneListDiscarded) == 0

        allDone = len(doneListAccepted) + len(doneListDiscarded) +\
                  len(newDoneAccepted) + len(newDoneDiscarded)

        # We have finished when there is not more input movies (stream closed)
        # and the number of processed movies is equal to the number of inputs
        maxMovieSize = len(set(self.allMovies1).intersection(set(self.allMovies2)))
        self.finished = (self.isStreamClosed and allDone == maxMovieSize)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        def readOrCreateOutputs(doneList, newDone, label=''):
            if len(doneList) > 0 or len(newDone) > 0:
                with self._lock:
                    movSet = self._loadOutputSet(SetOfMovies, 'movies'+label+'.sqlite')
                    micSet = self._loadOutputSet(SetOfMicrographs, 'micrographs'+label+'.sqlite')
                    label = ACCEPTED if label == '' else DISCARDED
                    self.fillOutput(movSet, micSet, newDone, label)
                    movSet.setSamplingRate(self.samplingRate)
                    micSet.setSamplingRate(self.samplingRate)
                    micSet.setAcquisition(self.acquisition.clone())
                    movSet.setAcquisition(self.acquisition.clone())

                return movSet, micSet
            return None, None

        movieSet, micSet = readOrCreateOutputs(doneListAccepted, newDoneAccepted)
        movieSetDiscarded, micSetDiscarded = readOrCreateOutputs(doneListDiscarded, newDoneDiscarded, DISCARDED)

        if not self.finished and not newDoneDiscarded and not newDoneAccepted:
        # If we are not finished and no new output have been produced
        # it does not make sense to proceed and updated the outputs
        # so we exit from the function here
            return

        def updateRelationsAndClose(movieSet, micSet, first, label=''):
            if os.path.exists(self._getPath('movies'+label+'.sqlite')):
                micsAttrName = 'outputMicrographs'+label
                self._updateOutputSet(micsAttrName, micSet, streamMode)
                self._updateOutputSet('outputMovies'+label, movieSet, streamMode)

                if first:
                    # We consider that Movies are 'transformed' into the Micrographs
                    # This will allow to extend the micrograph associated to a set of
                    # movies to another set of micrographs generated from a
                    # different movie alignment
                    self._defineTransformRelation(self.inputMovies1, micSet)

                micSet.close()
                movieSet.close()

        updateRelationsAndClose(movieSet, micSet, firstTimeAccepted)
        updateRelationsAndClose(movieSetDiscarded, micSetDiscarded, firstTimeDiscarded, DISCARDED)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

    def fillOutput(self, movieSet, micSet, newDone, label):
        if newDone:
            inputMovieSet = self._loadInputMovieSet(self.movieFn1)
            inputMicSet = self._loadInputMicrographSet(self.micsFn)

            for movieId in newDone:
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

                self._writeCertainDoneList(movieId, label)

                if self.trajectoryPlot.get():
                    firstFrame, _, _ = self.inputMovies1.get().getFramesRange()
                    self._createAndSaveTrajectoriesPlot(movieId, firstFrame, self.samplingRate)
                    mic.plotCart = Image()
                    mic.plotCart.setFileName(self._getTrajectoriesPlot(movieId))

                movieSet.append(movie)
                micSet.append(mic)

            inputMovieSet.close()
            inputMicSet.close()

    def _loadOutputSet(self, SetClass, baseName, fixSampling=True):
        """
        Load the output set if it exists or create a new one.
        """
        setFile = self._getPath(baseName)

        if os.path.exists(setFile) and os.path.getsize(setFile) > 0:
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

            inputMovies = self.inputMovies1.get()
            outputSet.copyInfo(inputMovies)

            if fixSampling:
                newSampling = inputMovies.getSamplingRate() * self._getBinFactor()
                outputSet.setSamplingRate(newSampling)

        return outputSet

    def _loadInputMovieSet(self, moviesFn):
        self.debug("Loading input db: %s" % moviesFn)
        movieSet = SetOfMovies(filename=moviesFn)
        movieSet.loadAllProperties()
        movieSet.close()
        self.debug("Closed db.")
        return movieSet

    def _loadInputMicrographSet(self, micsFn):
        self.debug("Loading input db: %s" % micsFn)
        micSet = SetOfMicrographs(filename=micsFn)
        micSet.loadAllProperties()
        micSet.close()
        self.debug("Closed db.")
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

        return errors


    # ------------------------------------ Utils functions ------------------------------------
    def _isMovieDone(self, id):
        """ A movie is done if the marker file exists. """
        return os.path.exists(self._getMovieDone(id))

    def _getMovieDone(self, id):
        """ Return the file that is used as a flag of termination. """
        return self._getExtraPath('DONE', 'movie_%06d.TXT' % id)

    def _readDoneList(self):
        """ Read from a file the id's of the items that have been done. """
        doneFile = self._getAllDone()
        doneList = []
        # Check what items have been previously done
        if os.path.exists(doneFile):
            with open(doneFile) as f:
                doneList += [int(line.strip()) for line in f]
        return doneList

    def _getAllDone(self):
        return self._getExtraPath('DONE_all.TXT')

    def _writeDoneList(self, partList):
        """ Write to a text file the items that have been done. """
        with open(self._getAllDone(), 'a') as f:
            for part in partList:
                f.write('%d\n' % part.getObjId())

    def _getMicsPath(self):
        prot1 = self.inputMovies1.getObjValue()  # pointer to previous protocol

        if hasattr(prot1, 'outputMicrographs'):
            path1 = prot1.outputMicrographs.getFileName()
            if os.path.getsize(path1) > 0:
                return path1
        elif hasattr(prot1, 'outputMicrographsDoseWeighted'):
            path2 = prot1.outputMicrographsDoseWeighted.getFileName()
            if os.path.getsize(path2) > 0:
                return path2
        else:
            return None

    def _readCertainDoneList(self, label):
        """ Read from a text file the id's of the items
        that have been done. """
        doneFile = self._getCertainDone(label)
        doneList = []
        # Check what items have been previously done
        if os.path.exists(doneFile):
            with open(doneFile) as f:
                doneList += [int(line.strip()) for line in f]
        return doneList

    def _writeCertainDoneList(self, movieId, label):
        """ Write to a text file the items that have been done. """
        doneFile = self._getCertainDone(label)
        with open(doneFile, 'a') as f:
            f.write('%d\n' % movieId)

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

    def _getCertainDone(self, label):
        return self._getExtraPath('DONE_'+label+'.TXT')

    def _getMovieSelecFileAccepted(self):
        return self._getExtraPath('selection-movie-accepted.txt')

    def _getMovieSelecFileDiscarded(self):
        return self._getExtraPath('selection-movie-discarded.txt')

    def _readtMovieId(self, accepted):
        if accepted:
            fn = self._getMovieSelecFileAccepted()
        else:
            fn = self._getMovieSelecFileDiscarded()
        moviesList = []
        # Check what items have been previously done
        if os.path.exists(fn):
            with open(fn) as f:
                moviesList += [int(line.strip().split()[0]) for line in f]
        return moviesList

    def _getEnable(self, movieId):
        fn = self._getMovieSelecFileAccepted()
        # Check what items have been previously done
        if os.path.exists(fn):
            with open(fn) as f:
                for line in f:
                    if movieId == int(line.strip().split()[0]):
                        if line.strip().split()[1] == 'T':
                            return True
                        else:
                            return False

def setAttribute(obj, label, value):
    if value is None:
        return
    setattr(obj, label, getScipionObj(value))

