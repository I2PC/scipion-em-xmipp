# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              Daniel Marchán Torres (da.marchan@cnb.csic.es)
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
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import copy

from pyworkflow import VERSION_3_0
from pyworkflow.object import Set
from pyworkflow.protocol.params import (PointerParam, IntParam, FloatParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
import pyworkflow.protocol.constants as cons
from pyworkflow import UPDATED, PROD
import pyworkflow.utils as pwutils

from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfMovies
from pwem.protocols import ProtProcessMovies

from xmipp3.convert import getScipionObj

THRESHOLD = 2
OUTPUT_MOVIES = "outputMovies"
OUTPUT_MOVIES_DISCARDED = "outputMoviesDiscarded"

class XmippProtMovieDoseAnalysis(ProtProcessMovies):
    """
    Analyzes the electron dose applied throughout a movie acquisition. This protocol helps assess dose accumulation and its effects on image quality, providing information essential for dose weighting and optimizing reconstruction..
    """
    # FIXME: WITH .mrcs IT DOES NOT FILL THE LABELS

    _devStatus = PROD
    _label = 'movie dose analysis'
    _lastUpdateVersion = VERSION_3_0
    _possibleOutputs = {
        OUTPUT_MOVIES: SetOfMovies,
        OUTPUT_MOVIES_DISCARDED: SetOfMovies
    }

    finished = False
    stats = {}
    meanDoseList = []
    medianDoseTemporal = []
    medianDifferences = []
    meanGlobal = 0
    usingExperimental = False
    PARALLEL_BATCH_SIZE = 8

    def __init__(self, **args):
        ProtProcessMovies.__init__(self, **args)
        self.stepsExecutionMode = cons.STEPS_PARALLEL

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label=Message.LABEL_INPUT_MOVS,
                      help='Select one or several movies. Dose analysis '
                           'be calculated for each one of them.')
        form.addParam('percentage_threshold', FloatParam, default=5,
                      label="Maximum percentage difference (%)",
                      help='By default, a difference of 5% against the median dose is used to '
                           'assume that the dose has an incorrect value.')
        form.addParam('n_samples', IntParam, default=20,
                      label="Samples to estimate the median dose", expertLevel=LEVEL_ADVANCED,
                      help='By default, 20 movies are used to '
                           'compute the global median.')
        form.addParam('window', IntParam, default=50,
                      label="Window step (movies)", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 50 movies (window=50) '
                           'the percentage of incorrect dose analysis is computed to check if there '
                           'is any anomally in the dose.')
        form.addParam('percentage_window', FloatParam, default=30,
                      label="Maximum faulty percentage (%)", expertLevel=LEVEL_ADVANCED,
                      help='By default, if 30% of the movies are discarded in a window step '
                           'it assumes that the dose has an incorrect value that endures in time.')

        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        """ Insert the steps to perform movie dose evaluation
                """
        self.initializeStep()
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=[], wait=True, needsGPU=False)

    def initializeStep(self):
        self.samplingRate = self.inputMovies.get().getSamplingRate()
        self.movsFn = self.inputMovies.get().getFileName()
        # Important to have both:
        self.insertedIds = []  # Contains images that have been inserted in a Step (checkNewInput).
        self.processedIds = []  # Contains images that have been processed in a Step (checkNewOutput).
        # Contains images that have been processed in a Step (checkNewOutput).
        self.isStreamClosed = self.inputMovies.get().isStreamClosed()
        self.framesRange = self.inputMovies.get().getFramesRange()
        dosePerFrame = self.inputMovies.get().getFirstItem().getAcquisition().getDosePerFrame()

        if dosePerFrame != 0 and dosePerFrame != None:
            self.dosePerFrame = dosePerFrame
        else:
            self.usingExperimental = True

    def createOutputStep(self):
        self._closeOutputSet()

    def _loadInputSet(self, movsFn):
        """ Load the input set of movies and create a list. """
        self.debug("Loading input db: %s" % movsFn)
        movSet = SetOfMovies(filename=movsFn)
        movSet.loadAllProperties()
        self.isStreamClosed = movSet.isStreamClosed()
        movSet.close()
        self.debug("Closed db.")
        return movSet

    def _checkNewInput(self):
        # Check if there are new micrographs to process from the input set
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = datetime.fromtimestamp(os.path.getmtime(self.movsFn))
        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input micrographs.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and self.insertedIds: # If this is empty it is due to a static "continue" action or it is the first round
            return None

        # Open input micrographs.sqlite and close it as soon as possible
        movSet = self._loadInputSet(self.movsFn)
        movSetIds = movSet.getIdSet()
        newIds = [idMov for idMov in movSetIds if idMov not in self.insertedIds]

        self.isStreamClosed = movSet.isStreamClosed()
        self.lastCheck = datetime.now()
        movSet.close()

        outputStep = self._getFirstJoinStep()

        if self.isContinued() and not self.insertedIds: # For "Continue" action and the first round
            doneIds, _, _, _ = self._getAllDoneIds()
            skipIds = list(set(newIds).intersection(set(doneIds)))
            newIds = list(set(newIds).difference(set(doneIds)))
            self.info("Skipping Mics with ID: %s, seems to be done" % skipIds)
            self.insertedIds = doneIds # During the first round of "Continue" action it has to be filled

        if newIds:
            fDeps = self._insertNewMoviesSteps(newIds)
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def _insertNewMoviesSteps(self, newIds):
        """ Insert the processMovieStep for a given movie.
        """
        deps = []
        # Loop through the image IDs in batches
        for i in range(0, len(newIds), self.PARALLEL_BATCH_SIZE):
            batchIds = newIds[i:i + self.PARALLEL_BATCH_SIZE]
            stepId = self._insertFunctionStep(self._processMovies, batchIds, needsGPU=False,
                                          prerequisites=[])
            for movId in batchIds:
                self.insertedIds.append(movId)

            deps.append(stepId)

        return deps

    def _processMovies(self, movieIds):
        inputMovies = self._loadInputSet(self.movsFn)
        for movieId in movieIds:
            movie = inputMovies.getItem("id", movieId).clone()
            movieId = movie.getObjId()
            stats = self.estimatePoissonCount(movie)
            if stats:
                self.stats[movieId] = stats
                self.info("movie_%d_poisson_count: mean=%f stdev=%f [min=%f,max=%f]\n" %
                         (movieId, stats['mean'], stats['std'], stats['min'], stats['max']))
            self.processedIds.append(movieId)

    def estimatePoissonCount(self, movie):
        mean_frames = []
        n = movie.getNumberOfFrames()
        frames = [1, n/2, n]
        try:
            for frame in frames:
                frame_image = ImageHandler().read("%d@%s" % (frame, movie.getFileName())).getData()
                mean_dose_per_pixel = np.mean(frame_image)
                mean_dose_per_angstrom2 = mean_dose_per_pixel/ self.samplingRate**2
                mean_frames.append(mean_dose_per_angstrom2)

            stats = computeStats(np.asarray(mean_frames))
            self.meanDoseList.append(stats['mean'])
        except Exception as e:
            self.error(e)
            self.info('Skipping movie with ID: %d' %movie.getObjId())
            stats = None # If it fails, then Stats should be empty as it could not be read
        return stats

    def _loadOutputSet(self, SetClass, baseName):
        """
        Load the output set if it exists or create a new one.
        fixSampling: correct the output sampling rate if binning was used,
        except for the case when the original movies are kept and shifts
        refers to that one.
        """
        setFile = self._getPath(baseName)
        if os.path.exists(setFile) and os.path.getsize(setFile) > 0:
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)
            inputMovies = self.inputMovies.get()
            outputSet.copyInfo(inputMovies)

        return outputSet

    def _checkNewOutput(self):
        if len(self.meanDoseList) >= self.n_samples.get() and not hasattr(self, 'mu'):
            medianDoseExperimental = np.median(self.meanDoseList)
            if hasattr(self, 'dosePerFrame'):
                refDose = self.dosePerFrame
                diff = abs((medianDoseExperimental/refDose)-1) * 100
                if diff < THRESHOLD:
                   self.mu = refDose
                else:
                    self.mu = medianDoseExperimental
                    self.usingExperimental = True
                    self.info("The given dose per frame %f does not match with the experimental one %f"
                              " having a difference of %f percent. "
                              "Therefore, using the experimental as global median."
                              %(refDose, medianDoseExperimental, diff))
            else:
                self.mu = medianDoseExperimental

        if hasattr(self, 'mu'):
            # load if first time in order to make dataSets relations
            doneListIds, _, _, _ = self._getAllDoneIds()
            processedIds = self.processedIds
            newDone = [micId for micId in processedIds if micId not in doneListIds]
            allDone = len(doneListIds) + len(newDone)
            maxMicSize = self._loadInputSet(self.movsFn).getSize()
            # We have finished when there is not more input movies
            # (stream closed) and the number of processed movies is
            # equal to the number of inputs
            self.finished = self.isStreamClosed and allDone == maxMicSize
            streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

            if not self.finished and not newDone:
                # If we are not finished and no new output have been produced
                # it does not make sense to proceed and updated the outputs
                # so we exit from the function here
                return

            # Update the file with the newly done movies
            # or exit from the function if no new done movies
            self.debug('_checkNewOutput: ')
            self.debug('  doneList: %s, newDone: %s'
                        % (len(doneListIds), len(newDone)))
            self.debug('        self.isStreamClosed (%s) AND' % self.isStreamClosed)
            self.debug('   streamMode: %s' % streamMode)
            # Find the acceptance intervals
            lower, upper = self.getLimitIntervals()
            self.info(f"Acceptance interval: ({lower:.4f}, {upper:.4f})")
            self.info(f"Global median: ({self.mu:.4f})")

            acceptedMovies = []
            discardedMovies = []

            inputMovieSet = self._loadInputSet(self.movsFn)

            for movieId in newDone:
                newMovie = inputMovieSet.getItem("id", movieId).clone()
                newMovie.setFramesRange(self.framesRange)
                movieId = newMovie.getObjId()
                if movieId in self.stats:
                    stats = self.stats[movieId]
                    mean = stats['mean']
                    std = stats['std']
                    minDose = stats['min']
                    maxDose = stats['max']
                    diff_median = ((mean/self.mu)-1)*100

                    setAttribute(newMovie, '_DIFF_TO_DOSE_PER_ANGSTROM2', diff_median)
                    setAttribute(newMovie, '_MEAN_DOSE_PER_ANGSTROM2', mean)
                    setAttribute(newMovie, '_STD_DOSE_PER_ANGSTROM2', std)
                    setAttribute(newMovie, '_MIN_DOSE_PER_FRAME', minDose)
                    setAttribute(newMovie, '_MAX_DOSE_PER_FRAME', maxDose)

                    self.medianDifferences.append(diff_median)
                    self.medianDoseTemporal.append(mean)
                    self.info('Movie with id %d has a mean dose per frame of %f and a diff of %f percent'
                              %(movieId, mean, diff_median))
                    if lower <= mean <= upper:
                        self.info('accepted')
                        acceptedMovies.append(newMovie)
                    else:
                        self.info('discarded')
                        discardedMovies.append(newMovie)

                    if len(self.medianDifferences) % self.window.get() == 0:
                        if self.usingExperimental:
                            # Update the median global
                            self.mu = np.median(self.meanDoseList)
                            self.info('Updating median global to %f' %self.mu)

                        windowList = self.medianDoseTemporal[-self.window.get():]
                        percentage = (1 - (len([dose for dose in windowList
                                          if lower < dose < upper]) / len(windowList)))*100
                        self.info('The faulty percentage of this window is %f' %percentage)

                        if percentage > self.percentage_window.get():
                            with open(self._getExtraPath('WARNING.TXT'), 'a') as f:
                                self.info('Percentage of wrong dose in a window surpass the threshold: {}% > {}%'
                                          .format(percentage, self.percentage_window.get()))
                                f.write('Percentage of wrong dose in a window surpass the threshold: {}% > {}% \n'
                                        .format(percentage, self.percentage_window.get()))
                                f.close()

            if len(acceptedMovies)>0:
                moviesSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite')
                for movie in acceptedMovies:
                    moviesSet.append(movie)
                self._updateOutputSet(OUTPUT_MOVIES, moviesSet, streamMode)
            if len(discardedMovies)>0:
                moviesSetDiscarded = self._loadOutputSet(SetOfMovies, 'movies_discarded.sqlite')
                for movie in discardedMovies:
                    moviesSetDiscarded.append(movie)
                self._updateOutputSet(OUTPUT_MOVIES_DISCARDED, moviesSetDiscarded, streamMode)

            tmpMeanDoseList = copy.deepcopy(self.meanDoseList)
            tmpMedianDifferences = copy.deepcopy(self.medianDifferences)
            plotDoseAnalysis(self.getDosePlot(), tmpMeanDoseList, self.mu, lower, upper)
            plotDoseAnalysisDiff(self.getDoseDiffPlot(), tmpMedianDifferences)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)

        self._store()

# ------------------------- UTILS functions --------------------------------
    def _getAllDoneIds(self):
        doneIds = []
        acceptedIds = []
        discardedIds = []
        sizeOutput = 0

        if hasattr(self, OUTPUT_MOVIES):
            sizeOutput += self.outputMovies.getSize()
            acceptedIds.extend(list(self.outputMovies.getIdSet()))
            doneIds.extend(acceptedIds)

        if hasattr(self, OUTPUT_MOVIES_DISCARDED):
            sizeOutput += self.outputMoviesDiscarded.getSize()
            discardedIds.extend(list(self.outputMoviesDiscarded.getIdSet()))
            doneIds.extend(discardedIds)

        return doneIds, sizeOutput, acceptedIds, discardedIds

    def getLimitIntervals(self):
        """ Funtion to obtain the acceptance interval limits."""
        lower = self.mu - self.mu * (self.percentage_threshold.get()/100)
        upper = self.mu + self.mu * (self.percentage_threshold.get()/100)

        return lower, upper

    def getDosePlot(self):
        return self._getExtraPath('dose_analysis_plot.png')

    def getDoseDiffPlot(self):
        return self._getExtraPath('dose_analysis_diff_plot.png')

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        if errors:
            errors.append("")
        return errors

    def _summary(self):
        fnSummary = self._getPath("summary.txt")
        if not os.path.exists(fnSummary):
            summary = ["No summary information yet."]
        else:
            fhSummary = open(fnSummary, "r")
            summary = []
            for line in fhSummary.readlines():
                summary.append(line.rstrip())
            fhSummary.close()
        return summary


# --------------------- WORKERS --------------------------------------
def setAttribute(obj, label, value):
    if value is None:
        return
    setattr(obj, label, getScipionObj(value))

def computeStats(mean_frames):
    mean = np.mean(mean_frames)
    std = np.std(mean_frames)
    max_dose = np.max(mean_frames)
    min_dose = np.min(mean_frames)

    stats = {'mean': mean,
             'std': std,
             'max': max_dose,
             'min': min_dose,
             }
    return stats

def plotDoseAnalysis(filename, doseValues, medianGlobal, lower, upper):
    x = np.arange(start=1, stop=len(doseValues)+1, step=1)
    plt.figure()
    plt.scatter(x, doseValues,s=10)
    plt.axhline(y=upper, color='r', linestyle='-.', label='Upper limit dose')
    plt.axhline(y=medianGlobal, color='g', linestyle='-', label='Median dose')
    plt.axhline(y=lower, color='r', linestyle='-.', label='Lower limit dose')
    plt.xlabel("Movies ID")
    plt.ylabel("Dose (e- impacts per A²)")
    plt.title('Dose vs time')
    plt.legend()
    plt.grid()
    plt.savefig(filename)

def plotDoseAnalysisDiff(filename, medianDifferences):
    medianDiff = np.median(medianDifferences)
    x = np.arange(start=1+1, stop=len(medianDifferences)+2, step=1)
    plt.figure()
    plt.scatter(x, medianDifferences, s=10)
    plt.axhline(y=5, color='r', linestyle='-.', label='Upper limit dose')
    plt.axhline(y=medianDiff, color='g', linestyle='-',  label='Median dose difference')
    plt.axhline(y=-5, color='r', linestyle='-.', label='Upper limit dose')
    plt.xlabel("Movies ID")
    plt.ylabel("Dose differences (%)")
    plt.title('Dose differences with respect to the global median vs time')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
