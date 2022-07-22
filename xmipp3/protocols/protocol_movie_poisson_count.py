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
import matplotlib.pyplot as plt
import numpy as np
import os
from pyworkflow import VERSION_3_0
from pyworkflow.object import Set
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam, FloatParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
import pyworkflow.protocol.constants as cons
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfMovies
from pwem.protocols import EMProtocol, ProtProcessMovies
from xmipp3.convert import getScipionObj
import statistics as stat

stats_template = {
    'mean': 0,
    'std': 0,
    'var': 0,
    'max': 0,
    'min': 0,
    'per25': 0,
    'per50': 0,
    'per75': 0,
    'per97.5': 0
             }

THRESHOLD = 0.05


class XmippProtMoviePoissonCount(ProtProcessMovies):
    """ Protocol for the dose analysis """
    ## FIX ME WITH MRC IT DOES NOT WORK


    _label = 'movie poisson count'
    _lastUpdateVersion = VERSION_3_0

    stats = {}
    estimatedIds = []
    meanDoseList = []
    medianDifferences = []
    last_multiple = 0
    meanGlobal = 0

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label=Message.LABEL_INPUT_MOVS,
                      help='Select one or several movies. A dose analysis '
                           'be calculated for each one of them.')
        form.addParam('frameStep', IntParam, default=5,
                      label="Frame step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 5th frame is used to compute '
                           'the movie poisson count. If you set this parameter to '
                           '2, 3, ..., then only every 2nd, 3rd, ... '
                           'frame will be used.')
        form.addParam('movieStep', IntParam, default=5,
                      label="Movie step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 5 movie (movieStep=5) is used to '
                           'compute the movie poisson count. If you set '
                           'this parameter to 2, 3, ..., then only every 2nd, '
                           '3rd, ... movie will be used.')
        form.addParam('window', IntParam, default=50,
                      label="Window step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 50 movies (window=50) is used to '
                           'compute the proportion of incorrect dose analysis.')
        form.addParam('threshold', FloatParam, default=0.05,
                      label="Threshold differences", expertLevel=LEVEL_ADVANCED,
                      help='By default, a difference of 5% against the median dose is used to '
                           'assume that a movie has an incorrect dose.')

        # It should be in parallel (>2) in order to be able of attaching
        #  new movies to the output while estimating residual gain
        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self):
        pass


    def _insertAllSteps(self):
        # Build the list of all processMovieStep ids by
        # inserting each of the steps for each movie
        self.insertedDict = {}
        self.samplingRate = self.inputMovies.get().getSamplingRate()
        # Initial steps
        self.initializeParams()

        # Gain and Dark conversion step
        self.convertCIStep = []
        convertStepId = self._insertFunctionStep('_convertInputStep',
                                                 prerequisites=[])
        self.convertCIStep.append(convertStepId)

        # Conversion step is part of processMovieStep because of streaming.
        movieSteps = self._insertNewMoviesSteps(self.insertedDict,
                                                self.inputMovies.get())
        finalSteps = self._insertFinalSteps(movieSteps)
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=finalSteps, wait=True)


    def initializeParams(self):
        self.framesRange = self.inputMovies.get().getFramesRange()


    def _processMovie(self, movie):
        movieId = movie.getObjId()
        if not self.doPoissonCountProcess(movieId):
            self.info('Movie ID {} NOT PROCESSED'.format(movieId))
            return

        self.info('Movie ID {} PROCESSED'.format(movieId))
        self.estimatedIds.append(movieId)
        stats = self.estimatePoissonCount(movie)
        self.stats[movieId] = stats

        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")
        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")

        fhSummary.write("movie_%06d_poisson_count: mean=%f std=%f [min=%f,max=%f]\n" %
                        (movieId, stats['mean'], stats['std'], stats['min'], stats['max']))

        fhSummary.close()
        fnMonitorSummary.close()


    def estimatePoissonCount(self, movie):
        movieImages = ImageHandler().read(movie.getLocation())
        steps = self.frameStep.get()
        mean_frames = []
        movieFrames = movieImages.getData()
        dimx, dimy, z, n = movieImages.getDimensions()

        for frame in range(n):
            if (frame % steps) == 0:
                mean = np.mean(movieFrames[frame, 0, :, :])
                mean_frames.append(mean)

        stats = computeStats(np.asarray(mean_frames))

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
            inputMovies.loadAllProperties()
            outputSet.copyInfo(inputMovies)
            outputSet.setFramesRange(inputMovies.getFramesRange())

        return outputSet


    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        newDone = [m for m in self.listOfMovies
                   if m.getObjId() not in doneList and self._isMovieDone(m)]

        # Update the file with the newly done movies
        # or exit from the function if no new done movies
        self.debug('_checkNewOutput: ')
        self.debug('   listOfMovies: %s, doneList: %s, newDone: %s'
                   % (len(self.listOfMovies), len(doneList), len(newDone)))

        firstTime = len(doneList) == 0
        allDone = len(doneList) + len(newDone)
        # We have finished when there is no more input movies
        # (stream closed) and the number of processed movies is
        # equal to the number of inputs
        self.finished = self.streamClosed and allDone == len(self.listOfMovies)
        streamMode = Set.STREAM_CLOSED if self.finished \
                     else Set.STREAM_OPEN

        if newDone:
            self._writeDoneList(newDone)
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        self.debug('   finished: %s ' % self.finished)
        self.debug('        self.streamClosed (%s) AND' % self.streamClosed)
        self.debug('        allDone (%s) == len(self.listOfMovies (%s)'
                   % (allDone, len(self.listOfMovies)))
        self.debug('   streamMode: %s' % streamMode)

        moviesSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite')
        index = round(self.window.get() / self.movieStep.get())

        for movie in newDone:
            newMovie = movie.clone()
            newMovie.setFramesRange(self.framesRange)
            movieId = newMovie.getObjId()
            if not self.doPoissonCountProcess(movieId):
                stats = stats_template
            else:
                stats = self.stats[movieId]

            mean = stats['mean']
            std = stats['std']
            min = stats['min']
            max = stats['max']

            setAttribute(newMovie, '_MEAN_DOSE_PER_FRAME', mean)
            setAttribute(newMovie, '_STD_DOSE_PER_FRAME', std)
            setAttribute(newMovie, '_MIN_DOSE_PER_FRAME', min)
            setAttribute(newMovie, '_MAX_DOSE_PER_FRAME', max)

            if self.doPoissonCountProcess(movieId):
                self.meanDoseList.append(mean)

            moviesSet.append(newMovie)
            multiple = len(self.meanDoseList)

            if self.last_multiple < multiple and len(self.meanDoseList) >= 2:
                self.medianGlobal = np.median(self.meanDoseList)
                medianDifference = (self.meanDoseList[-1] - self.medianGlobal)/self.medianGlobal
                self.medianDifferences.append(medianDifference)

                if len(self.medianDifferences) >= 2:
                    self.stdDiffGlobal = stat.stdev(self.medianDifferences)

                plotDoseAnalysis(self.getDosePlot(),
                                 self.meanDoseList, self.movieStep.get(), self.medianGlobal)
                plotDoseAnalysisDiff(self.getDoseDiffPlot(),
                                     self.medianDifferences, self.movieStep.get(), np.median(self.medianDifferences))

                if (len(self.meanDoseList)*self.movieStep.get()) % self.window.get() == 0:
                    windowDiffList = self.medianDifferences[-index:]
                    proportion = len([diff for diff in windowDiffList if abs(diff)
                                      > self.threshold.get()]) / len(windowDiffList)

                    if proportion > THRESHOLD:
                        with open(self._getExtraPath('WARNING.TXT'), 'x') as f:
                            self.info('Proportion of wrong dose in a window surpass the threshold: {} > {}'
                                      .format(proportion, THRESHOLD))
                            f.write('Proportion of wrong dose in a window surpass the threshold: {} > {}'
                                    .format(proportion, THRESHOLD))

                    self.info('Stdev global: {}'.format(self.stdDiffGlobal))
                    self.info('Proportion of wrong dose in a window: {}'.format(proportion))

                self.info('MovieID :   {}    LenMeansList: {}   MedianGlobal: {}   MedianDifference: {}'
                          .format(movieId, len(self.meanDoseList)*self.movieStep.get()
                                  , self.medianGlobal, medianDifference))
                self.last_multiple = multiple

        self._updateOutputSet('outputMovies', moviesSet, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)


    def _updateOutputSet(self, outputName, outputSet, state=Set.STREAM_OPEN):
        outputSet.setStreamState(state)

        if self.hasAttribute(outputName):
            outputSet.write()  # Write to commit changes
            outputAttr = getattr(self, outputName)
            # Copy the properties to the object contained in the protocol
            outputAttr.copy(outputSet, copyId=False)
            # Persist changes
            self._store(outputAttr)
        else:
            # Here the defineOutputs function will call the write() method
            self._defineOutputs(**{outputName: outputSet})
            self._store(outputSet)

        # Close set databaset to avoid locking it
        outputSet.close()


    # ------------------------- UTILS functions --------------------------------
    def doPoissonCountProcess(self, movieId):
        return (movieId - 1) % self.movieStep.get() == 0

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
    p = np.percentile(mean_frames, [25, 50, 75, 97.5])
    mean = np.mean(mean_frames)
    std = np.std(mean_frames)
    var = np.var(mean_frames)
    max = np.max(mean_frames)
    min = np.min(mean_frames)

    stats = {'mean': mean,
             'std': std,
             'var': var,
             'max': max,
             'min': min,
             'per25': p[0],
             'per50': p[1],
             'per75': p[2],
             'per97.5': p[3]
             }
    return stats

def plotDoseAnalysis(filename, doseValues, movieStep, medianGlobal):
    x = np.arange(start=1, stop=len(doseValues)*movieStep+1, step=movieStep)
    plt.figure()
    plt.scatter(x, doseValues)
    plt.axhline(y=medianGlobal, color='r', linestyle='-', label='Median dose')
    plt.xlabel("Movies ID")
    plt.ylabel("Dose")
    plt.title('Dose vs time')
    plt.legend()
    plt.savefig(filename)

def plotDoseAnalysisDiff(filename, medianDifferences, movieStep, medianDiff):
    x = np.arange(start=movieStep+1, stop=len(medianDifferences)*movieStep+2, step=movieStep)
    plt.figure()
    plt.scatter(x, medianDifferences)
    plt.axhline(y=medianDiff, color='r', linestyle='-',  label='Median dose difference')
    plt.xlabel("Movies ID")
    plt.ylabel("Dose differences")
    plt.title('Dose differences with respect to the global mean vs time')
    plt.legend()
    plt.savefig(filename)
