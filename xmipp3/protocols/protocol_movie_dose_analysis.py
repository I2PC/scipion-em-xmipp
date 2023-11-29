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
from pyworkflow import VERSION_3_0, NEW
from pyworkflow.object import Set
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam, FloatParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
import pyworkflow.protocol.constants as cons
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfMovies
from pwem.protocols import ProtProcessMovies
from xmipp3.convert import getScipionObj
import statistics as stat
from pyworkflow import BETA, UPDATED, NEW, PROD

THRESHOLD = 2
OUTPUT_ACCEPTED = 'outputMovies'
OUTPUT_DISCARDED = 'outputMoviesDiscarded'

class XmippProtMovieDoseAnalysis(ProtProcessMovies):
    """ Protocol for the dose analysis """
    # FIXME: WITH .mrcs IT DOES NOT FILL THE LABELS
    _devStatus = PROD

    _label = 'movie dose analysis'
    _lastUpdateVersion = VERSION_3_0
    _devStatus = NEW
    _possibleOutputs = {
        OUTPUT_ACCEPTED: SetOfMovies,
        OUTPUT_DISCARDED: SetOfMovies
    }

    finished = False
    stats = {}
    estimatedIds = []
    meanDoseList = []
    medianDoseTemporal = []
    medianDifferences = []
    meanGlobal = 0
    usingExperimental = False

    def __init__(self, **args):
        ProtProcessMovies.__init__(self, **args)

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label=Message.LABEL_INPUT_MOVS,
                      help='Select one or several movies. A dose analysis '
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
                      help='By default, every 50 movies (window=50) we '
                           'compute the percentage of incorrect dose analysis to check if there '
                           'is any anomally in the dose.')
        form.addParam('percentage_window', FloatParam, default=30,
                      label="Windows maximum faulty percentage (%)", expertLevel=LEVEL_ADVANCED,
                      help='By default, if 30% of the movies are discarded'
                           'it assume that the dose has an incorrect value that endures in time.')

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self):
        self._closeOutputSet()

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

        self._insertFunctionStep('createOutputStep',
                                 prerequisites=[], wait=True)

    def initializeParams(self):
        self.framesRange = self.inputMovies.get().getFramesRange()
        self.dims = self.inputMovies.get().getFirstItem().getDim()
        self.pixelSize =  self.inputMovies.get().getFirstItem().getSamplingRate()
        dosePerFrame = self.inputMovies.get().getFirstItem().getAcquisition().getDosePerFrame()
        if dosePerFrame != 0 and dosePerFrame != None:
            self.dosePerFrame = dosePerFrame
        else:
            self.usingExperimental = True

    def _processMovie(self, movie):
        movieId = movie.getObjId()
        self.estimatedIds.append(movieId)
        stats = self.estimatePoissonCount(movie)
        self.stats[movieId] = stats

        fnMonitorSummary = self._getPath("summaryForMonitor.txt")
        if not os.path.exists(fnMonitorSummary):
            fhMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhMonitorSummary = open(fnMonitorSummary, "a")

        fhMonitorSummary.write("movie_%06d_poisson_count: mean=%f stdev=%f [min=%f,max=%f]\n" %
                               (movieId, stats['mean'], stats['std'], stats['min'], stats['max']))
        fhMonitorSummary.close()


    def estimatePoissonCount(self, movie):
        mean_frames = []
        n = movie.getNumberOfFrames()
        frames = [1, n/2, n]

        for frame in frames:
            frame_image = ImageHandler().read("%d@%s" % (frame, movie.getFileName())).getData()
            mean_dose_per_pixel = np.mean(frame_image)
            mean_dose_per_angstrom2 = mean_dose_per_pixel/ self.samplingRate**2
            mean_frames.append(mean_dose_per_angstrom2)

        stats = computeStats(np.asarray(mean_frames))
        self.meanDoseList.append(stats['mean'])

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
        if getattr(self, 'finished', False):
            return

        # If continue then we need to define again some parameters
        if self.isContinued():
            self.initializeParams()

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
            # Find the acceptance intervals
            lower, upper = self.getLimitIntervals()
            self.info(f"Acceptance interval: ({lower:.4f}, {upper:.4f})")
            self.info(f"Global median: ({self.mu:.4f})")

            acceptedMovies = []
            discardedMovies = []

            for movie in newDone:
                newMovie = movie.clone()
                newMovie.setFramesRange(self.framesRange)
                movieId = newMovie.getObjId()
                stats = self.stats[movieId]
                mean = stats['mean']
                std = stats['std']
                minDose = stats['min']
                maxDose = stats['max']
                diff_median = ((mean/self.mu)-1)*100

                setAttribute(newMovie, '_DIFF_TO_DOSE_PER_ANGSTROM2', abs(diff_median))
                setAttribute(newMovie, '_MEAN_DOSE_PER_ANGSTROM2', mean)
                setAttribute(newMovie, '_STD_DOSE_PER_ANGSTROM2', std)
                setAttribute(newMovie, '_MIN_DOSE_PER_FRAME', minDose)
                setAttribute(newMovie, '_MAX_DOSE_PER_FRAME', maxDose)

                self.medianDifferences.append(diff_median)
                self.medianDoseTemporal.append(mean)
                self.info('Movie with id %d has a mean dose per frame of %f and a diff of %f percent'
                          %(movie.getObjId(), mean, diff_median))

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
                self._updateOutputSet(OUTPUT_ACCEPTED, moviesSet, streamMode)
            if len(discardedMovies)>0:
                moviesSetDiscarded = self._loadOutputSet(SetOfMovies, 'movies_discarded.sqlite')
                for movie in discardedMovies:
                    moviesSetDiscarded.append(movie)
                self._updateOutputSet(OUTPUT_DISCARDED, moviesSetDiscarded, streamMode)

            plotDoseAnalysis(self.getDosePlot(), self.meanDoseList, self.mu, lower, upper)
            plotDoseAnalysisDiff(self.getDoseDiffPlot(), self.medianDifferences)

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
    plt.axhline(y=medianGlobal, color='r', linestyle='-', label='Median dose')
    plt.axhline(y=upper, color='b', linestyle='-.', label='Upper limit dose')
    plt.axhline(y=lower, color='g', linestyle='-.', label='Lower limit dose')
    plt.xlabel("Movies ID")
    plt.ylabel("Dose (electrons impacts per angstrom**2 )")
    plt.title('Dose vs time')
    plt.legend()
    plt.grid()
    plt.savefig(filename)

def plotDoseAnalysisDiff(filename, medianDifferences):
    medianDiff = np.median(medianDifferences)
    x = np.arange(start=1+1, stop=len(medianDifferences)+2, step=1)
    plt.figure()
    plt.scatter(x, medianDifferences, s=10)
    plt.axhline(y=medianDiff, color='r', linestyle='-',  label='Median dose difference')
    plt.xlabel("Movies ID")
    plt.ylabel("Dose differences (%)")
    plt.title('Dose differences with respect to the global median vs time')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
