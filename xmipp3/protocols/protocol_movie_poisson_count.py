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
import pyworkflow.utils as pwutils
from pyworkflow.object import Set
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
import pyworkflow.protocol.constants as cons
from datetime import datetime
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfMovies
from pwem.protocols import EMProtocol, ProtProcessMovies
from xmipp3.convert import getScipionObj

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

WINDOW = 2


class XmippProtMoviePoissonCount(ProtProcessMovies):
    """ Protocol for the dose analysis """
    _label = 'movie poisson count'
    _lastUpdateVersion = VERSION_3_0

    stats = {}
    estimatedIds = []
    meanDoseList = []
    meanDifferences = []
    meanDerivatives = []
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
                      help='Select one or several movies. A gain image will '
                           'be calculated for each one of them.')
        form.addParam('frameStep', IntParam, default=5,
                      label="Frame step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 5th frame is used to compute '
                           'the movie poisson count. If you set this parameter to '
                           '2, 3, ..., then only every 2nd, 3rd, ... '
                           'frame will be used.')
        form.addParam('movieStep', IntParam, default=5,
                      label="Movie step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every movie (movieStep=1) is used to '
                           'compute the movie poisson count. If you set '
                           'this parameter to 2, 3, ..., then only every 2nd, '
                           '3rd, ... movie will be used.')

        # It should be in parallel (>2) in order to be able of attaching
        #  new movies to the output while estimating residual gain
        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self):
        # here you would create the plot and then save
        # print(self.meanDoseList)
        # plotDoseAnalysis(self.meanDoseList, self.movieStep.get(), self.meanGlobal)
        # plotDoseAnalysisDiff(self.meanDifferences)
        pass

    def _insertAllSteps(self):
        self.initializeParams()
        movieSteps = self._insertNewMoviesSteps(self.moviesDict)
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=movieSteps, wait=True)

    def initializeParams(self):
        self.insertedDict = {}
        self.framesRange = self.inputMovies.get().getFramesRange()
        self.samplingRate = self.inputMovies.get().getSamplingRate()
        self.moviesFn = self.inputMovies.get().getFileName()
        # Build the list of all processMovieStep ids by
        # inserting each of the steps for each movie
        self.moviesDict = self._loadInputList()
        self.streamClosed = self.inputMovies.get().isStreamClosed()
        pwutils.makePath(self._getExtraPath('DONE'))

    def _insertNewMoviesSteps(self, inputMovies):
        """ Insert steps to process new movies (from streaming)
        Params:
            insertedDict: contains already processed movies
            inputMovies: input movies set to be check
        """
        deps = []
        for movie in inputMovies.values():
            if movie.getObjId() not in self.insertedDict:
                stepId = self._insertMovieStep(movie)
                deps.append(stepId)
                self.insertedDict[movie.getObjId()] = stepId

        return deps

    def _insertMovieStep(self, movie):
        """ Insert the processMovieStep for a given movie. """
        # Note1: At this point is safe to pass the movie, since this
        # is not executed in parallel, here we get the params
        # to pass to the actual step that is gone to be executed later on
        # Note2: We are serializing the Movie as a dict that can be passed
        # as parameter for a functionStep
        movieDict = movie.getObjDict(includeBasic=True)
        movieStepId = self._insertFunctionStep('processMovieStep',
                                               movieDict,
                                               movie.hasAlignment(),
                                               prerequisites=[])

        return movieStepId

    def _processMovie(self, movie):
        movieId = movie.getObjId()
        if not self.doPoissonCountProcess(movieId):
            self.info('Movie ID {} not processed'.format(movieId))
            return

        self.info('Movie ID {} processed'.format(movieId))
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

    def _loadInputList(self):
        """ Load the input set of movies and create a list. """
        moviesFile = self.inputMovies.get().getFileName()
        self.debug("Loading input db: %s" % moviesFile)
        movieSet = SetOfMovies(filename=moviesFile)
        movieSet.loadAllProperties()
        newMovies = {movie.getObjId(): movie.clone() for movie in movieSet if movie.getObjId() not in self.insertedDict}
        self.streamClosed = movieSet.isStreamClosed()
        movieSet.close()
        self.debug("Closed db.")

        return newMovies

    def _checkNewInput(self):
        # Check if there are new movies to process from the input set
        localFile = self.inputMovies.get().getFileName()
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(os.path.getmtime(localFile))
        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input movies.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and hasattr(self, 'moviesDict'):
            return None

        self.lastCheck = now
        # Open input movies.sqlite and close it as soon as possible
        newMovies = self._loadInputList()
        self.moviesDict.update(newMovies)
        outputStep = self._getFirstJoinStep()

        if newMovies:
            fDeps = self._insertNewMoviesSteps(newMovies)

            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)

            self.updateSteps()


    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return

        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        newDone = [m.clone() for m in self.moviesDict.values()
                   if int(m.getObjId()) not in doneList and self._isMovieDone(m)]

        allDone = len(doneList) + len(newDone)
        # We have finished when there is not more input movies
        # (stream closed) and the number of processed movies is
        # equal to the number of inputs
        self.finished = self.streamClosed and \
                        allDone == len(self.moviesDict)
        streamMode = Set.STREAM_CLOSED if self.finished \
                     else Set.STREAM_OPEN

        if newDone:
            self._writeDoneList(newDone)
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        moviesSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite')

        for movie in newDone:
            movie.setFramesRange(self.framesRange)
            movieId = movie.getObjId()
            if not self.doPoissonCountProcess(movieId):
                stats = stats_template
            else:
                stats = self.stats[movieId]

            mean = stats['mean']
            std = stats['std']
            min = stats['min']
            max = stats['max']

            setAttribute(movie, '_MEAN_DOSE_PER_FRAME', mean)
            setAttribute(movie, '_STD_DOSE_PER_FRAME', std)
            setAttribute(movie, '_MIN_DOSE_PER_FRAME', min)
            setAttribute(movie, '_MAX_DOSE_PER_FRAME', max)

            if self.doPoissonCountProcess(movieId):
                self.meanDoseList.append(mean)

            moviesSet.append(movie)
            multiple = len(self.meanDoseList) / WINDOW

            # if len(self.meanDoseList) % WINDOW == 0 and self.last_multiple < multiple:
            #     meanWindow = np.mean(self.meanDoseList[-WINDOW:])
            #     print(self.meanDoseList[-WINDOW:])
            #     meanGlobal = np.mean(self.meanDoseList)
            #     meanDifference = abs(self.meanDoseList[-WINDOW] - self.meanDoseList[-1]) / meanGlobal #OR meanWindow?
            #     self.meanDifferences.append(meanDifference)
            #     print('Multiple: {}  MeanWindow: {}    MeanGlobal: {}    MeanDifference: {}'.format(
            #         multiple, meanWindow, meanGlobal, meanDifference))
            #
            #     self.last_multiple = multiple

            if self.last_multiple < multiple and len(self.meanDoseList) >= 2:
                meanWindow = np.mean(self.meanDoseList[-2:])
                print(self.meanDoseList[-2:])
                self.meanGlobal = np.mean(self.meanDoseList)
                #meanDifference = abs(self.meanDoseList[-2] - self.meanDoseList[-1]) / meanGlobal  # OR meanWindow?
                meanDifference = (self.meanDoseList[-1] - self.meanGlobal)/ self.meanGlobal # quitar absoluto
                self.meanDifferences.append(meanDifference)
                self.meanDerivatives.append((self.meanDoseList[-1] - self.meanDoseList[-2])/self.meanGlobal)

                print('Multiple: {}  MeanWindow: {}    MeanGlobal: {}    MeanDifference: {}'.format(
                    multiple, meanWindow, self.meanGlobal, meanDifference))

                plotDoseAnalysis(self.meanDoseList, self.movieStep.get(), self.meanGlobal)
                plotDoseAnalysisDiff(self.meanDifferences)
                plotDoseAnalysisDerivatives(self.meanDerivatives)

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

def plotDoseAnalysis(doseValues, movieStep, meanGlobal):
    x = np.arange(start=1, stop=len(doseValues)*movieStep, step=movieStep)
    plt.scatter(x, doseValues)
    plt.axhline(y=meanGlobal, color='r', linestyle='-')
    plt.xlabel("Movies ID")
    plt.ylabel("Dose")
    plt.title('Dose vs time')
    plt.show()

def plotDoseAnalysisDiff(meanDifferences):
    x = np.arange(start=1, stop=len(meanDifferences)+1, step=1)
    print(x)
    print(meanDifferences)
    plt.scatter(x, meanDifferences)
    plt.xlabel("Differences")
    plt.ylabel("Dose differences")
    plt.title('Dose differences with respect to the global mean vs time')
    plt.show()

def plotDoseAnalysisDerivatives(meanDerivatives):
    x = np.arange(start=1, stop=len(meanDerivatives)+1, step=1)
    plt.scatter(x, meanDerivatives)
    plt.xlabel("Differences")
    plt.ylabel("Dose mean derivative")
    plt.title('Dose tendency (last 2 diff) normalized by global mean vs time')
    plt.show()