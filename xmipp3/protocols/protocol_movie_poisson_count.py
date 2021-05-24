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

import numpy as np
import os
import sys

from pyworkflow import VERSION_1_1
import pyworkflow.utils as pwutils
from pyworkflow.object import Set
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
from pyworkflow.utils.path import moveFile
import pyworkflow.protocol.constants as cons

from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfMovies, Movie
from pwem.protocols import EMProtocol, ProtProcessMovies

from pwem import emlib
import xmipp3.utils as xmutils


class XmippProtMoviePoissonCount(ProtProcessMovies):
    """ Estimate the gain image of a camera, directly analyzing one of its movies.
    It can correct the orientation of an external gain image (by comparing it with the estimated).
    Finally, it estimates the residual gain (the gain of the movie after correcting with a gain).
    The gain used in the correction will be preferably the external gain, but can also be the estimated
    gain if the first is not found.
    The same criteria is used for assigning the gain to the output movies (external corrected > external > estimated)
    """
    _label = 'movie poisson count'
    _lastUpdateVersion = VERSION_1_1

    estimatedDatabase = 'estGains.sqlite'
    residualDatabase = 'resGains.sqlite'
    stats = {}


    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL
        self.estimatedIds = []

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies, '
                                                                'Movie',
                      label=Message.LABEL_INPUT_MOVS,
                      help='Select one or several movies. A gain image will '
                           'be calculated for each one of them.')
        form.addParam('frameStep', IntParam, default=5,
                      label="Frame step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 5th frame is used to compute '
                           'the movie gain. If you set this parameter to '
                           '2, 3, ..., then only every 2nd, 3rd, ... '
                           'frame will be used.')
        # form.addParam('movieStep', IntParam, default=250,
        #               label="Movie step", expertLevel=LEVEL_ADVANCED,
        #               help='By default, every movie (movieStep=1) is used to '
        #                    'compute the movie gain. If you set '
        #                    'this parameter to 2, 3, ..., then only every 2nd, '
        #                    '3rd, ... movie will be used.')

        # It should be in parallel (>2) in order to be able of attaching
        #  new movies to the output while estimating residual gain
        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self):
        pass
        # if self.estimateGain.get():
        #     estGainsSet = self._loadOutputSet(SetOfImages, self.estimatedDatabase)
        #     self._updateOutputSet('estimatedGains', estGainsSet, Set.STREAM_CLOSED)
        #
        # if self.estimateResidualGain.get():
        #     resGainsSet = self._loadOutputSet(SetOfImages, self.residualDatabase)
        #     self._updateOutputSet('residualGains', resGainsSet, Set.STREAM_CLOSED)

    def _insertNewMoviesSteps(self, insertedDict, inputMovies):
        """ Insert steps to process new movies (from streaming)
        Params:
            insertedDict: contains already processed movies
            inputMovies: input movies set to be check
        """
        deps = []
        if isinstance(self.inputMovies.get(), Movie):
            movie = self.inputMovies.get()
            if movie.getObjId() not in insertedDict:
                stepId = self._insertMovieStep(movie)
                deps.append(stepId)
                insertedDict[movie.getObjId()] = stepId
        else:
            # For each movie insert the step to process it
            for movie in self.inputMovies.get():
                if movie.getObjId() not in insertedDict:
                    stepId = self._insertMovieStep(movie)
                    deps.append(stepId)
                    insertedDict[movie.getObjId()] = stepId
        return deps


    def _processMovie(self, movie):
        movieId = movie.getObjId()
        if not self.doPoissonCountProcess(movieId):
            return

        if movieId not in self.estimatedIds:
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

        mean = 0
        dev = 0
        min = 0
        max = 0
        fhSummary.write("movie_%06d_poisson_count: mean=%f std=%f [min=%f,max=%f]\n" %
                        (movieId, mean, dev, min, max))
        fhSummary.close()

        # G = emlib.Image()
        # G.read(resid_gain)
        # mean, dev, min, max = G.computeStats()

        fnMonitorSummary.close()

    def estimatePoissonCount(self, movie):
        stats = {}
        movieImages = ImageHandler().read(movie.getLocation())  # This is an Xmipp Image DATA
        print(type(movieImages))
        print(np.shape(movieImages))
        dimx, dimy, z, n = movieImages.getDimensions()
        print(dimx)
        print(dimy)


        return stats


    def _loadOutputSet(self, SetClass, baseName, fixGain=False):
        """
        Load the output set if it exists or create a new one.
        fixSampling: correct the output sampling rate if binning was used,
        except for the case when the original movies are kept and shifts
        refers to that one.
        """
        setFile = self._getPath(baseName)
        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

            if isinstance(self.inputMovies.get(), SetOfMovies) or isinstance(self.inputMovies.get(), Movie):
                inputMovies = self.inputMovies.get()
                outputSet.copyInfo(inputMovies)

                if fixGain:
                    outputSet.setGain(self.getFinalGainPath(tifFlipped=True))

        return outputSet

    def _checkNewInput(self):
        if isinstance(self.inputMovies.get(), SetOfMovies):
            ProtProcessMovies._checkNewInput(self)

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        if isinstance(self.inputMovies.get(), Movie):
            movie = self.inputMovies.get()
            movieId = movie.getObjId()
            streamMode = Set.STREAM_CLOSED
            saveMovie = self.getAttributeValue('doSaveMovie', False)
            moviesSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite', fixGain=True)

            # Here we need to pass the statistical study of the mean per frame of each movie, std, max, min
            # movie.....
            moviesSet.append(movie)

            #
            self._updateOutputSet('outputMovies', moviesSet, streamMode)
            outputStep = self._getFirstJoinStep()
            outputStep.setStatus(cons.STATUS_NEW)
            self.finished = True
        else:
            # Load previously done items (from text file)
            doneList = self._readDoneList()
            # Check for newly done items
            newDone = [m.clone() for m in self.listOfMovies
                       if int(m.getObjId()) not in doneList and
                       self._isMovieDone(m)]

            allDone = len(doneList) + len(newDone)
            # We have finished when there is not more input movies
            # (stream closed) and the number of processed movies is
            # equal to the number of inputs
            self.finished = self.streamClosed and \
                            allDone == len(self.listOfMovies)
            streamMode = Set.STREAM_CLOSED if self.finished \
                else Set.STREAM_OPEN

            if newDone:
                self._writeDoneList(newDone)
            elif not self.finished:
                # If we are not finished and no new output have been produced
                # it does not make sense to proceed and updated the outputs
                # so we exit from the function here
                return

            moviesSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite', fixGain=True)
            for movie in newDone:
                moviesSet.append(movie)
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
            # Copy the properties to the object contained in the protcol
            outputAttr.copy(outputSet, copyId=False)
            # Persist changes
            self._store(outputAttr)
        else:
            # Here the defineOutputs function will call the write() method
            self._defineOutputs(**{outputName: outputSet})
            self._store(outputSet)

        # Close set databaset to avoid locking it
        outputSet.close()

    def match_orientation(self, exp_gain, est_gain):
        ''' Calculates the correct orientation of the experimental gain image
            with respect to the estimated
            Input: 2 Xmipp Images
        '''
        print('\nEstimating best orientation')
        sys.stdout.flush()
        best_cor = 0
        # Building conjugate of FT of estimated gain for correlations
        est_gain_array = est_gain.getData()
        est_gain_array = xmutils.normalize_array(est_gain_array)
        est_gain_array_FT_conj = np.conj(np.fft.fft2(est_gain_array))

        # Iterating for mirrors
        for imir in range(2):
            # Iterating for 90 rotations
            for irot in range(4):
                imag_array = np.asarray(exp_gain.getData(), dtype=np.float64)
                if imir == 1:
                    # Matrix for MirrorX
                    M = np.asarray([[-1, 0, imag_array.shape[1]], [0, 1, 0], [0, 0, 1]])
                else:
                    M = np.identity(3)
                angle = irot * 90
                # Transformating the imag array (mirror + rotation)
                imag_array, R = xmutils.rotation(imag_array, angle, est_gain_array.shape, M)

                # calculating correlation
                correlationFunction = arrays_correlation_FT(imag_array, est_gain_array_FT_conj)

                minVal = np.amin(correlationFunction)
                maxVal = np.amax(correlationFunction)
                minLoc = np.where(correlationFunction == minVal)
                maxLoc = np.where(correlationFunction == maxVal)

                if abs(minVal) > abs(best_cor):
                    corLoc = translation_correction(minLoc, est_gain_array.shape)
                    best_cor = minVal
                    best_transf = (angle, imir)
                    best_R = R
                    T = np.asarray([[1, 0, np.asscalar(corLoc[1])], [0, 1, np.asscalar(corLoc[0])], [0, 0, 1]])
                if abs(maxVal) > abs(best_cor):
                    corLoc = translation_correction(maxLoc, est_gain_array.shape)
                    best_cor = maxVal
                    best_transf = (angle, imir)
                    best_R = R
                    T = np.asarray([[1, 0, np.asscalar(corLoc[1])], [0, 1, np.asscalar(corLoc[0])], [0, 0, 1]])

        # Multiply by inverse of translation matrix
        best_M = np.matmul(np.linalg.inv(T), best_R)
        best_gain_array = xmutils.applyTransform(np.asarray(exp_gain.getData(), dtype=np.float64), best_M,
                                                 est_gain_array.shape)

        print('Best correlation: ', best_cor)
        print('Rotation angle: {}\nHorizontal mirror: {}'.format(best_transf[0], best_transf[1] == 1))

        inv_best_gain_array = invert_array(best_gain_array)
        if best_cor > 0:
            xmutils.writeImageFromArray(best_gain_array, self.getOrientedGainPath())
            # xmutils.writeImageFromArray(inv_best_gain_array, self.getBestCorrectionPath())
        else:
            xmutils.writeImageFromArray(inv_best_gain_array, self.getOrientedGainPath())
            # xmutils.writeImageFromArray(best_gain_array, self.getBestCorrectionPath())

    # ------------------------- UTILS functions --------------------------------
    def invertImage(self, img, outFn):
        array = img.getData()
        inv_array = invert_array(array)
        xmutils.writeImageFromArray(inv_array, outFn)

    def getInputGain(self):
        return self.inputMovies.get().getGain()

    def getEstimatedGainPath(self, movieId):
        return self._getExtraPath("movie_%06d_gain.xmp" % movieId)

    def getResidualGainPath(self, movieId):
        return self._getExtraPath("movie_%06d_residual_gain.xmp" % movieId)


    def getFinalGainPath(self, tifFlipped=False):
        fnBest = self.getOrientedGainPath()
        if os.path.exists(fnBest):
            # If the best orientatin has been calculated, take it
            finalGainFn = fnBest
        elif self.getInputGain() != None:
            # Elif, take the input gain provided
            finalGainFn = self.getInputGain()
        else:
            # Elif, take the estimated gain
            finalGainFn = self.searchEstimatedGainPath()
            if finalGainFn == None:
                # If no gains have been estimated, estimate one and use that
                firstMovie = self.inputMovies.get().getFirstItem()
                movieId = firstMovie.getObjId()
                if not movieId in self.estimatedIds:
                    self.estimatedIds.append(movieId)
                    self.estimateGainFun(firstMovie)
                finalGainFn = self.getEstimatedGainPath(movieId)

        ext = pwutils.getExt(self.inputMovies.get().getFirstItem().getFileName()).lower()
        if ext in ['.tif', '.tiff'] and tifFlipped:
            finalGainFn = xmutils.flipYImage(finalGainFn, outDir=self._getExtraPath())

        return finalGainFn

    def searchEstimatedGainPath(self):
        for fn in os.listdir(self._getExtraPath()):
            if fn.endswith('gain.xmp') and not 'residual' in fn:
                return self._getExtraPath(fn)
        return None

    def getArgs(self, movieFn, movieId, extraArgs='', residual=False):
        if residual:
            outbase = self._getExtraPath("movie_%06d_residual" % movieId)
        else:
            outbase = self._getExtraPath("movie_%06d" % movieId)
        return ("-i %s --oroot %s --iter 1 --singleRef --frameStep %d %s"
                % (movieFn, outbase, self.frameStep, extraArgs))

    def doPoissonCountProcess(self, movieId):
        return (movieId - 1) % self.movieStep.get() == 0

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        if errors:
            errors.append("An experimental gain can be associated with a "
                          "setOfMovies during its importing protocol. "
                          "Otherwise, no gain reorientation nor "
                          "gain normalization can be performed.")
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
def arrays_correlation_FT(ar1, ar2_ft_conj, normalize=True):
    '''Return the correlation matrix of an array and the FT_conjugate of a second array using the fourier transform
    '''
    if normalize:
        ar1 = xmutils.normalize_array(ar1)

    ar1_FT = np.fft.fft2(ar1)
    corr2FT = np.multiply(ar1_FT, ar2_ft_conj)
    correlationFunction = np.real(np.fft.ifft2(corr2FT)) / ar1_FT.size

    return correlationFunction


def translation_correction(Loc, shape):
    '''Return translation corrections given the max/min Location and the image shape
    '''
    correcs = []
    for i in range(2):
        if Loc[i] > shape[i] / 2:
            correcs += [Loc[i] - shape[i]]
        else:
            correcs += [Loc[i]]
    return correcs


def invert_array(gain, thres=0.01, depth=1):
    '''Return the inverted array by first converting the values under the threshold to the median of the surrounding'''
    gain = array_zeros_to_median(gain, thres, depth)
    return 1.0 / gain


def array_zeros_to_median(a, thres=0.01, depth=1):
    '''Return an array, replacing the zeros (values under a threshold) with the median of
    its surrounding values (with a depth)'''
    idxs = np.where(np.abs(a) < thres)[0]
    idys = np.where(np.abs(a) < thres)[1]

    for i in range(len(idxs)):
        sur_values = surrounding_values(a, idxs[i], idys[i], depth)
        a[idxs[i]][idys[i]] = np.median(sur_values)
    return a
