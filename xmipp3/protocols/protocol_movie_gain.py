# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              Tomas Majtner (tmajtner@cnb.csic.es)
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
from pyworkflow.protocol import STEPS_PARALLEL, Protocol
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
from pyworkflow.utils.path import moveFile
import pyworkflow.protocol.constants as cons

from pwem.objects import SetOfMovies, Movie, SetOfImages, Image
from pwem.protocols import EMProtocol, ProtProcessMovies
from pyworkflow import BETA, UPDATED, NEW, PROD

from pwem import emlib
import xmipp3.utils as xmutils

OUTPUT_ESTIMATED_GAINS = 'estimatedGains'
OUTPUT_ORIENTED_GAINS = 'orientedGain'
OUTPUT_RESIDUAL_GAINS = 'residualGains'
OUTPUT_MOVIES = 'outputMovies'

class XmippProtMovieGain(ProtProcessMovies, Protocol):
    """ Estimate the gain image of a camera, directly analyzing one of its movies.
    It can correct the orientation of an external gain image (by comparing it with the estimated).
    Finally, it estimates the residual gain (the gain of the movie after correcting with a gain).
    The gain used in the correction will be preferably the external gain, but can also be the estimated
    gain if the first is not found.
    The same criteria is used for assigning the gain to the output movies (external corrected > external > estimated)
    """
    _label = 'movie gain'
    _devStatus = UPDATED
    _lastUpdateVersion = VERSION_1_1
    _stepsCheckSecs = 60
    estimatedDatabase = 'estGains.sqlite'
    residualDatabase = 'resGains.sqlite'
    _possibleOutputs = {OUTPUT_ESTIMATED_GAINS: SetOfImages,
                        OUTPUT_ORIENTED_GAINS: SetOfImages,
                        OUTPUT_RESIDUAL_GAINS: SetOfImages,
                        OUTPUT_MOVIES: SetOfMovies}

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label=Message.LABEL_INPUT_MOVS,
                      help='Select several movies. A gain image will '
                           'be calculated for each one of them.')
        form.addParam('estimateGain', BooleanParam, default=True,
                      label="Estimate movies gain",
                      help='Estimate the gain from a set of movies using the algorith from xmipp')
        form.addParam('estimateOrientation', BooleanParam, default=True,
                      label="Estimate external gain orientation",
                      help='Estimate the relative orientation between the estimated '
                           'and the existing gain')
        form.addParam('estimateResidualGain', BooleanParam, default=True,
                      label="Estimate residual gain",
                      help='If there is a gain image associated with input '
                           'movies, you can decide to use it instead of '
                           'estimating raw/residual gain image. Location of '
                           'this gain image needs to be indicated in import '
                           'movies protocol.')
        form.addParam('normalizeGain', BooleanParam, default=True,
                      label="Normalize existing gain", expertLevel=LEVEL_ADVANCED,
                      help='Normalize the input gain so that it has a mean of 1')
        form.addParam('estimateSigma', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label="Estimate the sigma parameter",
                      help='Estimate the sigma parameter for the gain image computation')
        form.addParam('frameStep', IntParam, default=5,
                      label="Frame step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 5th frame is used to compute '
                           'the movie gain. If you set this parameter to '
                           '2, 3, ..., then only every 2nd, 3rd, ... '
                           'frame will be used.')
        form.addParam('movieStep', IntParam, default=250,
                      label="Movie step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 250 movies (movieStep=250) is used to '
                           'compute the movie gain. If you set '
                           'this parameter to 2, 3, ..., then every 2nd, '
                           '3rd, ... movie will be used.')

        # It should be in parallel (>2) in order to be able of attaching
        #  new movies to the output while estimating residual gain
        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self):
        if self.estimateGain.get():
            estGainsSet = self._loadOutputSet(SetOfImages, self.estimatedDatabase)
            self._updateOutputSet(OUTPUT_ESTIMATED_GAINS, estGainsSet, Set.STREAM_CLOSED)

        if self.estimateResidualGain.get():
            resGainsSet = self._loadOutputSet(SetOfImages, self.residualDatabase)
            self._updateOutputSet(OUTPUT_RESIDUAL_GAINS, resGainsSet, Set.STREAM_CLOSED)

    def _insertNewMoviesSteps(self, insertedDict, inputMovies):
        """ Insert steps to process new movies (from streaming)
        Params:
            insertedDict: contains already processed movies
            inputMovies: input movies set to be check
        """
        deps = []
        if len(insertedDict) == 0 and self.estimateOrientation.get():
            # Adding a first step to orientate the input gain
            firstMovie = inputMovies.getFirstItem()
            movieDict = firstMovie.getObjDict(includeBasic=True)
            orientStepId = self._insertFunctionStep('estimateOrientationStep',
                                                    movieDict,
                                                    prerequisites=self.convertCIStep)
            # adding orientStep as dependency for all other steps
            self.convertCIStep.append(orientStepId)

        if len(insertedDict) == 0 and self.normalizeGain.get():
            # Adding a step to normalize the gain (only one)
            normStepId = self._insertFunctionStep('normalizeGainStep',
                                                  prerequisites=self.convertCIStep)
            # adding normStep as dependency for all other steps
            self.convertCIStep.append(normStepId)
        self.estimatedIds, self.estimatedResIds = [], []
        # For each movie insert the step to process it
        for movie in inputMovies:
            if movie.getObjId() not in insertedDict:
                stepId = self._insertMovieStep(movie)
                deps.append(stepId)
                insertedDict[movie.getObjId()] = stepId
        return deps

    def estimateGainFun(self, movie, noSigma=False, residual=False):
        movieId = movie.getObjId()
        movieFn = movie.getFileName()

        if os.path.splitext(movieFn)[-1] == '.eer':
            # When the input movie is an EER file, use a single 
            movieFn += '#32,4K,uint16'

        # Check which estimated gain matches with the experimental gain
        args = self.getArgs(movieFn, movieId, residual=residual)
        if not self.estimateSigma.get() or noSigma:
            args += " --sigma 0"
        if residual:
            args += " --gainImage {}".format(self.getFinalGainPath())
        self.runJob("xmipp_movie_estimate_gain", args, numberOfMpi=1)

    def estimateOrientationStep(self, movieDict):
        movie = Movie()
        movie.setAttributesFromDict(movieDict, setBasic=True, ignoreMissing=True)
        movieId = movie.getObjId()
        estGainFn = self.getEstimatedGainPath(movieId)
        expGainFn = self.inputMovies.get().getGain()

        if not movieId in self.estimatedIds:
            self.estimatedIds.append(movieId)
            self.estimateGainFun(movie, noSigma=True)

        estGain = xmutils.readImage(estGainFn)
        expGain = xmutils.readImage(expGainFn)
        self.match_orientation(expGain, estGain)

        orientedSet = self._loadOutputSet(SetOfImages, 'orientedGain.sqlite')
        orientedSet = self.updateGainsOutput(movie, orientedSet, self.getOrientedGainPath())
        self._updateOutputSet(OUTPUT_ORIENTED_GAINS, orientedSet, Set.STREAM_CLOSED)

    def normalizeGainStep(self):
        gainFn = self.getFinalGainPath()

        oriGain = emlib.Image()
        oriGain.read(gainFn)
        oriArray = oriGain.getData()

        # normalize array to mean 1
        oriArray = oriArray / np.mean(oriArray)

        oriGain.setData(oriArray)
        oriGain.write(self.getFinalGainPath())

    def _processMovie(self, movie):
        movieId = movie.getObjId()
        if not self.doGainProcess(movieId):
            return
        inputGain = self.getInputGain()

        if self.estimateGain.get() and not movieId in self.estimatedIds:
                self.estimatedIds.append(movieId)
                self.estimateGainFun(movie)

        if self.estimateResidualGain.get() and not movieId in self.estimatedResIds:
            self.info('\nEstimating residual gain')
            self.estimatedResIds.append(movieId)
            self.estimateGainFun(movie, residual=True)

        # If the gain hasn't been oriented or normalized, we still need orientedGain
        if not os.path.exists(self.getOrientedGainPath()):
            # No previous gain: orientedGain is the estimated
            if not inputGain is None:
                G = emlib.Image()
                G.read(inputGain)
                G.write(self.getOrientedGainPath())

        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")
        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")

        resid_gain = self.getResidualGainPath(movieId)
        if os.path.exists(resid_gain):
            G = emlib.Image()
            G.read(resid_gain)
            mean, dev, min, max = G.computeStats()
            Gnp = G.getData()
            p = np.percentile(Gnp, [2.5, 25, 50, 75, 97.5])
            fhSummary.write("movie_%06d_residual: mean=%f std=%f [min=%f,max=%f]\n" %
                            (movieId, mean, dev, min, max))
            fhSummary.write(
                "            2.5%%=%f 25%%=%f 50%%=%f 75%%=%f 97.5%%=%f\n" %
                (p[0], p[1], p[2], p[3], p[4]))
            fhSummary.close()
            fnMonitorSummary.write("movie_%06d_residual: %f %f %f %f\n" %
                                   (movieId, dev, p[0], p[4], max))
        fnMonitorSummary.close()

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

            inputMovies = self.inputMovies.get()
            outputSet.copyInfo(inputMovies)

            if fixGain:
                outputSet.setGain(self.getFinalGainPath(tifFlipped=True))

        return outputSet

    def _checkNewInput(self):
        ProtProcessMovies._checkNewInput(self)

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return

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

        if any([self.doGainProcess(i.getObjId()) for i in newDone]):
            # update outputGains if any residualGain is processed in newDone
            if self.estimateGain.get():
                estGainsSet = self._loadOutputSet(SetOfImages, self.estimatedDatabase)
            if self.estimateResidualGain.get():
                resGainsSet = self._loadOutputSet(SetOfImages, self.residualDatabase)

            for movie in newDone:
                movieId = movie.getObjId()
                if not self.doGainProcess(movieId):
                    continue
                if self.estimateGain.get():
                    estGainsSet = self.updateGainsOutput(movie, estGainsSet, self.getEstimatedGainPath(movieId))
                if self.estimateResidualGain.get():
                    resGainsSet = self.updateGainsOutput(movie, resGainsSet, self.getResidualGainPath(movieId))

            if self.estimateGain.get():
                self._updateOutputSet(OUTPUT_ESTIMATED_GAINS, estGainsSet, streamMode)
            if self.estimateResidualGain.get():
                self._updateOutputSet(OUTPUT_RESIDUAL_GAINS, resGainsSet, streamMode)

        moviesSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite', fixGain=True)
        for movie in newDone:
            moviesSet.append(movie)
        self._updateOutputSet(OUTPUT_MOVIES, moviesSet, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)

    def updateGainsOutput(self, movie, imgSet, imageFile):
        movieId = movie.getObjId()
        imgOut = Image()
        imgOut.setObjId(movieId)
        imgOut.setSamplingRate(movie.getSamplingRate())
        imgOut.setFileName(imageFile)

        imgSet.setSamplingRate(movie.getSamplingRate())
        imgSet.append(imgOut)
        return imgSet

    def match_orientation(self, exp_gain, est_gain):
        ''' Calculates the correct orientation of the experimental gain image
            with respect to the estimated
            Input: 2 Xmipp Images
        '''
        self.info('\nEstimating best orientation')
        sys.stdout.flush()
        best_cor = 0
        #Building conjugate of FT of estimated gain for correlations
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
                #Transformating the imag array (mirror + rotation)
                imag_array, R = xmutils.rotation(imag_array, angle, est_gain_array.shape, M)

                # calculating correlation
                correlationFunction = arrays_correlation_FT(imag_array,est_gain_array_FT_conj)

                minVal = np.amin(correlationFunction)
                maxVal = np.amax(correlationFunction)
                minLoc = np.where(correlationFunction == minVal)
                maxLoc = np.where(correlationFunction == maxVal)

                if abs(minVal) > abs(best_cor):
                    corLoc = translation_correction(minLoc,est_gain_array.shape)
                    best_cor = minVal
                    best_transf = (angle,imir)
                    best_R = R
                    T = np.asarray([[1, 0, corLoc[1].item()],
                                    [0, 1, corLoc[0].item()],
                                    [0, 0, 1]])
                if abs(maxVal) > abs(best_cor):
                    corLoc = translation_correction(maxLoc, est_gain_array.shape)
                    best_cor = maxVal
                    best_transf = (angle, imir)
                    best_R = R
                    T = np.asarray([[1, 0, corLoc[1].item()], 
                                    [0, 1, corLoc[0].item()],
                                    [0, 0, 1]])

        # Multiply by inverse of translation matrix
        best_M = np.matmul(np.linalg.inv(T), best_R)
        best_gain_array = xmutils.applyTransform(np.asarray(exp_gain.getData(), dtype=np.float64), best_M, est_gain_array.shape)

        self.info('Best correlation: %f' %best_cor)
        self.info('Rotation angle: {}\nHorizontal mirror: {}'.format(best_transf[0],best_transf[1]==1))

        inv_best_gain_array = invert_array(best_gain_array)
        if best_cor > 0:
            xmutils.writeImageFromArray(best_gain_array, self.getOrientedGainPath())
            #xmutils.writeImageFromArray(inv_best_gain_array, self.getBestCorrectionPath())
        else:
            xmutils.writeImageFromArray(inv_best_gain_array, self.getOrientedGainPath())
            #xmutils.writeImageFromArray(best_gain_array, self.getBestCorrectionPath())

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

    def getFlippedOrientedGainPath(self):
        return self._getExtraPath("orientedGain_flipped.mrc")

    def getOrientedGainPath(self):
        return self._getExtraPath("orientedGain.mrc")

    def getOrientedCorrectionPath(self):
        return self._getExtraPath("orientedCorrection.mrc")

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
            finalGainFn = xmutils.flipYImage(finalGainFn, outDir = self._getExtraPath())

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

    def doGainProcess(self, movieId):
        return (movieId-1) % self.movieStep.get() == 0

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        if self.estimateOrientation.get() and not self.getInputGain():
            errors.append("Experimental gain needed to estimate its proper "
                         "orientation.")
        if self.normalizeGain.get() and not self.getInputGain():
            errors.append("Experimental gain needed to normalize it.")
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
def arrays_correlation_FT(ar1,ar2_ft_conj,normalize=True):
    '''Return the correlation matrix of an array and the FT_conjugate of a second array using the fourier transform
    '''
    if normalize:
        ar1=xmutils.normalize_array(ar1)

    ar1_FT = np.fft.fft2(ar1)
    corr2FT = np.multiply(ar1_FT, ar2_ft_conj)
    correlationFunction = np.real(np.fft.ifft2(corr2FT)) / ar1_FT.size

    return correlationFunction


def translation_correction(Loc,shape):
    '''Return translation corrections given the max/min Location and the image shape
    '''
    correcs=[]
    for i in range(2):
        if Loc[i]>shape[i]/2:
            correcs+=[Loc[i]-shape[i]]
        else:
            correcs+=[Loc[i]]
    return correcs


def invert_array(gain,thres=0.01,depth=1):
    '''Return the inverted array by first converting the values under the threshold to the median of the surrounding'''
    gain=array_zeros_to_median(gain, thres, depth)
    return 1.0/gain


def array_zeros_to_median(a, thres=0.01, depth=1):
    '''Return an array, replacing the zeros (values under a threshold) with the median of
    its surrounding values (with a depth)'''
    idxs = np.where(np.abs(a) < thres)[0]
    idys = np.where(np.abs(a) < thres)[1]

    for i in range(len(idxs)):
        sur_values = xmutils.surrounding_values(a, idxs[i], idys[i], depth)
        a[idxs[i]][idys[i]] = np.median(sur_values)
    return a
