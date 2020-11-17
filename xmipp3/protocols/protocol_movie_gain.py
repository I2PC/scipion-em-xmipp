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
import math
import sys

from pyworkflow import VERSION_1_1
from pyworkflow.object import Set
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
from pyworkflow.utils.path import moveFile
import pyworkflow.protocol.constants as cons

from pwem.objects import SetOfMovies, Movie, SetOfImages, Image
from pwem.protocols import EMProtocol, ProtProcessMovies

from pwem import emlib
from xmipp3.utils import normalize_array


class XmippProtMovieGain(ProtProcessMovies):
    """ Estimate the gain image of a camera, directly analyzing one of its movies.
    """
    _label = 'movie gain'
    _lastUpdateVersion = VERSION_1_1

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

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
        form.addParam('movieStep', IntParam, default=250,
                      label="Movie step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every movie (movieStep=1) is used to '
                           'compute the movie gain. If you set '
                           'this parameter to 2, 3, ..., then only every 2nd, '
                           '3rd, ... movie will be used.')
        form.addParam('estimateOrientation', BooleanParam, default=True,
                      label="Estimate gain orientation",
                      help='Estimate the relative orientation between the estimated '
                           'and the existing gain')
        form.addParam('normalizeGain', BooleanParam, default=True,
                      label="Normalize existing gain", expertLevel=LEVEL_ADVANCED,
                      help='Normalize the input gain so that it has a mean of 1')
        form.addParam('useExistingGainImage', BooleanParam, default=True,
                      label="Estimate residual gain",
                      help='If there is a gain image associated with input '
                           'movies, you can decide to use it instead of '
                           'estimating raw/residual gain image. Location of '
                           'this gain image needs to be indicated in import '
                           'movies protocol.')

        # It should be in parallel (>2) in order to be able of attaching
        #  new movies to the output while estimating residual gain
        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def createOutputStep(self):
        pass

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
            if len(insertedDict) == 0 and self.estimateOrientation.get():
                # Adding a first step to orientate the input gain
                firstMovie = self.inputMovies.get().getFirstItem()
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

            # For each movie insert the step to process it
            for movie in self.inputMovies.get():
                if movie.getObjId() not in insertedDict:
                    stepId = self._insertMovieStep(movie)
                    deps.append(stepId)
                    insertedDict[movie.getObjId()] = stepId
        return deps

    def estimateOrientationStep(self, movieDict):
        movie = Movie()
        movie.setAttributesFromDict(movieDict, setBasic=True, ignoreMissing=True)

        movieId = movie.getObjId()
        movieFn = movie.getFileName()
        expGainFn = self.inputMovies.get().getGain()
        resGainFn = self.getCurrentGain(movieId)

        # Check which estimated gain matches with the experimental gain
        args = self.getArgs(movieFn, movieId, " --sigma 0")
        self.runJob("xmipp_movie_estimate_gain", args, numberOfMpi=1)

        resGain = emlib.Image()
        resGain.read(resGainFn)
        expGain = emlib.Image()
        expGain.read(expGainFn)
        self.match_orientation(expGain, resGain)

    def normalizeGainStep(self):
        gainFn = self.getFinalGain()

        oriGain = emlib.Image()
        oriGain.read(gainFn)
        oriArray = oriGain.getData()

        # normalize array to mean 1
        oriArray = oriArray / np.mean(oriArray)

        oriGain.setData(oriArray)
        oriGain.write(self.getBestGain())

    def _processMovie(self, movie):
        movieId = movie.getObjId()
        if not self.doGainProcess(movieId):
            return
        fnMovie = movie.getFileName()
        inputGain = self.getInputGain()
        args = self.getArgs(fnMovie, movieId)

        if self.useExistingGainImage.get() and inputGain is not None:
            args += " --gainImage %s" % self.getFinalGain()

        self.runJob("xmipp_movie_estimate_gain", args, numberOfMpi=1)

        # We take the inverse of the estimated gain computed by xmipp,
        #  stored in correction file
        moveFile(self._getExtraPath("movie_%06d_correction.xmp" % movieId),
                 self.getCurrentGain(movieId))

        # If the gain hasn't been oriented or normalized, we still need bestGain
        if not os.path.exists(self.getBestGain()):
            # No previous gain: bestGain is the estimated
            if not inputGain is None:
                G = emlib.Image()
                G.read(inputGain)
                G.write(self.getBestGain())

        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")
        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")

        estim_gain = self.getCurrentGain(movieId)
        if os.path.exists(estim_gain):
            G = emlib.Image()
            G.read(estim_gain)
            mean, dev, min, max = G.computeStats()
            Gnp = G.getData()
            p = np.percentile(Gnp, [2.5, 25, 50, 75, 97.5])
            fhSummary.write("movie_%06d: mean=%f std=%f [min=%f,max=%f]\n" %
                            (movieId, mean, dev, min, max))
            fhSummary.write(
                "            2.5%%=%f 25%%=%f 50%%=%f 75%%=%f 97.5%%=%f\n" %
                (p[0], p[1], p[2], p[3], p[4]))
            fhSummary.close()
            fnMonitorSummary.write("movie_%06d: %f %f %f %f\n" %
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

        if isinstance(self.inputMovies.get(), SetOfMovies):
            inputMovies = self.inputMovies.get()
            outputSet.copyInfo(inputMovies)

        if fixGain and os.path.isfile(self.getBestGain()):
            outputSet.setGain(self.getFinalGain())

        return outputSet

    def _checkNewInput(self):
        if isinstance(self.inputMovies.get(), SetOfMovies):
            ProtProcessMovies._checkNewInput(self)

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        if isinstance(self.inputMovies.get(), Movie):
            saveMovie = self.getAttributeValue('doSaveMovie', False)
            imageSet = self._loadOutputSet(SetOfImages,
                                           'movies.sqlite')
            movie = self.inputMovies.get()
            imgOut = Image()
            imgOut.setObjId(movie.getObjId())
            imgOut.setSamplingRate(movie.getSamplingRate())
            imgOut.setFileName(self.getCurrentGain(movie.getObjId()))
            imageSet.setSamplingRate(movie.getSamplingRate())
            imageSet.append(imgOut)

            self._updateOutputSet('outputGains', imageSet, Set.STREAM_CLOSED)
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

            if any([self.doGainProcess(i.getObjId()) for i in newDone]):
                # update outputGains if any residualGain is processed in newDone
                imageSet = self._loadOutputSet(SetOfImages,
                                               'gains.sqlite')
                for movie in newDone:
                    movieId = movie.getObjId()
                    if not self.doGainProcess(movieId):
                        continue
                    imgOut = Image()
                    imgOut.setObjId(movieId)
                    imgOut.setSamplingRate(movie.getSamplingRate())
                    imgOut.setFileName(self.getCurrentGain(movieId))
                    imageSet.setSamplingRate(movie.getSamplingRate())
                    imageSet.append(imgOut)

                self._updateOutputSet('outputGains', imageSet, streamMode)

            moviesSet = self._loadOutputSet(SetOfMovies,
                                            'movies.sqlite',
                                            fixGain=True)
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
        print('Estimating best orientation')
        sys.stdout.flush()
        best_cor = 0
        #Building conjugate of FT of estimated gain for correlations
        est_gain_array = est_gain.getData()
        est_gain_array = normalize_array(est_gain_array)
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
                imag_array, R = rotation(imag_array, angle, est_gain_array.shape, M)

                # calculating correlation
                correlationFunction = arrays_correlation_FT(imag_array,est_gain_array_FT_conj)

                minVal = np.amin(correlationFunction)
                maxVal = np.amax(correlationFunction)
                minLoc = np.where(correlationFunction == minVal)
                maxLoc = np.where(correlationFunction == maxVal)
                #print(minVal,maxVal,minLoc,maxLoc)
                if abs(minVal) > abs(best_cor):
                    corLoc=translation_correction(minLoc,est_gain_array.shape)
                    best_cor = minVal
                    best_transf=(angle,imir)
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
        best_gain_array = applyTransform(np.asarray(exp_gain.getData(), dtype=np.float64), best_M, est_gain_array.shape)

        print('Best correlation: ',best_cor)
        print('Rotation angle: {}\nVertical mirror: {}'.format(best_transf[0],best_transf[1]==1))
        if best_cor > 0:
            best_gain_array = invert_array(best_gain_array)

        best_gain = emlib.Image()
        best_gain.setData(best_gain_array)
        best_gain.write(self.getBestGain())

    # ------------------------- UTILS functions --------------------------------
    def getCurrentGain(self, movieId):
        return self._getExtraPath("movie_%06d_gain.xmp" % movieId)

    def getBestGain(self):
        return self._getExtraPath("bestGain.mrc")

    def getFinalGain(self):
        fnBest = self.getBestGain()
        if os.path.exists(fnBest):
            # If the best orientatin has been calculated, take it
            finalGainFn = fnBest
        else:
            finalGainFn = self.getInputGain()
        return finalGainFn

    def getInputGain(self):
        return self.inputMovies.get().getGain()

    def getArgs(self, movieFn, movieId, extraArgs=''):
        return ("-i %s --oroot %s --iter 1 --singleRef --frameStep %d %s"
                % (movieFn, self._getExtraPath("movie_%06d" % movieId),
                   self.frameStep, extraArgs))

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

def applyTransform(imag_array, M, shape):
    ''' Apply a transformation(M) to a np array(imag) and return it in a given shape
    '''
    imag = emlib.Image()
    imag.setData(imag_array)
    imag = imag.applyWarpAffine(list(M.flatten()), shape, True)
    return imag.getData()


def rotation(imag, angle, shape, P):
    '''Rotate a np.array and return also the transformation matrix
    #imag: np.array
    #angle: angle in degrees
    #shape: output shape
    #P: transform matrix (further transformation in addition to the rotation)'''
    (hsrc, wsrc) = imag.shape
    angle *= math.pi / 180
    T = np.asarray([[1, 0, -wsrc / 2], [0, 1, -hsrc / 2], [0, 0, 1]])
    R = np.asarray([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    M = np.matmul(np.matmul(np.linalg.inv(T), np.matmul(R, T)), P)

    transformed = applyTransform(imag, M, shape)
    return transformed, M


def arrays_correlation_FT(ar1,ar2_ft_conj,normalize=True):
    '''Return the correlation matrix of an array and the FT_conjugate of a second array using the fourier transform
    '''
    if normalize:
        ar1=normalize_array(ar1)

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
        sur_values = surrounding_values(a, idxs[i], idys[i], depth)
        a[idxs[i]][idys[i]] = np.median(sur_values)
    return a
