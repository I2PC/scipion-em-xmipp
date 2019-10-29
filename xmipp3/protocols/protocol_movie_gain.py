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
import cv2
import sys

from pyworkflow import VERSION_1_1
from pyworkflow.em.data import SetOfMovies, Movie
from pyworkflow.em.protocol import EMProtocol, ProtProcessMovies
from pyworkflow.object import Set
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
from pyworkflow.utils.path import cleanPath
import pyworkflow.protocol.constants as cons
import pyworkflow.em as em

import xmippLib
from xmippLib import *
from xmipp_base import *
from xmipp3.utils import *

def cv2_applyTransform(imag, M, shape):
    ''' Apply a transformation(M) to a np array(imag) and return it in a given shape
    '''
    (hdst, wdst) = shape
    transformed = cv2.warpAffine(imag, M[:2][:], (wdst, hdst),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    return transformed

def cv2_rotation(imag, angle, shape, P):
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
    return cv2_applyTransform(imag, M, shape), M

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

class XmippProtMovieGain(ProtProcessMovies):
    """
    Estimate the gain image of a camera, directly analyzing one of its movies.
    """
    _label = 'movie gain'
    _lastUpdateVersion = VERSION_1_1

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.stepsExecutionMode = em.STEPS_PARALLEL

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

        form.addParallelSection(threads=1, mpi=1)

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
            # For each movie insert the step to process it
            for idx, movie in enumerate(self.inputMovies.get()):
                if idx % self.movieStep.get() != 0:
                    continue
                if movie.getObjId() not in insertedDict:
                    stepId = self._insertMovieStep(movie)
                    deps.append(stepId)
                    insertedDict[movie.getObjId()] = stepId
        return deps

    def _processMovie(self, movie):
        movieId = movie.getObjId()
        fnMovie = movie.getFileName()
        gain = self.inputMovies.get().getGain()
        args = "-i %s --oroot %s --iter 1 --singleRef --frameStep %d" % \
               (fnMovie, self._getPath("movie_%06d" % movieId),
                self.frameStep)
        fnGain = self._getPath("movie_%06d_gain.xmp" % movieId)

        # Check which estimated gain matches with the experimental gain
        if self.estimateOrientation.get() and gain is not None and not os.path.exists(
                self._getPath("bestGain.xmp")):
            self.runJob("xmipp_movie_estimate_gain", args+" --sigma 0", numberOfMpi=1)

            if os.path.exists(fnGain):
                G = xmippLib.Image()
                G.read(fnGain)
                exp_gain = xmippLib.Image()
                exp_gain.read(gain)
                self.match_orientation(exp_gain, G)

        if self.normalizeGain.get():
            fnBest=self._getPath("bestGain.xmp")
            if os.path.exists(fnBest):
                #If the best orientatin has been calculated, take it
                fnGain=fnBest
            else:
                fnGain=gain
            ori_gain = xmippLib.Image()
            ori_gain.read(fnGain)
            ori_array=ori_gain.getData()
            #normalize array to mean 1
            ori_array = ori_array / np.mean(ori_array)

            ori_gain.setData(ori_array)
            ori_gain.write(self._getPath("bestGain.xmp"))

        if self.useExistingGainImage.get() and gain is not None:
            if self.estimateOrientation.get() or self.normalizeGain.get():
                args += " --gainImage %s"%self._getPath("bestGain.xmp")
            else:
                args += " --gainImage %s" % gain
        self.runJob("xmipp_movie_estimate_gain", args, numberOfMpi=1)
        #cleanPath(self._getPath("movie_%06d_correction.xmp" % movieId))
        #We take the inverse of the estimated gain computed by xmipp, stored in correction file
        os.rename(self._getPath("movie_%06d_correction.xmp" % movieId),
                                self._getPath("movie_%06d_gain.xmp" % movieId))

        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")
        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")
        if os.path.exists(fnGain):
            G = xmippLib.Image()
            G.read(fnGain)
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

    def _loadOutputSet(self, SetClass, baseName, fixSampling=True, fixGain=False):
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

        if fixSampling:
            newSampling = inputMovies.getSamplingRate() * self._getBinFactor()
            outputSet.setSamplingRate(newSampling)

        if fixGain:
            outputSet.setGain(self._getPath("bestGain.xmp"))

        return outputSet

    def _checkNewInput(self):
        if isinstance(self.inputMovies.get(), SetOfMovies):
            ProtProcessMovies._checkNewInput(self)

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        if isinstance(self.inputMovies.get(), Movie):
            saveMovie = self.getAttributeValue('doSaveMovie', False)
            imageSet = self._loadOutputSet(em.data.SetOfImages,
                                           'movies.sqlite',
                                           fixSampling=saveMovie)
            movie = self.inputMovies.get()
            imgOut = em.data.Image()
            imgOut.setObjId(movie.getObjId())
            imgOut.setSamplingRate(movie.getSamplingRate())
            imgOut.setFileName(self._getPath("movie_%06d_gain.xmp" %
                                             movie.getObjId()))
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
                            allDone == int(math.ceil(
                len(self.listOfMovies) /
                float(self.movieStep.get())))
            streamMode = Set.STREAM_CLOSED if self.finished \
                else Set.STREAM_OPEN

            if newDone:
                self._writeDoneList(newDone)
            elif not self.finished:
                # If we are not finished and no new output have been produced
                # it does not make sense to proceed and updated the outputs
                # so we exit from the function here
                return

            saveMovie = self.getAttributeValue('doSaveMovie', False)
            imageSet = self._loadOutputSet(em.data.SetOfImages,
                                           'gains.sqlite',
                                           fixSampling=saveMovie)

            for movie in newDone:
                imgOut = em.data.Image()
                imgOut.setObjId(movie.getObjId())
                imgOut.setSamplingRate(movie.getSamplingRate())
                imgOut.setFileName(self._getPath("movie_%06d_gain.xmp"
                                                 % movie.getObjId()))
                imageSet.setSamplingRate(movie.getSamplingRate())
                imageSet.append(imgOut)


            inMovie = self.inputMovies.get()
            moviesSet = self._loadOutputSet(em.data.SetOfMovies,
                                            'movies.sqlite',
                                            fixSampling=saveMovie,
                                            fixGain=True)

            for movie in newDone:
                moviesSet.append(movie)

            self._updateOutputSet('outputMovies', moviesSet, streamMode)

            self._updateOutputSet('outputGains', imageSet, streamMode)

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
        ''' Calculates the correct orientation of the experimental gain image with respect to the estimated
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
                imag_array, R = cv2_rotation(imag_array, angle, est_gain_array.shape, M)

                # calculating correlation
                correlationFunction = arrays_correlation_FT(imag_array,est_gain_array_FT_conj)

                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(correlationFunction)
                if abs(minVal) > abs(best_cor):
                    # minloc o minloc-h
                    corLoc=translation_correction(minLoc,est_gain_array.shape)
                    T = np.asarray([[1, 0, corLoc[0]], [0, 1, corLoc[1]], [0, 0, 1]])
                    best_cor = minVal
                    #Multiply by inverse of translation matrix
                    best_M = np.matmul(np.linalg.inv(T), R)
                if abs(maxVal) > abs(best_cor):
                    corLoc = translation_correction(maxLoc, est_gain_array.shape)
                    T = np.asarray([[1, 0, corLoc[0]], [0, 1, corLoc[1]], [0, 0, 1]])
                    best_cor = maxVal
                    #Multiply by inverse of translation matrix
                    best_M = np.matmul(np.linalg.inv(T),R)

        best_gain_array = cv2_applyTransform(np.asarray(exp_gain.getData(), dtype=np.float64), best_M, est_gain_array.shape)

        print('Best correlation: ',best_cor)
        if best_cor > 0:
            best_gain_array = invert_array(best_gain_array)

        best_gain = xmippLib.Image()
        best_gain.setData(best_gain_array)
        best_gain.write(self._getPath("bestGain.xmp"))

    # --------------------------- INFO functions -------------------------------
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
