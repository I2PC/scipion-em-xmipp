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

def copy_image(imag):
    # Return a copy of a xmipp_image
    new_imag = xmippLib.Image()
    new_imag.setData(imag.getData())
    return new_imag

def cv2_applyTransform(imag,M,shape):
    (hdst, wdst) = shape
    transformed = cv2.warpAffine(imag, M[:2][:], (wdst, hdst),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    return transformed

def cv2_rotation(imag,angle, shape, P):
    (hsrc, wsrc) = imag.shape

    #M = cv2.getRotationMatrix2D((h/2,w/2), angle, 1.0)
    angle*=math.pi/180
    T=np.asarray([[1,0,-wsrc/2],[0,1,-hsrc/2],[0,0,1]])
    R=np.asarray([[math.cos(angle), math.sin(angle), 0],[-math.sin(angle),math.cos(angle), 0],[0,0,1]])
    M=np.matmul(np.matmul(np.linalg.inv(T),np.matmul(R,T)),P)
    print(M)
    return cv2_applyTransform(imag,M,shape),M

def matmul_serie(mat_list,size=4):
    # Return the matmul of several numpy arrays
    if len(mat_list)>0:
        res = np.identity(len(mat_list[0]))
    else:
        res = np.identity(size)
    for i in range(len(mat_list)):
        res = np.matmul(res, mat_list[i])
    return res

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
                      label="Estimate orientation",
                      help='Estimate the relative orientation between the estimated '
                           'and the existing gain')
        form.addParam('useExistingGainImage', BooleanParam, default=True,
                      label="Use existing gain image",
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
        if self.useExistingGainImage.get() and gain is not None:
            args += " --gainImage %s" % gain
        self.runJob("xmipp_movie_estimate_gain", args, numberOfMpi=1)
        cleanPath(self._getPath("movie_%06d_correction.xmp" % movieId))
        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")
        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")
        fnGain = self._getPath("movie_%06d_gain.xmp" % movieId)
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

            # Check which estimated gain matches with the experimental gain
            if self.estimateOrientation.get() and gain is not None and not os.path.exists(self._getPath("bestGain.xmp")):
                ori_gain=xmippLib.Image()
                ori_gain.read(gain)
                match_orientation(ori_gain,G)


    def _loadOutputSet(self, SetClass, baseName, fixSampling=True):
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

            self._updateOutputSet('outputMovies', imageSet, Set.STREAM_CLOSED)
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
                                           'movies.sqlite',
                                           fixSampling=saveMovie)

            for movie in newDone:
                imgOut = em.data.Image()
                imgOut.setObjId(movie.getObjId())
                imgOut.setSamplingRate(movie.getSamplingRate())
                imgOut.setFileName(self._getPath("movie_%06d_gain.xmp"
                                                 % movie.getObjId()))
                imageSet.setSamplingRate(movie.getSamplingRate())
                imageSet.append(imgOut)

            self._updateOutputSet('outputMovies', imageSet, streamMode)

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
        # input: two xmipp images
        # Calculates the correct orientation of the experimental gain image with respect to the estimated
        best_cor = 0
        est_gain_array = est_gain.getData()
        est_gain_array -= np.mean(est_gain_array)
        est_gain_array /= np.std(est_gain_array)
        est_gain_array_FT_conj = np.conj(np.fft.fft2(est_gain_array))

        # Iterating for mirrors
        for imir in range(2):
            # Iterating for 90 rotations
            for irot in range(4):
                imag_array = exp_gain.getData()
                if imir == 1:
                    # imag_array=cv2.flip(imag_array,1)
                    M = np.asarray([[-1, 0, imag_array.shape[1]], [0, 1, 0], [0, 0, 1]])
                else:
                    M = np.identity(3)
                angle = irot * 90
                imag_array, R = cv2_rotation(imag_array, angle, est_gain_array.shape, M)

                # calculating correlation
                imag_array -= np.mean(imag_array)
                imag_array /= np.std(imag_array)
                imag_array_FT = np.fft.fft2(imag_array)
                corr2FT = np.multiply(imag_array_FT, est_gain_array_FT_conj)
                correlationFunction = np.real(np.fft.ifft2(corr2FT)) / imag_array_FT.size

                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(correlationFunction)
                print('correlation: %i,%i' % (imir, irot), minVal, maxVal, minLoc, maxLoc)
                sys.stdout.flush()
                if abs(minVal) > abs(best_cor):
                    T = np.asarray([[1, 0, minLoc[0]], [0, 1, minLoc[1]], [0, 0, 1]])
                    best_cor = minVal
                    best_M = np.matmul(T, R)
                if abs(maxVal) > abs(best_cor):
                    T = np.asarray([[1, 0, maxLoc[0]], [0, 1, maxLoc[1]], [0, 0, 1]])
                    best_cor = maxVal
                    best_M = np.matmul(T, R)

        print(best_M)
        best_gain_array = cv2_applyTransform(exp_gain.getData(), best_M, est_gain_array.shape)
        if best_cor < 0:
            best_gain_array = np.where(np.abs(best_gain_array) > 1e-2, 1.0 / best_gain_array, 1.0)
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
