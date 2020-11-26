 #**************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              Daniel Marchán
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
from pyworkflow.protocol import STEPS_PARALLEL, params
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED)
from pyworkflow.utils.properties import Message
from pyworkflow.utils.path import moveFile
import pyworkflow.protocol.constants as cons

from pwem.objects import SetOfMicrographs, SetOfImages, Image, Micrograph
from pwem.emlib.image import ImageHandler
from pwem.protocols import EMProtocol,ProtPreprocessMicrographs , ProtMicrographs

from pwem import emlib
from xmipp3.utils import normalize_array


class XmippProtTiltEstimation(ProtPreprocessMicrographs):
    """ Estimate the tilt of a micrograph, by analyzing the PSD correlations of different segments of the image.
    """
    _label = 'tilt estimation'
    _lastUpdateVersion = VERSION_1_1
    results = {}
    registeredFiles = []

    def __init__(self, **args):
        #EMProtocol.__init__(self, **args)
        ProtPreprocessMicrographs.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMicrographs', PointerParam,
                      pointerClass='SetOfMicrographs, Micrograph',
                      label="Input micrographs", important=True,
                      help='Select the SetOfMicrograph to be preprocessed.')

        form.addParam('evaluationStep', IntParam, default=5,
                      label="Micrograph step", expertLevel=LEVEL_ADVANCED,
                      help='By default, every 5th micrograph is used to compute '
                           'the PSD and its correlations. If you set this parameter to '
                           '2, 3, ..., then only every 2nd, 3rd, ... '
                           'micrograph will be used.',validators= [params.Positive])

        form.addParam('n_segments', IntParam, default=4,
                      label="Number of segments", expertLevel=LEVEL_ADVANCED,
                      help='By default, the micrograph will be divided into n '
                           'the PSD and its correlations will be computed in'
                           'every n segments, remember that this value can only'
                           'be an even number.',validators= [params.Positive])
                    #we dont need the validators, since the _validate method is already doing it

        # It should be in parallel (>2) in order to be able of attaching
        #  new movies to the output while estimating residual gain
        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------

    #I believe this is only for static methods
    def _insertAllSteps(self):
        self._defineInputs()
        self.insertedDict = {}
        self.preprocessSteps = self._insertFunctionStep('calculateTiltCorrelationStep')
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=self.preprocessSteps, wait=True)

    def createOutputStep(self):
        pass

    def _insertNewMicrographSteps(self, insertedDict, inputMics):
        """ Insert steps to process new micrographs (from streaming)
        Params:
            insertedDict: contains already processed movies
            inputMovies: input mics set to be check
        """
        deps = []

        if isinstance(self.inputMicrographs.get(), Micrograph):
            micrograph = self.inputMicrographs.get()

            if micrograph.getObjId() not in insertedDict:
                tiltStepId  = self._insertMicrographStep(micrograph)
                deps.append(tiltStepId )
                insertedDict[micrograph.getObjId()] = tiltStepId
        else:
            # For each micrograph insert the step to process it
            for micrograph in self.inputMicrographs.get():
                if micrograph.getObjId() not in insertedDict:
                    stepId = self._insertMicrographStep(micrograph)
                    deps.append(stepId)
                    insertedDict[micrograph.getObjId()] = stepId

        return deps

    def _insertMicrographStep(self, micrograph):
        """ Insert the processMovieStep for a given movie. """
        # Note1: At this point is safe to pass the movie, since this
        # is not executed in parallel, here we get the params
        # to pass to the actual step that is gone to be executed later on
        # Note2: We are serializing the Movie as a dict that can be passed
        # as parameter for a functionStep

        micStepId = self._insertFunctionStep('processMicrograph', micrograph)

        return micStepId

    def processMicrograph(self, micrograph):
        micrographId = micrograph.getObjId()

        if not self.doEvaluationProcess(micrographId):
            return

        fnMicrograph = micrograph.getFileName()

        self.calculateTiltCorrelationStep(micrograph)


        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")

        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")


        G = emlib.Image()
        G.read(estim_gain)
        mean, dev, min, max = G.computeStats()
        Gnp = G.getData()
        p = np.percentile(Gnp, [2.5, 25, 50, 75, 97.5])
        fhSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f]\n" %
                        (micrographId, mean, dev, min, max))
        fhSummary.write(
            "            2.5%%=%f 25%%=%f 50%%=%f 75%%=%f 97.5%%=%f\n" %
            (p[0], p[1], p[2], p[3], p[4]))

        fhSummary.close()
        fnMonitorSummary.write("micrograph_%06d: %f %f %f %f\n" %
                               (micrographId, dev, p[0], p[4], max))
        fnMonitorSummary.close()


    def calculateTiltCorrelationStep(self, mic):
        micFn = mic.getFileName()
        micName = mic.getMicName()
        micBase = self._getMicBase(mic)
        micDir = self._getMicrographDir(mic)
        localParams = self.__params.copy()
        micId = mic.getObjId()

        x, y, z = mic.getDim()
        # alloc np arrays
        #create the binding for window
        #seg_image = window_binding()

        micImage = ImageHandler().read(mic)
        orig_matrix = micImage.getData()

        imag = Image()
        imag.setData(orig_matrix)
        imag = imag.computePSD()


        micImage.write(self._getExtraPath("average.mrc"))

        # Check which estimated gain matches with the experimental gain
        #self.runJob("xmipp_movie_estimate_gain", args, numberOfMpi=1)


    def _loadOutputSet(self, SetClass, baseName, fixGain=False):
        """
        Load the output set if it exists or create a new one.
        """
        setFile = self._getPath(baseName)
        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

        if isinstance(self.inputMicrographs.get(), SetOfMicrographs):
            inputMicrographs = self.inputMicrographs.get()
            outputSet.copyInfo(inputMicrographs)

        return outputSet

    def _stepsCheck(self):
        # Input micrograph set can be loaded or None when checked for new inputs
        # If None, we load it
        self._checkNewInput()
        self._checkNewOutput()

    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all micrographs
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

#MOvie process checNewInput
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
        if self.lastCheck > mTime and hasattr(self, 'listOfMovies'):
            return None

        self.lastCheck = now
        # Open input movies.sqlite and close it as soon as possible
        self._loadInputList()
        newMovies = any(m.getObjId() not in self.insertedDict
                        for m in self.listOfMovies)
        outputStep = self._getFirstJoinStep()

        if newMovies:
            fDeps = self._insertNewMoviesSteps(self.insertedDict,
                                               self.listOfMovies)
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

#Micrograph check input
    def _checkNewInput(self):
        if isinstance(self.inputMicrographs.get(), SetOfMicrographs):
        # Check if there are new micrographs to process from the input set
            micsFile = self.inputMicrographs.get().getFileName()
            micsSet = SetOfMicrographs(filename=micsFile)
            micsSet.loadAllProperties()
            self.SetOfMicrographs = [m.clone() for m in micsSet]
            self.streamClosed = micsSet.isStreamClosed()
            micsSet.close()
            newMics = any(m.getObjId() not in self.insertedDict
                          for m in self.inputMics)

            outputStep = self._getFirstJoinStep()

            if newMics:
                fDeps = self._insertNewMicrographSteps(self.insertedDict, self.inputMicrographs.get())
                if outputStep is not None:
                    outputStep.addPrerequisites(*fDeps)
                self.updateSteps()

#CTF micrograph
    def _checkNewInput(self):
        # Check if there are new micrographs to process from the input set
        localFile = self.getInputMicrographs().getFileName()
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(getmtime(localFile))
        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input micrographs.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and hasattr(self, 'listOfMics'):
            return None


    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        if isinstance(self.inputMicrographs.get(), Micrograph):
            saveMicrograph = self.getAttributeValue('doSaveMicrograph', False)
            imageSet = self._loadOutputSet(SetOfImages,
                                           'micrograph.sqlite')
            micrograph = self.inputMicrographs.get()
            imgOut = Image()
            imgOut.setObjId(micrograph.getObjId())
            imgOut.setSamplingRate(micrograph.getSamplingRate())
            imgOut.setFileName(self.getCurrentGain(micrograph.getObjId()))
            imageSet.setSamplingRate(micrograph.getSamplingRate())
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
                                            'micrograph.sqlite',
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



    # ------------------------- UTILS functions --------------------------------
    def doEvaluationProcess(self, micrographId):
        return (micrographId-1) % self.evaluationStep.get() == 0

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        if (self.n_segments.get() % 2 != 0) or (self.n_segments.get() < 0):
            errors.append("The number of segments must be an even number and positive number")

        if self.evaluationStep.get() <= 0:
            errors.append("The evaluation step should be a positive number")

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
