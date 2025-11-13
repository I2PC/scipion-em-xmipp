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
from itertools import combinations
import os
import math
from datetime import datetime
from pyworkflow import VERSION_3_0
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED, FloatParam, GE, GT, Range)
import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
import pyworkflow.protocol.constants as cons
from pwem.objects import SetOfMicrographs, Image, Set, Float
from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtMicrographs
from xmipp3 import emlib
from xmipp3.convert import getScipionObj
from pyworkflow import UPDATED, PROD

OUTPUT_MICS = "outputMicrographs"
OUTPUT_MICS_DISCARDED = "discardedMicrographs"
AUTOMATIC_WINDOW_SIZES = [4096, 2048, 1024, 512, 256]

class XmippProtTiltAnalysis(ProtMicrographs):
    """ Estimates the tilt angle of a micrograph by analyzing power spectral density correlations across different image quadrants. This helps discard the ones that have a tilt so high it could negatively affect the posterior processing.
    """
    _label = 'tilt analysis'
    _devStatus = PROD
    _lastUpdateVersion = VERSION_3_0
    _possibleOutputs = {OUTPUT_MICS: SetOfMicrographs,
                        OUTPUT_MICS_DISCARDED: SetOfMicrographs
                        }
    PARALLEL_BATCH_SIZE = 8

    def __init__(self, **args):
        ProtMicrographs.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMicrographs', PointerParam,
                      pointerClass='SetOfMicrographs',
                      label="Input micrographs", important=True,
                      help='Select the SetOfMicrograph to be preprocessed.')

        form.addParam('autoWindow', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
                      label='Estimate automatically the window size?',
                      help='Use this button to decide if you want to estimate the window size'
                           'based on the movie size. It will select between 512, 1024, 2048 or 4096. '
                           'Condition:  window_size <= Size/2.5 ')

        form.addParam('window_size', IntParam, label='Window size',
                      condition="not autoWindow",
                      default=1024, expertLevel=LEVEL_ADVANCED, validators=[GE(256, 'Error must be greater than 256')],
                      help=''' By default, the micrograph will be divided into windows of dimensions 1024x1024, '''
                           ''' the PSD its correlations will be computed in every segment.''')

        form.addParam('objective_resolution', FloatParam, label='Objective resolution',
                      default=3, expertLevel=LEVEL_ADVANCED, validators=[GT(0, 'Error must be Positive')],
                      help='''By default, micrographs PSD will be cropped into a central windows of dimensions (xdim*'''
                           '''(sampling rate/objective resolution)) x (ydim*(sampling rate/objective resolution)).''')

        form.addParam('meanCorr_threshold', FloatParam, label='Mean correlation threshold',
                      default=0.5, expertLevel=LEVEL_ADVANCED, validators=[Range(0, 1)],
                      help='''By default, micrographs will be divided into an output set and a discarded set based'''
                           ''' on the mean and std threshold''')

        form.addParam('stdCorr_threshold', FloatParam, label='STD correlation threshold',
                      default=0.1, expertLevel=LEVEL_ADVANCED, validators=[GT(0, 'Error must be greater than 0')],
                      help='''By default, micrographs will be divided into an output set and a discarded set based'''
                           ''' on the mean and std threshold.''')

        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        """ Insert the steps to perform CTF estimation, or re-estimation,
        on a set of micrographs.
        """
        self.initializeStep()
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=[], wait=True, needsGPU=False)

    def initializeStep(self):
        self.samplingRate = self.inputMicrographs.get().getSamplingRate()
        self.micsFn = self.inputMicrographs.get().getFileName()
        self.stats = {}
        # Important to have both:
        self.insertedIds = [] # Contains images that have been inserted in a Step (checkNewInput).
        self.processedIds = [] # Contains images that have been processed in a Step (checkNewOutput).
        self.isStreamClosed = self.inputMicrographs.get().isStreamClosed()
        self.windowSize = self.getWindowSize()

    def createOutputStep(self):
        self._closeOutputSet()

    def _loadInputSet(self, micsFn):
        """ Load the input set of mics and create a list. """
        self.debug("Loading input db: %s" % micsFn)
        micSet = SetOfMicrographs(filename=micsFn)
        micSet.loadAllProperties()
        self.isStreamClosed = micSet.isStreamClosed()
        micSet.close()
        self.debug("Closed db.")
        return micSet

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

    def _checkNewInput(self):
        # Check if there are new micrographs to process from the input set
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = datetime.fromtimestamp(os.path.getmtime(self.micsFn))
        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input micrographs.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and self.insertedIds: # If this is empty it is dut to a static "continue" action or it is the first round
            return None

        # Open input micrographs.sqlite and close it as soon as possible
        micSet = self._loadInputSet(self.micsFn)
        micSetIds = micSet.getIdSet()
        newIds = [idMic for idMic in micSetIds if idMic not in self.insertedIds]

        self.isStreamClosed = micSet.isStreamClosed()
        self.lastCheck = datetime.now()
        micSet.close()

        outputStep = self._getFirstJoinStep()

        if self.isContinued() and not self.insertedIds: # For "Continue" action and the first round
            doneIds, _, _, _ = self._getAllDoneIds()
            skipIds = list(set(newIds).intersection(set(doneIds)))
            newIds = list(set(newIds).difference(set(doneIds)))
            self.info("Skipping Mics with ID: %s, seems to be done" % skipIds)
            self.insertedIds = doneIds # During the first round of "Continue" action it has to be filled

        if newIds:
            fDeps = self._insertNewMicrographSteps(newIds)
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def _checkNewOutput(self):
        doneListIds, _, _, _ = self._getAllDoneIds()
        processedIds = self.processedIds
        newDone = [micId for micId in processedIds if micId not in doneListIds]
        allDone = len(doneListIds) + len(newDone)
        maxMicSize = self._loadInputSet(self.micsFn).getSize()
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

        micsAccepted = []
        micsDiscarded = []

        inputMicSet = self._loadInputSet(self.micsFn)

        for micId in newDone:
            mic = inputMicSet.getItem("id", micId).clone()
            corr_mean = Float(self.stats[micId]['mean'])
            corr_std = Float(self.stats[micId]['std'])
            corr_min = Float(self.stats[micId]['min'])
            corr_max = Float(self.stats[micId]['max'])
            psdImage = Image(location=self.getPSDs(self._getExtraPath(), micId))
            setAttribute(mic, '_tilt_mean_corr', corr_mean)
            setAttribute(mic, '_tilt_std_corr', corr_std)
            setAttribute(mic, '_tilt_min_corr', corr_min)
            setAttribute(mic, '_tilt_max_corr', corr_max)
            setAttribute(mic, '_tilt_psds_image', psdImage)
            # Double threshold
            if corr_mean > self.meanCorr_threshold.get() and corr_std < self.stdCorr_threshold.get():
                micsAccepted.append(mic)
            else:
                micsDiscarded.append(mic)

        if len(micsAccepted) > 0:
            micSet = self._loadOutputSet(SetOfMicrographs, 'micrograph.sqlite')
            for mic in micsAccepted:
                micSet.append(mic)
            self._updateOutputSet('outputMicrographs', micSet, streamMode)

        if len(micsDiscarded) > 0:
            micSet_discarded = self._loadOutputSet(SetOfMicrographs, 'micrograph' + 'DISCARDED' + '.sqlite')
            for mic in micsDiscarded:
                micSet_discarded.append(mic)
            self._updateOutputSet('discardedMicrographs', micSet_discarded, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)

        self._store()

    def _insertNewMicrographSteps(self, newIds):
        """ Insert steps to process new micrographs (from streaming)
        Params:
            newIds: input mics ids to be processed
        """
        deps = []
        # Loop through the image IDs in batches
        for i in range(0, len(newIds), self.PARALLEL_BATCH_SIZE):
            batchIds = newIds[i:i + self.PARALLEL_BATCH_SIZE]
            tiltStepId = self._insertFunctionStep(self.processMicrographListStep, batchIds, needsGPU=False,
                                              prerequisites=[])
            for micId in batchIds:
                self.insertedIds.append(micId)

            deps.append(tiltStepId)

        return deps

    def processMicrographListStep(self, micIds):
        inputMicSet = self._loadInputSet(self.micsFn)

        for micId in micIds:
            micrograph = inputMicSet.getItem("id", micId).clone()
            self._processMicrograph(micrograph)

    def _processMicrograph(self, micrograph):
        micFolderTmp = self._getOutputMicFolder(micrograph)
        pwutils.makePath(micFolderTmp)
        micrographId = micrograph.getObjId()
        correlations = self.calculateTiltCorrelationStep(micrograph)
        # Numpy array to compute all the
        correlations = np.asarray(correlations)
        # Calculate the mean, dev of the correlation
        stats = computeStats(correlations)
        self.stats[micrographId] = stats
        self.processedIds.append(micrographId)

        fnMonitorSummary = self._getPath("summaryForMonitor.txt")

        if not os.path.exists(fnMonitorSummary):
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fnMonitorSummary = open(fnMonitorSummary, "a")

        self.info("\nmicrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
                  (micrographId, stats['mean'], stats['std'], stats['min'], stats['max']))

        fnMonitorSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
                               (micrographId, stats['mean'], stats['std'], stats['min'], stats['max']))
        fnMonitorSummary.close()

    def calculateTiltCorrelationStep(self, mic):
        psds = []
        correlations = []
        autocorrelations = []
        # Read image
        micFolder = self._getOutputMicFolder(mic)
        micImage = ImageHandler().read(mic.getLocation())
        dimx, dimy, z, n = micImage.getDimensions()
        windStep = self.windowSize
        x_steps, y_steps = window_coordinates2D(dimx, dimy, windStep)
        subWindStep = int(windStep * (self.samplingRate / self.objective_resolution.get()))
        x_steps_psd, y_steps_psd = window_coordinates2D(subWindStep * 2, subWindStep * 2, subWindStep)
        # Extract windows
        window_image = ImageHandler().createImage()
        rotatedWind_psd = ImageHandler().createImage()
        output_image = ImageHandler().createImage()
        output_array = np.zeros(((subWindStep * 2)-1, (subWindStep * 2)-1))
        ih = ImageHandler()

        for i in range(0, 3, 2):
            for j in range(0, 3, 2):
                window = micImage.window2D(x_steps[i], y_steps[j], x_steps[i + 1], y_steps[j + 1])
                x_dim, y_dim, z, n = window.getDimensions()
                # NORMALIZED
                mean, dev, minCorr, maxCorr = window.computeStats()
                winMatrix = (window.getData() - mean) / dev
                window_image.setData(winMatrix)
                # Compute PSD
                wind_psd = window_image.computePSD(0.4, x_dim, y_dim, 1)
                wind_psd.convertPSD()
                # Rotate PSD
                psdMatrix = wind_psd.getData()
                P = np.identity(3)
                rotatedPSD_matrix, M = rotation(psdMatrix, 90, psdMatrix.shape, P)
                rotatedWind_psd.setData(rotatedPSD_matrix)
                # Window intro a sub window psd for the correlations
                x0_sub = int((x_dim / 2) - (subWindStep / 2))
                y0_sub = int((x_dim / 2) - (subWindStep / 2))

                subWind_psd = wind_psd.window2D(x0_sub, y0_sub, int(x0_sub + subWindStep - 1),
                                                int(y0_sub + subWindStep - 1))

                subRotatedWind_psd = rotatedWind_psd.window2D(x0_sub, y0_sub, int(x0_sub + subWindStep - 1),
                                                              int(y0_sub + subWindStep - 1))
                # SAVE images
                filename = "tmp" + str(i) + str(j) + '.mrc'
                window_image.write(os.path.join(micFolder, filename))
                filename_subwindPSD = os.path.join(micFolder,
                                                   "tmp_psd" + str(i) + str(j) + '.mrc')
                filename_rotatedPSD = os.path.join(micFolder,
                                                   "tmp_psd_rotated" + str(i) + str(j) + '.mrc')
                subWind_psd.write(filename_subwindPSD)
                subRotatedWind_psd.write(filename_rotatedPSD)
                # Filter this window using runJob
                filename_subwindPSD_filt = os.path.join(micFolder,
                                                        "tmp_psd_filtered" + str(i) + str(j) + '.mrc')
                filename_subwindRotatedPSD_filt = os.path.join(micFolder,
                                                               "tmp_psd_rot_filtered" + str(i) + str(j) + '.mrc')

                args1 = '-i %s -o %s --fourier low_pass 0.1' % (filename_subwindPSD, filename_subwindPSD_filt)
                args2 = '-i %s -o %s --fourier low_pass 0.1' % (filename_rotatedPSD, filename_subwindRotatedPSD_filt)

                self.runJob("xmipp_transform_filter", args1)
                self.runJob("xmipp_transform_filter", args2)
                # Calculate autocorrelation 90 degrees
                subWind_psd_filt = ih.read(filename_subwindPSD_filt)
                subRotatedWind_psd_filt = ih.read(filename_subwindRotatedPSD_filt)
                autocorrelation = subWind_psd_filt.correlation(subRotatedWind_psd_filt)
                # Paint the output array
                output_array[y_steps_psd[j]:y_steps_psd[j] + subWindStep, x_steps_psd[i]:x_steps_psd[i] + subWindStep] = \
                    subWind_psd_filt.getData()
                # Append
                autocorrelations.append(autocorrelation)
                psds.append(subWind_psd_filt)

        output_image.setData(output_array)
        filename = "psd_outputs" + str(mic.getObjId()) + '.mrc'
        output_image.write(self._getExtraPath(filename))

        correlation_pairs = list(combinations(psds, 2))
        for m1, m2 in correlation_pairs:
            correlation = m1.correlation(m2)
            correlations.append(correlation)

        correlations.extend(autocorrelations)
        return correlations

    def _loadOutputSet(self, SetClass, baseName):
        """
        Load the output set if it exists or create a new one.
        """
        setFile = self._getPath(baseName)
        # -----------------Si no lo pones asi no funciona
        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            if outputSet.__len__() == 0:
                pwutils.path.cleanPath(setFile)
        # ----------------
        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

        inputMicrographs = self.inputMicrographs.get()
        outputSet.copyInfo(inputMicrographs)

        return outputSet

    # ------------------------- UTILS functions --------------------------------
    def _getAllDoneIds(self):
        doneIds = []
        acceptedIds = []
        discardedIds = []
        sizeOutput = 0

        if hasattr(self, OUTPUT_MICS):
            sizeOutput += self.outputMicrographs.getSize()
            acceptedIds.extend(list(self.outputMicrographs.getIdSet()))
            doneIds.extend(acceptedIds)

        if hasattr(self, OUTPUT_MICS_DISCARDED):
            sizeOutput += self.discardedMicrographs.getSize()
            discardedIds.extend(list(self.discardedMicrographs.getIdSet()))
            doneIds.extend(discardedIds)

        return doneIds, sizeOutput, acceptedIds, discardedIds

    def getWindowSize(self):
        """ Function to get the window size, automatically or the one set by the user. """
        if self.autoWindow:
            dimX, dimY, _ = self.getInputMicrographs().getFirstItem().getDimensions()
            halfMic = int(min(dimX, dimY)/2.5)
            windowSize = halfMic  # In case there is not a suitable option, very unlikely
            for sizeAuto in AUTOMATIC_WINDOW_SIZES:
                if sizeAuto < halfMic:  # Exit in case you found a suitable option
                    windowSize = sizeAuto
                    break
        else:
            windowSize = self.window_size.get()

        self.info('The window size used is %d' % windowSize)

        return windowSize

    def _getOutputMicFolder(self, micrograph):
        """ Create a Mic folder where to work with it. """
        return self._getTmpPath('mic_%06d' % micrograph.getObjId())

    def getInputMicrographs(self):
        return self.inputMicrographs.get()

    @staticmethod
    def getPSDs(micFolder, ID):
        """ Return the Mic folder where find the PSDs in the tmp folder. """
        filename = 'psd_outputs' + str(ID) + '.mrc'
        return os.path.join(micFolder, filename)

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


# --------------------- WORKERS --------------------------------------
def applyTransform(imag_array, M, shape):
    '''Apply a transformation(M) to a np array(imag) and return it in a given shape'''
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


def window_coordinates2D(x, y, windStep):
    x0 = 0
    xF = x - 1
    y0 = 0
    yF = y - 1
    x_coor = []
    y_coor = []

    if windStep < x and windStep < y:
        x_coor.append(x0)
        x_coor.append(x0 + windStep - 1)
        x_coor.append(xF - windStep)
        x_coor.append(xF)

        y_coor.append(y0)
        y_coor.append(y0 + windStep - 1)
        y_coor.append(yF - windStep)
        y_coor.append(yF)

        return x_coor, y_coor
    else:
        print("Dimensions not correct")
        return 0, 0


def computeStats(correlations):
    p = np.percentile(correlations, [25, 50, 75, 97.5])
    mean = np.mean(correlations)
    std = np.std(correlations)
    var = np.var(correlations)
    maxCorr = np.max(correlations)
    minCorr = np.min(correlations)

    stats = {'mean': mean,
             'std': std,
             'var': var,
             'max': maxCorr,
             'min': minCorr,
             'per25': p[0],
             'per50': p[1],
             'per75': p[2],
             'per97.5': p[3]
             }
    return stats


def setAttribute(obj, label, value):
    if value is None:
        return
    setattr(obj, label, getScipionObj(value))
