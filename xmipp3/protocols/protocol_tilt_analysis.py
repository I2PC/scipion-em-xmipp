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
from os.path import join, basename, exists
import math
from datetime import datetime
from collections import OrderedDict
from pyworkflow import VERSION_3_0
from pyworkflow.protocol import STEPS_PARALLEL
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED, FloatParam, GE, GT, Range)
import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
from pyworkflow.utils.path import moveFile, getFiles
import pyworkflow.protocol.constants as cons
from pwem.objects import SetOfMicrographs, Image, Micrograph, Acquisition, String, Set, Float
from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtMicrographs
from xmipp3 import emlib
from xmipp3.convert import getScipionObj
from matplotlib import pyplot as plt


class XmippProtTiltAnalysis(ProtMicrographs):
    """ Estimate the tilt of a micrograph, by analyzing the PSD correlations of different segments of the image.
    """
    _label = 'tilt analysis'
    _lastUpdateVersion = VERSION_3_0
    stats = {}

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

        form.addParam('window_size', IntParam, label='Window size',
                      default=1024, expertLevel=LEVEL_ADVANCED, validators=[GE(256, 'Error must be greater than 256')],
                      help='''By default, the micrograph will be divided into windows of dimensions 512x512, the PSD '''
                           '''its correlations will be computed in every segment.''')

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

        form.addHidden('saveIntermediateResults', BooleanParam, default=False, label="Save intermediate results",
                       help='''Save the micrograph segments, the PSD of those segments'''
                            ''' and the correlation statistics of those segments.''')

        form.addParallelSection(threads=4, mpi=1)

    # -------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        """ Insert the steps to perform CTF estimation, or re-estimation,
        on a set of micrographs.
        """
        self.initializeStep()
        fDeps = self._insertNewMicrographSteps(self.insertedDict,
                                               self.micsDict.values())
        # For the streaming mode, the steps function have a 'wait' flag that can be turned on/off. For example, here we insert the
        # createOutputStep but it wait=True, which means that can not be executed until it is set to False
        # (when the input micrographs stream is closed)
        waitCondition = self._getFirstJoinStepName() == 'createOutputStep'
        finalSteps = self._insertFinalSteps(fDeps)

        self._insertFunctionStep('createOutputStep',
                                 prerequisites=finalSteps, wait=waitCondition)

    def initializeStep(self):
        self.insertedDict = OrderedDict()
        self.samplingRate = self.inputMicrographs.get().getSamplingRate()
        self.micsFn = self.inputMicrographs.get().getFileName()
        self.stats = {}
        self.streamClosed = self.inputMicrographs.get().isStreamClosed()
        self.micsDict = {mic.getObjId(): mic.clone() for mic in self._loadInputList(self.micsFn).iterItems()}
        pwutils.makePath(self._getExtraPath('DONE'))

    def createOutputStep(self):
        self._closeOutputSet()

    def _loadInputList(self, micsFn):
        """ Load the input set of mics and create a list. """
        self.debug("Loading input db: %s" % micsFn)
        micSet = SetOfMicrographs(filename=micsFn)
        micSet.loadAllProperties()
        self.streamClosed = micSet.isStreamClosed()
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
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(os.path.getmtime(self.micsFn))
        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input micrographs.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and hasattr(self, 'micsDict'):
            return None

        self.lastCheck = now
        # Open input micrographs.sqlite and close it as soon as possible
        micSet = self._loadInputList(self.micsFn)

        newMicsDict = {mic.getObjId(): mic.clone() for mic in micSet.iterItems()
                       if mic.getObjId() not in self.insertedDict}
        self.micsDict.update(newMicsDict)
        outputStep = self._getFirstJoinStep()

        if len(newMicsDict) > 0:
            fDeps = self._insertNewMicrographSteps(self.insertedDict, newMicsDict.values())

            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)

            self.updateSteps()

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        newDone = [m.clone() for m in self.micsDict.values() if
                   int(m.getObjId()) not in doneList and self._isMicDone(m)]
        allDone = len(doneList) + len(newDone)
        # We have finished when there is not more input movies
        # (stream closed) and the number of processed movies is
        # equal to the number of inputs
        self.finished = self.streamClosed and allDone == len(self.micsDict)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        if newDone:
            self._writeDoneList(newDone)
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        micsAccepted = []
        micsDiscarded = []
        for mic in newDone:
            micId = mic.getObjId()
            corr_mean = Float(self.stats[micId]['mean'])
            corr_std = Float(self.stats[micId]['std'])
            corr_min = Float(self.stats[micId]['min'])
            corr_max = Float(self.stats[micId]['max'])
            psdImage = Image(location=self.getPSDs(self._getExtraPath(), micId))
            new_Mic = mic.clone()
            setAttribute(new_Mic, '_tilt_mean_corr', corr_mean)
            setAttribute(new_Mic, '_tilt_std_corr', corr_std)
            setAttribute(new_Mic, '_tilt_min_corr', corr_min)
            setAttribute(new_Mic, '_tilt_max_corr', corr_max)
            setAttribute(new_Mic, '_tilt_psds_image', psdImage)
            # Double threshold
            if corr_mean > self.meanCorr_threshold.get() and corr_std < self.stdCorr_threshold.get():
                micsAccepted.append(new_Mic)
            else:
                micsDiscarded.append(new_Mic)

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

    def _closeOutputSet(self):
        if self.hasAttribute('outputMicrographs'):
            self.outputMicrographs.close()
            self.outputMicrographs.setStreamState(Set.STREAM_CLOSED)
        if self.hasAttribute('discardedMicrographs'):
            self.discardedMicrographs.close()
            self.discardedMicrographs.setStreamState(Set.STREAM_CLOSED)

    def _insertNewMicrographSteps(self, insertedDict, inputMics):
        """ Insert steps to process new micrographs (from streaming)
        Params:
            insertedDict: contains already processed micrographs
            inputMics: input mics set to be check
        """
        deps = []
        # For each micrograph insert the step to process it
        for micrograph in inputMics:
            if micrograph.getObjId() not in insertedDict:
                tiltStepId = self._insertMicrographStep(micrograph)
                deps.append(tiltStepId)
                insertedDict[micrograph.getObjId()] = tiltStepId

        return deps

    def _insertMicrographStep(self, micrograph):
        """ Insert the processMicStep for a given movie. """
        # Note1: At this point is safe to pass the micrograph, since this
        # is not executed in parallel, here we get the params
        # to pass to the actual step that is gone to be executed later on
        # Note2: We are serializing the Movie as a dict that can be passed
        # as parameter for a functionStep
        micDict = micrograph.getObjDict(includeBasic=True)
        micStepId = self._insertFunctionStep('processMicrographStep', micDict, prerequisites=[])
        return micStepId

    def processMicrographStep(self, micDict):
        micrograph = Micrograph()
        micrograph.setAcquisition(Acquisition())
        micrograph.setAttributesFromDict(micDict, setBasic=True, ignoreMissing=True)
        micFolderTmp = self._getOutputMicFolder(micrograph)
        micFn = micrograph.getFileName()
        micName = basename(micFn)
        micDoneFn = self._getMicrographDone(micrograph)

        if self.isContinued() and os.path.exists(micDoneFn):
            self.info("Skipping micrograph: %s, seems to be done" % micFn)
            return

        pwutils.cleanPath(micDoneFn)

        pwutils.makePath(micFolderTmp)
        pwutils.createLink(micFn, join(micFolderTmp, micName))
        newMicName = self._correctFormat(micName, micFn, micFolderTmp)

        # Just store the original name in case it is needed in _processMovie
        micrograph._originalFileName = String(objDoStore=False)
        micrograph._originalFileName.set(micrograph.getFileName())
        # Now set the new filename (either linked or converted)
        micrograph.setFileName(os.path.join(micFolderTmp, newMicName))
        self.info("Processing micrograph: %s" % micrograph.getFileName())

        self._processMicrograph(micrograph)

        if self.saveIntermediateResults.get():
            micOutputFn = self._getResultsMicFolder(micrograph)
            pwutils.makePath(micOutputFn)
            for file in getFiles(micFolderTmp):
                moveFile(file, micOutputFn)

        # Mark this movie as finished
        open(micDoneFn, 'w').close()

    def _processMicrograph(self, micrograph):
        micrographId = micrograph.getObjId()
        correlations = self.calculateTiltCorrelationStep(micrograph)
        # Numpy array to compute all the
        correlations = np.asarray(correlations)
        # Calculate the mean, dev of the correlation
        stats = computeStats(correlations)
        self.stats[micrographId] = stats

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
        wind_step = self.window_size.get()
        x_steps, y_steps = window_coordinates2D(dimx, dimy, wind_step)
        subWindStep = int(wind_step * (self.samplingRate / self.objective_resolution.get()))
        x_steps_psd, y_steps_psd = window_coordinates2D(subWindStep * 2, subWindStep * 2, subWindStep)
        # Extract windows
        window_image = ImageHandler().createImage()
        rotatedWind_psd = ImageHandler().createImage()
        output_image = ImageHandler().createImage()
        output_array = np.zeros((subWindStep * 2, subWindStep * 2))
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
        filename = "psd_outputs" + str(mic.getObjId()) + '.jpeg'
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
    def _correctFormat(self, micName, micFn, micFolderTmp):
        if micName.endswith('bz2'):
            newMicName = micName.replace('.bz2', '')
            # We assume that if compressed the name ends with .mrc.bz2
            if not exists(newMicName):
                self.runJob('bzip2', '-d -f %s' % micName, cwd=micFolderTmp)

        elif micName.endswith('tbz'):
            newMicName = micName.replace('.tbz', '.mrc')
            # We assume that if compressed the name ends with .tbz
            if not exists(newMicName):
                self.runJob('tar', 'jxf %s' % micName, cwd=micFolderTmp)

        elif micName.endswith('.txt'):
            # Support a list of frame as a simple .txt file containing
            # all the frames in a raw list, we could use a xmd as well,
            # but a plain text was choose to simply its generation
            micTxt = os.path.join(micFolderTmp, micName)
            with open(micTxt) as f:
                micOrigin = os.path.basename(os.readlink(micFn))
                newMicName = micName.replace('.txt', '.mrc')
                ih = emlib.image.ImageHandler()
                for i, line in enumerate(f):
                    if line.strip():
                        inputFrame = os.path.join(micOrigin, line.strip())
                        ih.convert(inputFrame, (i + 1, os.path.join(micFolderTmp, newMicName)))
        else:
            newMicName = micName

        return newMicName

    def _insertFinalSteps(self, deps):
        """ This should be implemented in subclasses"""
        return deps

    def _getOutputMicFolder(self, micrograph):
        """ Create a Mic folder where to work with it. """
        return self._getTmpPath('mic_%06d' % micrograph.getObjId())

    def _getResultsMicFolder(self, micrograph):
        """ Create a Mic folder where to work with it. """
        return self._getExtraPath('mic_%06d' % micrograph.getObjId())

    def getInputMicrographsPointer(self):
        return self.inputMicrographs

    def getInputMicrographs(self):
        return self.getInputMicrographsPointer().get()

    def _writeFailedList(self, micList):
        """ Write to a text file the items that have failed. """
        with open(self._getAllFailed(), 'a') as f:
            for mic in micList:
                f.write('%d\n' % mic.getObjId())

    def _readDoneList(self):
        """ Read from a text file the id's of the items that have been done. """
        doneFile = self._getAllDone()
        doneList = []
        # Check what items have been previously done
        if exists(doneFile):
            with open(doneFile) as f:
                doneList += [int(line.strip()) for line in f]
        return doneList

    def _getAllDone(self):
        return self._getExtraPath('DONE_all.TXT')

    def _writeDoneList(self, micList):
        """ Write to a text file the items that have been done. """
        with open(self._getAllDone(), 'a') as f:
            for mic in micList:
                f.write('%d\n' % mic.getObjId())

    def _isMicDone(self, mic):
        """ A mic is done if the marker file exists. """
        return exists(self._getMicrographDone(mic))

    def _getMicrographDone(self, mic):
        """ Return the file that is used as a flag of termination. """
        return self._getExtraPath('DONE', 'mic_%06d.TXT' % mic.getObjId())

    @staticmethod
    def getPSDs(micFolder, ID):
        """ Return the Mic folder where find the PSDs in the tmp folder. """
        filename = 'psd_outputs' + str(ID) + '.jpeg'
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


def window_coordinates2D(x, y, wind_step):
    x0 = 0
    xF = x - 1
    y0 = 0
    yF = y - 1
    x_coor = []
    y_coor = []

    if wind_step < x and wind_step < y:
        x_coor.append(x0)
        x_coor.append(x0 + wind_step - 1)
        x_coor.append(xF - wind_step)
        x_coor.append(xF)

        y_coor.append(y0)
        y_coor.append(y0 + wind_step - 1)
        y_coor.append(yF - wind_step)
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
