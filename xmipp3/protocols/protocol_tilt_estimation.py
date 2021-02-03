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
from itertools import combinations
import os
from os.path import join, basename, exists
import math
from datetime import datetime
from scipy import stats


from pyworkflow import VERSION_1_1
from pyworkflow.object import Set
from pyworkflow.protocol import STEPS_PARALLEL, params
from pyworkflow.protocol.params import (PointerParam, IntParam,
                                        BooleanParam, LEVEL_ADVANCED)

import pyworkflow.utils as pwutils
from pyworkflow.utils.properties import Message
from pyworkflow.utils.path import moveFile, getFiles
import pyworkflow.protocol.constants as cons


from pwem.objects import SetOfMicrographs, SetOfImages, Image, Micrograph, Acquisition, String, Set
from pwem.emlib.image import ImageHandler
from pwem.protocols import EMProtocol,ProtPreprocessMicrographs , ProtMicrographs

from pwem import emlib
import pwem.constants as emcts

from xmipp3.utils import normalize_array
from xmipp3 import emlib
from xmipp3.convert import setXmippAttribute



class XmippProtTiltEstimation(ProtMicrographs):
    """ Estimate the tilt of a micrograph, by analyzing the PSD correlations of different segments of the image.
    """
    _label = 'tilt estimation'
    _lastUpdateVersion = VERSION_1_1
    registeredFiles = []
    mean_correlations = []
    stats = {}
    #micDict = {}
    insertedDict = {}


    def __init__(self, **args):
        ProtMicrographs.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMicrographs', PointerParam,
                      pointerClass='SetOfMicrographs, Micrograph',
                      label="Input micrographs", important=True,
                      help='Select the SetOfMicrograph to be preprocessed.')

        form.addParam('window_size', IntParam, label='Window size',
                      default=1024, expertLevel=LEVEL_ADVANCED,
                      help='''By default, the micrograph will be divided into windows of size 1024x1024, 
                            the PSD and its correlations will be computed in every segment.''')

        form.addParam('saveMediumResults', BooleanParam, default=False,
                      label="Save intermediate results", expertLevel=LEVEL_ADVANCED,
                      help='''Save the micrograph segments, the PSD of those segments
                           and the correlation statistics of those segments''')

        form.addParallelSection(threads=3, mpi=1) #poner aqui 4

    # -------------------------- STEPS functions ------------------------------

    def createOutputStep(self):
        pass



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
        micStepId = self._insertFunctionStep('processMicrographStep', micDict)

        return micStepId


    def processMicrographStep(self, micDict):
        micrograph = Micrograph()
        micrograph.setAcquisition(Acquisition())
        micrograph.setAttributesFromDict(micDict, setBasic=True, ignoreMissing=True)
        micFolder = self._getOutputMicFolder(micrograph) #tmp/micID
        micFn = micrograph.getFileName()
        micName = basename(micFn)
        micDoneFn = self._getMicrographDone(micrograph) #EXTRAPath/Done_micrograph_ID.TXT

        if self.isContinued() and os.path.exists(micDoneFn):
            self.info("Skipping micrograph: %s, seems to be done" % micFn)
            return

        # Clean old finished files
        pwutils.cleanPath(micDoneFn)

        if self._filterMicrograph(micrograph):
            pwutils.makePath(micFolder)
            pwutils.createLink(micFn, join(micFolder, micName))

        #-------------------------------------ESTO creo q no hace falta
            if micName.endswith('bz2'):
                newMicName = micName.replace('.bz2', '')
                # We assume that if compressed the name ends with .mrc.bz2
                if not exists(newMicName):
                    self.runJob('bzip2', '-d -f %s' % micName, cwd=micFolder)

            elif micName.endswith('tbz'):
                newMicName = micName.replace('.tbz', '.mrc')
                # We assume that if compressed the name ends with .tbz
                if not exists(newMicName):
                    self.runJob('tar', 'jxf %s' % micName, cwd=micFolder)

            elif micName.endswith('.txt'):
                # Support a list of frame as a simple .txt file containing
                # all the frames in a raw list, we could use a xmd as well,
                # but a plain text was choose to simply its generation
                micTxt = os.path.join(micFolder, micName)
                with open(micTxt) as f:
                    micOrigin = os.path.basename(os.readlink(micFn))
                    newMicName = micName.replace('.txt', '.mrc')
                    ih = emlib.image.ImageHandler()
                    for i, line in enumerate(f):
                        if line.strip():
                            inputFrame = os.path.join(micOrigin, line.strip())
                            ih.convert(inputFrame, (i+1, os.path.join(micFolder, newMicName)))
            #-----------------------------HASTA AQUI
            else:
                newMicName = micName

            # Just store the original name in case it is needed in _processMovie
            micrograph._originalFileName = String(objDoStore=False)
            micrograph._originalFileName.set(micrograph.getFileName())
            # Now set the new filename (either linked or converted)
            micrograph.setFileName(os.path.join(micFolder, newMicName))
            self.info("Processing micrograph: %s" % micrograph.getFileName())

            self._processMicrograph(micrograph)

            if self._doMicFolderCleanUp():
                self._cleanMicFolder(micFolder)

        # Mark this movie as finished
        open(micDoneFn, 'w').close()


    def _processMicrograph(self, micrograph):
        #REMEBER TO COPY THE IMPORTANT FILES TO EXTRA_PATH WE ARE IN TMP NOW
        micrographId = micrograph.getObjId()
        fnMicrograph = micrograph.getFileName()
        micFolder = self._getResultsMicFolder(micrograph)
        micTmpFolder = self._getOutputMicFolder(micrograph)
        pwutils.makePath(micFolder)

        correlations = self.calculateTiltCorrelationStep(micrograph)
        # Numpy array to compute all the
        correlations = np.asarray(correlations)
        # Calculate the mean, dev of the correlation
        stats = computeStats(correlations)

        #EStoy entre uno y otro
        self.mean_correlations.append(stats['mean'])
        self.stats[micrographId] = stats

        if self.saveMediumResults.get():
            for file in getFiles(micTmpFolder):
                moveFile(file, micFolder)
        else:
            #Store only the statistics of the correlation object
            pass

        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")

        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")

        fhSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
                        (micrographId, stats['mean'], stats['dev'], stats['min'], stats['max']))
        fhSummary.close()

        fnMonitorSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
                        (micrographId, stats['mean'], stats['dev'], stats['min'], stats['max']))
        fnMonitorSummary.close()


    def calculateTiltCorrelationStep(self, mic):
        windows = []
        psds = []
        rotated_psds = []
        correlations = []
        autocorrelations = []
        # read image
        micFn = mic.getFileName()
        micName = mic.getMicName()
        micFolder = self._getOutputMicFolder(mic)  # tmp/micID

        micImage = ImageHandler().read(mic.getLocation())  # This is an Xmipp Image DATA
        micMatrix = micImage.getData()
        dimx, dimy, z, n = micImage.getDimensions()
        wind_step = self.window_size.get()
        overlap = 0.7
        x_steps, y_steps = window_coordinates2D(dimx, dimy, wind_step, overlap)

        # Extract windows
        for x0 in x_steps:
            for y0 in y_steps:
                window = micImage.window2D(x0, y0, x0 + (wind_step - 1), y0 + (wind_step - 1))
                mean, dev, min, max = window.computeStats()
                x_dim, y_dim, z, n = window.getDimensions()
                # NORMALIZED
                winMatrix = (window.getData() - mean) / dev  # Esta normalización se podría normalizar
                window_image = ImageHandler().createImage()
                window_image.setData(winMatrix)
                # Compute PSD
                wind_psd = window_image.computePSD(0.4, x_dim, y_dim, 1)
                wind_psd.convertPSD()
                # Rotate PSD
                psdMatrix = wind_psd.getData()
                P = np.identity(3)
                rotatedPSD_matrix, M = rotation(psdMatrix, 90, psdMatrix.shape, P)
                rotatedwind_psd = ImageHandler().createImage()
                rotatedwind_psd.setData(rotatedPSD_matrix)
                # Calculate autocorrelation 90 degrees
                autocorrelation = wind_psd.correlation(rotatedwind_psd)
                print("Autocorrelation with rotation: " + str(autocorrelation))
                autocorrelations.append(autocorrelation)
                # SAVE images
                filename = "tmp" + str(x_steps.index(x0)) + str(y_steps.index(y0)) + '.mrc'
                window_image.write(os.path.join(micFolder, filename))
                filename =  "tmp_psd" + str(x_steps.index(x0)) + str(y_steps.index(y0)) + '.mrc'
                wind_psd.write(os.path.join(micFolder, filename))
                filename =  "tmp_psd_rotated" + str(x_steps.index(x0)) + str(y_steps.index(y0)) + '.mrc'
                rotatedwind_psd.write(os.path.join(micFolder, filename))
                # Append
                windows.append(window_image)
                psds.append(wind_psd)
                rotated_psds.append(rotatedwind_psd)

        correlation_pairs = list(combinations(psds, 2))
        print("Correlation all the combinations with rotation 90")
        for m1, m2 in correlation_pairs:
            correlation = m1.correlation(m2)
            print(correlation)
            correlations.append(correlation)  # check this

        correlations.extend(autocorrelations)

        return correlations


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
        #localFile = self.getInputMicrographs().getFileName()
        localFile = self.inputMicrographs.get().getFileName()
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(os.path.getmtime(localFile))
        self.debug('Last check: %s, modification: %s'
                % (pwutils.prettyTime(self.lastCheck),
                    pwutils.prettyTime(mTime)))
        # If the input micrographs.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and hasattr(self, 'listOfMics'):
            return None

        self.lastCheck = now
        # Open input micrographs.sqlite and close it as soon as possible
        self._loadInputList()

        newMics = any(m.getObjId() not in self.insertedDict
                        for m in self.listOfMicrographs)

        outputStep = self._getFirstJoinStep()

        if newMicss:
            fDeps = self._insertNewMicrographSteps(self.insertedDict,
                                               self.listOfMicrographs)
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)

            self.updateSteps()


    def _loadInputList(self):
        """ Load the input set of mics and create a list. """
        micsFile = self.inputMicrographs.get().getFileName()
        self.debug("Loading input db: %s" % micsFile)
        micSet = SetOfMicrographs(filename=micsFile)
        micSet.loadAllProperties()
        self.listOfMicrographs = [m.clone() for m in micSet]
        self.streamClosed = micSet.isStreamClosed()
        micSet.close()
        self.debug("Closed db.")


    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        newDone = [m.clone() for m in self.listOfMicrographs
                    if int(m.getObjId()) not in doneList and
                    self._isMicDone(m)]

        allDone = len(doneList) + len(newDone)
        # We have finished when there is not more input movies
        # (stream closed) and the number of processed movies is
        # equal to the number of inputs
        self.finished = self.streamClosed and allDone == len(self.listOfMovies)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        if newDone:
            self._writeDoneList(newDone)
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        micSet = self._loadOutputSet(SetOfMicrographs,
                                        'micrograph.sqlite')

        for mic in newDone:
            id = mic.getObjId()
            corr_mean = Float(self.stats[id]['mean'])
            new_Mic = mic.clone()
            setXmippAttribute(new_Mic, emlib.MDL_TILT_ESTIMATION, corr_mean)
            micSet.append(new_Mic)
            # AQUI DEBERIA IR EL setXMIPP_atribute

        self._updateOutputSet('outputMicrographs', micSet, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)


    def _loadOutputSet(self, SetClass, baseName):
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

        inputMicrographs = self.inputMicrographs.get()
        outputSet.copyInfo(inputMicrographs)

        return outputSet


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


 # --------------------------- OVERRIDE functions --------------------------
    def _filterMicrograph(self, micrograph):
        """ Check if process or not this movie.
        """
        return True

    def _doMicFolderCleanUp(self):
        """ This functions allows subclasses to change the default behaviour
        of cleanup the movie folders after the _processMovie function.
        In some cases it makes sense that the protocol subclass take cares
        of when to do the clean up.
        """
        return True

    def _cleanMicFolder(self, micFolder):
        if pwutils.envVarOn(SCIPION_DEBUG_NOCLEAN):
            self.info('Clean micrograph data DISABLED. '
                      'Micrograph folder will remain in disk!!!')
        else:
            self.info("Erasing.....micrographFolder: %s" % micFolder)
            os.system('rm -rf %s' % micFolder)
            # cleanPath(movieFolder)
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


def window_coordinates2D(x, y, wind_step, overlap):
    x0 = 0
    xF = wind_step - 1
    y0 = 0
    yF = wind_step - 1
    x_coor = []
    y_coor = []
    if wind_step < x and wind_step < y:
        x_coor.append(x0)
        while xF != (x - 1):
            x0 = x0 + wind_step
            xF = xF + wind_step
            if xF > (x - 1):
                if (((xF - x) / wind_step) < overlap):
                    xF = x - 1
                    x0 = x - wind_step
                else:
                    break
            x_coor.append(x0)

        y_coor.append(y0)
        while yF != (y - 1):
            y0 = y0 + wind_step
            yF = yF + wind_step
            if yF > (y - 1):
                if (((yF - y) / wind_step) < overlap):
                    yF = y - 1
                    y0 = y - wind_step
                else:
                    break
            y_coor.append(y0)
        return x_coor, y_coor
    else:
        print("Dimensions not correct")
        return 0, 0

def computeStats(correlations):
    p = np.percentile(correlations, [25, 50, 75, 97.5])
    mean = np.mean(correlations)
    std = np.std(correlations)
    var = np.var(correlations)
    max = np.max(correlations)
    min = np.min(correlations)

    stats ={'mean': mean,
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





