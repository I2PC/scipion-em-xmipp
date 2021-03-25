# **************************************************************************
# *
# * Authors:     Federico P. de Isidro Gomez (fp.deisidro@cnb.csi.es) [1]
# *              Daniel Marchan
# *
# * [1] Centro Nacional de Biotecnologia, CSIC, Spain
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

import os
import numpy as np
from datetime import datetime
from os.path import join, basename, exists

import pyworkflow.utils as pwutils
from pyworkflow.protocol.constants import (STEPS_PARALLEL, STATUS_NEW)
import pyworkflow.protocol.params as params
from pwem.objects import SetOfMicrographs, Image, Micrograph, Acquisition, String, Set, Float
from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtMicrographs

from xmipp_base import createMetaDataFromPattern
from collections import OrderedDict
import pyworkflow.protocol.constants as cons
from xmipp3.convert import setXmippAttribute, getScipionObj, prefixAttribute
from xmipp3 import emlib


class XmippProtDeepDefocusMicrograph(ProtMicrographs):
    """Protocol to remove coordinates in carbon zones or large impurities"""
    _label = 'deep micrograph defocus'
    _conda_env = "xmipp_MicCleaner"

    def __init__(self, **kwargs):
        ProtMicrographs.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMicrographs',
                      params.PointerParam,
                      pointerClass='SetOfMicrographs',
                      important=True,
                      label="Input micrographs",
                      help='Select the SetOfMicrogrsphs ')

        form.addParam('defocusU_threshold', params.FloatParam, label='Defocus in U axis',
                      default=0.6, expertLevel=params.LEVEL_ADVANCED,
                      help='''By default, micrographs will be divided into an output set and a discarded set based
                                    on the defocus double threshold''')

        form.addParam('defocusV_threshold', params.FloatParam, label='Defocus in V axis',
                      default=0.1, expertLevel=params.LEVEL_ADVANCED,
                      help='''By default, micrographs will be divided into an output set and a discarded set based
                                            on the defocus double threshold''')

        form.addParam('test',
                      params.BooleanParam,
                      important=False,
                      label="Input micrographs",
                      help='If selected the micrographs need to have a defocus value associated')

        form.addParallelSection(threads=4, mpi=1)


    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        """ Insert the steps to perform CTF estimation, or re-estimation,
                on a set of micrographs. """
        self.insertedDict = OrderedDict()
        self.samplingRate = self.inputMicrographs.get().getSamplingRate()
        self.listOfMicrographs = []
        self._loadInputList()
        pwutils.makePath(self._getExtraPath('DONE'))
        fDeps = self._insertNewMicrographSteps(self.insertedDict,
                                               self.listOfMicrographs)
        # For the streaming mode, the steps function have a 'wait' flag that can be turned on/off. For example, here we insert the
        # createOutputStep but it wait=True, which means that can not be executed until it is set to False
        # (when the input micrographs stream is closed)
        waitCondition = self._getFirstJoinStepName() == 'createOutputStep'
        finalSteps = self._insertFinalSteps(fDeps)

        self._insertFunctionStep('createOutputStep',
                                 prerequisites=finalSteps, wait=waitCondition)


    def createOutputStep(self):
        pass


    def _loadInputList(self):
        """ Load the input set of mics and create a list. """
        micsFile = self.inputMicrographs.get().getFileName()
        self.debug("Loading input db: %s" % micsFile)
        micSet = SetOfMicrographs(filename=micsFile)
        micSet.loadAllProperties()
        newMics = [m.clone() for m in micSet if m.getObjId() not in self.insertedDict]
        self.listOfMicrographs.extend(newMics)
        self.streamClosed = micSet.isStreamClosed()
        micSet.close()
        self.debug("Closed db.")


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
    # ---------------------------STEPS checks ----------------------------------

    def _checkNewInput(self):
        # Check if there are new micrographs to process from the input set
        localFile = self.inputMicrographs.get().getFileName()
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(os.path.getmtime(localFile))
        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input micrographs.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and hasattr(self, 'listOfMicrographs'):
            return None

        self.lastCheck = now
        # Open input micrographs.sqlite and close it as soon as possible
        self._loadInputList()

        newMics = [m for m in self.listOfMicrographs if m.getObjId() not in self.insertedDict]  # SOLUTION
        newMicsBool = [len(newMics) > 0]
        outputStep = self._getFirstJoinStep()

        if newMicsBool:
            fDeps = self._insertNewMicrographSteps(self.insertedDict, newMics)  # SOLUTION

            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)

            self.updateSteps()


    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        newDone = [m.clone() for m in self.listOfMicrographs if
                   int(m.getObjId()) not in doneList and self._isMicDone(m)]

        allDone = len(doneList) + len(newDone)
        # We have finished when there is not more input movies
        # (stream closed) and the number of processed movies is
        # equal to the number of inputs
        self.finished = self.streamClosed and allDone == len(self.listOfMicrographs)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        if newDone:
            self._writeDoneList(newDone)
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        micSet = self._loadOutputSet(SetOfMicrographs, 'micrograph.sqlite')
        if self.tilt:
            micSet_discarded = self._loadOutputSet(SetOfMicrographs, 'micrograph' + 'DISCARDED' + '.sqlite')

        for mic in newDone:
            id = mic.getObjId()
            defocusU, defocusV = Float(self.defocus[id])
            new_Mic = mic.clone()
            #setattr(new_Mic, self.getDefocusULabel(), defocusU)
            #setattr(new_Mic, self.getDefocusVLabel(), defocusV)
            # Double threshold
            if defocusU > self.defocusU_threshold.get() and defocusV < self.defocusV_threshold.get():  # AND or OR
                micSet.append(new_Mic)
            else:
                if not self.tilt:
                    micSet_discarded = self._loadOutputSet(SetOfMicrographs, 'micrograph' + 'DISCARDED' + '.sqlite')
                    self.tilt = True

                micSet_discarded.append(new_Mic)

        self._updateOutputSet('outputMicrographs', micSet, streamMode)

        if self.tilt:
            self._updateOutputSet('discardedMicrographs', micSet_discarded, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)

    # --------------------------- STEPS functions ------------------------------

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
        micFolderTmp = self._getOutputMicFolder(micrograph)  # tmp/micID
        micFn = micrograph.getFileName()
        micID = micrograph.getObjId()
        micName = basename(micFn)
        micDoneFn = self._getMicrographDone(micrograph)  # EXTRAPath/Done/micrograph_ID.TXT

        if self.isContinued() and os.path.exists(micDoneFn):
            self.info("Skipping micrograph: %s, seems to be done" % micFn)
            return

        pwutils.cleanPath(micDoneFn)

        if self._filterMicrograph(micrograph):
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

            #if self.saveIntermediateResults.get():
            #    micOutputFn = self._getResultsMicFolder(micrograph)  # ExtraPath/micID
            #    pwutils.makePath(micOutputFn)
            #    for file in getFiles(micFolderTmp):
            #       moveFile(file, micOutputFn)
            #else:
            #    moveFile(self.getPSDs(micFolderTmp, micID), self._getExtraPath())

        # Mark this movie as finished
        open(micDoneFn, 'w').close()

    def _processMicrograph(self, micrograph):
        micrographId = micrograph.getObjId()

        correlations = self.calculateTiltCorrelationStep(micrograph)

        # Numpy array to compute all the
        correlations = np.asarray(correlations)
        # Calculate the mean, dev of the correlation
        stats = computeStats(correlations)
        self.mean_correlations.append(stats['mean'])
        self.stats[micrographId] = stats





        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")

        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")

        #fhSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
         #               (micrographId, stats['mean'], stats['std'], stats['min'], stats['max']))
        #fhSummary.close()

        #fnMonitorSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
         #                      (micrographId, stats['mean'], stats['std'], stats['min'], stats['max']))
        fnMonitorSummary.close()



    def _computeMaskForMicrographList(self, micList):
        """ Functional Step. Overrided in general protExtracParticle """

        micsFnDone = self.getDoneMics()
        micLisfFn = [mic.getFileName() for mic in micList
                     if not pwutils.removeBaseExt(mic.getFileName()) in micsFnDone]

        if len(micLisfFn) > 0:
            inputMicsPathMetadataFname = self._getTmpPath("inputMics" + str(hash(micLisfFn[0])) + ".xmd")
            mics_md = createMetaDataFromPattern(micLisfFn)
            mics_md.write(inputMicsPathMetadataFname)
            args = '-i %s' % inputMicsPathMetadataFname
            args += ' -c %s' % self._getExtraPath('inputCoords')
            args += ' -o %s' % self._getExtraPath('outputCoords')
            args += ' -b %d' % self.getBoxSize()
            args += ' -s 1'  # Downsampling is automatically managed by scipion
            args += ' -d %s' % self.getModel('deepMicrographCleaner', 'defaultModel.keras')

            if self.threshold.get() > 0:
                args += ' --deepThr %f ' % (1 - self.threshold.get())

            if self.saveMasks.get():
                args += ' --predictedMaskDir %s ' % (self._getExtraPath("predictedMasks"))

            if self.useGpu.get():
                if self.useQueueForSteps() or self.useQueue():
                    args += ' -g all '
                else:
                    args += ' -g %s' % (",".join([str(elem) for elem in self.getGpuList()]))
            else:
                args += ' -g -1'

            self.runJob('xmipp_deep_micrograph_cleaner', args)



    # --------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = self.validateDLtoolkit(assertModel=True,
                                        model=('deepMicrographCleaner', 'defaultModel.keras'))

        batchSize = self.streamingBatchSize.get()
        if batchSize == 1:
            errors.append('Batch size must be 0 (all at once) or larger than 1.')
        elif not self.isInStreaming() and batchSize > len(self.inputCoordinates.get().getMicrographs()):
            errors.append('Batch size (%d) must be <= that the number of micrographs '
                          '(%d) in static mode. Set it to 0 to use only one batch'
                          % (batchSize, self._getNumPickedMics()))

        return errors


    def _citations(self):
        return ['***']


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



    # --------------------------- UTILS functions ------------------------------
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
                ih = ImageHandler()
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
    def getDefocusULabel(): #Cambiar a una constante que exista
        return prefixAttribute(emlib.label2Str(emlib.MDL_TILT_ANALYSIS_MAX))

    @staticmethod
    def getDefocusVLabel(): #Cambiar a una constante que exista
        return prefixAttribute(emlib.label2Str(emlib.MDL_TILT_ANALYSIS_PSDs))


