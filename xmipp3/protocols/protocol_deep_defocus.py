# **************************************************************************
# *
# * Authors:     Federico P. de Isidro Gomez (fp.deisidro@cnb.csi.es) [1]
# *              Daniel Marchan
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

import pyworkflow.utils as pwutils
from pyworkflow.protocol.constants import (STEPS_PARALLEL, STATUS_NEW)
import pyworkflow.protocol.params as params
from pwem.protocols import ProtExtractParticles
from pyworkflow.object import Set, Pointer

from xmipp3.base import XmippProtocol
from xmipp_base import createMetaDataFromPattern
from xmipp3.convert import (writeMicCoordinates, readSetOfCoordinates)
from xmipp3.constants import SAME_AS_PICKING, OTHER


class XmippProtDeepMicrographScreen(ProtExtractParticles, XmippProtocol):
    """Protocol to remove coordinates in carbon zones or large impurities"""
    _label = 'deep micrograph cleaner'
    _conda_env = "xmipp_MicCleaner"

    def __init__(self, **kwargs):
        ProtExtractParticles.__init__(self, **kwargs)
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

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        # self._insertFunctionStep('fun1', param)
        # self._insertFunctionStep('fun2', param)
        pass

    # def _insertInitialSteps(self):
    #     # Just overwrite this function to load some info
    #     # before the actual processing
    #     pwutils.makePath(self._getExtraPath('inputCoords'))
    #     pwutils.makePath(self._getExtraPath('outputCoords'))
    #     if self.saveMasks.get():
    #         pwutils.makePath(self._getExtraPath("predictedMasks"))
    #     self._setupBasicProperties()
    #
    #     return []

    # --------------------------- STEPS functions ------------------------------
    def _insertNewMicsSteps(self, inputMics):
        """ Insert steps to process new mics (from streaming)
        Params:
            inputMics: input mics set to be check
        """
        return self._insertNewMics(inputMics,
                                   lambda mic: mic.getMicName(),
                                   self._insertExtractMicrographStepOwn,
                                   self._insertExtractMicrographListStepOwn,
                                   *self._getExtractArgs())

    def _insertExtractMicrographStepOwn(self, mic, prerequisites, *args):
        raise ValueError("Batch size must be >1")

    def _insertExtractMicrographListStepOwn(self, micList, prerequisites, *args):
        """ Basic method to insert a picking step for a given micrograph. """
        return self._insertFunctionStep('extractMicrographListStepOwn',
                                        [mic.getMicName() for mic in micList],
                                        *args, prerequisites=prerequisites)

    def extractMicrographListStepOwn(self, micKeyList, *args):
        micList = []

        for micName in micKeyList:
            mic = self.micDict[micName]
            micDoneFn = self._getMicDone(mic)
            micFn = mic.getFileName()
            coordList = self.coordDict[mic.getObjId()]
            self._convertCoordinates(mic, coordList)
            if self.isContinued() and os.path.exists(micDoneFn):
                self.info("Skipping micrograph: %s, seems to be done" % micFn)

            else:
                # Clean old finished files
                pwutils.cleanPath(micDoneFn)
                self.info("Extracting micrograph: %s " % micFn)
                micList.append(mic)

        self._computeMaskForMicrographList(micList, *args)

        for mic in micList:
            # Mark this mic as finished
            open(self._getMicDone(mic), 'w').close()

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

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return

        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        newDone = [m for m in self.micDict.values()
                   if m.getObjId() not in doneList and self._isMicDone(m)]

        # Update the file with the newly done mics
        # or exit from the function if no new done mics
        inputLen = len(self.micDict)
        self.debug('_checkNewOutput: ')
        self.debug('   input: %s, doneList: %s, newDone: %s'
                   % (inputLen, len(doneList), len(newDone)))

        firstTime = len(doneList) == 0
        allDone = len(doneList) + len(newDone)
        # We have finished when there is not more input mics (stream closed)
        # and the number of processed mics is equal to the number of inputs
        streamClosed = self._isStreamClosed()
        self.finished = streamClosed and allDone == inputLen
        self.debug(' is finished? %s ' % self.finished)
        self.debug(' is stream closed? %s ' % streamClosed)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        if newDone:
            self._updateOutputCoordSet(newDone, streamMode)
            self._writeDoneList(newDone)
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here

            # Maybe it would be good idea to take a snap to avoid
            # so much IO if this protocol does not have much to do now
            if allDone == len(self.micDict):
                self._streamingSleepOnWait()

            return

        self.debug('   finished: %s ' % self.finished)
        self.debug('        self.streamClosed (%s) AND' % streamClosed)
        self.debug('        allDone (%s) == len(self.listOfMics (%s)'
                   % (allDone, inputLen))
        self.debug('   streamMode: %s' % streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs

            # Close the output set
            self._updateOutputCoordSet([], Set.STREAM_CLOSED)
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

    def _getScale(self):
        if self.micsSource == SAME_AS_PICKING or self.useOtherScale.get() == 1:
            scale = 1
        else:
            scale = (1. / self.getBoxScale())
        return scale

    def _updateOutputCoordSet(self, micList, streamMode):

        # Do no proceed if there is not micrograph ready
        if not micList:
            return []

        outputDir = self._getExtraPath('outputCoords')
        outputCoords = self.getOutput()

        # If there are not outputCoordinates yet, it means that is the first
        # time we are updating output coordinates, so we need to first create
        # the output set
        firstTime = outputCoords is None

        if firstTime:
            if self.micsSource == SAME_AS_PICKING or self.useOtherScale.get() == 1:
                boxSize = self.getBoxSize()
                micSetPtr = self.getInputMicrographs()
            else:
                boxSize = self.inputCoordinates.get().getBoxSize()
                micSetPtr = self.inputCoordinates.get().getMicrographs()

            outputCoords = self._createSetOfCoordinates(micSetPtr,
                                                        suffix=self.getAutoSuffix())
            outputCoords.copyInfo(self.inputCoordinates.get())
            outputCoords.setBoxSize(boxSize)
        else:
            outputCoords.enableAppend()
        self.info("Reading coordinates from mics: %s" % ','.join([mic.strId() for mic in micList]))
        readSetOfCoordinates(outputDir, micList, outputCoords, scale=self._getScale())
        self.debug(" _updateOutputCoordSet Stream Mode: %s " % streamMode)
        self._updateOutputSet(self.getOutputName(), outputCoords, streamMode)

        if firstTime:
            self._defineSourceRelation(micSetPtr,
                                       outputCoords)

        return micList

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
        summary = []
        summary.append("Micrographs source: %s"
                       % self.getEnumText("micsSource"))
        summary.append("Coordinates scale: %d" % (1. / self.getBoxScale()))

        return summary

    def _methods(self):
        methodsMsgs = []

        return methodsMsgs

    # --------------------------- UTILS functions ------------------------------
    def _convertCoordinates(self, mic, coordList):
        writeMicCoordinates(mic, coordList, self._getMicPos(mic),
                            getPosFunc=self._getPos)

