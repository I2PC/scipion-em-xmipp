# **************************************************************************
# *
# * Authors:     Tomas Majtner (tmajtner@cnb.csic.es)
# *              David Maluenda (dmaluenda@cnb.csic.es)
# *              Daniel Marchán (da.marchan@cnb.csic.es)
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

import os
import time
from datetime import datetime

import pyworkflow.protocol.constants as cons
from pyworkflow import VERSION_3_0
from pyworkflow.protocol import Protocol
from pwem.protocols import EMProtocol
from pyworkflow.object import Set
from pyworkflow.protocol.params import BooleanParam, IntParam, PointerParam, GT

SIGNAL_FILENAME = "STOP_STREAM.TXT"

class XmippProtTriggerData(EMProtocol, Protocol):
    """
    Waits until certain number of images is prepared and then
    send them to output.
    It can be done in 3 ways:
        - If "Send all items to output?" is _No_:
            Once the number of items is reached, a setOfImages is returned and
            the protocol finishes (ending the streaming from this point).
        - If "Send all items to output?" is _Yes_ and:
            - If "Split items to multiple sets?" is _Yes_:
                Multiple closed outputs will be returned as soon as
                the number of items is reached.
            - If "Split items to multiple sets?" is _No_:
                Only one output is returned and it is growing up in batches of
                a certain number of items (completely in streaming).
    """
    _label = 'trigger data'
    _lastUpdateVersion = VERSION_3_0

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):

        form.addSection(label='Input')

        form.addParam('inputImages', PointerParam,
                      pointerClass='SetOfImages',
                      label='Input images', important=True)
        form.addParam('triggerWait', BooleanParam, default=False,
                      label='Wait for signal to stop the stream?',
                      help='If NO is selected, normal functionality.\n'
                           'If YES is selected it will wait for a signal to stop the stream.'
                           '\n For this option, select send all items to output with a '
                           'minimum size of 1')
        form.addParam('outputSize', IntParam, default=10000,
                      label='Minimum output size',
                      help='How many particles need to be on input to '
                           'create output set.')
        form.addParam('allImages', BooleanParam, default=True,
                      label='Send all items to output?',
                      help='If NO is selected, only a closed subset of '
                           '"Output size" items will be send to output.\n'
                           'If YES is selected it will still running in streaming.')
        form.addParam('splitImages', BooleanParam, default=False,
                      label='Split items to multiple sets?',
                      condition='allImages',
                      help='If YES is selected, multiple closed outputs of '
                           '"Output size" are returned.\n'
                           'If NO is selected, only one open and growing output '
                           'is returned')
        form.addParam('triggerSignal', BooleanParam, default=False,
                      label='Send signal to stop a stream?',
                      help='If NO is selected, normal functionality.\n'
                           'If YES is selected it will send a signal to a connected Trigger data protocol.'
                           '\n For this option, select the option send all items to output.')
        form.addParam('triggerProt', PointerParam,
                      pointerClass=self.getClassName(),
                      condition='triggerSignal',
                      label='Trigger data protocol',
                      help='Select the trigger data protocol that you will send a signal to stop the stream.')
        form.addParam('delay', IntParam, default=10, label="Delay (sec)",
                      validators=[GT(3, "must be larger than 3sec.")],
                      help="Delay in seconds before checking new output")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        # initializing variables
        self.finished = False
        self.images = []
        self.splitedImages = []
        self.outputCount = 0
        self.setImagesClass()
        self.setImagesType()

        # steps
        imsSteps = self._insertFunctionStep('delayStep')
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=[imsSteps], wait=True)

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    def createOutputStep(self):
        self._closeOutputSet()

    def _checkNewInput(self):
        imsFile = self.inputImages.get().getFileName()
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = datetime.fromtimestamp(os.path.getmtime(imsFile))

        # If the input's sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and hasattr(self, 'newImages'):
            return None

        # loading the input set in a dynamic way
        inputClass = self.getImagesClass()
        self.imsSet = inputClass(filename=imsFile)
        self.imsSet.loadAllProperties()

        # loading new images to process
        if len(self.images) > 0:  # taking the non-processed yet
            extraLimitLen = -1 if self.allImages.get() else self.outputSize.get() - len(self.images)
            self.newImages = [m.clone() for m in self.imsSet.iterItems(
                orderBy='creation',
                where='creation>"' + str(self.check) + '"',
                limit=extraLimitLen)]
        else:  # first time
            limitLen = -1 if self.allImages.get() else self.outputSize.get()
            self.newImages = [m.clone() for m in self.imsSet.iterItems(
                orderBy='creation',
                limit=limitLen)]

        self.splitedImages = self.splitedImages + self.newImages
        self.images = self.images + self.newImages
        if len(self.newImages) > 0:
            for item in self.imsSet.iterItems(orderBy='creation',
                                              direction='DESC'):
                self.check = item.getObjCreation()
                break

        self.lastCheck = datetime.now()
        self.streamClosed = self.imsSet.isStreamClosed()
        self.imsSet.close()

        # filling the output if needed
        self._fillingOutput()

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return

        if self.streamClosed:
            self.finished = True
        elif not self.allImages.get() and not self.triggerSignal.get():
            self.finished = len(self.images) >= self.outputSize
        else:
            self.finished = False

        # Send the signal to the connected protocol
        if self.triggerSignal.get() and len(self.images) >= self.outputSize:
            self.info('Sending signal to stop the input trigger data protocol')
            self.stopWait()

        # Wait for trigger data signal
        if self.triggerWait.get():
            self.info('Waiting for signal to stop the stream')
            if self.waitingHasFinished():
                self.info('Stopped by received signal from a trigger data protocol')
                self.finished = True

        outputStep = self._getFirstJoinStep()
        deps = []
        if self.finished:  # Unlock createOutputStep if finished all jobs
            self._fillingOutput()  # To do the last filling
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)
        else:
            delayId = self._insertFunctionStep('delayStep', prerequisites=[])
            deps.append(delayId)

        if outputStep is not None:
            outputStep.addPrerequisites(*deps)
        self.updateSteps()

    def _fillingOutput(self):
        imsSqliteFn = '%s.sqlite' % self.getImagesType('lower')
        outputName = self.getOututName()
        if len(self.images) >= self.outputSize or self.finished:
            if self.allImages:  # Streaming and semi-streaming
                if self.splitImages:  # Semi-streaming: Splitting the input
                    if len(self.splitedImages) >= self.outputSize or \
                            (self.finished and len(self.splitedImages) > 0):
                        splitLimIndex = self.outputSize.get() if not self.finished else None
                        numIter = 1 if len(self.splitedImages) < self.outputSize.get() \
                            else int(len(self.splitedImages) / self.outputSize.get())
                        for _ in range(numIter):
                            self.outputCount += 1
                            imageSet = self._loadOutputSet(self.getImagesClass(),
                                                           '%s%d.sqlite'
                                                           % (self.getImagesType('lower'),
                                                              self.outputCount),
                                                           self.splitedImages[:splitLimIndex
                                                                              or len(self.splitedImages)])
                            # The splitted outputSets are always closed
                            self._updateOutputSet("%s%d" % (outputName, self.outputCount),
                                                  imageSet, Set.STREAM_CLOSED)
                            self.splitedImages = self.splitedImages[splitLimIndex:] if splitLimIndex else []
                else:  # Full streaming case
                    if not os.path.exists(self._getPath(imsSqliteFn)):
                        imageSet = self._loadOutputSet(self.getImagesClass(),
                                                       imsSqliteFn,
                                                       self.images)
                    else:
                        # if finished no images to add, but we need to close the set
                        imagesToAdd = self.newImages if not self.finished else []
                        imageSet = self._loadOutputSet(self.getImagesClass(),
                                                       imsSqliteFn,
                                                       imagesToAdd)
                    streamMode = Set.STREAM_CLOSED if self.finished else \
                        Set.STREAM_OPEN
                    self._updateOutputSet(outputName, imageSet, streamMode)

            elif not os.path.exists(self._getPath(imsSqliteFn)):
                imageSet = self._loadOutputSet(self.getImagesClass(), imsSqliteFn,
                                               self.images)
                # The outputSet is always closed here
                self._updateOutputSet(outputName, imageSet, Set.STREAM_CLOSED)

    def _loadOutputSet(self, SetClass, baseName, newImages):
        setFile = self._getPath(baseName)
        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

        inputs = self.inputImages.get()
        outputSet.copyInfo(inputs)
        outputSet.copyItems(newImages)
        return outputSet

    # --------------------------- INFO functions ------------------------------
    def _summary(self):
        summary = []

        outputStr = self.getOututName()

        if self.allImages.get() and not self.splitImages.get():
            summary.append("MODE: *full streaming*.")
            triggeredMsg = ("'%s' released, it will be growing up "
                            "as soon as the input does." % outputStr)
        elif self.splitImages.get():
            summary.append("MODE: *semi streaming (batches)*.")
            triggeredMsg = ("%d '%s' are being released with %d items, "
                            "each. A new batch will be created when ready."
                            % (self.getOutputsSize(), outputStr, self.outputSize))
        else:
            summary.append("MODE: *static output*.")
            triggeredMsg = ("'%s' released and closed. Nothing else to do."
                            % outputStr)

        if self.getOutputsSize():
            summary.append(triggeredMsg)
        else:
            inputStr = self.getImagesType() if self.inputImages.get() else '(or not ready)'
            summary.append("Not enough input %s to release an output, yet."
                           % inputStr)
            summary.append("At least, %d items are needed to trigger an output."
                           % self.outputSize.get())

        if (self.isFinished() and
                self.outputSize.get() > [o for o in
                                         self.iterOutputAttributes()][0][1].getSize()):
            summary.append("Output released because streaming finished.")

        return summary

    def _validate(self):
        errors = []
        if self.triggerSignal.get():
            if not isinstance(self.triggerProt.get(), XmippProtTriggerData):
                errors.append("There is not a Trigger protocol connected to send a stop signal.")

    # --------------------------- UTILS functions -----------------------------
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

    def delayStep(self):
        time.sleep(self.delay)

    def setImagesClass(self):
        self._inputClass = self.inputImages.get().getClass()

    def getImagesClass(self):
        return self._inputClass

    def setImagesType(self):
        inputSet = self.inputImages.get()
        if inputSet:
            inputClassName = inputSet.getClassName()
            self._inputType = inputClassName.split('SetOf')[1]
        else:
            self._inputType = None

    def getImagesType(self, letters='default'):
        if not self.hasAttribute('_inputType'):
            self.setImagesType()
        typeStr = str(self._inputType)
        if letters == 'lower':
            return typeStr.lower()
        else:
            return typeStr

    def getOututName(self):
        return 'output%s' % self.getImagesType()

    def stopWait(self):
        f = open(self._getStopConnectingFilename(), 'w')
        f.close()

    def _getStopConnectingFilename(self):
        triggerProtocol = self.triggerProt.get()
        if triggerProtocol is not None:
            fileName = triggerProtocol._getExtraPath(SIGNAL_FILENAME)
        else:
            fileName = None

        return fileName

    def waitingHasFinished(self):
        return os.path.exists(self._getStopWaitingFilename())

    def _getStopWaitingFilename(self):
        return self._getExtraPath(SIGNAL_FILENAME)
