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

    AI Generated

    ## Overview

    The Trigger Data protocol controls when a set of images is released to the next
    step of a workflow.

    In streaming cryo-EM workflows, images may arrive progressively. Sometimes a
    downstream protocol should not start immediately with the first image, but only
    after a minimum number of items has accumulated. This protocol acts as a
    trigger: it waits until the input set contains enough images and then creates
    one or more output sets.

    The protocol can work in several modes:

    - release one closed subset and finish;
    - release one output set that keeps growing in streaming mode;
    - release multiple closed batches;
    - wait for a stop signal from another Trigger Data protocol;
    - send a stop signal to another Trigger Data protocol.

    This makes the protocol useful for coordinating streaming workflows, batching
    data, delaying downstream execution, or stopping a stream when another branch
    has collected enough data.

    ## Inputs and General Workflow

    The input is a Scipion **SetOfImages**.

    The protocol periodically checks the input set. When enough new images are
    available, it creates or updates the corresponding output set. The minimum
    number of items required to release an output is controlled by the **Minimum
    output size** parameter.

    Depending on the selected options, the output may be a single closed set, a
    single growing streaming set, or several closed batches.

    The protocol also includes optional signaling between Trigger Data protocols.
    One Trigger Data run can create a signal file that tells another Trigger Data
    run to stop waiting and close its output.

    ## Input Images

    The **Input images** parameter defines the image set to monitor.

    Although the form refers to images generically, the protocol works with the
    specific input set class selected by the user. For example, the input may be a
    set of particles, micrographs, or another Scipion image set type derived from
    SetOfImages.

    The output set keeps the same type and metadata as the input set.

    ## Minimum Output Size

    The **Minimum output size** parameter defines how many input items are needed
    before the protocol releases an output.

    For example, if the minimum output size is 10000, the protocol waits until at
    least 10000 images are available before releasing data.

    In static-output mode, this defines the size of the closed subset that is
    released. In streaming mode, it defines the threshold at which the output set
    starts being released. In batch mode, it defines the size of each output batch.

    If the input stream closes before this number is reached, the protocol can
    release the available items because no more data are expected.

    ## Send All Items to Output

    The **Send all items to output?** option controls whether the protocol behaves
    as a streaming pass-through or as a one-time trigger.

    If this option is **No**, the protocol releases only one closed subset with the
    selected minimum number of items and then finishes. This is useful when a
    downstream step should receive a fixed-size subset.

    If this option is **Yes**, the protocol continues running in streaming mode.
    After the threshold is reached, it keeps sending new input items to the output.

    This is the most common option when the user wants to delay the beginning of a
    streaming workflow but then allow the stream to continue.

    ## Static Output Mode

    Static output mode is obtained when **Send all items to output?** is set to
    **No**.

    In this mode, the protocol waits until the input set contains the selected
    number of images. It then creates one closed output set and finishes.

    No additional input items are sent to the output after that point.

    This mode is useful for workflows that require a fixed initial subset, for
    example to train a model, estimate parameters, or run a preliminary analysis on
    a controlled number of images.

    ## Full Streaming Mode

    Full streaming mode is obtained when **Send all items to output?** is set to
    **Yes** and **Split items to multiple sets?** is set to **No**.

    In this mode, once the minimum output size is reached, the protocol creates a
    single output set. This output set remains open and continues growing as new
    items arrive in the input.

    When the input stream closes, the output stream is also closed.

    This mode is useful when a downstream workflow should start only after enough
    data have accumulated, but should then continue processing all subsequent data.

    ## Semi-Streaming Batch Mode

    Semi-streaming batch mode is obtained when **Send all items to output?** is
    set to **Yes** and **Split items to multiple sets?** is set to **Yes**.

    In this mode, the protocol creates multiple closed output sets. Each output set
    contains approximately the selected minimum number of items.

    For example, if the minimum output size is 5000, the protocol releases a first
    closed batch of 5000 items, then a second closed batch when another 5000 items
    are available, and so on.

    If the input stream closes and there are remaining items that do not fill a
    complete batch, the final smaller batch is still released.

    This mode is useful for workflows that should process data in independent
    closed batches rather than as one continuously growing stream.

    ## Wait for Signal to Stop the Stream

    The **Wait for signal to stop the stream?** option makes the protocol wait for
    a stop signal from another Trigger Data protocol.

    When enabled, the protocol monitors a signal file named `STOP_STREAM.TXT` in
    its protocol directory. When that file appears, the protocol stops the stream
    and closes its output.

    This option is useful when two workflow branches need to coordinate. For
    example, one branch may collect a sufficient number of images and then signal
    another branch to stop waiting.

    The help recommends using this option together with **Send all items to
    output?** enabled and a minimum size of 1.

    ## Send Signal to Stop a Stream

    The **Send signal to stop a stream?** option makes this protocol send a stop
    signal to another Trigger Data protocol.

    When enabled, the user selects a **Trigger data protocol**. Once this protocol
    has accumulated the selected minimum number of images, it writes the
    `STOP_STREAM.TXT` signal file into the selected protocol directory.

    The connected Trigger Data protocol can then detect the signal and close its
    stream.

    This option is useful for coordinating two streaming branches.

    ## Trigger Data Protocol

    The **Trigger data protocol** parameter is used when **Send signal to stop a
    stream?** is enabled.

    It should point to another Trigger Data protocol that is configured to wait for
    a stop signal.

    The protocol validates that the selected target is indeed a Trigger Data
    protocol. If no valid connected protocol is selected, it reports an error.

    ## Delay

    The **Delay** parameter controls how often the protocol checks for new input
    items or trigger conditions.

    The value is expressed in seconds and must be greater than 3 seconds.

    A shorter delay makes the protocol respond more quickly to new data but checks
    the input more frequently. A longer delay reduces checking frequency but may
    delay output updates.

    The default value is 10 seconds.

    ## Output Name and Type

    The output name is generated from the input image type.

    For example, if the input is a set of particles, the output will be named
    `outputParticles`. If the input is a set of micrographs, the output will be
    named `outputMicrographs`.

    The output set has the same class and metadata structure as the input set.

    In batch mode, outputs are numbered, such as `outputParticles1`,
    `outputParticles2`, and so on.

    ## Stream Closure

    The protocol closes its outputs when the selected finishing condition is met.

    In static mode, the output is closed immediately after the fixed-size subset is
    released.

    In full streaming mode, the output remains open until the input stream closes,
    or until a stop signal is received when signal waiting is enabled.

    In batch mode, each batch output is closed immediately after it is created.

    If the input stream closes before the minimum size is reached, the protocol can
    still release the available images and close the output because no more input
    items are expected.

    ## Summary Information

    The protocol summary reports the current mode:

    - full streaming;
    - semi-streaming batches;
    - static output.

    It also reports whether enough input items have arrived to release an output,
    how many items are required, and whether an output has already been released.

    This helps the user understand whether the protocol is still waiting, actively
    streaming, or finished.

    ## Practical Recommendations

    Use static mode when you need exactly one closed subset of a given size.

    Use full streaming mode when you want to delay a streaming workflow until
    enough data are available, then continue with all new data.

    Use batch mode when downstream protocols should process independent closed
    chunks of data.

    Use signal waiting and signal sending when two streaming branches need to be
    coordinated.

    For signal-based workflows, configure the waiting protocol to send all items
    to output and use a small minimum size, as recommended by the form help.

    Choose a delay that balances responsiveness with unnecessary repeated checks.
    The default 10 seconds is a reasonable starting point.

    ## Final Perspective

    Trigger Data is a workflow-control protocol rather than an image-processing
    protocol.

    For biological users, its value is practical orchestration. It helps control
    when data are released during streaming workflows, when batches are created,
    and when one branch of a workflow should stop another.

    The protocol does not modify the images. It only copies items from an input
    image set to one or more output sets according to threshold, streaming, batch,
    and signaling rules.
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
