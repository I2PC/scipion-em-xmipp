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

from pwem import RELATION_CTF
import pyworkflow.utils as pwutils
from pyworkflow.protocol.constants import (STEPS_PARALLEL, STATUS_NEW)
import pyworkflow.protocol.params as params
from pwem.objects import SetOfMicrographs, SetOfCTF, Image, Micrograph,\
                         Acquisition, String, Set, Float, Integer, Boolean
from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtMicrographs

from xmipp_base import createMetaDataFromPattern
from collections import OrderedDict
import pyworkflow.protocol.constants as cons
from xmipp3.convert import setXmippAttribute, getScipionObj, prefixAttribute
from xmipp3 import emlib
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
from pyworkflow.mapper.sqlite import ID
from xmipp3.convert import setXmippAttribute, getXmippAttribute


SAMPLING_RATE1 = 1
SAMPLING_RATE2 = 1.75
SAMPLING_RATE3 = 2.75
DIMENSION_X = 512
DIMENSION_Y = 512


class XmippProtDeepDefocusMicrograph(ProtMicrographs):
    """Protocol to calcute the defocus Value"""

    _label = 'deep micrograph defocus'

    def __init__(self, **kwargs):
        ProtMicrographs.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL
        self.isFirstTime = Boolean(False)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMicrographs', params.PointerParam, pointerClass='SetOfMicrographs',
                      important=True, label="Input micrographs", help='Select the SetOfMicrogrsphs')

        # form.addParam('defocusU_threshold', params.FloatParam, label='Defocus in U axis',
        #               default=0.6, expertLevel=params.LEVEL_ADVANCED,
        #               help='''By default, micrographs will be divided into an output set and a discarded set based
        #                   on the defocus double threshold''')

        # form.addParam('defocusV_threshold', params.FloatParam, label='Defocus in V axis',
        #              default=0.1, expertLevel=params.LEVEL_ADVANCED,
        #              help='''By default, micrographs will be divided into an output set and a discarded set based
        #                      on the defocus double threshold''')

        form.addParam('ownModel_boolean', params.BooleanParam, default=True,
                      label='Use your own model',
                      help='Setting "yes" '
                            'you can choose your own model trained. If you choose'
                            '"no" a general model pretrained will be assign')

        form.addParam('ownModel', params.FileParam,
                      condition= 'ownModel_boolean',
                      label='Set your model',
                      help='Choose the protocol where your model is trained')

        form.addParam('doTest', params.BooleanParam, label="test model",
                      default=False, expertLevel=params.LEVEL_ADVANCED,
                      help='Use defoci from a previous CTF estimation to test the model')

        form.addParam('ctfRelations', params.RelationParam, allowsNull=True,
                      condition='doTest',
                      relationName=RELATION_CTF,
                      attributeName='inputMicrographs',
                      label='Previous CTF estimation',
                      help='Choose some CTF estimation related to input '
                           'micrographs, in case you want to use the defocus '
                           'values found previously to test the method',
                      expertLevel=params.LEVEL_ADVANCED)

        form.addParam("streamingBatchSize", params.IntParam, default=1,
                      label="Batch size",  expertLevel=params.LEVEL_ADVANCED,
                      help="This value allows to group several items to be "
                           "processed inside the same protocol step. You can "
                           "use the following values: \n"
                           "*1*    The default behavior, the items will be "
                           "processed one by one.\n"
                           "*0*    Put in the same step all the items "
                           "available. If the sleep time is short, it could be "
                           "practically the same of one by one. If not, you "
                           "could have steps with more items. If the steps will "
                           "be executed in parallel, it is better not to use "
                           "this option.\n"
                           "*>1*   The number of items that will be grouped into "
                           "a step.")


        form.addParallelSection(threads=4, mpi=1)

    def _getStreamingBatchSize(self):
        return self.getAttributeValue('streamingBatchSize', 1)


    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        """ Insert the steps to perform CTF estimation, or re-estimation,
                on a set of micrographs. """

        self.micDict = OrderedDict()
        self.initialIds = self._insertInitialSteps()
        micDict, self.streamClosed = self._loadInputList()
        fDeps = self._insertNewMicSteps(micDict.values())
        self._insertFinalSteps(fDeps)
        # For the streaming mode, the steps function have a 'wait' flag that can be turned on/off. For example, here we insert the
        # createOutputStep but it wait=True, which means that can not be executed until it is set to False
        # (when the input micrographs stream is closed)
        waitCondition = self._getFirstJoinStepName() == 'createOutputStep'

        # Comentar esto como segunda opcion a ver que pasa
        if self.isFirstTime:
            # Insert previous estimation
            self._insertPreviousSteps()
            self.isFirstTime.set(False)

        self._insertFunctionStep('createOutputStep',
                                 prerequisites=fDeps, wait=waitCondition)

    def _insertInitialSteps(self):
        """ Override this function to insert some steps before the
        estimate ctfs steps.
        Should return a list of ids of the initial steps. """

        #self.lastMicIdFound = Integer(0)  # Last micId found in the input set
        self.samplingRate = self.inputMicrographs.get().getSamplingRate()
        #self.listOfMicrographs = []
        self.predictions = {}
        self.real = {}
        self.errors = {}
        pwutils.makePath(self._getExtraPath('DONE'))
        if not hasattr(self, "ctfDict"):
            self.getPreviousParameters()
        # Load the model one time only-----
        # modelFname = self.getModel('deepDefocus', 'ModelTrained.h5') #deepDefocus is the directory and ModelTrained.h5 is the model
        self.model = load_model(self.ownModel.get())
        print(self.model.summary())

        return []

    def createOutputStep(self):
        pass

    def _loadInputList(self):
        """ Load the input set of micrographs that are ready to be estimated. """
        return self._loadSet(self.getInputMicrographs(), SetOfMicrographs,
                             lambda mic: mic.getMicName())


    def _loadSet(self, inputSet, SetClass, getKeyFunc):
        """ method overrided in order to check if the previous CTF estimation
            is ready when doInitialCTF=True and streaming is activated
        """
        setFn = inputSet.getFileName()
        self.debug("Loading input db: %s" % setFn)
        updatedSet = SetClass(filename=setFn)
        updatedSet.loadAllProperties()
        streamClosed = updatedSet.isStreamClosed()
        initCtfCheck = lambda idItem: True
        if self.doTest.get():
            ctfSet = SetOfCTF(filename=self.ctfRelations.get().getFileName())
            ctfSet.loadAllProperties()
            streamClosed = streamClosed and ctfSet.isStreamClosed()
            if not streamClosed:
                initCtfCheck = lambda idItem: idItem in ctfSet

        newItemDict = OrderedDict()
        for item in updatedSet:
            micKey = item.getObjId()  # getKeyFunc(item)
            if micKey not in self.micDict and initCtfCheck(micKey):
                newItemDict[micKey] = item.clone()
        updatedSet.close()
        self.debug("Closed db.")
        return newItemDict, streamClosed


    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None


    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all micrographs
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'createOutputStep'


    # ---------------------------STEPS checks ----------------------------------
    def _stepsCheck(self):
        # Input micrograph set can be loaded or None when checked for new inputs
        # If None, we load it
        self._checkNewInput()
        self._checkNewOutput()


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
        micDict, self.streamClosed = self._loadInputList()
        newMics = micDict.values()
        outputStep = self._getFirstJoinStep()

        if newMics:
            fDeps = self._insertNewMicSteps(newMics)
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()


    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        listOfMics = self.micDict.values()
        nMics = len(listOfMics)
        newDone = [m for m in listOfMics
                   if m.getObjId() not in doneList and self._isMicDone(m)]

        # Update the file with the newly done mics
        # or exit from the function if no new done mics
        self.debug('_checkNewOutput: ')
        self.debug('   listOfMics: %s, doneList: %s, newDone: %s'
                   % (nMics, len(doneList), len(newDone)))

        allDone = len(doneList) + len(newDone)
        # We have finished when there is not more input mics (stream closed)
        # and the number of processed mics is equal to the number of inputs
        self.finished = self.streamClosed and allDone == nMics
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN
        self.debug('   streamMode: %s newDone: %s' % (streamMode,
                                                      not (newDone == [])))

        if newDone:
            newDone_tagged = []
            for mic in newDone:
                id = mic.getObjId()
                #defocusU, defocusV = self.predictions[id]
                defocusU = self.predictions[id]
                defocusV = self.predictions[id]
                newMic = mic.clone()
                #setXmippAttribute(newMic, getXmippAttribute('DEFOCUS_U'), defocusU)
                #setXmippAttribute(newMic, getXmippAttribute('DEFOCUS_V'), defocusU)

                #setXmippAttribute(newMic, prefixAttribute('DEFOCUS_U'), defocusU)
                #setXmippAttribute(newMic, prefixAttribute('DEFOCUS_V'), defocusV)

                setAttribute(newMic, 'DEFOCUS_U', defocusU)
                setAttribute(newMic, 'DEFOCUS_V', defocusV)

                newDone_tagged.append(newMic)

            #newDoneUpdated = self._updateOutputMicSet(newDone, streamMode)
            #print('Writing the mic finished')
            #self._writeDoneList(newDoneUpdated)
            newDoneUpdated = self._updateOutputMicSet(newDone_tagged, streamMode)
            print('Writing the mic finished')
            self._writeDoneList(newDoneUpdated)



        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            if allDone == nMics:
                self._streamingSleepOnWait()

            return

        self.debug('   finished: %s ' % self.finished)
        self.debug('        self.streamClosed (%s) AND' % self.streamClosed)
        self.debug('        allDone (%s) == len(self.listOfMics (%s)'
                   % (allDone, nMics))

        #micSet = self._loadOutputSet(SetOfMicrographs, 'micrograph.sqlite')
        #if self.tilt:
            #micSet_discarded = self._loadOutputSet(SetOfMicrographs, 'micrograph' + 'DISCARDED' + '.sqlite')

        #for mic in newDone:
        #   id = mic.getObjId()
        #    defocusU, defocusV = Float(self.defocus[id])
        #    new_Mic = mic.clone()
            #setattr(new_Mic, self.getDefocusULabel(), defocusU)
            #setattr(new_Mic, self.getDefocusVLabel(), defocusV)
            # Double threshold
        #    if defocusU > self.defocusU_threshold.get() and defocusV < self.defocusV_threshold.get():  # AND or OR
        #        micSet.append(new_Mic)
        #    else:
        #        if not self.tilt:
        #            micSet_discarded = self._loadOutputSet(SetOfMicrographs, 'micrograph' + 'DISCARDED' + '.sqlite')
        #            self.tilt = True

        #        micSet_discarded.append(new_Mic)

        #self._updateOutputSet('outputMicrographs', micSet, streamMode)

        #if self.tilt:
        #    self._updateOutputSet('discardedMicrographs', micSet_discarded, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            print('WE FINISHED')
            if self.doTest.get():
                # print(type(np.concatenate(list(self.predictions.values()))))
                print(list(self.predictions.values()))
                print(list(self.real.values()))
                # print(np.concatenate(list(self.predictions.values())))

                mae = mean_absolute_error(np.array(list(self.predictions.values())),
                                          np.array(list(self.real.values())))
                print(mae)

            self._updateStreamState(streamMode)
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)



    def _updateOutputMicSet(self, micList, streamMode):
        doneFailed = []
        micDoneList = [mic for mic in micList]
        # Do no proceed if there is not micrograph ready
        if not micDoneList:
            return []

        outputName = 'outputMicrographs'
        outputMics = getattr(self, outputName, None)

        # If there is not outputCTF yet, it means that is the first
        # time we are updating output CTFs, so we need to first create
        # the output set
        firstTime = outputMics is None

        if firstTime:
            outputMics = self._createSetOfMicrographs()
            inputMicrographs = self.getInputMicrographs()
            outputMics.copyInfo(inputMicrographs)

        else:
            outputMics.enableAppend()

        for micFn, mic in self._iterMicrographs(micList):
            try:
                #ctf = self._createCtfModel(mic)
                # OJOOOO Here we should add the values of the defocus to the mic
                outputMics.append(mic)
            except Exception as ex:
                print(pwutils.yellowStr("Missing CTF?: Couldn't update CTF set with mic: %s" % micFn))
                doneFailed.append(mic)

        self.debug(" _updateOutputMicSet Stream Mode: %s " % streamMode)
        self._updateOutputSet(outputName, outputMics, streamMode)
        if doneFailed:
            self._writeFailedList(doneFailed)

        if firstTime:  # define relation just once
            # Using a pointer to define the relations is more robust to
            # scheduling and id changes between the protocol run.db and
            # the main project database.get
            pass
            #self._defineCtfRelation(self.getInputMicrographsPointer(),
             #                       outputMics)

        return micDoneList

    # DEPENDE DEL SET QUE QUIERAS SACAR
    def _updateStreamState(self, streamMode):
        outputName = 'outputMicrographs'
        outputMics = getattr(self, outputName, None)

        # If there are not outputCTFs yet, it means that is the first
        # time we are updating output CTF, so we need to first create
        # the output set
        firstTime = outputMics is None

        if firstTime:
            outputMics = self._createSetOfMicrographs()
        else:
            outputMics.enableAppend()

        self.debug(" _updateStreamState Stream Mode: %s " % streamMode)
        self._updateOutputSet(outputName, outputMics, streamMode)

    # --------------------------- STEPS functions ------------------------------
    def _insertNewMicSteps(self, inputMics):
        """ Insert steps to process new mics (from streaming)
        Params:
            inputMics: input mics set to be inserted
        """
        return self._insertNewMics(inputMics,
                                   lambda mic: mic.getMicName(),
                                   self._insertMicrographStep,
                                   self._insertMicrographListStep)


    def _insertMicrographStep(self, mic, prerequisites):
        """ Insert the processMicStep for a given movie. """
        micStepId = self._insertFunctionStep('processMicrographStep',
                                             mic.getMicName(),
                                             prerequisites=prerequisites)
        return micStepId


    def _insertMicrographListStep(self, micSubset, prerequisites):
        """ Basic method to insert an estimation step for a given micrograph. """
        micDictList = [mic.getMicName() for mic in micSubset]
        micStepId = self._insertFunctionStep('processMicrographListStep',
                                             micDictList,
                                             prerequisites=prerequisites)
        return micStepId


    def _insertNewMics(self, inputMics, getMicKeyFunc,
                       insertStepFunc, insertStepListFunc, *args):
        """ Insert steps of new micrographs taking into account the batch size.
        It is assumed that a self.micDict exists mapping between micKey and mic.
        It is also assumed that self.streamClosed is defined...with True value
        if the input stream is closed.
        This function can be used from several base protocols that support
        streaming and batch:

        - ProtCTFMicrographs
        - ProtParticlePickingAuto
        - ProtExtractParticles
        - ProtDeepDefocus

        Params:
            inputMics: the input micrographs to be inserted into steps
            getMicKeyFunc: function to get the key of a micrograph
                (usually mic.getMicName()
            insertStepFunc: function used to insert a single step
            insertStepListFunc: function used to insert many steps.
            *args: argument list to be passed to step functions
        Returns:
            The list of step Ids that can be used as dependencies.
        """
        deps = []
        insertedMics = inputMics

        # Despite this function only should insert new micrographs
        # let's double check that they are not inserted already
        micList = [mic for mic in inputMics
                   if getMicKeyFunc(mic) not in self.micDict]

        def _insertSubset(micSubset):
            stepId = insertStepListFunc(micSubset, self.initialIds, *args)
            deps.append(stepId)

        # Now handle the steps depending on the streaming batch size
        batchSize = self._getStreamingBatchSize()

        if batchSize == 1:  # This is one by one, as before the batch size
            for mic in micList:
                stepId = insertStepFunc(mic, self.initialIds, *args)
                deps.append(stepId)
        elif batchSize == 0:  # Greedy, take all available ones
            _insertSubset(micList)
        else:  # batchSize > 0, insert only batches of this size
            n = len(inputMics)
            d = int(n / batchSize)  # number of batches to insert
            nd = d * batchSize
            for i in range(d):
                _insertSubset(micList[i * batchSize:(i + 1) * batchSize])

            if n > nd and self.streamClosed:  # insert last ones
                _insertSubset(micList[nd:])
            else:
                insertedMics = micList[:nd]

        for mic in insertedMics:
            self.micDict[getMicKeyFunc(mic)] = mic

        return deps

    def processMicrographStep(self, micName):
        micrograph = self.micDict[micName]
        # micrograph.setAcquisition(Acquisition())
        # micrograph.setAttributesFromDict(micDict, setBasic=True, ignoreMissing=True)
        micDoneFn = self._getMicrographDone(micrograph)  # EXTRAPath/Done/micrograph_ID.TXT
        micFolderTmp = self._getTmpMicFolder(micrograph)  # tmp/micID
        micFn = micrograph.getFileName()
        # micName = basename(micFn)

        if self.isContinued() and os.path.exists(micDoneFn):
            self.info("Skipping micrograph: %s, seems to be done" % micFn)
            return

        # Clean old finished files
        pwutils.cleanPath(micDoneFn)

        if self._filterMicrograph(micrograph):
            pwutils.makePath(micFolderTmp)
            self.info("Processing micrograph: %s" % micrograph.getFileName())
            self._processMicrograph(micrograph)

            # Quitar esto, es solo prueba para ver que se guarda el attr ---
            if hasattr(micrograph, 'DEFOCUS_U'):
                print('TENGO EL ATRIBUTO DEFOCUS_U')
                print(getattr(micrograph, 'DEFOCUS_U'))

            if hasattr(micrograph, 'DEFOCUS_V'):
                print('TENGO EL ATRIBUTO DEFOCUS_V')
                print(getattr(micrograph, 'DEFOCUS_V'))

            # Quitar esto, es solo prueba para ver que se guarda el attr ---

            # if self.saveIntermediateResults.get():
            #    micOutputFn = self._getResultsMicFolder(micrograph)  # ExtraPath/micID
            #    pwutils.makePath(micOutputFn)
            #    for file in getFiles(micFolderTmp):
            #       moveFile(file, micOutputFn)
            # else:
            #    moveFile(self.getPSDs(micFolderTmp, micID), self._getExtraPath())

        # Mark this movie as finished
        open(micDoneFn, 'w').close()

    def processMicrographListStep(self, micDictList):
        micList = []

        for micName in micDictList:
            micrograph = self.micDict[micName]
            #micrograph.setAcquisition(Acquisition())
            micDoneFn = self._getMicrographDone(micrograph) # EXTRAPath/Done/micrograph_ID.TXT
            micFolderTmp = self._getTmpMicFolder(micrograph)  # tmp/micID

            if self.isContinued() and self._isMicDone(micrograph):
                self.info("Skipping micrograph: %s, seems to be done"
                          % micrograph.getFileName())
            else:
                # Clean old finished files
                pwutils.cleanPath(micDoneFn)

                if self._filterMicrograph(micrograph):
                    pwutils.makePath(micFolderTmp)
                    print('HEllo world')
                    micList.append(micrograph)

                    # if self.saveIntermediateResults.get():
                    #    micOutputFn = self._getResultsMicFolder(micrograph)  # ExtraPath/micID
                    #    pwutils.makePath(micOutputFn)
                    #    for file in getFiles(micFolderTmp):
                    #       moveFile(file, micOutputFn)
                    # else:
                    #    moveFile(self.getPSDs(micFolderTmp, micID), self._getExtraPath())

        self.info("Estimating CTF defocus for micrographs: %s"
                  % [mic.getObjId() for mic in micList])

        self._processMicrographList(micList)

        for mic in micList:
            # Mark this mic as finished
            open(self._getMicrographDone(mic), 'w').close()


    def _processMicrograph(self, micrograph):
        micName = micrograph.getMicName()
        micId =micrograph.getObjId()
        input_NN = self.inputPreparationStep(micrograph)
        print(np.shape(input_NN))
        print(type(input_NN))
        model = self.model
        print('------------New prediction')
        imagPrediction = model.predict(input_NN)
        defocus_pred = np.abs(float(imagPrediction[[0]]))
        #print(defocus_pred)
        self.predictions[micrograph.getObjId()] = defocus_pred
        print('The prediction for mic %s is %f' %(micrograph.getObjId(), self.predictions[micrograph.getObjId()]))

        if self.doTest.get():
            prevValues = (self.ctfDict[micName] if micName in self.ctfDict
                          else self.getSinglePreviousParameters(micId))

            defocusU, defocusV, defocusAngle, phaseShift0 = prevValues
            defocus_real = (defocusU + defocusV) * 0.5
            print(defocusU)
            print(defocusV)
            print(defocus_real)
            self.real[micrograph.getObjId()] = defocus_real
            self.errors[micId] = np.abs(defocus_real-defocus_pred)
            print(self.errors[micId])


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


    def _processMicrographList(self, micList):
        """ This function can be implemented by subclasses if it is a more
        efficient way to estimate many micrographs at once.
         Default implementation will just call the _estimateCTF. """
        input_NN = self.inputPreparationListStep(micList)
        print(np.shape(input_NN))
        print(type(input_NN))
        model = self.model
        print('------------New prediction')
        imagPrediction = model.predict(input_NN)
        #print(imagPrediction)
        # mae = mean_absolute_error(defocusVector, imagPrediction)
        i = 0
        for mic in micList:
            self.predictions[mic.getObjId()] = float(imagPrediction[[i]])
            i = i + 1
            print('The prediction for mic %s is %f' %(mic.getObjId(), self.predictions[mic.getObjId()]))

        fnSummary = self._getPath("summary.txt")
        fnMonitorSummary = self._getPath("summaryForMonitor.txt")

        if not os.path.exists(fnSummary):
            fhSummary = open(fnSummary, "w")
            fnMonitorSummary = open(fnMonitorSummary, "w")
        else:
            fhSummary = open(fnSummary, "a")
            fnMonitorSummary = open(fnMonitorSummary, "a")

        # fhSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
        #               (micrographId, stats['mean'], stats['std'], stats['min'], stats['max']))
        # fhSummary.close()

        # fnMonitorSummary.write("micrograph_%06d: mean=%f std=%f [min=%f,max=%f] \n" %
        #                      (micrographId, stats['mean'], stats['std'], stats['min'], stats['max']))
        fnMonitorSummary.close()


    def inputPreparationStep(self, micrograph):
        samplingRate1 = SAMPLING_RATE1
        samplingRate2 = SAMPLING_RATE2
        samplingRate3 = SAMPLING_RATE3
        ih = ImageHandler()
        micFn = micrograph.getFileName()
        micID = micrograph.getObjId()
        micFolder = self._getTmpMicFolder(micrograph)  # tmp/micID

        imagMatrix = np.zeros((1, DIMENSION_X, DIMENSION_Y, 3), dtype=np.float64)
        # Normalize the micrograph
        filename_micNormalized = os.path.join(micFolder, "micNormalized_" + str(micID) + '.mrc')
        micImage = ih.read(micFn)
        normalizedMatrix = normalizeData(micImage.getData())
        micImage.setData(normalizedMatrix)
        micImage.write(filename_micNormalized)

        # Downsample into 1 A/px, 1.75 A/px, 2.75 A/px
        factor1 = samplingRate1 / self.samplingRate
        factor2 = samplingRate2 / self.samplingRate
        factor3 = samplingRate3 / self.samplingRate

        filename_mic1 = os.path.join(micFolder, "tmp_mic1_" + str(micID) + '.mrc')
        filename_mic2 = os.path.join(micFolder, "tmp_mic2_" + str(micID) + '.mrc')
        filename_mic3 = os.path.join(micFolder, "tmp_mic3_" + str(micID) + '.mrc')

        args1 = "-i %s -o %s --step %f --method fourier" \
                % (filename_micNormalized, filename_mic1,
                   factor1)  # Aqui habria que poner filename_micNormalized en lugar de micFn
        args2 = "-i %s -o %s --step %f --method fourier" \
                % (filename_micNormalized, filename_mic2, factor2)
        args3 = "-i %s -o %s --step %f --method fourier" \
                % (filename_micNormalized, filename_mic3, factor3)

        #TAKE THIS IN MIND
        #if downFactor > 1:
        #    self.runJob("xmipp_transform_downsample",
        #                "-i %s -o %s --step %f --method fourier"
        #                % (micFn, finalName, downFactor))
        #else:
        #    self.runJob("xmipp_image_resize",
        #                "-i %s -o %s --factor %f --interp linear"
        #                % (micFn, finalName, 1.0 / downFactor))


        self.runJob("xmipp_transform_downsample", args1)
        self.runJob("xmipp_transform_downsample", args2)
        self.runJob("xmipp_transform_downsample", args3)

        # Compute PSDs
        image1 = ih.read(filename_mic1)
        image2 = ih.read(filename_mic2)
        image3 = ih.read(filename_mic3)

        psd1 = image1.computePSD(0.4, DIMENSION_X, DIMENSION_Y, 1)
        psd1.convertPSD()
        psd2 = image2.computePSD(0.4, DIMENSION_X, DIMENSION_Y, 1)
        psd2.convertPSD()
        psd3 = image3.computePSD(0.4, DIMENSION_X, DIMENSION_Y, 1)
        psd3.convertPSD()

        # Store data to check
        filename_psd1 = os.path.join(micFolder, "tmp_psd1_" + str(micID) + '.mrc')
        filename_psd2 = os.path.join(micFolder, "tmp_psd2_" + str(micID) + '.mrc')
        filename_psd3 = os.path.join(micFolder, "tmp_psd3_" + str(micID) + '.mrc')

        psd1.write(filename_psd1)
        psd2.write(filename_psd2)
        psd3.write(filename_psd3)

        # Filter the PSD OJO:no parece haber mucha diff probar con ambos
        filename_filt_psd1 = os.path.join(micFolder, "tmp_psd1_filt" + str(micID) + '.mrc')
        filename_filt_psd2 = os.path.join(micFolder, "tmp_psd2_filt" + str(micID) + '.mrc')
        filename_filt_psd3 = os.path.join(micFolder, "tmp_psd3_filt_" + str(micID) + '.mrc')

        args1 = '-i %s -o %s --fourier low_pass 0.1' % (filename_psd1, filename_filt_psd1)
        args2 = '-i %s -o %s --fourier low_pass 0.1' % (filename_psd2, filename_filt_psd2)
        args3 = '-i %s -o %s --fourier low_pass 0.1' % (filename_psd3, filename_filt_psd3)

        self.runJob("xmipp_transform_filter", args1)
        self.runJob("xmipp_transform_filter", args2)
        self.runJob("xmipp_transform_filter", args3)

        img1 = ih.read(filename_filt_psd1).getData()
        img2 = ih.read(filename_filt_psd2).getData()
        img3 = ih.read(filename_filt_psd3).getData()

        imagMatrix[0, :, :, 0] = img1
        imagMatrix[0, :, :, 1] = img2
        imagMatrix[0, :, :, 2] = img3

        if self.doTest.get():
            # md = xmipp.MetaData(fnRoot.replace("_xmipp_ctf_enhanced_psd.xmp", "_xmipp_ctf.xmd"))
            # objId = md.firstObject()
            # dU = md.getValue(xmipp.MDL_CTF_DEFOCUSU, objId)
            # dV = md.getValue(xmipp.MDL_CTF_DEFOCUSV, objId)

            # prevValues = (self.ctfDict[micName] if micName in self.ctfDict
            #             else self.getSinglePreviousParameters(mic.getObjId()))
            # localParams['defocusU'], localParams['defocusV'], localParams['defocusAngle'], localParams['phaseShift0'] = \
            # prevValues

            # dU = localParams['defocusU']
            # dV = localParams['defocusV']

            # defocus = 0.5*(dU+dV)
            # self.defocusVector[micId] = defocus
            pass

        return imagMatrix

    def inputPreparationListStep(self, micList):
        samplingRate1 = SAMPLING_RATE1
        samplingRate2 = SAMPLING_RATE2
        samplingRate3 = SAMPLING_RATE3
        ih = ImageHandler()
        imagMatrix = np.zeros((len(micList), DIMENSION_X, DIMENSION_Y, 3), dtype=np.float64)
        i = 0
        for micrograph in micList:
            micFn = micrograph.getFileName()
            micID = micrograph.getObjId()
            micFolder = self._getTmpMicFolder(micrograph)  # tmp/micID

            # Normalize the micrograph
            filename_micNormalized = os.path.join(micFolder, "micNormalized_" + str(micID) + '.mrc')
            micImage = ih.read(micFn)
            normalizedMatrix = normalizeData(micImage.getData())
            micImage.setData(normalizedMatrix)
            micImage.write(filename_micNormalized)

            # Downsample into 1 A/px, 1.75 A/px, 2.75 A/px
            factor1 = samplingRate1 / self.samplingRate
            factor2 = samplingRate2 / self.samplingRate
            factor3 = samplingRate3 / self.samplingRate

            filename_mic1 = os.path.join(micFolder, "tmp_mic1_" + str(micID) + '.mrc')
            filename_mic2 = os.path.join(micFolder, "tmp_mic2_" + str(micID) + '.mrc')
            filename_mic3 = os.path.join(micFolder, "tmp_mic3_" + str(micID) + '.mrc')

            args1 = "-i %s -o %s --step %f --method fourier" \
                    % (filename_micNormalized, filename_mic1,
                       factor1)  # Aqui habria que poner filename_micNormalized en lugar de micFn
            args2 = "-i %s -o %s --step %f --method fourier" \
                    % (filename_micNormalized, filename_mic2, factor2)
            args3 = "-i %s -o %s --step %f --method fourier" \
                    % (filename_micNormalized, filename_mic3, factor3)

            self.runJob("xmipp_transform_downsample", args1)
            self.runJob("xmipp_transform_downsample", args2)
            self.runJob("xmipp_transform_downsample", args3)

            # Compute PSDs
            image1 = ih.read(filename_mic1)
            image2 = ih.read(filename_mic2)
            image3 = ih.read(filename_mic3)

            psd1 = image1.computePSD(0.4, DIMENSION_X, DIMENSION_Y, 1)
            psd1.convertPSD()
            psd2 = image2.computePSD(0.4, DIMENSION_X, DIMENSION_Y, 1)
            psd2.convertPSD()
            psd3 = image3.computePSD(0.4, DIMENSION_X, DIMENSION_Y, 1)
            psd3.convertPSD()

            # Store data to check
            filename_psd1 = os.path.join(micFolder, "tmp_psd1_" + str(micID) + '.mrc')
            filename_psd2 = os.path.join(micFolder, "tmp_psd2_" + str(micID) + '.mrc')
            filename_psd3 = os.path.join(micFolder, "tmp_psd3_" + str(micID) + '.mrc')

            psd1.write(filename_psd1)
            psd2.write(filename_psd2)
            psd3.write(filename_psd3)

            # Filter the PSD OJO:no parece haber mucha diff probar con ambos
            filename_filt_psd1 = os.path.join(micFolder, "tmp_filt_psd1_" + str(micID) + '.mrc')
            filename_filt_psd2 = os.path.join(micFolder, "tmp_filt_psd2_filt_" + str(micID) + '.mrc')
            filename_filt_psd3 = os.path.join(micFolder, "tmp_filt_psd3_filt_" + str(micID) + '.mrc')

            args1 = '-i %s -o %s --fourier low_pass 0.1' % (filename_psd1, filename_filt_psd1)
            args2 = '-i %s -o %s --fourier low_pass 0.1' % (filename_psd2, filename_filt_psd2)
            args3 = '-i %s -o %s --fourier low_pass 0.1' % (filename_psd3, filename_filt_psd3)

            self.runJob("xmipp_transform_filter", args1)
            self.runJob("xmipp_transform_filter", args2)
            self.runJob("xmipp_transform_filter", args3)

            img1 = ih.read(filename_filt_psd1).getData()
            img2 = ih.read(filename_filt_psd2).getData()
            img3 = ih.read(filename_filt_psd3).getData()

            imagMatrix[i, :, :, 0] = img1
            imagMatrix[i, :, :, 1] = img2
            imagMatrix[i, :, :, 2] = img3

            i = i + 1

            if self.doTest.get():
                # md = xmipp.MetaData(fnRoot.replace("_xmipp_ctf_enhanced_psd.xmp", "_xmipp_ctf.xmd"))
                # objId = md.firstObject()
                # dU = md.getValue(xmipp.MDL_CTF_DEFOCUSU, objId)
                # dV = md.getValue(xmipp.MDL_CTF_DEFOCUSV, objId)

                # prevValues = (self.ctfDict[micName] if micName in self.ctfDict
                #             else self.getSinglePreviousParameters(mic.getObjId()))
                # localParams['defocusU'], localParams['defocusV'], localParams['defocusAngle'], localParams['phaseShift0'] = \
                # prevValues

                # dU = localParams['defocusU']
                # dV = localParams['defocusV']

                # defocus = 0.5*(dU+dV)
                # self.defocusVector[micId] = defocus
                pass

        return imagMatrix

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
    #def _validate(self):
    #    errors = self.validateDLtoolkit(assertModel=True,
    #                                    model=('deepMicrographCleaner', 'defaultModel.keras'))

    #    batchSize = self.streamingBatchSize.get()
    #    if batchSize == 1:
    #        errors.append('Batch size must be 0 (all at once) or larger than 1.')
    #   elif not self.isInStreaming() and batchSize > len(self.inputCoordinates.get().getMicrographs()):
    #        errors.append('Batch size (%d) must be <= that the number of micrographs '
    #                      '(%d) in static mode. Set it to 0 to use only one batch'
    #                      % (batchSize, self._getNumPickedMics()))

    #    return errors


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
    def getSinglePreviousParameters(self, micId):
        if self.ctfRelations.hasValue():
            ctf = self.ctfRelations.get()[micId]
            return self.getPreviousValues(ctf)

    def getPreviousValues(self, ctf):
        phaseShift0 = 0.0
        if ctf.hasPhaseShift():
            phaseShift0 = ctf.getPhaseShift()
        else:
            phaseShift0 = 1.57079  # pi/2

        ctfValues = (ctf.getDefocusU(), ctf.getDefocusV(), ctf.getDefocusAngle(), phaseShift0)
        return ctfValues

    def getPreviousParameters(self):
        if self.ctfRelations.hasValue():
            self.ctfDict = {}
            for ctf in self.ctfRelations.get():
                ctfName = ctf.getMicrograph().getMicName()
                self.ctfDict[ctfName] = self.getPreviousValues(ctf)


    def _iterMicrographs(self, inputMics=None):
        """ Iterate over micrographs and yield
        micrograph name. """
        if inputMics is None:
            inputMics = self.getInputMicrographs()

        for mic in inputMics:
            micFn = mic.getFileName()
            yield micFn, mic





    def updateLastMicIdFound(self, mics):
        """ Updates the last input check attribute with the maximum id in the micSet"""
        micIds = [m.getObjId() for m in mics]
        self.lastMicIdFound.set(max(micIds))

    def getSinglePreviousParameters(self, micId):
        if self.ctfRelations.hasValue():
            ctf = self.ctfRelations.get()[micId]
            return self.getPreviousValues(ctf)


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

    def _getTmpMicFolder(self, micrograph):
        """ Create a Mic folder where to work with it. """
        return self._getTmpPath('mic_%06d' % micrograph.getObjId())

    def _getOutputMicFolder(self, micrograph):
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
        doneFile = self._getAllDone()

        if not exists(doneFile):
            pwutils.makeFilePath(doneFile)

        with open(doneFile, 'a') as f:
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

    # --------------------------- OVERRIDE functions --------------------------

    def _filterMicrograph(self, micrograph):
        """ Check if process or not this movie.
        """
        return True


# --------------------- WORKERS --------------------------------------
def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def setAttribute(obj, label, value):
    if value is None:
        return
    setattr(obj, label, getScipionObj(value))