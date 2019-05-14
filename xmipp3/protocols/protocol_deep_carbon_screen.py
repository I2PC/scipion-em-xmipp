# **************************************************************************
# *
# * Authors:     Ruben Sanchez Garcia (rsanchez@cnb.csic.es)
# *              David Maluenda (dmaluenda@cnb.csic.es)
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

from glob import glob
import os
from os.path import exists, basename

import pyworkflow.em.metadata as md
import pyworkflow.utils as pwutils
from pyworkflow.protocol.constants import (STEPS_PARALLEL, LEVEL_ADVANCED,
                                           STATUS_FINISHED, STATUS_NEW)
import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtExtractParticles
from pyworkflow.em.data import Particle
from pyworkflow.object import Set, Pointer
from pyworkflow.em.constants import RELATION_CTF

from xmipp3 import Plugin
from xmipp3.base import XmippProtocol, createMetaDataFromPattern
from xmipp3.convert import (micrographToCTFParam, writeMicCoordinates,
                            xmippToLocation, setXmippAttributes, readSetOfCoordinates)
from xmipp3.constants import SAME_AS_PICKING, OTHER
from xmipp3.utils import validateDLtoolkit


class XmippProtDeepCarbonScreen(ProtExtractParticles, XmippProtocol):
    """Protocol to remove coordinates in carbon zones or large impurities"""
    _label = 'deep carbon screen'
    
    def __init__(self, **kwargs):
        ProtExtractParticles.__init__(self, **kwargs)
        self.stepsExecutionMode = STEPS_PARALLEL

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      important=True,
                      label="Input coordinates",
                      help='Select the SetOfCoordinates ')

        # The name for the followig param is because historical reasons
        # now it should be named better 'micsSource' rather than
        # 'downsampleType', but this could make inconsistent previous executions
        # of this protocols, we will keep the name
        form.addParam('downsampleType', params.EnumParam,
                      choices=['same as picking', 'other'],
                      default=0, important=True,
                      display=params.EnumParam.DISPLAY_HLIST,
                      label='Micrographs source',
                      help='By default the particles will be extracted '
                           'from the micrographs used in the picking '
                           'step ( _same as picking_ option ). \n'
                           'If you select _other_ option, you must provide '
                           'a different set of micrographs to extract from. \n'
                           '*Note*: In the _other_ case, ensure that provided '
                           'micrographs and coordinates are related '
                           'by micName or by micId. Difference in pixel size '
                           'will be handled automatically.')

        form.addParam('inputMicrographs', params.PointerParam,
                      pointerClass='SetOfMicrographs',
                      condition='downsampleType != %s' % SAME_AS_PICKING,
                      important=True, label='Input micrographs',
                      help='Select the SetOfMicrographs from which to extract.')

        form.addParam("threshold", params.FloatParam, default=-1,expertLevel=params.LEVEL_ADVANCED,
                      label="Threshold", help="Deep learning goodness score to rule out coordinates. The bigger the treshold "+
                           "the more coordiantes will be ruled out. Ranges from 0 to 1. Use -1 to pospone thresholding until "+
                           "analyze results")

        form.addParam("streamingBatchSize", params.IntParam, default=12,
                      label="Batch size", expertLevel=params.LEVEL_ADVANCED,
                      help="This value allows to group several items to be "
                           "processed inside the same protocol step. You can "
                           "use the following values: \n"
                           "*0*    Put in the same step all the items "
                           "available. If the sleep time is short, it could be "
                           "practically the same of one by one. If not, you "
                           "could have steps with more items. If the steps will "
                           "be executed in parallel, it is better not to use "
                           "this option.\n"
                           "*>1*   The number of items that will be grouped into "
                           "a step.")

        form.addParam("saveMasks", params.BooleanParam, default=False,expertLevel=params.LEVEL_ADVANCED,
                      label="saveMasks", help="Save predicted masks?")

        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation. "
                            "Select the one you want to use. CPU may become "
                            "quite slow.")

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used.")

        # form.addParallelSection(threads=4, mpi=1)
    
    #--------------------------- INSERT steps functions ------------------------
    def _insertInitialSteps(self):
        # Just overwrite this function to load some info
        # before the actual processing
        pwutils.makePath(self._getExtraPath('inputCoords'))
        pwutils.makePath(self._getExtraPath('outputCoords'))
        if self.saveMasks.get():
          pwutils.makePath(self._getExtraPath("predictedMasks"))
        self._setupBasicProperties()

        return []

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
        """ Basic method to insert a picking step for a given micrograph. """
        micStepId = self._insertFunctionStep('extractMicrographStepOwn',
                                             mic.getMicName(), *args,
                                             prerequisites=prerequisites)
        return micStepId


    def _insertExtractMicrographListStepOwn(self, micList, prerequisites, *args):
        """ Basic method to insert a picking step for a given micrograph. """
        return self._insertFunctionStep('extractMicrographListStepOwn',
                                        [mic.getMicName() for mic in micList],
                                        *args, prerequisites=prerequisites)


    def extractMicrographStepOwn(self, micKey, *args):
        """ Step function that will be common for all extraction protocols.
        It will take an id and will grab the micrograph from a micDict map.
        The micrograph will be passed as input to the _extractMicrograph
        function.
        """
        # Retrieve the corresponding micrograph with this key and the
        # associated list of coordinates
        mic = self.micDict[micKey]

        micDoneFn = self._getMicDone(mic)
        micFn = mic.getFileName()

        if self.isContinued() and os.path.exists(micDoneFn):
            self.info("Skipping micrograph: %s, seems to be done" % micFn)
            return

        coordList = self.coordDict[mic.getObjId()]
        self._convertCoordinates(mic, coordList)

        # Clean old finished files
        pwutils.cleanPath(micDoneFn)

        self.info("Extracting micrograph: %s " % micFn)
        self._extractMicrographOwn(mic, *args)

        # Mark this mic as finished
        open(micDoneFn, 'w').close()


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

        self._extractMicrographListOwn(micList, *args)

        for mic in micList:
            # Mark this mic as finished
            open(self._getMicDone(mic), 'w').close()


    def _extractMicrographListOwn(self, micList):
        """ Functional Step. Overrided in general protExtracParticle """

        print("micList: %s" % str(micList))

        micsFnDone = self.getDoneMics()
        micLisfFn = [mic.getFileName() for mic in micList
                     if not pwutils.removeBaseExt(mic.getFileName()) in micsFnDone]
        inputMicsPathMetadataFname= self._getTmpPath("inputMics.xmd")
        mics_md= createMetaDataFromPattern( micLisfFn )
        mics_md.write(inputMicsPathMetadataFname)
        args  =  '-i %s' % inputMicsPathMetadataFname
        args += ' -c %s' % self._getExtraPath('inputCoords')
        args += ' -o %s' % self._getExtraPath('outputCoords')
        args += ' -b %d' % self.getBoxSize()
        args += ' -s 1' #Downsampling is automatically managed by scipion
        args += ' -d %s' % Plugin.getModel('deepCarbonCleaner', 'defaultModel.keras')

        if self.threshold.get() > 0:
            args += ' --deepThr %f ' % (1-self.threshold.get())
        
        if self.saveMasks.get():
            args += ' --predictedMaskDir %s ' % (self._getExtraPath("predictedMasks"))
            
        if self.useGpu.get():
            args += ' -g %s' % self.gpuList.get()
        else:
            args += ' -g -1'
            
        self.runJob('xmipp_deep_carbon_cleaner', args)



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


    def _updateOutputCoordSet(self, micList, streamMode):
        print("in _updateOutputCoordSet > micList: %s" % micList)
        micDoneList = micList  # [mic for mic in micList if self._micIsReady(mic)]

        # Do no proceed if there is not micrograph ready
        if not micDoneList:
            return []

        outputDir = self._getExtraPath('outputCoords')
        outputCoords = self.getOutput()

        # If there are not outputCoordinates yet, it means that is the first
        # time we are updating output coordinates, so we need to first create
        # the output set
        firstTime = outputCoords is None

        if firstTime:
            micSetPtr = self.getInputMicrographs()
            outputCoords = self._createSetOfCoordinates(micSetPtr,
                                                        suffix=self.getAutoSuffix())
            outputCoords.copyInfo(self.inputCoordinates.get())
            outputCoords.setBoxSize(self.getBoxSize())
        else:
            outputCoords.enableAppend()

        self.info("Reading coordinates from mics: %s" % ','.join([mic.strId() for mic in micList]))
        self.readCoordsFromMics(outputDir, micDoneList, outputCoords)
        self.debug(" _updateOutputCoordSet Stream Mode: %s " % streamMode)
        self._updateOutputSet(self.getOutputName(), outputCoords, streamMode)

        if firstTime:
            self._defineSourceRelation(micSetPtr,
                                       outputCoords)

        return micDoneList


    def readCoordsFromMics(self, workingDir, micList, coordSet):
        readSetOfCoordinates(workingDir, micList, coordSet)

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        # errors =[]
        errors = validateDLtoolkit(assertModel=True,
                                   model=('deepCarbonCleaner', 'defaultModel.keras'))

        if self.streamingBatchSize.get() == 1:
            errors.append('Batch size must be 0 (all at once) or larger than 1.')

        return errors

    
    def _citations(self):
        return ['***']
    
    def _summary(self):
        summary = []
        summary.append("Micrographs source: %s"
                        % self.getEnumText("downsampleType"))
        summary.append("Coordinates box size: %d" % (1./self.getBoxScale()) )
        
        return summary
    
    def _methods(self):
        methodsMsgs = []

        return methodsMsgs

    # --------------------------- UTILS functions ------------------------------
    def _convertCoordinates(self, mic, coordList):
        writeMicCoordinates(mic, coordList, self._getMicPos(mic),
                            getPosFunc=self._getPos)
    
    def _micsOther(self):
        """ Return True if other micrographs are used for extract. """
        return self.downsampleType == OTHER

    def _doDownsample(self):
        return False

    def notOne(self, value):
        return abs(value - 1) > 0.0001

    def _getNewSampling(self):
        newSampling = self.samplingMics

        if self._doDownsample():
            # Set new sampling, it should be the input sampling of the used
            # micrographs multiplied by the downFactor
            newSampling *= self.downFactor.get()

        return newSampling

    def _setupBasicProperties(self):
        # Set sampling rate (before and after doDownsample) and inputMics
        # according to micsSource type
        inputCoords = self.getCoords()
        mics = inputCoords.getMicrographs()
        self.samplingInput = inputCoords.getMicrographs().getSamplingRate()
        self.samplingMics = self.getInputMicrographs().getSamplingRate()
        self.samplingFactor = float(self.samplingMics / self.samplingInput)

        scale = self.getBoxScale()
        self.debug("Scale: %f" % scale)
        if self.notOne(scale):
            # If we need to scale the box, then we need to scale the coordinates
            getPos = lambda coord: (int(coord.getX() * scale),
                                    int(coord.getY() * scale))
        else:
            getPos = lambda coord: coord.getPosition()
        # Store the function to be used for scaling coordinates
        self._getPos = getPos

    def getInputMicrographs(self):
        """ Return the micrographs associated to the SetOfCoordinates or
        Other micrographs. """
        if not self._micsOther():
            return self.inputCoordinates.get().getMicrographs()
        else:
            return self.inputMicrographs.get()

    def getCoords(self):
        return self.inputCoordinates.get()

    def getAutoSuffix(self):
        return '_Full' if self.threshold.get() < 0 else '_Auto_%03d'%int(self.threshold.get()*100)

    def getOutputName(self):
        return 'outputCoordinates' + self.getAutoSuffix()

    def getOutput(self):
        if (self.hasAttribute(self.getOutputName()) and
            getattr(self, self.getOutputName()).hasValue()):
            return getattr(self, self.getOutputName())
        else:
            return None

    def getCoordSampling(self):
        return self.getCoords().getMicrographs().getSamplingRate()

    def getMicSampling(self):
        return self.getInputMicrographs().getSamplingRate()

    def getBoxScale(self):
        """ Computing the sampling factor between input and output.
        We should take into account the differences in sampling rate between
        micrographs used for picking and the ones used for extraction.
        The downsampling factor could also affect the resulting scale.
        """
        samplingPicking = self.getCoordSampling()
        samplingExtract = self.getMicSampling()
        f = samplingPicking / samplingExtract
        return f / self.downFactor.get() if self._doDownsample() else f

    def getBoxSize(self):
        # This function is needed by the wizard
        return int(self.getCoords().getBoxSize() * self.getBoxScale())

    def _getOutputImgMd(self):
        return self._getPath('images.xmd')

    def _getMicPos(self, mic):
        """ Return the corresponding .pos file for a given micrograph. """
        micBase = pwutils.removeBaseExt(mic.getFileName())
        return self._getExtraPath('inputCoords', micBase + ".pos")

    def _getMicXmd(self, mic):
        """ Return the corresponding .xmd with extracted particles
        for this micrograph. """
        micBase = pwutils.removeBaseExt(mic.getFileName())
        return self._getExtraPath(micBase + ".xmd")

    def getDoneMics(self):
        out = set([])
        for fName in os.listdir(self._getExtraPath('outputCoords')):
            out.add(pwutils.removeBaseExt(fName))

        return out

    def registerCoords(self, coordsDir):
        """ This method is usually inherited by all Pickers
        and it is used from the Java picking GUI to register
        a new SetOfCoordinates when the user click on +Particles button.
        """

        inputset = self.getInputMicrographs()
        
        mySuffix = '_Manual_%s' % coordsDir.split('manualThresholding_')[1]
        outputName = 'outputCoordinates' + mySuffix

        outputset = self._createSetOfCoordinates(inputset, suffix=mySuffix)
        readSetOfCoordinates(coordsDir, outputset.getMicrographs(), outputset)
        # summary = self.getSummary(outputset)
        # outputset.setObjComment(summary)
        outputs = {outputName: outputset}
        self._defineOutputs(**outputs)

        # Using a pointer to define the relations is more robust to scheduling
        # and id changes between the protocol run.db and the main project
        # database. The pointer defined below points to the outputset object
        self._defineSourceRelation(self.getInputMicrographs(),
                                   Pointer(value=self, extended=outputName))
        self._store()
