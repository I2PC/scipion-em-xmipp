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
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippProtDeepMicrographScreen(ProtExtractParticles, XmippProtocol):
    """Removes coordinates located in carbon regions or large impurities in micrographs using a pre-trained deep learning model. This screening improves particle picking accuracy by filtering out false positives from contaminated areas."""
    _label = 'deep micrograph cleaner'
    _conda_env= "xmipp_MicCleaner"
    _devStatus = UPDATED

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

        form.addParam('micsSource', params.EnumParam,
                      choices=['same as coordinates', 'other'],
                      default=0, important=True,
                      display=params.EnumParam.DISPLAY_HLIST,
                      label='Micrographs source',
                      help='By default, the micrographs from which the computation '
                           'will be performed will be the ones used in the picking '
                           'step ( _same as coordinates_ option ). \n'
                           'If you select other option, you must provide '
                           'a different set of micrographs to evaluate its regions. \n'
                           '*Note*: In the _other_ case, ensure that provided '
                           'micrographs and coordinates are related '
                           'by micName or by micId. Difference in pixel size '
                           'will be handled automatically.\n'
                           '*Note2*: *Particles must be dark* over a bright '
                           'background. If not, use the _other_ option to provide '
                           'an inverted setOfMicrograph.')

        form.addParam('inputMicrographs', params.PointerParam,
                      pointerClass='SetOfMicrographs',
                      condition='micsSource != %s' % SAME_AS_PICKING,
                      important=True, label='Input micrographs',
                      help='Select the SetOfMicrographs from which to extract.')

        form.addParam('useOtherScale',params.EnumParam,
                      choices=['same as coordinates', 'scale to micrographs'],
                      default=0, condition='micsSource != %s' % SAME_AS_PICKING,
                      display=params.EnumParam.DISPLAY_HLIST,
                      label='Coordinates scale',
                      help='If you select _same as coordinates_ option output coordinates '
                           'will be mapped to the original micrographs and thus, they will preserve '
                           'the scale.\nIf you select _scale to micrographs_ option, output coordinates '
                           'will be mapped to the new micrographs and rescaled accordingly.')


        form.addParam("threshold", params.FloatParam, default=-1,
                      label="Threshold", help="Deep learning goodness score to select/discard coordinates. The bigger the threshold "+
                           "the more coordiantes will be ruled out. Ranges from 0 to 1. Use -1 to skip thresholding. "+
                           "Manual thresholding can be performed after execution through analyze results button. "+
                           "\n0.75 <= Recommended threshold <= 0.9")

        form.addParam("streamingBatchSize", params.IntParam, default=-1,
                      label="Batch size", expertLevel=params.LEVEL_ADVANCED,
                      help="This value allows to group several items to be "
                           "processed inside the same protocol step. You can "
                           "use the following values: \n"
                           "*0*    Put in the same step all the items  available.\n "
                           "*>1*   The number of items that will be grouped into "
                           "a step. -1, automatic decission")

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

    def getGpusList(self, separator):
        strGpus = ""
        for elem in self._stepsExecutor.getGpuList():
            strGpus = strGpus + str(elem) + separator
        return strGpus[:-1]

    def setGPU(self, oneGPU=False):
        if oneGPU:
            gpus = self.getGpusList(",")[0]
        else:
            gpus = self.getGpusList(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self.info(f'Visible GPUS: {gpus}')
        return gpus


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

        if len(micLisfFn)>0:
          inputMicsPathMetadataFname= self._getTmpPath("inputMics"+str(hash(micLisfFn[0]))+".xmd")
          mics_md= createMetaDataFromPattern( micLisfFn )
          mics_md.write(inputMicsPathMetadataFname)
          args  =  '-i %s' % inputMicsPathMetadataFname
          args += ' -c %s' % self._getExtraPath('inputCoords')
          args += ' -o %s' % self._getExtraPath('outputCoords')
          args += ' -b %d' % self.getBoxSize()
          args += ' -s 1' #Downsampling is automatically managed by scipion
          args += ' -d %s' % self.getModel('deepMicrographCleanerTF2', 'defaultModel.h5')

          if self.threshold.get() > 0:
              args += ' --deepThr %f ' % (1-self.threshold.get())

          if self.saveMasks.get():
              args += ' --predictedMaskDir %s ' % (self._getExtraPath("predictedMasks"))

          if self.useGpu.get():
              gpuId = self.setGPU(oneGPU=False)
              args += f' -g {gpuId} '

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
      if self.micsSource==SAME_AS_PICKING or self.useOtherScale.get()==1:
        scale= 1
      else:
        scale=(1./self.getBoxScale())
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
            if self.useOtherScale.get() == 1:
              boxSize = self.getBoxSize()
            else:
              boxSize = self.inputCoordinates.get().getBoxSize()

            micSetPtr = self.getInputMicrographsPointer()
            outputCoords = self._createSetOfCoordinates(micSetPtr, suffix=self.getAutoSuffix())
            outputCoords.copyInfo(self.inputCoordinates.get())
            outputCoords.setBoxSize(boxSize)
        else:
            outputCoords.enableAppend()
        self.info("Reading coordinates from mics: %s" % ','.join([mic.strId() for mic in micList]))
        readSetOfCoordinates(outputDir, micList, outputCoords, scale= self._getScale())
        self.debug(" _updateOutputCoordSet Stream Mode: %s " % streamMode)
        self._updateOutputSet(self.getOutputName(), outputCoords, streamMode)

        if firstTime:
            self._defineSourceRelation(micSetPtr,
                                       outputCoords)

        return micList

    #--------------------------- INFO functions --------------------------------
    def _getStreamingBatchSize(self):
      self.firstBatch = True
      if self.streamingBatchSize.get() == -1:
        if not hasattr(self, "actualBatchSize"):
          if self.isInStreaming():
            self.actualBatchSize = 16
            batchSize = self.actualBatchSize
          else:
            if self.firstBatch:
              self.firstBatch = False
              batchSize = 4
            else:
              nPickMics = self._getNumPickedMics()
              self.actualBatchSize = min(50, nPickMics)
              batchSize = self.actualBatchSize
        else:
            batchSize = self.actualBatchSize
      else:
        batchSize = self.streamingBatchSize.get()

      return batchSize

    def _getNumPickedMics(self):
      nPickMics = 0
      lastId=None
      for coord in self.inputCoordinates.get():
        curId=coord.getMicId()
        if lastId!=curId:
          lastId=curId
          nPickMics+=1
      return nPickMics

    def _validate(self):
        errors = self.validateDLtoolkit(assertModel=True,
                                        model=('deepMicrographCleaner', 'defaultModel.keras'))

        batchSize = self.streamingBatchSize.get()
        if batchSize == 1:
            errors.append('Batch size must be 0 (all at once) or larger than 1.')
        elif not self.isInStreaming() and batchSize > len(self.inputCoordinates.get().getMicrographs()):
            errors.append('Batch size (%d) must be <= that the number of micrographs '
                          '(%d) in static mode. Set it to 0 to use only one batch'
                          %(batchSize, self._getNumPickedMics()))

        return errors

    
    def _citations(self):
        return ['***']
    
    def _summary(self):
        summary = []
        summary.append("Micrographs source: %s"
                        % self.getEnumText("micsSource"))
        summary.append("Coordinates scale: %d" % (1./self.getBoxScale()) )
        
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
        return self.micsSource == OTHER

    def notOne(self, value):
        return abs(value - 1) > 0.0001

    def _setupBasicProperties(self):
        # Set sampling rate (before and after doDownsample) and inputMics
        # according to micsSource type
        inputCoords = self.getCoords()
        mics = inputCoords.getMicrographs()
        self.samplingInput = inputCoords.getMicrographs().getSamplingRate()
        self.samplingMics = self.getInputMicrographs().getSamplingRate()
        self.samplingFactor = float(self.samplingMics / float(self.samplingInput))

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

    def getInputMicrographsPointer(self):
        """ Return the micrographs pointer associated to the SetOfCoordinates or
        Other micrographs. """
        if not self._micsOther():
            inMicsPointer = self.getCoords().getMicrographs(asPointer=True)
            return inMicsPointer
        else:
            return self.inputMicrographs

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
        f = float(samplingPicking) / samplingExtract
        return f

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
        inputset = self.getInputMicrographsPointer()
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
        self._defineSourceRelation(inputset,
                                   Pointer(value=self, extended=outputName))
        self._store()
