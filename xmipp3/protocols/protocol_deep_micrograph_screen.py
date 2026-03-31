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

import mrcfile
from scipy.ndimage import zoom
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
matplotlib.use('Agg')
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from xmipp3.base import XmippProtocol
from xmipp_base import createMetaDataFromPattern
from xmipp3.convert import (writeMicCoordinates, readSetOfCoordinates)
from xmipp3.constants import SAME_AS_PICKING, OTHER
from pyworkflow import BETA, UPDATED, NEW, PROD
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

MAX_SIZE_THUMB=512
NUM_THUMBNAILS=45

class XmippProtDeepMicrographScreen(ProtExtractParticles, XmippProtocol):
    """Removes coordinates located in carbon regions or large impurities in
    micrographs using a pre-trained deep learning model. This screening
    improves particle picking accuracy by filtering out false positives from
    contaminated areas.

    AI Generated

    ## Overview

    The Deep Micrograph Cleaner protocol removes particle coordinates that fall
    in undesirable regions of the micrograph, such as carbon film, large
    contaminants, or other strongly non-particle areas. It uses a pre-trained
    deep learning model to predict which regions of each micrograph are
    suitable for particle picking and which are likely to generate false
    positives.

    In practical cryo-EM workflows, this protocol is typically used after
    particle picking, not before. Its role is to **screen an existing set of
    picked coordinates** and eliminate those that are likely to come from
    visually misleading regions rather than from real particles. This is
    especially useful in datasets containing thick carbon edges, crystalline
    ice, dirt, broken support film, or large impurities.

    For a biological user, this protocol should be understood as a cleaning
    and quality-control step for coordinates. It does not detect new particles;
    instead, it improves an existing picking result by rejecting coordinates
    in problematic regions.

    ## Inputs and General Workflow

    The protocol requires as input a **set of coordinates**. These coordinates
    are the candidate particle positions that will be screened.

    By default, the protocol uses the same micrographs from which those
    coordinates were originally picked. Alternatively, the user may provide a
    different set of micrographs. This is useful, for example, when the
    screening should be performed on inverted micrographs or on a different
    representation of the same images.

    The workflow is conceptually simple. For each micrograph, the deep learning
    model predicts a mask that identifies bad regions. The protocol then checks
    which coordinates fall in or near those regions and removes them according
    to the model score and the threshold chosen by the user.

    The output is therefore a new set of coordinates representing the cleaned
    picking result.

    ## What the Deep Learning Model Detects

    The model is trained to identify micrograph regions that tend to generate
    false positives during picking. These usually include:
    * carbon support or carbon edges,
    * large contaminants,
    * thick or irregular ice,
    * bright or dark artifacts,
    * regions with poor particle-like signal.

    Biologically, the important point is that these regions often contain
    patterns that resemble particles to a generic picker, but they do not
    correspond to real projections of the biological specimen. By removing
    coordinates from such areas, the protocol helps enrich the dataset in more
    meaningful particle candidates.

    However, as with any deep learning method, the output is probabilistic
    rather than absolute. The model is highly useful, but it should not replace
    visual inspection in critical datasets.

    ## Choosing the Micrograph Source

    The protocol allows two possibilities for the micrograph source.

    The most common option is **same as coordinates**, which means that the same
    micrographs used in the picking step are also used for cleaning. This is
    the standard choice and the safest one when the picking and cleaning are
    meant to operate in exactly the same image space.

    The second option is **other**, which allows the user to provide a different
    set of micrographs. This is useful in more specialized situations, such as
    when the original micrographs are not in the most appropriate contrast
    convention for the model or when the user wants to screen coordinates using
    an alternative micrograph representation.

    A particularly important practical note is that the model expects
    **particles to be dark over a bright background**. If the micrographs have
    the opposite contrast, it may be necessary to provide an inverted set
    through the “other” option.

    ## Coordinate Scale and Rescaling

    When different micrographs are used for screening, the coordinate system
    may not match exactly the one used during picking. The protocol handles
    this automatically and offers a choice about how the output coordinates
    should be expressed.

    If the user keeps the coordinates at the original scale, the cleaned
    output remains directly comparable to the original picking result. If the
    user chooses to scale coordinates to the new micrographs, the coordinates
    are rescaled accordingly.

    From a practical point of view, this matters mainly when screening is
    performed on a micrograph set with a different sampling rate or box scale.
    In standard workflows using the same micrographs, the issue does not arise.

    ## Threshold: Controlling How Strict the Cleaning Is

    The most important user parameter is the **threshold**. This controls how
    strict the protocol is when deciding whether a coordinate should be
    discarded.

    Higher threshold values produce a more aggressive cleaning, removing more
    coordinates. Lower values are more permissive and retain more picks. The
    protocol documentation recommends values roughly between **0.75 and 0.9**,
    which is a reasonable practical range for many datasets.

    If the threshold is set to **-1**, the protocol skips automatic thresholding
    during execution. In that case, the model scores are still computed,
    and manual thresholding can be performed afterwards using the
    result-analysis tools.

    Biologically, this parameter determines the balance between two risks:
    * being too permissive and keeping many false positives,
    * being too strict and removing true particles located near difficult regions.

    The best value depends on the dataset. Micrographs with extensive carbon or
    strong contamination may benefit from stricter thresholds, whereas clean
    datasets may only need mild screening.

    ## Batch Processing and Streaming

    The protocol is designed to work both in standard execution and in
    **streaming mode**. This makes it suitable for facility pipelines or
    ongoing acquisitions where micrographs and coordinates arrive progressively.

    To improve efficiency, micrographs can be processed in **batches**. The
    batch size determines how many micrographs are grouped together in a single
    processing step. In streaming scenarios, this helps balance turnaround time
    and GPU efficiency. In static datasets, larger batches can improve
    throughput.

    For most biological users, the default automatic behavior is appropriate
    unless there is a specific need to optimize performance on a given machine.

    ## GPU Acceleration

    This protocol is designed to take advantage of **GPU acceleration**, and in
    practice this is usually the preferred way to run it. A CPU implementation
    exists, but for realistic datasets it may be considerably slower.

    In high-throughput workflows, GPU execution is often essential if one wants
    the cleaning step to keep pace with acquisition or with large-scale picking.

    ## Predicted Masks and Thumbnails

    Optionally, the protocol can save the **predicted masks** generated by the
    deep learning model. These masks show which micrograph regions were
    identified as problematic.

    It can also generate a limited number of **thumbnails** in which the
    micrograph, the predicted mask, and the coordinate positions are overlaid.
    These thumbnails are very useful for visual inspection because they allow
    the user to see, at a glance, whether the model is behaving sensibly.

    From a practical standpoint, these visual outputs are strongly recommended
    when first using the protocol on a new dataset type. They provide immediate
    intuition about whether the selected threshold is reasonable and whether
    the model is correctly identifying contamination or carbon regions.

    ## Outputs and Their Interpretation

    The protocol produces a new **set of cleaned coordinates**. Depending on the
    thresholding strategy, the output name reflects whether the cleaning was
    fully automatic or whether all model scores were retained for later manual
    thresholding.

    These output coordinates represent the subset of the original picking
    result that survived the screening. In other words, the protocol does not
    add coordinates, only removes them.

    Biologically, the cleaned set should contain fewer false positives
    associated with clearly bad regions of the micrograph. This often
    translates into better downstream particle extraction, cleaner 2D classes,
    and a lower burden on later classification steps.

    ## Practical Recommendations

    In most workflows, this protocol is best applied after an initial picking
    step and before particle extraction or early classification. It is
    especially useful when the dataset contains obvious carbon edges,
    contamination, or strong regional heterogeneity in image quality.

    A good starting point is to use the same micrographs as the coordinates and
    a threshold in the recommended range, then inspect the resulting thumbnails
    or masks. If too many clearly bad coordinates remain, the threshold can be
    increased. If obviously good particles are being removed, the threshold
    should be relaxed.

    When using a different micrograph source, users should pay close attention
    to coordinate scaling and to the contrast convention expected by the model.

    As with any automatic cleaning step, it is wise not to treat the output as
    infallible. Visual inspection of at least a representative subset of
    micrographs is strongly advisable, particularly for important biological
    datasets.

    ## Final Perspective

    The Deep Micrograph Cleaner protocol is a practical deep-learning-based
    tool for improving an existing picking result by removing coordinates in
    contaminated or otherwise unsuitable micrograph regions. Its strength lies
    in automating a task that is often visually obvious to an experienced user
    but tedious to perform manually at scale.

    For most cryo-EM users, it should be seen as a screening and
    quality-improvement step that reduces false positives before downstream
    analysis. When used thoughtfully, it can substantially improve the overall
    quality of the particle set while saving considerable manual effort.
    """
    _label = 'deep micrograph cleaner'
    _conda_env= "xmipp_MicCleaner"
    _devStatus = PROD

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

        form.addParam("saveMicThumbnailWithMask", params.BooleanParam, default=True, expertLevel=params.LEVEL_ADVANCED,
                      condition='saveMasks == True',
                      label="Save thumbnails (mics and mask)",
                      help="Save a set of 50 micrographs with the predicted masks stamp and coords stamp")


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
        pwutils.makePath(self._getExtraPath('thumbnails'))
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

        thumbnailCounter = 0
        for mic in micList:
            if self.saveMicThumbnailWithMask.get():
                if thumbnailCounter <= NUM_THUMBNAILS:
                    self._generateThumbnail(mic)
                    thumbnailCounter += 1
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

    def _generateThumbnail(self, mic):
        """
        Generate a thumbnail PNG for a given micrograph.

        Steps performed:
        1. Load the micrograph from MRC file and normalize pixel values to 0-255.
        2. Optionally read particle coordinates from a .pos file.
        3. Optionally read a predicted mask from a .mrc file.
        4. Resize the image, mask, and scale coordinates if larger than maxSize.
        5. Overlay the mask in blue and draw particles as red circles.
        6. Save the final thumbnail as a PNG with minimal padding and compression.

        Parameters
        ----------
        mic : Micrograph object
            The micrograph to generate the thumbnail for.
        maxSize : int, optional
            Maximum size (in pixels) for the thumbnail (default is 512).

        Returns
        -------
        None
            The thumbnail is saved directly to the 'thumbnails' directory.
        """
        micFn = mic.getFileName()
        micBase = pwutils.removeBaseExt(os.path.basename(micFn))

        posFn = self._getExtraPath('outputCoords', micBase + '.pos')
        maskFn = self._getExtraPath('predictedMasks', micBase + '.mrc')
        thumbFn = self._getExtraPath('thumbnails', micBase + '.png')
        if os.path.exists(thumbFn):
            return

        # --- Read micrograph ---
        with mrcfile.open(micFn, permissive=True) as mrc:
            img = mrc.data.astype(np.float32)

        # --- Normalize and convert to uint8 ---
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip((img - p1) / (p99 - p1), 0, 1)
        img = (img * 255).astype(np.uint8)

        # --- Read coordinates ---
        if  os.path.exists(posFn):
            coords = self.read_star_coordinates(posFn)

        # --- Read mask ---
        mask = None
        if os.path.exists(maskFn):
            with mrcfile.open(maskFn, permissive=True) as mrc:
                mask = mrc.data.astype(np.float32)
            mask = np.clip(mask, 0.0, 1)

        # --- Resize image, mask, and scale coordinates/radius ---
        h, w = img.shape
        scale = max(MAX_SIZE_THUMB / h, MAX_SIZE_THUMB / w)

        if scale < 1.0:
            # Bilinear resize for image
            img = zoom(img, (scale, scale), order=1, prefilter=False).astype(np.uint8)
            # Nearest neighbor resize for mask
            if mask is not None:
                mask = zoom(mask, (scale, scale), order=0)
            # Scale coordinates
            if coords:
                coords = [(x * scale, y * scale) for x, y in coords]

        radius = (self.getBoxSize() * scale) / 4
        h, w = img.shape

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        ax.set_position([0, 0, 1, 1])  # axes fill figure
        ax.axis('off')
        # --- Show image ---
        ax.imshow(img, cmap='gray', origin='upper', vmin=0, vmax=255)
        # --- Overlay mask in blue ---
        if mask is not None:
            ax.imshow(mask, cmap='YlGnBu', origin='upper', alpha=mask)

        # --- Fix axes, remove background and borders ---
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_facecolor('none')

        # --- Draw particles using PatchCollection ---
        if coords:
            patches = [Circle((x, y), radius=radius) for x, y in coords]
            collection = PatchCollection(patches, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_collection(collection)

        # --- Save thumbnail ---
        plt.savefig(thumbFn, dpi=80, bbox_inches=None, pad_inches=0, pil_kwargs={"compress_level": 4})
        plt.close(fig)

    def read_star_coordinates(self, posFn):
        coords = []
        with open(posFn) as f:
            lines = f.readlines()

        in_data_particles = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('data_particles'):
                in_data_particles = True
                continue
            if not in_data_particles:
                continue

            if line.startswith('loop_') or line.startswith('_'):
                continue

            parts = line.split()
            if len(parts) >= 4:
                x = float(parts[2])
                y = float(parts[3])
                coords.append((x, y))

        return coords
