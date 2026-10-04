# *****************************************************************************
# *
# * Authors:     Tomas Majtner         tmajtner@cnb.csic.es (2017)
# *              David Maluenda        dmaluenda@cnb.csic.es
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
# *****************************************************************************

import json
import os, sys

import pwem.emlib.metadata as md
import pyworkflow.protocol.constants as cons
import pyworkflow.protocol.params as param
from pyworkflow import VERSION_2_0
from pyworkflow.object import Set
from pyworkflow.utils import cleanPath
from pyworkflow.utils.properties import Message

from pwem import ALIGN_NONE
from pwem.protocols import ProtClassify2D
from pwem.objects import SetOfParticles, SetOfAverages, SetOfClasses2D, Class2D, SetOfClasses, SetOfImages
from pyworkflow.protocol import ProtStreamingBase
from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

from xmipp3.convert import (writeSetOfParticles, readSetOfParticles,
                            setXmippAttributes)


class XmippProtEliminateEmptyBase(XmippStreamingBase, ProtStreamingBase, ProtClassify2D):
    """ Base to eliminate images using statistical methods
    (variance of variances of sub-parts of input image) eliminates those samples,
    where there is no object/particle (only noise is presented there).
    Threshold parameter can be used for fine-tuning the algorithm for type of
    data.

    AI Generated

    ## Overview

    The Eliminate Empty protocols identify images that appear to contain little
    or no meaningful particle signal and separate them from the rest of the
    dataset. The decision is based on statistical image features that measure
    whether the image behaves more like structured particle content or like
    mostly background noise.

    There are two variants of the protocol. One works on
    **individual particles**, and the other works on **2D classes or averages**.
    In both cases, the goal is similar: to remove images that look empty,
    noise-dominated, or otherwise uninformative.

    In practical cryo-EM workflows, these protocols are quality-control tools.
    They are especially useful when a dataset contains a substantial fraction
    of bad particle picks, empty boxes, very weak classes, or classes dominated
    by background. Used carefully, they can reduce the burden on later
    classification and interpretation steps. However, because they rely on
    statistical criteria rather than biological understanding, they should
    always be used with some caution.

    ## General Principle

    The underlying idea is that images containing a real particle tend to show
    more meaningful internal structure than images containing only background
    noise. The protocol evaluates this using statistical descriptors based on
    the variability of subregions within the image.

    In simple terms, truly empty images tend to look more uniform in a
    statistical sense, whereas particle-containing images tend to exhibit
    localized variations associated with the molecular projection. The protocol
    converts this idea into an **emptiness score** and then uses a threshold to
    decide whether the image is kept or rejected.

    For biological users, the important point is that this is a
    **screening method**, not a classification of biological states. It
    distinguishes likely signal from likely emptiness, but it does not know
    whether a particle is biologically interesting, well centered, damaged,
    or heterogeneous.

    ## Two Variants: Particles and Classes

    ### Eliminate Empty Particles

    This variant operates on a **set of particles**. It is intended to remove
    extracted images that appear to contain mostly noise rather than a
    recognizable particle.

    This can be useful after particle picking and extraction, especially when
    the picking step has produced many false positives from ice, contaminants,
    or empty background regions.

    ### Eliminate Empty Classes

    This variant operates on a **set of 2D classes or averages**. It is
    intended to remove classes that do not appear to contain a real particle
    signal, or that are too poorly populated to be considered useful.

    This is especially useful after 2D classification, when some classes
    correspond to noise, carbon edges, contamination, or very weak particle
    content.

    ## Threshold: Main Control of the Selection

    The most important parameter in both variants is the
    **threshold used in elimination**.

    Higher threshold values produce a more aggressive selection, meaning that
    more particles or classes will be rejected as empty. Lower values are more
    permissive and retain more data.

    A special case is **threshold = -1**. In this mode, the protocol does not
    actually eliminate anything, but it still computes and stores the emptiness
    score for each image. This is often a very good strategy when using the
    protocol for the first time on a dataset, because it allows the user to
    inspect the scores before deciding how strict the filtering should be.

    From a practical point of view, the threshold controls the balance between:
    * removing obviously bad data, and
    * accidentally rejecting weak but real particle images or classes.

    For difficult datasets with low contrast or small particles, overly
    aggressive thresholds can be risky.

    ## Optional Denoising

    Both variants offer an advanced option to use **denoising** during the
    computation of the emptiness feature. This applies a Gaussian blurring step
    before the emptiness score is evaluated.

    The purpose of this denoising is to suppress fine-grained noise and make
    the broad distinction between empty and non-empty images more robust. In
    some datasets this improves the discrimination, especially when the raw
    images are extremely noisy.

    The **denoising factor** controls how strong this smoothing is. Higher
    values apply stronger blurring.

    Biologically, the denoising does not improve the particle itself; it only
    changes the way the emptiness feature is computed. For this reason, it is
    mainly a technical aid for the scoring process.

    ## Add Features Option

    An advanced option allows the protocol to
    **attach the computed ranking features** to the input particles or classes.
    This is useful when the user wants not only the accepted and rejected
    outputs, but also the underlying numerical information used for the
    decision.

    This option is particularly valuable in exploratory workflows, since it
    allows later inspection, sorting, or manual analysis of the images
    according to their emptiness-related score.

    ## Eliminate Empty Particles: Typical Use

    The particle variant is best understood as a cleaning step for extracted
    particles. It evaluates each particle individually and separates the
    input into:
    * **accepted particles**, and
    * **eliminated particles**.

    The output particles also carry the computed emptiness score, which can be
    useful for later inspection.

    In practical workflows, this protocol is often useful after particle
    extraction and before expensive downstream classification. It can reduce
    the number of obvious false positives and make subsequent 2D classification
    more efficient.

    However, users should be careful with weak or low-contrast particles. A
    particle may be real but still appear statistically close to noise,
    especially if it is small, poorly centered, or strongly affected by CTF
    and low dose.

    ## Eliminate Empty Classes: Typical Use

    The class variant works on **2D classes or averages** and can use two
    complementary criteria:
    * the image-based emptiness criterion, and
    * optionally, the **class population**.

    The image-based criterion evaluates whether the class average itself
    appears to contain meaningful structure. This is useful for identifying
    classes that are mostly noise or background.

    The population criterion rejects classes whose size is too small relative
    to the mean class population. The parameter **minimum population (%)**
    expresses how large a class must be, relative to the average class size,
    in order to be accepted.

    This is biologically useful because very small classes are often unstable
    or poorly representative, although they may sometimes correspond to rare
    but meaningful states. For that reason, the population criterion should
    be used carefully when minority conformations are of interest.

    ## Population Filtering in Class Mode

    When **use class population** is enabled, a class can be rejected not
    because it is visually empty, but because too few particles contributed to
    it.

    This criterion only works when the input is a true **set of classes** with
    membership information. It is not available when the input consists only of
    standalone averages, because in that case the protocol does not know how
    many particles contributed to each average.

    From a practical perspective, population filtering is a good way to remove
    weak, unstable classes in large datasets. But in heterogeneous samples it
    can also remove rare but biologically important states. Users interested in
    low-population conformations should therefore apply this criterion
    conservatively, or disable it.

    ## Streaming Behavior

    These protocols are designed to work in **streaming mode** as well as in
    standard batch mode. This means that they can process new particles or
    classes as they arrive and progressively update the accepted and rejected
    outputs.

    This is especially useful in facility pipelines or automated workflows
    where image data are generated continuously and early filtering is
    desirable.

    For most users, the streaming logic remains mostly transparent, but it
    explains why the protocol can produce outputs incrementally rather than
    only at the very end.

    ## Outputs and Their Interpretation

    ### Particle Variant

    The protocol can produce:
    * **outputParticles**: accepted particles
    * **eliminatedParticles**: rejected particles

    ### Class Variant
    The protocol can produce:
    * **outputAverages** and, when applicable, **outputClasses**
    * **eliminatedAverages** and, when applicable, **eliminatedClasses**

    The average outputs contain the representative images, while the class
    outputs preserve the class structure and their member particles when the
    input was a real set of classes.

    In all cases, the outputs should be interpreted as a split of the original
    dataset into a more promising subset and a more questionable subset. The
    rejected subset is not useless: it is often worth inspecting, since it helps
    the user understand what kind of low-quality data were present in the input.

    ## Practical Recommendations

    For particles, a good strategy is often to begin conservatively, especially
    in low-SNR datasets. Running with **threshold = -1** first can be very
    helpful because it lets the user inspect the emptiness scores before
    deciding how much to reject.

    For classes, this protocol is particularly useful after 2D classification,
    when many classes are clearly noise-like. The population criterion can be
    very effective in large datasets, but it should be used carefully if rare
    states are biologically important.

    The denoising option is often helpful when the images are extremely noisy,
    but it should still be regarded as a technical aid rather than a universal
    improvement.

    As with many automatic filtering protocols, the best practice is not to
    rely exclusively on the numerical decision. Visual inspection of at least
    a representative subset of accepted and eliminated items is strongly
    recommended.

    ## Final Perspective

    The Eliminate Empty protocols are statistical cleaning tools designed to
    separate informative images from images that are likely dominated by
    background or noise. They are useful for both particle-level and class-level
    quality control and can simplify downstream processing by reducing obviously
    poor data.

    For most cryo-EM users, their main value lies in providing an automatic
    first pass over the dataset. Used thoughtfully, they can save substantial
    effort and improve the quality of later analysis, but they should always be
    combined with visual inspection and biological judgment.
    """
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        self.stepsExecutionMode = cons.STEPS_PARALLEL

    def addAdvancedParams(self, form):
        form.addParam('addFeatures', param.BooleanParam, default=False,
                      label='Add features', expertLevel=param.LEVEL_ADVANCED,
                      help='Add features used for the ranking to each '
                           'one of the input particles.')
        form.addParam('useDenoising', param.BooleanParam, default=True,
                      label='Turning on denoising',
                      expertLevel=param.LEVEL_ADVANCED,
                      help='Option for turning on denoising method '
                           'while computing emptiness feature.')
        form.addParam('denoising', param.IntParam, default=5,
                      expertLevel=param.LEVEL_ADVANCED,
                      condition='useDenoising',
                      label='Denoising factor:',
                      help='Factor to be used during Gaussian blurring. '
                           'Higher value applies stronger denoising, '
                           'could be more precise but also slower.')

        self._defineStreamingParams(form)
        form.addParallelSection(threads=3, mpi=1)

    # --------------------------- INSERT steps functions ----------------------
    def _prepareStreamingGenerator(self):
        self.lenPartsSet = 0
        self.outputSize = 0
        self.stepCount = 0
        self.streamClosed = False
        self.finished = False
        self._lastInputId = 0
        self._scheduledBatchIds = set()
        self._activeBatchIndexes = set()
        self.fnInputMd = self._getExtraPath("input%d.xmd")
        originalRunMode = getattr(self, '_originalRunMode', self.getRunMode())
        existingInfos = [self._getEliminationStepInfo(step) for step in self._iterEliminationSteps()]
        self.stepCount = max([info[0] for info in existingInfos if info] or [0])
        if originalRunMode == cons.MODE_RESUME:
            self._restoreStreamingState(existingInfos)

    def stepsGeneratorStep(self) -> None:
        self._prepareStreamingGenerator()
        self.newDeps = []
        while not getattr(self, 'finished', False):
            self._checkNewInput()
            self._checkNewOutput()
            if getattr(self, 'finished', False):
                break
            self._streamingSleepOnWait()
        self._insertFunctionStep(self.createOutputStep, prerequisites=self.newDeps, needsGPU=False)

    @staticmethod
    def _getEliminationStepInfo(step):
        funcName = getattr(step, 'funcName', None)
        if hasattr(funcName, 'get'):
            funcName = funcName.get()
        if funcName != 'eliminationStep':
            return None
        argsStr = getattr(step, 'argsStr', None)
        if hasattr(argsStr, 'get'):
            argsStr = argsStr.get('[]')
        try:
            args = json.loads(argsStr or '[]')
        except (TypeError, ValueError):
            return None
        if len(args) < 2 or not isinstance(args[1], list):
            return None
        return int(args[0]), [int(itemId) for itemId in args[1]]

    def _iterEliminationSteps(self):
        for step in self._iterKnownStreamingSteps():
            if self._getEliminationStepInfo(step) is not None:
                yield step

    def _getPersistedProcessedIds(self):
        processedIds = set()
        for outputName in self._getResumeOutputNames():
            processedIds.update(self._getPersistedOutputIds(outputName))
        return processedIds

    def _getReadyEliminationBatches(self):
        processedIds = self._getPersistedProcessedIds()
        ready = []
        for step in self._iterEliminationSteps():
            info = self._getEliminationStepInfo(step)
            if info is None or info[0] not in self._activeBatchIndexes or not step.isFinished():
                continue
            batchIndex, batchIds = info
            if not set(batchIds).issubset(processedIds):
                ready.append((batchIndex, batchIds))
        return ready

    def _restoreStreamingState(self, existingInfos=None):
        existingInfos = existingInfos if existingInfos is not None else [self._getEliminationStepInfo(step) for step in self._iterEliminationSteps()]
        processedIds = self._getPersistedProcessedIds()
        for info in existingInfos:
            if info:
                self._activeBatchIndexes.add(info[0])
                self._scheduledBatchIds.update(info[1])
        knownIds = processedIds.union(self._scheduledBatchIds)
        self.outputSize = len(processedIds)
        self._lastInputId = max(knownIds) if knownIds else 0
        inputSet = self.getInput()
        inputSet.loadAllProperties()
        self.lenPartsSet = inputSet.getSize()
        self.streamClosed = inputSet.isStreamClosed()
        inputSet.close()
        self.info("Restored streaming state: %d persisted of %d input images" % (self.outputSize, self.lenPartsSet))

    def _insertNewPartsSteps(self, batchIds):
        batchIds = sorted(set(batchIds))
        if not batchIds:
            return []
        self.stepCount += 1
        batchIndex = self.stepCount
        stepId = self._insertFunctionStep('eliminationStep', batchIndex, batchIds, prerequisites=[])
        self._activeBatchIndexes.add(batchIndex)
        self._scheduledBatchIds.update(batchIds)
        return [stepId]

    def _getResumeOutputNames(self):
        return ()

    def specialBehavoir(self, inSet):
        pass

    def eliminationStep(self, batchIndex, batchIds):
        fnInputMd = self.fnInputMd % batchIndex
        inputSet = self.getInput()
        inputSet.loadAllProperties()
        batchItems = self._loadLogicalSetItemsByIds(inputSet, batchIds)
        inputSet.close()
        partsSet = self.prepareImages(batchItems, batchIndex)
        writeSetOfParticles(partsSet, fnInputMd, alignType=ALIGN_NONE, orderBy='id')
        self.specialBehavoir(partsSet)
        fnOutputMd = self._getExtraPath("output%d.xmd" % batchIndex)
        fnElimMd = self._getExtraPath("eliminated%d.xmd" % batchIndex)
        args = "-i %s -o %s -e %s -t %f" % (fnInputMd, fnOutputMd, fnElimMd, self.threshold.get())
        if self.addFeatures:
            args += " --addFeatures"
        if self.useDenoising:
            args += " --useDenoising -d %f" % self.denoising.get()
        self.runJob("xmipp_image_eliminate_empty_particles", args)

    def createOutputStep(self):
        pass

    def _checkNewInput(self):
        inputSet = self.getInput()
        inputSet.loadAllProperties()
        knownIds = self._getPersistedProcessedIds().union(self._scheduledBatchIds)
        newIds, self._lastInputId = self._discoverIdsAfter(inputSet, self._lastInputId)
        producerClosed = inputSet.isStreamClosed()
        newIds, terminalConsistent = self._reconcileClosedStreamIds(inputSet, newIds, knownIds, producerClosed)
        newIds = [itemId for itemId in newIds if itemId not in knownIds]
        self.lenPartsSet = inputSet.getSize()
        self.streamClosed = producerClosed and terminalConsistent
        inputSet.close()
        if newIds:
            self.newDeps.extend(self._insertNewPartsSteps(newIds))
            self.updateSteps()

    def prepareImages(self, batchItems, batchIndex):
        pass

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        self.createOutputs()
        self.outputSize = len(self._getPersistedProcessedIds())
        self.finished = self.streamClosed and self.outputSize == self.lenPartsSet and not self._getReadyEliminationBatches()
        if self.finished:
            self.createOutputs()

    def createOutputs(self):
        """ To be implemented by child. (create, fill and close the outputSet)
        """
        pass

    def _loadOrCreateOutputSet(self, outputName, SetClass, suffix=''):
        if SetClass is not SetOfClasses2D:
            return XmippStreamingBase._loadOrCreateOutputSet(self, outputName, SetClass, suffix)
        outputSet = getattr(self, outputName, None)
        if outputSet is not None:
            outputSet.loadAllProperties()
            outputSet.enableAppend()
            return outputSet, False
        outputSet = self._createSetOfClasses2D(self.getInput(), suffix)
        outputSet.setStreamState(outputSet.STREAM_OPEN)
        return outputSet, True

    def _loadOutputSet(self, SetClass, outputName):
        suffixByOutputName = {'outputParticles': '', 'eliminatedParticles': 'Eliminated', 'outputAverages': '', 'eliminatedAverages': 'Eliminated'}
        if outputName not in suffixByOutputName:
            raise ValueError("Unknown EliminateEmpty output: %s" % outputName)
        outputSet, created = self._loadOrCreateOutputSet(outputName, SetClass, suffixByOutputName[outputName])
        if created:
            outputSet.copyInfo(self.inputImages)
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
            self._defineOutputs(**{outputName: outputSet})
            self._defineTransformRelation(self.getInput(), outputSet)
            self._store(outputSet)

        # Close set databaset to avoid locking it
        outputSet.close()

    def _validate(self):
        return self._validateParallelProcessing()

    def _validateParallelProcessing(self):
        # pyworkflow's executor always reserves one thread out of
        # numberOfThreads for its own bookkeeping, and one more of the
        # remaining slots is permanently held by the streaming generator
        # step for the whole run - at least 3 threads are needed to leave
        # a worker slot free for eliminationStep.
        if self.numberOfThreads.get() < 3:
            return ['Please assign at least 3 threads: one is reserved by '
                    'the executor for its own bookkeeping and another is '
                    'permanently held by the streaming generator.']
        return []

    # --------------------------- UTILS functions -----------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_SCORE_BY_EMPTINESS)
        if row.getValue(md.MDL_ENABLED) <= 0:
            item._appendItem = False
        else:
            item._appendItem = True

    def getInput(self):
        """ Get the input as it is in the form """
        pass


class XmippProtEliminateEmptyParticles(XmippProtEliminateEmptyBase):
    """ Takes a set of particles and using statistical methods
    (variance of variances of sub-parts of input image) eliminates those samples,
    where there is no object/particle (only noise is presented there).
    Threshold parameter can be used for fine-tuning the algorithm for type of data.
    """

    _label = 'eliminate empty particles'

    def __init__(self, **args):
        XmippProtEliminateEmptyBase.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        # - - - F O R   P A R T I C L E S - - -
        form.addParam('inputParticles', param.PointerParam, important=True,
                      label="Input particles", pointerClass='SetOfParticles',
                      help='Select the input particles to be classified.')
        form.addParam('threshold', param.FloatParam, default=1.1,
                      label='Threshold used in elimination:',
                      help='Higher threshold => more particles will be '
                           'eliminated. Set to -1 for no elimination, even so '
                           'the "xmipp_scoreEmptiness" value will be attached to '
                           'every paricle for a posterior inspection.')

        self.addAdvancedParams(form)

    # --------------------------- INSERT steps functions ----------------------
    def _getResumeOutputNames(self):
        return ('outputParticles', 'eliminatedParticles')

    def specialBehavoir(self, partsSet):
        partsSet.close()

    def createOutputs(self):
        streamMode = Set.STREAM_CLOSED if getattr(self, 'finished', False) else Set.STREAM_OPEN
        for batchIndex, batchIds in self._getReadyEliminationBatches():
            for mdFn, outputName, auxSuffix in ((self._getExtraPath("output%d.xmd" % batchIndex), 'outputParticles', "AUXOUT%d" % batchIndex), (self._getExtraPath("eliminated%d.xmd" % batchIndex), 'eliminatedParticles', "AUXELIM%d" % batchIndex)):
                if not os.path.exists(mdFn):
                    continue
                outSet = self._loadOutputSet(SetOfParticles, outputName)
                partsSet = self._createSetOfParticles(auxSuffix)
                readSetOfParticles(mdFn, partsSet)
                existingIds = self._getOutputIds(outSet)
                def updateItem(item, row):
                    self._updateParticle(item, row)
                    if item.getObjId() in existingIds:
                        item._appendItem = False
                outSet.copyItems(partsSet, updateItemCallback=updateItem, itemDataIterator=md.iterRows(mdFn, sortByLabel=md.MDL_ITEM_ID))
                self._updateOutputSet(outputName, outSet, streamMode)
                cleanPath(mdFn)
            self.outputSize = len(self._getPersistedProcessedIds())
        if self.finished:
            for outputName in self._getResumeOutputNames():
                if hasattr(self, outputName):
                    self._updateOutputSet(outputName, self._loadOutputSet(SetOfParticles, outputName), Set.STREAM_CLOSED)

    def getInput(self):
        return self.inputParticles.get()

    def prepareImages(self, batchItems, batchIndex):
        self.inputImages = self._createSetOfParticles("AUX%d" % batchIndex)
        self.inputImages.copyInfo(self.getInput())
        for item in batchItems:
            self.inputImages.append(item)
        return self.inputImages



DISCARDED = 0
ACCEPTED = 1
ATTACHED = 2

class XmippProtEliminateEmptyClasses(XmippProtEliminateEmptyBase):
    """ Takes a set of classes (or averages) and using statistical methods
    (variances of sub-parts of input image) eliminates those samples,
    where there is no object/particle (only noise is presented there).
    Threshold parameter can be used for fine-tuning the algorithm for
    type of data. Also discards classes with less population than a given
    percentage.
    """

    _label = 'eliminate empty classes'

    def __init__(self, **args):
        XmippProtEliminateEmptyBase.__init__(self, **args)
        self.enableCls = {}

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        # - - - F O R   C L A S S E S - - -
        form.addParam('inputClasses', param.PointerParam, important=True,
                      pointerClass='SetOfClasses, SetOfAverages',
                      label="Input classes",
                      help='Select the input averages to be classified.')
        form.addParam('threshold', param.FloatParam, default=8.0,
                      label='Threshold used in elimination',
                      help='Higher threshold => more particles will be '
                           'eliminated. Set to -1 for no elimination, even so '
                           'the "xmipp_scoreEmptiness" value will be attached to '
                           'every paricle for a posterior inspection.')
        form.addParam('usePopulation', param.BooleanParam, default=True,
                      label='Use class population',
                      help="Consider class population to reject a class.")
        form.addParam('minPopulation', param.FloatParam, default=20,
                      label='Min. population (%)',
                      condition="usePopulation",
                      expertLevel=param.LEVEL_ADVANCED,
                      help="Minimum population to accept a class.\n"
                           "Classes with less population than the mean population "
                           "times this value will be rejected.")

        self.addAdvancedParams(form)

    # --------------------------- INSERT steps functions ----------------------
    def _validate(self):
        errors = super()._validate()
        if (not isinstance(self.getInput(), SetOfClasses)
                and self.usePopulation.get()):
            errors.append("Using population to reject classes is not possible "
                          "with Averages as input.\nPlease, introduce a "
                          "setOfClasses to analyse or disable the _use class "
                          "population_ option.")
        return errors

    def _getResumeOutputNames(self):
        return ('outputAverages', 'eliminatedAverages')

    def specialBehavoir(self, partSet):
        idsToCheck = [item.getObjId() for item in partSet]
        partSet.close()
        self.rejectByPopulation(idsToCheck)

    def createOutputs(self):
        streamMode = Set.STREAM_CLOSED if getattr(self, 'finished', False) else Set.STREAM_OPEN
        for batchIndex, batchIds in self._getReadyEliminationBatches():
            enableByOutput = {}
            for mdFn, outputName, auxSuffix in ((self._getExtraPath("output%d.xmd" % batchIndex), 'outputAverages', "AUXOUT%d" % batchIndex), (self._getExtraPath("eliminated%d.xmd" % batchIndex), 'eliminatedAverages', "AUXELIM%d" % batchIndex)):
                enableOut = {}
                if os.path.exists(mdFn):
                    outSet = self._loadOutputSet(SetOfAverages, outputName)
                    partsSet = self._createSetOfParticles(auxSuffix)
                    readSetOfParticles(mdFn, partsSet)
                    accepted = outputName == 'outputAverages'
                    existingIds = self._getOutputIds(outSet)
                    for part in partsSet:
                        partId = part.getObjId()
                        currentStatus = self.enableCls.get(partId, ACCEPTED)
                        enableOut[partId] = ACCEPTED if accepted and currentStatus == ACCEPTED else DISCARDED
                    def updateItem(item, row):
                        self._updateParticle(item, row)
                        if item.getObjId() in existingIds:
                            item._appendItem = False
                    outSet.copyItems(partsSet, updateItemCallback=updateItem, itemDataIterator=md.iterRows(mdFn, sortByLabel=md.MDL_ITEM_ID))
                    self._updateOutputSet(outputName, outSet, streamMode)
                    cleanPath(mdFn)
                enableByOutput[outputName] = enableOut
            self.createOutputClasses('output', streamMode, enableByOutput.get('outputAverages', {}))
            self.createOutputClasses('eliminated', streamMode, enableByOutput.get('eliminatedAverages', {}))
            self.outputSize = len(self._getPersistedProcessedIds())

    # ------------- UTILS Fuctions ------------------------------------
    def prepareImages(self, batchItems, batchIndex):
        inSet = self.getInput()
        isImages = isinstance(inSet, SetOfImages)
        self.inputImages = self._createSetOfAverages("AUX%d" % batchIndex)
        self.inputImages.enableAppend()
        self.classesDict = getattr(self, 'classesDict', {}) if not isImages else None
        if batchItems:
            firstRep = batchItems[0] if isImages else batchItems[0].getRepresentative()
            self.inputImages.copyAttributes(firstRep, '_samplingRate')
        for item in batchItems:
            image = item.clone() if isImages else item.getRepresentative().clone()
            if not isImages:
                image.setObjId(item.getObjId())
                self.classesDict[item.getObjId()] = item.getSize()
            self.inputImages.append(image)
        self.inputImages.copyAttributes(inSet, '_streamState')
        self.streamClosed = self.inputImages.isStreamClosed()
        self.lenPartsSet = inSet.getSize()
        return self.inputImages

    def rejectByPopulation(self, ids):
        if self.usePopulation.get() and self.classesDict is not None:
            sizeDict = {clsId: size for clsId, size
                        in self.classesDict.items()
                        if clsId in ids}

            meanPop = sum(sizeDict.values())/len(sizeDict)

            for clsId, size in sizeDict.items():
                # minPopulation is normalized to 100% not to 1
                decision = int(size*100 > meanPop * self.minPopulation.get())
                self.enableCls[clsId] = ACCEPTED if decision else DISCARDED
        else:
            self.enableCls = dict.fromkeys(ids, ACCEPTED)

    def createOutputClasses(self, suffix, streamingState, enableDict):
        if not self.classesDict or not enableDict:
            return

        outputName = '%sClasses' % suffix
        outputSet, created = self._loadOrCreateOutputSet(outputName, SetOfClasses2D, '' if suffix == 'output' else 'Eliminated')
        if created:
            outputSet.copyInfo(self.getInput())

        decision = ACCEPTED if suffix == 'output' else DISCARDED
        desiredIds = [objId for objId, enable in enableDict.items() if enable == decision]
        outputSet.appendFromClasses(self.getInput(), lambda cls: cls.getObjId() in desiredIds)
        outputSet.setStreamState(streamingState)

        if self.hasAttribute(outputName):
            outputSet.write()
            outputAttr = getattr(self, outputName)
            if outputAttr is not outputSet:
                outputAttr.copy(outputSet, copyId=False)
            self._store(outputAttr)
        else:
            self._defineOutputs(**{outputName: outputSet})
            self._defineSourceRelation(self.inputClasses, outputSet)
            self._store(outputSet)

        outputSet.close()

    def getInput(self):
        return self.inputClasses.get()
