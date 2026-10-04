# **************************************************************************
# *
# * Authors: Daniel Marchan (da.marchan@cnb.csic.es)
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
import random
import time
from collections import defaultdict

from pyworkflow import VERSION_3_0
from pwem.objects import SetOfCTF, SetOfMicrographs
from pyworkflow.object import Pointer, CsvList
import pyworkflow.protocol.params as params

from pwem.protocols import ProtCTFMicrographs
from pyworkflow.protocol import ProtStreamingBase
from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase
from pyworkflow.protocol.constants import MODE_RESUME, MODE_RESTART
from pyworkflow import UPDATED, NEW

OUTPUT_CTF =  "outputCTF"
OUTPUT_MICS = "outputMicrographs"

class XmippProtMicDefocusSampler(XmippStreamingBase, ProtStreamingBase, ProtCTFMicrographs):
    """
    Protocol to make a balanced subsample of meaningful CTFs in basis of the
    defocus values. Both CTFs and micrographs will be output. CTFs with
    different defocus values look differently, while low defocus images
    will have bigger fringes, higher defocus values will have more compact rings.
    Micrographs with a greater defocus will have more contrast.

    AI Generated

    ## Overview

    The Micrographs Defocus Sampler protocol selects a subset of micrographs and
    their associated CTF estimations so that the selected images are balanced across
    the defocus range of the dataset.

    In cryo-EM, micrographs acquired at different defocus values can look quite
    different. Low-defocus micrographs usually show wider CTF fringes and lower
    image contrast, while high-defocus micrographs show more compact Thon rings and
    stronger contrast. Because of this, a random subset of micrographs may not
    represent the full optical diversity of the dataset.

    This protocol addresses that problem by sampling micrographs according to their
    defocus values. It divides the defocus range into bins and selects images from
    each bin, producing a more balanced subset than a purely random selection.

    The protocol outputs both the selected CTF estimations and the corresponding
    micrographs.

    ## Inputs and General Workflow

    The main input is a **SetOfCTF**. Each CTF item is linked to the micrograph
    from which it was estimated.

    The protocol reads the defocus value of each CTF estimation, specifically the
    defocusU value, and uses these values to organize the micrographs across the
    observed defocus range.

    The defocus range is divided into a fixed number of bins. The protocol then
    samples micrographs from each bin so that the final output contains a more
    uniform representation of low, medium, and high defocus values.

    The final outputs are:

    - a selected set of CTF estimations;
    - the corresponding selected set of micrographs.

    These outputs can be used for inspection, quality control, representative
    visualization, or downstream tests on a manageable but defocus-balanced subset.

    ## Input CTF

    The **Input CTF** parameter should point to the CTF estimations that will be
    sampled.

    Each CTF must be associated with a micrograph. The protocol uses this
    relationship to produce both output CTFs and output micrographs.

    The quality of the result depends on the quality and completeness of the input
    CTF set. If the CTF estimation contains incorrect defocus values, the balanced
    sampling will reflect those incorrect values.

    This protocol does not estimate CTF parameters. It assumes that CTF estimation
    has already been performed.

    ## Sample Size

    The **Sample size** parameter defines the number of images that the protocol
    will try to select.

    For example, if the sample size is 25, the protocol will output up to 25
    micrographs and their corresponding CTF estimations.

    The sample size should be chosen according to the intended use. A small sample
    is useful for quick visual inspection or illustrative examples. A larger sample
    may be preferable for more robust quality-control checks or benchmark tests.

    The selected subset is not intended to replace the full dataset for final
    processing. Its purpose is to provide a representative defocus-balanced sample.

    ## Minimum Number of Images to Make Sampling

    The **Minimum number of images to make sampling** parameter controls when the
    protocol starts the balanced sampling.

    This is especially relevant in streaming workflows, where CTF estimations may
    arrive gradually as micrographs are processed. The protocol waits until at
    least this number of CTF estimations is available before performing the
    sampling, unless the input stream is closed.

    This avoids selecting a sample too early, before the dataset contains enough
    micrographs to represent the defocus distribution.

    For non-streaming datasets, this parameter acts as a practical threshold for
    when sampling should be performed.

    ## Defocus-Based Balanced Sampling

    The protocol uses defocusU values to organize the input CTFs.

    Conceptually, the procedure is:

    1. Read the defocus value for each CTF.
    2. Determine the minimum and maximum defocus values.
    3. Divide this interval into several bins.
    4. Select images from each bin.
    5. If the selected set is still smaller than the requested sample size, add
       additional images from the remaining pool.

    This strategy tries to prevent the selected subset from being dominated by the
    most common defocus values. Instead, it increases the chance that the final
    sample contains examples from across the full defocus range.

    Because the selection involves random sampling within bins, running the
    protocol again may produce a different subset.

    ## Why Balance by Defocus?

    Defocus affects both image contrast and the appearance of the CTF. Therefore,
    a balanced defocus sample is useful when the user wants to inspect whether the
    dataset behaves consistently across optical conditions.

    For example, such a subset may help answer questions such as:

    - Do low-defocus and high-defocus micrographs both look acceptable?
    - Are Thon rings visible across the whole defocus range?
    - Are some defocus ranges associated with poorer images?
    - Does a downstream protocol behave similarly for different defocus values?

    This can be especially useful for quality-control reports, visual summaries,
    training examples, or testing processing workflows on a representative subset.

    ## Output Micrographs

    The **outputMicrographs** object contains the micrographs associated with the
    selected CTF estimations.

    These micrographs are not modified. They are simply copied into a new Scipion
    set so that the user can inspect or process the selected subset separately from
    the full dataset.

    This output is useful for visual inspection, manual checking, or running quick
    tests on representative micrographs.

    ## Output CTF

    The **outputCTF** object contains the selected CTF estimations.

    The output CTF set remains linked to the selected output micrographs. This
    means that downstream protocols can use the selected micrographs together with
    their corresponding CTF information.

    This output is useful when the user wants to inspect the CTFs themselves, plot
    their defocus distribution, or run downstream tests that require both
    micrographs and CTF metadata.

    ## Summary Statistics

    The protocol reports basic statistics of the defocus values in the input group
    used for sampling. These include the defocus range, minimum, maximum, mean, and
    standard deviation.

    These values help the user understand the defocus distribution from which the
    sample was drawn.

    A wide defocus range indicates that the dataset contains substantial optical
    variation. A narrow range means that the micrographs were acquired with more
    similar defocus values.

    These statistics are descriptive. They are not a quality criterion by
    themselves, but they provide useful context for interpreting the selected
    sample.

    ## Streaming Behavior

    The protocol is designed to work with streaming input.

    In a streaming workflow, new CTF estimations may appear progressively. The
    protocol checks whether new input CTFs are available and waits until either
    enough CTFs have accumulated or the input stream has closed.

    Once a balanced sample has been selected and the outputs have been created, the
    protocol finishes.

    This behavior is useful during online or near-real-time processing, where the
    user may want a representative subset for early inspection without waiting for
    all downstream processing to finish.

    ## Practical Recommendations

    Use this protocol after CTF estimation, not before. The protocol needs defocus
    values and micrograph associations from an existing CTF set.

    Choose a sample size large enough to cover the defocus range meaningfully. Very
    small samples may miss some parts of the distribution even if the sampling is
    balanced.

    Use a larger minimum number of images when working with streaming data, so that
    the protocol does not sample too early from an incomplete and unrepresentative
    defocus distribution.

    Remember that the selection is balanced by defocus, not by all possible quality
    criteria. A selected micrograph may still be poor because of drift, ice
    contamination, astigmatism, poor CTF fit, or other problems.

    Inspect the output micrographs and CTFs together. The purpose of this protocol
    is to make such inspection more representative across the defocus range.

    If reproducibility of the exact selected subset is important, note that the
    sampling includes random choices within defocus bins.

    ## Final Perspective

    The Micrographs Defocus Sampler is a practical quality-control and dataset
    selection tool. It does not modify images or estimate new CTF parameters.
    Instead, it selects a representative subset of micrographs and CTFs across the
    observed defocus range.

    For biological users and facility workflows, this can be useful for quickly
    checking whether different defocus conditions are well represented and whether
    data quality is consistent across the acquisition strategy.

    The protocol is especially helpful when the full dataset is large and the user
    needs a small, interpretable, defocus-balanced subset for inspection, reporting,
    or preliminary testing.
    """
    _label = 'micrographs defocus sampler'
    _devStatus = NEW
    _lastUpdateVersion = VERSION_3_0
    _possibleOutputs = {OUTPUT_MICS: SetOfMicrographs,
                        OUTPUT_CTF: SetOfCTF}

    CTF_VISIBILITY_MAX_ATTEMPTS = 3
    CTF_VISIBILITY_RETRY_DELAY = 1  # seconds


    def __init__(self, **args):
        ProtCTFMicrographs.__init__(self, **args)
        self.sampledIds = CsvList(pType=int)


    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCTF', params.PointerParam, pointerClass='SetOfCTF',
                      label="Input CTF", important=True,
                      help='Select the estimated CTF to evaluate.')
        form.addSection(label='Sampling')
        form.addParam('numImages', params.IntParam,
                      default=25, label='Sample size',
                      help='Number of images after the defocus balanced sampling.')
        form.addParam('minImages', params.IntParam,
                      default=100, label='Minimum number of images to make sampling',
                      help='Minimum number of images to make the defocus balanced sampling.')

        self._defineStreamingParams(form)
        form.addParallelSection(threads=3, mpi=1)

    def _validate(self):
        errors = []
        if self.numImages.get() <= 0:
            errors.append('Sample size must be greater than zero.')
        if self.minImages.get() <= 0:
            errors.append('Minimum number of images must be greater than zero.')
        errors.extend(self._validateParallelProcessing())
        return errors

    def _validateParallelProcessing(self):
        # pyworkflow's executor always reserves one thread out of
        # numberOfThreads for its own bookkeeping, and one more of the
        # remaining slots is permanently held by the streaming generator
        # step for the whole run - at least 3 threads are needed to leave
        # a worker slot free for extractBalancedDefocus.
        if self.numberOfThreads.get() < 3:
            return ['Please assign at least 3 threads: one is reserved by '
                    'the executor for its own bookkeeping and another is '
                    'permanently held by the streaming generator.']
        return []

# --------------------------- INSERT steps functions -------------------------
    def stepsGeneratorStep(self) -> None:
        self.initializeParams()
        self.newDeps = []

        while not getattr(self, 'finished', False):
            self._checkNewInput()
            self._checkNewOutput()

            if getattr(self, 'finished', False):
                break

            self._streamingSleepOnWait()

        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=self.newDeps, needsGPU=False)

    def createOutputStep(self):
        self._closeOutputSet()

    def initializeParams(self):
        self.finished = False
        self.insertedIds = []
        self._lastInputId = 0
        self._pendingInputIds = set()

        # initializeParams now runs as part of the streaming generator
        # step, i.e. AFTER Protocol._runSteps() has already forced
        # runMode to MODE_RESUME (unlike the old _insertAllSteps, which
        # ran before that override) - so self.runMode.get() alone can no
        # longer tell Restart and Continue apart here.
        originalRunMode = getattr(self, '_originalRunMode', self.getRunMode())
        if originalRunMode == MODE_RESTART and self.sampledIds:
            self.sampledIds.clear()
            self._store()

        self.sampled_images = list(self.sampledIds)

    def _insertNewCtfsSteps(self, newIds):
        deps = []
        stepId = self._insertFunctionStep(self.extractBalancedDefocus, newIds,  needsGPU=False, prerequisites=[])
        deps.append(stepId)
        self.insertedIds.extend(newIds)

        return deps

    def _checkNewInput(self):
        if self.sampled_images:
            return

        isResume = (
            getattr(
                self,
                '_originalRunMode',
                self.runMode.get(),
            )
            == MODE_RESUME
        )

        if isResume and not self.insertedIds:
            doneIds, _ = self._getAllDoneIds()

            if doneIds:
                self.sampled_images = doneIds
                self.sampledIds.set(doneIds)
                self._store()
                self.info(
                    'Restoring the previously selected CTFs.'
                )
                return

        if self.insertedIds:
            return

        ctfsSet = self._loadLogicalSet(self.inputCTF)

        try:
            newIds, self._lastInputId = self._discoverIdsAfter(
                ctfsSet,
                self._lastInputId,
            )
            producerClosed = ctfsSet.isStreamClosed()

            knownIds = set(self.insertedIds).union(
                self._pendingInputIds,
            )
            newIds, terminalConsistent = (
                self._reconcileClosedStreamIds(
                    ctfsSet,
                    newIds,
                    knownIds,
                    producerClosed,
                )
            )
        finally:
            ctfsSet.close()

        self._pendingInputIds.update(newIds)

        streamClosed = (
            producerClosed
            and terminalConsistent
        )

        if self._pendingInputIds and (
            len(self._pendingInputIds) >= self.minImages.get()
            or streamClosed
        ):
            pendingIds = sorted(self._pendingInputIds)
            fDeps = self._insertNewCtfsSteps(pendingIds)
            self._pendingInputIds.clear()

            self.newDeps.extend(fDeps)
            self.updateSteps()

        elif streamClosed and not self.insertedIds:
            self.finished = True
            self.info(
                'Input stream is closed and no CTFs are available '
                'for sampling.'
            )

    def extractBalancedDefocus(self, ctfIds):
        pendingIds = sorted(set(ctfIds))
        ctfDefocus = {}

        for attempt in range(self.CTF_VISIBILITY_MAX_ATTEMPTS):
            if not pendingIds:
                break

            if attempt > 0:
                time.sleep(self.CTF_VISIBILITY_RETRY_DELAY)

            inputCtfSet = self._loadLogicalSet(self.inputCTF)
            try:
                visibleCtfs = self._loadLogicalSetItemsByIds(inputCtfSet, pendingIds)
            finally:
                inputCtfSet.close()

            visibleIds = set()
            for ctf in visibleCtfs:
                ctfId = ctf.getObjId()
                visibleIds.add(ctfId)
                try:
                    defocusU = ctf.getDefocusU()
                    if defocusU is None:
                        raise ValueError("CTF has no defocusU value")
                except Exception as e:
                    self.error("CTF with id %d failed while reading its defocus value (%s); excluding it from the defocus sampling pool." % (ctfId, e))
                    continue
                ctfDefocus[ctfId] = defocusU

            pendingIds = [ctfId for ctfId in pendingIds if ctfId not in visibleIds]

        if pendingIds:
            self.error("CTF(s) with id %s never became visible in the input Set after %d attempts; excluding them from the defocus sampling pool." % (pendingIds, self.CTF_VISIBILITY_MAX_ATTEMPTS))

        self.sampled_images = balanced_sampling(image_dict=ctfDefocus, N=self.numImages.get(), bins=10)
        self.info('The number of CTFs selected for defocus balanced sampling is the following: %d' % len(self.sampled_images))

        stats = compute_statistics(list(ctfDefocus.values()))
        message = ("The defocus statistics are the following: range %d   min %d   max %d   mean %d   std %.1f" % (stats["range"], stats["min"], stats["max"], stats["mean"], stats["std"]))
        self.summaryVar.set(message)
        self.sampledIds.set(self.sampled_images)

        # Persist the selected ids before output creation so Resume can reuse
        # exactly the same sample.
        self._store()


    def _checkNewOutput(self):
        """ Check for already selected CTF and update the output set. """
        if self.finished:
            return

        if self.sampled_images:
            ctfSet, micSet = self.createOutputs(self.sampled_images)
            self.updateRelations(ctfSet, micSet)
            self.finished = True
            self._store()  # Update the summary dictionary

    def createOutputs(self, newDone):
        cSet = self._loadOutputSet(SetOfCTF, OUTPUT_CTF)
        mSet = self._loadOutputSet(SetOfMicrographs, OUTPUT_MICS)
        self.fillOutput(cSet, mSet, newDone)

        return cSet, mSet

    def _loadOutputSet(self, SetClass, outputName):
        outputSet = getattr(self, outputName, None)

        if outputSet is not None:
            outputSet.enableAppend()
            return outputSet

        if issubclass(SetClass, SetOfCTF):
            outputSet = self._createSetOfCTF()
        elif issubclass(SetClass, SetOfMicrographs):
            outputSet = self._createSetOfMicrographs()
        else:
            raise TypeError(
                "Unsupported MicDefocusSampler output Set class: %s"
                % SetClass
            )

        micSet = self.inputCTF.get().getMicrographs()

        if isinstance(outputSet, SetOfMicrographs):
            outputSet.copyInfo(micSet)
        elif isinstance(outputSet, SetOfCTF):
            outputSet.setMicrographs(micSet)

        return outputSet

    def fillOutput(self, ctfSet, micSet, newDone):
        inputCtfSet = self._loadLogicalSet(self.inputCTF)
        try:
            visibleCtfs = self._loadLogicalSetItemsByIds(inputCtfSet, newDone)
        finally:
            inputCtfSet.close()

        visibleById = {ctf.getObjId(): ctf for ctf in visibleCtfs}
        missingIds = sorted(set(newDone).difference(visibleById))
        for ctfId in missingIds:
            self.error("CTF with id %d is not visible in the input Set; excluding it from the output." % ctfId)

        ctfIds = self._getOutputIds(ctfSet) if ctfSet.getSize() else set()
        micIds = self._getOutputIds(micSet) if micSet.getSize() else set()
        persistedCtfIds = []
        persistedMicIds = []

        for ctfId in newDone:
            ctf = visibleById.get(ctfId)
            if ctf is None:
                continue

            mic = ctf.getMicrograph().clone()

            if ctf.getObjId() not in ctfIds:
                ctfSet.append(ctf)
                ctfIds.add(ctf.getObjId())
            persistedCtfIds.append(ctf.getObjId())

            if mic.getObjId() not in micIds:
                micSet.append(mic)
                micIds.add(mic.getObjId())
            persistedMicIds.append(mic.getObjId())

        self._markOutputIdsPersisted(OUTPUT_CTF, persistedCtfIds)
        self._markOutputIdsPersisted(OUTPUT_MICS, persistedMicIds)


    def updateRelations(self, cSet, mSet):
        micsAttrName = OUTPUT_MICS
        self._updateOutputSet(micsAttrName, mSet)
        # Set micrograph as pointer to protocol to prevent pointer end up as another attribute (String, Boolean,...)
        # that happens somewhere while scheduling.
        cSet.setMicrographs(Pointer(self, extended=micsAttrName))
        self._updateOutputSet(OUTPUT_CTF, cSet)

        # Rebuild relations atomically from the protocol point of view. This makes
        # repeating updateRelations on Resume safe after a partial output publication.
        if self.mapper is not None:
            self.mapper.deleteRelations(self)

        self._defineTransformRelation(self.inputCTF.get().getMicrographs(), mSet)
        self._defineTransformRelation(self.inputCTF, cSet)
        self._defineCtfRelation(mSet, cSet)

    def _getAllDoneIds(self):
        if hasattr(self, OUTPUT_CTF):
            doneIds = sorted(self._getKnownPersistedOutputIds(OUTPUT_CTF))
            return doneIds, len(doneIds)

        if hasattr(self, OUTPUT_MICS):
            cached = getattr(self, "_micFallbackCtfIds", None)
            if cached is None:
                micIds = self._getKnownPersistedOutputIds(OUTPUT_MICS)
                doneIds = []

                if micIds:
                    inputCtfSet = self._loadLogicalSet(self.inputCTF)
                    try:
                        for ctf in inputCtfSet:
                            mic = ctf.getMicrograph()
                            if mic is not None and mic.getObjId() in micIds:
                                doneIds.append(ctf.getObjId())
                    finally:
                        inputCtfSet.close()

                cached = set(doneIds)
                self._micFallbackCtfIds = cached

            doneIds = sorted(cached)
            return doneIds, len(doneIds)

        return [], 0


    def _summary(self):
        summary = []
        if not hasattr(self, OUTPUT_MICS):
            summary.append("Output set not ready yet.")
        else:
            outputSize = self.outputMicrographs.getSize()
            summary.append("Balanced defocus sample: %d micrographs" % outputSize)
            summary.append(self.summaryVar.get())

        return summary


def balanced_sampling(image_dict, N, bins=10):
    """
    Perform balanced sampling of N images based on defocus values.

    Parameters:
    - image_dict (dict): Dictionary where keys are image IDs and values are defocus values.
    - N (int): Total number of images to sample.
    - bins (int): Number of bins to divide defocus values into (default is 10).

    Returns:
    - sampled_images (list): List of sampled image IDs.
    """

    if not image_dict or N <= 0:
        return []

    target = min(N, len(image_dict))
    bins = max(1, bins)
    defocus_values = list(image_dict.values())

    if min(defocus_values) == max(defocus_values):
        return random.sample(list(image_dict.keys()), target)

    bin_edges = np.linspace(min(defocus_values), max(defocus_values), bins + 1)

    binned_images = defaultdict(list)
    for image_id, defocus in image_dict.items():
        bin_index = np.digitize(defocus, bin_edges) - 1
        bin_index = max(0, min(bin_index, bins - 1))
        binned_images[bin_index].append(image_id)

    non_empty_bins = sorted(
        bin_index for bin_index, images in binned_images.items() if images
    )

    if target <= len(non_empty_bins):
        return _sampleAcrossBins(
            binned_images,
            non_empty_bins,
            target,
        )

    return _sampleRoundRobin(
        binned_images,
        non_empty_bins,
        target,
    )


def _sampleAcrossBins(binned_images, non_empty_bins, target):
    if target == 1:
        selected_positions = [len(non_empty_bins) // 2]
    else:
        selected_positions = [
            round(i * (len(non_empty_bins) - 1) / (target - 1))
            for i in range(target)
        ]

    return [
        random.choice(binned_images[non_empty_bins[position]])
        for position in selected_positions
    ]


def _sampleRoundRobin(binned_images, non_empty_bins, target):
    available_by_bin = {}

    for bin_index in non_empty_bins:
        available = list(binned_images[bin_index])
        random.shuffle(available)
        available_by_bin[bin_index] = available

    sampled_images = []

    while len(sampled_images) < target:
        added = _appendRoundRobinPass(
            sampled_images,
            available_by_bin,
            non_empty_bins,
            target,
        )
        if not added:
            break

    return sampled_images


def _appendRoundRobinPass(
    sampled_images,
    available_by_bin,
    non_empty_bins,
    target,
):
    added = False

    for bin_index in non_empty_bins:
        available = available_by_bin[bin_index]
        if not available:
            continue

        sampled_images.append(available.pop())
        added = True

        if len(sampled_images) == target:
            break

    return added


def compute_statistics(values):
    """
    Compute basic statistics for a list of numerical values.

    Parameters:
    - values (list or array-like): A list of numerical values (e.g., defocus values).

    Returns:
    - dict: A dictionary containing the statistics: min, max, mean, median, std, variance, and range.
    """

    values = np.array(values)
    if values.size == 0:
        raise ValueError('Cannot compute statistics for an empty collection.')

    ddof = 1 if values.size > 1 else 0

    stats = {
        "min": np.min(values),
        "max": np.max(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values, ddof=ddof),
        "variance": np.var(values, ddof=ddof),
        "range": np.max(values) - np.min(values),
    }

    return stats
