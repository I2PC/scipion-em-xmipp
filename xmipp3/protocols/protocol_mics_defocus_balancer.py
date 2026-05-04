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

import os
from datetime import datetime
import numpy as np
import random
from collections import defaultdict

from pyworkflow import VERSION_3_0
from pwem.objects import SetOfCTF, SetOfMicrographs
from pyworkflow.object import Pointer
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils

from pwem.protocols import ProtCTFMicrographs
from pyworkflow.protocol.constants import (STATUS_NEW)
from pyworkflow import UPDATED, NEW


OUTPUT_CTF =  "outputCTF"
OUTPUT_MICS = "outputMicrographs"

class XmippProtMicDefocusSampler(ProtCTFMicrographs):
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


    def __init__(self, **args):
        ProtCTFMicrographs.__init__(self, **args)


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

# --------------------------- INSERT steps functions -------------------------
    def _insertAllSteps(self):
        self.initializeParams()
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=[], wait=True, needsGPU=False)

    def createOutputStep(self):
        self._closeOutputSet()

    def initializeParams(self):
        self.finished = False
        # Important to have both:
        self.insertedIds = []   # Contains images that have been inserted in a Step (checkNewInput).
        self.sampled_images = [] # Ids to be sample
        # Contains images that have been processed in a Step (checkNewOutput).
        self.ctfFn = self.inputCTF.get().getFileName()

    def _getFirstJoinStepName(self):
        ''' This function will be used for streaming, to check which is
        the first function that need to wait for all ctfs
        to have completed, this can be overriden in subclasses
        (e.g., in Xmipp 'sortPSDStep')
        '''
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

    def _insertNewCtfsSteps(self, newIds):
        deps = []
        stepId = self._insertFunctionStep(self.extractBalancedDefocus, newIds,  needsGPU=False, prerequisites=[])
        deps.append(stepId)
        self.insertedIds.extend(newIds)

        return deps

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        # Check if there are new ctf to process from the input set
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = datetime.fromtimestamp(os.path.getmtime(self.ctfFn))
        self.debug('Last check: %s, modification: %s'
                    % (pwutils.prettyTime(self.lastCheck),
                        pwutils.prettyTime(mTime)))
        # If the input movies.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and self.insertedIds:  # If this is empty it is dut to a static "continue" action or it is the first round
            return None

        ctfsSet = self._loadInputCtfSet(self.ctfFn)
        ctfSetIds = ctfsSet.getIdSet()
        newIds = [idCTF for idCTF in ctfSetIds if idCTF not in self.insertedIds]

        self.lastCheck = datetime.now()
        isStreamClosed = ctfsSet.isStreamClosed()

        ctfsSet.close()

        outputStep = self._getFirstJoinStep()

        if self.isContinued() and not self.insertedIds:  # For "Continue" action and the first round
            doneIds, _ = self._getAllDoneIds()
            if doneIds:
                self.finished = True
                self.info('The sampling images are already created.')
                return

        if (newIds and len(newIds) >= self.minImages.get()) or isStreamClosed:
            fDeps = self._insertNewCtfsSteps(newIds)

            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def _loadInputCtfSet(self, ctfFn):
        self.debug("Loading input db: %s" % ctfFn)
        ctfSet = SetOfCTF(filename=ctfFn)
        ctfSet.loadAllProperties()
        return ctfSet

    def extractBalancedDefocus(self, ctfIds):
        inputCtfSet = self._loadInputCtfSet(self.ctfFn)
        ctfDefocus = {}

        for ctfId in ctfIds:
            ctf = inputCtfSet.getItem("id", ctfId).clone()
            defocusU = ctf.getDefocusU()
            ctfDefocus[ctfId] = defocusU

        self.sampled_images = balanced_sampling(image_dict=ctfDefocus, N=self.numImages.get(), bins=10)
        self.info('The number of CTFs selected for defocus balanced sampling is the following: %d'
                  %len(self.sampled_images))

        stats = compute_statistics(list(ctfDefocus.values()))
        message = ("The defocus statistics are the following: range %d   min %d   max %d   mean %d   std %.1f"
                   %(stats["range"], stats["min"], stats["max"],stats["mean"], stats["std"]))
        self.summaryVar.set(message)

    def _checkNewOutput(self):
        """ Check for already selected CTF and update the output set. """
        # Check for results: we have finished when there is results in sample_images list
        if not self.finished:
            if self.sampled_images:
                ctfSet, micSet = self.createOutputs(self.sampled_images)
                self.updateRelations(ctfSet, micSet)
                self.finished = True
                self._store()  # Update the summary dictionary
        else:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

    def createOutputs(self, newDone):
        cSet = self._loadOutputSet(SetOfCTF, 'ctfs.sqlite')
        mSet = self._loadOutputSet(SetOfMicrographs,
                                   'micrographs.sqlite')
        self.fillOutput(cSet, mSet, newDone)

        return cSet, mSet

    def _loadOutputSet(self, SetClass, baseName):
        """
        Create the output set.
        """
        setFile = self._getPath(baseName)
        outputSet = SetClass(filename=setFile)

        micSet = self.inputCTF.get().getMicrographs()

        if isinstance(outputSet, SetOfMicrographs):
            outputSet.copyInfo(micSet)
        elif isinstance(outputSet, SetOfCTF):
            outputSet.setMicrographs(micSet)

        return outputSet

    def fillOutput(self, ctfSet, micSet, newDone):
        inputCtfSet = self._loadInputCtfSet(self.ctfFn)

        for ctfId in newDone:
            ctf = inputCtfSet[ctfId].clone()
            mic = ctf.getMicrograph().clone()
            ctfSet.append(ctf)
            micSet.append(mic)

        inputCtfSet.close()

    def updateRelations(self, cSet, mSet):
        micsAttrName = OUTPUT_MICS
        self._updateOutputSet(micsAttrName, mSet)
        # Set micrograph as pointer to protocol to prevent pointer end up as another attribute (String, Booelan,...)
        # that happens somewhere while scheduling.
        cSet.setMicrographs(Pointer(self, extended=micsAttrName))
        self._updateOutputSet(OUTPUT_CTF, cSet)
        self._defineTransformRelation(self.inputCTF.get().getMicrographs(), mSet)
        self._defineTransformRelation(self.inputCTF, cSet)
        self._defineCtfRelation(mSet, cSet)

    def _getAllDoneIds(self):
        doneIds = []
        sizeOutput = 0

        if hasattr(self, OUTPUT_CTF):
            sizeOutput = self.outputCTF.getSize()
            doneIds.extend(list(self.outputCTF.getIdSet()))

        return doneIds, sizeOutput

    def _summary(self):
        summary = []
        if not hasattr(self, OUTPUT_MICS):
            summary.append("Output set not ready yet.")
        else:
            populationSize = self.minImages.get()
            outputSize = self.outputMicrographs.getSize()
            summary.append("From %d micrographs extract a balanced defocus sample of: %d micrographs"
                           % (populationSize, outputSize))
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


    # Step 1: Get all defocus values and determine the bin edges
    defocus_values = list(image_dict.values())
    bin_edges = np.linspace(min(defocus_values), max(defocus_values), bins + 1)

    # Step 2: Organize image IDs by bins
    binned_images = defaultdict(list)
    for image_id, defocus in image_dict.items():
        # Find the bin index for the current defocus value
        bin_index = np.digitize(defocus, bin_edges) - 1
        # Avoid indexing beyond the available bins
        bin_index = min(bin_index, bins - 1)
        binned_images[bin_index].append(image_id)

    # Step 3: Calculate how many images to sample per bin
    images_per_bin = max(1, N // bins)
    sampled_images = []

    for bin_index in range(bins):
        images_in_bin = binned_images[bin_index]

        if len(images_in_bin) > images_per_bin:
            # Randomly sample from the bin if there are more images than needed
            sampled_images.extend(random.sample(images_in_bin, images_per_bin))
        else:
            # If fewer images than needed, take all images in this bin
            sampled_images.extend(images_in_bin)

    # If we have fewer than N images, randomly sample additional images to reach N
    if len(sampled_images) < N:
        remaining_images = list(set(image_dict.keys()) - set(sampled_images))
        sampled_images.extend(random.sample(remaining_images, N - len(sampled_images)))

    # Limit to N images in case there are extra
    return sampled_images[:N]


def compute_statistics(values):
    """
    Compute basic statistics for a list of numerical values.

    Parameters:
    - values (list or array-like): A list of numerical values (e.g., defocus values).

    Returns:
    - dict: A dictionary containing the statistics: min, max, mean, median, std, variance, and range.
    """

    # Convert to a NumPy array for efficient computation
    values = np.array(values)

    # Compute statistics
    stats = {
        "min": np.min(values),
        "max": np.max(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values, ddof=1),  # Sample standard deviation
        "variance": np.var(values, ddof=1),  # Sample variance
        "range": np.max(values) - np.min(values),
    }

    return stats
