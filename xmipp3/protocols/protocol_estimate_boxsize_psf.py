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
from pwem.objects import SetOfCTF, Integer
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils

from pwem.protocols import ProtCTFMicrographs
import pwem.emlib.metadata as md
from pwem import emlib

from pyworkflow.protocol.constants import (STATUS_NEW)
from pyworkflow import UPDATED, NEW
import xmippLib

from xmipp3.convert import ctfModelToRow, acquisitionToRow


OUTPUT_BOXSIZE_A =  "extractionBoxSizeA"
OUTPUT_BOXSIZE_PX = "extractionBoxSizePx"

class XmippProtEstimateBoxsizePSF(ProtCTFMicrographs):
    """
    Protocol to estimate meaningful particle extraction boxsize based on the CTFs estimated values.
    An integer representing the recommended value will be output. CTFs with different defocus values look differently,
    while particles with higher defocus values will be recommended a bigger box size for extraction,
    particles with lower defocus values will be recommended a more compact one.
    """
    _label = 'PSF particle extraction box size'
    _devStatus = NEW
    _lastUpdateVersion = VERSION_3_0
    _possibleOutputs = {OUTPUT_BOXSIZE_A: Integer,
                        OUTPUT_BOXSIZE_PX: Integer}


    def __init__(self, **args):
        ProtCTFMicrographs.__init__(self, **args)

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCTF', params.PointerParam, pointerClass='SetOfCTF',
                      label="Input CTF", important=True,
                      help='Select the estimated CTF to evaluate.')
        form.addParam('boxSize', params.IntParam,
                      label='Particle box size (px)',
                      allowsPointers=True,
                      important=True,
                      help='This is size of the boxed particles (in pixels).\n'
                           'For sanity check if it is not, it will be transform to an even number.')
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
        self.estimatedBoxSizesA = []
        self.estimatedBoxSizesPx = []
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
        stepId = self._insertFunctionStep(self.estimateBoxSizePSF, newIds,  needsGPU=False, prerequisites=[])
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
            if hasattr(self, OUTPUT_BOXSIZE_A):
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

    def estimateBoxSizePSF(self, ctfIds):
        inputCtfSet = self._loadInputCtfSet(self.ctfFn)
        pickingBoxSize = self.boxSize.get()
        ts = self.inputCTF.get().getMicrographs().getSamplingRate()
        print(ts)
        ctfDefocus = {}

        for ctfId in ctfIds:
            ctf = inputCtfSet.getItem("id", ctfId).clone()
            defocusU = ctf.getDefocusU()
            ctfDefocus[ctfId] = defocusU

        sampled_images = balanced_sampling(image_dict=ctfDefocus, N=self.numImages.get(), bins=10)
        self.info('The number of CTFs selected for defocus balanced sampling is the following: %d'
                  %len(sampled_images))

        for ctfId in sampled_images:
            ctf = inputCtfSet.getItem("id", ctfId).clone()
            mic = ctf.getMicrograph()
            baseMicName = mic.getBaseName()
            fnCTF = self._getTmpPath("%s.ctfParam" % baseMicName)
            ctfToCTFParam(ctf, fnCTF)
            psf = np.abs(xmippLib.getPSF(fnCTF, ts))
            thPSF = 0.001 * np.max(psf)
            supportPSF_A = np.sum(psf > thPSF) * ts
            supportPSF_Px = np.sum(psf > thPSF)
            estimatedBoxSizeA = pickingBoxSize * ts + supportPSF_A
            estimatedBoxSizePx = pickingBoxSize + supportPSF_Px
            self.estimatedBoxSizesA.append(estimatedBoxSizeA)
            self.estimatedBoxSizesPx.append(estimatedBoxSizePx)
            self.info("Defocus %0.1f - estimated box size %0.1f(A)" % (ctfDefocus[ctfId], estimatedBoxSizeA))

    def _checkNewOutput(self):
        """ Check for already selected CTF and update the output set. """
        if not self.finished:
            if self.estimatedBoxSizesA:
                boxSizeA, boxSizePx = self.estimateFinalBoxSize(self.estimatedBoxSizesA, self.estimatedBoxSizesPx)
                self._defineOutputs(**{OUTPUT_BOXSIZE_A: boxSizeA, OUTPUT_BOXSIZE_PX: boxSizePx})
                message = ("The estimated extraction box size are the following: box size in Angstroms %d and box size in pixels %d"
                           % (boxSizeA, boxSizePx))
                self.summaryVar.set(message)
                self.finished = True
                self._store()  # Update the summary dictionary
        else:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

    def estimateFinalBoxSize(self, boxSizesA, boxSizesPx):
        maxBoxSizeA = max(boxSizesA)
        maxBoxSizePx = max(boxSizesPx)
        boxSizeA = transform2EvenNumber(maxBoxSizeA)
        boxSizePx = transform2EvenNumber(maxBoxSizePx)

        return  Integer(boxSizeA),  Integer(boxSizePx)

    def _summary(self):
        summary = []
        if not hasattr(self, OUTPUT_BOXSIZE_A):
            summary.append("Output set not ready yet.")
        else:
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

def transform2EvenNumber(var):
    if var % 2 != 0:
        var = round(var / 2) * 2
    return var

def ctfToCTFParam(ctf, ctfparam):
    """ This function is used to convert a CTF object
    to the .ctfparam metadata needed by some Xmipp programs.
    If the micrograph already comes from xmipp, the ctfparam
    will be returned, if not, the new file.
    """
    mdCtf = emlib.MetaData()
    mdCtf.setColumnFormat(False)
    row = md.Row()
    ctfModelToRow(ctf, row)
    acquisitionToRow(ctf.getMicrograph().getAcquisition(), row)
    row.writeToMd(mdCtf, mdCtf.addObject())
    mdCtf.write(ctfparam)
    return ctfparam