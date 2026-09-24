# **************************************************************************
# *
# * Authors:     Laura del Cano (laura.cano@cnb.csic.es)
# *              Jose Gutierrez (jose.gutierrez@cnb.csic.es)
# *              I. Foche (ifoche@cnb.csic.es)
# *              Tomas Majtner (tmajtner@cnb.csic.es)   -- streaming version
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

import pyworkflow.protocol.constants as cons
from pyworkflow.utils import cleanPath
from pyworkflow.object import Set, Float
from pyworkflow.protocol.params import (EnumParam, IntParam, Positive,
                                        Range, LEVEL_ADVANCED, FloatParam,
                                        BooleanParam)

from pwem.constants import ALIGN_NONE
from pwem.emlib.metadata import getSize, isEmpty
from pwem.objects import SetOfParticles
from pwem.protocols import ProtProcessParticles

from pwem import emlib
from xmipp3.convert import readSetOfParticles, writeSetOfParticles


class XmippProtScreenParticles(ProtProcessParticles):
    """Protocol to attach different merit values to every particle metadata for subsequent pruning the set.
There are different merit values to be calculated:
    - zScore evaluates the similarity of a particles with an average (lower zScore -> higher similarity).
    - SSNR evaluates the signal/noise ration in the Fourier space.
    - Variance evaluates the varaince on the micrographs context where the particle was picked.

    AI Generated

    ## Overview

    The Screen Particles protocol computes several quality-related scores for a
    set of particles and can automatically reject particles according to selected
    criteria.

    Particle datasets often contain images that are poorly centered, noisy,
    contaminated, damaged, extracted from bad micrograph regions, or otherwise
    inconsistent with the rest of the dataset. These particles can reduce the
    quality of 2D classification, 3D refinement, and final reconstruction.

    This protocol evaluates particles using several Xmipp statistical criteria.
    The main criteria are:

    - Z-score, which measures how different a particle is from the average or from
      expected particle statistics;
    - SSNR, which evaluates signal-to-noise behavior in Fourier space;
    - variance and Gini-related measures, which evaluate the local micrograph
      context where the particle was picked.

    The protocol writes an output particle set containing the input particles with
    the computed metadata. Depending on the selected rejection options, some
    particles may be disabled and excluded from the final output.

    ## Inputs and General Workflow

    The input is a set of particles.

    The protocol converts the input particles to Xmipp metadata format and runs
    several screening programs. First, it computes statistical Z-score information.
    Then, it computes SSNR-related information. If requested, it also applies
    variance-based rejection using metadata already associated with the particles.

    The output is updated progressively and supports streaming input. New particles
    can be processed as they arrive, and the output particle set remains open until
    the input stream is closed and all particles have been processed.

    The final output particle set is also written to an Xmipp metadata file sorted
    by Z-score.

    ## Input Particles

    The **Input particles** parameter defines the particle set to be screened.

    The particles can come from extraction, previous particle-processing steps, or
    other Scipion-compatible workflows.

    The protocol does not perform alignment, classification, or reconstruction. It
    only computes screening metadata and optionally disables particles according to
    the selected automatic rejection rules.

    The output particles preserve the input particle information and add or update
    Xmipp screening attributes.

    ## Automatic Rejection by Z-score

    The **Automatic rejection by Zscore** parameter controls whether particles are
    rejected according to their Z-score statistics.

    There are three options:

    **None** computes and attaches Z-score information but does not reject
    particles based on this criterion.

    **MaxZscore** rejects particles whose Z-score is larger than the selected
    threshold.

    **Percentage** rejects a selected percentage of the worst particles according
    to several Z-score-related metadata labels.

    Z-score rejection is useful for removing particles that are unusually different
    from the rest of the dataset.

    ## Z-score Threshold

    The **zScore threshold** parameter is used when the rejection mode is
    **MaxZscore**.

    Particles with a Z-score larger than this value are rejected.

    A lower threshold is stricter and removes more particles. A higher threshold is
    more permissive.

    The default value is intended to remove strong outliers without being overly
    aggressive. Users should inspect the output distribution before using very
    strict thresholds.

    ## Percentage Rejection by Z-score

    The **Percentage** parameter is used when Z-score rejection is set to
    **Percentage**.

    The protocol rejects the selected percentage of worst particles according to
    Z-score-related labels, including shape, signal-to-noise, and histogram
    statistics.

    For example, if the percentage is 5, the protocol disables approximately the
    worst 5% of particles according to the Z-score screening criteria.

    This mode is useful when the user wants to remove a fixed fraction of likely
    outliers rather than choosing an absolute threshold.

    ## Automatic Rejection by SSNR

    The **Automatic rejection by SSNR** parameter controls whether particles are
    rejected according to SSNR.

    SSNR stands for spectral signal-to-noise ratio. It evaluates the relation
    between signal and noise in Fourier space.

    There are two options:

    **None** computes SSNR-related metadata but does not reject particles by SSNR.

    **Percentage** rejects the selected percentage of particles with the lowest
    SSNR values.

    Low SSNR particles may correspond to noisy, weak, damaged, or poorly extracted
    particles.

    ## Percentage Rejection by SSNR

    The **Percentage** parameter under SSNR rejection defines the percentage of
    particles to reject according to SSNR.

    For example, a value of 5 rejects approximately the worst 5% of particles by
    SSNR.

    This is useful when the user wants to remove particles with the weakest
    Fourier-space signal while keeping the rejection rate controlled.

    ## Automatic Rejection by Variance

    The **Automatic rejection by Variance** parameter controls whether particles
    are rejected according to variance-related information from the micrograph
    context.

    There are three options:

    **None** does not reject by variance.

    **Variance** rejects particles using the variance score alone.

    **Var. and Gini** rejects particles using a combined score based on variance
    and the Gini coefficient.

    This criterion is intended to identify particles extracted from problematic
    micrograph regions, such as areas with abnormal background, contamination,
    carbon, ice artifacts, or strong local intensity variation.

    ## Variance and Gini Scores

    Variance measures the amount of local intensity variation around the region
    where the particle was picked.

    The Gini coefficient provides additional information about inequality or
    concentration of intensity values. In the combined mode, the protocol uses a
    score based on variance multiplied by a Gini-related term.

    The threshold for variance-based rejection is estimated automatically from the
    score distribution using a histogram method.

    This option requires that particles already contain the necessary
    score-by-variance metadata. If the required attribute is missing, the protocol
    reports a validation error and suggests using Xmipp extraction to generate the
    needed information.

    ## Add Features

    The **Add features** option asks the protocol to attach additional ranking
    features to each input particle.

    These features are used internally by the ranking and screening program and can
    be useful for later inspection or analysis.

    This is an advanced option. It increases the amount of metadata associated
    with each particle but may be helpful when the user wants to examine screening
    criteria in more detail.

    ## Output Particles

    The main output is **outputParticles**.

    This output contains the screened particle set. Particles that pass the
    selected criteria are retained as enabled particles. Particles rejected by the
    selected automatic rejection rules are disabled and do not appear as accepted
    items in the usual output iteration.

    The output particle set copies the input metadata and adds Xmipp screening
    attributes such as Z-score and, when computed, SSNR and variance-related
    information.

    The output can be used directly in downstream protocols such as classification,
    refinement, or further particle cleaning.

    ## Metadata File Sorted by Z-score

    The protocol also writes an Xmipp metadata file named `images.xmd`, sorted by
    the particle Z-score.

    This file is useful for inspection, debugging, or advanced workflows where the
    user wants to examine particles in order of their screening score.

    Particles with worse scores can be inspected to understand what types of
    images are being rejected.

    ## Streaming Behavior

    The protocol supports streaming particle input.

    When the input particle set grows, the protocol detects new particles,
    processes only the new items, and appends the screened results to the output
    particle set.

    The output stream remains open while the input stream is open and closes when
    all input particles have been processed.

    This makes the protocol suitable for online or automated workflows, where
    particles are extracted progressively during data acquisition or early
    processing.

    ## Summary Information

    The protocol summary reports the selected rejection methods and, once output
    particles are available, basic Z-score statistics.

    These include:

    - minimum Z-score;
    - maximum Z-score;
    - mean Z-score.

    If variance rejection is enabled, the summary also reports the estimated
    variance threshold.

    This information helps the user understand how strict the screening was and
    whether the score distribution looks reasonable.

    ## Interpreting the Scores

    The scores should be interpreted as quality-control indicators, not as absolute
    biological truth.

    A particle with a poor Z-score, low SSNR, or abnormal variance may be a false
    positive, contaminant, noisy particle, or poorly extracted image. However,
    unusual particles can sometimes correspond to rare views, flexible states, or
    minority conformations.

    Automatic rejection is therefore useful, but it should be checked visually and
    validated through downstream classification.

    ## Practical Recommendations

    Use this protocol after particle extraction and before expensive classification
    or refinement steps.

    Start with no automatic rejection or a small rejection percentage to inspect
    the score distributions.

    Use Z-score rejection to remove strong statistical outliers.

    Use SSNR rejection to remove particles with weak Fourier-space signal.

    Use variance rejection when particles were extracted by Xmipp and contain the
    necessary micrograph-context variance metadata.

    Avoid overly aggressive rejection at early stages, especially if the dataset
    may contain rare views or conformational variability.

    Inspect rejected particles or low-ranked particles to confirm that the selected
    criteria behave as expected.

    Use the screened output as input for 2D classification or further particle
    cleaning.

    ## Final Perspective

    Screen Particles is a particle-quality assessment and pruning protocol.

    For biological users, its main value is that it attaches interpretable
    screening scores to particles and optionally removes likely outliers before
    downstream processing.

    The protocol is most useful as an early particle-cleaning step, especially in
    large datasets where manual inspection of all extracted particles is not
    practical. Its decisions should be combined with visual inspection, 2D
    classification, and later reconstruction behavior.
    """

    _label = 'screen particles'

    # Automatic Particle rejection enum
    ZSCORE_CHOICES = ['None', 'MaxZscore', 'Percentage']
    SSNR_CHOICES = ['None', 'Percentage']
    VAR_CHOICES = ['None', 'Variance', 'Var. and Gini']
    REJ_NONE = 0
    REJ_MAXZSCORE = 1
    REJ_PERCENTAGE = 2
    REJ_PERCENTAGE_SSNR = 1
    REJ_VARIANCE = 1
    REJ_VARGINI = 2

    # --------------------------- DEFINE param functions ---------------------
    def _defineProcessParams(self, form):
        # --- zScore rejection ---
        form.addParam('autoParRejection', EnumParam,
                      choices=self.ZSCORE_CHOICES,
                      label="Automatic rejection by Zscore",
                      default=self.REJ_NONE,
                      display=EnumParam.DISPLAY_COMBO,
                      help='zScore evaluates the similarity of a particles '
                           'with an average. The rejection can be:\n'
                           '  None (no rejection)\n'
                           '  MaxZscore (reject a particle if its zScore '
                           'is larger than this value).\n '
                           '  Percentage (reject a given percentage for '
                           'this criteria).')
        form.addParam('maxZscore_', FloatParam, default=3, validators=[Positive],
                      condition='autoParRejection==1', label='zScore threshold',
                      help='Maximum Zscore.')
        form.addParam('percentage', IntParam, default=5, label='Percentage (%)',
                      condition='autoParRejection==2',
                      help='The worse percentage of particles according to '
                           'metadata labels: ZScoreShape1, ZScoreShape2, '
                           'ZScoreSNR1, ZScoreSNR2, ZScoreHistogram are '
                           'automatically disabled.',
                      validators=[Range(0, 100, error="Percentage must be "
                                                      "between 0 and 100.")])
        # --- SSNR rejection ---
        form.addParam('autoParRejectionSSNR', EnumParam,
                      choices=self.SSNR_CHOICES,
                      label="Automatic rejection by SSNR",
                      default=self.REJ_NONE, display=EnumParam.DISPLAY_COMBO,
                      help='SSNR evaluates the signal/noise ration in the '
                           'Fourier space. The rejection can be:\n'
                           '  None (no rejection)\n'
                           '  Percentage (reject a given percentage of the '
                           'lowest SSNRs).')
        form.addParam('percentageSSNR', IntParam, default=5,
                      condition='autoParRejectionSSNR==1',
                      label='Percentage (%)',
                      help='The worse percentage of particles according to '
                           'SSNR are automatically disabled.',
                      validators=[Range(0, 100, error="Percentage must be "
                                                      "between 0 and 100.")])
        # --- Variance rejection ---
        form.addParam('autoParRejectionVar', EnumParam, default=self.REJ_NONE,
                      choices=self.VAR_CHOICES,
                      label='Automatic rejection by Variance',
                      help='Variance evaluates the varaince on the micrographs '
                           'context where the particle was picked. '
                           'The rejection can be:\n'
                           '  None (no rejection)\n'
                           '  Variance (taking into account only the variance)\n'
                           '  Var. and Gini (taking into account also the Gini '
                           'coeff.)')
        # --- Add features ---
        form.addParam('addFeatures', BooleanParam, default=False,
                      label='Add features', expertLevel=LEVEL_ADVANCED,
                      help='Add features used for the ranking to each one '
                           'of the input particles')

        form.addParallelSection(threads=0, mpi=0)

    def _getDefaultParallel(self):
        """This protocol doesn't have mpi version"""
        return (0, 0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initializeZscores()
        self.outputSize = 0
        self.inputSize = 0
        self.check = None
        self.fnInputMd = self._getExtraPath("input.xmd")
        self.fnInputOldMd = self._getExtraPath("inputOld.xmd")
        self.fnOutputMd = self._getExtraPath("output.xmd")

        self.inputSize, self.streamClosed = self._loadInput()
        partsSteps = self._insertNewPartsSteps()
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=partsSteps, wait=True)

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

    def createOutputStep(self):
        pass

    def _insertNewPartsSteps(self):
        deps = []
        stepId = self._insertFunctionStep('sortImagesStep', prerequisites=[])
        deps.append(stepId)
        return deps

    def _stepsCheck(self):
        # Input particles set can be loaded or None when checked for new inputs
        # If None, we load it
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        # Check if there are new particles to process from the input set
        partsFile = self.inputParticles.get().getFileName()
        mTime = datetime.fromtimestamp(os.path.getmtime(partsFile))
        # If the input movies.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime:
            return None

        self.inputSize, self.streamClosed = self._loadInput()
        if not isEmpty(self.fnInputMd):
            fDeps = self._insertNewPartsSteps()
            outputStep = self._getFirstJoinStep()
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def _loadInput(self):
        self.lastCheck = datetime.now()
        partsFile = self.inputParticles.get().getFileName()
        inPartsSet = SetOfParticles(filename=partsFile)
        inPartsSet.loadAllProperties()

        check = None
        for p in inPartsSet.iterItems(orderBy='creation', direction='DESC'):
            check = p.getObjCreation()
            break
        if self.check is None:
            writeSetOfParticles(inPartsSet, self.fnInputMd,
                                alignType=ALIGN_NONE, orderBy='creation')
        else:
            writeSetOfParticles(inPartsSet, self.fnInputMd,
                                alignType=ALIGN_NONE, orderBy='creation',
                                where='creation>"' + str(self.check) + '"')
            writeSetOfParticles(inPartsSet, self.fnInputOldMd,
                                alignType=ALIGN_NONE, orderBy='creation',
                                where='creation<"' + str(self.check) + '"')
        self.check = check

        streamClosed = inPartsSet.isStreamClosed()
        inputSize = inPartsSet.getSize()

        inPartsSet.close()

        return inputSize, streamClosed

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        self.finished = self.streamClosed and \
                        self.outputSize == self.inputSize

        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        newData = os.path.exists(self.fnOutputMd)
        lastToClose = self.finished and hasattr(self, 'outputParticles')
        if newData or lastToClose:

            outSet = self._loadOutputSet(SetOfParticles, 'outputParticles.sqlite')

            if newData:
                partsSet = self._createSetOfParticles()
                readSetOfParticles(self.fnOutputMd, partsSet)
                outSet.copyItems(partsSet)
                for item in partsSet:
                    self._calculateSummaryValues(item)
                self._store()

                writeSetOfParticles(outSet.iterItems(orderBy='_xmipp_zScore'),
                                    self._getPath("images.xmd"),
                                    alignType=ALIGN_NONE)
                cleanPath(self.fnOutputMd)

            self._updateOutputSet('outputParticles', outSet, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)

    def _loadOutputSet(self, SetClass, baseName):
        setFile = self._getPath(baseName)
        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)
            self._store(outputSet)
            self._defineTransformRelation(self.inputParticles, outputSet)

        outputSet.copyInfo(self.inputParticles.get())

        return outputSet

    # --------------------------- STEP functions -----------------------------
    def sortImagesStep(self):
        args = "-i Particles@%s -o %s --addToInput " % (self.fnInputMd,
                                                        self.fnOutputMd)
        if os.path.exists(self.fnInputOldMd):
            args += "-t Particles@%s " % self.fnInputOldMd

        if self.autoParRejection == self.REJ_MAXZSCORE:
            args += "--zcut " + str(self.maxZscore_.get())
        elif self.autoParRejection == self.REJ_PERCENTAGE:
            args += "--percent " + str(self.percentage.get())

        if self.addFeatures:
            args += "--addFeatures "

        self.runJob("xmipp_image_sort_by_statistics", args)

        args = "-i Particles@%s -o %s" % (self.fnInputMd, self.fnOutputMd)
        if self.autoParRejectionSSNR == self.REJ_PERCENTAGE_SSNR:
            args += " --ssnrpercent " + str(self.percentageSSNR.get())
        self.runJob("xmipp_image_ssnr", args)

        if self.autoParRejectionVar != self.REJ_NONE:
            print('Rejecting by variance:')
            if self.outputSize == 0:
                varList = []
                giniList = []
                print('  - Reading metadata')
                mdata = emlib.MetaData(self.fnInputMd)
                for objId in mdata:
                    varList.append(mdata.getValue(emlib.MDL_SCORE_BY_VAR, objId))
                    giniList.append(mdata.getValue(emlib.MDL_SCORE_BY_GINI, objId))

                if self.autoParRejectionVar == self.REJ_VARIANCE:
                    valuesList = varList
                    self.mdLabels = [emlib.MDL_SCORE_BY_VAR]
                else:  # not working pretty well
                    valuesList = [var * (1 - gini) for var, gini in zip(varList, giniList)]
                    self.mdLabels = [emlib.MDL_SCORE_BY_VAR, emlib.MDL_SCORE_BY_GINI]

                self.varThreshold.set(histThresholding(valuesList))
                print('  - Variance threshold: %f' % self.varThreshold)

            rejectByVariance(self.fnInputMd, self.fnOutputMd, self.varThreshold,
                             self.autoParRejectionVar)

        # update the processed particles
        self.outputSize += getSize(self.fnInputMd)

    def _initializeZscores(self):
        # Store the set for later access , ;-(
        self.minZScore = Float()
        self.maxZScore = Float()
        self.sumZScore = Float()
        self.varThreshold = Float()
        self._store()

    def _calculateSummaryValues(self, particle):
        zScore = particle._xmipp_zScore.get()

        self.minZScore.set(min(zScore, self.minZScore.get(1000)))
        self.maxZScore.set(max(zScore, self.maxZScore.get(0)))
        self.sumZScore.set(self.sumZScore.get(0) + zScore)

    # -------------------------- INFO functions ------------------------------
    def _summary(self):
        summary = []

        sumRejMet = {}  # A dict with the form choices
        if self.autoParRejection is not None:
            metStr = self.ZSCORE_CHOICES[self.autoParRejection.get()]
            if self.autoParRejection.get() == self.REJ_MAXZSCORE:
                metStr += " = %.2f" % self.maxZscore_.get()
            elif self.autoParRejection.get() == self.REJ_PERCENTAGE:
                metStr += " = %.2f" % self.percentage.get()
            sumRejMet['Zscore'] = ("Zscore rejection method: " + metStr)

        if self.autoParRejectionSSNR is not None:
            metStr = self.SSNR_CHOICES[self.autoParRejectionSSNR.get()]
            if self.autoParRejectionSSNR.get() == self.REJ_PERCENTAGE_SSNR:
                metStr += " = %.2f" % self.percentageSSNR.get()
            sumRejMet['SSNR'] = ("SSNR rejection method: " + metStr)

        if self.autoParRejectionVar is not None:
            sumRejMet['Var'] = ("Variance rejection method: " +
                                self.VAR_CHOICES[self.autoParRejectionVar.get()])

        # If no output yet, just the form choices are shown plus a no-ready text
        if not hasattr(self, 'outputParticles'):
            summary += sumRejMet.values()
            summary.append("Output particles not ready yet.")
        else:
            if 'Zscore' in sumRejMet:
                summary.append(sumRejMet['Zscore'])
            if hasattr(self, 'sumZScore'):
                summary.append(" - The minimum ZScore is %.2f" % self.minZScore)
                summary.append(" - The maximum ZScore is %.2f" % self.maxZScore)
                meanZScore = self.sumZScore.get() * 1.0 / len(self.outputParticles)
                summary.append(" - The mean ZScore is %.2f" % meanZScore)
            else:
                summary.append(
                    "Summary values not calculated during processing.")
            if 'SSNR' in sumRejMet:
                summary.append(sumRejMet['SSNR'])
            if 'Var' in sumRejMet:
                summary.append(sumRejMet['Var'])
            if self.autoParRejectionVar != self.REJ_NONE:
                if hasattr(self, 'varThreshold'):
                    summary.append(" - Variance threshold: %.2f" % self.varThreshold)
                else:
                    summary.append(" - Variance threshold not calculed yet.")
        return summary

    def _validate(self):
        validateMsgs = []
        if self.autoParRejectionVar != self.REJ_NONE:
            part = self.inputParticles.get().getFirstItem()
            if not part.hasAttribute('_xmipp_scoreByVariance'):
                validateMsgs.append('The auto-rejection by Variance can not be '
                                    'done because the particles have not the '
                                    'scoreByVariance attribute. Use Xmipp to '
                                    'extract the particles.')
        return validateMsgs

    def _citations(self):
        return ['Vargas2013b']

    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            outParticles = (len(self.outputParticles) if self.outputParticles
                                                         is not None else None)
            particlesRejected = (len(self.inputParticles.get()) - outParticles
                                 if outParticles is not None else None)
            particlesRejectedText = (' (' + str(particlesRejected) + ')' if
                                     particlesRejected is not None else '')
            rejectionText = ['',  # REJ_NONE
                             ' and removing those not reaching %s%s'
                             % (str(self.maxZscore_.get()),
                                particlesRejectedText),  # REJ_MAXZSCORE
                             ' and removing worst %s percent %s'
                             % (str(self.percentage.get()),
                                particlesRejectedText)  # REJ_PERCENTAGE
                             ]
            methods.append('Input dataset %s of %s particles was sorted by'
                           ' its ZScore using xmipp_image_sort_by_statistics'
                           ' program%s. '
                           % (self.getObjectTag('inputParticles'),
                              len(self.inputParticles.get()),
                              rejectionText[self.autoParRejection.get()]))
            methods.append('Output set is %s.'
                           % self.getObjectTag('outputParticles'))
        return methods


# -------------------------- EXTERNAL functions ------------------------------
def histThresholding(valuesList, nBins=256, portion=4, takeNegatives=False):
    """ returns the threshold to reject those values above a portionth of 
        the peak. i.e: if portion is 4, the threshold correponds to the
        4th of the peak (in the right part).
    """
    if not takeNegatives:
        # take only the positive values, negative are considered corrupted
        valuesList = [x for x in valuesList if not x < 0]

    import numpy as np
    while len(valuesList) * 1.0 / nBins < 5:
        nBins = int(nBins / 2)

    print('Thresholding with %d bins for the histogram.' % nBins)

    hist, bin_edges = np.histogram(valuesList, bins=nBins)

    histRight = hist
    histRight[0:hist.argmax()] = 0

    idx = (np.abs(histRight - hist.max() / portion)).argmin()
    return bin_edges[idx]


def rejectByVariance(inputMdFn, outputMdFn, threshold, mode):
    """ Sets MDL_ENABLED to -1 to those items with a higher value
        than the threshold
    """
    mdata = emlib.MetaData(inputMdFn)
    for objId in mdata:
        if mode == XmippProtScreenParticles.REJ_VARIANCE:
            if mdata.getValue(emlib.MDL_SCORE_BY_VAR, objId) > threshold:
                mdata.setValue(emlib.MDL_ENABLED, -1, objId)
        elif mode == XmippProtScreenParticles.REJ_VARGINI:
            if (mdata.getValue(emlib.MDL_SCORE_BY_VAR, objId) *
                    (1 - mdata.getValue(emlib.MDL_SCORE_BY_GINI, objId)) > threshold):
                mdata.setValue(emlib.MDL_ENABLED, -1, objId)

    mdata.write(outputMdFn)
