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

import pyworkflow.em as em
import pyworkflow.protocol.constants as cons

from pyworkflow.em.data import SetOfParticles
from pyworkflow.em.protocol import ProtProcessParticles
from pyworkflow.object import Set, Float
from pyworkflow.protocol.params import (EnumParam, IntParam, Positive,
                                        Range, LEVEL_ADVANCED, FloatParam,
                                        BooleanParam)

import xmippLib
from xmipp3.convert import readSetOfParticles, writeSetOfParticles


class XmippProtScreenParticles(ProtProcessParticles):
    """ Classify particles according their similarity to the others in order
    to detect outliers. """

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

        form.addParam('autoParRejection', EnumParam,
                      choices=self.ZSCORE_CHOICES,
                      label="Automatic particle rejection based on Zscore",
                      default=self.REJ_NONE,
                      display=EnumParam.DISPLAY_COMBO,
                      expertLevel=LEVEL_ADVANCED,
                      help='How to automatically reject particles. It can be:\n'
                           '  None (no rejection)\n'
                           '  MaxZscore (reject a particle if its Zscore [a '
                           'similarity index] is larger than this value).\n '
                           '  Percentage (reject a given percentage in each '
                           'one of the screening criteria).')
        form.addParam('maxZscore', FloatParam, default=3,
                      condition='autoParRejection==1',
                      label='Maximum Zscore', expertLevel=LEVEL_ADVANCED,
                      help='Maximum Zscore.', validators=[Positive])
        form.addParam('percentage', IntParam, default=5,
                      condition='autoParRejection==2',
                      label='Percentage (%)', expertLevel=LEVEL_ADVANCED,
                      help='The worse percentage of particles according to '
                           'metadata labels: ZScoreShape1, ZScoreShape2, '
                           'ZScoreSNR1, ZScoreSNR2, ZScoreHistogram are '
                           'automatically disabled. Therefore, the total '
                           'number of disabled particles belongs to ['
                           'percetage, 5*percentage]',
                      validators=[Range(0, 100, error="Percentage must be "
                                                      "between 0 and 100.")])
        form.addParam('autoParRejectionSSNR', EnumParam,
                      choices=self.SSNR_CHOICES,
                      label="Automatic particle rejection based on SSNR",
                      default=self.REJ_NONE, display=EnumParam.DISPLAY_COMBO,
                      expertLevel=LEVEL_ADVANCED,
                      help='How to automatically reject particles based on '
                           'SSNR. It can be:\n'
                           '  None (no rejection)\n'
                           'Percentage (reject a given percentage of the '
                           'lowest SSNRs).')
        form.addParam('percentageSSNR', IntParam, default=5,
                      condition='autoParRejectionSSNR==1',
                      label='Percentage (%)', expertLevel=LEVEL_ADVANCED,
                      help='The worse percentage of particles according to '
                           'SSNR are automatically disabled.',
                      validators=[Range(0, 100, error="Percentage must be "
                                                      "between 0 and 100.")])
        form.addParam('autoParRejectionVar', EnumParam, default=self.REJ_NONE,
                      choices=self.VAR_CHOICES,
                      label='Automatic particle rejection based on Variance',
                      expertLevel=LEVEL_ADVANCED,
                      help='How to automatically reject particles based on '
                           'Variance. It can be:\n'
                           '  None (no rejection)\n'
                           '  Variance (taking into account only the variance)\n'
                           '  Var. and Gini (taking into account also the Gini coeff.)')
        form.addParam('addFeatures', BooleanParam, default=False,
                      label='Add features', expertLevel=LEVEL_ADVANCED,
                      help='Add features used for the ranking to each one '
                           'of the input particles')
        form.addParallelSection(threads=0, mpi=0)


    def _getDefaultParallel(self):
        """This protocol doesn't have mpi version"""
        return (0, 0)

    #--------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initializeZscores()
        self.outputSize = 0
        self.check = None
        self.streamClosed = self.inputParticles.get().isStreamClosed()
        partsSteps = self._insertNewPartsSteps()
        self._insertFunctionStep('createOutputStep',
                                 prerequisites=partsSteps, wait=True)

    def createOutputStep(self):
        pass

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

    def _insertNewPartsSteps(self):
        deps = []
        stepId = self._insertFunctionStep('sortImages',
                                          self._getExtraPath("input.xmd"),
                                          self._getExtraPath("inputOld.xmd"),
                                          self._getExtraPath("output.xmd"),
                                          prerequisites=[])
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
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(os.path.getmtime(partsFile))
        # If the input movies.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime:
            return None
        self.lastCheck = now
        outputStep = self._getFirstJoinStep()
        fDeps = self._insertNewPartsSteps()
        if outputStep is not None:
            outputStep.addPrerequisites(*fDeps)
        self.updateSteps()

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return
        self.finished = self.streamClosed and \
                        self.outputSize == len(self.outputParticles)
        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)

    def _loadOutputSet(self, SetClass, baseName, fnMd):
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

        inputs = self.inputParticles.get()
        outputSet.copyInfo(inputs)
        partsSet = self._createSetOfParticles()
        readSetOfParticles(fnMd, partsSet)
        os.remove(fnMd)
        self.outputSize = self.outputSize + len(partsSet)
        outputSet.copyItems(partsSet)
        for item in partsSet:
            self._calculateSummaryValues(item)
        self._store()   # Persist zScore values for summary and testing
        writeSetOfParticles(outputSet.iterItems(orderBy='_xmipp_zScore'),
                            self._getPath("images.xmd"),
                            alignType=em.ALIGN_NONE)
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
            self._store(outputSet)

        # Close set databaset to avoid locking it
        outputSet.close()

    #--------------------------- STEPS functions -----------------------------
    def sortImages(self, fnInputMd, fnInputOldMd, fnOutputMd):
        partsFile = self.inputParticles.get().getFileName()
        self.outputParticles = SetOfParticles(filename=partsFile)
        self.outputParticles.loadAllProperties()
        self.streamClosed = self.outputParticles.isStreamClosed()
        if self.check == None:
            writeSetOfParticles(self.outputParticles, fnInputMd,
                                alignType=em.ALIGN_NONE, orderBy='creation')
        else:
            writeSetOfParticles(self.outputParticles, fnInputMd,
                                alignType=em.ALIGN_NONE, orderBy='creation',
                                where='creation>"' + str(self.check) + '"')
            writeSetOfParticles(self.outputParticles, fnInputOldMd,
                                alignType=em.ALIGN_NONE, orderBy='creation',
                                where='creation<"' + str(self.check) + '"')
        if (self.outputSize >= len(self.outputParticles)):
            return
        args = "-i Particles@%s -o %s --addToInput " % (fnInputMd, fnOutputMd)
        if self.check != None:
            args += "-t Particles@%s " % fnInputOldMd
        for p in self.outputParticles.iterItems(orderBy='creation',
                                                direction='DESC'):
            self.check = p.getObjCreation()
            break
        self.outputParticles.close()
        if self.autoParRejection == self.REJ_MAXZSCORE:
            args += "--zcut " + str(self.maxZscore.get())
        elif self.autoParRejection == self.REJ_PERCENTAGE:
            args += "--percent " + str(self.percentage.get())
        if self.addFeatures:
            args += "--addFeatures "
        self.runJob("xmipp_image_sort_by_statistics", args)

        args = "-i Particles@%s -o %s" % (fnInputMd, fnOutputMd)
        if self.autoParRejectionSSNR == self.REJ_PERCENTAGE_SSNR:
            args += " --ssnrpercent " + str(self.percentageSSNR.get())
        self.runJob("xmipp_image_ssnr", args)

        if self.autoParRejectionVar != self.REJ_NONE:
            print('Rejecting by variance:')
            if self.outputSize == 0:
                varList = []
                giniList = []
                print('  - Reading metadata')
                mdata = xmippLib.MetaData(fnInputMd)
                for objId in mdata:
                    varList.append(mdata.getValue(xmippLib.MDL_SCORE_BY_VAR, objId))
                    giniList.append(mdata.getValue(xmippLib.MDL_SCORE_BY_GINI, objId))

                if self.autoParRejectionVar == self.REJ_VARIANCE:
                    valuesList = varList
                    self.mdLabels = [xmippLib.MDL_SCORE_BY_VAR]
                else:  # not working pretty well
                    valuesList = [var*(1-gini) for var, gini in zip(varList, giniList)]
                    self.mdLabels = [xmippLib.MDL_SCORE_BY_VAR, xmippLib.MDL_SCORE_BY_GINI]

                self.varThreshold.set(histThresholding(valuesList))
                print('  - Variance threshold: %f' % self.varThreshold)

            rejectByVariance(fnInputMd, fnOutputMd, self.varThreshold,
                             self.autoParRejectionVar)

        streamMode = Set.STREAM_CLOSED \
            if getattr(self, 'finished', False) else Set.STREAM_OPEN
        outSet = self._loadOutputSet(SetOfParticles, 'outputParticles.sqlite',
                                     fnOutputMd)
        self._updateOutputSet('outputParticles', outSet, streamMode)

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
                metStr += " = %.2f" % self.maxZscore.get()
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
            summary.append(sumRejMet['Zscore'])
            if hasattr(self, 'sumZScore'):
                summary.append(" - The minimum ZScore is %.2f" % self.minZScore)
                summary.append(" - The maximum ZScore is %.2f" % self.maxZScore)
                meanZScore = self.sumZScore.get() * 1.0 / len(self.outputParticles)
                summary.append(" - The mean ZScore is %.2f" % meanZScore)
            else:
                summary.append(
                    "Summary values not calculated during processing.")
            summary.append(sumRejMet['SSNR'])
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
            particlesRejected = (len(self.inputParticles.get())-outParticles
                                 if outParticles is not None else None)
            particlesRejectedText = (' ('+str(particlesRejected)+')' if
                                     particlesRejected is not None else '')
            rejectionText = ['',  # REJ_NONE
                             ' and removing those not reaching %s%s'
                             % (str(self.maxZscore.get()),
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


# -------------------------- UTILS functions ------------------------------
def histThresholding(valuesList, nBins=256, portion=4):
    """ returns the threshold to reject those values above a portionth of 
        the peak. i.e: if portion is 4, the threshold correponds to the
        4th of the peak (in the right part).
    """
    import numpy as np
    while len(valuesList)*1.0/nBins < 5:
        nBins = nBins/2

    print('Thresholding with %d bins for the histogram.' % nBins)

    hist, bin_edges = np.histogram(valuesList, bins=nBins)

    histRight = hist
    histRight[0:hist.argmax()] = 0

    idx = (np.abs(histRight-hist.max()/portion)).argmin()
    return bin_edges[idx] 

def rejectByVariance(inputMdFn, outputMdFn, threshold, mode):
    """ Sets MDL_ENABLED to -1 to those items with a higher value
        than the threshold
    """
    mdata = xmippLib.MetaData(inputMdFn)
    for objId in mdata:
        if mode == XmippProtScreenParticles.REJ_VARIANCE:
            if mdata.getValue(xmippLib.MDL_SCORE_BY_VAR, objId) > threshold:
                mdata.setValue(xmippLib.MDL_ENABLED, -1, objId)
        elif mode == XmippProtScreenParticles.REJ_VARGINI:
            if (mdata.getValue(xmippLib.MDL_SCORE_BY_VAR, objId) *
                (1 - mdata.getValue(xmippLib.MDL_SCORE_BY_GINI, objId)) > threshold):
                mdata.setValue(xmippLib.MDL_ENABLED, -1, objId)

    mdata.write(outputMdFn)
