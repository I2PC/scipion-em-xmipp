# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Roberto Marabini (roberto@cnb.csic.es)
# *              Tomas Majtner (tmajtner@cnb.csic.es)  -- streaming version
# *              Amaya Jimenez (ajimenez@cnb.csic.es)
# *              Daniel Marchan (da.marchan@cnb.csic.es)  --refactor streaming
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
from cmath import rect, phase
from math import radians, degrees

from pyworkflow import VERSION_3_0
from pwem.objects import SetOfCTF, SetOfMicrographs
from pyworkflow.object import Set, Integer, Pointer
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils

from pwem.protocols import ProtCTFMicrographs
from pwem.emlib.metadata import Row
from pyworkflow.protocol.constants import (STATUS_NEW)
from pyworkflow import BETA, UPDATED, NEW, PROD

from pwem import emlib
import xmipp3
from xmipp3.convert import setXmippAttribute, getScipionObj

ACCEPTED = 'Accepted'
DISCARDED = 'Discarded'
INPUT1 = 1
INPUT2 = 2

OUTPUT_CTF =  "outputCTF"
OUTPUT_MICS = "outputMicrographs"
OUTPUT_CTF_DISCARDED = "outputCTFDiscarded"
OUTPUT_MICS_DISCARDED = "outputMicrographsDiscarded"

class XmippProtCTFConsensus(ProtCTFMicrographs):
    """
    Protocol to make a selection of meaningful CTFs in basis of the defocus
    values, the astigmatism, the resolution, other Xmipp parameters, and
    the agreement with a secondary CTF for the same set of micrographs.
    """
    _label = 'ctf consensus'
    _devStatus = PROD
    _lastUpdateVersion = VERSION_3_0
    _possibleOutputs = {OUTPUT_MICS: SetOfMicrographs,
                        OUTPUT_MICS_DISCARDED: SetOfMicrographs,
                        OUTPUT_CTF: SetOfCTF,
                        OUTPUT_CTF_DISCARDED: SetOfCTF
                        }


    def __init__(self, **args):
        ProtCTFMicrographs.__init__(self, **args)
        self._freqResol = {}

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCTF', params.PointerParam, pointerClass='SetOfCTF',
                      label="Input CTF", important=True,
                      help='Select the estimated CTF to evaluate')

        form.addParam('useDefocus', params.BooleanParam, default=True,
                      label='Use Defocus for selection',
                      help='Use this button to decide if carry out the '
                           'selection taking into account or not the defocus '
                           'values.')
        line = form.addLine('Defocus (A)', condition="useDefocus",
                            help='Minimum and maximum values for defocus in '
                                 'Angstroms.\nMicrographs out of this range '
                                 'will left out of the output.')
        line.addParam('minDefocus', params.FloatParam, default=4000,
                      label='Min')
        line.addParam('maxDefocus', params.FloatParam,
                      default=40000, label='Max')

        form.addParam('useAstigmatism', params.BooleanParam, default=False,
                      label='Use Astigmatism for selection',
                      help='Use this button to decide if carry out the '
                           'selection taking into account or not the '
                           'astigmatism value.')
        form.addParam('astigmatism', params.FloatParam, default=1000,
                      label='Astigmatism (A)', condition="useAstigmatism",
                      help='Maximum value allowed for astigmatism in '
                           'Angstroms. If the evaluated CTF has a '
                           'larger Astigmatism, it will be discarded.')

        form.addParam('useAstigmatismPercentage', params.BooleanParam, default=True,
                      label='Use Astigmatism percentage for selection',
                      help='Use this button to decide if carry out the '
                           'selection taking into account or not the '
                           'astigmatism value.')
        form.addParam('astigmatismPer', params.FloatParam, default=0.10,
                      label='Astigmatism percentage', condition="useAstigmatismPercentage",
                      help='Maximum value allowed for astigmatism '
                           'percentage (|defocus_U-defocus_V|/mean_defocus). If the evaluated CTF has a '
                           'larger Astigmatism Percentage, it will be discarded.')

        form.addParam('useResolution', params.BooleanParam, default=True,
                      label='Use Resolution for selection',
                      help='Use this button to decide if carry out the '
                           'selection taking into account or not the '
                           'resolution value.')
        form.addParam('resolution', params.FloatParam, default=17,
                      label='Resolution (A)',
                      condition="useResolution",
                      help='Minimum value for resolution in Angstroms. '
                           'If the evaluated CTF has not reached that minimum, '
                           'it will be discarded.')

        form.addSection(label='Xmipp criteria')
        form.addParam('useCritXmipp', params.BooleanParam, default=False,
                      label='Use Xmipp criteria for selection',
                      help='Use this button to decide if carrying out the '
                           'selection taking into account the Xmipp parameters.\n'
                           'Only available when Xmipp CTF estimation was used '
                           'for the _Input CTF_ or for the _Secondary CTF_.')

        form.addParam('critFirstZero', params.FloatParam, default=5,
                      condition="useCritXmipp", label='Minimum 1st zero',
                      help='Minimun value of CritFirstZero')
        line = form.addLine('First zero astigmatism', condition="useCritXmipp",
                            help='Minimum and maximum values for '
                                 'CritFirstZeroRatio')
        line.addParam('minCritFirstZeroRatio', params.FloatParam, default=0.9,
                      label='Min')
        line.addParam('maxCritFirstZeroRatio', params.FloatParam, default=1.1,
                      label='Max')
        form.addParam('critCorr', params.FloatParam, default=0.05,
                      condition="useCritXmipp", label='Correlation 1st-3rd zero',
                      help='Minimum value of correlation between 1st and 3rd zeros')
        form.addParam('critCtfMargin', params.FloatParam, default=1.5,
                      condition="useCritXmipp", label='CTF Margin',
                      help='Minimum value of CritCtfMargin')
        form.addParam('critIceness', params.FloatParam, default=1,
                      condition="useCritXmipp", label='CritIceness',
                      help='Minimum value of the iceness.')
        line = form.addLine('Non astigmatic validity',
                            condition="useCritXmipp",
                            help='Minimum and maximum values for '
                                 'CritNonAstigmaticValidity')
        line.addParam('minCritNonAstigmaticValidity', params.FloatParam,
                      default=0.3, label='Min')
        line.addParam('maxCritNonAstigmaticValidity', params.FloatParam,
                      default=9, label='Max')

        form.addSection(label='Consensus')
        form.addParam('calculateConsensus', params.BooleanParam, default=False,
                      label='Calculate Consensus Resolution',
                      help='Option for calculating consensus resolution. '
                           'The algorithm assumes that two CTF are '
                           'consistent if the phase (wave aberration function) '
                           'of the two CTFs are closer than 90 degrees.\n'
                           'The reported consensusResolution is the resolution '
                           'at which the two CTF phases differ in 90 degrees.')
        form.addParam('inputCTF2', params.PointerParam,
                      pointerClass='SetOfCTF', condition="calculateConsensus",
                      label="Secondary CTF",
                      help='CTF to be compared with reference CTF')
        form.addParam('minConsResol', params.FloatParam,
                      condition="calculateConsensus", default=15.0,
                      label='Minimum consensus resolution (A).',
                      help="Minimum value for the consensus resolution in "
                           "Angstroms.\nIf there are noticeable discrepancies "
                           "between the two estimations below this resolution, "
                           "it will be discarded.")
        form.addParam('averageDefocus', params.BooleanParam,
                      condition="calculateConsensus", default=False,
                      label='Average equivalent metadata?',
                      help='If *Yes*, making an average of those metadata present '
                           'in both CTF estimations (defocus, astigmatism angle...)\n '
                           'If *No*, the primary estimation metadata will persist.')
        form.addParam('includeSecondary', params.BooleanParam,
                      condition="calculateConsensus", default=False,
                      label='Include all secondary metadata?',
                      help='If *Yes*, all metadata in the *Secondary CTF* will '
                           'be included in the resulting CTF.\n '
                           'If *No*, only the primary metadata (plus consensus '
                           'scores) will be in the resulting CTF.')


# --------------------------- INSERT steps functions -------------------------
    def _insertAllSteps(self):
        self.initializeParams()
        if self.calculateConsensus:
            self.ctfFn2 = self.inputCTF2.get().getFileName()

        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=[], wait=True, needsGPU=False)

    def createOutputStep(self):
        self._closeOutputSet()

    def initializeParams(self):
        self.finished = False
        self.isStreamClosed = False
        # Important to have both:
        self.insertedIds = []   # Contains images that have been inserted in a Step (checkNewInput).
        # Contains images that have been processed in a Step (checkNewOutput).
        self.acceptedIds = {}
        self.discardedIds = {}
        self.initializeRejDict()
        self.setSecondaryAttributes()
        self.ctfFn1 = self.inputCTF.get().getFileName()

    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all ctfs
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

    def _insertNewCtfsSteps(self, newIds):
        deps = []
        stepId = self._insertFunctionStep(self.selectCtfStep, newIds,  needsGPU=False,
                                              prerequisites=[])
        deps.append(stepId)

        for ctfId in newIds:
            self.insertedIds.append(ctfId)

        return deps

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        if self.calculateConsensus:
            # Check if there are new ctf to process from the input set
            self.lastCheck = getattr(self, 'lastCheck', datetime.now())
            mTime = max(datetime.fromtimestamp(os.path.getmtime(self.ctfFn1)),
                        datetime.fromtimestamp(os.path.getmtime(self.ctfFn2)))
            self.debug('Last check: %s, modification: %s'
                       % (pwutils.prettyTime(self.lastCheck),
                          pwutils.prettyTime(mTime)))
            # If the input movies.sqlite have not changed since our last check,
            # it does not make sense to check for new input data
            if self.lastCheck > mTime and self.insertedIds:  # If this is empty it is dut to a static "continue" action or it is the first round
                return None

            ctfsSet1 = self._loadInputCtfSet(self.ctfFn1)
            ctfsSet2 = self._loadInputCtfSet(self.ctfFn2)

            ctfSet1Ids = ctfsSet1.getIdSet()
            ctfSet2Ids = ctfsSet2.getIdSet()

            newIds1 = [idCTF for idCTF in ctfSet1Ids if idCTF not in self.insertedIds]
            newIds2 = [idCTF for idCTF in ctfSet2Ids if idCTF not in self.insertedIds]

            newIds = list(set(newIds1).intersection(set(newIds2)))

            self.lastCheck = datetime.now()
            self.isStreamClosed = ctfsSet1.isStreamClosed() and \
                                  ctfsSet2.isStreamClosed()
            ctfsSet1.close()
            ctfsSet2.close()
        else:
            self.lastCheck = getattr(self, 'lastCheck', datetime.now())
            mTime = datetime.fromtimestamp(os.path.getmtime(self.ctfFn1))
            self.debug('Last check: %s, modification: %s'
                       % (pwutils.prettyTime(self.lastCheck),
                          pwutils.prettyTime(mTime)))
            # If the input ctfs.sqlite have not changed since our last check,
            # it does not make sense to check for new input data
            if self.lastCheck > mTime and self.insertedIds:
                return None

            # Open input ctfs.sqlite and close it as soon as possible
            ctfSet = self._loadInputCtfSet(self.ctfFn1)
            ctfSetIds = ctfSet.getIdSet()
            newIds = [idCTF for idCTF in ctfSetIds if idCTF not in self.insertedIds]

            self.lastCheck = datetime.now()
            self.isStreamClosed = ctfSet.isStreamClosed()
            ctfSet.close()

        outputStep = self._getFirstJoinStep()

        if self.isContinued() and not self.insertedIds:  # For "Continue" action and the first round
            doneIds, _, _, _ = self._getAllDoneIds()
            skipIds = list(set(newIds).intersection(set(doneIds)))
            newIds = list(set(newIds).difference(set(doneIds)))
            self.info("Skipping CTFs with ID: %s, seems to be done" % skipIds)
            self.insertedIds = doneIds  # During the first round of "Continue" action it has to be filled

        if newIds:
            fDeps = self._insertNewCtfsSteps(newIds)
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()


    def _checkNewOutput(self):
        """ Check for already selected CTF and update the output set. """
        _, _, doneListAccepted, doneListDiscarded = self._getAllDoneIds()
        # Check for newly done items
        acceptedIds = list(self.acceptedIds)
        discardedIds = list(self.discardedIds)
        newDoneAccepted = [ctfId for ctfId in acceptedIds
                           if ctfId not in doneListAccepted]
        newDoneDiscarded = [ctfId for ctfId in discardedIds
                            if ctfId not in doneListDiscarded]

        firstTimeAccepted = len(doneListAccepted) == 0
        firstTimeDiscarded = len(doneListDiscarded) == 0
        allDone = len(doneListAccepted) + len(doneListDiscarded) +\
                  len(newDoneAccepted) + len(newDoneDiscarded)
        # We have finished when there is not more input ctf (stream closed)
        # and the number of processed ctf is equal to the number of inputs
        if self.calculateConsensus:
            inputCtfSet = self._loadInputCtfSet(self.ctfFn1)
            inputCtfSet2 = self._loadInputCtfSet(self.ctfFn2)

            ctfSet1Ids = inputCtfSet.getIdSet()
            ctfSet2Ids = inputCtfSet2.getIdSet()

            newIds = list(set(ctfSet1Ids).intersection(set(ctfSet2Ids)))

        else:
            inputCtfSet = self._loadInputCtfSet(self.ctfFn1)
            ctfSetIds = inputCtfSet.getIdSet()
            newIds = list(set(ctfSetIds))

        self.finished = (self.isStreamClosed and allDone == len(newIds))

        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        if not self.finished and (not newDoneDiscarded and not newDoneAccepted):
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        def readOrCreateOutputs(doneList, newDone, label=''):
            if len(doneList) > 0 or len(newDone) > 0:
                cSet = self._loadOutputSet(SetOfCTF, 'ctfs'+label+'.sqlite')
                mSet = self._loadOutputSet(SetOfMicrographs,
                                             'micrographs'+label+'.sqlite')
                label = ACCEPTED if label == '' else DISCARDED
                self.fillOutput(cSet, mSet, newDone, label)

                return cSet, mSet
            return None, None

        ctfSet, micSet = readOrCreateOutputs(doneListAccepted, newDoneAccepted)
        ctfSetDiscarded, micSetDiscarded = readOrCreateOutputs(doneListDiscarded,
                                                               newDoneDiscarded,
                                                               DISCARDED)

        def updateRelationsAndClose(cSet, mSet, first, label=''):

            if os.path.exists(self._getPath('ctfs'+label+'.sqlite')):

                micsAttrName = OUTPUT_MICS+label
                self._updateOutputSet(micsAttrName, mSet, streamMode)
                # Set micrograph as pointer to protocol to prevent pointer end up as another attribute (String, Boolean,...)
                # that happens somewhere while scheduling.
                cSet.setMicrographs(Pointer(self, extended=micsAttrName))

                self._updateOutputSet(OUTPUT_CTF+label, cSet, streamMode)

                if first:
                    self._defineTransformRelation(self.inputCTF.get().getMicrographs(), mSet)
                    self._defineTransformRelation(self.inputCTF, cSet)
                    self._defineCtfRelation(mSet, cSet)

                mSet.close()
                cSet.close()

        updateRelationsAndClose(ctfSet, micSet, firstTimeAccepted)
        updateRelationsAndClose(ctfSetDiscarded, micSetDiscarded,
                                firstTimeDiscarded, DISCARDED)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

        self._store()  # Update the summary dictionary


    def fillOutput(self, ctfSet, micSet, newDone, label):
        if newDone:
            inputCtfSet = self._loadInputCtfSet(self.ctfFn1)
            if self.calculateConsensus:
                inputCtfSet2 = self._loadInputCtfSet(self.ctfFn2)
            for ctfId in newDone:
                ctf = inputCtfSet[ctfId].clone()
                mic = ctf.getMicrograph().clone()

                ctf.setEnabled(self._getEnable(ctfId, label))
                mic.setEnabled(self._getEnable(ctfId, label))

                if self.calculateConsensus:
                    ctf2 = inputCtfSet2[ctfId]
                    conRes = self._freqResol[ctfId]
                    setAttribute(ctf, '_consensus_resolution', conRes)
                    setAttribute(ctf, '_ctf2_defocus_diff',
                                 max(abs(ctf.getDefocusU()-ctf2.getDefocusU()),
                                     abs(ctf.getDefocusV()-ctf2.getDefocusV())))
                    setAttribute(ctf, '_ctf2_defocusAngle_diff',
                                 anglesDifference(ctf.getDefocusAngle(),
                                                  ctf2.getDefocusAngle()))
                    if ctf.hasPhaseShift() and ctf2.hasPhaseShift():
                        setAttribute(ctf, '_ctf2_phaseShift_diff',
                                     anglesDifference(ctf.getPhaseShift(),
                                                      ctf2.getPhaseShift()))

                    setAttribute(ctf, '_ctf2_resolution', ctf2.getResolution())
                    setAttribute(ctf, '_ctf2_fitQuality', ctf2.getFitQuality())
                    if ctf2.hasAttribute('_xmipp_ctfmodel_quadrant'):
                        # To check CTF in Xmipp _quadrant is the best
                        copyAttribute(ctf2, ctf, '_xmipp_ctfmodel_quadrant')
                    else:
                        setAttribute(ctf, '_ctf2_psdFile', ctf2.getPsdFile())

                    if self.averageDefocus:
                        newDefocusU = 0.5*(ctf.getDefocusU() + ctf2.getDefocusU())
                        newDefocusV = 0.5*(ctf.getDefocusV() + ctf2.getDefocusV())
                        newDefocusAngle = averageAngles(ctf.getDefocusAngle(),
                                                        ctf2.getDefocusAngle())
                        ctf.setStandardDefocus(newDefocusU, newDefocusV,
                                               newDefocusAngle)
                        if ctf.hasPhaseShift() and ctf2.hasPhaseShift():
                            newPhaseShift = averageAngles(ctf.getPhaseShift(),
                                                          ctf2.getPhaseShift())
                            ctf.setPhaseShift(newPhaseShift)
                    else:
                        setAttribute(ctf, '_ctf2_defocusRatio', ctf2.getDefocusRatio())
                        setAttribute(ctf, '_ctf2_astigmatism',
                                     abs(ctf2.getDefocusU() - ctf2.getDefocusV()))

                    if self.includeSecondary:
                        for attr in self.secondaryAttributes:
                            copyAttribute(ctf2, ctf, attr)

                # main _astigmatism always but after consensus if so
                setAttribute(ctf, '_astigmatism',
                             abs(ctf.getDefocusU() - ctf.getDefocusV()))

                # percentage _astigmatism always but after consensus if so
                astigmatismPer = abs(ctf.getDefocusU() - ctf.getDefocusV())/(0.5 * (ctf.getDefocusU() + ctf.getDefocusV()))
                setAttribute(ctf, '_astigmatismPercentage', astigmatismPer)

                ctfSet.append(ctf)
                micSet.append(mic)

            inputCtfSet.close()
            if self.calculateConsensus:
                inputCtfSet2.close()

    def setSecondaryAttributes(self):
        if self.calculateConsensus and self.includeSecondary:
            item = self.inputCTF.get().getFirstItem()
            ctf1Attr = set(item.getObjDict().keys())

            item = self.inputCTF2.get().getFirstItem()
            ctf2Attr = set(item.getObjDict().keys())
            self.secondaryAttributes = ctf2Attr - ctf1Attr
        else:
            self.secondaryAttributes = set()


    def _loadOutputSet(self, SetClass, baseName):
        """
        Load the output set if it exists or create a new one.
        """
        setFile = self._getPath(baseName)

        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            if (outputSet.__len__() == 0):
                pwutils.path.cleanPath(setFile)

        if os.path.exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

        micSet = self.inputCTF.get().getMicrographs()

        if isinstance(outputSet, SetOfMicrographs):
            outputSet.copyInfo(micSet)
        elif isinstance(outputSet, SetOfCTF):
            outputSet.setMicrographs(micSet)
        return outputSet

    def _ctfToMd(self, ctf, ctfMd):
        """ Write the proper metadata for Xmipp from a given CTF """
        ctfMd.clear()
        ctfRow = Row()
        xmipp3.convert.ctfModelToRow(ctf, ctfRow)
        xmipp3.convert.micrographToRow(ctf.getMicrograph(), ctfRow,
                                       alignType=xmipp3.convert.ALIGN_NONE)
        ctfRow.addToMd(ctfMd)

    def initializeRejDict(self):
        self.discDict = {'defocus': 0,
                         'astigmatism': 0,
                         'astigmatismPer': 0,
                         'singleResolution': 0,
                         '_xmipp_ctfCritFirstZero': 0,
                         '_xmipp_ctfCritfirstZeroRatio': 0,
                         '_xmipp_ctfCritCorr13': 0,
                         '_xmipp_ctfCritIceness': 0,
                         '_xmipp_ctfCritCtfMargin': 0,
                         '_xmipp_ctfCritNonAstigmaticValidty': 0,
                         'consensusResolution': 0
                         }
        for k in self.discDict:
            setattr(self, "rejBy"+k, Integer(0))
        self._store()

    def selectCtfStep(self, ctfIds):
        # Depending on the flags selected by the user, we set the values of the params to compare with
        minDef, maxDef = self._getDefociValues()
        maxAstig = self._getMaxAstisgmatism()
        maxAstigPer = self._getMaxAstigmatismPer()
        minResol = self._getMinResol()

        inputCtfSet = self._loadInputCtfSet(self.ctfFn1)

        if self.calculateConsensus:
            inputCtfSet2 = self._loadInputCtfSet(self.ctfFn2)

        def compareValue(ctf, label, comp, crit):
            """ Returns True if the ctf.label NOT complain the crit by comp
            """
            if hasattr(ctf, label):
                if comp == 'lt':
                    discard = getattr(ctf, label).get() < crit
                elif comp == 'bt':
                    discard = getattr(ctf, label).get() > crit
                else:
                    raise Exception("'comp' must be either 'lt' or 'bt'.")
            else:
                print("%s not found. Skipping evaluation on that." % label)
                return False
            if discard:
                self.discDict[label] += 1
            return discard

        for ctfId in ctfIds:
            ctf = inputCtfSet.getItem("id", ctfId).clone()

            defocusU = ctf.getDefocusU()
            defocusV = ctf.getDefocusV()
            astigm = abs(defocusU - defocusV)
            astigmPer = abs(defocusU - defocusV)/((defocusU+defocusV)/2)
            resol = self._getCtfResol(ctf)

            defRangeCrit = (defocusU < minDef or defocusU > maxDef or
                            defocusV < minDef or defocusV > maxDef)
            if defRangeCrit:
                self.discDict['defocus'] += 1

            astigCrit = astigm > maxAstig
            if astigCrit:
                self.discDict['astigmatism'] += 1

            astigPer = (astigmPer > maxAstigPer)
            if astigPer:
                self.discDict['astigmatismPer'] += 1

            singleResolCrit = resol > minResol
            if singleResolCrit:
                self.discDict['singleResolution'] += 1

            firstCondition = defRangeCrit or astigCrit or singleResolCrit or astigPer

            consResolCrit = False

            if self.calculateConsensus:
                ctf2 = inputCtfSet2.getItem("id", ctfId)
                freqResolConsensus = self.calculateConsensusResolution(ctfId, ctf, ctf2)
                consResolCrit = self.minConsResol < freqResolConsensus
                if consResolCrit:
                    self.discDict['consensusResolution'] += 1

                self._freqResol[ctfId] = freqResolConsensus

            secondCondition = False
            if self.useCritXmipp:
                firstZero = self._getCritFirstZero()
                minFirstZero, maxFirstZero = self._getCritFirstZeroRatio()
                corr = self._getCritCorr()
                iceness = self._getIceness()
                ctfMargin = self._getCritCtfMargin()
                minNonAstigmatic, maxNonAstigmatic = \
                    self._getCritNonAstigmaticValidity()

                if self.xmippCTF == INPUT1:
                    ctfX = ctf
                else:
                    ctfX = ctf2

                secondCondition = (
                    compareValue(ctfX, '_xmipp_ctfCritFirstZero', 'lt', firstZero) or
                    compareValue(ctfX, '_xmipp_ctfCritfirstZeroRatio', 'lt', minFirstZero) or
                    compareValue(ctfX, '_xmipp_ctfCritfirstZeroRatio', 'bt', maxFirstZero) or
                    compareValue(ctfX, '_xmipp_ctfCritCorr13', 'lt', corr) or
                    compareValue(ctfX, '_xmipp_ctfCritIceness', 'bt', iceness) or
                    compareValue(ctfX, '_xmipp_ctfCritCtfMargin', 'lt', ctfMargin) or
                    compareValue(ctfX, '_xmipp_ctfCritNonAstigmaticValidty', 'lt', minNonAstigmatic) or
                    compareValue(ctfX, '_xmipp_ctfCritNonAstigmaticValidty', 'bt', maxNonAstigmatic))

            """ Write to a text file the items that have been done. """
            if firstCondition or consResolCrit or secondCondition:
                self.discardedIds[ctfId] = 'F'
            else:
                if ctf.isEnabled():
                    self.acceptedIds[ctfId] = 'T'
                else:
                    self.acceptedIds[ctfId] = 'F'

            for k, v in self.discDict.items():
                setattr(self, "rejBy"+k, Integer(v))

    def calculateConsensusResolution(self, ctfId, ctf1, ctf2):
        md1 = emlib.MetaData()
        md2 = emlib.MetaData()
        try:
            self._ctfToMd(ctf1, md1)
            self._ctfToMd(ctf2, md2)
            freqResol = emlib.errorMaxFreqCTFs2D(md1, md2)
        except TypeError as exc:
            print("Error reading ctf for id:%s. %s" % (ctfId, exc))
            freqResol = 9999

        return freqResol

    def _getAllDoneIds(self):
        doneIds = []
        acceptedIds = []
        discardedIds = []
        sizeOutput = 0

        if hasattr(self, OUTPUT_CTF):
            sizeOutput += self.outputCTF.getSize()
            acceptedIds.extend(list(self.outputCTF.getIdSet()))
            doneIds.extend(acceptedIds)

        if hasattr(self, OUTPUT_CTF_DISCARDED):
            sizeOutput += self.outputCTFDiscarded.getSize()
            discardedIds.extend(list(self.outputCTFDiscarded.getIdSet()))
            doneIds.extend(discardedIds)

        return doneIds, sizeOutput, acceptedIds, discardedIds

    def _citations(self):
        return ['Marabini2014a']

    def _summary(self):

        if not (hasattr(self, "outputCTF") or hasattr(self, "outputCTFDiscarded")):
            return ['No CTF processed, yet.']

        acceptedSize = (self.outputCTF.getSize()
                        if hasattr(self, "outputCTF") else 0)

        discardedSize = (self.outputCTFDiscarded.getSize()
                         if hasattr(self, "outputCTFDiscarded") else 0)

        message = ["%d/%d CTF processed (%d accepted and %d discarded)."
                   % (acceptedSize+discardedSize,
                      self.inputCTF.get().getSize(),
                      acceptedSize, discardedSize)]

        def addDiscardedStr(label):
            obj = getattr(self, "rejBy%s" % label, Integer(0))
            number = obj.get()
            return "" if number == 0 else "  (%d discarded)" % number
        if any([self.useDefocus, self.useAstigmatism, self.useResolution, self.useAstigmatismPercentage]):
            message.append("*General Criteria*:")
        if self.useDefocus:
            message.append(" - _Defocus_. Range: %.0f - %.0f %s"
                           % (self.minDefocus, self.maxDefocus,
                              addDiscardedStr('defocus')))

        if self.useAstigmatism:
            message.append(" - _Astigmatism_. Threshold: %.0f %s"
                           % (self.astigmatism,
                              addDiscardedStr('astigmatism')))

        if self.useAstigmatismPercentage:
            message.append(" - _AstigmatismPer_. Threshold: %.3f %s"
                           % (self.astigmatismPer,
                              addDiscardedStr('astigmatismPer')))

        if self.useResolution:
            message.append(" - _Resolution_. Threshold: %.2f %s"
                           % (self.resolution,
                              addDiscardedStr('singleResolution')))

        if self.useCritXmipp:
            message.append("*Xmipp criteria*:")
            message.append(" - _First zero_. Threshold: %.2f %s"
                           % (self.critFirstZero,
                              addDiscardedStr('_xmipp_ctfCritFirstZero')))
            message.append(" - _First zero astigmatism_. Range: %.2f - %.2f %s"
                           % (self.minCritFirstZeroRatio,
                              self.maxCritFirstZeroRatio,
                              addDiscardedStr('_xmipp_ctfCritfirstZeroRatio')))
            message.append(" - _Correlation Experimental-Estimated_. Threshold: "
                           "%.2f %s" % (self.critCorr,
                                        addDiscardedStr('_xmipp_ctfCritCorr13')))
            message.append(" - _CTF margin_. Threshold: %.2f %s"
                           % (self.critCtfMargin,
                              addDiscardedStr('_xmipp_ctfCritCtfMargin')))
            message.append(" - _Iceness_. Threshold: %.2f %s"
                           % (self.critIceness,
                              addDiscardedStr('_xmipp_ctfCritIceness')))
            message.append(" - _Non Astigmatic validation range_: %.2f - %.2f %s"
                           % (self.minCritNonAstigmaticValidity,
                              self.maxCritNonAstigmaticValidity,
                              addDiscardedStr('_xmipp_ctfCritNonAstigmaticValidty')))

        if self.calculateConsensus:
            def getProtocolInfo(inCtf):
                protocol = self.getMapper().getParent(inCtf)
                runName = protocol.getRunName()
                classLabel = protocol.getClassLabel()
                if runName == classLabel:
                    infoStr = runName
                else:
                    infoStr = "%s (%s)" % (runName, classLabel)

                return infoStr

            message.append("*CTF consensus*:")
            message.append(" - _Consensus resolution. Threshold_: %.2f %s"
                           % (self.minConsResol,
                              addDiscardedStr('consensusResolution')))
            message.append("   > _Primary CTF_: %s"
                           % getProtocolInfo(self.inputCTF.get()))
            message.append("   > _Reference CTF_: %s"
                           % getProtocolInfo(self.inputCTF2.get()))

        return message

    def _validate(self):
        """ The function of this hook is to add some validation before the
        protocol is launched to be executed. It should return a list of
        errors. If the list is empty the protocol can be executed.
        """
        # same micrographs in both CTF??
        errors = []
        if self.useCritXmipp.get() and not self.calculateConsensus.get():
            if self.usingXmipp(self.inputCTF.get().getFirstItem()):
                self.xmippCTF = INPUT1
            else:
                errors.append("The primary CTF input ( _Input CTF_ ) must be "
                              "estimated using the _Xmipp - CTF estimation_ "
                              "protocol.")
        if self.useCritXmipp.get() and self.calculateConsensus.get():
            if self.usingXmipp(self.inputCTF.get().getFirstItem()):
                self.xmippCTF = INPUT1
            elif self.usingXmipp(self.inputCTF2.get().getFirstItem()):
                self.xmippCTF = INPUT2
            else:
                errors.append("One of the CTF inputs ( _Input CTF_ or "
                              "_Secundary CTF_) must be estimated using the "
                              "_Xmipp - CTF estimation_ protocol.")
        return errors

    def usingXmipp(self, ctf):
        return ctf.hasAttribute('_xmipp_ctfCritFirstZero')

    def _getCtfResol(self, ctf):
        resolution = ctf.getResolution()
        if resolution is not None:
            return resolution
        else:
            return 0

    def _getEnable(self, ctfId, label):
        if label == ACCEPTED:
            enableValue = self.acceptedIds[ctfId]
        else:
            enableValue = self.discardedIds[ctfId]

        if enableValue == 'T':
            return True
        else:
            return False

    def _loadInputCtfSet(self, ctfFn):
        self.debug("Loading input db: %s" % ctfFn)
        ctfSet = SetOfCTF(filename=ctfFn)
        ctfSet.loadAllProperties()
        return ctfSet

    def _getDefociValues(self):
        if not self.useDefocus:
            return 0, 1000000
        else:
            return self.minDefocus.get(), self.maxDefocus.get()

    def _getMaxAstigmatismPer(self):
        if not self.useAstigmatismPercentage:
            return 0.1
        else:
            return self.astigmatismPer.get()

    def _getMaxAstisgmatism(self):
        if not self.useAstigmatism:
            return 10000000
        else:
            return self.astigmatism.get()

    def _getMinResol(self):
        if not self.useResolution:
            return 1000000
        else:
            return self.resolution.get()

    def _getIceness(self):
        return self.critIceness.get()

    def _getCritFirstZero(self):
        return self.critFirstZero.get()

    def _getCritFirstZeroRatio(self):
        return self.minCritFirstZeroRatio.get(),\
               self.maxCritFirstZeroRatio.get()

    def _getCritCorr(self):
        return self.critCorr.get()

    def _getCritCtfMargin(self):
        return self.critCtfMargin.get()

    def _getCritNonAstigmaticValidity(self):
        return (self.minCritNonAstigmaticValidity.get(),
                self.maxCritNonAstigmaticValidity.get())


def averageAngles(angle1, angle2):
    c1 = rect(1, radians(angle1*2))
    c2 = rect(1, radians(angle2*2))
    return degrees(phase((c1 + c2)*0.5))/2


def anglesDifference(angle1, angle2):
    if (angle1 > angle2) == (abs(angle2 - angle1) > 90):
        aux = angle1
        angle1 = angle2
        angle2 = aux
    return (angle1 - angle2) % 180


def setAttribute(obj, label, value):
    if value is None:
        return
    setattr(obj, label, getScipionObj(value))


def copyAttribute(src, dst, label, default=None):
    setAttribute(dst, label, getattr(src, label, default))
