# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
# *             Daniel Marchán Torres (da.marchan@cnb.csic.es)  -- streaming version
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************
"""
Consensus picking protocol
"""

import os
from datetime import datetime
from cmath import rect, phase
from math import radians, degrees

from pyworkflow import VERSION_2_0
from pwem.objects import SetOfMovies, SetOfMicrographs
from pyworkflow.object import Set, Integer, Pointer
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAlignMovies
from pwem.emlib.metadata import Row
from pyworkflow.protocol.constants import (STATUS_NEW)

from pwem import emlib
import xmipp3
from xmipp3.convert import setXmippAttribute, getScipionObj
from pwem.constants import (NO_INDEX, ALIGN_NONE, ALIGN_2D, ALIGN_3D,
                            ALIGN_PROJ, ALIGNMENTS)

ACCEPTED = 'Accepted'
DISCARDED = 'Discarded'


class XmippProtConsensusMovieAlignment(ProtAlignMovies):
    """
    Protocol to estimate the agreement between different movie alignment
    algorithms in the Global Shifts ...).
    """

    _label = 'movie alignment consensus'
    outputName = 'consensusAlignments'
    #FN_PREFIX = 'consensusAlignments_'

    def __init__(self, **args):
        ProtAlignMovies.__init__(self, **args)
        self._freqResol = {}
        self.stepsExecutionMode = params.STEPS_SERIAL
        #self.stepsExecutionMode = STEPS_PARALLEL


    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputMovies1', params.PointerParam, pointerClass='SetOfMovies',
                      label="Aligned Movies", important=True,
                      help='Select the aligned movies to evaluate (this first set will give the global shifts)')

        #form.addSection(label='Xmipp criteria')
        #form.addParam('useCritXmipp', params.BooleanParam, default=False,
         #             label='Use Xmipp criteria for selection',
          #            help='Use this button to decide if carrying out the '
           #                'selection taking into account the Xmipp parameters.\n'
            #               'Only available when Xmipp Flex Align algorithm was used '
             #              'for the _Aligned Input Movies_ or for the _Secondary Aligned Movies_.')

        form.addSection(label='Consensus')
        form.addParam('calculateConsensus', params.BooleanParam, default=False,
                      label='Calculate Consensus on Global Shifts ',
                      help='Option for calculating consensus on global shifts. '
                           'The algorithm assumes that movies shifts are '
                           'consistent if the the shift trajectory'
                           'of the two alignments are correlated.')

        form.addParam('inputMovies2', params.PointerParam,
                      pointerClass='SetOfMovies', condition="calculateConsensus",
                      label="Secondary Movies",
                      help='Shift to be compared with reference alignment')

        form.addParam('minConsCorrelation', params.FloatParam,
                      condition="calculateConsensus", default=0.8,
                      label='Minimum consensus shifts correlation.',
                      help="Minimum value for the consensus correlations between shifts trajectories."
                           "\nIf there are noticeable discrepancies "
                           "between the two estimations below this correlation, "
                           "it will be discarded.")

        form.addParam('averageParams', params.BooleanParam,
                      condition="calculateConsensus", default=False,
                      label='Average equivalent metadata?',
                      help='If *Yes*, making an average of those metadata present '
                           'in both alignments (shift_x, shift_y...)\n '
                           'If *No*, the primary estimation metadata will persist.')

        form.addParam('includeSecondary', params.BooleanParam,
                      condition="calculateConsensus", default=False,
                      label='Include all secondary metadata?',
                      help='If *Yes*, all metadata in the *Secondary Movie ALignment* will '
                           'be included in the resulting movies.\n '
                           'If *No*, only the primary metadata (plus consensus '
                           'scores) will be in the resulting movies.')

        form.addParallelSection(threads=0, mpi=0)

# --------------------------- INSERT steps functions -------------------------
    def _insertAllSteps(self):
        self.finished = False
        self.insertedDict = {}
        self.processedDict = []
        self.outputDict = []
        self.allMovies1 = []
        self.allMovies2 = []
        self.initializeRejDict()
        self.setSecondaryAttributes()

        self.movieFn1 = self.inputMovies1.get().getFileName()
        self.numberOfFrames1 = self.inputMovies1.get().getFirstItem().getNumberOfFrames()
        print(self.numberOfFrames1)


        if self.calculateConsensus.get():
            self.movieFn2 = self.inputMovies2.get().getFileName()
            self.numberOfFrames2 = self.inputMovies2.get().getFirstItem().getNumberOfFrames()
            print(self.numberOfFrames2)
            movieSteps = self._checkNewInput()

        else:
            movieSteps = self._insertNewSelectionSteps(self.insertedDict, self.inputMovies1.get())

        self._insertFunctionStep('createOutputStep',
                                 prerequisites=movieSteps, wait=True)

    def createOutputStep(self):
        pass

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

    def _insertNewMovieSteps(self, setOfMovies1, setOfMovies2, insDict):
        deps = []
        movies1Dict = setOfMovies1.getObjDict(includeBasic=True)
        movies2Dict = setOfMovies2.getObjDict(includeBasic=True)
        discrepId = self._insertFunctionStep("movieDiscrepancyStep",
                                             movies1Dict, movies2Dict,
                                             prerequisites=[])
        deps.append(discrepId)
        if len(setOfMovies1) > len(setOfMovies2):
            setOfMovies = setOfMovies2
        else:
            setOfMovies = setOfMovies1
        for movie in setOfMovies:
            movieId = movie.getObjId()
            if movieId not in insDict:
                stepId = self._insertFunctionStep('selectMovieStep', movieId,
                                                  prerequisites=[discrepId])
                deps.append(stepId)
                insDict[movieId] = stepId

        return deps

    def _insertNewSelectionSteps(self, insertedDict, inputMovies):
        deps = []
        # For each ctf insert the step to process it
        for movie in inputMovies:
            movieId = movie.getObjId()
            if movieId not in insertedDict:
                stepId = self._insertFunctionStep('selectMovieStep', movieId,
                                                  prerequisites=[])
                deps.append(stepId)
                insertedDict[movieId] = stepId
        return deps

    def _stepsCheck(self):
        self._checkNewInput()
        #self._checkNewOutput()

    def _checkNewInput(self):
        if self.calculateConsensus.get():
            # Check if there are new ctf to process from the input set
            self.lastCheck = getattr(self, 'lastCheck', datetime.now())
            mTime = max(datetime.fromtimestamp(os.path.getmtime(self.movieFn1)),
                        datetime.fromtimestamp(os.path.getmtime(self.movieFn2)))
            # If the input movies.sqlite have not changed since our last check,
            # it does not make sense to check for new input data
            if self.lastCheck > mTime and hasattr(self, 'SetOfMovies1'):
                return None
            movieSet1 = self._loadInputMovieSet(self.movieFn1)
            movieSet2 = self._loadInputMovieSet(self.movieFn2)
            if len(self.allMovies1) > 0:
                newMovies1 = [movie.clone() for movie in
                              movieSet1.iterItems(orderBy='creation',
                                                  where='creation>"' + str(
                                                  self.checkMovies1) + '"')]
            else:
                newMovies1 = [movie.clone() for movie in movieSet1]
            self.allMovies1 = self.allMovies1 + newMovies1
            if len(newMovies1) > 0:
                for movie in movieSet1.iterItems(orderBy='creation',
                                              direction='DESC'):
                    self.checkMovies1 = movie.getObjCreation()
                    break
            if len(self.allMovies2) > 0:
                newMovies2 = [movie.clone() for movie in
                           movieSet2.iterItems(orderBy='creation',
                                               where='creation>"' + str(
                                                  self.checkMovies2) + '"')]
            else:
                newMovies2 = [movie.clone() for movie in movieSet2]
            self.allMovies2 = self.allMovies2 + newMovies2
            if len(newMovies2) > 0:
                for movie in movieSet2.iterItems(orderBy='creation',
                                                 direction='DESC'):
                    self.checkMovies2 = movie.getObjCreation()
                    break

            self.lastCheck = datetime.now()
            self.isStreamClosed = movieSet1.isStreamClosed() and \
                                  movieSet2.isStreamClosed()

            movieSet1.close()
            movieSet2.close()

            outputStep = self._getFirstJoinStep()

            if len(set(self.allMovies1)) > len(set(self.processedDict)) and \
               len(set(self.allMovies2)) > len(set(self.processedDict)):
                fDeps = self._insertNewMovieSteps(movieSet1, movieSet2, self.insertedDict)
                if outputStep is not None:
                    outputStep.addPrerequisites(*fDeps)
                self.updateSteps()
        else:
            now = datetime.now()
            self.lastCheck = getattr(self, 'lastCheck', now)
            mTime = datetime.fromtimestamp(os.path.getmtime(self.movieFn1))
            self.debug('Last check: %s, modification: %s'
                       % (pwutils.prettyTime(self.lastCheck),
                          pwutils.prettyTime(mTime)))

            # Open input movies.sqlite and close it as soon as possible
            movieSet = self._loadInputMovieSet(self.movieFn1)
            self.isStreamClosed = movieSet.isStreamClosed()
            self.allMovies1 = [m.clone() for m in movieSet]
            movieSet.close()

            # If the input movies.sqlite have not changed since our last check,
            # it does not make sense to check for new input data
            if self.lastCheck > mTime and hasattr(self, 'allMovies1'):
                return None

            self.lastCheck = now
            newMovies = any(movie.getObjId() not in
                         self.insertedDict for movie in self.allMovies1)
            outputStep = self._getFirstJoinStep()

            if newMovies:
                fDeps = self._insertNewSelectionSteps(self.insertedDict,
                                                      self.allMovies1)
                if outputStep is not None:
                    outputStep.addPrerequisites(*fDeps)
                self.updateSteps()


    def _checkNewOutput(self):
        """ Check for already selected CTF and update the output set. """

        # Load previously done items (from text file)
        doneListDiscarded = self._readCertainDoneList(DISCARDED)
        doneListAccepted = self._readCertainDoneList(ACCEPTED)

        # Check for newly done items
        ctfListIdAccepted = self._readtCtfId(True)
        ctfListIdDiscarded = self._readtCtfId(False)

        newDoneAccepted = [ctfId for ctfId in ctfListIdAccepted
                           if ctfId not in doneListAccepted]
        newDoneDiscarded = [ctfId for ctfId in ctfListIdDiscarded
                            if ctfId not in doneListDiscarded]
        firstTimeAccepted = len(doneListAccepted) == 0
        firstTimeDiscarded = len(doneListDiscarded) == 0
        allDone = len(doneListAccepted) + len(doneListDiscarded) +\
                  len(newDoneAccepted) + len(newDoneDiscarded)

        # We have finished when there is not more input ctf (stream closed)
        # and the number of processed ctf is equal to the number of inputs
        if self.calculateConsensus:
            maxCtfSize = min(len(self.allCtf1), len(self.allCtf2))
        else:
            maxCtfSize = len(self.allCtf1)

        self.finished = (self.isStreamClosed and allDone == maxCtfSize)

        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN


        def readOrCreateOutputs(doneList, newDone, label=''):
            if len(doneList) > 0 or len(newDone) > 0:
                cSet = self._loadOutputSet(SetOfMovies, 'ctfs'+label+'.sqlite')
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

        if not self.finished and not newDoneDiscarded and not newDoneAccepted:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        def updateRelationsAndClose(cSet, mSet, first, label=''):

            if os.path.exists(self._getPath('ctfs'+label+'.sqlite')):

                micsAttrName = 'outputMicrographs'+label
                self._updateOutputSet(micsAttrName, mSet, streamMode)
                # Set micrograph as pointer to protocol to prevent pointee end up as another attribute (String, Booelan,...)
                # that happens somewhere while scheduling.
                cSet.setMicrographs(Pointer(self, extended=micsAttrName))

                self._updateOutputSet('outputCTF'+label, cSet, streamMode)

                if first:
                    self._defineTransformRelation(self.inputCTF.get().getMicrographs(),
                                                  mSet)
                    # self._defineTransformRelation(cSet, mSet)
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


    def fillOutput(self, movieSet, micSet, newDone, label):
        if newDone:
            inputMovieSet = self._loadInputMovieSet(self.movieFn1)
            if self.calculateConsensus.get():
                inputMovieSet2 = self._loadInputMovieSet(self.movieFn2)
            for movieId in newDone:
                movie = inputMovieSet[movieId].clone()
                mic = movie.getMicrograph().clone()

                movie.setEnabled(self._getEnable(movieId))
                mic.setEnabled(self._getEnable(movieId))

                if self.calculateConsensus.get():
                    movie2 = inputMovieSet2[movieId]
                    setAttribute(movie, '_consensus_resolution',
                                 self._freqResol[movieId])
                    setAttribute(movie, '_ctf2_defocus_diff',
                                 max(abs(movie.getDefocusU()-movie2.getDefocusU()),
                                     abs(movie.getDefocusV()-movie2.getDefocusV())))
                    setAttribute(movie, '_ctf2_defocusAngle_diff',
                                 anglesDifference(movie.getDefocusAngle(),
                                                  movie2.getDefocusAngle()))
                    if movie.hasPhaseShift() and movie2.hasPhaseShift():
                        setAttribute(movie, '_ctf2_phaseShift_diff',
                                     anglesDifference(movie.getPhaseShift(),
                                                      movie2.getPhaseShift()))

                    setAttribute(movie, '_ctf2_resolution', movie2.getResolution())
                    setAttribute(movie, '_ctf2_fitQuality', movie2.getFitQuality())

                    if movie2.hasAttribute('_xmipp_ctfmodel_quadrant'):
                        # To check CTF in Xmipp _quadrant is the best
                        copyAttribute(movie2, movie, '_xmipp_ctfmodel_quadrant')
                    else:
                        setAttribute(movie, '_ctf2_psdFile', movie2.getPsdFile())

                    if self.averageDefocus:
                        newDefocusU = 0.5*(movie.getDefocusU() + movie2.getDefocusU())
                        newDefocusV = 0.5*(movie.getDefocusV() + movie2.getDefocusV())
                        newDefocusAngle = averageAngles(movie.getDefocusAngle(),
                                                        movie2.getDefocusAngle())
                        movie.setStandardDefocus(newDefocusU, newDefocusV,
                                               newDefocusAngle)
                        if movie.hasPhaseShift() and movie2.hasPhaseShift():
                            newPhaseShift = averageAngles(movie.getPhaseShift(),
                                                          movie2.getPhaseShift())
                            movie.setPhaseShift(newPhaseShift)
                    else:
                        setAttribute(movie, '_ctf2_defocusRatio', movie2.getDefocusRatio())
                        setAttribute(movie, '_ctf2_astigmatism',
                                     abs(movie2.getDefocusU() - movie2.getDefocusV()))

                    if self.includeSecondary:
                        for attr in self.secondaryAttributes:
                            copyAttribute(movie2, movie, attr)

                # main _astigmatism always but after consensus if so
                setAttribute(movie, '_astigmatism',
                             abs(movie.getDefocusU() - movie.getDefocusV()))

                movieSet.append(movie)
                micSet.append(mic)
                self._writeCertainDoneList(movieId, label)

            inputMovieSet.close()
            if self.calculateConsensus.get():
                inputMovieSet2.close()


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
        elif isinstance(outputSet, SetOfMovies):
            outputSet.setMicrographs(micSet)
        return outputSet


    def _movieToMd(self, movie, movieMd):
        """ Write the proper metadata for Xmipp from a given CTF """
        movieMd.clear()
        movieRow = Row()

        xmipp3.convert.writeMovieMd(movie, movieMd)
        xmipp3.convert.micrographToRow(movie.getMicrograph(), movieRow,
                                       alignType=xmipp3.convert.ALIGN_NONE)

        movieRow.addToMd(movieMd)

    def movieDiscrepancyStep(self, met1Dict, met2Dict):
        # TODO must be same micrographs
        # move to a single step, each step takes 5 sec while the function
        # takes 0.03 sec
        # convert to md

        method1 = SetOfMovies()
        method1.setAttributesFromDict(met1Dict, setBasic=True, ignoreMissing=True)
        method2 = SetOfMovies()
        method2.setAttributesFromDict(met2Dict, setBasic=True, ignoreMissing=True)
        md1 = emlib.MetaData()
        md2 = emlib.MetaData()

        for movie1 in method1:  # reference CTF
            movieId = movie1.getObjId()
            if movieId in self.processedDict:
                continue
            for movie2 in method2:
                movieId2 = movie2.getObjId()
                if movieId2 != movieId:
                    continue
                self.processedDict.append(movieId)
                try:
                    self._movieToMd(movie1, md1)
                    self._movieToMd(movie2, md2)
                    self._freqResol[movieId] = emlib.errorMaxFreqCTFs2D(md1, md2)

                except TypeError as exc:
                    print("Error reading movie for id:%s. %s" % (movieId, exc))
                    self._freqResol[movieId] = 9999

    def initializeRejDict(self):
        self.discDict = {'shiftX': 0,
                         'shiftY': 0,
                         'consensusCorrelation': 0
                         }
        for k in self.discDict:
            setattr(self, "rejBy"+k, Set(0))
        self._store()


    def selectMovieStep(self, movieId):
        # Depending on the flags selected by the user, we set the values of
        # the params to compare with

        def compareValue(movie, label, comp, crit):
            """ Returns True if the ctf.label NOT complain the crit by comp
            """
            if hasattr(movie, label):
                if comp == 'lt':
                    discard = getattr(movie, label).get() < crit
                elif comp == 'bt':
                    discard = getattr(movie, label).get() > crit
                else:
                    raise Exception("'comp' must be either 'lt' or 'bt'.")
            else:
                print("%s not found. Skipping evaluation on that." % label)
                return False

            if discard:
                self.discDict[label] += 1
            return discard

        # TODO: Change this way to get the ctf.
        movie = self.inputMovies1.get()[movieId]

        # FIXME: this is a workaround to skip errors, but it should be treat at checkNewInput
        if movie is None:
            return

        alignment = movie.getAlignment()
        shiftX, shiftY = alignment.getShifts()
        print('SHIFTTSSSSSSSSS')
        print(shiftX)
        print(shiftY)

        self.discDict['shiftsX'] = shiftX
        self.discDict['shiftsY'] = shiftY
        self.discDict['consensusCorrelation'] = self.minConsCorrelation.get()

        """ Write to a text file the items that have been done. """
        fn = movie.getFileName()
        with open(fn, 'a') as f:
            f.write('%d \n' % movie.getObjId())

        self._store()


    def _readDoneList(self):
        """ Read from a file the id's of the items that have been done. """
        doneFile = self._getAllDone()
        doneList = []
        # Check what items have been previously done
        if os.path.exists(doneFile):
            with open(doneFile) as f:
                doneList += [int(line.strip()) for line in f]
        return doneList

    def _getAllDone(self):
        return self._getExtraPath('DONE_all.TXT')

    def _writeDoneList(self, partList):
        """ Write to a text file the items that have been done. """
        with open(self._getAllDone(), 'a') as f:
            for part in partList:
                f.write('%d\n' % part.getObjId())

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

        if any([self.useDefocus, self.useAstigmatism, self.useResolution]):
            message.append("*General Criteria*:")


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
            message.append(" - _Consensus resolution. Threshold_: %.0f %s"
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

        if (self.inputMovies1.get().hasAlignment() ==  ALIGN_NONE) or \
                (self.inputMovies2.get().hasAlignment() == ALIGN_NONE):
            errors.append("The inputs ( _Input Movies 1_ or _Input Movies 2_ must be aligned before")

        return errors


    # def _readDoneListDiscarded(self):
    #     """ Read from a text file the id's of the items
    #     that have been done. """
    #     DiscardedFile = self._getDiscardedDone()
    #     DiscardedList = []
    #     # Check what items have been previously done
    #     if os.path.exists(DiscardedFile):
    #         with open(DiscardedFile) as f:
    #             DiscardedList += [int(line.strip()) for line in f]
    #     return DiscardedList

    def _readCertainDoneList(self, label):
        """ Read from a text file the id's of the items
        that have been done. """
        doneFile = self._getCertainDone(label)
        doneList = []
        # Check what items have been previously done
        if os.path.exists(doneFile):
            with open(doneFile) as f:
                doneList += [int(line.strip()) for line in f]
        return doneList

    def _writeCertainDoneList(self, ctfId, label):
        """ Write to a text file the items that have been done. """
        doneFile = self._getCertainDone(label)
        with open(doneFile, 'a') as f:
            f.write('%d\n' % ctfId)

    def _getCertainDone(self, label):
        return self._getExtraPath('DONE_'+label+'.TXT')

    def _getCtfSelecFileAccepted(self):
        return self._getExtraPath('selection-ctf-accepted.txt')

    def _getCtfSelecFileDiscarded(self):
        return self._getExtraPath('selection-ctf-discarded.txt')

    def _readtCtfId(self, accepted):
        if accepted:
            fn = self._getCtfSelecFileAccepted()
        else:
            fn = self._getCtfSelecFileDiscarded()
        ctfList = []
        # Check what items have been previously done
        if os.path.exists(fn):
            with open(fn) as f:
                ctfList += [int(line.strip().split()[0]) for line in f]
        return ctfList

    def _getEnable(self, ctfId):
        fn = self._getCtfSelecFileAccepted()
        # Check what items have been previously done
        if os.path.exists(fn):
            with open(fn) as f:
                for line in f:
                    if ctfId == int(line.strip().split()[0]):
                        if line.strip().split()[1] == 'T':
                            return True
                        else:
                            return False

    def _loadInputMovieSet(self, moviesFn):
        self.debug("Loading input db: %s" % moviesFn)
        movieSet = SetOfMovies(filename=moviesFn)
        movieSet.loadAllProperties()
        return movieSet

    def _getMinCorrelation(self):
       return self.minConsCorrelation.get()


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
