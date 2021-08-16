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

import numpy as np
from pyworkflow import VERSION_2_0
from pwem.objects import SetOfMovies, SetOfMicrographs, MovieAlignment
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


        form.addParallelSection(threads=0, mpi=0)

# --------------------------- INSERT steps functions -------------------------
    def _insertAllSteps(self):
        self.finished = False
        self.insertedDict = {}
        self.processedDict = []
        self.outputDict = []
        self.allMovies1 = []
        self.allMovies2 = []
        self.movieFn1 = self.inputMovies1.get().getFileName()
        self.samplingRate = self.inputMovies1.get().getFirstItem().getSamplingRate()
        self.micsFn = '/home/dmarchan/ScipionUserData/projects/MovieAligmentConsensus/Runs/000753_XmippProtMovieCorr/'
        self.stats = {}

        if self.calculateConsensus.get():
            self.movieFn2 = self.inputMovies2.get().getFileName()
            movieSteps = self._checkNewInput()

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
                stepId = self._insertFunctionStep('alignmentCorrelationMovieStep', movieId,
                                                  prerequisites=[discrepId])
                deps.append(stepId)
                insDict[movieId] = stepId

        return deps


    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

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



    def _checkNewOutput(self):
        """ Check for already selected CTF and update the output set. """

        # Load previously done items (from text file)
        doneListDiscarded = self._readCertainDoneList(DISCARDED)
        doneListAccepted = self._readCertainDoneList(ACCEPTED)

        # Check for newly done items
        movieListIdAccepted = self._readtMovieId(True)
        movieListIdDiscarded = self._readtMovieId(False)

        newDoneAccepted = [movieId for movieId in movieListIdAccepted
                           if movieId not in doneListAccepted]
        newDoneDiscarded = [movieId for movieId in movieListIdDiscarded
                            if movieId not in doneListDiscarded]

        firstTimeAccepted = len(doneListAccepted) == 0
        firstTimeDiscarded = len(doneListDiscarded) == 0
        allDone = len(doneListAccepted) + len(doneListDiscarded) +\
                  len(newDoneAccepted) + len(newDoneDiscarded)

        # We have finished when there is not more input ctf (stream closed)
        # and the number of processed ctf is equal to the number of inputs
        if self.calculateConsensus.get():
            maxMovieSize = min(len(self.allMovies1), len(self.allMovies2))
        else:
            maxMovieSize = len(self.allMovies1)

        self.finished = (self.isStreamClosed and allDone == maxMovieSize)

        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN


        def readOrCreateOutputs(doneList, newDone, label=''):
            if len(doneList) > 0 or len(newDone) > 0:
                movSet = self._loadOutputSet(SetOfMovies, 'movies'+label+'.sqlite')
                micSet = self._loadOutputSet(SetOfMicrographs, 'micrographs'+label+'.sqlite')
                label = ACCEPTED if label == '' else DISCARDED
                self.fillOutput(movSet, micSet, newDone, label)
                movSet.setSamplingRate(self.samplingRate)
                micSet.setSamplingRate(self.samplingRate)

                return movSet, micSet

            return None, None

        movieSet, micSet = readOrCreateOutputs(doneListAccepted, newDoneAccepted)
        movieSetDiscarded, micSetDiscarded = readOrCreateOutputs(doneListDiscarded, newDoneDiscarded, DISCARDED)


        if not self.finished and not newDoneDiscarded and not newDoneAccepted:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        def updateRelationsAndClose(movieSet, micSet, first, label=''):

            if os.path.exists(self._getPath('movies'+label+'.sqlite')):

                micsAttrName = 'outputMicrographs'+label
                self._updateOutputSet(micsAttrName, micSet, streamMode)
                self._updateOutputSet('outputMovies'+label, movieSet, streamMode)

                if first:
                    #self._defineTransformRelation(self.inputMovies1.get(),
                    #                             micSet)
                    #self._defineTransformRelation(self.inputMovies1, cSet)
                    #self._defineCtfRelation(mSet, cSet)
                    pass

                micSet.close()
                movieSet.close()

        updateRelationsAndClose(movieSet, micSet, firstTimeAccepted)
        updateRelationsAndClose(movieSetDiscarded, micSetDiscarded,
                                firstTimeDiscarded, DISCARDED)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)


    def fillOutput(self, movieSet, micSet, newDone, label):
        if newDone:
            inputMovieSet = self._loadInputMovieSet(self.movieFn1)
            inputMicSet = self._loadInputMicrographSet(self.micsFn)

            if self.calculateConsensus.get():
                inputMovieSet2 = self._loadInputMovieSet(self.movieFn2)

            for movieId in newDone:
                movie = inputMovieSet[movieId].clone()
                movie.setSamplingRate(self.samplingRate) #DMT
                mic = inputMicSet[movieId].clone()
                mic.setSamplingRate(self.samplingRate)  # DMT
                movie.setEnabled(self._getEnable(movieId))
                mic.setEnabled(self._getEnable(movieId))

                if self.calculateConsensus.get():
                    alignment1 = movie.getAlignment()
                    shiftX_1, shiftY_1 = alignment1.getShifts()


                    setAttribute(mic, '_alignment_corr', self.stats[movieId]['shift_corr'])
                    setAttribute(mic, '_alignment_mean_error', self.stats[movieId]['max_error'])
                    setAttribute(mic, '_alignment_max_error', self.stats[movieId]['mean_error'])

                    alignment = MovieAlignment(xshifts=shiftX_1, yshifts=shiftY_1)
                    movie.setAlignment(alignment)

                movieSet.append(movie)
                micSet.append(mic)

                self._writeCertainDoneList(movieId, label)

            inputMovieSet.close()
            inputMicSet.close()

            if self.calculateConsensus.get():
                inputMovieSet2.close()


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

        return outputSet

    # def _loadOutputSet_previous_protocol(self, SetClass, baseName):
    #     """
    #     Load the output set if it exists or create a new one.
    #     """
    #
    #     baseName_prev = 'micrographs.sqlite'
    #     setFile_previous = self._getPath_previous_protocol(baseName_prev)
    #     setFile = self._getPath(baseName)
    #
    #     if os.path.exists(setFile_previous):
    #         inputSet = SetClass(filename=setFile_previous)
    #
    #         if os.path.exists(setFile):
    #             outputSet = SetClass(filename=setFile)
    #             if (outputSet.__len__() == 0):
    #                 pwutils.path.cleanPath(setFile)
    #
    #         if os.path.exists(setFile):
    #             outputSet = SetClass(filename=setFile)
    #             outputSet.loadAllProperties()
    #             outputSet.enableAppend()
    #         else:
    #             outputSet = SetClass(filename=setFile)
    #             outputSet.setStreamState(outputSet.STREAM_OPEN)
    #
    #
    #
    #     return outputSet

    def _getPath_previous_protocol(self, baseName):
        """ Return a path inside the workingDir. """
        return os.path.join(self.micsFn, baseName)


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

        for movie1 in method1:  # reference Movies
            movieId = movie1.getObjId()
            if movieId in self.processedDict:
                continue
            for movie2 in method2:
                movieId2 = movie2.getObjId()
                if movieId2 != movieId:
                    continue
                self.processedDict.append(movieId)

                #This part is the one that I dont understand
                #try:
                 #   self._movieToMd(movie1, md1)
                  #  self._movieToMd(movie2, md2)

                #except TypeError as exc:
                 #   print("Error reading movie for id:%s. %s" % (movieId, exc))


    def alignmentCorrelationMovieStep(self, movieId):

        # TODO: Change this way to get the MOVIE.
        movie1 = self.inputMovies1.get()[movieId]
        movie2 = self.inputMovies2.get()[movieId]

        # FIXME: this is a workaround to skip errors, but it should be treat at checkNewInput
        if movie1 is None:
            return

        fn1 = movie1.getFileName()
        fn2 = movie1.getFileName()

        movieID1 = movie1.getObjId()
        movieID2 = movie2.getObjId()

        if movieID1 == movieID2:
            if fn1 == fn2:
                print('NAME AND ID ARE THE SAME')
                print(fn1)
                print(fn2)
                print(movieID1)
                print(movieID2)
            else:
                print('NAME AND ID ARE NOT THE SAME')
                return

        alignment1 = movie1.getAlignment()
        alignment2 = movie2.getAlignment()

        shiftX_1, shiftY_1 = alignment1.getShifts()
        shiftX_2, shiftY_2 = alignment2.getShifts()

        # Transformation of the shifts to calculate the shifts trajectory correlation
        S1 = np.ones([3, len(shiftX_1)])
        S2 = np.ones([3, len(shiftX_2)])

        S1[0, :] = shiftX_1
        S1[1, :] = shiftY_1

        S2[0, :] = shiftX_2
        S2[2, :] = shiftY_2

        A = np.dot(np.dot(S1, S2.T), np.linalg.inv(np.dot(S2, S2.T)))
        S2_p = np.dot(A, S2)

        # mse = ((S1 - S2_p) ** 2).mean(axis=1)
        # maxe = np.max(S1 - S2_p)
        # error = S1 -S2_p
        # print('S2_PRIME')
        # print(S2_p)
        # print('S1')
        # print(S1)
        # print('HOMOGENOUS COORDINATES')
        # print('General error')
        # print(error)
        # print('MSE per dimension')
        # print(mse)
        # print('MAX ERROR')
        # print(maxe)


        S1_cart = np.array([S1[0, :]/S1[2, :], S1[1, :]/S1[2, :]])
        S2_p_cart = np.array([S2_p[0, :] / S2_p[2, :], S2_p[1, :] / S2_p[2, :]])
        meane_cart = (S1_cart - S2_p_cart).mean()
        maxe_cart = np.max(S1_cart - S2_p_cart)
        corrX_cart = np.corrcoef(S1_cart[0, :], S2_p_cart[0, :])[0, 1]
        corrY_cart = np.corrcoef(S1_cart[1, :], S2_p_cart[1, :])[0, 1]
        corr_cart = np.corrcoef(S1_cart[1, :], S2_p_cart[1, :])[0, 1]

        # error_cart = S1_cart - S2_p_cart
        # print('CARTESIANS COORDINATES')
        print('Mean error')
        print(meane_cart)
        print('MAX ERROR')
        print(maxe_cart)
        print('Correlation x')
        print(corrX_cart)
        print('Correlation y')
        print(corrY_cart)
        print('general corr')
        print(corr_cart)


        if corr_cart >= self.minConsCorrelation.get():
            print('Alignment correct')
            fn = self._getMovieSelecFileAccepted()
            with open(fn, 'a') as f:
                f.write('%d T\n' % movieID1)

        elif corr_cart < self.minConsCorrelation.get():
            print('Discrepancy in the alignment')
            print(corr_cart)
            fn = self._getMovieSelecFileDiscarded()
            with open(fn, 'a') as f:
                f.write('%d F\n' % movieID1)

        stats_loc = {'shift_corr': corr_cart, 'shift_corr_X': corrX_cart, 'shift_corr_Y': corrY_cart,
                     'max_error': maxe_cart, 'mean_error': meane_cart}

        self.stats[movieID1] = stats_loc
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

    def _getMovieSelecFileAccepted(self):
        return self._getExtraPath('selection-movie-accepted.txt')

    def _getMovieSelecFileDiscarded(self):
        return self._getExtraPath('selection-movie-discarded.txt')

    def _readtMovieId(self, accepted):
        if accepted:
            fn = self._getMovieSelecFileAccepted()
        else:
            fn = self._getMovieSelecFileDiscarded()
        moviesList = []
        # Check what items have been previously done
        if os.path.exists(fn):
            with open(fn) as f:
                moviesList += [int(line.strip().split()[0]) for line in f]
        return moviesList

    def _getEnable(self, ctfId):
        fn = self._getMovieSelecFileAccepted()
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

    def _loadInputMicrographSet(self, micsFn):
        baseName_prev = 'micrographs.sqlite'
        micsFn_prev = os.path.join(micsFn, baseName_prev)
        self.debug("Loading input db: %s" % micsFn_prev)
        micSet = SetOfMicrographs(filename=micsFn_prev)
        micSet.loadAllProperties()
        return micSet

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
