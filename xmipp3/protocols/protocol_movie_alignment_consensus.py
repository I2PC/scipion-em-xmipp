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
Consensus alignment protocol
"""

import os
from datetime import datetime

import numpy as np
import re
from pwem.objects import SetOfMovies, SetOfMicrographs, MovieAlignment
from pyworkflow.object import Set
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils

from pwem.protocols import ProtAlignMovies
from pyworkflow.protocol.constants import (STATUS_NEW)
from xmipp3.convert import getScipionObj
from pwem.constants import ALIGN_NONE

ACCEPTED = 'Accepted'
DISCARDED = 'Discarded'


class XmippProtConsensusMovieAlignment(ProtAlignMovies):
    """
    Protocol to estimate the agreement between different movie alignment
    algorithms in the Global Shifts).
    """

    _label = 'movie alignment consensus'
    outputName = 'consensusAlignments'

    def __init__(self, **args):
        ProtAlignMovies.__init__(self, **args)
        self._freqResol = {}
        self.stepsExecutionMode = params.STEPS_SERIAL
        #self.stepsExecutionMode = STEPS_PARALLEL


    def _defineParams(self, form):
        form.addSection(label='Input Consensus')
        form.addParam('inputMovies1', params.PointerParam, pointerClass='SetOfMovies',
                      label="Reference Aligned Movies", important=True,
                      help='Select the aligned movies to evaluate (this first set will give the global shifts)')

        form.addParam('inputMovies2', params.PointerParam,
                      pointerClass='SetOfMovies',
                      label="Secondary Aligned Movies",
                      help='Shift to be compared with reference alignment')

        form.addParam('minConsCorrelation', params.FloatParam, default=-1,
                      label='Minimum consensus shifts correlation',
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
        self.newIDs = []
        self.allMovies1 = {}
        self.allMovies2 = {}
        self.movieFn1 = self.inputMovies1.get().getFileName()
        self.movieFn2 = self.inputMovies2.get().getFileName()
        self.samplingRate = self.inputMovies1.get().getFirstItem().getSamplingRate()
        self.micsFn = self._getMicsPath(self.movieFn1)
        self.stats = {}

        movieSteps = self._checkNewInput()

        self._insertFunctionStep('createOutputStep',
                                 prerequisites=movieSteps, wait=True)

    def createOutputStep(self):
        pass

    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all movies
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()


    def _checkNewInput(self):
        # Check if there are new movies to process from the input set
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = max(datetime.fromtimestamp(os.path.getmtime(self.movieFn1)),
                    datetime.fromtimestamp(os.path.getmtime(self.movieFn2)))
        # If the input movies.sqlite have not changed since our last check,
        # it does not make sense to check for new input data

        if self.lastCheck < mTime:
            return None

        movieSet1 = self._loadInputMovieSet(self.movieFn1)
        movieSet2 = self._loadInputMovieSet(self.movieFn2)

        movieDict1 = {movie.getObjId(): movie.clone() for movie in movieSet1.iterItems()}
        movieDict2 = {movie.getObjId(): movie.clone() for movie in movieSet2.iterItems()}


        if len(self.allMovies1) > 0:
            newIds1 = [idMovie for idMovie in movieDict1.keys() if idMovie not in self.processedDict]
            #newMoviesDict1 = {idMovie: self.allMovies1.get(idMovie) for idMovie in newIds1} #Comentar esta linea
        else:
            newIds1 = list(movieDict1.keys())
            #newMoviesDict1 = movieDict1#Comentar esta linea


        #self.allMovies1.update(newMoviesDict1)#Comentar esta linea
        self.allMovies1.update(movieDict1)

        if len(self.allMovies2) > 0:
            newIds2 = [idMovie for idMovie in movieDict2.keys() if idMovie not in self.processedDict]
            #newMoviesDict2 = {idMovie: self.allMovies2.get(idMovie) for idMovie in newIds2}#Comentar esta linea
        else:
            newIds2 = list(movieDict2.keys())
            #newMoviesDict2 = movieDict2#Comentar esta linea

        #self.allMovies2.update(newMoviesDict2)#Comentar esta linea
        self.allMovies2.update(movieDict2)

        self.lastCheck = datetime.now()
        self.isStreamClosed = movieSet1.isStreamClosed() and \
                              movieSet2.isStreamClosed()

        movieSet1.close()
        movieSet2.close()

        outputStep = self._getFirstJoinStep()

        #Valora poner un OR porque a veces uno no va a ser mas grande porque ya termino
        if len(set(self.allMovies1)) > len(set(self.processedDict)) and \
           len(set(self.allMovies2)) > len(set(self.processedDict)):

            #fDeps = self._insertNewMovieSteps(newMoviesDict1, newMoviesDict2, self.insertedDict)
            fDeps = self._insertNewMovieSteps(newIds1, newIds2, self.insertedDict)

            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()


    def _insertNewMovieSteps(self, movies1Dict, movies2Dict, insDict):
        deps = []
        # movies1Dict = setOfMovies1.getObjDict(includeBasic=True)
        # movies2Dict = setOfMovies2.getObjDict(includeBasic=True)
        discrepId = self._insertFunctionStep("movieDiscrepancyStep",
                                             movies1Dict, movies2Dict,
                                             prerequisites=[])
        deps.append(discrepId)
        for movieID in self.newIDs:
            if movieID not in insDict:
                stepId = self._insertFunctionStep('alignmentCorrelationMovieStep', movieID,
                                                  prerequisites=[discrepId])
                deps.append(stepId)
                insDict[movieID] = stepId
                self.processedDict.append(movieID)

        self.newIDs = []
        return deps


    def movieDiscrepancyStep(self, newMoviesDict1, newMoviesDict2):
        #newIDs = set(newMoviesDict1.keys()).intersection(set(newMoviesDict2.keys()))
        newIDs = list(set(newMoviesDict1).intersection(set(newMoviesDict2))) #LIsta de cosas

        if len(newIDs) > 0:
            #self.newIDs = []
            #self.newIDs = [id for id in newIDs if id not in self.processedDict]
            self.newIDs.extend(newIDs)
        #     print('Discrepancy step pass, new IDs to process')
        # else:
        #     print('Discrepancy no pass')


    def alignmentCorrelationMovieStep(self, movieId):
        movie1 = self.allMovies1.get(movieId)
        movie2 = self.allMovies2.get(movieId)

        # # FIXME: this is a workaround to skip errors, but it should be treat at checkNewInput
        if (movie1 is None) or (movie2 is None):
            print('Maybe is here AlignmentCorrelationMOvieStep movie1 or movie2 are None')
            return

        fn1 = movie1.getFileName()
        fn2 = movie1.getFileName()
        movieID1 = movie1.getObjId()
        movieID2 = movie2.getObjId()

        print(fn1)
        print(fn2)
        print(movieID1)
        print(movieID2)

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

        S1_cart = np.array([S1[0, :]/S1[2, :], S1[1, :]/S1[2, :]])
        S2_p_cart = np.array([S2_p[0, :] / S2_p[2, :], S2_p[1, :] / S2_p[2, :]])
        rmse_cart = np.sqrt((np.square(S1_cart - S2_p_cart)).mean())
        maxe_cart = np.max(S1_cart - S2_p_cart)
        corrX_cart = np.corrcoef(S1_cart[0, :], S2_p_cart[0, :])[0, 1]
        corrY_cart = np.corrcoef(S1_cart[1, :], S2_p_cart[1, :])[0, 1]
        corr_cart = np.min([corrY_cart, corrX_cart])

        print('Root Mean Squared Error %f' %rmse_cart)
        print('Max Error %f' %maxe_cart)
        print('Correlation X %f' %corrX_cart)
        print('Correlation Y %f' %corrY_cart)
        print('General Corr (min X&Y) %f' %corr_cart)

        if corr_cart >= self.minConsCorrelation.get():
            print('Alignment shift trajectory correlated')
            fn = self._getMovieSelecFileAccepted()
            with open(fn, 'a') as f:
                f.write('%d T\n' % movieID1)

        elif corr_cart < self.minConsCorrelation.get():
            print('Discrepancy in the alignment with correlation %f' %corr_cart)
            fn = self._getMovieSelecFileDiscarded()
            with open(fn, 'a') as f:
                f.write('%d F\n' % movieID1)

        stats_loc = {'shift_corr': corr_cart, 'shift_corr_X': corrX_cart, 'shift_corr_Y': corrY_cart,
                     'max_error': maxe_cart, 'rmse_error': rmse_cart}

        self.stats[movieID1] = stats_loc
        self._store()

    def _checkNewOutput(self):
        """ Check for already selected movies and update the output set. """
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

        # We have finished when there is not more input movies (stream closed)
        # and the number of processed movies is equal to the number of inputs
        maxMovieSize = min(len(self.allMovies1), len(self.allMovies2))
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
            inputMovieSet2 = self._loadInputMovieSet(self.movieFn2)

            for movieId in newDone:
                movie = inputMovieSet[movieId].clone()
                movie.setSamplingRate(self.samplingRate)
                mic = inputMicSet[movieId].clone()
                mic.setSamplingRate(self.samplingRate)
                movie.setEnabled(self._getEnable(movieId))
                mic.setEnabled(self._getEnable(movieId))

                alignment1 = movie.getAlignment()
                shiftX_1, shiftY_1 = alignment1.getShifts()
                setAttribute(mic, '_alignment_corr', self.stats[movieId]['shift_corr'])
                setAttribute(mic, '_alignment_rmse_error', self.stats[movieId]['rmse_error'])
                setAttribute(mic, '_alignment_max_error', self.stats[movieId]['max_error'])

                alignment = MovieAlignment(xshifts=shiftX_1, yshifts=shiftY_1)
                movie.setAlignment(alignment)

                movieSet.append(movie)
                micSet.append(mic)

                self._writeCertainDoneList(movieId, label)

            inputMovieSet.close()
            inputMicSet.close()
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


    def _getMicsPath(self, moviesFn1):
        pattern = r'movies.sqlite'
        fnDirBase = re.sub(pattern, "", moviesFn1)
        path1 = fnDirBase + 'micrographs.sqlite'
        path2 = fnDirBase + 'micrographs_dose-weighted.sqlite'

        if (os.path.exists(path1)) and (os.path.getsize(path1) > 0):
            return path1
        elif os.path.exists(path2) and (os.path.getsize(path2) > 0):
            return path2
        else:
            return None


    def _summary(self):
        pass
        #return message


    def _validate(self):
        """ The function of this hook is to add some validation before the
        protocol is launched to be executed. It should return a list of
        errors. If the list is empty the protocol can be executed.
        """
        errors = []
        # if (self.inputMovies1.get().hasAlignment() == ALIGN_NONE) or \
        #         (self.inputMovies2.get().hasAlignment() == ALIGN_NONE):
        #     errors.append("The inputs ( _Input Movies 1_ or _Input Movies 2_ must be aligned before")

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


    def _writeCertainDoneList(self, movieId, label):
        """ Write to a text file the items that have been done. """
        doneFile = self._getCertainDone(label)
        with open(doneFile, 'a') as f:
            f.write('%d\n' % movieId)


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


    def _getEnable(self, movieId):
        fn = self._getMovieSelecFileAccepted()
        # Check what items have been previously done
        if os.path.exists(fn):
            with open(fn) as f:
                for line in f:
                    if movieId == int(line.strip().split()[0]):
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
        self.debug("Loading input db: %s" % micsFn)
        micSet = SetOfMicrographs(filename=micsFn)
        micSet.loadAllProperties()
        return micSet


def setAttribute(obj, label, value):
    if value is None:
        return
    setattr(obj, label, getScipionObj(value))

