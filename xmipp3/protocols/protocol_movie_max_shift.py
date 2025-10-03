# **************************************************************************
# *
# * Authors:    David Maluenda    (dmaluenda@cnb.csic.es)
# *             Daniel Marchan (da.marchan@cnb.csic.es) REFACTOR
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
from os.path import exists
import numpy as np
import copy

from pyworkflow import VERSION_3_0
from pyworkflow import UPDATED
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Set
from pyworkflow.protocol.constants import STATUS_NEW
from pyworkflow.protocol.params import PointerParam
from pyworkflow.utils.properties import Message

from pwem.protocols import ProtProcessMovies
from pwem.objects import SetOfMicrographs, SetOfMovies


OUTPUT_MICS = "outputMicrographs"
OUTPUT_MICS_DISCARDED = "outputMicrographs"
OUTPUT_MOVIES = "outputMovies"
OUTPUT_MOVIES_DISCARDED = "outputMoviesDiscarded"
OUTPUT_MICS_DW = "outputMicrographsDoseWeighted"
OUTPUT_MICS_DW_DISCARDED = "outputMicrographsDoseWeightedDiscarded"

class XmippProtMovieMaxShift(ProtProcessMovies):
    """
    Protocol to make an automatic rejection of those movies whose
    frames move more than a given threshold.
        Rejection criteria:
            - *by frame*: Rejects movies with drifts between frames
                              bigger than a certain maximum.
            - *by whole movie*: Rejects movies with a total travel
                                         bigger than a certain maximum.
            - *by frame and movie*: Rejects movies if both conditions
                                                above are met.
            - *by frame or movie*: Rejects movies if one of the conditions
                                             above are met.
    """
    _label = 'movie maxshift'
    _devStatus = UPDATED
    _lastUpdateVersion = VERSION_3_0
    _possibleOutputs = {OUTPUT_MICS: SetOfMicrographs,
                        OUTPUT_MICS_DW: SetOfMicrographs,
                        OUTPUT_MICS_DISCARDED: SetOfMicrographs,
                        OUTPUT_MICS_DW_DISCARDED: SetOfMicrographs,
                        OUTPUT_MOVIES: SetOfMovies,
                        OUTPUT_MOVIES_DISCARDED: SetOfMovies
                        }
    
    REJ_TYPES = ['by frame', 'by whole movie', 'by frame and movie', 
                 'by frame or movie']
    REJ_FRAME = 0
    REJ_MOVIE = 1
    REJ_AND = 2
    REJ_OR = 3

    def __init__(self, **args):
        ProtProcessMovies.__init__(self, **args)
        #self.stepsExecutionMode = STEPS_PARALLEL

    # -------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputMovies', PointerParam, important=True,
                      label=Message.LABEL_INPUT_MOVS,
                      pointerClass='SetOfMovies', 
                      help='Select a set of previously aligned Movies.')

        form.addParam('rejType', params.EnumParam, choices=self.REJ_TYPES,
                      label='Rejection type', default=self.REJ_OR,
                      help='Rejection criteria:\n'
                           ' - *by frame*: Rejects movies with drifts between '
                                    'frames bigger than a certain maximum.\n'
                           ' - *by whole movie*: Rejects movies with a total '
                                    'travel bigger than a certain maximmum.\n'
                           ' - *by frame and movie*: Rejects movies if both '
                                    'conditions above are met.\n'
                           ' - *by frame or movie*: Rejects movies if one of '
                                    'the conditions above are met.')

        form.addParam('maxFrameShift', params.FloatParam, default=10,
                       label='Max. frame shift (A)',
                       condition='rejType==%s or rejType==%s or rejType==%s'
                                  % (self.REJ_FRAME, self.REJ_AND, self.REJ_OR),
                       help='Maximum drift between consecutive frames '
                            'to evaluate the frame condition.')
        form.addParam('maxMovieShift', params.FloatParam, default=45,
                       label='Max. movie shift (A)',
                       condition='rejType==%s or rejType==%s or rejType==%s'
                                  % (self.REJ_MOVIE, self.REJ_AND, self.REJ_OR),
                       help='Maximum total travel to evaluate the whole movie '
                            'condition.')

        
    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        """ Insert the steps to perform movie alignment evaluation
        """
        self.initializeStep()
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=[], wait=True, needsGPU=False)

    def initializeStep(self):
        self.samplingRate = self.inputMovies.get().getSamplingRate()
        self.movsFn = self.inputMovies.get().getFileName()
        # Important to have both:
        self.insertedIds = []  # Contains images that have been inserted in a Step (checkNewInput).
        # Contains images that have been processed in a Step (checkNewOutput).
        self.acceptedIds = []
        self.discardedIds = []
        self.isStreamClosed = self.inputMovies.get().isStreamClosed()
        self.alreadyLoad = False

    def createOutputStep(self):
        self._closeOutputSet()

    def _loadInputSet(self, movsFn):
        """ Load the input set of movies and create a list. """
        self.debug("Loading input db: %s" % movsFn)
        movSet = SetOfMovies(filename=movsFn)
        movSet.loadAllProperties()
        movSet.close()
        self.debug("Closed db.")
        return movSet

    def _loadMicAssociatedInputSet(self):
        """ Load the input set of mics and create a list. """
        parentProt = self.getMapper().getParent(self.inputMovies.get())
        micSet = getattr(parentProt, self.outMicName, None)
        micSet.loadAllProperties()
        micSet.close()

        return micSet

    def _stepsCheck(self):
        # Input micrograph set can be loaded or None when checked for new inputs
        # If None, we load it
        self._checkNewInput()
        self._checkNewOutput()

    def _checkNewInput(self):
        # Check if there are new micrographs to process from the input set
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = datetime.fromtimestamp(os.path.getmtime(self.movsFn))
        self.debug('Last check: %s, modification: %s'
                   % (pwutils.prettyTime(self.lastCheck),
                      pwutils.prettyTime(mTime)))
        # If the input micrographs.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and self.insertedIds: # If this is empty it is dut to a static "continue" action or it is the first round
            return None

        # Open input micrographs.sqlite and close it as soon as possible
        movSet = self._loadInputSet(self.movsFn)
        movSetIds = movSet.getIdSet()
        newIds = [idMic for idMic in movSetIds if idMic not in self.insertedIds]

        self.isStreamClosed = movSet.isStreamClosed()
        self.lastCheck = datetime.now()
        movSet.close()

        outputStep = self._getFirstJoinStep()

        if self.isContinued() and not self.insertedIds: # For "Continue" action and the first round
            doneIds, _, _, _ = self._getAllDoneIds()
            skipIds = list(set(newIds).intersection(set(doneIds)))
            newIds = list(set(newIds).difference(set(doneIds)))
            self.info("Skipping Mics with ID: %s, seems to be done" % skipIds)
            self.insertedIds = doneIds # During the first round of "Continue" action it has to be filled

        if newIds:
            fDeps = self._insertNewMoviesSteps(newIds)
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def _insertNewMoviesSteps(self, newIds):
        """ Insert the processMovieStep for a given movie.
        """
        if not self.alreadyLoad:
            # looking for a setOfMicrographs related to the inputMovies
            self.setInputMics()

        # Insert the selection/rejection step process
        deps = []
        stepId = self._insertFunctionStep(self._evaluateMovieAlign, newIds,  needsGPU=False,
                                          prerequisites=[])
        deps.append(stepId)

        for movId in newIds:
            self.insertedIds.append(movId)

        return deps

    def _evaluateMovieAlign(self, movIds):
        """ Fill the accepted or the rejected list with the movie.
        """
        inputMovies = self._loadInputSet(self.movsFn)
        sampling = self.samplingRate

        for movieId in movIds:
            movie = inputMovies.getItem("id", movieId).clone()
            alignment = movie.getAlignment()
            # getShifts() returns the absolute shifts from a certain reference
            shiftListX, shiftListY = alignment.getShifts()
            # initialize the criteria values
            rejectedByMovie = False
            rejectedByFrame = False

            if any(shiftListX) or any(shiftListY):
                # we use np.arrays to use np.diff()
                shiftArrayX = np.asarray(shiftListX)
                shiftArrayY = np.asarray(shiftListY)

                evalBoth = self.rejType==self.REJ_AND or self.rejType==self.REJ_OR

                # --- Evaluation for accumulated displacement ---
                if self.rejType == self.REJ_MOVIE or evalBoth:
                    deltaX = np.diff(shiftArrayX)
                    deltaY = np.diff(shiftArrayY)
                    # Magnitude for each step
                    stepDistances = np.sqrt(deltaX ** 2 + deltaY ** 2)
                    # Total sum of the displacement
                    totalPath = np.sum(stepDistances) * sampling
                    rejectedByMovie = totalPath > self.maxMovieShift.get()

                # --- Evaluation by maximum displacement between frames ---
                if self.rejType == self.REJ_FRAME or evalBoth:
                    frameShiftX = np.diff(shiftArrayX)
                    frameShiftY = np.diff(shiftArrayY)
                    frameShifts = np.sqrt(frameShiftX ** 2 + frameShiftY ** 2)
                    maxShiftM = np.max(frameShifts) * sampling
                    rejectedByFrame = maxShiftM > self.maxFrameShift.get()

                if self.rejType == self.REJ_AND:
                    if rejectedByFrame and rejectedByMovie:
                        self.discardedIds.append(movieId)
                    else:
                        self.acceptedIds.append(movieId)
                else:  # for the OR and the individuals evaluations
                    if rejectedByFrame or rejectedByMovie:
                        self.discardedIds.append(movieId)
                    else:
                        self.acceptedIds.append(movieId)
            else:  # we accept the movie if no shifts is associated
                self.acceptedIds.append(movieId)

    def _checkNewOutput(self):
        """ Check for already selected Movies and update the output set. """
        # load if first time in order to make dataSets relations
        _, _, doneListAccepted, doneListDiscarded = self._getAllDoneIds()
        # Check for newly done items
        acceptedIds = copy.deepcopy(self.acceptedIds)
        discardedIds = copy.deepcopy(self.discardedIds)

        newDoneAccepted = [movId for movId in acceptedIds
                           if movId not in doneListAccepted]
        newDoneDiscarded = [movId for movId in discardedIds
                            if movId not in doneListDiscarded]

        firstTimeAccepted = len(doneListAccepted) == 0
        firstTimeDiscarded = len(doneListDiscarded) == 0

        allDone = len(doneListAccepted) + len(doneListDiscarded) + \
                  len(newDoneAccepted) + len(newDoneDiscarded)
        maxMicSize = self._loadInputSet(self.movsFn).getSize()
        # We have finished when there is not more input movies
        # (stream closed) and the number of processed movies is
        # equal to the number of inputs
        self.finished = self.isStreamClosed and allDone == maxMicSize
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        # Some checks to debug
        self.debug('_checkNewOutput: ')
        self.debug('   newDoneAccepted: %d.' % len(newDoneAccepted))
        self.debug('   newDoneDiscarded: %d,' % len(newDoneDiscarded))

        if not self.finished and (not newDoneDiscarded and not newDoneAccepted):
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        def fillOutput(newDoneList, firstTime, AccOrDisc='Accepted'):
            """ A single function is used to fill the two sets (Accepted/Rej.)
            """
            suffix1 = '' if self.outMicName == OUTPUT_MICS else '_dose-weighted'
            suffix = 'Discarded' if AccOrDisc=='Discarded' else ''

            enable = False if AccOrDisc=='Discarded' else True
                
            movieSet = self._loadOutputSet(SetOfMovies, 'movies%s.sqlite' % suffix)
            micsSet = self._loadOutputSet(SetOfMicrographs,'micrographs%s%s.sqlite'%(suffix1, suffix))
            print('micrographs%s%s.sqlite'%(suffix1, suffix))

            def tryToAppend(outSet, micOut):
                """ When micrograph is very big, sometimes it's not ready to be read
                Then we will wait for it up to a minute in 6 time-growing tries. 
                Returns True if fails! """
                if micOut is None:
                    print('Mic/Movie is None do not introduce')
                else:
                    micOut.setEnabled(enable)
                    outSet.append(micOut)

            inputMovies = self._loadInputSet(self.movsFn)
            inputMics = self._loadMicAssociatedInputSet()
            inputMicsIds = inputMics.getIdSet()

            for movieId in newDoneList:
                movie = inputMovies.getItem("id", movieId).clone()
                tryToAppend(movieSet, movie)
                if movieId in inputMicsIds:
                    mic = inputMics.getItem("id", movieId).clone()
                    tryToAppend(micsSet, mic)
                else:
                    self.info("Movie with id %d has not a micrograph associated" %movieId)
            
            if movieSet.getSize() > 0:
                self._updateOutputSet(OUTPUT_MOVIES + suffix, movieSet,
                                      streamMode)
                                      
            if self.inputMics is not None and micsSet.getSize() > 0:
                self._updateOutputSet(self.outMicName + suffix, micsSet,
                                      streamMode)
            if firstTime:  # define relation just the first time
                if movieSet.getSize() > 0:
                    self._defineTransformRelation(self.inputMovies.get(), movieSet)
                if self.inputMics is not None and micsSet.getSize() > 0:
                    self._defineTransformRelation(self.inputMics, micsSet)
            
            movieSet.close()
            if self.inputMics is not None:
                micsSet.close()

        # We fill/update the output if there are something new or to close sets
        if newDoneAccepted:
            fillOutput(newDoneAccepted, firstTimeAccepted, AccOrDisc='Accepted')
        # We fill/update the output for the discarded movies
        if newDoneDiscarded:
            fillOutput(newDoneDiscarded, firstTimeDiscarded, AccOrDisc='Discarded')

        # Unlock createOutputStep if finished all jobs
        if self.finished:  
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

        self._store()

    #--------------------------- UTILS functions -------------------------------
    def _getAllDoneIds(self):
        doneIds = []
        acceptedIds = []
        discardedIds = []
        sizeOutput = 0

        if hasattr(self, OUTPUT_MOVIES):
            sizeOutput += self.outputMovies.getSize()
            acceptedIds.extend(list(self.outputMovies.getIdSet()))
            doneIds.extend(acceptedIds)

        if hasattr(self, OUTPUT_MOVIES_DISCARDED):
            sizeOutput += self.outputMoviesDiscarded.getSize()
            discardedIds.extend(list(self.outputMoviesDiscarded.getIdSet()))
            doneIds.extend(discardedIds)

        return doneIds, sizeOutput, acceptedIds, discardedIds

    def _loadOutputSet(self, SetClass, baseName):
        """ Load the output set if it exists or create a new one based on the inputs.
        """
        if SetClass == SetOfMicrographs:
            if self.inputMics is None:
                # if no mics to do, do nothing and exit
                return None
            inputSet = self.inputMics
        else:
            inputSet = self._loadInputSet(self.movsFn)

        setFile = self._getPath(baseName)
        print(setFile)

        if exists(setFile):
            outputSet = SetClass(filename=setFile)
            if len(outputSet) == 0:
                pwutils.path.cleanPath(setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)

        outputSet.copyInfo(inputSet)

        return outputSet

    def setInputMics(self):
        """ Setting the self.inputMics to the SetOfMics associated to
            the input movies (or None if no relation is found).
            The same for DoseWeighted mics
        """
        self.alreadyLoad = True
        parentProt = self.getMapper().getParent(self.inputMovies.get())
        self.outMicName = None
        self.inputMics = None

        if hasattr(parentProt, OUTPUT_MICS):
            self.outMicName = OUTPUT_MICS
            self.inputMics = getattr(parentProt, OUTPUT_MICS, None)

        if hasattr(parentProt, OUTPUT_MICS_DW):
            self.outMicName = OUTPUT_MICS_DW
            self.inputMics = getattr(parentProt, OUTPUT_MICS_DW, None)

        if not self.inputMics:
            self.warning("WARNING: The outputMovies has no outputMicrographs associated. Then, no outputMicrographs will be "
                         "produced.")

    # ---------------------- INFO functions ------------------------------------
    def _validate(self):
        errors = []
        if not self.inputMovies.get().getFirstItem().hasAlignment():
            errors.append('The _Input Movies_ must come from an alignment '
                          'protocol.')
        return errors

    def _summary(self):
        def getSize(outNameCondition):
            for outName, outObj in self.iterOutputAttributes():
                if outNameCondition(outName):
                    return outObj.getSize()
            return 0

        moviesAcc = getSize(lambda x: not x.endswith('Discarded'))
        outDisc = getSize(lambda x: x.endswith('Discarded'))

        summary = ['Movies processed: %d' % (moviesAcc+outDisc),
                   'Movies rejected: *%d*' % outDisc]

        return summary
