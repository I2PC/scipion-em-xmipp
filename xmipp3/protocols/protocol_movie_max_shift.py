# **************************************************************************
# *
# * Authors:     David Maluenda    (dmaluenda@cnb.csic.es)
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

from os.path import exists
import numpy as np

from pyworkflow import VERSION_2_0
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Set
from pyworkflow.protocol.constants import STATUS_NEW
from pyworkflow.protocol.params import PointerParam
from pyworkflow.utils.properties import Message

from pwem.protocols import ProtProcessMovies
from pwem.objects import SetOfMicrographs, SetOfMovies


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
    _lastUpdateVersion = VERSION_2_0
    
    REJ_TYPES = ['by frame', 'by whole movie', 'by frame and movie', 
                 'by frame or movie']
    REJ_FRAME = 0
    REJ_MOVIE = 1
    REJ_AND = 2
    REJ_OR = 3

    def __init__(self, **args):
        ProtProcessMovies.__init__(self, **args)
        self.acceptedMoviesList = []
        self.discardedMoviesList = []
        self.acceptedDone = 0
        self.discardedDone = 0
        self.alreadyLoad = False

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

        form.addParam('maxFrameShift', params.FloatParam, default=5,
                       label='Max. frame shift (A)',
                       condition='rejType==%s or rejType==%s or rejType==%s'
                                  % (self.REJ_FRAME, self.REJ_AND, self.REJ_OR),
                       help='Maximum drift between consecutive frames '
                            'to evaluate the frame condition.')
        form.addParam('maxMovieShift', params.FloatParam, default=15,
                       label='Max. movie shift (A)',
                       condition='rejType==%s or rejType==%s or rejType==%s'
                                  % (self.REJ_MOVIE, self.REJ_AND, self.REJ_OR),
                       help='Maximum total travel to evaluate the whole movie '
                            'condition.')
        
    #--------------------------- INSERT steps functions ------------------------
    def _insertMovieStep(self, movie):
        """ Insert the processMovieStep for a given movie.
        """
        if not self.alreadyLoad:
            # looking for a setOfMicrographs related to the inputMovies
            self.setInputMics()
        # Insert the selection/rejection step
        movieStepId = self._insertFunctionStep('_evaluateMovieAlign',
                                               movie.getObjId(),
                                               prerequisites=[])
        return movieStepId

    def _evaluateMovieAlign(self, movieId):
        """ Fill the accepted or the rejected list with the movie.
        """
        movie = self.inputMovies.get()[movieId].clone()
        alignment = movie.getAlignment()
        sampling = self.samplingRate

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
            if self.rejType == self.REJ_MOVIE or evalBoth:
                rangeX = np.max(shiftArrayX) - np.min(shiftArrayX)
                rangeY = np.max(shiftArrayY) - np.min(shiftArrayY)
                rangeM = max(rangeX, rangeY) * sampling
                rejectedByMovie = rangeM > self.maxMovieShift.get()

            if self.rejType == self.REJ_FRAME or evalBoth:
                frameShiftX = np.abs(np.diff(shiftArrayX))
                frameShiftY = np.abs(np.diff(shiftArrayY))
                maxShiftX = np.max(frameShiftX)
                maxShiftY = np.max(frameShiftY)
                maxShiftM = max(maxShiftX, maxShiftY) * sampling
                rejectedByFrame = maxShiftM > self.maxFrameShift.get()

            if self.rejType == self.REJ_AND:
                if rejectedByFrame and rejectedByMovie:
                    self.discardedMoviesList.append(movie)
                else:
                    self.acceptedMoviesList.append(movie)
            else:  # for the OR and the individuals evaluations
                if rejectedByFrame or rejectedByMovie:
                    self.discardedMoviesList.append(movie)
                else:
                    self.acceptedMoviesList.append(movie)
        else:  # we accept the movie if no shifts is associated
            self.acceptedMoviesList.append(movie)

    def _checkNewOutput(self):
        """ Check for already selected Movies and update the output set. """
        if getattr(self, 'finished', False):
            return

        # load if first time in order to make dataSets relations
        firstTimeAcc = self.acceptedDone==0
        firstTimeDisc = self.discardedDone==0

        # Load previously done items
        preDone = self.acceptedDone + self.discardedDone

        # Taking newly done items in movie lists
        newDoneAccepted = self.acceptedMoviesList[self.acceptedDone:]
        newDoneDiscarded = self.discardedMoviesList[self.discardedDone:]

        # Updating the done items
        self.acceptedDone = len(self.acceptedMoviesList)
        self.discardedDone = len(self.discardedMoviesList)
        
        allDone = preDone + len(newDoneAccepted) + len(newDoneDiscarded)

        # Some checks to debug
        self.debug('_checkNewOutput: ')
        self.debug('   listOfMovies: %d,' %len(self.listOfMovies))
        self.debug('   doneList: %d,' %preDone)
        self.debug('   newDoneAccepted: %d.' %len(newDoneAccepted))
        self.debug('   newDoneDiscarded: %d,' %len(newDoneDiscarded))
    
        # We have finished when there is not more input movies (stream closed)
        # and the number of processed movie is equal to the number of inputs
        self.finished = self.streamClosed and allDone == len(self.listOfMovies)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        if not self.finished and not newDoneDiscarded and not newDoneAccepted:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            return

        def fillOutput(newDoneList, firstTime, AccOrDisc='Accepted'):
            """ A single function is used to fill the two sets (Accepted/Rej.)
            """
            suffix = 'Discarded' if AccOrDisc=='Discarded' else ''
            enable = False if AccOrDisc=='Discarded' else True
                
            movieSet = self._loadOutputSet(SetOfMovies,
                                                     'movies%s.sqlite' % suffix)
            micsSet = self._loadOutputSet(SetOfMicrographs,
                                                  'micrographs%s.sqlite'%suffix)
            micsDwSet = self._loadOutputSet(SetOfMicrographs,
                                  'micrographs_dose-weighted%s.sqlite' % suffix)

            def tryToAppend(outSet, micOut, tries=1, labelPrefix='movie'):
                """ When micrograph is very big, sometimes it's not ready to be read
                Then we will wait for it up to a minute in 6 time-growing tries. 
                Returns True if fails! """
                if micOut is None:
                    return
                try:
                    micOut.setEnabled(enable)
                    outSet.append(micOut)
                except Exception as ex:
                    if tries < 10:
                        from time import sleep
                        sleep(tries*3)
                        tryToAppend(outSet, micOut, tries+1)
                    else:
                        labelStr = ' '.join([labelPrefix, micOut.getMicName()])
                        self.warning("The %s seems corrupted. Skipping it...\n "
                                     " > %s" % (labelStr, ex))

            for movie in newDoneList:
                tryToAppend(movieSet, movie,
                            labelPrefix='movie')
                if self.inputMics is not None:
                    mic = self.getMicFromMovie(movie, isDoseWeighted=False)
                    tryToAppend(micsSet, mic,
                                labelPrefix='micrograph')
                if self.inputDwMics is not None:
                    micDw = self.getMicFromMovie(movie, isDoseWeighted=True)
                    tryToAppend(micsDwSet, micDw,
                                labelPrefix='micDW')
            
            if movieSet.getSize() > 0:
                self._updateOutputSet('outputMovies%s' % suffix, movieSet,
                                      streamMode)
                                      
            if self.inputMics is not None and micsSet.getSize() > 0:
                self._updateOutputSet('outputMicrographs%s' % suffix, micsSet,
                                      streamMode)
            if self.inputDwMics is not None and micsDwSet.getSize() > 0:
                self._updateOutputSet('outputMicrographsDoseWeighted%s' % suffix,
                                      micsDwSet, streamMode)
            if firstTime:  # define relation just the first time
                if movieSet.getSize() > 0:
                    self._defineTransformRelation(self.inputMovies.get(), movieSet)
                if self.inputMics is not None and micsSet.getSize() > 0:
                    self._defineTransformRelation(self.inputMics, micsSet)
                if self.inputDwMics is not None and micsDwSet.getSize() > 0:
                    self._defineTransformRelation(self.inputDwMics, micsDwSet)
            
            movieSet.close()
            if self.inputMics is not None:
                micsSet.close()
            if self.inputDwMics is not None:
                micsDwSet.close()

        # We fill/update the output if there are something new or to close sets
        if newDoneAccepted or (self.finished and hasattr(self, 'outputMovies')):
            fillOutput(newDoneAccepted, firstTimeAcc, AccOrDisc='Accepted')

        # We fill/update the output for the discarded movies
        if newDoneDiscarded or (self.finished and
                                hasattr(self, 'outputMoviesDiscarded')):
            fillOutput(newDoneDiscarded, firstTimeDisc, AccOrDisc='Discarded')

        # Unlock createOutputStep if finished all jobs
        if self.finished:  
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

    # FIXME: Methods will change when using the streaming for the output
    def createOutputStep(self):
        # Do nothing now, the output should be ready.
        pass

    #--------------------------- UTILS functions -------------------------------
    def _loadOutputSet(self, SetClass, baseName):
        """ Load the output set if it exists or create a new one based on the inputs.
        """
        if SetClass == SetOfMicrographs:
            if 'dose-weighted' in baseName:
                if self.inputDwMics is None:
                    # if no DwMics to do, do nothing and exit
                    return None
                inputSet = self.inputDwMics
            else:
                if self.inputMics is None:
                    # if no mics to do, do nothing and exit
                    return None
                inputSet = self.inputMics
        else:
            inputSet = self.inputMovies.get()

        setFile = self._getPath(baseName)

        if exists(setFile):
            outputSet = SetClass(filename=setFile)
            if len(outputSet) == 0:
                pwutils.path.cleanPath(setFile)

        if exists(setFile):
            outputSet = SetClass(filename=setFile)
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
        inNotFound = []
        inNotFound2 = []
        self.alreadyLoad = True
        parentProt = self.getMapper().getParent(self.inputMovies.get())

        self.inputMics = getattr(parentProt, 'outputMicrographs', None)
        if self.inputMics is None:
            inNotFound.append('outputMicrographs')
            inNotFound2.append('Mics')

        self.inputDwMics = getattr(parentProt, 'outputMicrographsDoseWeighted', None)
        if self.inputDwMics is None:
            inNotFound.append('outputMicrographsDoseWeighted')
            inNotFound2.append('Dose Weighted Mics')

        if inNotFound:
            self.warning("WARNING: The '%s' has no %s. Then, no %s will be "
                         "produced%s." % (parentProt.getObjLabel(),
                                          ' nor '.join(inNotFound),
                                          ' nor '.join(inNotFound2),
                                          '' if len(inNotFound)<2
                                                else ', only movies'))

    def getMicFromMovie(self, movie, isDoseWeighted):
        """ Get the Micrograph/DwMicrograph related with a certain movie
        """
        movieMicName = movie.getMicName()
        inMics = self.inputDwMics if isDoseWeighted else self.inputMics
        for mic in inMics:  # self.inputMics:
            if mic.getMicName() == movieMicName:
                return mic
        micStr = 'DwMic' if isDoseWeighted else 'Mic'
        self.warning('%s NOT found for movie "%s"' % (micStr, movieMicName))


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
