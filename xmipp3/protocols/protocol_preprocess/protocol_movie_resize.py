# **************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
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

from pyworkflow import VERSION_1_2
from pyworkflow.utils.properties import Message
from pyworkflow.protocol.params import (PointerParam, FloatParam, EnumParam,
                                       IntParam)
from pyworkflow.object import Set
import pyworkflow.protocol.constants as cons
from pyworkflow.protocol.constants import STEPS_PARALLEL

from pwem.protocols import EMProtocol, ProtProcessMovies
from pwem.objects import SetOfMovies, Movie


RESIZE_SAMPLINGRATE = 0
RESIZE_DIMENSIONS = 1
RESIZE_FACTOR = 2


class XmippProtMovieResize(ProtProcessMovies):
    """
    Resize a set of movies. Only downsampling is allowed.

    AI Generated

    ## Overview

    The Movie Resize protocol downsamples a set of cryo-EM movies.

    Movies can be very large, and many early processing or quality-control steps do
    not require the original full pixel size. Downsampling movies reduces file
    size, memory use, and computational cost. It can also prepare movies for
    workflows that expect a specific sampling rate or image dimension.

    This protocol resizes each movie in Fourier space and produces a new set of
    movies with updated sampling rate metadata.

    Only downsampling is allowed. The protocol does not support increasing movie
    dimensions or decreasing the sampling rate.

    ## Inputs and General Workflow

    The input is a set of movies.

    For each movie, the protocol computes the new movie dimensions and output
    sampling rate according to the selected resize option. It then runs
    `xmipp_image_resize` on the movie stack, preserving the number of frames while
    changing the X and Y image size.

    The output movies are written as MRC movie stacks with names such as:

    `movie_000001_resize.mrcs`

    The protocol supports streaming. As new movies arrive in the input set, they
    are resized and appended to the output movie set.

    ## Input Movies

    The **Input movies** parameter defines the movie set to be resized.

    All movies are expected to share the dimensions and sampling rate stored in the
    input movie set. The protocol uses those values to compute the output size and
    sampling rate.

    The input movies are not modified. The protocol writes new resized movie files.

    ## Resize Option

    The **Resize option** parameter defines how the downsampling is specified.

    There are three options:

    **Sampling Rate**: the user provides the desired output sampling rate.

    **Dimensions**: the user provides the desired output image size in pixels.

    **Factor**: the user provides a downsampling factor.

    In all cases, the protocol computes a new X and Y dimension and keeps the same
    number of movie frames.

    ## Resize by Sampling Rate

    When **Sampling Rate** is selected, the user provides the desired output
    sampling rate in angstroms per pixel.

    The requested sampling rate must be larger than the original sampling rate,
    because only downsampling is allowed.

    The protocol computes the downsampling factor as:

    \[
    \text{factor} =
    \frac{\text{new sampling rate}}{\text{original sampling rate}}
    \]

    and then computes the new image dimension from that factor.

    This option is useful when the user wants the output movies to match a specific
    physical pixel size.

    ## Resize by Dimensions

    When **Dimensions** is selected, the user provides the desired output image
    size in pixels.

    The new dimension must be smaller than or equal to the original movie
    dimension, because only downsampling is allowed.

    The protocol computes the new sampling rate from the ratio between the
    original and new dimensions.

    This option is useful when a downstream workflow requires a specific movie
    width and height.

    ## Resize by Factor

    When **Factor** is selected, the user provides the downsampling factor.

    The factor must be greater than or equal to 1. A factor of 2 halves the image
    dimensions and doubles the sampling rate. A factor of 4 reduces the dimensions
    by four and multiplies the sampling rate by four.

    This option is useful when the user wants a simple integer or numerical
    downsampling ratio.

    ## Fourier Resize

    The protocol uses Fourier resizing.

    For each movie, it calls `xmipp_image_resize` with the Fourier option and the
    new X, Y, and frame dimensions. The number of frames is kept unchanged.

    Fourier resizing is appropriate for downsampling because it changes the spatial
    sampling while preserving the frequency-domain interpretation of the movie
    data.

    ## Output Movies

    The main output is **outputMovies**.

    This output is a new Scipion movie set containing the resized movies. Each
    output movie:

    - keeps the same object identifier as the corresponding input movie;
    - points to the resized `.mrcs` file;
    - copies the acquisition information;
    - stores the updated sampling rate;
    - keeps the same frame range as the input movie set.

    The output set can be used in downstream movie-processing workflows.

    ## Streaming Behavior

    The protocol supports streaming input.

    As new movies appear in the input set, the protocol inserts processing steps
    for them. Each movie is resized independently. When a movie has been processed,
    it is appended to the output movie set.

    The output set remains open while the input stream is open. When the input
    stream closes and all movies have been processed, the output stream is closed.

    This makes the protocol suitable for online processing workflows where movies
    arrive progressively during data acquisition.

    ## Summary Information

    The protocol summary reports the number of input movies, their original size,
    the number of output movies, and the resized output size.

    This allows the user to quickly verify that the expected downsampling was
    performed.

    ## Validation Rules

    The protocol validates that only downsampling is requested.

    For **Factor**, the factor must be greater than or equal to 1.

    For **Dimensions**, the requested new dimension must not be larger than the
    input movie dimension.

    For **Sampling Rate**, the requested sampling rate must be larger than the
    original sampling rate.

    If these conditions are not met, the protocol reports an error.

    ## Interpreting the Result

    The resized movies should be interpreted as downsampled versions of the input
    movies.

    The protocol does not align, dose-weight, correct motion, estimate CTF, or
    change the number of frames. It only changes the spatial sampling of each movie.

    Downsampling reduces high-resolution information. This is often acceptable for
    early quality-control or low-resolution processing, but it may not be
    appropriate if the full-resolution movie information is needed later.

    ## Practical Recommendations

    Use this protocol to reduce movie size and accelerate early processing.

    Choose resizing by sampling rate when the output should match a target pixel
    size.

    Choose resizing by dimensions when a downstream protocol expects a specific
    image size.

    Choose resizing by factor for simple downsampling ratios.

    Do not use downsampled movies for final high-resolution workflows unless the
    chosen sampling rate still supports the target resolution.

    Keep the original movies available if later full-resolution processing may be
    needed.

    In streaming workflows, use the output movie set as the downsampled stream for
    downstream protocols.

    ## Final Perspective

    Movie Resize is a movie-preprocessing protocol for spatial downsampling.

    For biological users, its value is that it creates smaller and faster-to-process
    movie stacks while preserving movie metadata such as acquisition information,
    frame range, and updated sampling rate.

    The protocol is useful for reducing computational cost and preparing movies for
    early processing, but it should be used with awareness that downsampling limits
    the highest resolution that can be recovered from the resized data.
    """
    _label = 'movie resize'
    _lastUpdateVersion = VERSION_1_2

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label=Message.LABEL_INPUT_MOVS,
                      help='Select a set of movies to be resized.')
        form.addParam('resizeOption', EnumParam,
                      choices=['Sampling Rate', 'Dimensions', 'Factor'],
                      default=RESIZE_SAMPLINGRATE,
                      label="Resize option", display=EnumParam.DISPLAY_COMBO,
                      help='Select an option to resize the images: \n '
                           '_Sampling Rate_: Set the desire sampling rate '
                           'to resize. \n'
                           '_Dimensions_: Set the output dimensions. \n'
                           '_Factor_: Set a resize factor to resize. \n ')
        form.addParam('resizeSamplingRate', FloatParam, default=4.0,
                      condition='resizeOption==%d' % RESIZE_SAMPLINGRATE,
                      label='Resize sampling rate (A/px)',
                      help='Set the new output sampling rate.')
        form.addParam('resizeDim', IntParam, default=1024,
                      condition='resizeOption==%d' % RESIZE_DIMENSIONS,
                      label='New image size (px)',
                      help='Size in pixels of the particle images '
                           '<x> <y=x> <z=x>.')
        form.addParam('resizeFactor', FloatParam, default=2.0,
                      condition='resizeOption==%d' % RESIZE_FACTOR,
                      label='Downsampling factor',
                      help='New size is the old one x resize factor.')
        form.addParallelSection(threads=1, mpi=1)

    # ------------------------ INSERT STEPS functions --------------------------

    def _insertNewMoviesSteps(self, insertedDict, inputMovies):
        """ Insert steps to process new movies (from streaming)
        Params:
            insertedDict: contains already processed movies
            inputMovies: input movies set to be check
        """
        deps = []
        # For each movie insert the step to process it
        for idx, movie in enumerate(self.inputMovies.get()):
            if movie.getObjId() not in insertedDict:
                stepId = self._insertMovieStep(movie)
                deps.append(stepId)
                insertedDict[movie.getObjId()] = stepId
        return deps

    # --------------------------- STEPS functions ------------------------------

    def _processMovie(self, movie):
        movieId = movie.getObjId()
        fnMovie = movie.getFileName()

        dim, _, numMov = self.inputMovies.get().getDim()
        samplingRate = self.inputMovies.get().getSamplingRate()
        if self.resizeOption==RESIZE_FACTOR:
            factor = self.resizeFactor.get()
            newDim = int(dim/factor)
            self.newSamplingRate =  samplingRate*factor
        if self.resizeOption==RESIZE_DIMENSIONS:
            newDim = self.resizeDim.get()
            self.newSamplingRate = float(samplingRate)*(float(dim)/float(
                newDim))
        if self.resizeOption==RESIZE_SAMPLINGRATE:
            self.newSamplingRate = self.resizeSamplingRate.get()
            factor = self.newSamplingRate/samplingRate
            newDim = int(dim/factor)

        args = "-i %s -o %s --fourier %d %d %d" % \
               (fnMovie, self._getPath("movie_%06d_resize.mrcs" % movieId),
                newDim, newDim, numMov)

        self.runJob("xmipp_image_resize", args, numberOfMpi=1)

    # -------------- Methods for Streaming --------------------

    def createOutputStep(self):
        # Do nothing now, the output should be ready.
        pass

    def _checkNewOutput(self):
        if getattr(self, 'finished', False):
            return

        # Load previously done items (from text file)
        doneList = self._readDoneList()
        # Check for newly done items
        newDone = [m.clone() for m in self.listOfMovies
                   if int(m.getObjId()) not in doneList
                   and self._isMovieDone(m)]

        allDone = len(doneList) + len(newDone)
        # We have finished when there is not more input movies (stream closed)
        # and the number of processed movies is equal to the number of inputs
        self.finished = self.streamClosed and allDone == len(self.listOfMovies)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        if newDone:
            self._writeDoneList(newDone)
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        imageSet = self._loadOutputSet(SetOfMovies, 'movies.sqlite')

        for movie in newDone:
            imgOut = Movie()
            imgOut.setObjId(movie.getObjId())
            imgOut.setFileName(self._getPath(
                "movie_%06d_resize.mrcs" % movie.getObjId()))
            imgOut.setAcquisition(movie.getAcquisition())
            imgOut.setSamplingRate(self.newSamplingRate)
            imgOut.setFramesRange(self.inputMovies.get().getFramesRange())
            
            if imageSet.isEmpty():
                imageSet.setDim(imgOut.getDim())
            
            imageSet.append(imgOut)

        self._updateOutputSet('outputMovies', imageSet, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(cons.STATUS_NEW)

    # --------------------------- UTILS functions ------------------------------

    def _loadOutputSet(self, SetClass, baseName):
        """
        Load the output set if it exists or create a new one.
        """
        setFile = self._getPath(baseName)
        if exists(setFile):
            outputSet = SetClass(filename=setFile)
            outputSet.loadAllProperties()
            outputSet.enableAppend()
        else:
            inputMovies = self.inputMovies.get()
            outputSet = SetClass(filename=setFile)
            outputSet.setStreamState(outputSet.STREAM_OPEN)
            outputSet.copyInfo(inputMovies)
            outputSet.setSamplingRate(self.newSamplingRate)

        return outputSet

    def _updateOutputSet(self, outputName, outputSet, state=Set.STREAM_OPEN):
        outputSet.setStreamState(state)

        if self.hasAttribute(outputName):
            outputSet.write()  # Write to commit changes
            outputAttr = getattr(self, outputName)
            # Copy the properties to the object contained in the protcol
            outputAttr.copy(outputSet, copyId=False)
            # Persist changes
            self._store(outputAttr)
        else:
            # Here the defineOutputs function will call the write() method
            self._defineOutputs(**{outputName: outputSet})
            self._store(outputSet)

        # Close set database to avoid locking it
        outputSet.close()

# --------------------------- INFO functions -------------------------------
    def _summary(self):
        if not hasattr(self, 'outputMovies'):
            summary = ["No summary information yet."]
        else:
            xIn, yIn, zIn = self.inputMovies.get().getDim()
            xOut, yOut, zOut = self.outputMovies.getDim()
            summary = ["%d input movies of size %dx%dx%d resized to %d "
                       "output movies of size %dx%dx%d."
                       %(self.inputMovies.get().getSize(), xIn, yIn, zIn,
                       self.outputMovies.getSize(), xOut, yOut, zOut)]
        return summary

    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        dim, _, numMov = self.inputMovies.get().getDim()
        samplingRate = self.inputMovies.get().getSamplingRate()
        if self.resizeOption == RESIZE_FACTOR:
            if self.resizeFactor.get()<1.0:
                errors.append('Please provide a resizeFactor higher than 1, '
                              'only downsampling is allowed.')
        if self.resizeOption == RESIZE_DIMENSIONS:
            if self.resizeDim.get() > dim:
                errors.append('Please provide a resizeDim higher than the '
                              'size of the input set, only downsampling is '
                              'allowed.')
        if self.resizeOption == RESIZE_SAMPLINGRATE:
            if self.resizeSamplingRate.get() <= samplingRate:
                errors.append('Please provide a resizeSamplingRate higher '
                              'than the original one, only downsampling is '
                              'allowed.')
        return errors