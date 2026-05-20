# **************************************************************************
# *
# * Authors:     Jose Luis Vilas Prieto (jlvilas@cnb.csic.es)
# *              Eduardo García Delgado (eduardo.garcia@cnb.csic.es)
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

import os, shutil
from os.path import basename
import pyworkflow.utils as pwutils
from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, BooleanParam, EnumParam)

from pwem.protocols import ProtPreprocessMicrographs

from pwem.objects import Movie, Micrograph
from pyworkflow import UPDATED, PROD


class XmippProtSplitFrames(ProtPreprocessMicrographs):
    """
    Wrapper protocol for Xmipp Split Odd Even. It separates movie frames into
    two outputs based on whether their index is odd or even. This allows for
    independent processing of each group, helping to reduce noise and artifacts
    in cryo-EM data analysis.

    AI Generated

    ## Overview

    The Split Frames protocol separates the frames of each input movie into two
    independent subsets: one containing the odd-numbered frames and the other
    containing the even-numbered frames.

    This type of split is useful when the user wants to create two related but
    partially independent versions of the same acquisition. Odd and even frame
    subsets can be processed separately for validation, comparison, noise
    assessment, or diagnostic purposes.

    The protocol can produce either two output movie sets, preserving the odd and
    even frames as movies, or two output micrograph sets, where the odd and even
    frames have been summed into separate micrographs.

    ## Inputs and General Workflow

    The main input is a set of movies.

    For each movie, the protocol separates the frame sequence into two groups:
    odd frames and even frames. It first writes temporary metadata files describing
    the odd and even frame selections, then converts those selections into MRC
    stacks or summed micrographs.

    The resulting files are organized into two output folders:

    - one folder for odd-frame outputs;
    - one folder for even-frame outputs.

    Finally, the protocol creates Scipion output sets pointing to those files.

    ## Input Movies

    The **Input movies** parameter defines the movie set whose frames will be
    split.

    Each movie is processed independently. The protocol preserves the relationship
    between each original movie and its odd/even outputs. The output items inherit
    the relevant metadata from the original movie, such as sampling and acquisition
    information.

    The protocol is intended for movie data. It does not perform movie alignment,
    dose weighting, CTF estimation, or particle processing. Its purpose is only to
    split the frames into two complementary subsets.

    ## Type of Set

    The **Type of Set** parameter controls the type of output produced by the
    protocol.

    If **Movies** is selected, the protocol creates two output movie sets:

    - **oddMovie**, containing movies made from the odd frames;
    - **evenMovie**, containing movies made from the even frames.

    If **Micrographs** is selected, the protocol sums the odd and even frames
    separately and creates two output micrograph sets:

    - **oddMicrograph**, containing summed micrographs from odd frames;
    - **evenMicrograph**, containing summed micrographs from even frames.

    The choice depends on the intended downstream analysis.

    Use **Movies** if the odd and even frame subsets should still be processed as
    movies, for example by a movie-alignment or dose-dependent analysis protocol.

    Use **Micrographs** if the goal is to obtain two static images per original
    movie, one from odd frames and one from even frames.

    ## Odd and Even Movie Outputs

    When the output type is **Movies**, each input movie produces two new movies.

    The odd movie contains the frames with odd indices. The even movie contains the
    frames with even indices. These outputs are stored as MRC stacks.

    The protocol also adjusts the frame range metadata of the output movies to
    reflect the reduced number of frames. If the original movie has an even number
    of frames, the odd and even outputs contain the same number of frames. If the
    original movie has an odd number of frames, one subset contains one more frame
    than the other.

    These outputs can be useful when the user wants to run equivalent downstream
    processing on two interleaved subsets of the same acquisition.

    ## Odd and Even Micrograph Outputs

    When the output type is **Micrographs**, the protocol sums the odd frames and
    even frames separately.

    For each input movie, this produces two micrographs:

    - one micrograph from the odd frames;
    - one micrograph from the even frames.

    These micrographs are useful for direct comparison of two independent-looking
    sums from the same exposure. They can help reveal artifacts, frame-dependent
    instabilities, or differences between interleaved frame subsets.

    Because the odd and even micrographs are generated from alternating frames,
    they share the same specimen, acquisition geometry, and general imaging
    conditions, but they are not identical sums.

    ## Why Split Odd and Even Frames?

    Odd/even splitting is a simple way to create two related subsets of the same
    movie data.

    This can be useful for several purposes:

    - checking whether signal is reproducible between frame subsets;
    - comparing independently processed versions of the same movie;
    - estimating noise or variability;
    - testing whether artifacts appear in both subsets or only in one;
    - creating half-data inputs for validation workflows.

    The split is not the same as dividing the movie into early and late frames.
    Odd and even subsets are interleaved across the exposure, so both subsets
    sample the full time range of the movie.

    This makes them more comparable than a first-half/second-half split when the
    goal is to create two similar subsets.

    ## Interpretation of the Outputs

    The odd and even outputs should be interpreted as complementary views of the
    same movie acquisition.

    They are not independent experimental acquisitions, because they come from the
    same exposure. However, they are useful because they are built from different
    frames. Differences between them may reveal noise, instability, or
    frame-dependent artifacts.

    If the odd and even outputs are very similar after appropriate processing, this
    suggests that the signal is stable across frames. If they differ strongly, the
    user may need to investigate movie quality, radiation damage, drift, detector
    artifacts, or other acquisition problems.

    ## Practical Recommendations

    Use **Movies** as output if the odd and even subsets should undergo further
    movie-level processing, such as alignment.

    Use **Micrographs** as output if the goal is quick visual comparison or
    micrograph-level downstream analysis.

    Remember that odd and even subsets contain fewer frames than the original
    movie. Therefore, each subset has lower total signal than the full movie.

    When comparing odd and even outputs, focus on reproducibility of the main
    features rather than expecting them to be identical. Some differences are
    expected because each output uses only part of the dose.

    If the original movies have an odd number of frames, the odd and even subsets
    will not contain exactly the same number of frames. This should be considered
    when interpreting small intensity or noise differences.

    ## Final Perspective

    Split Frames is a simple but useful protocol for generating odd and even
    subsets from cryo-EM movies.

    For biological users, its main value is diagnostic and validation-oriented. It
    allows the same acquisition to be divided into two interleaved frame subsets,
    which can then be compared or processed separately.

    The protocol does not improve the data by itself, but it creates useful paired
    outputs for checking consistency, exploring noise behavior, and supporting
    quality-control workflows.
    """
    _label = 'split frames'
    _lastUpdateVersion = VERSION_1_1
    _devStatus = PROD

    # --------------------------- DEFINE param functions ------------------------

    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label="Input movies", important=True,
                      help='Select a set of movies to be split into two sets (odd and even).'
                           'It means, the set of frames splits into two subsets.')
        form.addParam('set', EnumParam, choices=['Movies', 'Micrographs'],
                      label="Type of Set", important=False, default=0,
                      help='Set Movies to get a set of movies, or set Micrographs to get a set of micrographs.')

    # --------------------------- STEPS functions -------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep(self.splittingStep)
        self._insertFunctionStep(self.convertXmdToStackStep)
        self._insertFunctionStep(self.separateOddEvenOutputs)
        self._insertFunctionStep(self.createOutputStep)

    def splittingStep(self):

        inSet = self.inputMovies.get()

        for movie in inSet:

            fnMovie = movie.getFileName()
            if fnMovie.endswith(".mrc"):
                fnMovie += ":mrcs"

            fnMovieOdd = pwutils.removeExt(basename(fnMovie)) + "_odd.xmd"
            fnMovieEven = pwutils.removeExt(basename(fnMovie)) + "_even.xmd"

            args = '--img "%s" ' % fnMovie
            args += '-o "%s" ' % self._getTmpPath(fnMovieOdd)
            args += '-e %s ' % self._getTmpPath(fnMovieEven)
            args += '--type frames '
            if self.set.get() == 1:
                args += '--sum_frames '

            self.runJob('xmipp_image_odd_even', args)

    def convertXmdToStackStep(self):

        inSet = self.inputMovies.get()

        for movie in inSet:
            fnMovie = movie.getFileName()

            fnMovieOdd = pwutils.removeExt(basename(fnMovie)) + "_odd.xmd"
            fnMovieEven = pwutils.removeExt(basename(fnMovie)) + "_even.xmd"

            newfnMovieOdd = pwutils.removeExt(basename(fnMovieOdd)) + ".mrcs"
            newfnMovieEven = pwutils.removeExt(basename(fnMovieEven)) + ".mrcs"

            args = '-i "%s" ' % self._getTmpPath(fnMovieOdd)
            args += '-o "%s" ' % self._getExtraPath(newfnMovieOdd)

            self.runJob('xmipp_image_convert', args)

            args = '-i "%s" ' % self._getTmpPath(fnMovieEven)
            args += '-o "%s" ' % self._getExtraPath(newfnMovieEven)

            self.runJob('xmipp_image_convert', args)

    def separateOddEvenOutputs(self):

        inSet = self.inputMovies.get()

        oddDir = os.path.join(self._getExtraPath(), 'oddFrames')
        evenDir = os.path.join(self._getExtraPath(), 'evenFrames')

        os.makedirs(oddDir, exist_ok=True)
        os.makedirs(evenDir, exist_ok=True)

        for movie in inSet:

            fnMovie = movie.getFileName()

            if self.set.get() == 1:

                fnMicOdd = self._getTmpPath(pwutils.removeExt(basename(fnMovie)) + "_odd_aligned.mrc")
                shutil.move(fnMicOdd, os.path.join(oddDir, pwutils.removeExt(basename(fnMovie))
                                                   + "_aligned.mrc"))

                fnMicEven = self._getTmpPath(pwutils.removeExt(basename(fnMovie)) + "_even_aligned.mrc")
                shutil.move(fnMicEven, os.path.join(evenDir, pwutils.removeExt(basename(fnMovie))
                                                    + "_aligned.mrc"))

            else:

                fnMovieOdd = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_odd.mrcs")
                shutil.move(fnMovieOdd, os.path.join(oddDir, pwutils.removeExt(basename(fnMovie))
                                                     + ".mrcs"))

                fnMovieEven = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_even.mrcs")
                shutil.move(fnMovieEven, os.path.join(evenDir, pwutils.removeExt(basename(fnMovie))
                                                      + ".mrcs"))

    def createOutputStep(self):

        inSet = self.inputMovies.get()

        if self.set.get() == 1:

            oddSetMic = self._createSetOfMicrographs(suffix='oddMic')
            evenSetMic = self._createSetOfMicrographs(suffix='evenMic')

            oddSetMic.copyInfo(inSet)
            evenSetMic.copyInfo(inSet)

            oddDir = os.path.join(self._getExtraPath(), 'oddFrames')
            evenDir = os.path.join(self._getExtraPath(), 'evenFrames')

            for movie in inSet:
                fnMovie = movie.getFileName()

                fnMicOdd = os.path.join(oddDir, pwutils.removeExt(basename(fnMovie))
                                        + "_aligned.mrc")
                fnMicEven = os.path.join(evenDir, pwutils.removeExt(basename(fnMovie))
                                         + "_aligned.mrc")

                imgOutOdd = Micrograph()
                imgOutEven = Micrograph()

                imgOutOdd.setFileName(fnMicOdd)
                imgOutEven.setFileName(fnMicEven)

                imgOutOdd.copyInfo(movie)
                imgOutEven.copyInfo(movie)

                oddSetMic.append(imgOutOdd)
                evenSetMic.append(imgOutEven)

            self._defineOutputs(oddMicrograph=oddSetMic)
            self._defineOutputs(evenMicrograph=evenSetMic)

            self._defineSourceRelation(inSet, oddSetMic)
            self._defineSourceRelation(inSet, evenSetMic)

        else:

            oddSet = self._createSetOfMovies(suffix='odd')
            evenSet = self._createSetOfMovies(suffix='even')

            oddSet.copyInfo(inSet)
            evenSet.copyInfo(inSet)

            oddDir = os.path.join(self._getExtraPath(), 'oddFrames')
            evenDir = os.path.join(self._getExtraPath(), 'evenFrames')

            for movie in inSet:

                fnMovie = movie.getFileName()

                fnMovieOdd = os.path.join(oddDir, pwutils.removeExt(basename(fnMovie)) + ".mrcs")
                fnMovieEven = os.path.join(evenDir, pwutils.removeExt(basename(fnMovie)) + ".mrcs")

                imgOutOdd = Movie()
                imgOutEven = Movie()

                imgOutOdd.setFileName(fnMovieOdd)
                imgOutEven.setFileName(fnMovieEven)

                imgOutOdd.copyInfo(movie)
                imgOutEven.copyInfo(movie)

                if movie.getNumberOfFrames() % 2 == 0:
                    imgOutOdd.setFramesRange([1, imgOutOdd.getNumberOfFrames() / 2, 1])
                    imgOutEven.setFramesRange([1, imgOutEven.getNumberOfFrames() / 2, 1])
                else:
                    imgOutOdd.setFramesRange([1, round(imgOutOdd.getNumberOfFrames() / 2), 1])
                    imgOutEven.setFramesRange([1, round(imgOutEven.getNumberOfFrames() / 2) - 1, 1])

                oddSet.append(imgOutOdd)
                evenSet.append(imgOutEven)

            oddSet.setFramesRange([1, oddSet.getFirstItem().getNumberOfFrames(), 1])
            evenSet.setFramesRange([1, evenSet.getFirstItem().getNumberOfFrames(), 1])

            self._defineOutputs(oddMovie=oddSet)
            self._defineOutputs(evenMovie=evenSet)

            self._defineSourceRelation(inSet, oddSet)
            self._defineSourceRelation(inSet, evenSet)

    # --------------------------- INFO functions ------------------------------

    def _summary(self):

        message = []

        inSet = self.inputMovies.get()

        if self.set.get() == 1 and (hasattr(self, "oddMicrograph") or hasattr(self, "evenMicrograph")):

            oddMicrographs = getattr(self, "oddMicrograph", None)
            evenMicrographs = getattr(self, "evenMicrograph", None)

            message.append("%d/%d Odd Micrographs processed."
                           % (oddMicrographs.getSize(), inSet.getSize()))
            message.append("%d/%d Even Micrographs processed."
                           % (evenMicrographs.getSize(), inSet.getSize()))

        elif self.set.get() == 0 and (hasattr(self, "oddMovie") or hasattr(self, "evenMovie")):

            oddMovies = getattr(self, "oddMovie")
            evenMovies = getattr(self, "evenMovie")

            message.append("%d/%d Movies with %d Odd Frames processed."
                           % (oddMovies.getSize(), inSet.getSize(),
                              oddMovies.getFirstItem().getNumberOfFrames()))
            message.append("%d/%d Movies with %d Even Frames processed."
                           % (evenMovies.getSize(), inSet.getSize(),
                              evenMovies.getFirstItem().getNumberOfFrames()))

        return message