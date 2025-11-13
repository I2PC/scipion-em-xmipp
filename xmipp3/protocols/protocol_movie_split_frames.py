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
    Wrapper protocol for Xmipp Split Odd Even. It separates movie frames into two outputs based on whether their index is odd or even. This allows for independent processing of each group, helping to reduce noise and artifacts in cryo-EM data analysis.
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