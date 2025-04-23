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
from pyworkflow.protocol.params import (PointerParam, BooleanParam)

from pwem.protocols import ProtPreprocessMicrographs

from pwem.objects import Movie, Micrograph


class XmippProtSplitFrames(ProtPreprocessMicrographs):
    """
    Wrapper protocol to Xmipp split Odd Even. Applies the possibility of creating 2 different outputs of movies with
    a quantity of frames related to their odd or even nature (that is, if each frame represents an odd or even number).
    """
    _label = 'split frames'
    _lastUpdateVersion = VERSION_1_1

    #--------------------------- DEFINE param functions ------------------------

    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label="Input movies", important=True,
                      help='Select a set of movies to be split into two sets (odd and even).'
                      'It means, the set of frames splits into two subsets.')
        group = form.addGroup('Alignment')
        group.addParam('Micrographs', BooleanParam,
                       label="Micrographs Selection", important=False, default=False,
                       help='Set *Yes* to select a set of micrographs just in case alignment has been processed '
                            'in the input movies, so that they can also be split into two sets (odd  and even) of micrographs. '
                            'Set *No* if alignment has not been processed in the input movies.')

        group.addParam('inputMicrographs', PointerParam, pointerClass='SetOfMicrographs',
                      label="Input micrographs", condition="Micrographs",
                      help='Select a set of micrographs from the previous protocol.')

    #--------------------------- STEPS functions -------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep(self.movieProcessStep)
        self._insertFunctionStep(self.splittingStep)
        self._insertFunctionStep(self.convertXmdToStackStep)
        self._insertFunctionStep(self.separateOddEvenOutputs)
        self._insertFunctionStep(self.createOutputStep)

    def movieProcessStep(self):

        if self.inputMicrographs.get() is not None:
            self.info('Input set of movies incoming from a previous alignment, so its corresponding set of micrographs will be split.')
            self.alignment = 'T'
        else:
            self.info('No existing alignment process for input set of movies, so its corresponding set of micrographs will not be split.')
            self.alignment = 'F'

    def splittingStep(self):

        for movie in self.inputMovies.get():

            fnMovie = movie.getFileName()
            if fnMovie.endswith(".mrc"):
                fnMovie+=":mrcs"

            fnMovieOdd = pwutils.removeExt(basename(fnMovie)) + "_odd.xmd"
            fnMovieEven = pwutils.removeExt(basename(fnMovie)) + "_even.xmd"

            args = '--img "%s" ' % fnMovie
            args += '-o "%s" ' % self._getTmpPath(fnMovieOdd)
            args += '-e %s ' % self._getTmpPath(fnMovieEven)
            args += '--type frames '
            if self.alignment == 'T':
                args += '--sum_frames '

            self.runJob('xmipp_image_odd_even', args)

    def convertXmdToStackStep(self):

        for movie in self.inputMovies.get():

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

        oddDir = os.path.join(self._getExtraPath(), 'oddFrames')
        evenDir = os.path.join(self._getExtraPath(), 'evenFrames')

        os.makedirs(oddDir, exist_ok=True)
        os.makedirs(evenDir, exist_ok=True)

        for movie in self.inputMovies.get():

            fnMovie = movie.getFileName()

            if self.alignment == 'T':

                fnMicOdd = self._getTmpPath(pwutils.removeExt(basename(fnMovie)) + "_odd_aligned.mrc")
                shutil.move(fnMicOdd, os.path.join(oddDir, pwutils.removeExt(basename(fnMovie))
                                                     + "_aligned.mrc"))

                fnMicEven = self._getTmpPath(pwutils.removeExt(basename(fnMovie)) + "_even_aligned.mrc")
                shutil.move(fnMicEven, os.path.join(evenDir, pwutils.removeExt(basename(fnMovie))
                                                   + "_aligned.mrc"))

            else:

                fnMovieOdd = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_odd.mrcs")
                shutil.move(fnMovieOdd, os.path.join(oddDir,pwutils.removeExt(basename(fnMovie))
                                                         + ".mrcs"))

                fnMovieEven = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_even.mrcs")
                shutil.move(fnMovieEven, os.path.join(evenDir, pwutils.removeExt(basename(fnMovie))
                                                          + ".mrcs"))

    def createOutputStep(self):

        inSet = self.inputMovies.get()

        if self.alignment == 'T':

            oddSetMic = self._createSetOfMicrographs(suffix='oddMic')
            evenSetMic = self._createSetOfMicrographs(suffix='evenMic')

            oddSetMic.setSamplingRate(inSet.getSamplingRate())
            evenSetMic.setSamplingRate(inSet.getSamplingRate())

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

                imgOutOdd.setSamplingRate(movie.getSamplingRate())
                imgOutEven.setSamplingRate(movie.getSamplingRate())

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

        if self.alignment == 'T':

            oddMicrographs = getattr(self, "oddMicrograph")
            evenMicrographs = getattr(self, "evenMicrograph")

            message.append("%d/%d Odd Micrographs processed."
                           % (oddMicrographs.getSize(), inSet.getSize()))
            message.append("%d/%d Even Micrographs processed."
                           % (evenMicrographs.getSize(), inSet.getSize()))
        else:

            oddMovies = getattr(self, "oddMovie")
            evenMovies = getattr(self, "evenMovie")

            message.append("%d/%d Movies with %d Odd Frames processed."
                           % (oddMovies.getSize(), inSet.getSize(),
                              oddMovies.getFirstItem().getNumberOfFrames()))
            message.append("%d/%d Movies with %d Even Frames processed."
                           % (evenMovies.getSize(), inSet.getSize(),
                              evenMovies.getFirstItem().getNumberOfFrames()))

        return message