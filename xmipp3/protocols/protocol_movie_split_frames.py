# **************************************************************************
# *
# * Authors:     Jose Luis Vilas Prieto (jlvilas@cnb.csic.es)
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

from os.path import basename
import pyworkflow.utils as pwutils
from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, BooleanParam)

from pwem.protocols import ProtPreprocessMicrographs

from pwem.objects import Movie, Micrograph


class XmippProtSplitFrames(ProtPreprocessMicrographs):
    """
    Wrapper protocol to Xmipp split Odd Even
    """
    _label = 'split frames'
    _lastUpdateVersion = VERSION_1_1

    #--------------------------- DEFINE param functions ------------------------

    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMovies', PointerParam, pointerClass='SetOfMovies',
                      label="Input movies", important=True,
                      help='Select a set of movies to be split into two sets (odd and even).'
                      'It means, the set of frames is split in two subsets.')
        
        form.addParam('sumFrames', BooleanParam, 
                      label="Sum Frames", important=False,
                      help='Set yes to get a set of micrograms, or no to get a set of movies.')

    #--------------------------- STEPS functions -------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('splittingStep')
        self._insertFunctionStep('convertXmdToStackStep')
        self._insertFunctionStep('createOutputStep')

    def splittingStep(self):
        
        for movie in self.inputMovies.get():
            fnMovie = movie.getFileName()

            fnMovieOdd = pwutils.removeExt(basename(fnMovie)) + "_odd.xmd"
            fnMovieEven = pwutils.removeExt(basename(fnMovie)) + "_even.xmd"
            
            args = '--img "%s" ' % fnMovie
            args += '-o "%s" ' % self._getExtraPath(fnMovieOdd)
            args += '-e %s ' % self._getExtraPath(fnMovieEven)
            args += '--type frames '
            if (self.sumFrames.get() is True):
                args += '--sum_frames'
            
            self.runJob('xmipp_image_odd_even', args)

    def convertXmdToStackStep(self):

        for movie in self.inputMovies.get():
            fnMovie = movie.getFileName()

            fnMovieOdd = pwutils.removeExt(basename(fnMovie)) + "_odd.xmd"
            fnMovieEven = pwutils.removeExt(basename(fnMovie)) + "_even.xmd"

            fnMovieOddMrc = pwutils.removeExt(basename(fnMovieOdd)) + ".mrc"
            fnMovieEvenMrc = pwutils.removeExt(basename(fnMovieEven)) + ".mrc"

            args = '-i "%s" ' % self._getExtraPath(fnMovieOdd)
            args += '-o "%s" ' % self._getExtraPath(fnMovieOddMrc)

            self.runJob('xmipp_image_convert', args)

            args = '-i "%s" ' % self._getExtraPath(fnMovieEven)
            args += '-o "%s" ' % self._getExtraPath(fnMovieEvenMrc)

            self.runJob('xmipp_image_convert', args)


    def createOutputStep(self):

        oddSet = self._createSetOfMovies(suffix='odd')
        evenSet = self._createSetOfMovies(suffix='even')

        for movie in self.inputMovies.get():
            fnMovie = movie.getFileName()

            fnMovieOddMrc = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_odd.mrc")
            fnMovieEvenMrc = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_even.mrc")
            
            imgOutOdd = Movie()
            imgOutEven = Movie()
            
            imgOutOdd.setFileName(fnMovieOddMrc)
            imgOutEven.setFileName(fnMovieEvenMrc)
            
            imgOutOdd.setSamplingRate(movie.getSamplingRate())
            imgOutEven.setSamplingRate(movie.getSamplingRate())

            oddSet.append(imgOutOdd)
            evenSet.append(imgOutEven)

        oddSet.setSamplingRate(self.inputMovies.get().getSamplingRate())
        evenSet.setSamplingRate(self.inputMovies.get().getSamplingRate())

        self._defineOutputs(oddMovie=oddSet)
        self._defineOutputs(evenMovie=evenSet)
        
        self._defineSourceRelation(self.inputMovies, oddSet)
        self._defineSourceRelation(self.inputMovies, evenSet)
        
        if (self.sumFrames.get() is True):
            oddSetAligned = self._createSetOfMicrographs(suffix='oddMic')
            evenSetAligned = self._createSetOfMicrographs(suffix='evenMic')

            for movie in self.inputMovies.get():
            
                fnMovie = movie.getFileName()

                fnMicOdd = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_odd_aligned.mrc")
                fnMicEven = self._getExtraPath(pwutils.removeExt(basename(fnMovie)) + "_even_aligned.mrc")

                imgOutOdd = Micrograph()
                imgOutEven = Micrograph()
                
                imgOutOdd.setFileName(fnMicOdd)
                imgOutEven.setFileName(fnMicEven)

                imgOutOdd.setSamplingRate(movie.getSamplingRate())
                imgOutEven.setSamplingRate(movie.getSamplingRate())

                oddSetAligned.append(imgOutOdd)
                evenSetAligned.append(imgOutEven)

            oddSetAligned.copyInfo(self.inputMovies.get())
            evenSetAligned.copyInfo(self.inputMovies.get())

            oddSetAligned.setSamplingRate(self.inputMovies.get().getSamplingRate())
            evenSetAligned.setSamplingRate(self.inputMovies.get().getSamplingRate())

            self._defineOutputs(oddMicrographs=oddSetAligned)
            self._defineOutputs(evenMicrographs=evenSetAligned)

            self._defineSourceRelation(self.inputMovies, oddSetAligned)
            self._defineSourceRelation(self.inputMovies, evenSetAligned)
            


    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ')
        return messages
    
    def _summary(self):
        summary = []

        return summary

    def _citations(self):
        return ['']
