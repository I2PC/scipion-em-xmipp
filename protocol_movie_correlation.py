# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Vahid Abrishami (vabrishami@cnb.csic.es)
# *              Josue Gomez Blanco (jgomez@cnb.csic.es)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
Protocol wrapper around the xmipp correlation alignment for movie alignment
"""

import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
from pyworkflow.em.protocol import ProtAlignMovies
import pyworkflow.em.metadata as md
from convert import getMovieFileName



class XmippProtMovieCorr(ProtAlignMovies):
    """
    Wrapper protocol to Xmipp Movie Alignment by cross-correlation
    """
    _label = 'correlation alignment'

    #--------------------------- DEFINE param functions --------------------------------------------

    def _defineAlignmentParams(self, form):
        ProtAlignMovies._defineAlignmentParams(self, form)

        form.addParam('splineOrder', params.IntParam, default=3,
                      expertLevel=cons.LEVEL_ADVANCED,
                      label='B-spline order',
                      help="1 for linear interpolation (faster but lower quality) "
                           "3 for cubic interpolation (slower but more accurate).")

        form.addParam('maxFreq', params.FloatParam, default=4,
                       label='Filter at (A)',
                       help="For the calculation of the shifts with Xmipp, micrographs are "
                            "filtered (and downsized accordingly) to this resolution. "
                            "Then shifts are calculated, and they are applied to the "
                            "original frames without any filtering and downsampling.")

        form.addParam('maxShift', params.IntParam, default=30,
                      expertLevel=cons.LEVEL_ADVANCED,
                      label="Maximum shift (pixels)",
                      help='Maximum allowed distance (in pixels) that each frame '
                           'can be shifted with respect to the next.')
        
        form.addParallelSection(threads=1, mpi=1)
    
    #--------------------------- STEPS functions ---------------------------------------------------

    def _processMovie(self, movie):
        inputMd = getMovieFileName(movie)
        x, y, n = movie.getDim()
        a0, aN = self._getFrameRange(n, 'align')
        s0, sN = self._getFrameRange(n, 'sum')

        args  = '-i %s ' % inputMd
        args += '-o %s ' % self._getShiftsFile(movie)
        args += '--sampling %f ' % movie.getSamplingRate()
        args += '--max_freq %f ' % self.maxFreq

        if self.binFactor > 1:
            args += '--bin %f ' % self.binFactor
        # Assume that if you provide one cropDim, you provide all
        if self.cropDimX.get():
            args += '--cropULCorner %d %d ' % (self.cropOffsetX, self.cropOffsetY)
            args += '--cropDRCorner %d %d ' % (self.cropOffsetX.get() + self.cropDimX.get() -1,
                                                  self.cropOffsetY.get() + self.cropDimY.get() -1)
        
        args += ' --frameRange %d %d ' % (a0-1, aN-1)
        args += ' --frameRangeSum %d %d ' % (s0-1, sN-1)
        args += ' --max_shift %d ' % self.maxShift

        if self.doSaveAveMic:
            args += ' --oavg %s' % self._getExtraPath(self._getOutputMicName(movie))

        if self.doSaveMovie:
            args += ' --oaligned %s' % self._getExtraPath(self._getOutputMovieName(movie))

        if self.inputMovies.get().getDark():
            args += ' --dark ' + self.inputMovies.get().getDark()

        if self.inputMovies.get().getGain():
            args += ' --gain ' + self.inputMovies.get().getGain()

        self.runJob('xmipp_movie_alignment_correlation', args, numberOfMpi=1)
        
    #--------------------------- INFO functions --------------------------------------------

    def _summary(self):
        summary = []
        return summary
    
    def _validate(self):
        errors = []
        return errors
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _getShiftsFile(self, movie):
        return self._getExtraPath(self._getMovieRoot(movie) + '_shifts.xmd')

    def _getMovieShifts(self, movie):
        """ Returns the x and y shifts for the alignment of this movie.
         The shifts should refer to the original micrograph without any binning.
         In case of a bining greater than 1, the shifts should be scaled.
        """
        shiftsMd = md.MetaData(self._getShiftsFile(movie))
        shiftsMd.removeDisabled()

        return (shiftsMd.getColumnValues(md.MDL_SHIFT_X),
                shiftsMd.getColumnValues(md.MDL_SHIFT_Y))

