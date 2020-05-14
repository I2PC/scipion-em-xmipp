# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Vahid Abrishami (vabrishami@cnb.csic.es)
# *              Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *              David Strelak (davidstrelak@gmail.com)
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
from math import ceil

import pyworkflow.utils as pwutils
from pyworkflow.utils import yellowStr
import pyworkflow.object as pwobj
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
import pwem.emlib.metadata as md
from pwem.objects import Image
from pwem.protocols.protocol_align_movies import createAlignmentPlot
from pyworkflow import VERSION_1_1
from pwem.protocols import ProtAlignMovies

from xmipp3.convert import writeMovieMd


class XmippProtMovieCorr(ProtAlignMovies):
    """
    Wrapper protocol to Xmipp Movie Alignment by cross-correlation
    """
    OUTSIDE_WRAP = 0
    OUTSIDE_AVG = 1
    OUTSIDE_VALUE = 2

    INTERP_LINEAR = 0
    INTERP_CUBIC = 1

    # Map to xmipp interpolation values in command line
    INTERP_MAP = {INTERP_LINEAR: 1, INTERP_CUBIC: 3}

    _label = 'correlation alignment'
    _lastUpdateVersion = VERSION_1_1

    def __init__(self, **args):
        ProtAlignMovies.__init__(self, **args)
        self.stepsExecutionMode = cons.STEPS_PARALLEL

    #--------------------------- DEFINE param functions ------------------------

    def _defineAlignmentParams(self, form):
        ProtAlignMovies._defineAlignmentParams(self, form)

        form.addParam('splineOrder', params.EnumParam, condition="doSaveAveMic or doSaveMovie",
                      default=self.INTERP_CUBIC, choices=['linear', 'cubic'],
                      expertLevel=cons.LEVEL_ADVANCED,
                      label='Interpolation',
                      help="linear (faster but lower quality), "
                           "cubic (slower but more accurate).")

        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=cons.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addParam('maxResForCorrelation', params.FloatParam, default=30,
                       label='Maximal resolution (A)',
                       help="Maximal resolution in A that will be preserved during correlation.")

        form.addParam('doComputePSD', params.BooleanParam, default=True,
                      label="Compute PSD (before/after)?",
                      help="If Yes, the protocol will compute for each movie "
                           "the PSD of the average micrograph (without CC "
                           "alignement) and after that, to compare each PSDs")

        form.addParam('maxShift', params.IntParam, default=30,
                      expertLevel=cons.LEVEL_ADVANCED,
                      label="Maximum shift (pixels)",
                      help='Maximum allowed distance (in pixels) that each '
                           'frame can be shifted with respect to the next.')

        form.addParam('outsideMode', params.EnumParam,
                      choices=['Wrapping','Average','Value'],
                      default=self.OUTSIDE_WRAP,
                      expertLevel=cons.LEVEL_ADVANCED,
                      label="How to fill borders",
                      help='How to fill the borders when shifting the frames')

        #Local alignment params
        group = form.addGroup('Local alignment')

        group.addParam('doLocalAlignment', params.BooleanParam, default=True,
                      label="Compute local alignment?",
                      help="If Yes, the protocol will try to determine local shifts, similarly to MotionCor2.")

        group.addParam('autoControlPoints', params.BooleanParam, default=True,
                      label="Auto control points",
                      expertLevel=cons.LEVEL_ADVANCED,
                      condition='doLocalAlignment',
                      help="If on, protocol will automatically determine necessary number of control points.")
        group.addParam('minLocalRes', params.FloatParam, default=500,
                       expertLevel=cons.LEVEL_ADVANCED,
                       label='Min size of the patch (A)',
                       help="How many A should contain each patch?")

        group.addParam('skipAutotuning', params.BooleanParam, default=False,
                       expertLevel=cons.LEVEL_ADVANCED,
                       label='Skip autotuning',
                       help="We try to faster settings of for the FFT library. This takes some time, "
                            "but consecutive executions will be faster and use less memory."
                            "Set to True (autotuning will be turned off) if you process just few movies")

        line = group.addLine('Number of control points',
                    expertLevel=cons.LEVEL_ADVANCED,
                    help='Number of control points use for BSpline.',
                    condition='not autoControlPoints')
        line.addParam('controlPointX', params.IntParam, default=6, label='X')
        line.addParam('controlPointY', params.IntParam, default=6, label='Y')
        line.addParam('controlPointT', params.IntParam, default=5, label='t')

        group.addParam('groupNFrames', params.IntParam, default=3,
                    expertLevel=cons.LEVEL_ADVANCED,
                    label='Group N frames',
                    help='Group every specified number of frames by adding them together. \
                        The alignment is then performed on the summed frames.',
                    condition='doLocalAlignment')

        form.addParam('outsideValue', params.FloatParam, default=0.0,
                       expertLevel=cons.LEVEL_ADVANCED,
                       condition="outsideMode==2",
                       label='Fill value',
                       help="Fixed value for filling borders")

        form.addParallelSection(threads=1, mpi=1)

    #--------------------------- STEPS functions -------------------------------
    def _processMovie(self, movie):
        try:
            self.tryProcessMovie(movie)
        except Exception as ex:
            print(yellowStr("We cannot process %s" % movie.getFileName()))
            print(ex)

    def tryProcessMovie(self, movie):
        movieFolder = self._getOutputMovieFolder(movie)

        x, y, n = movie.getDim()
        a0, aN = self._getFrameRange(n, 'align')
        s0, sN = self._getFrameRange(n, 'sum')

        inputMd = os.path.join(movieFolder, 'input_movie.xmd')
        writeMovieMd(movie, inputMd, a0, aN, useAlignment=False)

        args = '-i "%s" ' % inputMd
        args += '-o "%s" ' % self._getShiftsFile(movie)
        args += '--sampling %f ' % movie.getSamplingRate()
        args += '--maxResForCorrelation %f ' % self.maxResForCorrelation
        args += '--Bspline %d ' % self.INTERP_MAP[self.splineOrder.get()]

        if self.binFactor > 1:
            args += '--bin %f ' % self.binFactor
        # Assume that if you provide one cropDim, you provide all

        offsetX = self.cropOffsetX.get()
        offsetY = self.cropOffsetY.get()
        cropDimX = self.cropDimX.get()
        cropDimY = self.cropDimY.get()

        args += '--cropULCorner %d %d ' % (offsetX, offsetY)

        if cropDimX <= 0:
            dimX = x - 1
        else:
            dimX = offsetX + cropDimX - 1

        if cropDimY <= 0:
            dimY = y - 1
        else:
            dimY = offsetY + cropDimY - 1

        args += '--cropDRCorner %d %d ' % (dimX, dimY)

        if self.outsideMode == self.OUTSIDE_WRAP:
            args += "--outside wrap"
        elif self.outsideMode == self.OUTSIDE_AVG:
            args += "--outside avg"
        elif self.outsideMode == self.OUTSIDE_AVG:
            args += "--outside value %f" % self.outsideValue

        args += ' --frameRange %d %d ' % (0, aN-a0)
        args += ' --frameRangeSum %d %d ' % (s0-a0, sN-a0)
        args += ' --max_shift %d ' % self.maxShift

        if self.doSaveAveMic or self.doComputePSD:
            fnAvg = self._getExtraPath(self._getOutputMicName(movie))
            args += ' --oavg "%s"' % fnAvg

        if self.doComputePSD:
            fnInitial = os.path.join(movieFolder, "initialMic.mrc")
            args  += ' --oavgInitial %s' % fnInitial

        if self.doSaveMovie:
            args += ' --oaligned %s' % self._getExtraPath(self._getOutputMovieName(movie))

        if self.inputMovies.get().getDark():
            args += ' --dark ' + self.inputMovies.get().getDark()

        if self.inputMovies.get().getGain():
            args += ' --gain ' + self.inputMovies.get().getGain()

        if self.autoControlPoints.get():
            self._setControlPoints()

        if self.minLocalRes.get():
            args += ' --minLocalRes %f' % self.minLocalRes

        if self.useGpu.get():
            args += ' --device %(GPU)s'
            if self.doLocalAlignment.get():
                args += ' --processLocalShifts '
            if self.skipAutotuning.get():
                args += " --skipAutotuning"
            args += ' --storage ' + self._getExtraPath("fftBenchmark.txt")
            args += ' --controlPoints %d %d %d' % (self.controlPointX, self.controlPointY, self.controlPointT)
            args += ' --patchesAvg %d' % self.groupNFrames
            self.runJob('xmipp_cuda_movie_alignment_correlation', args, numberOfMpi=1)
        else:
            self.runJob('xmipp_movie_alignment_correlation', args, numberOfMpi=1)

        if self.doComputePSD:
            self.computePSDImages(movie, fnInitial, fnAvg)
            # If the micrograph was only saved for computing the PSD
            # we can remove it
            pwutils.cleanPath(fnInitial)
            if not self.doSaveAveMic:
                pwutils.cleanPath(fnAvg)

        self._saveAlignmentPlots(movie)

    #--------------------------- UTILS functions ------------------------------
    def _getShiftsFile(self, movie):
        return self._getExtraPath(self._getMovieRoot(movie) + '_shifts.xmd')

    def _stepsCheck(self):
        # Input movie set can be loaded or None when checked for new inputs
        # If None, we load it
        self._checkNewInput()
        self._checkNewOutput()
        self.inputMovies.get().close()

    def _setControlPoints(self):
            x, y,frames = self.inputMovies.get().getDim()
            Ts = self.inputMovies.get().getSamplingRate()
            # one control point each 1000 A
            self.controlPointX.set(int(x * Ts) / 1000 + 2)
            self.controlPointY.set(int(y * Ts) / 1000 + 2)
            self.controlPointT.set(ceil(frames/7.) + 2)

    def _getMovieShifts(self, movie):
        from ..convert import readShiftsMovieAlignment
        """ Returns the x and y shifts for the alignment of this movie.
         The shifts should refer to the original micrograph without any binning.
         In case of a bining greater than 1, the shifts should be scaled.
        """
        shiftsMd = md.MetaData("frameShifts@" + self._getShiftsFile(movie))
        return readShiftsMovieAlignment(shiftsMd)

    def _storeSummary(self, movie):
        if self.doSaveAveMic and movie.hasAlignment():
            s0, sN = self._getFrameRange(movie.getNumberOfFrames(), 'sum')
            fstFrame, lstFrame = movie.getAlignment().getRange()
            if fstFrame > s0 or lstFrame < sN:
                self.summaryVar.set("Warning!!! You have selected a frame range "
                                    "wider than the range selected to align. All "
                                    "the frames selected without alignment "
                                    "information, will be aligned by setting "
                                    "alignment to 0")

    def _loadMeanShifts(self, movie):
        alignMd = md.MetaData("frameShifts@" + self._getShiftsFile(movie))
        meanX = alignMd.getColumnValues(md.MDL_SHIFT_X)
        meanY = alignMd.getColumnValues(md.MDL_SHIFT_Y)

        return meanX, meanY

    def _getPlotCart(self,movie):
        return self._getExtraPath(self._getMovieRoot(movie)+"_plot_cart.png")

    def _getPsdCorr(self,movie):
        return self._getExtraPath(self._getMovieRoot(movie)+"_aligned_corrected.psd")

    def _saveAlignmentPlots(self, movie):
        """ Compute alignment shifts plot and save to file as a png image. """
        meanX, meanY = self._loadMeanShifts(movie)
        plotter = createAlignmentPlot(meanX, meanY)
        plotter.savefig(self._getPlotCart(movie))
        plotter.close()

    def _setAlignmentInfo(self, movie, obj):
        """ Set alignment info such as plot and psd filename, and
        the cumulative shifts values.
        Params:
            movie: Pass the reference movie
            obj: should pass either the created micrograph or movie
        """
        obj.plotCart = Image()
        obj.plotCart.setFileName(self._getPlotCart(movie))
        if self.doComputePSD:
            obj.psdCorr = Image()
            obj.psdCorr.setFileName(self._getPsdCorr(movie))

        meanX, meanY = self._loadMeanShifts(movie)
        obj._xmipp_ShiftX = pwobj.CsvList()
        obj._xmipp_ShiftX.set(meanX)
        obj._xmipp_ShiftY = pwobj.CsvList()
        obj._xmipp_ShiftY.set(meanY)

    def _preprocessOutputMicrograph(self, mic, movie):
        self._setAlignmentInfo(movie, mic)

    def _createOutputMovie(self, movie):
        alignedMovie = ProtAlignMovies._createOutputMovie(self, movie)
        self._setAlignmentInfo(movie, alignedMovie)
        return alignedMovie

    def _validate(self):
        if self.autoControlPoints.get():
            self._setControlPoints() # make sure we work with proper values
        errors = ProtAlignMovies._validate(self)
        getXmippHome = self.getClassPackage().Plugin.getHome
        if self.doLocalAlignment.get():
            cudaBinaryFn = getXmippHome('bin', 'xmipp_cuda_movie_alignment_correlation')
            if not os.path.isfile(cudaBinaryFn):
                errors.append('GPU version not found, make sure that Xmipp is '
                              'compiled with GPU\n'
                              '( *CUDA=True* in _scipion.conf_ + '
                              '_run_: $ *scipion installb xmippSrc* ).')
                return errors
            elif not self.useGpu.get():
                errors.append("GPU is needed to do local alignment.")
                return errors

        nGpus = len(self.gpuList.get().split())
        nMPI = self.numberOfMpi.get()
        nThreads = self.numberOfThreads.get()
        validThreads = nGpus if nGpus == 1 else nGpus + 1
        if self.useGpu and nMPI * nThreads != validThreads:
            coreStr = ('threads' if nThreads > 1 else
                       'MPI' if nMPI > 1 else
                       'threads or MPI')
            errors.append("The number of threads/mpi must be one more than "
                          "the number of GPUs to use (single threading is also "
                          "valid for one single GPU)\n"
                          "i.e. *%s = %d* , when using *%d GPUs*"
                          % (coreStr, validThreads, nGpus))
        else:
            cpuBinaryFn = getXmippHome('bin', 'xmipp_movie_alignment_correlation')
            if not os.path.isfile(cpuBinaryFn):
                errors.append('CPU version not found for some reason, try to GPU=True.')
                return errors
        if (self.controlPointX < 3):
            errors.append("You have to use at least 3 control points in X dim")
            return errors # to avoid possible division by zero later
        if (self.controlPointY < 3):
            errors.append("You have to use at least 3 control points in Y dim")
            return errors # to avoid possible division by zero later
        if (self.controlPointT < 3):
            errors.append("You have to use at least 3 control points in T dim")
            return errors # to avoid possible division by zero later
        return errors
