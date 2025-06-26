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

import os.path
import numpy as np
from math import ceil

import pyworkflow.utils as pwutils
from pyworkflow.utils import yellowStr
import pyworkflow.object as pwobj
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
import pwem.emlib.metadata as md
from pwem import emlib
from pwem.objects import Image, SetOfMovies
from pwem.protocols.protocol_align_movies import createAlignmentPlot
from pyworkflow import VERSION_1_1
from pwem.protocols import ProtAlignMovies

from xmipp3.convert import writeMovieMd, isEerMovie
from xmipp3.base import isXmippCudaPresent
import xmipp3.utils as xmutils
from pyworkflow import BETA, UPDATED, NEW, PROD

class XmippProtFlexAlign(ProtAlignMovies):
    """
    Wrapper protocol for Xmipp Movie Alignment using cross-correlation methods. It aligns movie frames to produce beam-induced motion corrected micrographs.
    """
    NO_ROTATION = 0
    NO_FLIP = 0
    _devStatus = PROD
    _label = 'FlexAlign'
    _lastUpdateVersion = VERSION_1_1

    def __init__(self, **args):
        ProtAlignMovies.__init__(self, **args)
        self.stepsExecutionMode = cons.STEPS_PARALLEL

    #--------------------------- DEFINE param functions ------------------------

    def _defineAlignmentParams(self, form):
        ProtAlignMovies._defineAlignmentParams(self, form)
        EER_CONDITION = 'inputMovies is not None and len(inputMovies) > 0 and next(iter(inputMovies.getFiles())).endswith(".eer")'
        form.addParam('nFrames', params.IntParam, label='Number of EER frames',
                      condition=EER_CONDITION, default=40, validators=[params.GT(0, "Number of EER frames must be a positive integer (> 0).")],
                      help='Number of frames to be generated. EER files contain subframes, that will be grouped into the selected number of frames.')

        # FlexAlign does not support cropping
        form._paramsDict['Alignment']._paramList.remove('Crop_offsets__px_')
        form._paramsDict['Alignment']._paramList.remove('Crop_dimensions__px_')

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

        line = group.addLine('Number of control points',
                    expertLevel=cons.LEVEL_ADVANCED,
                    help='Number of control points use for BSpline.',
                    condition='not autoControlPoints')
        line.addParam('controlPointX', params.IntParam, default=6, label='X')
        line.addParam('controlPointY', params.IntParam, default=6, label='Y')
        line.addParam('controlPointT', params.IntParam, default=5, label='t')

        group.addParam('autoPatches', params.BooleanParam, default=True,
                      label="Auto patches",
                      expertLevel=cons.LEVEL_ADVANCED,
                      condition='doLocalAlignment',
                      help="If on, protocol will automatically determine necessary number of patches.")

        line = group.addLine('Number of patches',
                    expertLevel=cons.LEVEL_ADVANCED,
                    help='Number of patches used for local alignment.',
                    condition='not autoPatches')
        line.addParam('patchesX', params.IntParam, default=7, label='X')
        line.addParam('patchesY', params.IntParam, default=7, label='Y')

        group.addParam('minLocalRes', params.FloatParam, default=500,
                       expertLevel=cons.LEVEL_ADVANCED,
                       label='Min size of the patch (A)',
                       help="How many A should contain each patch?")

        group.addParam('groupNFrames', params.IntParam, default=3,
                    expertLevel=cons.LEVEL_ADVANCED,
                    label='Group N frames',
                    help='Group every specified number of frames by adding them together. \
                        The alignment is then performed on the summed frames.',
                    condition='doLocalAlignment')
        form.addParam('maxResForCorrelation', params.FloatParam, default=30,
                      label='Maximum resolution (A)',
                      help="Maximum resolution in A that will be preserved during correlation.")

        form.addParam('doComputePSD', params.BooleanParam, default=True,
                      label="Compute PSD?",
                      help="If Yes, the protocol will compute PSD for each movie "
                           "before and after the alignment")

        form.addParam('maxShift', params.IntParam, default=50,
                      expertLevel=cons.LEVEL_ADVANCED,
                      label="Maximum shift (A)",
                      help='Maximum allowed distance (in A) that each '
                           'frame can be shifted with respect to the next.')

        #Gain Section
        form.addSection(label="Gain orientation")
        form.addParam('gainRot', params.EnumParam,
                      choices=['no rotation', '90 degrees',
                               '180 degrees', '270 degrees'],
                      label="Rotate gain reference:",
                      default=self.NO_ROTATION,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help="Rotate gain reference counter-clockwise.")

        form.addParam('gainFlip', params.EnumParam,
                      choices=['no flip', 'upside down', 'left right'],
                      label="Flip gain reference:", default=self.NO_FLIP,
                      display=params.EnumParam.DISPLAY_COMBO,
                      help="Flip gain reference after rotation. "
                           "For tiff movies, gain is automatically upside-down flipped")

        form.addParallelSection(threads=1, mpi=1)
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=cons.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True, label="Use GPU or CPU implementation of the algorithm.")



    #--------------------------- STEPS functions -------------------------------
    def _processMovie(self, movie):
        try:
            self.tryProcessMovie(movie)
        except Exception as ex:
            print(yellowStr("We cannot process %s" % movie.getFileName()))
            print(ex)

    def getUserAngle(self):
      anglesDic = {0:0, 1:90, 2:180, 3:270}
      return anglesDic[self.gainRot.get()]

    def getUserFlip(self, imag_array):
      flipDic = {0: np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                 1: np.asarray([[1, 0, 0], [0, -1, imag_array.shape[0]], [0, 0, 1]]),
                 2: np.asarray([[-1, 0, imag_array.shape[1]], [0, 1, 0], [0, 0, 1]])}
      return flipDic[self.gainFlip.get()]

    def getGPUArgs(self):
        args = ' --device %(GPU)s'
        args += ' --storage "%s"' % self._getExtraPath("fftBenchmark.txt")
        args += ' --controlPoints %d %d %d' % (self.controlPointX, self.controlPointY, self.controlPointT)
        args += ' --patchesAvg %d' % self.groupNFrames
        return args

    def tryProcessMovie(self, movie):
        movieFolder = self._getOutputMovieFolder(movie)

        if self._isInputEer():
            n = self.nFrames.get()
        else:
            _, _, n = movie.getDim()
        a0, aN = self._getFrameRange(n, 'align')
        s0, sN = self._getFrameRange(n, 'sum')

        inputMd = os.path.join(movieFolder, 'input_movie.xmd')
        writeMovieMd(movie, inputMd, a0, aN, useAlignment=False, eerFrames=self.nFrames.get())

        args = '-i "%s" ' % inputMd
        args += ' -o "%s"' % self._getShiftsFile(movie)
        args += ' --sampling %f' % movie.getSamplingRate()
        args += ' --maxResForCorrelation %f' % self.maxResForCorrelation

        if self.binFactor > 1:
            args += ' --bin %f' % self.binFactor

        args += ' --frameRange %d %d' % (0, aN-a0)
        args += ' --frameRangeSum %d %d' % (s0-a0, sN-a0)
        args += ' --maxShift %d' % self.maxShift

        if self.doSaveAveMic or self.doComputePSD:
            fnAvg = self._getExtraPath(self._getOutputMicName(movie))
            args += ' --oavg "%s"' % fnAvg

        if self.doComputePSD:
            fnInitial = os.path.join(movieFolder, "initialMic.mrc")
            args  += ' --oavgInitial "%s"' % fnInitial

        if self.doSaveMovie:
            args += ' --oaligned "%s"' % self._getExtraPath(self._getOutputMovieName(movie))

        if self.inputMovies.get().getDark():
            args += ' --dark "%s"' % self.inputMovies.get().getDark()

        if self.inputMovies.get().getGain():
            ext = pwutils.getExt(self.inputMovies.get().getFirstItem().getFileName()).lower()
            if ext in ['.tif', '.tiff', '.gain']:
              self.flipY = True
              inGainFn = self.inputMovies.get().getGain()
              gainFn = xmutils.flipYImage(inGainFn, outDir = self._getExtraPath())
            else:
              gainFn = self.inputMovies.get().getGain()

            if self.gainRot.get() != 0 or self.gainFlip.get() != 0:
              gainFn = self.transformGain(gainFn, self._getTmpPath('gain.tif'))
            args += ' --gain "%s"' % gainFn

        if self.autoControlPoints.get():
            self._setControlPoints()

        if not self.autoPatches.get():
            args += ' --patches %d %d ' % (self.patchesX, self.patchesY)

        if self.minLocalRes.get():
            args += ' --minLocalRes %f' % self.minLocalRes

        if not self.doLocalAlignment.get():
            args += ' --skipLocalAlignment '

        if self.useGpu.get():
            args += self.getGPUArgs()
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
    def transformGain(self, gainFn, outFn=None):
      '''Transforms the gain image with the user specifications'''
      if outFn == None:
        ext = pwutils.getExt(gainFn)
        baseName = os.path.basename(gainFn).replace(ext, '_transformed' + ext)
        outFn = os.path.abspath(self._getExtraPath(baseName))

      if os.path.isfile(outFn):
        return outFn 

      gainImg = xmutils.readImage(gainFn)
      imag_array = np.asarray(gainImg.getData(), dtype=np.float64)

      # Building the transformation matrix
      angle = self.getUserAngle()
      M = self.getUserFlip(imag_array)
      print('Transforming gain: {} degrees rotation, {} flip'.
            format(angle, ['no', 'vertical', 'horizontal'][self.gainFlip.get()]))
      flipped_array, M = xmutils.rotation(imag_array, angle, imag_array.shape, M)
      xmutils.writeImageFromArray(flipped_array, outFn)
      return outFn

    def _isInputEer(self):
        return isEerMovie(self.inputMovies.get())

    def _getShiftsFile(self, movie):
        return self._getExtraPath(self._getMovieRoot(movie) + '_shifts.xmd')

    def _setControlPoints(self):
        x, y, frames = self.inputMovies.get().getDim()
        if self._isInputEer():
            frames = self.nFrames.get()
        Ts = self.inputMovies.get().getSamplingRate()
        # one control point each 1000 A
        self.controlPointX.set(max([int(x * Ts) / 1000 + 2, 3]))
        self.controlPointY.set(max([int(y * Ts) / 1000 + 2, 3]))
        self.controlPointT.set(max([ceil(frames/7.) + 2, 3]))

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

    def _validateParallelProcessing(self):
        nGpus = len(self.gpuList.get().split())
        nThreads = self.numberOfThreads.get()
        errors = []
        neededThreads = 1 if nGpus == 1 else nGpus + 1
        if nThreads != neededThreads:
            errors.append('Please assign the number of threads so that it corresponds to the amount of GPUs + 1.')
        return errors

    def _validateBinary(self):
        errors = []
        getXmippHome = self.getClassPackage().Plugin.getHome
        cpuBinaryFn = getXmippHome('bin', 'xmipp_movie_alignment_correlation')
        if self.useGpu.get() and not isXmippCudaPresent("xmipp_cuda_movie_alignment_correlation"):
            errors.append('GPU version not found, make sure that Xmipp is '
                          'compiled with GPU\n'
                          '( *CUDA=True* in _scipion.conf_ + '
                          '_run_: $ *scipion installb xmippSrc* ).')
        elif not os.path.isfile(cpuBinaryFn):
            errors.append('CPU version not found for some reason, try to use the GPU version.')
        return errors

    def _validate(self):
        if self.autoControlPoints.get():
            self._setControlPoints()  # make sure we work with proper values
        # check execution issues
        errors = ProtAlignMovies._validate(self)
        errors.extend(self._validateBinary())
        errors.extend(self._validateParallelProcessing())
        if errors:
            return errors

        # check settings issues
        if self.doLocalAlignment.get() and not self.useGpu.get():
            errors.append("GPU is needed to do local alignment.")
        if (self.controlPointX < 3):
            errors.append("You have to use at least 3 control points in X dim")
        if (self.controlPointY < 3):
            errors.append("You have to use at least 3 control points in Y dim")
        if (self.controlPointT < 3):
            errors.append("You have to use at least 3 control points in T dim")

        if (self.binFactor.get() < 1):
            errors.append("Bin factor must be >= 1")
        
        return errors

    def _citations(self):
        return ['strelak2020flexalign', 'Strelak2023performance']
