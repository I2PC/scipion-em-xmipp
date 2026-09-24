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
    Wrapper protocol for Xmipp Movie Alignment using cross-correlation methods.
    It aligns movie frames to produce beam-induced motion corrected micrographs.

    AI Generated

    ## Overview

    The FlexAlign protocol aligns cryo-EM movie frames to correct for beam-induced
    motion and produce motion-corrected micrographs. During exposure, the electron
    beam causes the sample, ice, and support film to move. If this motion is not
    corrected, high-resolution information is blurred when the frames are summed.

    FlexAlign estimates the shifts between movie frames using cross-correlation
    methods. It can perform both global movie alignment and local alignment, where
    different regions of the image are allowed to move slightly differently. This
    local correction is especially important for modern direct-electron detector
    data, where beam-induced motion may vary across the field of view.

    The main outputs are aligned movies, averaged micrographs, motion-shift plots,
    and, optionally, power spectral density diagnostics before and after alignment.
    These outputs help the user assess whether the movie alignment has improved the
    data and whether the corrected micrographs are suitable for later CTF
    estimation, particle picking, and reconstruction.

    ## Inputs and General Workflow

    The protocol takes a set of input movies. Each movie is processed independently.
    For each movie, FlexAlign reads the selected frame range, estimates frame
    shifts, optionally estimates local motion, applies the correction, and produces
    the requested output objects.

    Depending on the selected options, the protocol may save:

    - an aligned average micrograph;
    - an aligned movie;
    - estimated frame shifts;
    - shift plots;
    - PSD images before and after alignment.

    The averaged micrographs are usually the most important output for the standard
    single-particle workflow. They are used for CTF estimation and particle picking.
    The aligned movies are useful if further movie-level processing is needed.

    ## EER Movies

    If the input movies are in EER format, the protocol allows the user to define
    the **Number of EER frames**. EER files contain very fine-grained subframes,
    which must be grouped into a practical number of frames before alignment.

    A larger number of EER frames gives finer temporal sampling of the motion, but
    also increases computational cost and may reduce the signal available in each
    frame group. A smaller number of frames gives stronger signal per frame group,
    but may describe the motion less precisely.

    The default value is a reasonable starting point for many datasets, but users
    may adjust it depending on dose, exposure time, and the expected amount of
    motion.

    ## Frame Ranges for Alignment and Summation

    As in other Scipion movie-alignment protocols, the user can define which frames
    are used to estimate the alignment and which frames are used to generate the
    final summed micrograph.

    The **alignment frame range** determines the frames used to estimate motion.
    The **summation frame range** determines the frames included in the final
    average.

    These two ranges do not necessarily have to be identical. For example, very
    early frames may contain strong initial motion, and very late frames may contain
    more radiation damage. Depending on the dataset, users may exclude some frames
    from the final sum while still using enough frames to estimate a stable motion
    trajectory.

    From a biological point of view, this choice controls the compromise between
    signal, radiation damage, and motion correction quality.

    ## Global and Local Alignment

    FlexAlign can perform local alignment in addition to global alignment.

    Global alignment estimates a single shift per frame. This corrects the average
    motion of the whole movie. It is often sufficient when the motion is small or
    spatially homogeneous.

    Local alignment estimates position-dependent motion. In this mode, different
    regions of the micrograph can have different shifts. This is useful because
    beam-induced motion is frequently not uniform across the image.

    The option **Compute local alignment?** enables this local correction. In most
    modern cryo-EM workflows, local alignment is recommended, especially for
    high-resolution single-particle analysis.

    If local alignment is disabled, the protocol performs a simpler correction.
    This may be faster, but it can leave residual local blurring in the corrected
    micrograph.

    ## Control Points

    For local alignment, FlexAlign models the motion field using control points.
    These control points define how flexibly the estimated deformation can vary
    across space and time.

    The protocol can determine the number of control points automatically. This is
    the recommended option for most users. Automatic control-point selection uses
    the movie dimensions, sampling rate, and number of frames to choose values that
    are appropriate for the dataset.

    Advanced users can disable automatic control-point selection and manually set
    the number of control points in X, Y, and time. Larger numbers allow a more
    flexible motion model, but they may also make the estimation less stable if the
    movie does not contain enough signal. Too few control points may underfit the
    motion and fail to correct local deformation.

    At least three control points are required in each dimension.

    ## Patches for Local Alignment

    Local alignment also uses image patches. These patches provide local regions
    from which the motion can be estimated.

    The **Auto patches** option lets the protocol choose the number of patches
    automatically. This is recommended for routine use.

    If the number of patches is set manually, the user controls how finely the
    micrograph is divided for local motion estimation. More patches can capture
    more local variation, but each patch contains less signal. Fewer patches are
    more stable but may miss spatially varying motion.

    The correct balance depends on micrograph size, particle distribution, dose,
    ice quality, and the amount of beam-induced movement.

    ## Minimum Patch Size

    The **Min size of the patch** parameter defines the minimum physical size, in
    angstroms, that each local-alignment patch should cover.

    This parameter helps prevent the local-alignment model from using patches that
    are too small to provide reliable correlation signal. If patches are too small,
    their estimated shifts may become noisy. If they are too large, local motion may
    be averaged out.

    For most users, the default value should be left unchanged unless there is a
    specific reason to tune local alignment behavior.

    ## Grouping Frames

    The **Group N frames** parameter controls whether several consecutive frames
    are summed before estimating local alignment.

    Grouping frames increases the signal-to-noise ratio of the images used for
    alignment, which can make shift estimation more stable. However, grouping also
    reduces temporal resolution. If too many frames are grouped together, rapid
    motion may be smoothed out and not fully corrected.

    The default value is a practical compromise. Increasing this value may help for
    very noisy movies; decreasing it may help when motion changes rapidly across
    the exposure.

    ## Maximum Resolution for Correlation

    The **Maximum resolution** parameter defines the highest-resolution information
    preserved during the correlation step.

    Movie alignment is usually driven by relatively low- and medium-resolution
    features, which are more robust to noise. Very high-resolution information may
    be too noisy to help the correlation and can sometimes make alignment less
    stable.

    The default value limits the correlation to a resolution range that is usually
    suitable for estimating motion. Users should be cautious when changing this
    parameter. Preserving too much high-resolution information during correlation
    does not necessarily improve the final micrograph.

    ## Maximum Shift

    The **Maximum shift** parameter defines the maximum allowed displacement, in
    angstroms, between consecutive frames.

    This parameter acts as a safeguard against unrealistic frame-to-frame jumps. If
    the value is too small, genuine motion may be artificially restricted. If it is
    too large, the algorithm may accept unstable or incorrect correlations.

    The default value is suitable for many datasets. Datasets with very strong
    initial beam-induced motion may require a larger value, but such changes should
    be checked carefully using the resulting shift plots.

    ## Binning

    If a binning factor is used, the movie is processed at reduced image size. This
    can speed up alignment and reduce memory requirements.

    Binning may be useful for very large movies or for preliminary processing.
    However, excessive binning can remove information needed for accurate alignment
    and may reduce the quality of the final corrected micrographs.

    The bin factor must be greater than or equal to 1. A value of 1 means that no
    binning is applied.

    ## Gain and Dark Correction

    If the input movie set has associated gain or dark references, FlexAlign can
    use them during processing.

    Gain correction compensates for pixel-to-pixel sensitivity differences in the
    detector. Dark correction compensates for detector background signal. These
    corrections are important because uncorrected detector artifacts can interfere
    with frame alignment, CTF estimation, and later particle processing.

    The protocol includes a **Gain orientation** section that allows the user to
    rotate or flip the gain reference. This is useful when the gain image and the
    movie frames have different orientations.

    For TIFF movies, the gain reference may require an automatic vertical flip.
    The protocol also allows explicit gain rotation by 90, 180, or 270 degrees, and
    flipping upside down or left-right.

    Users should pay special attention to gain orientation. An incorrectly oriented
    gain reference can introduce strong artifacts into the corrected micrographs.

    ## PSD Computation

    The option **Compute PSD?** makes the protocol compute power spectral density
    diagnostics before and after alignment.

    PSD images are useful for assessing data quality and the effect of motion
    correction. After successful alignment, Thon rings and other frequency-domain
    features may become clearer, especially at higher resolution.

    These diagnostics are particularly useful when checking whether motion
    correction has improved the data before proceeding to CTF estimation.

    If PSD computation is enabled only for diagnostic purposes, temporary average
    micrographs used for PSD calculation may be removed automatically unless the
    user has also requested to save the average micrographs.

    ## Shift Plots

    For each processed movie, FlexAlign stores a plot of the estimated motion
    trajectory. This plot shows the cumulative shifts in X and Y across frames.

    Shift plots are one of the most useful practical diagnostics of movie
    alignment. Smooth trajectories usually indicate stable alignment. Very abrupt
    jumps may indicate poor correlation, very low signal, incorrect gain correction,
    bad frames, or excessive motion.

    Users should inspect representative shift plots, especially when processing a
    new dataset or when changing alignment parameters.

    ## GPU and CPU Execution

    FlexAlign can use either the GPU or CPU implementation, depending on the
    available installation and selected options.

    Local alignment requires the GPU implementation. If local alignment is enabled,
    the protocol must be run with GPU support.

    The protocol checks that the required Xmipp binaries are available. If the GPU
    version is selected but the CUDA-enabled binary is not present, the protocol
    will report an error. In that case, the Xmipp installation should be checked.

    When using GPUs, the number of Scipion threads should be consistent with the
    number of selected GPU devices. In typical use, the protocol expects the number
    of threads to correspond to the number of GPUs plus one.

    ## Outputs and Their Interpretation

    Depending on the selected options, the protocol can produce aligned
    micrographs, aligned movies, shift metadata, alignment plots, and PSD
    diagnostics.

    The aligned average micrographs are generally used as input for CTF estimation
    and particle picking. They represent the motion-corrected sum of the selected
    movie frames.

    The aligned movies preserve the corrected frame sequence and may be useful for
    additional movie-level processing.

    The shift metadata and plots describe the estimated motion. They should be used
    to evaluate whether the correction behaved sensibly.

    The PSD diagnostics help assess whether the alignment improves the frequency
    content of the micrograph and whether the corrected images are suitable for
    subsequent processing.

    ## Practical Recommendations

    For most modern single-particle cryo-EM datasets, local alignment should be
    enabled and automatic control points and patches should be used.

    Keep PSD computation enabled when processing a new dataset, because it provides
    a useful diagnostic of alignment quality.

    Use the default maximum resolution for correlation unless there is a specific
    reason to change it. Movie alignment does not usually benefit from using very
    high-resolution noisy information during correlation.

    Inspect the shift plots for several movies. Smooth, physically plausible motion
    trajectories are a good sign. Sudden jumps or erratic behavior should be
    investigated.

    Check gain orientation carefully. Many alignment problems are caused not by the
    motion algorithm itself, but by incorrect gain correction.

    Use frame ranges thoughtfully. Excluding damaged late frames or unstable early
    frames from the final sum can improve the quality of the averaged micrographs.

    ## Final Perspective

    FlexAlign is an early but crucial step in the cryo-EM image-processing
    workflow. It converts raw detector movies into motion-corrected images that can
    be reliably used for CTF estimation, particle picking, classification, and
    reconstruction.

    Good movie alignment preserves high-resolution information that would otherwise
    be blurred by beam-induced motion. Poor alignment, incorrect gain correction,
    or inappropriate frame selection can limit the achievable resolution of the
    entire project.

    For biological users, the main goal is to obtain corrected micrographs with
    stable motion trajectories, clear diagnostic PSDs, and no obvious artifacts.
    These corrected micrographs form the foundation for the rest of the
    single-particle analysis workflow.
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
