# **************************************************************************
# *
# * Authors:     Roberto Marabini (roberto@cnb.csic.es)
# *              J.M. de la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Vahid Abrishami (vabrisahmi@cnb.csic.es)
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
In this module are protocol base classes related to EM Micrographs
"""

import sys
from os.path import join
import numpy as np

from pyworkflow.object import String, Integer
from pyworkflow.protocol.params import IntParam, FloatParam, StringParam, BooleanParam, LEVEL_ADVANCED, LEVEL_ADVANCED, EnumParam
from pyworkflow.utils.path import moveFile
import pyworkflow.em as em
from pyworkflow.em.protocol import ProtProcessMovies
from pyworkflow.gui.plotter import Plotter
import matplotlib.pyplot as plt
import xmipp

#Alignment methods enum
AL_OPTICAL = 0
AL_AVERAGE = 1

class ProtMovieAlignment(ProtProcessMovies):
    """ Aligns movies, from direct detectors cameras, into micrographs.
    """
    _label = 'optical alignment'

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        ProtProcessMovies._defineParams(self, form)

        form.addParam('alignMethod', EnumParam, choices=['optical flow', 'average'],
                      label="Alignment method", default=AL_OPTICAL,
                      display=EnumParam.DISPLAY_COMBO,
                      help='Method to use for movie alignment. ')
        line = form.addLine('Skip frames for alignment',
                            help='Skip frames for alignment.\n'
                                  'The first frame in the stack is *0*.' )
        line.addParam('alignFrame0', IntParam, default=0, label='Begin')
        line.addParam('alignFrameN', IntParam, default=0, label='End',
                      help='The number of frames to cut from the front and end')

        # GROUP GPU PARAMETERS
        group = form.addGroup('GPU', condition="alignMethod==%d" % AL_OPTICAL)
        group.addParam('doGPU', BooleanParam, default=False,
                      label="Use GPU (vs CPU)",
                      help="Set to true if you want the GPU implementation of Optical Flow")
        group.addParam('GPUCore', IntParam, default=0, expertLevel=LEVEL_ADVANCED,
                      label="Choose GPU core",
                      help="GPU may have several cores. Set it to zero if you do not know what we are talking about. First core index is 0, second 1 and so on.")
        
        # GROUP OPTICAL FLOW PARAMETERS
        group = form.addGroup('Parameters', expertLevel=LEVEL_ADVANCED, condition="alignMethod==%d" % AL_OPTICAL)
        group.addParam('winSize', IntParam, default=150,
                      label="Window size", help="Window size (shifts are assumed to be constant within this window).")
        group.addParam('groupSize', IntParam, default=1,
                      label="Group Size", help="In cases with low SNR, the average of a number of frames can be used in alignment")
        group.addParam('doSaveMovie', BooleanParam, default=False,

    #--------------------------- STEPS functions ---------------------------------------------------
    def createOutputStep(self):

        inputMovies = self.inputMovies.get()
        micSet = self._createSetOfMicrographs()
        micSet.copyInfo(inputMovies)
        # Also create a Set of Movies with the alignment parameters
        if self.doSaveMovie:
            movieSet = self._createSetOfMovies()
            movieSet.copyInfo(inputMovies)
        alMethod = self.alignMethod.get()
        for movie in self.inputMovies.get():
            micName = self._getNameExt(movie.getFileName(),'_aligned', 'mrc')
            metadataName = self._getNameExt(movie.getFileName(), '_aligned', 'xmd')
            plotCartName = self._getNameExt(movie.getFileName(), '_plot_cart', 'png')
            psdCorrName = self._getNameExt(movie.getFileName(),'_aligned_corrected', 'psd')
            # Parse the alignment parameters and store the log files
            alignedMovie = movie.clone()
            #if self.run:
                #alignedMovie.setFileName(self._getExtraPath(self._getNameExt(movie.getFileName(),'_aligned', 'mrcs')))
            ####>>>This is wrong. Save an xmipp metadata
            alignedMovie.alignMetaData = String(self._getExtraPath(metadataName))
            alignedMovie.plotCart = self._getExtraPath(plotCartName)
            alignedMovie.psdCorr = self._getExtraPath(psdCorrName)
            if alMethod == AL_OPTICAL:
                movieCreatePlot(alignedMovie, True)
            if self.doSaveMovie:
                movieSet.append(alignedMovie)
            mic = em.Micrograph()
            # All micrograph are copied to the 'extra' folder after each step
            mic.setFileName(self._getExtraPath(micName))
            # The micName of a micrograph MUST be the same as the original movie
            #mic.setMicName(micName)
            mic.setMicName(movie.getMicName())
            if alMethod == AL_OPTICAL:
                mic.plotCart = em.Image()
                mic.plotCart.setFileName(self._getExtraPath(plotCartName))
            mic.psdCorr = em.Image()
            mic.psdCorr.setFileName(self._getExtraPath(psdCorrName))
            micSet.append(mic)

        self._defineOutputs(outputMicrographs=micSet)
        self._defineSourceRelation(self.inputMovies, micSet)
        if self.doSaveMovie:
            self._defineOutputs(outputMovies=movieSet)

    #--------------------------- UTILS functions ---------------------------------------------------
    #TODO: In methods calling 2 protocols we should:
    #      1) work with the original movie and not the resampled one
    #      2) output metadata with shifts should be given over
    #         orignal movie not intermediate one
    def _processMovie(self, movieId, movieName, movieFolder):
        """ Process the movie actions, remember to:
        1) Generate all output files inside movieFolder (usually with cwd in runJob)
        2) Copy the important result files after processing (movieFolder will be deleted!!!)
        """

        # Read the parameters
        #micName = self._getMicName(movieId)
        micName = self._getNameExt(movieName, '_aligned', 'mrc')
        metadataName = self._getNameExt(movieName, '_aligned', 'xmd')
        psdCorrName = self._getNameExt(movieName,'_aligned_corrected', 'psd')
        firstFrame = self.alignFrame0.get()
        lastFrame = self.alignFrameN.get()
        gpuId = self.GPUCore.get()
        alMethod = self.alignMethod.get()

        # Some movie have .mrc or .mrcs format but it is recognized as a volume
        if movieName.endswith('.mrcs') or movieName.endswith('.mrc'):
            movieSuffix = ':mrcs'
        else:
            movieSuffix = ''
        command = '-i %(movieName)s%(movieSuffix)s -o %(micName)s ' % locals()
        command += '--cutf %d --cute %d ' % (firstFrame, lastFrame)
        program = 'xmipp_movie_optical_alignment_cpu'
        if self.inputMovies.get().getDark():
            command += '--dark '+self.inputMovies.get().getDark()
        if self.inputMovies.get().getGain():
            command += '--gain '+self.inputMovies.get().getGain()

        # For simple average execution
        if alMethod == AL_AVERAGE:
            command += '--simpleAverage'
        # For optical flow execution
        else:
            winSize = self.winSize.get()
            doSaveMovie = self.doSaveMovie.get()
            groupSize = self.groupSize.get()
            command += '--winSize %(winSize)d --groupSize %(groupSize)d ' % locals()
            if self.doGPU:
                program = 'xmipp_movie_optical_alignment_gpu'
                command += '--gpu %d ' % gpuId
            if doSaveMovie:
                command += '--ssc'
        try:
            self.runJob(program, command, cwd=movieFolder)
        except:
            print >> sys.stderr, program, " failed for movie %(movieName)s" % locals()
        if alMethod == AL_OPTICAL:
            moveFile(join(movieFolder, metadataName), self._getExtraPath())
            if doSaveMovie:
                outMovieName = self._getNameExt(movieName,'_aligned', 'mrcs')
                moveFile(join(movieFolder, outMovieName), self._getExtraPath())

        # Compute half-half PSD
        ih = em.ImageHandler()
        print join(movieFolder, '%(movieName)s' % locals())
        avg = ih.computeAverage(join(movieFolder, movieName))
        avg.write(join(movieFolder, 'uncorrectedmic.mrc'))
        command = '--micrograph uncorrectedmic.mrc --oroot uncorrectedpsd --dont_estimate_ctf --pieceDim 400 --overlap 0.7'
        program = 'xmipp_ctf_estimate_from_micrograph'
        self.runJob(program, command, cwd=movieFolder)
        command = '--micrograph %(micName)s --oroot correctedpsd --dont_estimate_ctf --pieceDim 400 --overlap 0.7' % locals()
        self.runJob(program, command, cwd=movieFolder)
        correctedPSD = em.ImageHandler().createImage()
        unCorrectedPSD = em.ImageHandler().createImage()
        correctedPSD.read(join(movieFolder, 'correctedpsd.psd'))
        unCorrectedPSD.read(join(movieFolder, 'uncorrectedpsd.psd'))
        x, y, z, n = correctedPSD.getDimensions()
        for i in range(1,y):
            for j in range(1,x//2):
                unCorrectedPSD.setPixel(i, j, correctedPSD.getPixel(i,j))
        unCorrectedPSD.write(join(movieFolder, psdCorrName))

        # Move output micrograph and related information to 'extra' folder
        moveFile(join(movieFolder, micName), self._getExtraPath())
        moveFile(join(movieFolder, psdCorrName), self._getExtraPath())

     #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        numThreads = self.numberOfThreads;
        alMethod = self.alignMethod.get()
        if numThreads>1:
            if self.doGPU:
                errors.append("GPU and Parallelization can not be used together")
        return errors

    def _citations(self):
        return ['Abrishami2015']

    def _methods(self):
        methods = []
        alMethod = self.alignMethod.get()
        gpuId = self.GPUCore.get()
        if alMethod == AL_AVERAGE:
            methods.append('Aligning method: Simple average')
        else:
            methods.append('Aligning method: Optical Flow')
            methods.append('- Used a window size of: *%d*' % self.winSize.get())
            methods.append('- Used a pyramid size of: *6*')
            if self.doGPU:
                methods.append('- Used GPU *%d* for processing' % gpuId)

        return methods

    def _summary(self):
        firstFrame = self.alignFrame0.get()
        lastFrame = self.alignFrameN.get()
        summary = []
        if self.inputMovies.get():
            summary.append('Number of input movies: *%d*' % self.inputMovies.get().getSize())
        summary.append('The number of frames to cut from the front: *%d* to *%s* (first frame is 0)' % (firstFrame, 'Last Frame'))

        return summary

def createPlots(plotType, protocol, movieId):

    movie = protocol.outputMovies[movieId]
    return movieCreatePlot(movie, False)

def movieCreatePlot(movie, saveFig):
    meanX = []
    meanY = []
    figureSize = (8, 6)

    alignedMovie = movie.alignMetaData
    md = xmipp.MetaData(alignedMovie)
    plotter = Plotter(*figureSize)
    figure = plotter.getFigure()

    preX = 0.0
    preY = 0.0
    meanX.append(0.0)
    meanY.append(0.0)
    ax = figure.add_subplot(111)
    ax.grid()
    ax.set_title('Cartesian representation')
    ax.set_xlabel('Drift x (pixels)')
    ax.set_ylabel('Drift y (pixels)')
    ax.plot(0, 0, 'yo-')
    for objId in md:
        preX += md.getValue(xmipp.MDL_OPTICALFLOW_MEANX, objId)
        preY += md.getValue(xmipp.MDL_OPTICALFLOW_MEANY, objId)
        meanX.append(preX)
        meanY.append(preY)
        ax.plot(preX, preY, 'yo-')
        ax.text(preX-0.02, preY+0.01, str(objId+1))
    ax.plot(np.asarray(meanX), np.asarray(meanY))
    if saveFig:
        plotter.savefig(movie.plotCart)
    return plotter
