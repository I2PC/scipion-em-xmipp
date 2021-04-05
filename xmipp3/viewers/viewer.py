# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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

from pwem.viewers import showj, DataViewer, EmPlotter, CtfView, ObjectView
from pyworkflow.utils import removeBaseExt
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO
from pwem.protocols import *

from pwem.viewers.showj import *
from pwem import emlib

from xmipp3.convert import *
from xmipp3.protocols import XmippProtCompareReprojections
from xmipp3.protocols import XmippProtCompareAngles
from xmipp3.protocols import XmippProtConsensusLocalCTF
from xmipp3.protocols import XmippProtExtractParticles
from xmipp3.protocols import XmippProtExtractParticlesPairs
from xmipp3.protocols import XmippProtKerdensom
from xmipp3.protocols import XmippParticlePickingAutomatic
from xmipp3.protocols import XmippProtParticlePickingPairs
from xmipp3.protocols import XmippProtRotSpectra
from xmipp3.protocols import XmippProtScreenParticles
from xmipp3.protocols import XmippProtCTFMicrographs
from xmipp3.protocols import XmippProtValidateNonTilt
from xmipp3.protocols import XmippProtMultiRefAlignability
from xmipp3.protocols import XmippProtAssignmentTiltPair
from xmipp3.protocols import XmippProtMovieGain
from xmipp3.protocols import XmippProtDeepDenoising
from xmipp3.protocols import XmippProtParticleBoxsize
from .plotter import XmippPlotter


class XmippViewer(DataViewer):
    """ Wrapper to visualize different type of objects
    with the Xmipp program xmipp_showj
    """
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [
                XmippProtCompareReprojections,
                XmippProtCompareAngles,
                XmippProtConsensusLocalCTF,
                XmippParticlePickingAutomatic,
                XmippProtExtractParticles,
                XmippProtExtractParticlesPairs,
                XmippProtKerdensom,
                XmippProtParticlePickingPairs,
                XmippProtRotSpectra,
                XmippProtScreenParticles,
                XmippProtCTFMicrographs,
                XmippProtValidateNonTilt,
                XmippProtAssignmentTiltPair,
                XmippProtMultiRefAlignability,
                XmippProtMovieGain,
                XmippProtDeepDenoising,
                XmippProtParticleBoxsize
                ]

    def __createTemporaryCtfs(self, obj, setOfMics):
        pwutils.cleanPath(obj._getPath("ctfs_temporary.sqlite"))
        self.protocol._createFilenameTemplates()
        ctfSet = self.protocol._createSetOfCTF("_temporary")

        for mic in setOfMics:
            micDir = obj._getExtraPath(removeBaseExt(mic.getFileName()))
            ctfparam = self.protocol._getFileName('ctfparam', micDir=micDir)

            if exists(ctfparam) or exists('xmipp_default_ctf.ctfparam'):
                if not os.path.exists(ctfparam):
                    ctfparam = 'xmipp_default_ctf.ctfparam'
                ctfModel = readCTFModel(ctfparam, mic)
                self.protocol._setPsdFiles(ctfModel, micDir)
                ctfSet.append(ctfModel)

        if not ctfSet.isEmpty():
            ctfSet.write()
            ctfSet.close()

        return ctfSet

    def _visualize(self, obj, **kwargs):
        cls = type(obj)

        if issubclass(cls, XmippProtCTFMicrographs):
            if obj.hasAttribute('outputCTF'):
                ctfSet = obj.outputCTF
            else:
                mics = obj.inputMicrographs.get()
                ctfSet = self.__createTemporaryCtfs(obj, mics)

            if ctfSet.isEmpty():
                self._views.append(self.infoMessage("No CTF estimation has finished yet"))
            else:
                self.getCTFViews(ctfSet)

        # elif issubclass(cls, SetOfNormalModes):
        #     from .viewer_nma import OBJCMD_NMA_PLOTDIST, OBJCMD_NMA_VMD
        #     fn = obj.getFileName()
        #     objCommands = "'%s' '%s'" % (OBJCMD_NMA_PLOTDIST, OBJCMD_NMA_VMD)
        #     self._views.append(ObjectView(self._project, self.protocol.strId(),
        #                                   fn, obj.strId(),
        #                                   viewParams={OBJCMDS: objCommands},
        #                                   **kwargs))

        elif (issubclass(cls, XmippProtExtractParticles) or
              issubclass(cls, XmippProtScreenParticles)):
            particles = obj.outputParticles
            self._visualize(particles)

            fn = obj._getPath('images.xmd')
            if os.path.exists(fn): # it doesnt unless cls is Xmipp
                md = emlib.MetaData(fn)
                # If Zscore on output images plot Zscore particle sorting
                if md.containsLabel(emlib.MDL_ZSCORE):
                    xplotter = XmippPlotter(windowTitle="Zscore particles sorting")
                    xplotter.createSubPlot("Particle sorting", "Particle number", "Zscore")
                    xplotter.plotMd(md, False, mdLabelY=emlib.MDL_ZSCORE)
                    self._views.append(xplotter)
                # If VARscore on output images plot VARscore particle sorting
                if md.containsLabel(emlib.MDL_SCORE_BY_VAR):
                    xplotter = XmippPlotter(windowTitle="Variance particles sorting")
                    xplotter.createSubPlot("Variance Histogram", "Variance", "Number of particles")
                    xplotter.plotMd(md, False, mdLabelY=emlib.MDL_SCORE_BY_VAR, nbins=100)
                    self._views.append(xplotter)


        elif issubclass(cls, XmippProtDeepDenoising):
            fn = obj.outputParticles.getFileName()
            self._views.append(ObjectView(self._project, obj.outputParticles.strId(), fn,
                                          viewParams={VISIBLE: 'enabled id _filename '
                                                               '_xmipp_corrDenoisedProjection '
                                                               '_xmipp_corrDenoisedNoisy '
                                                               '_xmipp_imageOriginal _xmipp_imageRef',
                                                      RENDER: '_filename _xmipp_imageOriginal '
                                                              '_xmipp_imageRef',
                                                      SORT_BY: 'id',
                                                      MODE: MODE_MD}))

            md = emlib.MetaData(obj._getExtraPath('particlesDenoised.xmd'))
            if (md.containsLabel(emlib.MDL_CORR_DENOISED_PROJECTION) and
                md.containsLabel(emlib.MDL_CORR_DENOISED_NOISY)):
                xplotter = XmippPlotter(windowTitle="denoised vs proj & denoised vs original")
                xplotter.createSubPlot("Correlations", "corr_denoised_projection", "corr_denoised_original")
                xplotter.plotScatterMd(md, mdLabelX=emlib.MDL_CORR_DENOISED_PROJECTION,
                                mdLabelY=emlib.MDL_CORR_DENOISED_NOISY )
                self._views.append(xplotter)

        elif issubclass(cls, XmippProtMovieGain):
            if obj.hasAttribute('outputGains'):
                self._visualize(obj.outputGains)
            # movieGainMonitor = MonitorMovieGain(obj,
            #                                     workingDir=obj.workingDir.get(),
            #                                     samplingInterval=60,
            #                                     monitorTime=300,
            #                                     stddevValue=0.04,
            #                                     ratio1Value=1.15,
            #                                     ratio2Value=4.5)
            # self._views.append(MovieGainMonitorPlotter(movieGainMonitor))

        elif issubclass(cls, XmippProtRotSpectra):
            self._visualize(obj.outputClasses,
                            viewParams={'columns': obj.SomXdim.get(),
                                        RENDER: ' spectraPlot._filename average._filename',
                                        ZOOM: 30,
                                        VISIBLE: 'enabled id _size average._filename '
                                                 'spectraPlot._filename',
                                        'labels': 'id _size',
                                        SORT_BY: 'id'})

        elif issubclass(cls, XmippProtKerdensom):
            self._visualize(obj.outputClasses, viewParams={'columns': obj.SomXdim.get(),
                                                           'render': 'average._filename '
                                                                     '_representative._filename',
                                                           'labels': '_size',
                                                           'sortby': 'id'})

        elif issubclass(cls, XmippProtCompareReprojections):
                fn = obj.reprojections.getFileName()
                labels = ('id enabled _index _xmipp_image._filename _xmipp_imageRef._filename '
                          '_xmipp_imageResidual._filename _xmipp_imageCovariance._filename '
                          '_xmipp_cost _xmipp_zScoreResCov _xmipp_zScoreResMean _xmipp_zScoreResVar '
                          '_xmipp_continuousA _xmipp_continuousB _xmipp_continuousX _xmipp_continuousY')
                labelRender = ("_xmipp_image._filename _xmipp_imageRef._filename "
                              "_xmipp_imageResidual._filename _xmipp_imageCovariance._filename")
                self._views.append(ObjectView(self._project, obj.reprojections.strId(), fn,
                                              viewParams={ORDER: labels,
                                                          VISIBLE: labels,
                                                          SORT_BY: '_xmipp_cost asc',
                                                          RENDER: labelRender,
                                                          MODE: MODE_MD}))

        elif issubclass(cls, XmippProtCompareAngles):
                fn = obj.outputParticles.getFileName()
                labels = 'id enabled _index _filename _xmipp_shiftDiff _xmipp_angleDiff'
                labelRender = "_filename"
                self._views.append(ObjectView(self._project, obj.outputParticles.strId(), fn,
                                              viewParams={ORDER: labels,
                                                          VISIBLE: labels,
                                                          SORT_BY: '_xmipp_angleDiff asc',
                                                          RENDER: labelRender,
                                                          MODE: MODE_MD}))

        elif issubclass(cls, XmippProtConsensusLocalCTF):
                fn = obj.outputParticles.getFileName()
                labels = ('id enabled _index _filename _ctfModel._xmipp_ctfDefocusA '
                          '_ctfModel._xmipp_ctfDefocusResidual')
                labelRender = "_filename"
                self._views.append(ObjectView(self._project, obj.outputParticles.strId(), fn,
                                              viewParams={ORDER: labels,
                                                          VISIBLE: labels,
                                                          SORT_BY: '_ctfModel._xmipp_ctfDefocusResidual',
                                                          RENDER: labelRender,
                                                          MODE: MODE_MD}))

        elif issubclass(cls, XmippParticlePickingAutomatic):
            micSet = obj.getInputMicrographs()
            mdFn = getattr(micSet, '_xmippMd', None)
            inTmpFolder = False
            if mdFn:
                micsfn = mdFn.get()
            else:  # happens if protocol is not an xmipp one
                micsfn = self._getExtraPath(micSet.getName() + '_micrographs.xmd')
                writeSetOfMicrographs(micSet, micsfn)
                inTmpFolder = True

            posDir = obj._getExtraPath()
            memory = showj.getJvmMaxMemory()
            launchSupervisedPickerGUI(micsfn, posDir, obj, mode='review',
                                      memory=memory, inTmpFolder=inTmpFolder)

        elif issubclass(cls, XmippProtParticleBoxsize):
            """ Launching a Coordinates viewer with only one coord in the center
                with the estimated boxsize.
            """
            micSet = obj.inputMicrographs.get()

            coordsFn = self._getExtraPath(micSet.getName()+'_coords_to_view.sqlite')
            if not os.path.exists(coordsFn):
                # Just creating the coords once
                coordsSet = SetOfCoordinates(filename=coordsFn)
                coordsSet.setBoxSize(obj.boxsize)
                for mic in micSet:
                    coord = Coordinate()
                    coord.setMicrograph(mic)
                    coord.setPosition(mic.getXDim()/2, mic.getYDim()/2)
                    coordsSet.append(coord)
                coordsSet.write()
            else:
                coordsSet = SetOfCoordinates(filename=coordsFn)
                coordsSet.loadAllProperties()

            coordsSet.setMicrographs(micSet)
            coordsSet.setName(micSet.getName())
            self._visualize(coordsSet)

         # We need this case to happens before the ProtParticlePicking one
        elif issubclass(cls, XmippProtAssignmentTiltPair):
            if obj.getOutputsSize() >= 1:
                coordsSet = obj.getCoordsTiltPair()
                self._visualize(coordsSet)

        elif issubclass(cls, XmippProtValidateNonTilt):
            outputVols = obj.outputVolumes
            labels = 'id enabled comment _filename weight'
            self._views.append(ObjectView(self._project, outputVols.strId(),
                                          outputVols.getFileName(),
                                          viewParams={MODE: MODE_MD,
                                                      VISIBLE: labels,
                                                      ORDER: labels,
                                                      SORT_BY: 'weight desc',
                                                      RENDER: '_filename'}))

        elif issubclass(cls, XmippProtMultiRefAlignability):
            outputVols = obj.outputVolumes
            labels = ('id enabled comment _filename weightAlignabilityPrecision '
                      'weightAlignabilityAccuracy')
            self._views.append(ObjectView(self._project, outputVols.strId(),
                                          outputVols.getFileName(),
                                          viewParams={MODE: MODE_MD,
                                                      VISIBLE: labels,
                                                      ORDER: labels,
                                                      SORT_BY: 'weightAlignabilityAccuracy desc',
                                                      RENDER: '_filename'}))

            fn = obj.outputParticles.getFileName()
            labels = ('id enabled _index _filename _xmipp_scoreAlignabilityAccuracy '
                      '_xmipp_scoreAlignabilityPrecision')
            labelRender = "_filename"
            self._views.append(ObjectView(self._project, obj.outputParticles.strId(), fn,
                                          viewParams={ORDER: labels,
                                                      VISIBLE: labels,
                                                      SORT_BY: '_xmipp_scoreAlignabilityAccuracy desc',
                                                      RENDER: labelRender,
                                                      MODE: MODE_MD}))

            fn = obj._getExtraPath('vol001_pruned_particles_alignability.xmd')
            md = emlib.MetaData(fn)
            plotter = XmippPlotter()
            plotter.createSubPlot('Soft-alignment validation plot',
                                  'Angular Precision', 'Angular Accuracy')
            plotter.plotMdFile(md, emlib.MDL_SCORE_BY_ALIGNABILITY_PRECISION,
                               emlib.MDL_SCORE_BY_ALIGNABILITY_ACCURACY,
                               marker='.', markersize=.55, color='red', linestyle='')
            self._views.append(plotter)

        elif issubclass(cls, XmippProtExtractParticlesPairs):
            self._visualize(obj.outputParticlesTiltPair)

        else:
            # Use default visualization defined in base class
            DataViewer._visualize(self, obj, **kwargs)

        return self._views

    def getCTFViews(self, ctfSet):
        # This could be used by any CTF viewer to show CTF plus, phaseShift plot
        # if applies.
        # Return phaseShift plot if apply
        firstCtf = ctfSet.getFirstItem()

        if firstCtf.hasPhaseShift():
            phase_shift = []

            for ctf in ctfSet.iterItems():
                phShift = ctf.getPhaseShift()
                phase_shift.append(phShift)

            plotter = EmPlotter()
            plotter.createSubPlot("Phase Shift estimation",
                                  "Number of CTFs", "Phase Shift")
            plotter.plotData(np.arange(0, len(phase_shift)), phase_shift)
            self._views.append(plotter)

        # Return Standard CTF view (showJ)
        self._views.append(CtfView(self._project, ctfSet))
