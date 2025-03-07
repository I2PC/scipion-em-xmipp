# **************************************************************************
# *
# * Authors:     J.L. Vilas (jlvilas@cnb.csic.es)
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

from pyworkflow.gui.plotter import Plotter
from pyworkflow.protocol.params import LabelParam, StringParam, EnumParam, IntParam, LEVEL_ADVANCED
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pyworkflow.utils import getExt, removeExt

from pwem.emlib.metadata import MetaData
from pwem.wizards import ColorScaleWizardBase
from pwem.emlib.image import ImageHandler
from pwem.viewers import ChimeraView, DataView, LocalResolutionViewer
from pwem import emlib


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

from xmipp3.protocols.protocol_resolution_fso import \
    XmippProtFSO, OUTPUT_3DFSC, OUTPUT_DIRECTIONAL_FILTER
from xmipp3.viewers.plotter import XmippPlotter
import xmippLib

# Axis code
AX_X = 0
AX_Y = 1
AX_Z = 2


class XmippProtFSOViewer(LocalResolutionViewer):
    """
    Visualization tools for the FSO, FSC, and 3DFSC.
    
    """
    _label = 'viewer FSO'
    _targets = [XmippProtFSO]
    _environments = [DESKTOP_TKINTER]

    @staticmethod
    def getColorMapChoices():
        return plt.colormaps()

    def _defineParams(self, form):
        form.addSection(label='Visualization')

        form.addParam('doShowOriginalVolumeSlices', LabelParam,
                      label="Original Half Maps slices")

        if self.protocol.estimate3DFSC:
            form.addParam('doShowDirectionalFilter', LabelParam,
                          label="Directionally filtered map")

        groupDirFSC = form.addGroup('FSO and Resolution Analysis')

        groupDirFSC.addParam('doShowAnisotropyCurve', LabelParam,
                             label="Show FSO curve")

        groupDirFSC.addParam('doShowFSC', LabelParam, label="Global FSC")

        if self.protocol.estimate3DFSC:
            groupDirFSC.addParam('doShow3DFSC', LabelParam, label="3DFSC map")
            groupDirFSC.addParam('doShow3DFSCcolorSlices', LabelParam,
                                 label="Show 3DFSC Color slices")

        groupDirFSC.addParam('doShowDirectionalResolution', LabelParam,
                             label="Show Directional Resolution on sphere")

        group = form.addGroup('Choose a Color Map')

        if self.protocol.estimate3DFSC:
            group.addParam('sliceAxis', EnumParam, default=AX_Z,
                           choices=['x', 'y', 'z'],
                           display=EnumParam.DISPLAY_HLIST,
                           label='Slice axis')
            group.addParam('doShowOneColorslice', LabelParam,
                           expertLevel=LEVEL_ADVANCED,
                           label='Show selected slice')
            group.addParam('sliceNumber', IntParam, default=-1,
                           expertLevel=LEVEL_ADVANCED,
                           label='Show slice number')

        ColorScaleWizardBase.defineColorScaleParams(group, defaultLowest=-1, defaultHighest=-1)

    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        if self.protocol.estimate3DFSC:
            return {'doShowOriginalVolumeSlices': self._showOriginalVolumeSlices,
                    'doShow3DFSC': self._show3DFSC,
                    'doShowFSC': self._showFSCCurve,
                    'doShow3DFSCcolorSlices': self._show3DFSCcolorSlices,
                    'doShowOneColorslice': self._showOneColorslice,
                    'doShowAnisotropyCurve': self._showAnisotropyCurve,
                    'doShowDirectionalFilter': self._showDirectionalFilter,
                    'doShowDirectionalResolution': self._showDirectionalResolution,
                    }
        else:
            return {'doShowOriginalVolumeSlices': self._showOriginalVolumeSlices,
                    'doShowFSC': self._showFSCCurve,
                    'doShowAnisotropyCurve': self._showAnisotropyCurve,
                    'doShowDirectionalResolution': self._showDirectionalResolution,
                    }

    def _showOriginalVolumeSlices(self, param=None):
        """
        This function opens the two half maps to visualize the slices 
        """
        cm = DataView(self.protocol.half1.get().getFileName())
        cm2 = DataView(self.protocol.half2.get().getFileName())
        return [cm, cm2]

    def _show3DFSC(self, param=None):
        """
        This function opens the slices of the 3DFSC
        """
        cm = DataView(self.protocol._getExtraPath(OUTPUT_3DFSC))
        return [cm]



    def _showFSCCurve(self, paramName=None):
        """
        It shows the FSC curve in terms of the resolution
        The horizontal axis is linear in this plot. Note
        That this is not the normal representation of hte FSC
        """
        fnmd = self.protocol._getExtraPath('GlobalFSC.xmd')
        title = 'Global FSC'
        xTitle = 'Resolution (1/A)'
        yTitle = 'FSC (a.u.)'
        mdLabelX = emlib.MDL_RESOLUTION_FREQ
        mdLabelY = emlib.MDL_RESOLUTION_FRC
        self._plotCurveFSC(fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY)

    def _show3DFSCcolorSlices(self, param=None):
        """
        It opens 4 colores slices of the 3DFSC
        """
        errors = []
        if self.protocol.estimate3DFSC:
            img = ImageHandler().read(self.protocol._getExtraPath(OUTPUT_3DFSC))
            imgData = img.getData()

            xplotter = XmippPlotter(x=2, y=2, mainTitle="3DFSC Color Slices"
                                                        "along %s-axis."
                                                        % self._getAxis())
            # The slices to be shown are close to the center. Volume size is divided in
            # 9 segments, the fouth central ones are selected i.e. 3,4,5,6
            for i in range(3, 7):
                sliceNumber = self.getSlice(i, imgData)
                a = xplotter.createSubPlot("Slice %s" % (sliceNumber + 1), '', '')
                matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
                plot = xplotter.plotMatrix(a, matrix, 0, 1,
                                           cmap=self.getColorMap(),
                                           interpolation="nearest")
            xplotter.getColorBar(plot)
            return [xplotter]
        else:
            errors.append("The 3dFSC estimation of the 3dFSC was not selected"
                          "in the protocol form.")
            return errors

    def _showAnisotropyCurve(self, paramName=None):
        """
        It shows the FSO curve in terms of the resolution
        The horizontal axis is linear in this plot.
        """
        fnmd = self.protocol._getExtraPath('fso.xmd')
        title = 'Anisotropy Curve'
        xTitle = 'Resolution (1/A)'
        yTitle = 'Anisotropy (a.u.)'
        mdLabelX = emlib.MDL_RESOLUTION_FREQ
        mdLabelY1 = emlib.MDL_RESOLUTION_FSO
        mdLabelY2 = emlib.MDL_RESOLUTION_ANISOTROPY
        self._plotCurveAnisotropy(fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY1, mdLabelY2)

    def _showDirectionalFilter(self, param=None):
        """
        The directionally filtered map using the 3DFSC as low pass filter
        """
        cm = DataView(self.protocol._getExtraPath(OUTPUT_DIRECTIONAL_FILTER))
        return [cm]

    def _showOneColorslice(self, param=None):
        """
        It shows a single colored slice of the 3DFSC
        """
        imageFile = self.protocol._getExtraPath(OUTPUT_3DFSC)
        img = ImageHandler().read(imageFile)
        imgData = img.getData()

        xplotter = XmippPlotter(x=1, y=1, mainTitle="3DFSC Slices "
                                                    "along %s-axis."
                                                    % self._getAxis())
        sliceNumber = self.sliceNumber.get()
        if sliceNumber < 0:
            x, _, _, _ = ImageHandler().getDimensions(imageFile)
            sliceNumber = int(x / 2)
        else:
            sliceNumber -= 1
        # sliceNumber has no sense to start in zero
        a = xplotter.createSubPlot("Slice %s" % (sliceNumber + 1), '', '')
        matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
        plot = xplotter.plotMatrix(a, matrix, 0, 1,
                                   cmap=self.getColorMap(),
                                   interpolation="nearest")
        xplotter.getColorBar(plot)

        return [plt.show(xplotter)]

    def _formatFreq(self, value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999.
        if value:
            inv = 1 / value
        return "1/%0.2f" % inv

    def _plotCurveFSC(self, fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY):
        """
        This function is called by _showFSCCurve.
        It shows the FSC curve in terms of the resolution
        That this is not the normal representation of the FSC
        """
        md = xmippLib.MetaData(fnmd)
        xplotter = XmippPlotter(figure=None)
        xplotter.plot_title_fontsize = 11

        a = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        xplotter.plotMdFile(md, mdLabelX, mdLabelY, 'g')

        a.xaxis.set_major_formatter(FuncFormatter(self._formatFreq))
        xx, yy = self._prepareDataForPlot(md, mdLabelX, mdLabelY)
        a.hlines(0.143, xx[0], xx[-1], colors='k', linestyles='dashed')
        a.grid(True)

        return plt.show()

    def interpolRes(self, thr, x, y):
        """
        This function is called by _showAnisotropyCurve.
        It provides the cut point of the curve defined
        by the points (x,y) with a threshold thr.
        The flag okToPlot shows if there is no intersection points
        """
        idx = np.arange(0, len(x))
        aux = np.array(y) <= thr
        idx_x = idx[aux]
        okToPlot = True
        resInterp = []
        if not idx_x.any():
            if len(idx_x) > 1:
                idx_2 = idx_x[0]
                idx_1 = idx_2 - 1
                if idx_1 < 0:
                    idx_2 = idx_x[1]
                    idx_1 = idx_2 - 1
                y2 = x[idx_2]
                y1 = x[idx_1]
                x2 = y[idx_2]
                x1 = y[idx_1]
                slope = (y2 - y1) / (x2 - x1)
                ny = y2 - slope * x2
                resInterp = 1.0 / (slope * thr + ny)
            else:
                okToPlot = False
        else:
            okToPlot = False

        return resInterp, okToPlot

    def _plotCurveAnisotropy(self, fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY1, mdLabelY2):
        """
        This function is called by _showAnisotropyCurve
        It shows the FSO curve in terms of the resolution
        The horizontal axis is linear in this plot.
        """
        md = xmippLib.MetaData(fnmd)
        xplotter = XmippPlotter(figure=None)
        xplotter.plot_title_fontsize = 11

        a = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        xplotter.plotMdFile(md, mdLabelX, mdLabelY1, 'g')

        xx, yy = self._prepareDataForPlot(md, mdLabelX, mdLabelY1)
        _, yyBingham = self._prepareDataForPlot(md, mdLabelX, mdLabelY2)

        from matplotlib.ticker import FuncFormatter
        a.axes.xaxis.set_major_formatter(FuncFormatter(self._formatFreq))
        a.axes.set_ylim([-0.1, 1.1])
        a.axes.plot(xx, yy, 'g')
        a.axes.set_xlabel('Resolution (A)')
        a.axes.set_ylabel('FSO (a.u)')
        hthresholds = [0.1, 0.5, 0.9]
        a.axes.hlines(hthresholds, xx[0], xx[-1], colors='k', linestyles='dashed')
        a.axes.grid(True)
        textstr = ''

        res_01, okToPlot_01 = self.interpolRes(0.1, xx, yy)
        res_05, okToPlot_05 = self.interpolRes(0.5, xx, yy)
        res_09, okToPlot_09 = self.interpolRes(0.9, xx, yy)

        textstr = ''

        if okToPlot_09:
           textstr += str(0.9) + ' --> ' + str("{:.2f}".format(res_09)) + 'A\n'
        if okToPlot_05:
           textstr += str(0.5) + ' --> ' + str("{:.2f}".format(res_05)) + 'A\n'
        if okToPlot_01:
           textstr += str(0.1) + ' --> ' + str("{:.2f}".format(res_01)) + 'A'

        props = dict(boxstyle='round', facecolor='white')
        a.axes.text(0.0, 0.0, textstr, fontsize=12, ha="left", va="bottom", bbox=props)
        
        if self.protocol.halfVolumesFile:
           sampling = self.protocol.inputHalves.get().getSamplingRate()
        else:
           sampling = self.protocol.half1.get().getSamplingRate()

        if okToPlot_09 and okToPlot_01:
           t = round((2*sampling/(res_01))*len(yyBingham)) + 3
           if t<len(yyBingham):
              for component in range(t, len(yyBingham)-1):
                 yyBingham[component] = 0
           a.axes.plot(xx, yyBingham, 'r--')
        else:
           a.axes.plot(xx, yyBingham, 'r--')

        if not okToPlot_01:
           res_01 = 2*sampling

        if not okToPlot_09:
           res_09 = 2*sampling

        a.axes.axvspan(1.0 / res_09, 1.0 / res_01, alpha=0.3, color='green')

        return plt.show()

    def _prepareDataForPlot(self, md, mdLabelX, mdLabelY):
        """ plot metadata columns mdLabelX and mdLabelY
            if nbins is in args then and histogram over y data is made
        """
        xx = md.getColumnValues(mdLabelX)
        yy = md.getColumnValues(mdLabelY)
        return xx, yy

    def _showDirectionalResolution(self, zparam=None):
        """
        This function shows the angular distribution of the resolution
        """
        fnmd = self.protocol._getExtraPath('Resolution_Distribution.xmd')
        self._showPolarPlot(fnmd)

    def _showPolarPlot(self, fnmd):
        """
        It is called by _showDirectionalResolution
        This function shows the angular distribution of the resolution
        """
        md = emlib.MetaData(fnmd)

        radius = md.getColumnValues(emlib.MDL_ANGLE_ROT)
        azimuth = md.getColumnValues(emlib.MDL_ANGLE_TILT)
        counts = md.getColumnValues(emlib.MDL_RESOLUTION_FRC)

        # define binning
        azimuths = np.radians(np.linspace(0, 360, 360))
        zeniths = np.arange(0, 91, 1)

        r, theta = np.meshgrid(zeniths, azimuths)

        values = np.zeros((len(azimuths), len(zeniths)))

        for i in range(0, len(azimuth)):
            values[int(radius[i]), int(azimuth[i])] = counts[i]

        # ------ Plot ------
        stp = 0.1
        # the names are wrong in the base class
        formLowLim = self.highest.get()
        formHighLim = self.lowest.get()

        if formLowLim>0 or formHighLim>0:
            lowlim = formLowLim
            highlim = formHighLim
        else:
            lowlim = max(0.0, values.min())
            highlim = values.max() + stp

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        pc = plt.contourf(theta, r, values, np.arange(lowlim, highlim, stp), cmap=self.getColorMap(), extend="max")

        plt.colorbar(pc)
        plt.show()

    def _getAxis(self):
        return self.getEnumText('sliceAxis')

    def getColorMap(self):
        cmap = cm.get_cmap(self.colorMap.get())
        if cmap is None:
            cmap = cm.jet
        return cmap

