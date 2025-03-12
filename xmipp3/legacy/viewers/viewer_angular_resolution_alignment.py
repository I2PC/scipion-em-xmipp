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


from pyworkflow.protocol.params import LabelParam, StringParam, EnumParam, IntParam, LEVEL_ADVANCED
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER

from pwem.emlib.metadata import MetaData
from pwem import emlib


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

from xmipp3.protocols.protocol_angular_resolution_alignment import \
    XmippProtResolutionAlignment, RADIAL_RESOLUTION_FN
from xmipp3.viewers.plotter import XmippPlotter
import xmippLib


class XmippProtAngResAlignViewer(ProtocolViewer):
    """
    Visualization tools for the validation of alignment based on resolution.
    
    """
    _label = 'viewer angular resolution alignment'
    _targets = [XmippProtResolutionAlignment]
    _environments = [DESKTOP_TKINTER]

    @staticmethod
    def getColorMapChoices():
        return plt.colormaps()

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        groupDirFSC = form.addGroup('Resolution Analysis')
        groupDirFSC.addParam('doShowRadialResolution', LabelParam, label="Radial FSC")

    def _getVisualizeDict(self):
        return {'doShowRadialResolution': self._showRadialResolution}

    def _showRadialResolution(self, params=None):
        """
        It shows the FSC curve in terms of the resolution
        The horizontal axis is linear in this plot. Note
        That this is not the normal representation of the FSC
        """
        fnmd = self.protocol._getExtraPath(RADIAL_RESOLUTION_FN)
        title = 'Radial Average of directional resolution'
        xTitle = 'Radius (px)'
        yTitle = 'Resolution (A)'
        mdLabelX = emlib.MDL_IDX
        mdLabelY = emlib.MDL_RESOLUTION_FRC
        self._plotCurveFSC(fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY)

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

        xx, yy = self._prepareDataForPlot(md, mdLabelX, mdLabelY)
        a.grid(True)

        return plt.show()

    def _formatFreq(self, value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999.
        if value:
            inv = 1 / value
        return "1/%0.2f" % inv

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
            okToPlot = False
        else:
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

        return resInterp, okToPlot


    def _prepareDataForPlot(self, md, mdLabelX, mdLabelY):
        """ plot metadata columns mdLabelX and mdLabelY
            if nbins is in args then and histogram over y data is made
        """
        xx = md.getColumnValues(mdLabelX)
        yy = md.getColumnValues(mdLabelY)
        return xx, yy

