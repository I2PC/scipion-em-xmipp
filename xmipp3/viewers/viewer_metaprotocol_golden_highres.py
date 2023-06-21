# **************************************************************************
# *
# * Authors:  Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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

from glob import glob
from os.path import exists, join

from pyworkflow.protocol.params import EnumParam, NumericRangeParam, LabelParam, \
    IntParam, FloatParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pwem.viewers import ObjectView, DataView
import pwem.viewers.showj as showj
from pwem import emlib

from xmipp3.protocols.protocol_metaprotocol_golden_highres import \
    XmippMetaProtGoldenHighRes
from .plotter import XmippPlotter

class XmippMetaprotocolGoldenHighResViewer(ProtocolViewer):
    """ Visualize the output of protocol reconstruct highres """
    _label = 'viewer golden highres'
    _targets = [XmippMetaProtGoldenHighRes]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('showResolutionPlotsGlob', LabelParam, default=True,
                      label='Display resolution plots of global iterations (FSC)')
        form.addParam('showResolutionPlotsLoc', LabelParam, default=True,
                      label='Display resolution plots of local iterations (FSC)')
        form.addParam('resolutionThreshold', FloatParam, default=0.5,
                      expertLevel=LEVEL_ADVANCED,
                      label='Threshold in resolution plots')

    def _getVisualizeDict(self):
        self._load()
        return {
            'showResolutionPlotsGlob': self._showFSCGlob,
            'showResolutionPlotsLoc': self._showFSCLoc
        }

    def _validate(self):
        pass

    def _load(self):
        from matplotlib.ticker import FuncFormatter
        self._plotFormatter = FuncFormatter(self._formatFreq)

    def _showFSCGlob(self, paramName=None):
        self.minInv = []
        self.maxInv = []
        fnFSCs = open(self.protocol._getExtraPath('fnFSCs.txt'), 'r')
        title="FSC of global iterations"
        xplotter = XmippPlotter(windowTitle=title)
        a = xplotter.createSubPlot("FSC", "Frequency (1/A)", "FSC")
        legends = []
        for i, line in enumerate(fnFSCs):
            if i>=10:
                continue
            fnFSC = line[:-2]
            if exists(fnFSC):
                legends.append('Group %s' % chr(65 + i))
                self._plotFSC(a, fnFSC)
                xplotter.showLegend(legends)

        a.plot([min(self.minInv), max(self.maxInv)],
               [self.resolutionThreshold.get(), self.resolutionThreshold.get()],
               color='black', linestyle='--')
        a.grid(True)
        views = []
        views.append(xplotter)
        return views


    def _showFSCLoc(self, paramName=None):
        self.minInv = []
        self.maxInv = []
        fnFSCs = open(self.protocol._getExtraPath('fnFSCs.txt'), 'r')
        title="FSC of local iterations"
        xplotter = XmippPlotter(windowTitle=title)
        a = xplotter.createSubPlot("FSC", "Frequency (1/A)", "FSC")
        legends = []
        for i, line in enumerate(fnFSCs):
            if i<10:
                continue
            fnFSC = line[:-2]
            if exists(fnFSC):
                legends.append('Group %s' % chr(65 + i))
                self._plotFSC(a, fnFSC)
                xplotter.showLegend(legends)

        a.plot([min(self.minInv), max(self.maxInv)],
               [self.resolutionThreshold.get(), self.resolutionThreshold.get()],
               color='black', linestyle='--')
        a.grid(True)
        views = []
        views.append(xplotter)
        return views


    def _plotFSC(self, ab, fnFSC):
        md = emlib.MetaData(fnFSC)
        resolution_inv = [md.getValue(emlib.MDL_RESOLUTION_FREQ, f) for f in md]
        frc = [md.getValue(emlib.MDL_RESOLUTION_FRC, f) for f in md]
        # self.maxFrc = max(frc)
        self.minInv.append(min(resolution_inv))
        self.maxInv.append(max(resolution_inv))
        ab.plot(resolution_inv, frc)
        ab.xaxis.set_major_formatter(self._plotFormatter)
        ab.set_ylim([-0.1, 1.1])

    def _formatFreq(self, value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999
        if value:
            inv = 1 / value
        return "1/%0.2f" % inv