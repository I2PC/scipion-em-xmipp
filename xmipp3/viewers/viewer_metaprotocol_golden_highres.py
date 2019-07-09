# **************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), May 2013
# *           Slavica Jonic                (jonic@impmc.upmc.fr)
# * Ported to Scipion:
# *           J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es), Nov 2014
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

from pyworkflow.protocol.params import EnumParam, NumericRangeParam, LabelParam, IntParam, FloatParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pyworkflow.em.viewers import ObjectView, DataView, ChimeraClientView
import pyworkflow.em.viewers.showj as showj

from xmippLib import (MDL_SAMPLINGRATE, MDL_ANGLE_ROT, MDL_ANGLE_TILT,
                   MDL_RESOLUTION_FREQ, MDL_RESOLUTION_FRC, MetaData)
from xmipp3.convert import getImageLocation
from xmipp3.protocols.protocol_metaprotocol_golden_highres import XmippMetaProtGoldenHighRes
from .plotter import XmippPlotter

ITER_LAST = 0
ITER_SELECTION = 1

ANGDIST_2DPLOT = 0
ANGDIST_CHIMERA = 1

VOLUME_SLICES = 0
VOLUME_CHIMERA = 1

class XmippMetaprotocolGoldenHighResViewer(ProtocolViewer):
    """ Visualize the output of protocol reconstruct highres """
    _label = 'viewer golden highres'
    _targets = [XmippMetaProtGoldenHighRes]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('showResolutionPlots', LabelParam, default=True,
                      label='Display resolution plots (FSC)')
        form.addParam('resolutionThreshold', FloatParam, default=0.5,
                      expertLevel=LEVEL_ADVANCED,
                      label='Threshold in resolution plots')

    def _getVisualizeDict(self):
        self._load()
        return {
                'showResolutionPlots': self._showFSC
                }
    
    def _validate(self):
        pass

    def _load(self):
        from matplotlib.ticker import FuncFormatter
        self._plotFormatter = FuncFormatter(self._formatFreq)


    def _showFSC(self, paramName=None):
        self.minInv = []
        self.maxInv = []
        fnFSCs = open(self.protocol._getExtraPath('fnFSCs.txt'), 'r')
        xplotter = XmippPlotter(windowTitle="FSC")
        a = xplotter.createSubPlot("FSC", "Frequency (1/A)", "FSC")
        legends = []
        i=0
        for i, line in enumerate(fnFSCs):
            fnFSC = line[:-2]
            if exists(fnFSC):
                if i<10:
                    legends.append('Group %s' % chr(65+i))
                else:
                    legends.append('Local %d'%(i-9))
                self._plotFSC(a, fnFSC)
                xplotter.showLegend(legends)
            i=i+1
        a.plot([min(self.minInv), max(self.maxInv)],[self.resolutionThreshold.get(), self.resolutionThreshold.get()], color='black', linestyle='--')
        a.grid(True)
        views = []
        views.append(xplotter)
        return views

    def _plotFSC(self, a, fnFSC):
        md = MetaData(fnFSC)
        resolution_inv = [md.getValue(MDL_RESOLUTION_FREQ, f) for f in md]
        frc = [md.getValue(MDL_RESOLUTION_FRC, f) for f in md]
        self.maxFrc = max(frc)
        self.minInv.append(min(resolution_inv))
        self.maxInv.append(max(resolution_inv))
        a.plot(resolution_inv, frc)
        a.xaxis.set_major_formatter(self._plotFormatter)
        a.set_ylim([-0.1, 1.1])

    def _formatFreq(self, value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999
        if value:
            inv = 1/value
        return "1/%0.2f" % inv


