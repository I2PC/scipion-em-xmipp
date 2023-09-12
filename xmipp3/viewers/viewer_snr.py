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
from pyworkflow.protocol.params import LabelParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pyworkflow.utils import getExt, removeExt

from pwem.emlib.metadata import MetaData
from pwem.viewers import LocalResolutionViewer
from pwem import emlib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

from xmipp3.protocols.protocol_snr import XmippProtSNR
from xmipp3.viewers.plotter import XmippPlotter
import xmippLib


class XmippProtSNRViewer(LocalResolutionViewer):
    """
    Visualization tools for the FSO, FSC, and 3DFSC.
    
    """
    _label = 'viewer FSO'
    _targets = [XmippProtSNR]
    _environments = [DESKTOP_TKINTER]

    def _defineParams(self, form):
        form.addSection(label='Visualization')

        form.addParam('doShowSNRHistogram', LabelParam,
                             label="Show SNR histogram")

        form.addParam('doShowSNRCurve', LabelParam,
                      label="Show SNR curve")

    def _getVisualizeDict(self):
        return {'doShowSNRHistogram': self._showSNRHistogram,
                'doShowSNRCurve': self._showSNREvolution,
                }

    def _showSNRHistogram(self, paramName=None):
        """
        It shows the FSC curve in terms of the resolution
        The horizontal axis is linear in this plot. Note
        That this is not the normal representation of hte FSC
        """
        fnmd = self.protocol._getExtraPath('SNR.xmd')
        title = 'Histogram of SNR'
        xlabel = 'SNR'
        mdLabelX = emlib.MDL_RESOLUTION_SSNR
        md = xmippLib.MetaData(fnmd)
        xx = md.getColumnValues(mdLabelX)
        plt.hist(xx)
        plt.title(title)
        plt.xlabel(xlabel)
        return plt.show()

    def _showSNREvolution(self, paramName=None):
        """
        It shows the FSC curve in terms of the resolution
        The horizontal axis is linear in this plot. Note
        That this is not the normal representation of hte FSC
        """
        fnmd = self.protocol._getExtraPath('SNR.xmd')
        title = 'Histogram of SNR'
        xlabel = 'SNR'
        mdLabelX = emlib.MDL_IDX
        mdLabelY = emlib.MDL_RESOLUTION_FRC
        md = xmippLib.MetaData(fnmd)
        xx = md.getColumnValues(mdLabelX)
        yy = md.getColumnValues(mdLabelY)
        plt.plot(xx, yy)
        plt.title(title)
        plt.xlabel(xlabel)
        return plt.show()
