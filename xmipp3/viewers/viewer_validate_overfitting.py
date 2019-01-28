# **************************************************************************
# *
# * Authors:     Mohsen Kazemi  (mkazemi@cnb.csic.es)
# *              C.O.S. Sorzano (coss@cnb.csic.es)
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

from pyworkflow.viewer import Viewer, DESKTOP_TKINTER, WEB_DJANGO
import pyworkflow.em.metadata as md
from pyworkflow.gui.plotter import plt

import xmippLib
from xmipp3.protocols.protocol_validate_overfitting import XmippProtValidateOverfitting
from .plotter import XmippPlotter


class XmippValidateOverfittingViewer(Viewer):
    """ Wrapper to visualize different type of data objects
    with the Xmipp program xmipp_showj
    """
    _label = 'viewer validate_overfitting'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtValidateOverfitting]
        
    def _visualize(self, e=None):
        fnOutput = self.protocol._defineResultsName()
        if not os.path.exists(fnOutput):
            return [self.errorMessage('The necessary metadata was not produced\n'
                                      'Execute again the protocol\n',
                                      title='Missing result file')]
        plotter = self._createPlot("Validation 3D Reconstruction (Overfitting)",
                                    "Number of Particles", 
                                    "Resolution for FSC=0.5 (A)", 
                                    fnOutput, xmippLib.MDL_COUNT, xmippLib.MDL_AVG)
        #for noise
        fnOutputN = self.protocol._defineResultsNoiseName()
        if not os.path.exists(fnOutputN):
            return [self.errorMessage('The necessary metadata was not produced\n'
                                      'Execute again the protocol\n',
                                      title='Missing noise result file')]
               
        return [plotter]         
               
    def _createPlot(self, title, xTitle, yTitle, fnOutput, mdLabelX,
                    mdLabelY, color = 'g', figure=None):        
        xplotter = XmippPlotter(figure=figure)
        xplotter.plot_title_fontsize = 11
        ax=xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        ax.set_yscale('log')
        ax.set_xscale('log')
                        
        #plot noise and related errorbar
        fnOutputN = self.protocol._defineResultsNoiseName()
        md = xmippLib.MetaData(fnOutputN)
        xValueN = md.getColumnValues(xmippLib.MDL_COUNT)
        yValueN = md.getColumnValues(xmippLib.MDL_AVG)
        plt.plot(xValueN, yValueN, '--', color='r',
                label='Aligned gaussian noise')
        
        # putting error bar
        md = xmippLib.MetaData(fnOutputN)
        yErrN = md.getColumnValues(xmippLib.MDL_STDDEV)
        xValueNe = md.getColumnValues(xmippLib.MDL_COUNT)
        yValueNe = md.getColumnValues(xmippLib.MDL_AVG)
        plt.errorbar(xValueNe, yValueNe, yErrN, fmt='o', color='k')
                
        #plot real data-set
        fnOutput = self.protocol._defineResultsName()
        md = xmippLib.MetaData(fnOutput)
        xValue = md.getColumnValues(xmippLib.MDL_COUNT)
        yValue = md.getColumnValues(xmippLib.MDL_AVG)
        plt.plot(xValue, yValue, color='g', label='Aligned particles')
                
        # putting error bar 
        md = xmippLib.MetaData(fnOutput)
        yErr = md.getColumnValues(xmippLib.MDL_STDDEV)
        xValue = md.getColumnValues(xmippLib.MDL_COUNT)
        yValue = md.getColumnValues(xmippLib.MDL_AVG)
        plt.errorbar(xValue, yValue, yErr, fmt='o')        
            
        plt.legend(loc='upper right' , fontsize = 11)
        
        return xplotter
    
    