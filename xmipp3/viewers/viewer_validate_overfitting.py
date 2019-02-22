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
from xmipp3.protocols.protocol_validate_overfitting import \
    XmippProtValidateOverfitting
from .plotter import XmippPlotter
from os.path import exists
from math import log10


class XmippValidateOverfittingViewer(Viewer):
    """ Wrapper to visualize different type of data objects
    with the Xmipp program xmipp_showj
    """
    _label = 'viewer validate_overfitting'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtValidateOverfitting]

    def _visualize(self, e=None):
        views=[]
        fnOutput = self.protocol._defineResultsTxt()
        if not os.path.exists(fnOutput):
            return [
                self.errorMessage('The necessary metadata was not produced\n'
                                  'Execute again the protocol\n',
                                  title='Missing result file')]
        plotter = self._createPlotInv()
        views.append(plotter)

        plotter = self._createPlot()
        views.append(plotter)

        # for noise
        # fnOutputN = self.protocol._defineResultsNoiseName()
        # if not os.path.exists(fnOutputN):
        #    return [self.errorMessage('The necessary metadata was not produced\n'
        #                              'Execute again the protocol\n',
        #                              title='Missing noise result file')]

        return views

    def _createPlotInv(self, figure=None):

        xplotter = XmippPlotter(figure=figure)
        xplotter.plot_title_fontsize = 11
        title = 'Validation 3D Reconstruction (Overfitting)'
        xTitle = 'log10(# Particles)'
        yTitle = '1/Resolution^2 in 1/A^2'
        ax = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)

        # plot real data-set
        fnOutput = self.protocol._defineResultsTxt()
        if exists(fnOutput):
            fileValues = open(fnOutput,'r')
            xValInv=[]
            yValInv=[]
            yErrInv=[]
            for line in fileValues:
                values = line.split()
                xValInv.append(log10(float(values[0])))
                yValInv.append(float(values[3]))
                yErrInv.append(float(values[4]))

            plt.plot(xValInv, yValInv, color='g', label='Aligned particles')
            plt.errorbar(xValInv, yValInv, yErrInv, fmt='o')
            plt.legend(loc='upper right', fontsize=11)

        # plot noise and related errorbar
        fnOutputN = self.protocol._defineResultsNoiseTxt()
        if exists(fnOutputN):
            fileNoise = open(fnOutputN,'r')
            yValNInv=[]
            yErrNInv=[]
            for line in fileNoise:
                values = line.split()
                yValNInv.append(float(values[3]))
                yErrNInv.append(float(values[4]))

            plt.plot(xValInv, yValNInv, '--', color='r',
                     label='Aligned gaussian noise')
            plt.errorbar(xValInv, yValNInv, yErrNInv, fmt='o', color='k')

        return xplotter


    def _createPlot(self, figure=None):

        xplotter = XmippPlotter(figure=figure)
        xplotter.plot_title_fontsize = 11
        title = 'Validation 3D Reconstruction (Overfitting)'
        xTitle = '# Particles'
        yTitle = 'Resolution in A'
        ax = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        ax.set_yscale('log')
        ax.set_xscale('log')

        # plot real data-set
        fnOutput = self.protocol._defineResultsTxt()
        if exists(fnOutput):
            fileValues = open(fnOutput,'r')
            xVal=[]
            yVal=[]
            yErr=[]
            for line in fileValues:
                values = line.split()
                xVal.append(float(values[0]))
                yVal.append(float(values[1]))
                yErr.append(float(values[2]))

            plt.plot(xVal, yVal, color='g', label='Aligned particles')
            plt.errorbar(xVal, yVal, yErr, fmt='o')
            plt.legend(loc='upper right', fontsize=11)

        # plot noise and related errorbar
        fnOutputN = self.protocol._defineResultsNoiseTxt()
        if exists(fnOutputN):
            fileNoise = open(fnOutputN,'r')
            yValN=[]
            yErrN=[]
            for line in fileNoise:
                values = line.split()
                yValN.append(float(values[1]))
                yErrN.append(float(values[2]))

            plt.plot(xVal, yValN, '--', color='r',
                     label='Aligned gaussian noise')
            plt.errorbar(xVal, yValN, yErrN, fmt='o', color='k')

        return xplotter
