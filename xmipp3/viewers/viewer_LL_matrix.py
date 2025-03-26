# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Slavica Jonic  (slavica.jonic@upmc.fr)
# *              James Krieger (jmkrieger@cnb.csic.es)
# *              Ricardo Serrano Gutiérrez (rserranogut@hotmail.com)  
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
"""
This module implement the wrappers around ProDy GNM 
visualization programs.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

from pwem.viewers.plotter import EmPlotter
from pwem.objects.data import Volume

from pyworkflow.protocol.params import LabelParam, IntParam, FloatParam, BooleanParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO

from xmipp3.protocols.protocol_compute_likelihood import XmippProtComputeLikelihood

_invalidInputStr = 'Invalid input'

class XmippLogLikelihoodViewer(ProtocolViewer):
    """ Visualization of results from the Xmipp log likelihood protocol.
    """
    _label = 'Log likelihood matrix viewer'
    _targets = [XmippProtComputeLikelihood]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def _defineParams(self, form):

        self.particles = self.protocol.inputParticles.get()
        self.refs = self.protocol.inputRefs.get()
        self.outputs = self.protocol.reprojections

        if isinstance(self.refs, Volume):
            self.refs = [self.refs]

        form.addSection(label='Visualization')

        group = form.addGroup('Particles range')
        group.addParam('partNumber1', IntParam, default=-1,
                      label='Initial particle id',
                      help='')
        group.addParam('partNumber2', IntParam, default=-1,
                      label='Final particle id')
        
        group = form.addGroup('Volumes range')
        group.addParam('volNumber1', IntParam, default=-1,
                      label='Initial volume number')
        group.addParam('volNumber2', IntParam, default=-1,
                      label='Final volume number')

        group = form.addGroup('Values for matrix plot')
        group.addParam('normalise', BooleanParam, default=False,
                      label='Normalise LL matrix by dividing by mean value of each column?',
                      help='This may increase the contrast to help with interpretability. '
                           'Note: we keep the sign negative to keep the order of values.')

        group.addParam('subtract', BooleanParam, default=True,
                      label='Shift LL matrix by subtracting mean value of each column?',
                      help='This may increase the contrast to help with interpretability. ')

        group.addParam('vmin', FloatParam, default=-1,
                      label='Minimum blue value',
                      help='All values below this will be coloured blue if not -1')
        group.addParam('vmax', FloatParam, default=-1,
                      label='Maximum yellow value',
                      help='All values above this will be coloured yellow if not -1')
        group.addParam('percentile', FloatParam, default=-1,
                      label='Percentile alternative to vmin and vmax',
                      help='If either of the above values is -1, they will be estimated as this percentile')

        group.addParam('displayLL', LabelParam, default=False,
                label="Plot log likelihood matrix?",
                help="Matrices are shown as heatmaps.")
        
        group = form.addGroup('Values for histogram')
        group.addParam('nBins', IntParam, default=100, label='Number of bins for histogram')

        group.addParam('vminH', FloatParam, default=-1,
                      label='Minimum axis value',
                      help='All values below this will be cropped off the axis if not -1')
        group.addParam('vmaxH', FloatParam, default=-1,
                      label='Maximum axis value',
                      help='All values above this will be cropped off the axis if not -1')
        group.addParam('percentileH', FloatParam, default=-1,
                      label='Percentile alternative to axis vmin and vmax',
                      help='If either of the above values is -1, they will be estimated as this percentile')

        group.addParam('displayRelativeLL', LabelParam, default=False,
                label="Plot relative log likelihood histogram?",
                help="Subtracted log likelihood is shown as a histogram.")

    def _getVisualizeDict(self):
        return {'displayLL': self._viewLL,
                'displayRelativeLL': self._viewRelativeLL}

    def _viewLL(self, paramName):
        """ visualization of log likelihood matrix for selected particles and ref volumes."""
        x = self._checkNumbers('particle')
        if x is not True:
            return x

        x = self._checkNumbers('volume')
        if x is not True:
            return x
        
        matrix = np.load(self.protocol._getExtraPath('matrix.npy'))

        if self.normalise.get():
            matrix = np.divide(matrix, np.mean(matrix, axis=0)) * np.sign(np.mean(matrix))

        elif self.subtract.get():
            matrix = np.subtract(matrix, np.mean(matrix, axis=0))

        if self.vmin.get() != -1:
            vmin = self.vmin.get()
        elif self.percentile.get() != -1:
            vmin = np.percentile(matrix, self.percentile.get())
        else:
            vmin = None

        if self.vmax.get() != -1:
            vmax = self.vmax.get()
        elif self.percentile.get() != -1:
            vmax = np.percentile(matrix, 100-self.percentile.get())
        else:
            vmax = None

        plotter = EmPlotter()
        im = plt.imshow(matrix, aspect='auto',
                        vmin=vmin, vmax=vmax)
        plt.colorbar(mappable=im)

        plt.ylabel('Reference volumes')
        plt.ylim([self.volNum1-1.5, self.volNum2-0.5])
        ylocs, _ = plt.yticks()
        plt.yticks([int(loc) for loc in ylocs[:-1] if loc >= self.volNum1-1],
                   [str(int(loc)+1) for loc in ylocs[:-1] if loc >= self.volNum1-1])

        plt.xlabel('Input particles')
        plt.xlim([self.partNum1-1.5, self.partNum2-0.5])
        xlocs, _ = plt.xticks()
        plt.xticks([int(loc) for loc in xlocs[:-1] if loc >= self.partNum1-1],
                   [str(int(loc)+1) for loc in xlocs[:-1] if loc >= self.partNum1-1])

        return [plotter]

    def _viewRelativeLL(self, paramName):
        """ visualization of relative log likelihood histogram for  for selected particles and ref volumes."""
        x = self._checkNumbers('particle')
        if x is not True:
            return x

        x = self._checkNumbers('volume')
        if x is not True:
            return x

        if isinstance(self.refs, Volume):
            self.refs = [self.refs]

        matrix = np.load(self.protocol._getExtraPath('matrix.npy'))
        matrix = np.subtract(matrix[self.volNum1-1, self.partNum1-1:self.partNum2],
                             matrix[self.volNum2-1, self.partNum1-1:self.partNum2])

        plotter = EmPlotter()
        _ = plt.hist(matrix, bins=self.nBins.get())
        plt.xlabel('Relative log likelihood')

        if self.vminH.get() != -1:
            vmin = self.vminH.get()
        elif self.percentileH.get() != -1:
            vmin = np.percentile(matrix, self.percentileH.get())
        else:
            vmin = None

        if self.vmaxH.get() != -1:
            vmax = self.vmaxH.get()
        elif self.percentileH.get() != -1:
            vmax = np.percentile(matrix, 100-self.percentileH.get())
        else:
            vmax = None

        plt.xlim([vmin, vmax])

        return [plotter]


    def _checkNumbers(self, string):

        if self.normalise.get() and self.subtract.get():
                return [self.errorMessage("Please select normalise or subtract, not both",
                                        title=_invalidInputStr)]

        items, number1, number2 = self._setNumbers(string)

        if len(self.refs) == 1:
            return True

        if number1+1 > number2:
            return [self.errorMessage("Invalid {0} range\n"
                                      "Initial {0} number can not be " 
                                      "bigger than the final one.".format(string), 
                                      title=_invalidInputStr)]

        elif number1 < -1:
            return [self.errorMessage("Invalid {0} range\n"
                                      "Initial {0} number can not be " 
                                      "smaller than -1.".format(string), 
                                      title=_invalidInputStr)]

        elif number2 < -1:
            return [self.errorMessage("Invalid {0} range\n"
                                      "Final {0} number can not be " 
                                      "smaller than -1.".format(string), 
                                      title=_invalidInputStr)]
        
        if number1 != -1:
            try:
                _ = items[number1]
            except IndexError:
                return [self.errorMessage("Invalid initial {0} number {1}\n"
                                         "Display the output {0}s to see "
                                         "the availables ones.".format(string, number1+1),
                                         title=_invalidInputStr)]
            
        if number2 != -1:
            try:
                _ = items[number2-1]
            except IndexError:
                return [self.errorMessage("Invalid final {0} number {1}\n"
                                         "Display the output {0}s to see "
                                         "the availables ones.".format(string, number2),
                                         title=_invalidInputStr)]

        return True

    def _setNumbers(self, string):

        if string == 'particle':
            items = self.particles
            number1 = self.partNum1 = self.partNumber1.get() if self.partNumber1.get() != -1 else 1
            number2 = self.partNum2 = self.partNumber2.get() if self.partNumber2.get() != -1 else len(self.particles)
        else:
            items = self.refs
            number1 = self.volNum1 = self.volNumber1.get() if self.volNumber1.get() != -1 else 1
            number2 = self.volNum2 = self.volNumber2.get() if self.volNumber2.get() != -1 else len(self.refs)

        return items, number1, number2
