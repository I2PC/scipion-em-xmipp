# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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

import os, matplotlib, math
from scipy import ndimage
import tkinter as tk
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pyworkflow.gui.plotter import plt
import pyworkflow.protocol.params as params

from xmipp3.protocols.protocol_structure_map import XmippProtStructureMap


class XmippProtStructureMapViewer(ProtocolViewer):
    """ Wrapper to visualize different type of data objects
    with the Xmipp program xmipp_showj
    """
    
    _label = 'viewer structure map'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtStructureMap]


    def _defineParams(self, form):
        form.addSection(label='Show StructMap')
        form.addParam('doShowPlot', params.LabelParam,
                      label="Display the StructMap")
    
    def _getVisualizeDict(self):
        return {'doShowPlot': self._visualize}
    
        
    def _visualize(self, e=None):
        fnOutput = self.protocol._defineResultsName(3)
        if not os.path.exists(fnOutput):
            return [self.errorMessage('The necessary metadata was not produced\n'
                                      'Execute again the protocol\n',
                                      title='Missing result file')]
        coordinates = np.loadtxt(fnOutput)
        if os.path.isfile(self._getExtraPath('weigths.txt')):
            weights = np.loadtxt(self._getExtraPath('weigths.txt'))
        else:
            weights = None
        plot = projectionPlot(coordinates, weights)
        plot.initializePlot()
        return plot
        
    def _validate(self):
        errors = []
        return errors


class projectionPlot(object):

    def __init__(self, coords, weights=None):
        self.coords = coords
        self.weights = weights
        self.proj_coords = None
        self.radio = None
        self.cb = None
        self.prevlabel = 'Scatter'
        self.root = tk.Tk()
        self.fig = plt.Figure(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.ax_3d = self.fig.add_subplot(121, projection="3d")
        self.ax_3d.set_title("3D Scatter Plot")
        self.fig.canvas.mpl_connect('button_release_event', self.onRelease)
        self.ax_2d = self.fig.add_subplot(122)
        self.ax_2d.set_title("Projection Scatter Plot")

    def projectMatrix(self, M, coords):
        proj_coords = []
        for coord in coords:
            coord = np.append(coord, 1.0)
            proj_coord = M.dot(coord)
            proj_coords.append(proj_coord)
        return np.asarray(proj_coords)

    def onRelease(self, event):
        if event.inaxes == self.ax_3d:
            M = event.inaxes.get_proj()
            self.proj_coords = self.projectMatrix(M, self.coords)
            self.plotType(self.prevlabel)

    def plotType(self, label):
        self.prevlabel = label
        x = self.proj_coords[:, 0]
        y = self.proj_coords[:, 1]
        if self.cb != None:
            self.cb.remove()
            self.cb = None
        if label == 'Scatter':
            self.ax_2d.clear()
            self.ax_2d.scatter(x, y, color="green")
            self.ax_2d.set_title("Projection Scatter Plot")
            self.fig.canvas.draw()
        elif label == 'Contour':
            self.ax_2d.clear()
            rangeX = np.max(x) - np.min(x)
            rangeY = np.max(y) - np.min(y)
            if rangeX > rangeY:
                sigma = rangeX / 50
            else:
                sigma = rangeY / 50
            xi = np.linspace(min(x) - 0.1, max(x) + 0.1, 100)
            yi = np.linspace(min(x) - 0.1, max(x) + 0.1, 100)
            z = np.zeros((100, 100), float)
            zSize = z.shape
            N = len(x)
            for c in range(zSize[1]):
                for r in range(zSize[0]):
                    for d in range(N):
                        z[r, c] = z[r, c] + (1.0 / N) * (1.0 / ((2 * math.pi) * sigma ** 2)) * math.exp(
                            -((xi[c] - x[d]) ** 2 + (yi[r] - y[d]) ** 2) / (2 * sigma ** 2))
            zMax = np.max(z)
            z = z / zMax
            self.ax_2d.contour(xi, yi, z, 15, linewidths=0.5, colors='k')
            cf = self.ax_2d.contourf(xi, yi, z, 15, cmap=plt.cm.jet)
            self.ax_2d.set_title("Projection Scatter Plot")
            cbaxes = self.fig.add_axes([0.92, 0.1, 0.01, 0.8])
            self.cb = self.fig.colorbar(mappable=cf, cax=cbaxes)
            self.cb.set_ticks([])
            self.fig.canvas.draw()
        elif label == 'Convolution':
            coordinates = np.stack((x, y), axis=1)
            Xr = np.round(coordinates, decimals=3)
            size_grid = 2 * np.amax(Xr)
            grid_coords = np.arange(-size_grid, size_grid, 0.001)
            R, C = np.meshgrid(grid_coords, grid_coords, indexing='ij')
            S = np.zeros(R.shape)
            sigma = R.shape[0] / (200 / 5)
            lbox = int(6 * sigma)
            if lbox % 2 == 0:
                lbox += 1
            mid = int((lbox - 1) / 2 + 1)
            kernel = np.zeros((lbox, lbox))
            kernel[mid, mid] = 1
            kernel = gaussian_filter(kernel, sigma=sigma)
            for p in range(Xr.shape[0]):
                indx = np.argmin(np.abs(R[:, 0] - Xr[p, 0]))
                indy = np.argmin(np.abs(C[0, :] - Xr[p, 1]))
                if self.weights != None:
                    S[indx - mid:indx + mid - 1, indy - mid:indy + mid - 1] += kernel * self.weights[p]
                else:
                    S[indx - mid:indx + mid - 1, indy - mid:indy + mid - 1] += kernel
            S = S[~np.all(S == 0, axis=1)]
            S = S[:, ~np.all(S == 0, axis=0)]
            S = ndimage.rotate(S, 90)
            cf = self.ax_2d.imshow(S)
            cbaxes = self.fig.add_axes([0.92, 0.1, 0.01, 0.8])
            self.ax_2d.set_title('Projection Scatter Plot')
            self.cb = self.fig.colorbar(mappable=cf, cax=cbaxes)
            self.cb.set_ticks([])
            self.fig.canvas.draw()

    def initializePlot(self):
        self.ax_3d.scatter3D(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], color="green")
        M = self.ax_3d.get_proj()
        self.proj_coords = self.projectMatrix(M, self.coords)
        self.ax_2d.scatter(self.proj_coords[:, 0], self.proj_coords[:, 1], color="green")

        # Buttons
        axcolor = 'silver'
        rax = self.fig.add_axes([0.01, 0.4, 0.12, 0.25], facecolor=axcolor)
        self.radio = RadioButtons(rax, ('Scatter', 'Contour', 'Convolution'))
        self.radio.on_clicked(self.plotType)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        tk.mainloop()
