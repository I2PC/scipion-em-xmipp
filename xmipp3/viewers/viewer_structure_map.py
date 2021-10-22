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
from scipy.spatial import KDTree
import tkinter as tk
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pyworkflow.gui.plotter import plt
import pyworkflow.protocol.params as params

from xmipp3.protocols.protocol_structure_map import XmippProtStructureMap
from xmipp3.protocols.protocol_structure_map_zernike3d import XmippProtStructureMapZernike3D


class XmippProtStructureMapViewer(ProtocolViewer):
    """ Wrapper to visualize different type of data objects
    with the Xmipp program xmipp_showj
    """
    
    _label = 'viewer structure map'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtStructureMap, XmippProtStructureMapZernike3D]

    def _defineParams(self, form):
        form.addSection(label='Show StructMap')
        if isinstance(self.protocol, XmippProtStructureMapZernike3D):
            form.addParam('map', params.EnumParam, choices=['Deformation', 'Correlation', 'Consensus'],
                          default=0, help='Choose the type of metric to display the coordinates.')
            form.addParam('twoSets', params.BooleanParam, default=False, label='Two set analysis',
                          help='Activate if the analysis of the deformation included the option of analysing '
                               'two sets of volumes independently.')
        form.addParam('doShowPlot', params.LabelParam,
                      label="Display the StructMap")
    
    def _getVisualizeDict(self):
        return {'doShowPlot': self._visualize}

    def getOutputFile(self):
        fnOutput = ['']
        if isinstance(self.protocol, XmippProtStructureMapZernike3D):
            if self.map.get() == 0 and not self.twoSets:
                fnOutput = [self.protocol._defineResultsName(3)]
            elif self.map.get() == 1 and not self.twoSets:
                fnOutput = [self.protocol._defineResultsName2(3)]
            elif self.map.get() == 0 and self.twoSets:
                fnOutput = [self.protocol._defineResultsName(3, 'Sub_1_'),
                            self.protocol._defineResultsName(3, 'Sub_2_')]
            elif self.map.get() == 1 and self.twoSets:
                fnOutput = [self.protocol._defineResultsName2(3, 'Sub_1_'),
                            self.protocol._defineResultsName2(3, 'Sub_2_')]
            elif self.map.get() == 2 and not self.twoSets:
                fnOutput = [self.protocol._defineResultsName3(3)]
        else:
            fnOutput = [self.protocol._defineResultsName(3)]
        return fnOutput

    def _visualize(self, e=None):
        fnOutput = self.getOutputFile()
        for file in fnOutput:
            if not os.path.exists(file):
                return [self.errorMessage('The necessary metadata was not produced\n'
                                          'Execute again the protocol\n',
                                          title='Missing result file')]

        self.coordinates = (np.loadtxt(file) for file in fnOutput)
        self.coordinates = np.vstack(self.coordinates)
        labels = [str(idp) for idp in range(1, self.coordinates.shape[0] + 1)]
        if os.path.isfile(self._getExtraPath('weigths.txt')):
            weights = np.loadtxt(self._getExtraPath('weigths.txt'))
        else:
            weights = None
        plot = projectionPlot(self.coordinates, labels, weights)
        plot.initializePlot()
        return plot
        
    def _validate(self):
        errors = []
        return errors


class projectionPlot(object):

    def __init__(self, coords, labels, weights):
        self.coords = coords
        self.labels = labels
        self.weights = weights
        try:
            self.minimum_spanning_tree()
        except IndexError:
            self.T = None
        self.proj_coords = None
        self.radio = None
        self.cb = None
        self.prevlabel = 'Scatter'
        self.root = tk.Tk()
        self.fig = plt.Figure(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.ax_3d = self.fig.add_subplot(121, projection="3d")
        self.ax_3d.set_title("3D view")
        self.ax_3d.set_axis_off()
        self.fig.canvas.mpl_connect('button_release_event', self.onRelease)
        self.ax_2d = self.fig.add_subplot(122)
        self.ax_2d.set_title("Projection from current 3D view")

    def minimum_spanning_tree(self):
        N = self.coords.shape[0]
        tree = KDTree(self.coords)
        distances, indices = tree.query(self.coords, k=10)
        nn_mat = np.zeros((N, N))
        for idn in range(N):
            nn_mat[idn, indices[idn]] += distances[idn].reshape(-1)
            nn_mat[idn, idn] = 0

        edges = ((int(e[0]), int(e[1])) for e in zip(*np.asarray(nn_mat).nonzero()))
        triples = list(((u, v, float(nn_mat[u, v])) for u, v in edges))
        edges = [(triple[0], triple[1]) for triple in triples]
        weigths_edge = [triple[2] for triple in triples]

        edge_matrix = np.zeros((N, N))
        for edge, weigth in zip(edges, weigths_edge):
            edge_matrix[edge] = weigth
        self.T = self.KruskalMST(triples, N)

    def KruskalMST(self, triples, N):
        result = []
        i = 0
        e = 0
        graph = sorted(triples, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(N):
            parent.append(node)
            rank.append(0)
        while e < N - 1:
            u, v, w = graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        return result

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def mst_3D(self):
        N = self.coords.shape[0]
        degree = [0] * N
        edges_mst = [0] * (N - 1)
        if self.T is not None:
            for idn in range(N - 1):
                degree[self.T[idn][0]] += 1
                degree[self.T[idn][1]] += 1
                edges_mst[idn] = (self.T[idn][0], self.T[idn][1])
            edge_max = max(degree)
            colors = [plt.cm.plasma(val / edge_max) for val in degree]
            for edge in edges_mst:
                if edge != 0:
                    x = np.array((self.coords[edge[0]][0], self.coords[edge[1]][0]))
                    y = np.array((self.coords[edge[0]][1], self.coords[edge[1]][1]))
                    z = np.array((self.coords[edge[0]][2], self.coords[edge[1]][2]))
                    self.ax_3d.plot(x, y, z, c='black', alpha=0.5)
        else:
            colors = [plt.cm.plasma(1) for _ in self.coords]
        for idn, row in enumerate(self.coords):
            xi = row[0]
            yi = row[1]
            zi = row[2]
            self.ax_3d.scatter(xi, yi, zi, c=[colors[idn]], s=20 + 20 * degree[idn], edgecolors='k', alpha=0.7)
        annotate3D(self.ax_3d, s=self.labels, xyz=self.coords, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')

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

    def plotScatter(self, x, y):
        self.ax_2d.clear()
        self.ax_2d.scatter(x, y, color="green")
        self.ax_2d.set_title("Projection Minimum Spanning Tree")
        self.fig.canvas.draw()

    def plotContour(self, x, y):
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

    def plotConvolution(self, x, y):
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
            if 'weights' in locals():
                S[indx - mid:indx + mid - 1, indy - mid:indy + mid - 1] += kernel * self.weights
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

    def plotType(self, label):
        self.prevlabel = label
        x = self.proj_coords[:, 0]
        y = self.proj_coords[:, 1]
        if self.cb != None:
            self.cb.remove()
            self.cb = None
        if label == 'Scatter':
            self.plotScatter(x, y)
        elif label == 'Contour':
            self.plotContour(x, y)
        elif label == 'Convolution':
            self.plotConvolution(x, y)

    def initializePlot(self):
        self.mst_3D()
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

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, "", xy=(0, 0), bbox=dict(boxstyle="round,pad=0.3", fc="whitesmoke", ec="lightgray", lw=2),
                            *args, **kwargs)
        self._verts3d = xyz
        self.s = s

    def draw(self, renderer):
        for coord, text in zip(self._verts3d, self.s):
            xs3d, ys3d, zs3d = coord[0], coord[1], coord[2]
            xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.xy = (xs, ys)
            Annotation.set_text(self, text)
            Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
