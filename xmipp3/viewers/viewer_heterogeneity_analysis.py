# **************************************************************************
# *
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

from typing import Tuple

from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import IntParam, LabelParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range

from pwem.viewers import DataView
from pwem import emlib

from xmipp3.protocols.protocol_heterogeneity_analysis import XmippProtHetAnalysis

import math
import itertools
import numpy as np
import scipy.sparse
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d

class XmippViewerHetAnalysis(ProtocolViewer):
    _label = 'viewer heterogeneity analysis'
    _targets = [XmippProtHetAnalysis]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Analysis')
        form.addParam('displayHistograms', LabelParam, label='Histograms',
                      help='Shows the 2D histograms of pairwise principal components')
        form.addParam('displayScatter2d', LabelParam, label='2D point scatter',
                      help='Shows the 2D histogram of pairwise principal components')
        form.addParam('displayScatter3d', LabelParam, label='3D point scatter',
                      help='Shows the 2D histogram of pairwise principal components')
        form.addParam('displayInformationCriterion', LabelParam, label='Information criterion',
                      help='Shows Akaike and Bayesian information criterions')
        form.addParam('displayCrossCorrelation', LabelParam, label='Cross correlation')

        form.addSection(label='Directional data')
        form.addParam('displayDirectionalMd', LabelParam, label='Directional metadata',
                      help='Shows a metadata with the directional averages and eigenimages')
        form.addParam('displayDirectionalHistograms', IntParam, label='Directional histograms',
                      default=1,
                      help='Shows the 2D histograms')
        
        form.addSection(label='Graph')
        form.addParam('displayCrossCorrelations', LabelParam, label='Cross correlations',
                      help='Shows a block matrix with cross correlations')
        form.addParam('displayWeights', LabelParam, label='Weights',
                      help='Shows a block matrix with weights')
        form.addParam('displayWeightedCrossCorrelation', LabelParam, label='Weighted cross correlation',
                      help='Shows a block matrix with weighted cross correlations')
        form.addParam('display3dGraph', LabelParam, label='3D adjacency graph')

    #--------------------------- INFO functions ----------------------------------------------------

    # --------------------------- DEFINE display functions ----------------------
    def _getVisualizeDict(self):
        return {
            'displayHistograms': self._displayHistograms,
            'displayScatter2d': self._displayScatter2d,
            'displayScatter3d': self._displayScatter3d,
            'displayInformationCriterion': self._displayInformationCriterion,
            'displayCrossCorrelation': self._displayCrossCorrelation,
            'displayDirectionalMd': self._displayDirectionalMd,
            'displayDirectionalHistograms': self._displayDirectionalHistograms,
            'displayCrossCorrelations': self._displayCrossCorrelations,
            'displayWeights': self._displayWeights,
            'displayWeightedCrossCorrelation': self._displayWeightedCrossCorrelations,
            'display3dGraph': self._display3dGraph,
        }

    def _displayHistograms(self, e):
        projections = self._readProjections()
        return self._showHistograms(projections)

    def _displayScatter2d(self, e):
        projections = self._readProjections()
        percentiles = np.percentile(projections, q=(1, 99), axis=0)
    
        fig, ax = plt.subplots()
        ax.scatter(projections[:,-1], projections[:,-2], s=1, marker='.', alpha=0.1)
        ax.set_xlim(percentiles[0,-1], percentiles[1,-1])
        ax.set_ylim(percentiles[0,-2], percentiles[1,-2])
            
        return[fig]

    def _displayScatter3d(self, e):
        projections = self._readProjections()
        percentiles = np.percentile(projections, q=(1, 99), axis=0)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(projections[:,-1], projections[:,-2], projections[:,-3], s=1, marker='.', alpha=0.1)
        ax.set_xlim(percentiles[0,-1], percentiles[1,-1])
        ax.set_ylim(percentiles[0,-2], percentiles[1,-2])
        ax.set_zlim(percentiles[0,-3], percentiles[1,-3])
        
        return[fig]
    
    def _displayInformationCriterion(self, e):
        analysis = self._readGmmAnalysis()

        fig, ax = plt.subplots()
        
        x = np.array(analysis['param_n_components'], dtype=float)
        y = -np.array(analysis['mean_test_score'], dtype=float)

        ax.plot(x, y)
        ax.set_ylabel('Bayesian information criterion')
        ax.set_xlabel('Number of classes')
            
        return [fig]
        
    def _displayCrossCorrelation(self, e):
        projections = self._readProjections()
        projections /= np.linalg.norm(projections, axis=0, keepdims=True)
        crossCorrelation = projections.T @ projections

        fig, ax = plt.subplots()
        ax.imshow(crossCorrelation)
        return [fig]
        
    def _displayDirectionalMd(self, e):
        dv = DataView(self._getDirectionalMdFilename())
        return [dv]

    def _displayDirectionalHistograms(self, e):
        directionId = self.displayDirectionalHistograms.get()
        md = emlib.MetaData(self.protocol._getDirectionalClassificationMdFilename(directionId))
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        return self._showHistograms(projections)
    
    def _displayCrossCorrelations(self, e):
        crossCorrelations = self._readCrossCorrelations()
        fig, ax = plt.subplots()
        self._showSparseBlockMatrix(
            ax, 
            crossCorrelations, 
            self._getBlockStride(),
            vmin=-1.0, vmax=+1.0,
            cmap=mpl.cm.coolwarm
        )
        return [fig]
    
    def _displayWeights(self, e):
        weights = self._readWeights()
        fig, ax = plt.subplots()
        self._showSparseBlockMatrix(
            ax, 
            weights, 
            self._getBlockStride()
        )
        return [fig]
    
    def _displayWeightedCrossCorrelations(self, e):
        crossCorrelations = self._readCrossCorrelations()
        weights = self._readWeights()
        fig, ax = plt.subplots()
        self._showSparseBlockMatrix(
            ax, 
            crossCorrelations.multiply(weights), 
            self._getBlockStride(),
            vmin=-1.0, vmax=+1.0,
            cmap=mpl.cm.coolwarm
        )
        return [fig]
   
    def _display3dGraph(self, e) -> plt.Figure:
        fig = plt.figure()
        ax: mpl3d.Axes3D = fig.add_subplot(projection='3d')
        
        directionMd = emlib.MetaData(self._getDirectionalMdFilename())
        objIds = list(directionMd)
        xs = directionMd.getColumnValues(emlib.MDL_X)
        ys = directionMd.getColumnValues(emlib.MDL_Y)
        zs = directionMd.getColumnValues(emlib.MDL_Z)
        points = np.column_stack((xs, ys, zs))
        
        ax.scatter(xs, ys, zs, color='black')
        for objId, x, y, z in zip(objIds, xs, ys, zs):
            ax.text(x, y, z, str(objId))
        
        adjacency = self._readAdjacencyGraph()
        edges = adjacency.nonzero()
        segments = np.stack((points[edges[0]], points[edges[1]]), axis=1)
        counts = adjacency[edges].A1
        
        # Display the graph edges
        colormap = mpl.cm.plasma
        normalize = mpl.colors.LogNorm(1.0, counts.max())
        lines = mpl3d.art3d.Line3DCollection(segments, linewidths=0.5, colors=colormap(normalize(counts)))
        ax.add_collection(lines)
        
        #  Show colorbar
        fig.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=colormap), label='Particle count')
        
        return [fig]
    
    # --------------------------- UTILS functions -----------------------------   
    def _getBlockStride(self) -> int:
        return self.protocol._getPrincipalComponentsCount()
    
    def _getDirectionalMdFilename(self):
        return self.protocol._getDirectionalMdFilename()
    
    def _readCrossCorrelations(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readCrossCorrelations()
        
    def _readWeights(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readWeights()
    
    def _readAdjacencyGraph(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readAdjacencyGraph()
    
    def _readProjections(self) -> np.ndarray:
        md = emlib.MetaData(self.protocol._getClassificationMdFilename())
        return np.array(md.getColumnValues(emlib.MDL_DIMRED))
    
    def _readGmmAnalysis(self):
        return self.protocol._readGmmAnalysis()
    
    def _gridLayout(self, n: int) -> Tuple[int, int]:
        cols = math.floor(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols
    
    def _showBlockMatrix(self, 
                         ax: plt.Axes,
                         matrix: np.ndarray, 
                         stride: int,
                         **kwargs  ):
        n = len(matrix)
        start = 0.5
        end = n / stride + 0.5
        extent = (start, end, end, start)
        
        plt.colorbar(ax.imshow(matrix, extent=extent, **kwargs))
        
        # Plot delimiting lines
        lines = np.linspace(start=start+1, stop=end-1, num=(n//stride)-1)
        ax.vlines(x=lines, ymin=start, ymax=end, colors='k')
        ax.hlines(y=lines, xmin=start, xmax=end, colors='k')
        
    def _showSparseBlockMatrix(self, 
                               ax: plt.Axes,
                               matrix: scipy.sparse.spmatrix, 
                               stride: int,
                               **kwargs):
        self._showBlockMatrix(ax, matrix.todense(), stride=stride, **kwargs)
    
    def _showHistograms(self, projections: np.ndarray):
        _, k = projections.shape
        percentiles = np.percentile(projections, q=(1, 99), axis=0)

        fig = plt.figure()
        
        for i, j in itertools.combinations(range(k), r=2):
            ax = fig.add_subplot(k, k, k*j+i+1)

            x = projections[:,i]
            y = projections[:,j]
            xmin = percentiles[0,i]
            xmax = percentiles[1,i]
            ymin = percentiles[0,j]
            ymax = percentiles[1,j]

            ax.hist2d(x, y, bins=32, range=((xmin, xmax), (ymin, ymax)))
        
        for i in range(k):
            ax = fig.add_subplot(k, k, (k+1)*i+1)
            
            v = projections[:,i]
            vmin = percentiles[0,i]
            vmax = percentiles[1,i]
            
            ax.hist(v, bins=32, range=(vmin, vmax))
    
        return [fig]
        