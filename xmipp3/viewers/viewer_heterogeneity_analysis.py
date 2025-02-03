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
from pyworkflow.protocol.params import IntParam, LabelParam, PointerParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range

from pwem.viewers import DataView, ChimeraView
from pwem import emlib
from pwem.objects import Volume

from xmipp3.protocols.protocol_heterogeneity_analysis import XmippProtHetAnalysis

import os
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
        form.addParam('consensusVolume', PointerParam, label='Consensus volume',
                      pointerClass=Volume, allowsNull=True)
        form.addParam('displayHistograms', LabelParam, label='Histograms',
                      help='Shows the 2D histograms of pairwise principal components')
        form.addParam('displayEigenVolume', LabelParam, label='Eigen-volumes')

        form.addSection(label='Directional data')
        form.addParam('displayDirectionalMd', LabelParam, label='Directional metadata',
                      help='Shows a metadata with the directional averages and eigenimages')
        form.addParam('displayDirectionalHistograms', IntParam, label='Directional histograms',
                      default=1 )
        
        form.addSection(label='Graph')
        form.addParam('display3dGraph', LabelParam, label='3D adjacency graph')
        form.addParam('displayAdjacencyMatrix', LabelParam, label='Adjacency matrix')
        
        form.addSection(label='Cross correlations')
        form.addParam('displayCrossCorrelations', LabelParam, label='Cross correlations',
                      help='Shows a block matrix with cross correlations')

    #--------------------------- INFO functions ----------------------------------------------------

    # --------------------------- DEFINE display functions ----------------------
    def _getVisualizeDict(self):
        return {
            'displayHistograms': self._displayHistograms,
            'displayEigenvalues': self._displayEigenvalues,
            'displayEigenVolume': self._displayEigenVolume,
            'displayDirectionalMd': self._displayDirectionalMd,
            'displayDirectionalHistograms': self._displayDirectionalHistograms,
            'display3dGraph': self._display3dGraph,
            'displayAdjacencyMatrix': self._displayAdjacencyMatrix,
            'displayCrossCorrelations': self._displayCrossCorrelations,
        }

    def _displayHistograms(self, e):
        projections = self._readProjections()
        return self._showHistograms(projections)

    def _displayInformationCriterion(self, e):
        analysis = self._readGmmAnalysis()

        fig, ax = plt.subplots()
        
        x = np.array(analysis['param_n_components'], dtype=float)
        y = -np.array(analysis['mean_test_score'], dtype=float)

        ax.plot(x, y)
        ax.set_ylabel('Bayesian information criterion')
        ax.set_xlabel('Number of classes')
            
        return [fig]
        
    def _displayEigenvalues(self, e):
        eigenvalues = self._readEigenvalues()
        x = np.arange(1, 1+len(eigenvalues))
        y = eigenvalues
        dy = np.diff(y)

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        
        ax.bar(x, y, width=1.0, label='Eigen-values')
        ax2.plot(x[:-1]+0.5, dy, label='Adjacent difference', c='orange')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)
        
        return [fig]
        
    def _displayEigenVolume(self, e):
        command = self._writeChimeraScript(self.consensusVolume.get().getFileName())
        return [ChimeraView(command)]
        
    def _displayDirectionalMd(self, e):
        dv = DataView(self._getDirectionalMdFilename())
        return [dv]

    def _displayCorrectedDirectionalMd(self, e):
        dv = DataView(self._getCorrectedDirectionalMdFilename())
        return [dv]
    
    def _displayDirectionalHistograms(self, e):
        directionId = self.displayDirectionalHistograms.get()
        md = emlib.MetaData(self.protocol._getDirectionalClassificationMdFilename(directionId))
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        return self._showHistograms(projections)
    
    def _displayCorrectedDirectionalHistograms(self, e):
        directionId = self.displayCorrectedDirectionalHistograms.get()
        md = emlib.MetaData(self.protocol._getCorrectedDirectionalClassificationMdFilename(directionId))
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        return self._showHistograms(projections)
    
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
        normalize = mpl.colors.Normalize(1.0, counts.max())
        lines = mpl3d.art3d.Line3DCollection(
            segments, 
            linewidths=0.2, 
            colors=colormap(normalize(counts)),
        )
        ax.add_collection(lines)
        
        #  Show colorbar
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=normalize, cmap=colormap), 
            label='Particle count'
        )
        
        return [fig]
    
    def _displayAdjacencyMatrix(self, e):
        adjcency = self._readAdjacencyGraph().todense()
        fig, ax = plt.subplots()
        fig.colorbar(ax.imshow(adjcency), label='Intersection size')
        return [fig]
    
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
   
    def _displayReconstructedCrossCorrelations(self, e):
        bases = self._readBases()
        bases = bases.reshape(math.prod(bases.shape[:2]), bases.shape[2])
        pairwise = bases @ bases.T
        fig, ax = plt.subplots()
        self._showBlockMatrix(
            ax, 
            pairwise, 
            self._getBlockStride(),
            vmin=-1.0, vmax=+1.0,
            cmap=mpl.cm.coolwarm
        )
        return [fig]
    
    # --------------------------- UTILS functions -----------------------------   
    def _getBlockStride(self) -> int:
        return self.protocol._getPrincipalComponentsCount()
    
    def _getDirectionalMdFilename(self):
        return self.protocol._getDirectionalMdFilename()
    
    def _getCorrectedDirectionalMdFilename(self):
        return self.protocol._getCorrectedDirectionalMdFilename()

    def _readCrossCorrelations(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readCrossCorrelations()
        
    def _readWeights(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readWeights()
    
    def _readAdjacencyGraph(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readAdjacencyGraph()

    def _readPairwise(self) -> np.ndarray:
        return self.protocol._readPairwise()
    
    def _readBases(self) -> np.ndarray:
        return self.protocol._readBases()

    def _readProjections(self) -> np.ndarray:
        md = emlib.MetaData(self.protocol._getClassificationMdFilename())
        return np.array(md.getColumnValues(emlib.MDL_DIMRED))
    
    def _readGmmAnalysis(self):
        return self.protocol._readGmmAnalysis()

    def _readEigenvalues(self):
        return self.protocol._readEigenvalues()
    
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
            x = projections[:,i]
            y = projections[:,j]
            
            xmin = percentiles[0,i]
            xmax = percentiles[1,i]
            ymin = percentiles[0,j]
            ymax = percentiles[1,j]

            ax1 = fig.add_subplot(k, k, k*j+i+1)
            ax1.hist2d(x, y, bins=32, range=((xmin, xmax), (ymin, ymax)))

            cov = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            ax2 = fig.add_subplot(k, k, k*i+j+1)
            ax2.imshow(cov[None, None], vmin=-1.0, vmax=1.0, cmap='coolwarm')
            ax2.text(0, 0, '%0.3f' % float(cov), ha='center', va='center')
            ax2.set_axis_off()
            
        for i in range(k):
            ax = fig.add_subplot(k, k, (k+1)*i+1)
            
            v = projections[:,i]
            vmin = percentiles[0,i]
            vmax = percentiles[1,i]
            
            ax.hist(v, bins=32, range=(vmin, vmax))
    
        return [fig]
        
    def _writeChimeraScript(self, consensusVol: str) -> str:
        scriptFile = self.protocol._getExtraPath('fusion_chimera.cxc')
        consensusVolFilename = os.path.abspath(consensusVol)
        n = self.protocol.outputPrincipalComponentCount.get()
        
        ih = emlib.Image()
        with open(scriptFile, 'w') as f:
            for i in range(n):
                basisFilename = os.path.abspath(self.protocol._getEigenVolumeFilename(i+1))
                volId = i*2 + 1
                mapId = volId + 1
                ih.read(basisFilename)
                eigenVolume = ih.getData()
                limit = abs(eigenVolume).max()
                f.write("open %s\n" % consensusVolFilename)
                f.write("open %s\n" % basisFilename)
                f.write("vol #%d hide\n" % mapId)
                f.write("color sample #%d map #%d range %f,%f\n" % (volId, mapId, -limit, limit))
            f.write("tile\n")
        return scriptFile
    