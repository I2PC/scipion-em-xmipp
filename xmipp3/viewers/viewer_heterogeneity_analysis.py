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
        form.addParam('displayHistogram1d', LabelParam, label='1D histograms',
                      help='Shows the histogram of each principal component')
        form.addParam('displayHistogram2d', LabelParam, label='2D histograms',
                      help='Shows the 2D histogram of pairwise principal components')
        form.addParam('displayInformationCriterion', LabelParam, label='Information criterion',
                      help='Shows Akaike and Bayesian information criterions')

        form.addSection(label='Directional data')
        form.addParam('displayDirectionalMd', LabelParam, label='Directional metadata',
                      help='Shows a metadata with the directional averages and eigenimages')
        form.addParam('displayDirectionalHistogram1d', IntParam, label='Directional 1D histograms',
                      default=1,
                      help='Shows the 2D histogram of the first 2 principal components')
        form.addParam('displayDirectionalHistogram2d', IntParam, label='Directional 2D histograms',
                      default=1,
                      help='Shows the 2D histogram of the first 2 principal components')
        
        form.addSection(label='Graph')
        form.addParam('displayCrossCorrelations', LabelParam, label='Cross correlations',
                      help='Shows a block matrix with cross correlations')
        form.addParam('displayWeights', LabelParam, label='Weights',
                      help='Shows a block matrix with weights')
        form.addParam('displayWeightedCrossCorrelation', LabelParam, label='Weighted cross correlation',
                      help='Shows a block matrix with weighted cross correlations')

    #--------------------------- INFO functions ----------------------------------------------------

    # --------------------------- DEFINE display functions ----------------------
    def _getVisualizeDict(self):
        return {
            'displayHistogram1d': self._displayHistogram1d,
            'displayHistogram2d': self._displayHistogram2d,
            'displayInformationCriterion': self._displayInformationCriterion,
            'displayDirectionalMd': self._displayDirectionalMd,
            'displayDirectionalHistogram1d': self._displayDirectionalHistogram1d,
            'displayDirectionalHistogram2d': self._displayDirectionalHistogram2d,
            'displayCrossCorrelations': self._displayCrossCorrelations,
            'displayWeights': self._displayWeights,
            'displayWeightedCrossCorrelation': self._displayWeightedCrossCorrelations,
        }

    def _displayHistogram1d(self, e):
        projections = self._readProjections()
        return self._showHistogram1d(projections)
    
    def _displayHistogram2d(self, e):
        projections = self._readProjections()
        return self._showHistogram2d(projections)

    def _displayInformationCriterion(self, e):
        analysis = self._readGmmAnalysis()
        
        fig, ax = plt.subplots()
        
        x = np.array(analysis['param_n_components'])
        y = -np.array(analysis['mean_test_score'])

        ax.plot(x, y)
        ax.set_title('Bayesian Information Criterion')
            
        return [fig]
        
    def _displayDirectionalMd(self, e):
        dv = DataView(self._getDirectionalMdFilename())
        return [dv]

    def _displayDirectionalHistogram1d(self, e):
        directionId = self.displayDirectionalHistogram2d.get()
        md = emlib.MetaData(self.protocol._getDirectionalClassificationMdFilename(directionId))
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        return self._showHistogram1d(projections)
    
    def _displayDirectionalHistogram2d(self, e):
        directionId = self.displayDirectionalHistogram2d.get()
        md = emlib.MetaData(self.protocol._getDirectionalClassificationMdFilename(directionId))
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        return self._showHistogram2d(projections)

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
    
    # --------------------------- UTILS functions -----------------------------   
    def _getBlockStride(self) -> int:
        k = self.protocol.principalComponents.get()
        k2 = k+2
        return k2
    
    def _getDirectionalMdFilename(self):
        return self.protocol._getDirectionalMdFilename()
    
    def _readCrossCorrelations(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readCrossCorrelations()
        
    def _readWeights(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readWeights()
    
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
    
    def _showHistogram1d(self, projections: np.ndarray):
        percentiles = np.percentile(projections, q=(1, 99), axis=0)

        k = projections.shape[1]
        fig, axs = plt.subplots(1, k)
        for i, ax in enumerate(axs):
            v = projections[:,i]
            vmin = percentiles[0,i]
            vmax = percentiles[1,i]
            ax.hist(v, bins=32, range=(vmin, vmax))
            ax.set_title(f'Principal component {i}')
            
        return [fig]
    
    def _showHistogram2d(self, projections: np.ndarray):
        percentiles = np.percentile(projections, q=(1, 99), axis=0)
        
        k = projections.shape[1]
        fig, axs = plt.subplots(k-1, k-1)
        for idx0, idx1 in itertools.combinations(range(k), r=2):
            if k > 2:
                ax = axs[idx1-1][idx0]
            else:
                ax = axs
            x = projections[:,idx0]
            y = projections[:,idx1]
            xmin = percentiles[0,idx0]
            xmax = percentiles[1,idx0]
            ymin = percentiles[0,idx1]
            ymax = percentiles[1,idx1]
            ax.hist2d(x, y, bins=32, range=((xmin, xmax), (ymin, ymax)))

        return [fig]
    