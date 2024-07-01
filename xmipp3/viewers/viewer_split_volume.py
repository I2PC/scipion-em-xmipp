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

from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import IntParam, LabelParam, BooleanParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range

from pwem.viewers import DataView, ObjectView
from pwem import emlib

from xmipp3.protocols.protocol_split_volume import XmippProtSplitVolume

import math
import collections
import pickle
import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.mixture
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d

class XmippViewerSplitVolume(ProtocolViewer):
    _label = 'viewer split volume'
    _targets = [XmippProtSplitVolume]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Analysis')
        form.addParam('displayHistogram1d', IntParam, label='1D histogram',
                      default=1,
                      help='Shows the histogram of a principal component')
        form.addParam('displayHistogram2d', LabelParam, label='2D histogram',
                      help='Shows the 2D histogram of the first 2 principal components')
        form.addParam('displayInformationCriterion', LabelParam, label='Information criterion',
                      help='Shows Akaike and Bayesian information criterions')

        form.addSection(label='Directional data')
        form.addParam('displayDirectionalMd', LabelParam, label='Directional metadata',
                      help='Shows a metadata with the directional averages and eigenimages')
        form.addParam('displayDirectionalHistogram2d', IntParam, label='Directional 2D histogram',
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
            'displayDirectionalHistogram2d': self._displayDirectionalHistogram2d,
            'displayCrossCorrelations': self._displayCrossCorrelations,
            'displayWeights': self._displayWeights,
            'displayWeightedCrossCorrelation': self._displayWeightedCrossCorrelations,
        }

    def _displayHistogram1d(self, e):
        i: int = self.displayHistogram1d.get()
        md = emlib.MetaData(self.protocol._getClassificationMdFilename())
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        
        v = projections[:,i]
        vmin, vmax = np.percentile(v, (1, 99))

        fig, ax = plt.subplots()
        ax.hist(v, bins=64, range=(vmin, vmax))
        return [fig]
    
    def _displayHistogram2d(self, e):
        md = emlib.MetaData(self.protocol._getClassificationMdFilename())
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        
        x = projections[:,-1] # TODO
        y = projections[:,-2] # TODO
        xmin, xmax = np.percentile(x, (1, 99))
        ymin, ymax = np.percentile(y, (1, 99))
        
        fig, ax = plt.subplots()
        ax.hist2d(x, y, bins=32, range=((xmin, xmax), (ymin, ymax)))
        return [fig]
        
    def _displayInformationCriterion(self, e):
        aic = self._readAicArray()
        bic = self._readBicArray()
        x = np.arange(1, 1+len(aic))
        
        fig, ax = plt.subplots()
        ax.plot(x, aic, label='BIC')
        ax.plot(x, bic, label='AIC')
        ax.legend()
        return [fig]
    
    def _displayDirectionalMd(self, e):
        dv = DataView(self._getDirectionalMdFilename())
        return [dv]

    def _displayDirectionalHistogram2d(self, e):
        directionId = self.displayDirectionalHistogram2d.get()
        md = emlib.MetaData(self.protocol._getDirectionalClassificationMdFilename(directionId))
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        
        x = projections[:,-1] # TODO
        y = projections[:,-2] # TODO
        xmin, xmax = np.percentile(x, (1, 99))
        ymin, ymax = np.percentile(y, (1, 99))
        
        fig, ax = plt.subplots()
        ax.hist2d(x, y, bins=32, range=((xmin, xmax), (ymin, ymax)))
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
    
    # --------------------------- UTILS functions -----------------------------
    def _getBlockStride(self) -> int:
        return self.protocol.principalComponents.get()
    
    def _getDirectionalMdFilename(self):
        return self.protocol._getDirectionalMdFilename()
    
    def _readCrossCorrelations(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readCrossCorrelations()
        
    def _readWeights(self) -> scipy.sparse.csr_matrix:
        return self.protocol._readWeights()
    
    def _readAicArray(self):
        return self.protocol._readAicArray()

    def _readBicArray(self):
        return self.protocol._readBicArray()
    
    def _showBlockMatrix(self, 
                         ax: plt.Axes,
                         matrix: np.ndarray, 
                         stride: int,
                         **kwargs  ):
        n = len(matrix)
        start = -0.5 / stride
        end = (n - 0.5) / stride
        extent = (start, end, end, start)
        
        plt.colorbar(ax.imshow(matrix, extent=extent, **kwargs))
        
        # Plot delimiting lines
        lines = np.linspace(start=start+1, stop=end-1, num=n//stride)
        ax.vlines(x=lines, ymin=start, ymax=end, colors='k')
        ax.hlines(y=lines, xmin=start, xmax=end, colors='k')
        
    def _showSparseBlockMatrix(self, 
                               ax: plt.Axes,
                               matrix: scipy.sparse.spmatrix, 
                               stride: int,
                               **kwargs):
        self._showBlockMatrix(ax, matrix.todense(), stride=stride, **kwargs)
        