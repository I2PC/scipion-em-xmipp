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

from xmipp3.protocols.protocol_heterogeneous_reconstruct import XmippProtHetReconstruct

import math
import collections
import itertools
import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.mixture
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d

class XmippViewerHetReconstruct(ProtocolViewer):
    _label = 'viewer heterogeneous reconstruct'
    _targets = [XmippProtHetReconstruct]
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
    

    #--------------------------- INFO functions ----------------------------------------------------

    # --------------------------- DEFINE display functions ----------------------
    def _getVisualizeDict(self):
        return {
            'displayHistogram1d': self._displayHistogram1d,
            'displayHistogram2d': self._displayHistogram2d,
        }

    def _displayHistogram1d(self, e):
        projections = self._readProjections()
        gmm = self._readGaussianMixtureModel()
        return self._showHistogram1d(projections, gmm)
    
    def _displayHistogram2d(self, e):
        projections = self._readProjections()
        gmm = self._readGaussianMixtureModel()
        return self._showHistogram2d(projections, gmm)

    # --------------------------- UTILS functions -----------------------------   
    def _readProjections(self) -> np.ndarray:
        md = emlib.MetaData(self.protocol._getClassificationMdFilename())
        return np.array(md.getColumnValues(emlib.MDL_DIMRED))
    
    def _readGaussianMixtureModel(self) -> sklearn.mixture.GaussianMixture:
        return self.protocol._readGaussianMixtureModel()
    
    def _gridLayout(self, n: int) -> Tuple[int, int]:
        cols = math.floor(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols
    
    def _showHistogram1d(self, projections: np.ndarray, gmm: sklearn.mixture.GaussianMixture):
        percentiles = np.percentile(projections, q=(1, 99), axis=0)

        k = projections.shape[1]
        fig, axs = plt.subplots(1, k)
        for i, ax in enumerate(axs):
            v = projections[:,i]
            vmin = percentiles[0,i]
            vmax = percentiles[1,i]
            weights = gmm.weights_
            means = gmm.means_[:,i]
            variances = gmm.covariances_[:,i,i]
            
            _, bins, _ = ax.hist(v, bins=32, range=(vmin, vmax), density=True, color='k')
            
            x = (bins[:-1] + bins[1:]) / 2
            ys = weights[:,None]*scipy.stats.norm.pdf(x, loc=means[:,None], scale=np.sqrt(variances[:,None]))
            
            ax.plot(x, np.sum(ys, axis=0))
            for y in ys:
                ax.plot(x, y, linestyle='--')
            
            ax.set_title(f'Principal component {i}')
            
        return [fig]
    
    def _showHistogram2d(self, projections: np.ndarray, gmm: sklearn.mixture.GaussianMixture):
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
            weights = gmm.weights_
            means = gmm.means_[:,(idx0,idx1)]
            covariances = gmm.covariances_[:,i,i]

            ax.hist2d(x, y, bins=32, range=((xmin, xmax), (ymin, ymax)), density=True)

        return [fig]
    