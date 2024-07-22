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
        form.addParam('displayClassSizes', LabelParam, label='Class sizes',
                      help='Shows the 2D histogram of pairwise principal components')
        form.addParam('displayHistograms', LabelParam, label='Histograms',
                      help='Shows the 2D histogram of pairwise principal components')
        form.addParam('displayScatter3d', LabelParam, label='3D point scatter',
                      help='Shows the 2D histogram of pairwise principal components')
    

    #--------------------------- INFO functions ----------------------------------------------------

    # --------------------------- DEFINE display functions ----------------------
    def _getVisualizeDict(self):
        return {
            'displayClassSizes': self._displayClassSizes,
            'displayHistograms': self._displayHistograms,
            'displayScatter3d': self._displayScatter3d,
        }

    def _displayClassSizes(self, e):
        md = emlib.MetaData(self._getClassificationFilename())
        counts = collections.Counter(md.getColumnValues(emlib.MDL_REF3D))

        fig, ax = plt.subplots()
        ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
        
        return [fig]

    def _displayHistograms(self, e):
        projections = self._readProjections()
        gmm = self._readGaussianMixtureModel()
        return self._showHistograms(projections, gmm)

    def _displayScatter3d(self, e):
        projections, classes = self._readProjectionsAndClasses()
        percentiles = np.percentile(projections, q=(1, 99), axis=0)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(projections[:,-1], projections[:,-2], projections[:,-3], c=classes, marker='.', alpha=0.1)
        ax.set_xlim(percentiles[0,-1], percentiles[1,-1])
        ax.set_ylim(percentiles[0,-2], percentiles[1,-2])
        ax.set_zlim(percentiles[0,-3], percentiles[1,-3])
        
        return[fig]
    
    # --------------------------- UTILS functions -----------------------------   
    def _getClassificationFilename(self):
        return self.protocol._getClassificationMdFilename()
    
    def _readProjections(self) -> np.ndarray:
        md = emlib.MetaData(self.protocol._getClassificationMdFilename())
        return np.array(md.getColumnValues(emlib.MDL_DIMRED))
    
    def _readProjectionsAndClasses(self) -> np.ndarray:
        md = emlib.MetaData(self.protocol._getClassificationMdFilename())
        projections = np.array(md.getColumnValues(emlib.MDL_DIMRED))
        classes = np.array(md.getColumnValues(emlib.MDL_REF3D))
        return projections, classes
    
    def _readGaussianMixtureModel(self) -> sklearn.mixture.GaussianMixture:
        return self.protocol._readGaussianMixtureModel()
    
    def _showHistograms(self, 
                        projections: np.ndarray,
                        gmm: sklearn.mixture.GaussianMixture ):
        _, k = projections.shape
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        percentiles = np.percentile(projections, q=(1, 99), axis=0)
        weights = gmm.weights_
        means = gmm.means_
        covariances = gmm.covariances_

        fig = plt.figure()
        
        for i, j in itertools.combinations(range(k), r=2):
            ax = fig.add_subplot(k, k, k*j+i+1)

            x = projections[:,i]
            y = projections[:,j]
            xmin = percentiles[0,i]
            xmax = percentiles[1,i]
            ymin = percentiles[0,j]
            ymax = percentiles[1,j]

            ax.hist2d(
                x, y, 
                bins=32, 
                density=True, 
                range=((xmin, xmax), (ymin, ymax)), 
                cmap=mpl.cm.binary
            )
            
            indices = np.array([i,j])
            subMeans = means[:,indices]
            subCovariance = covariances[:,indices[:,None],indices]
            for mean, covariance, color in zip(subMeans, subCovariance, colors):
                w, v = np.linalg.eigh(covariance)
                w = 2.0 * np.sqrt(2.0) * np.sqrt(w)
                angle = np.rad2deg(np.arctan2(v[0,1], v[0,0]))
                ellipse = mpl.patches.Ellipse(
                    xy=mean,
                    width=w[0], height=w[1],
                    angle=180 + angle,
                    color=color,
                    fill=False,
                    clip_box=ax
                )
                ax.add_artist(ellipse)
        
        for i in range(k):
            ax = fig.add_subplot(k, k, (k+1)*i+1)
            
            v = projections[:,i]
            vmin = percentiles[0,i]
            vmax = percentiles[1,i]
            
            _, bins, _ = ax.hist(
                v, 
                bins=32, 
                density=True, 
                range=(vmin, vmax), 
                color='k'
            )

            x = (bins[:-1] + bins[1:]) / 2
            norm = scipy.stats.norm(
                loc=means[:,i], 
                scale=np.sqrt(covariances[:,i,i])
            )
            ys = weights*norm.pdf(x[:,None])
            ax.plot(x, ys, linestyle='dashed')
            ax.plot(x, np.sum(ys, axis=1))
    
        return [fig]
        