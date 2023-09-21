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

from xmipp3.protocols.protocol_aligned_solid_angles import XmippProtAlignedSolidAngles

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

class XmippViewerAlignedSolidAngles(ProtocolViewer):
    _label = 'viewer aligned solid angles'
    _targets = [XmippProtAlignedSolidAngles]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='PCA')
        form.addParam('displayDirectionalMd', LabelParam, label='Directional metadata',
                      help='Shows a metadata with the directional averages and eigenimages')
        form.addParam('displayDirectionClassification', IntParam, label='Directional classification',
                      default=1,
                      help='Shows a histogram of the directional classification')
        
        form.addSection(label='Graph')
        form.addParam('display3dGraph', LabelParam, label='Distance graph',
                      help='Shows a 3D representation of the distance graph')


    #--------------------------- INFO functions ----------------------------------------------------

    # --------------------------- DEFINE display functions ----------------------
    
    def _getVisualizeDict(self):
        return {
            'displayDirectionalMd': self._displayDirectionalMd,
            'displayDirectionClassification': self._displayDirectionClassification,
            'display3dGraph': self._display3dGraph,
        }
        
    def _displayDirectionalMd(self, e):
        dv = DataView(self._getDirectionalMdFilename())
        return [dv]
    
    def _displayDirectionClassification(self, e):
        # Read data from disk
        directionId = self.displayDirectionClassification.get()
        directionRow = self._readDirectionRow(directionId)
        projections = emlib.MetaData(directionRow.getValue(emlib.MDL_SELFILE)).getColumnValues(emlib.MDL_SCORE_BY_PCA_RESIDUAL)
        average = emlib.Image(directionRow.getValue(emlib.MDL_IMAGE)).getData()
        basis = emlib.Image(directionRow.getValue(emlib.MDL_IMAGE_RESIDUAL)).getData()
        gmm = self._readDirectionalGaussianMixtureModel(directionId)
        
        # Obtain GMM model parameters
        weights = gmm.weights_
        means = gmm.means_[:,0]
        variances = gmm.covariances_[:,0,0]
        stddevs = np.sqrt(variances)
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(2)
        
        # Show histogram and gaussian mixture
        _, bins, _ = ax1.hist(projections, bins=64, density=True, color='black', picker=True)
        x = (bins[1:] + bins[:-1]) / 2
        ys = weights*scipy.stats.norm.pdf(x[:,None], means, stddevs)
        n = ys.shape[-1]
        if n > 1:
            for i in range(n):
                ax1.plot(x, ys[:,i], linestyle='dashed')
        ax1.plot(x, ys.sum(axis=-1))
        
        # Show image
        line = ax1.axvline(x=0, color='yellow')
        ax2.imshow(average)
        
        # Setup interactive picking
        def on_pick(event):
            projection = event.mouseevent.xdata
            line.set_xdata([projection, ]*2)
            image = average + basis*projection
            ax2.clear()
            ax2.imshow(image)
            fig.canvas.draw()
            
        fig.canvas.mpl_connect('pick_event', on_pick)
        
        return [fig]
        
    def _display3dGraph(self, e):
        # Read from disk
        points = self._readDirectionVectors()
        graph = scipy.sparse.tril(self._readGraph(), format='csr')
        edges = graph.nonzero()
        segments = np.stack((points[edges[0]], points[edges[1]]), axis=1)
        weights = graph[edges].A1
        colormap = self._getScalarColorMap()
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        # Display de direction vectors
        ax.scatter(points[:,0], points[:,1], points[:,2], c='b')
        
        # Display the graph edges
        normalize = mpl.colors.Normalize()
        lines = mpl3d.art3d.Line3DCollection(segments, linewidths=0.5, colors=colormap(normalize(weights)))
        ax.add_collection(lines)
        
        #  Show colorbar
        fig.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=colormap), label='Edge weights')
        
        return [fig]
        


    # --------------------------- UTILS functions -----------------------------
    def _getScalarColorMap(self):
        return mpl.cm.plasma
    
    def _getMaskGalleryAnglesMdFilename(self):
        return self.protocol._getMaskGalleryAnglesMdFilename()
    
    def _getGraphFilename(self):
        return self.protocol._getGraphFilename()
    
    def _getDirectionalGaussianMixtureModelFilename(self, direction_id: int):
        return self.protocol._getDirectionalGaussianMixtureModelFilename(direction_id)

    def _getDirectionalMdFilename(self):
        return self.protocol._getDirectionalMdFilename()
    
    def _readGraph(self) -> scipy.sparse:
        return scipy.sparse.load_npz(self._getGraphFilename())
        
    def _readDirectionVectors(self):
        directionMd = emlib.MetaData(self._getMaskGalleryAnglesMdFilename())
        x = directionMd.getColumnValues(emlib.MDL_X)
        y = directionMd.getColumnValues(emlib.MDL_Y)
        z = directionMd.getColumnValues(emlib.MDL_Z)
        return np.stack((x, y, z), axis=-1)
    
    def _readDirectionRow(self, directionId: int):
        result = emlib.metadata.Row()
        md = emlib.MetaData(self._getDirectionalMdFilename())
        result.readFromMd(md, directionId)
        return result
    
    def _readDirectionalGaussianMixtureModel(self, directionId) -> sklearn.mixture.GaussianMixture:
        with open(self._getDirectionalGaussianMixtureModelFilename(directionId), 'rb') as f:
            return pickle.load(f)