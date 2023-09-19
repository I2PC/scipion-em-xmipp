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
import numpy as np
import scipy.sparse
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
        form.addSection(label='Graph')
        form.addParam('display3dGraph', LabelParam, label='Distance graph',
                      help='Shows a 3D representation of the distance graph')


    #--------------------------- INFO functions ----------------------------------------------------

    # --------------------------- DEFINE display functions ----------------------
    
    def _getVisualizeDict(self):
        return {
            'display3dGraph': self._display3dGraph,
        }
        
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
        #fig.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=colormap), label='Edge weights')
        
        return [fig]
        


    # --------------------------- UTILS functions -----------------------------
    def _getScalarColorMap(self):
        return mpl.cm.plasma
    
    def _getMaskGalleryAnglesMdFilename(self):
        return self.protocol._getMaskGalleryAnglesMdFilename()
    
    def _getGraphFilename(self):
        return self.protocol._getGraphFilename()
    
    def _readGraph(self) -> scipy.sparse:
        return scipy.sparse.load_npz(self._getGraphFilename())
        
    def _readDirectionVectors(self):
        directionMd = emlib.MetaData(self._getMaskGalleryAnglesMdFilename())
        x = directionMd.getColumnValues(emlib.MDL_X)
        y = directionMd.getColumnValues(emlib.MDL_Y)
        z = directionMd.getColumnValues(emlib.MDL_Z)
        return np.stack((x, y, z), axis=-1)