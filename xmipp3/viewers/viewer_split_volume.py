# **************************************************************************
# *
# * Authors:     L. del Cano (ldelcano@cnb.csic.es)
# *              Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
"""
This module implement the wrappers aroung Xmipp CL2D protocol
visualization program.
"""
from cProfile import label
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import IntParam, LabelParam, BooleanParam

from xmipp3.protocols.protocol_split_volume import XmippProtSplitvolume

import numpy as np
import matplotlib.pyplot as plt

class XmippViewerSplitVolume(ProtocolViewer):
    """ Viewer for Split Volumes protocol
    """
    _label = 'viewer split volume'
    _targets = [XmippProtSplitvolume]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Graph')
        form.addParam('displayDistanceImage', LabelParam, label='Angular distance matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the angular distance computed for each image pair')
        form.addParam('displayCorrelationImage', LabelParam, label='Correlation matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the correlation computed for each image pair')
        form.addParam('displayWeightImage', LabelParam, label='Weight matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the weight of each graph edge')
        form.addParam('displayLabelImage', LabelParam, label='Classification',
                        help='Shows an image where each column\'s colour corresponds to '
                        'the label assigned to each image')
        form.addParam('displayLabelHistogram', LabelParam, label='Class sizes',
                        help='Shows a bar plot with the sizes of each class')
        form.addParam('displayProjectionClassification', LabelParam, label='Projection classification',
                        help='Shows a 3D representation of the classification')
        form.addParam('displayAngularGraph', LabelParam, label='3D Graph',
                        help='Shows a 3D representation of the graph')

    def _getVisualizeDict(self):
        return {
            'displayDistanceImage': self._displayDistanceImage,
            'displayCorrelationImage': self._displayCorrelationImage,
            'displayWeightImage': self._displayWeightImage,
            'displayLabelImage': self._displayLabelImage,
            'displayLabelHistogram': self._displayLabelHistogram,
            'displayProjectionClassification': self._displayProjectionClassification,
            'displayAngularGraph': self._displayAngularGraph,
        }
    
    # --------------------------- DEFINE display functions ----------------------
    def _displayDistanceImage(self, e):
        distances = self._readAngularDistances()
        return self._showImagePairImage(distances, 'Angular distances')

    def _displayCorrelationImage(self, e):
        correlations = self._readCorrelations()
        return self._showImagePairImage(correlations, 'Correlation')
    
    def _displayWeightImage(self, e):
        weights = self._readWeights()
        return self._showImagePairImage(weights, 'Weights')

    def _displayLabelImage(self, e):
        labels = self._readLabels()
        labels = np.array([labels]) # It needs to be bidimensional to work

        fig, ax = plt.subplots()
        fig.colorbar(ax.imshow(labels, origin='lower', aspect='auto', interpolation='none'))
        ax.set_xlabel('Image number')
        ax.set_title('Classification')
        return [fig]

    def _displayLabelHistogram(self, e):
        labels = self._readLabels()
        nClasses = int(labels.max())+1

        fig, ax = plt.subplots()
        ax.hist(labels, bins=nClasses)
        ax.set_xlabel('Class number')
        ax.set_ylabel('Class size')
        ax.set_title('Class sizes')
        return [fig]

    def _displayProjectionClassification(self, e):
        images = self._readImages()
        points = np.array(self._getProjectionUnitSphere(images))
        labels = self._readLabels()

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Projections')
        fig.colorbar(ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels))

        return [fig]

    def _displayAngularGraph(self, e):
        images = self._readImages()
        points = np.array(self._getProjectionUnitSphere(images))
        labels = self._readLabels()
        weights = self._readWeights()

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('3D graph')
        fig.colorbar(ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels))

        # Plot the edges
        for edge in zip(*np.nonzero(weights)):
            ax.plot3D(points[edge,0], points[edge,1], points[edge,2], 'gray')

        return [fig]

    # --------------------------- UTILS functions -----------------------------
    def _readImages(self):
        return self.protocol.directionalClasses.get()

    def _readAngularDistances(self):
        return self.protocol._readAngularDistances()

    def _readCorrelations(self):
        return self.protocol._readCorrelations()

    def _readWeights(self):
        return self.protocol._readWeights()

    def _readLabels(self):
        return self.protocol._readLabels()

    def _getProjectionUnitSphere(self, images):
        # Multiply the transform of the images by the unit x vector.
        # This is equivalent to selecting the first column.
        f = lambda img : img.getTransform().getMatrix()[0:3, 0]
        points = list(map(f, images))

        assert(len(images) == len(points))
        return points

    def _showImagePairImage(self, img, title):
        fig, ax = plt.subplots()
        fig.colorbar(ax.imshow(img, origin='lower', aspect='auto', interpolation='none'))
        ax.set_xlabel('Image number')
        ax.set_ylabel('Image number')
        ax.set_title(title)
        return [fig]

