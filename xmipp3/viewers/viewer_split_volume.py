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
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import IntParam, LabelParam, BooleanParam
from pwem.viewers import DataView

from xmipp3.protocols.protocol_split_volume import XmippProtSplitvolume

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d
from collections import Counter

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
        form.addSection(label='Input classes')
        form.addParam('displayInputClassification', LabelParam, label='Input classification',
                        help='Shows 3D representation of the input classes')

        form.addSection(label='Output classes')
        form.addParam('displayLabelHistogram', LabelParam, label='Class sizes',
                        help='Shows a bar plot with the sizes of each class')
        form.addParam('displayLabelImage', LabelParam, label='Classification',
                        help='Shows an image where each column\'s colour corresponds to '
                        'the label assigned to each image')
        form.addParam('displayProjectionClassification', LabelParam, label='Projection classification',
                        help='Shows a 3D representation of the classification')
        form.addParam('displayProjectionClassificationDisjoint', LabelParam, label='Disjoint projection classification',
                        help='Shows a 3D representation of the classification with a projection sphere for each class')

        form.addSection(label='Image pairs')
        form.addParam('displayDistanceImage', LabelParam, label='Angular distance matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the angular distance computed for each image pair')
        form.addParam('displayComparisonImage', LabelParam, label='Comparison matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the correlation computed for each image pair')
        form.addParam('displayWeightImage', LabelParam, label='Weight matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the weight of each graph edge')

        form.addSection(label='Fiedler vector')
        form.addParam('displayFiedlerVector', LabelParam, label='Sorted Fiedler Vector',
                        help='Shows a graph of the sorted Fiedler vector')
        form.addParam('displayFiedlerVectorDerivative', LabelParam, label='Sorted Fiedler Vector derivative',
                        help='Shows a graph of derivative the sorted Fiedler vector')
        form.addParam('displaySortedWeightImage', LabelParam, label='Sorted weight matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the weight of each graph edge sorted according to the fiedler vector')
        form.addParam('displayGraphCutMetric', LabelParam, label='Graph cut metric',
                        help='Shows the graph cut cost function in terms of the sorted Fiedler Vector')
        form.addParam('displayRatioCutMetric', LabelParam, label='Ratio cut metric',
                        help='Shows the ratio cut cost function in terms of the sorted Fiedler Vector')
        form.addParam('displayNormalizedCutMetric', LabelParam, label='Normalized cut metric',
                        help='Shows the normalized cut cost function in terms of the sorted Fiedler Vector')
        form.addParam('displayQuotientCutMetric', LabelParam, label='Quotient cut metric',
                        help='Shows the quotient cut cost function in terms of the sorted Fiedler Vector')

        form.addSection(label='Networks')
        form.addParam('displayComparison3dNetwork', LabelParam, label='3D comparison network',
                        help='Shows a 3D representation of the correlation network')
        form.addParam('displayComparison3dNetworkDisjoint', LabelParam, label='Disjoint 3D comparison network',
                        help='Shows a 3D representation of the correlation network with a projection sphere for each class')
        form.addParam('displayWeight3dNetwork', LabelParam, label='3D weight network',
                        help='Shows a 3D representation of the weight network')
        form.addParam('displayWeight3dNetworkDisjoint', LabelParam, label='Disjoint 3D weight network',
                        help='Shows a 3D representation of the weight network with a projection sphere for each class')

    def _getVisualizeDict(self):
        return {
            'displayInputClassification': self._displayInputClassification,
            'displayLabelHistogram': self._displayLabelHistogram,
            'displayLabelImage': self._displayLabelImage,
            'displayProjectionClassification': self._displayProjectionClassification,
            'displayProjectionClassificationDisjoint': self._displayProjectionClassificationDisjoint,
            'displayDistanceImage': self._displayDistanceImage,
            'displayComparisonImage': self._displayComparisonImage,
            'displayWeightImage': self._displayWeightImage,
            'displayFiedlerVector': self._displayFiedlerVector,
            'displayFiedlerVectorDerivative': self._displayFiedlerVectorDerivative,
            'displaySortedWeightImage': self._displaySortedWeightImage,
            'displayGraphCutMetric': self._displayGraphCutMetric,
            'displayRatioCutMetric': self._displayRatioCutMetric,
            'displayNormalizedCutMetric': self._displayNormalizedCutMetric,
            'displayQuotientCutMetric': self._displayQuotientCutMetric,
            'displayComparison3dNetwork': self._displayComparison3dNetwork,
            'displayComparison3dNetworkDisjoint': self._displayComparison3dNetworkDisjoint,
            'displayWeight3dNetwork': self._displayWeight3dNetwork,
            'displayWeight3dNetworkDisjoint': self._displayWeight3dNetworkDisjoint,
        }
    
    # --------------------------- DEFINE display functions ----------------------
    def _displayInputClassification(self, e):
        images = self._readImages()
        directionIds = self._readDirectionIds()
        points = self._getProjectionSphere(images)
        labels = self._getDirectionIdLabels(directionIds)
        
        # Scale the points to be concentric
        scales = self._calculateConcentricProjectionSphereScales(labels, 0.2)
        points *= np.column_stack((scales, )*3)

        # Obtain the edges joining members of the same class
        edges = self._getDirectionIdEdgeLines(points, directionIds)

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Input Classification')
        self._plotProjectionClassification(fig, ax, points, images, labels)
        self._plotNetworkEdges(fig, ax, edges)
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

    def _displayLabelImage(self, e):
        labels = self._readLabels()

        fig, ax = plt.subplots()
        self._plotClassification(fig, ax, labels)
        ax.set_title('Classification')
        return [fig]
    
    def _displayProjectionClassification(self, e):
        images = self._readImages()
        labels = self._readLabels()
        points = self._getProjectionSphere(images)

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Projection Classification')
        self._plotProjectionClassification(fig, ax, points, images, labels)
        return [fig]

    def _displayProjectionClassificationDisjoint(self, e):
        images = self._readImages()
        labels = self._readLabels()
        points = self._getProjectionSphere(images)

        # Apply an offset to the points
        offsets = self._calculateDisjointProjectionSphereOffsets(labels)
        points += offsets

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Projection Classification')
        self._plotProjectionClassification(fig, ax, points, images, labels)
        return [fig]

    def _displayDistanceImage(self, e):
        distances = self._readAngularDistances()

        fig, ax = plt.subplots()
        self._plotMatrix(fig, ax, distances, 'Angle (rad)')
        ax.set_title('Angular distances')
        return [fig]

    def _displayComparisonImage(self, e):
        comparisons = self._readComparisons()
        
        fig, ax = plt.subplots()
        self._plotMatrix(fig, ax, comparisons, 'Similarity')
        ax.set_title('Comparisons')
        return [fig]
    
    def _displayWeightImage(self, e):
        weights = self._readWeights()

        fig, ax = plt.subplots()
        self._plotMatrix(fig, ax, weights, 'Weight')
        ax.set_title('Weights')
        return [fig]

    def _displayFiedlerVector(self, e):
        fiedler = self._readFiedlerVector()

        x = np.arange(len(fiedler))
        y = np.sort(fiedler)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title('Sorted Fiedler Vector')
        return [fig]

    def _displayFiedlerVectorDerivative(self, e):
        fiedler = self._readFiedlerVector()

        x = np.arange(len(fiedler)-1)
        y = np.diff(np.sort(fiedler))
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title('Sorted Fiedler Vector derivative')
        return [fig]

    def _displaySortedWeightImage(self, e):
        weights = self._readWeights()
        fiedler = self._readFiedlerVector()

        indices = np.argsort(fiedler)
        weights = weights[:,indices][indices,:]

        fig, ax = plt.subplots()
        self._plotMatrix(fig, ax, weights, 'Weight')
        ax.set_title('Sorted weights')
        return [fig]

    def _displayGraphCutMetric(self, e):
        weights = self._readWeights()
        fiedler = self._readFiedlerVector()

        fig, ax = plt.subplots()
        self._plotCutMetric(fig, ax, self.protocol._calculateGraphCutMetric, weights, fiedler)
        ax.set_title('Graph cut metric')
        return [fig]
        
    def _displayRatioCutMetric(self, e):
        weights = self._readWeights()
        fiedler = self._readFiedlerVector()

        fig, ax = plt.subplots()
        self._plotCutMetric(fig, ax, self.protocol._calculateRatioCutMetric, weights, fiedler)
        ax.set_title('Ratio cut metric')
        return [fig]
    
    def _displayNormalizedCutMetric(self, e):
        weights = self._readWeights()
        fiedler = self._readFiedlerVector()

        fig, ax = plt.subplots()
        self._plotCutMetric(fig, ax, self.protocol._calculateNormalizedCutMetric, weights, fiedler)
        ax.set_title('Normalized cut metric')
        return [fig]

    def _displayQuotientCutMetric(self, e):
        weights = self._readWeights()
        fiedler = self._readFiedlerVector()

        fig, ax = plt.subplots()
        self._plotCutMetric(fig, ax, self.protocol._calculateQuotientCutMetric, weights, fiedler)
        ax.set_title('Quotient cut metric')
        return [fig]

    def _displayComparison3dNetwork(self, e):
        images = self._readImages()
        comparisons = self._readComparisons()
        labels = self._readLabels()
        points = self._getProjectionSphere(images)

        # Obtain the edges of the graph
        edges, edgeWeights = self._getEdgeLines(points, comparisons)

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('3D Comparison Network')
        self._plotProjectionClassification(fig, ax, points, images, labels)
        self._plotWeightedNetworkEdges(fig, ax, edges, edgeWeights)
        return [fig]

    def _displayComparison3dNetworkDisjoint(self, e):
        images = self._readImages()
        comparisons = self._readComparisons()
        labels = self._readLabels()
        points = self._getProjectionSphere(images)

        # Apply a offset to the points
        offsets = self._calculateDisjointProjectionSphereOffsets(labels)
        points += offsets

        # Obtain the edges of the graph
        edges, edgeWeights = self._getEdgeLines(points, comparisons)

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('3D Comparison Network')
        self._plotProjectionClassification(fig, ax, points, images, labels)
        self._plotWeightedNetworkEdges(fig, ax, edges, edgeWeights)
        return [fig]

    def _displayWeight3dNetwork(self, e):
        images = self._readImages()
        weights = self._readWeights()
        labels = self._readLabels()
        points = self._getProjectionSphere(images)

        # Obtain the edges of the graph
        edges, edgeWeights = self._getEdgeLines(points, weights)

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('3D Weight Network')
        self._plotProjectionClassification(fig, ax, points, images, labels)
        self._plotWeightedNetworkEdges(fig, ax, edges, edgeWeights)
        return [fig]

    def _displayWeight3dNetworkDisjoint(self, e):
        images = self._readImages()
        weights = self._readWeights()
        labels = self._readLabels()
        points = self._getProjectionSphere(images)

        # Apply a offset to the points
        offsets = self._calculateDisjointProjectionSphereOffsets(labels)
        points += offsets

        # Obtain the edges of the graph
        edges, edgeWeights = self._getEdgeLines(points, weights)

        # Plot the projection angles with the classification
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('3D Weight Network')
        self._plotProjectionClassification(fig, ax, points, images, labels)
        self._plotWeightedNetworkEdges(fig, ax, edges, edgeWeights)
        return [fig]

    # --------------------------- UTILS functions -----------------------------
    def _readImages(self):
        return self.protocol._readInputImages()

    def _readDirectionIds(self):
        return self.protocol._readDirectionIds()

    def _readAngularDistances(self):
        return self.protocol._readAngularDistances()

    def _readComparisons(self):
        return self.protocol._readComparisons()

    def _readWeights(self):
        return self.protocol._readWeights()

    def _readFiedlerVector(self):
        return self.protocol._readFiedlerVector()

    def _readLabels(self):
        return self.protocol._readLabels()

    def _getDirectionIdLabels(self, directionIds):
        result = np.zeros_like(directionIds)

        # Count items with the same id before it
        for i in range(len(result)):
            result[i] = np.count_nonzero(directionIds[:i] == directionIds[i])

        return result

    def _getProjectionSphere(self, images):
        points = np.array(list(map(self.protocol._getProjectionDirection, images)))

        # Force points to the upper hemisphere
        scale = np.ones(len(points))
        scale = np.where(points[:,2] < 0, -scale, scale)
        points *= np.column_stack((scale, )*3)

        return points

    def _calculateDistanceMinimizationMirrors(self, points, directionIds):
        result = np.zeros(len(directionIds))

        for idx0 in range(len(result)):
            # Find a prior occurrence of the same class directionId
            idx1 = np.argmax(directionIds[:idx0+1]==directionIds[idx0])
            assert(idx1 <= idx0)

            # Only consider flipping when there is a prior occurrence
            if idx0 != idx1:
                # Flip the projection when the distance is larger than 90º
                if np.dot(points[idx0], points[idx1]) < 0:
                    result[idx0] = -1

        return result

    def _calculateConcentricProjectionSphereScales(self, labels, step=0.2):
        return 1 + step*labels

    def _calculateDisjointProjectionSphereOffsets(self, labels):
        nClasses = int(max(labels)) + 1

        # Calculate a offset in a circumference for each point
        thetas = 2*math.pi*labels / nClasses
        radius = 1.5*max(nClasses/math.pi, 1) # For N<=3 the limiting factor is the radius. The perimeter is otherwise
        offsets = radius*np.column_stack((np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)))
        
        return offsets

    def _getDirectionIdEdgeLines(self, points, directionIds):
        result = []

        for id in np.unique(directionIds):
            result.append(points[directionIds==id])

        return result

    def _getEdgeLines(self, points, weights):
        edges = []
        edgeWeights = []

        # Make the matrix symmetric as we do not consider directionality
        weights = np.maximum(weights, weights.T)

        for (src, dst) in zip(*np.nonzero(weights)):
            # Only consider if it is in the lower triangle
            if src <= dst:
                srcPoint = points[src]
                dstPoint = points[dst]
                line = np.row_stack((srcPoint, dstPoint))
                edges.append(line)
                edgeWeights.append(weights[src,dst])
        assert(len(edges) == len(edgeWeights))

        return edges, edgeWeights

    def _getScalarColorMap(self):
        return mpl.cm.plasma

    def _getClassificationColorMap(self, labels):
        nLabels = int(max(labels)) + 1
        colours = [mpl.cm.jet(i / (nLabels-1)) for i in range(nLabels)]
        return mpl.colors.ListedColormap(colours)


    def _plotMatrix(self, fig, ax, img, label):
        colormap = self._getScalarColorMap()
        norm = mpl.colors.Normalize()
        ax.imshow(img, origin='lower', aspect='auto', interpolation='none', cmap=colormap, norm=norm)
        ax.set_xlabel('Image number')
        ax.set_ylabel('Image number')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label=label)

    def _plotClassification(self, fig, ax, labels):
        colormap = self._getClassificationColorMap(labels)
        norm = mpl.colors.Normalize()
        ax.imshow(np.array([labels]), origin='lower', aspect='auto', interpolation='none', cmap=colormap, norm=norm)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Classes', ticks=np.arange(colormap.N))

    def _plotProjectionClassification(self, fig, ax, points, images, labels):
        colormap = self._getClassificationColorMap(labels)
        norm = mpl.colors.Normalize()
        ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels, cmap=colormap, norm=norm, picker=True)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Classes', ticks=np.arange(colormap.N))
        self._openImageOnClick(fig, ax, images)

    def _plotNetworkEdges(self, fig, ax, edges):
        lines = mpl3d.art3d.Line3DCollection(edges, linewidths=0.5)
        ax.add_collection(lines)

    def _plotWeightedNetworkEdges(self, fig, ax, edges, weights):
        colormap = self._getScalarColorMap()
        norm = mpl.colors.Normalize()
        lines = mpl3d.art3d.Line3DCollection(edges, linewidths=0.5, colors=colormap(norm(weights)))
        ax.add_collection(lines)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), label='Edge weights')

    def _plotCutMetric(self, fig, ax, metric, graph, fiedler):
        indices = np.argsort(fiedler)

        x = np.arange(1, len(indices))
        y = self.protocol._calculateMetricValues(graph, indices, metric) 
        assert(len(x) == len(y))
        
        ax.plot(x, y)
        ax.set_xlabel('Cut position')
        ax.set_ylabel('Metric value')

    def _openImageOnClick(self, fig, ax, images):
        def callback(event):
            if event.mouseevent.inaxes == ax and event.mouseevent.dblclick:
                indices = event.ind
                if len(indices) == 1:
                    index = event.ind[0]
                    image = images[index]
                    path = '%s@%s' % image.getLocation()
                    DataView(path).show()

        return fig.canvas.callbacks.connect('pick_event', callback)