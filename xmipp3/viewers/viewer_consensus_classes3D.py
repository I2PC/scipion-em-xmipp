# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, LabelParam, IntParam
from pyworkflow.protocol.params import GE, GT, LE, LT

from pwem.viewers import TableView, ObjectView
from pwem.viewers.showj import *

from xmipp3.protocols.protocol_consensus_classes3D import XmippProtConsensusClasses3D

from scipy.cluster import hierarchy

import matplotlib.pyplot as plt
import numpy as np


class XmippConsensusClasses3DViewer(ProtocolViewer):
    """ Visualization of results from the consensus classes 3D protocol
    """
    _label = 'viewer consensus classes 3D'
    _targets = [XmippProtConsensusClasses3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    def _defineParams(self, form):
        # Particles section
        form.addSection(label='Classes')
        form.addParam('displayClasses', IntParam, label='Custom number of classes',
                      validators=[GE(1)], default=1,
                      help='Open a GUI to visualize the class of the given iteration. Warning: It takes time to open')

        form.addParam('displayInitialClasses', LabelParam, label='Initial classes')
        form.addParam('displayManualClasses', LabelParam, label='Manual number of classes')
        form.addParam('displayOriginClasses', LabelParam, label='Number of classes by proximity to origin')
        form.addParam('displayAngleClasses', LabelParam, label='Number of classes by angle')
        form.addParam('displayPllClasses', LabelParam, label='Number of classes by profile likelihood')

        # Graph section
        form.addSection(label='Graphs')
        form.addParam('displayDendrogram', LabelParam, label='Dendrogram',
                        help='Open a GUI to visualize the dendogram each number of clusters')
        form.addParam('displayDendrogramLog', LabelParam, label='Dendrogram (log)',
                        help='Open a GUI to visualize the dendogram each number'
                        ' of clusters with a logarithmic "y" axis')

        form.addParam('displayObjectiveFunction', LabelParam, label='Objective Function',
                        help='Open a GUI to visualize the objective function for each number of clusters')
        form.addParam('displayObjectiveFunctionLog', LabelParam, label='Objective Function (log)',
                        help='Open a GUI to visualize the objective function'
                        ' for each number of clusters with a logarithmic "y" axis')

        # Reference classification section (only if exists)
        if self._readReferenceClassificationSizes() is not None:
            form.addSection(label='Reference classification consensus')
            form.addParam('displayReferenceClassificationSizePercentiles', LabelParam, label='Reference classification consensus size percentiles',
                            help='Open a GUI to visualize a table with the most'
                            ' common percentiles of the sizes of a random classification consensus')

    def _getVisualizeDict(self):
        return {
            'displayClasses': self._visualizeClasses,
            'displayInitialClasses': self._visualizeInitialClasses,
            'displayManualClasses': self._visualizeManualClasses,
            'displayOriginClasses': self._visualizeOriginClasses,
            'displayAngleClasses': self._visualizeAngleClasses,
            'displayPllClasses': self._visualizePllClasses,
            'displayDendrogram': self._visualizeDendrogram,
            'displayDendrogramLog': self._visualizeDendrogramLog,
            'displayObjectiveFunction': self._visualizeObjectiveFunction,
            'displayObjectiveFunctionLog': self._visualizeObjectiveFunctionLog,
            'displayReferenceClassificationSizePercentiles': self._visualizeReferenceClassificationSizes,
        }

    # --------------------------- UTILS functions ------------------------------
    def _getInputParticles(self):
        return self.protocol._getInputParticles()

    def _getInputClassifications(self):
        return self.protocol._getInputClassifications()

    def _readIntersections(self):
        return self.protocol._readIntersections()

    def _readClustering(self, iter):
        return self.protocol._readClustering(iter)

    def _readClusteringCount(self):
        return self.protocol._readClusteringCount()

    def _readAllClusterings(self):
        return self.protocol._readAllClusterings()

    def _readObjectiveValues(self):
        return self.protocol._readObjectiveValues()
    
    def _readElbows(self):
        return self.protocol._readElbows()

    def _readReferenceClassificationSizes(self):
        return self.protocol._readReferenceClassificationSizes()

    def _readReferenceClassificationRelativeSizes(self):
        return self.protocol._readReferenceClassificationRelativeSizes()

    def _visualizeClasses(self, e):
        # Read data
        count = self.displayClasses.get()
        particles = self._getInputParticles()
        clustering = self._readClustering(count)
        randomConsensusSizes = self._readReferenceClassificationSizes()
        randomConsensusRelativeSizes = self._readReferenceClassificationRelativeSizes()

        # Create the output classes and show them
        outputClasses = self.protocol._createOutputClasses3D(
            particles,
            clustering, 
            'viewer_'+str(count),
            randomConsensusSizes, 
            randomConsensusRelativeSizes
        )

        return self._showSetOfClasses3D(outputClasses)
    
    def _visualizeInitialClasses(self, e):
        return self._showSetOfClasses3D(self.protocol.outputClasses_initial)

    def _visualizeManualClasses(self, e):
        return self._showSetOfClasses3D(self.protocol.outputClasses_manual)

    def _visualizeOriginClasses(self, e):
        return self._showSetOfClasses3D(self.protocol.outputClasses_origin)

    def _visualizeAngleClasses(self, e):
        return self._showSetOfClasses3D(self.protocol.outputClasses_angle)

    def _visualizePllClasses(self, e):
        return self._showSetOfClasses3D(self.protocol.outputClasses_pll)

    def _visualizeDendrogram(self, e):
        clusterings = self._readAllClusterings()
        objectiveFunction = self._readObjectiveValues()

        fig, ax = plt.subplots()
        self._plotDendrogram(fig, ax, list(reversed(clusterings)), list(reversed(objectiveFunction)))
        return [fig]
        
    def _visualizeDendrogramLog(self, e):
        clusterings = self._readAllClusterings()
        objectiveFunction = self._readObjectiveValues()

        fig, ax = plt.subplots()
        self._plotDendrogram(fig, ax, list(reversed(clusterings)), list(reversed(objectiveFunction)))
        ax.set_yscale('log')
        return [fig]
        
    def _visualizeObjectiveFunction(self, e):
        objectiveFunction = self._readObjectiveValues()
        elbows = self._readElbows()

        fig, ax = plt.subplots()
        self._plotObjectiveFunctionAndElbows(fig, ax, objectiveFunction, elbows)
        return [fig]
    
    def _visualizeObjectiveFunctionLog(self, e):
        objectiveFunction = self._readObjectiveValues()
        elbows = self._readElbows()

        fig, ax = plt.subplots()
        self._plotObjectiveFunctionAndElbows(fig, ax, objectiveFunction[:-1], elbows)
        ax.set_yscale('log')
        return [fig]

    def _visualizeReferenceClassificationSizes(self, e):
        randomConsensusSizes = self._readReferenceClassificationSizes()
        randomConsensusRelativeSizes = self._readReferenceClassificationRelativeSizes()

        # Calculate the percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        randomConsensusSizePercentiles = np.percentile(randomConsensusSizes, percentiles)
        randomConsensusRelativeSizePercentiles = np.percentile(randomConsensusRelativeSizes, percentiles)
        
        # Create a table with the data
        data = list(zip(percentiles, randomConsensusSizePercentiles, randomConsensusRelativeSizePercentiles))
        header = ['Percentile (%)', 'Size percentile', 'Relative size percentile']
        title = 'Reference classification consensus size percentiles'
        return [TableView(header, data, title=title)]

    def _showSetOfClasses3D(self, classes):
        labels = 'enabled id _size _representative._filename _xmipp_classIntersectionSizePValue _xmipp_classIntersectionRelativeSizePValue'
        labelRender = '_representative._filename'
        return [ObjectView( self._project, classes.strId(), classes.getFileName(),
                            viewParams={ORDER: labels,
                                        VISIBLE: labels,
                                        RENDER: labelRender,
                                        SORT_BY: '_size desc',
                                        MODE: MODE_MD})]

    def _plotDendrogram(self, fig, ax, clusterings, obValues):
        """ Plot the dendrogram from the objective functions of the merge
            between the groups of images """

        # Initialize required values
        linkageMatrix = np.zeros((len(clusterings)-1, 4))
        clsIds = np.arange(len(clusterings))
        nIntersections = len(clusterings[0])

        # Loop over each iteration of clustering
        for i in range(len(clusterings)-1):
            # Find the sets that were merged
            clsMerged = []
            assert(len(clusterings[i]) == len(clsIds))
            for cls, clsId in zip(clusterings[i], clsIds):
                if cls not in clusterings[i+1]:
                    clsMerged.append(clsId)
            assert(len(clsMerged) == 2)
            clsMerged = np.array(clsMerged)
            
            # Find original number of sets within new set
            nCls = list(map(lambda x : linkageMatrix[x,3] if (x >= 0) else 1, clsMerged - nIntersections))

            # Create linkage matrix
            linkageMatrix[i, 0:2] = clsMerged  # ids of merged sets
            linkageMatrix[i, 2] = obValues[i+1]   # objective function as distance
            linkageMatrix[i, 3] = sum(nCls)  # total number of original sets

            # Change set ids to reflect new set of clusters
            for id in clsMerged:
                clsIds = np.delete(clsIds, np.argwhere(clsIds == id))
            clsIds = np.append(clsIds, len(clusterings)+i)

        # Plot resulting dendrogram
        hierarchy.dendrogram(linkageMatrix, ax=ax)
        ax.set_title('Dendrogram')
        ax.set_xlabel('sets ids')
        ax.set_ylabel('objective function')
        ax.set_ylim([np.min(linkageMatrix[:, 2]), np.max(linkageMatrix[:, 2])])

    def _plotObjectiveFunctionAndElbows(self, fig, ax, obValues, elbows):
        """ Plots the objective function and elbows """

        # Shorthands for some variables
        numClusters = np.arange(1, 1+len(obValues))

        # Plot obValues vs numClusters
        ax.plot(numClusters, obValues)

        # Show elbows as scatter points
        for key, value in elbows.items():
            value -= 1 # Value uses 1-based indexing
            label = key + ': ' + str(numClusters[value])
            ax.scatter([numClusters[value]], [obValues[value]], label=label, color='green')

        # Configure the figure
        ax.legend()
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Objective values')
        ax.set_title('Objective values for each number of clusters')
        ax.set_ylim([np.min(obValues), np.max(obValues)])
        
