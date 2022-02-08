# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
from pyworkflow.gui import IconButton, Icon
from pyworkflow.gui.form import FormWindow
from pyworkflow.object import Object
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, LabelParam, IntParam
from pyworkflow.protocol.params import GE, GT, LE, LT

from pwem.viewers import TableView, ObjectView
from pwem.objects import Class3D
from pwem.viewers.showj import *

from xmipp3.protocols.protocol_consensus_classes3D import XmippProtConsensusClasses3D

from scipy.cluster import hierarchy

import pickle
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
        self._loadAttributes()

    def _defineParams(self, form):
        # Particles section
        form.addSection(label='Classes')
        if hasattr(self, '_clusterings'):
            form.addParam('displayClasses', IntParam, label='Custom number of classes',
                            validators=[GT(0), LE(len(self._clusterings))], default=len(self._clusterings),
                            help='Open a GUI to visualize the class of the given iteration. Warning: It takes time to open')

        if hasattr(self.protocol, 'outputClasses_initial'):
            form.addParam('displayInitialClasses', LabelParam, label='Initial classes')
        
        if hasattr(self.protocol, 'outputClasses_manual'):
            form.addParam('displayManualClasses', LabelParam, label='Manual number of classes')
        
        if hasattr(self.protocol, 'outputClasses_origin'):
            form.addParam('displayOriginClasses', LabelParam, label='Number of classes by proximity to origin')

        if hasattr(self.protocol, 'outputClasses_angle'):
            form.addParam('displayAngleClasses', LabelParam, label='Number of classes by angle')

        if hasattr(self.protocol, 'outputClasses_pll'):
            form.addParam('displayPllClasses', LabelParam, label='Number of classes by profile likelihood')

        # Graph section
        form.addSection(label='Graphs')
        if hasattr(self, '_clusterings') and hasattr(self, '_objectiveFunctionValues'):
            form.addParam('displayDendrogram', LabelParam, label='Dendrogram',
                          help='Open a GUI to visualize the dendogram each number of clusters')
            form.addParam('displayDendrogramLog', LabelParam, label='Dendrogram (log)',
                          help='Open a GUI to visualize the dendogram each number'
                          ' of clusters with a logarithmic "y" axis')

        if hasattr(self, '_objectiveFunctionValues') and hasattr(self, '_elbows'):
            form.addParam('displayObjectiveFunction', LabelParam, label='Objective Function',
                          help='Open a GUI to visualize the objective function for each number of clusters')
            form.addParam('displayObjectiveFunctionLog', LabelParam, label='Objective Function (log)',
                          help='Open a GUI to visualize the objective function'
                          ' for each number of clusters with a logarithmic "y" axis')

        # Reference classification section
        form.addSection(label='Reference classification consensus')
        if hasattr(self, '_sizeRatioPercentiles'):
            form.addParam('displayReferenceClassificationSizePercentiles', LabelParam, label='Reference Classification Size Percentiles',
                          help='Open a GUI to visualize a table with the most'
                          ' common percentiles of the sizes of a random classification consensus')

        if hasattr(self, '_sizeRatioPercentiles'):
            form.addParam('displayReferenceClassificationSizeRatioPercentiles', LabelParam, label='Reference Classification Size Ratio Percentiles',
                          help='Open a GUI to visualize a table with the most common percentiles'
                          ' sizes of a random classification consensus'
                          ' compared to the original cluster size')

        #form.addParam('chooseNumberOfClusters', IntParam, default=-1,
        #              label='Number of Clusters',
        #              help='Choose the final number of clusters in the consensus clustering. Press the eye')

        #TODO: set an apply button to change the number of clusters instead of using the eye
        '''command = self._chooseNumberOfClusters()
        btn = IconButton(self.getWindow(), "Apply", Icon.ACTION_CONTINUE, command=command)
        btn.config(relief="flat", activebackground=None, compound='left',
                   fg='black', overrelief="raised")
        btn.bind('<Button-1>')
        btn.grid(row=2, column=2, sticky='nw')'''

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
            'displayReferenceClassificationSizeRatioPercentiles': self._visualizeReferenceClassificationSizeRatios,
        }

        # TODO: remove when done
        #return {'chooseNumberOfClusters': self._chooseNumberOfClusters,
        #        'displayObjectiveFunction': self._visualizeObjectiveFunction}

    # --------------------------- UTILS functions ------------------------------
    def _loadAttributes(self):
        self._loadAttributeIfExists(self.protocol._getFileName('clusterings'), '_clusterings')
        self._loadAttributeIfExists(self.protocol._getFileName('objective_function'), '_objectiveFunctionValues')
        self._loadAttributeIfExists(self.protocol._getFileName('elbows'), '_elbows')
        self._loadAttributeIfExists(self.protocol._getFileName('random_consensus_sizes'), '_randomConsensusSizes')
        self._loadAttributeIfExists(self.protocol._getFileName('random_consensus_size_ratios'), '_randomConsensusSizeRatios')
        self._loadAttributeIfExists(self.protocol._getFileName('size_percentiles'), '_sizePercentiles')
        self._loadAttributeIfExists(self.protocol._getFileName('size_ratio_percentiles'), '_sizeRatioPercentiles')

    def _visualizeClasses(self, e):
        # Shorthands:
        ite = self.displayClasses.get() - 1
        clustering = self._clusterings[ite]

        # Create the output classes and show them
        outputClasses = self.protocol._createOutput3DClass(clustering, 'viewer_'+str(ite), self._randomConsensusSizes, self._randomConsensusSizeRatios)
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
        plot_dendrogram(list(reversed(self._clusterings)), list(reversed(self._objectiveFunctionValues)), False)
        
    def _visualizeDendrogramLog(self, e):
        plot_dendrogram(list(reversed(self._clusterings)), list(reversed(self._objectiveFunctionValues)), True)
        
    def _visualizeObjectiveFunction(self, e):
        plot_function_and_elbows(self._objectiveFunctionValues, self._elbows, False)
    
    def _visualizeObjectiveFunctionLog(self, e):
        plot_function_and_elbows(self._objectiveFunctionValues, self._elbows, True)

    def _visualizeReferenceClassificationSizes(self, e):
        show_percentile_table(self._sizePercentiles, 'Size percentiles')

    def _visualizeReferenceClassificationSizeRatios(self, e):
        show_percentile_table(self._sizeRatioPercentiles, 'Size ratio percentiles')

    def _loadAttributeIfExists(self, filename, attribute):
        self._loadAttribute(filename, attribute) # TODO: add exception handling for when the file does not exist

    def _loadAttribute(self, filename, attribute):
        setattr(self, attribute, self._loadObject(filename))

    def _loadObject(self, filename):
        path = self.protocol._getExtraPath(filename)
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result

    def _showSetOfClasses3D(self, classes):
        labels = 'enabled id _size _representative._filename _xmipp_classIntersectionSizePValue _xmipp_classIntersectionRelativeSizePValue'
        labelRender = '_representative._filename'
        return [ObjectView( self._project, classes.strId(), classes.getFileName(),
                            viewParams={ORDER: labels,
                                        VISIBLE: labels,
                                        RENDER: labelRender,
                                        SORT_BY: '_size desc',
                                        MODE: MODE_MD})]

    def _chooseNumberOfClusters(self, e=None):
        views=[]
        nClusters = list(range(1, 1+len(self._clusterings)))
        elbow_nclust = nClusters[self._elbowIdx]

        nclust = self.chooseNumberOfClusters.get()
        if nclust == -1:
            nclust = elbow_nclust
        elif nclust > max(nClusters):
            nclust = max(nClusters)

        scipion_clustering = self._buildSetOfClasses(nclust)
        self.protocol._defineOutputs(outputClasses=scipion_clustering)
        for item in self.protocol.inputMultiClasses:
            self.protocol._defineSourceRelation(item, scipion_clustering)

        #TODO: esto es un objeto SetOfClasses3D pero no sabemos como lanzar un viewer con el sin tener el protocolo
        #views.append(ObjectView(self._project, self.protocol.strId()))
        return views

    def _buildSetOfClasses(self, nclust):
        '''From a list of clusterings, nclust index it and that clustering
        is converted into a scipion setOfClasses'''
        clustIdx = self.nclusters.index(nclust)
        clustering = self._allClusterings[clustIdx]

        inputParticles = self.protocol.inputMultiClasses[0].get().getImages()
        outputClasses = self.protocol._createSetOfClasses3D(inputParticles)
        for classItem in clustering:
            numOfPart = classItem[0]
            partIds = classItem[1]
            setRepId = classItem[2]
            clsRepId = classItem[3]

            setRep = self.protocol.inputMultiClasses[setRepId].get()
            clRep = setRep[clsRepId]

            newClass = Class3D()
            # newClass.copyInfo(clRep)
            newClass.setAcquisition(clRep.getAcquisition())
            newClass.setRepresentative(clRep.getRepresentative())

            outputClasses.append(newClass)

            enabledClass = outputClasses[newClass.getObjId()]
            enabledClass.enableAppend()
            for itemId in partIds:
                enabledClass.append(inputParticles[itemId])
            outputClasses.update(enabledClass)

        return outputClasses



# Plotting functions

def plot_dendrogram(clusterings, obValues, logarithmic=False):
    """ Plot the dendrogram from the objective functions of the merge
        between the groups of images """

    # Initialize required values
    linkage_matrix = np.zeros((len(clusterings)-1, 4))
    set_ids = np.arange(len(clusterings))
    original_num_sets = len(clusterings)

    # Loop over each iteration of clustering
    for i in range(len(clusterings)-1):
        # Find the two sets that were merged
        sets_merged = []
        for set_info, set_id in zip(clusterings[i], set_ids):
            if set_info not in clusterings[i+1]:
                sets_merged.append(set_id)
        
        # Find original number of sets within new set
        if sets_merged[0] - original_num_sets < 0:
            n1 = 1
        else:
            n1 = linkage_matrix[sets_merged[0]-original_num_sets, 3]
        if sets_merged[1] - original_num_sets < 0:
            n2 = 1
        else:
            n2 = linkage_matrix[sets_merged[1]-original_num_sets, 3]

        # Create linkage matrix
        linkage_matrix[i, 0] = sets_merged[0]  # set id of merged set
        linkage_matrix[i, 1] = sets_merged[1]  # set id of merged set
        linkage_matrix[i, 2] = obValues[i+1]   # objective function as distance
        linkage_matrix[i, 3] = n1+n2  # total number of original sets

        # Change set ids to reflect new set of clusters
        set_ids = np.delete(set_ids, np.argwhere(set_ids == sets_merged[0]))
        set_ids = np.delete(set_ids, np.argwhere(set_ids == sets_merged[1]))
        set_ids = np.append(set_ids, len(clusterings)+i)

    # Plot resulting dendrogram
    plt.figure()
    dn = hierarchy.dendrogram(linkage_matrix)
    plt.title('Dendrogram')
    plt.xlabel('sets ids')
    plt.ylabel('objective function')
    plt.tight_layout()

    if logarithmic is True:
        plt.yscale('log')
        plt.ylim([np.min(linkage_matrix[:, 2]), np.max(linkage_matrix[:, 2])])

    plt.show()

def plot_function_and_elbows(obValues, elbows, logarithmic=False):
    """ Plots the objective function and elbows """

    # Shorthands for some variables
    numClusters = np.arange(1, 1+len(obValues))

    # Begin drawing
    plt.figure()

    # Plot obValues vs numClusters
    plt.plot(numClusters, obValues)

    # Show elbows as scatter points
    for key, value in elbows.items():
        label = key + ': ' + str(numClusters[value])
        plt.scatter([numClusters[value]], [obValues[value]], label=label, color='green')

    # Configure the figure
    plt.legend()
    plt.xlabel('Number of clusters')
    plt.ylabel('Objective values')
    plt.title('Objective values for each number of clusters')
    plt.tight_layout()

    if logarithmic is True:
        plt.yscale('log')
        plt.ylim([np.min(obValues), np.max(obValues)])
    
    plt.show()

def show_percentile_table(data, title=None):
    header = ('Percentile', 'Value')
    data = list(data.items())
    return TableView(header, data, None, title)