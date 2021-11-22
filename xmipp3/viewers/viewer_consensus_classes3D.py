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
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, LabelParam, IntParam

from xmipp3.protocols.protocol_consensus_classes3D import XmippProtConsensusClasses3D, ob_function
from pwem.objects import Class3D
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
        self._objectiveFunctionValues = self._loadObjectiveFunctionData()
        self._elbows = self._loadElbows()
        self._clusterings = self._loadClusterings()

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayDendrogram', LabelParam,
                      label='Visualize the dendrogram of the merging process',
                      help='Open a GUI to visualize the dendogram each number of clusters ')
        form.addParam('displayObjectiveFunction', LabelParam,
                      label='Visualize Objective Function',
                      help='Open a GUI to visualize the objective function for each number of clusters ')

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
            'displayDendrogram': self._visualizeDendrogram,
            'displayObjectiveFunction': self._visualizeObjectiveFunction
        }

        # TODO: remove when done
        #return {'chooseNumberOfClusters': self._chooseNumberOfClusters,
        #        'displayObjectiveFunction': self._visualizeObjectiveFunction}

    # --------------------------- UTILS functions ------------------------------
    def _visualizeDendrogram(self, e):
        self._plotDendrogram()
        
    def _visualizeObjectiveFunction(self, e):
        self._plotFunctionAndElbows()

    def _loadObjectiveFunctionData(self):
        loadPath = self.protocol._getExtraPath(self.protocol.objective_function_pkl)
        with open(loadPath, 'rb') as f:
            result = pickle.load(f)
        return result

    def _loadClusterings(self):
        loadPath = self.protocol._getExtraPath(self.protocol.clusterings_pkl)
        with open(loadPath, 'rb') as f:
            result = pickle.load(f)
        return result

    def _loadElbows(self):
        loadPath = self.protocol._getExtraPath(self.protocol.elbows_pkl)
        with open(loadPath, 'rb') as f:
            result = pickle.load(f)
        return result

    def _plotDendrogram(self):
        """ Plot the dendrogram from the objective functions of the merge
            between the groups of images """

        # Initialize required values
        clusterings = list(reversed(self._clusterings))
        obValues = list(reversed(self._objectiveFunctionValues))
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
        plt.yscale('log')
        plt.ylim([np.min(linkage_matrix[:, 2]), np.max(linkage_matrix[:, 2])])
        plt.show()

    def _plotFunctionAndElbows(self):
        """ Plots the objective function and elbows """

        # Shorthands for some variables
        obValues = self._objectiveFunctionValues
        elbows = self._elbows
        numClusters = list(range(1, 1+len(obValues)))

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
        plt.show()

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
