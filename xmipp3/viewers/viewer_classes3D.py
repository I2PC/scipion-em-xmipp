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

from xmipp3.protocols.protocol_consensus_classes3D import XmippProtConsensusClasses3D
from pwem.objects import Class3D

import pickle
import matplotlib.pyplot as plt

FILE_OBJECTIVE_FDATA = 'ObjectiveFData.pkl'
FILE_CLUSTERINGS = 'clusterings.pkl'
FILE_ELBOWCLUSTERS = 'elbowclusters.pkl'


class XmippClasses3DViewer(ProtocolViewer):
    """ Visualization of results from the consensus classes 3D protocol
    """
    _label = 'viewer classes 3D'
    _targets = [XmippProtConsensusClasses3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self.nclusters, self.objFValues = self.getObjectiveFData()
        self._elbowIdx = self.getElbowIndex()
        self._allClusterings = self.getClusterings()

    def getObjectiveFData(self):
        loadPath = self.protocol._getExtraPath(FILE_OBJECTIVE_FDATA)
        with open(loadPath, 'rb') as f:
            self._ObjFData = pickle.load(f)
        return self._ObjFData

    def getClusterings(self):
        loadPath = self.protocol._getExtraPath(FILE_CLUSTERINGS)
        with open(loadPath, 'rb') as f:
            self._allClusterings = pickle.load(f)
        return self._allClusterings

    def getElbowIndex(self):
        loadPath = self.protocol._getExtraPath(FILE_ELBOWCLUSTERS)
        with open(loadPath, 'rb') as f:
            self.elbowIndex = pickle.load(f)
        return self.elbowIndex

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayObjectiveFunction', LabelParam,
                      label='Visualize Objective Function',
                      help='Open a GUI to visualize the objective function for each number of clusters ')
        form.addParam('chooseNumberOfClusters', IntParam, default=-1,
                      label='Number of Clusters',
                      help='Choose the final number of clusters in the consensus clustering. Press the eye')

        #TODO: set an apply button to change the number of clusters instead of using the eye
        '''command = self._chooseNumberOfClusters()
        btn = IconButton(self.getWindow(), "Apply", Icon.ACTION_CONTINUE, command=command)
        btn.config(relief="flat", activebackground=None, compound='left',
                   fg='black', overrelief="raised")
        btn.bind('<Button-1>')
        btn.grid(row=2, column=2, sticky='nw')'''


    def _getVisualizeDict(self):
        return {'chooseNumberOfClusters': self._chooseNumberOfClusters,
                'displayObjectiveFunction': self._visualizeObjectiveFunction}

    def _visualizeObjectiveFunction(self, e=None):
        nclusters, ob_values = self.nclusters, self.objFValues
        elbowIdx = self._elbowIdx

        plt.plot(nclusters, ob_values)
        plt.scatter([nclusters[elbowIdx]], [ob_values[elbowIdx]], color='green')
        plt.xlabel('Number of clusters')
        plt.ylabel('Objective values')
        plt.title('Objective values for each number of clusters')
        plt.show()

    def _chooseNumberOfClusters(self, e=None):
        views=[]
        nclusters = self.nclusters
        elbow_nclust = nclusters[self._elbowIdx]

        nclust = self.chooseNumberOfClusters.get()
        if nclust == -1:
            nclust = elbow_nclust
        elif nclust > max(nclusters):
            nclust = max(nclusters)

        scipion_clustering = self.buildSetOfClasses(nclust)
        self.protocol._defineOutputs(outputClasses=scipion_clustering)
        for item in self.protocol.inputMultiClasses:
            self.protocol._defineSourceRelation(item, scipion_clustering)

        #TODO: esto es un objeto SetOfClasses3D pero no sabemos como lanzar un viewer con el sin tener el protocolo
        #views.append(ObjectView(self._project, self.protocol.strId()))
        return views

    def buildSetOfClasses(self, nclust):
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
