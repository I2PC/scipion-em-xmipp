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
This module implement the wrappers around
visualization program.
"""
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import Form, LabelParam, IntParam, HiddenBooleanParam
from pyworkflow.protocol.params import GE

from pwem.viewers.showj import *
from pwem.viewers import ObjectView
from pwem.objects import SetOfClasses

from xmipp3.protocols.protocol_consensus_classes import XmippProtConsensusClasses

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.cluster

class XmippConsensusClassesViewer(ProtocolViewer):
    """ Visualization of results from the consensus classes 3D protocol
    """
    _label = 'viewer consensus classes'
    _targets = [XmippProtConsensusClasses]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    def _defineParams(self, form: Form):
        form.addSection(label='Classes')
        form.addParam('visualizeAllClasses', HiddenBooleanParam,
                      label='All classes' )
        form.addParam('visualizeClasses', IntParam,
                      validators=[GE(1)], default=1,
                      label='Specific class' )

        form.addSection(label='Graphs')
        form.addParam('visualizeDendrogram', LabelParam,
                       label='Dendrogram' )
        form.addParam('visualizeCostFunction', LabelParam,
                       label='Cost function' )


    def _getVisualizeDict(self):
        return {
            'visualizeAllClasses': self._visualizeAllClasses,
            'visualizeClasses': self._visualizeClasses,
            'visualizeDendrogram': self._visualizeDendrogram,
            'visualizeCostFunction': self._visualizeCostFunction,
        }
    
    # --------------------------- UTILS functions ------------------------------
    def _getLinkageMatrix(self) -> np.ndarray:
        return self.protocol._readLinkageMatrix()
    
    def _getElbows(self) -> dict:
        return self.protocol._readElbows()
    
    def _getMergedIntersections(self, size) -> SetOfClasses:
        return self.protocol._obtainMergedIntersections(size)
    
    def _visualizeAllClasses(self, param=None):
        labels = 'enabled id _size _representative._filename _xmipp_classIntersectionSizePValue _xmipp_classIntersectionRelativeSizePValue'
        labelRender = '_representative._filename'
        outputClasses = self.protocol.outputClasses
        return [ObjectView( self._project, outputClasses.strId(), outputClasses.getFileName(),
                            viewParams={ORDER: labels,
                                        VISIBLE: labels,
                                        RENDER: labelRender,
                                        SORT_BY: '_size desc',
                                        MODE: MODE_MD})]

    def _visualizeClasses(self, param=None):
        count = self.visualizeClasses.get()
        classes = self._getMergedIntersections(count)
        return self._showSetOfClasses3D(classes)

    def _visualizeDendrogram(self, param=None):
        linkage = self._getLinkageMatrix()
        elbows = self._getElbows()
        labels = np.arange(1, len(linkage)+2)
        y = linkage[:,2]
        
        fig, ax = plt.subplots()
        scipy.cluster.hierarchy.dendrogram(linkage, ax=ax, labels=labels)
        
        # Plot the elbows
        for key, value in elbows.items():
            index = len(y) - value
            label = key + ': ' + str(value)
            ax.axhline((y[index] + y[index+1])/2, label=label, color='black', linestyle='--')
        
        ax.legend()
        ax.set_ylabel('cost')
        ax.set_xlabel('classId')
        ax.set_title('Dendrogram')
        
        return [fig]
        
    def _visualizeCostFunction(self, param=None):
        linkage = self._getLinkageMatrix()
        elbows = self._getElbows()
        y = linkage[:,2]
        x = np.arange(len(y), 0, -1)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        
        # Plot the elbows
        for key, value in elbows.items():
            index = len(x) - value
            label = key + ': ' + str(value)
            ax.scatter([x[index]], [y[index]], label=label)
        
        ax.legend()
        ax.set_ylabel('cost')
        ax.set_xlabel('class count')
        ax.set_title('Cost function')
        
        return [fig]
    
    def _showSetOfClasses3D(self, classes):
        labels = 'enabled id _size _representative._filename _xmipp_classIntersectionSizePValue _xmipp_classIntersectionRelativeSizePValue'
        labelRender = '_representative._filename'
        return [ObjectView( self._project, classes.strId(), classes.getFileName(),
                            viewParams={ORDER: labels,
                                        VISIBLE: labels,
                                        RENDER: labelRender,
                                        SORT_BY: '_size desc',
                                        MODE: MODE_MD})]
        