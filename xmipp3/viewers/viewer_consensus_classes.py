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
from pyworkflow.protocol.params import LabelParam, IntParam
from pyworkflow.protocol.params import GE

from pwem.viewers.showj import *
from pwem.viewers import TableView, ObjectView
from pwem.objects import SetOfClasses

from xmipp3.protocols.protocol_consensus_classes import XmippProtConsensusClasses

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster

class XmippConsensusClassesViewer(ProtocolViewer):
    """ Visualization of results from the consensus classes 3D protocol
    """
    _label = 'viewer consensus classes'
    _targets = [XmippProtConsensusClasses]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    def _defineParams(self, form):
        form.addSection(label='Classes')
        form.addParam('visualizeClasses', IntParam,
                      validators=[GE(1)], default=1,
                      label='Classes' )

        form.addSection(label='Graphs')
        form.addParam('visualizeDendrogram', LabelParam,
                       label='Dendrogram' )
        form.addParam('visualizeCostFunction', LabelParam,
                       label='Cost function' )


    def _getVisualizeDict(self):
        return {
            'visualizeClasses': self._visualizeClasses,
            'visualizeDendrogram': self._visualizeDendrogram,
            'visualizeCostFunction': self._visualizeCostFunction
        }
    
    # --------------------------- UTILS functions ------------------------------
    def _getLinkageMatrix(self) -> np.ndarray:
        return np.load(self.protocol._getLinkageMatrixFilename())
    
    def _getMergedIntersections(self, size) -> SetOfClasses:
        t = type(self.protocol._getInputClassification(0))
        print(t)
        suffix = self.protocol._getMergedIntersectionSuffix(size)
        filename =  self.protocol._getOutputSqliteFilename(suffix)
        return t(filename=filename)
    
    def _visualizeClasses(self, param=None):
        count = self.visualizeClasses.get()
        classes = self._getMergedIntersections(count)
        return self._showSetOfClasses3D(classes)
        
    def _visualizeDendrogram(self, param=None):
        linkage = self._getLinkageMatrix()
        labels = np.arange(1, len(linkage)+2)
        
        fig, ax = plt.subplots()
        scipy.cluster.hierarchy.dendrogram(linkage, ax=ax, labels=labels)
        ax.set_ylabel('cost')
        ax.set_xlabel('classId')
        
        return [fig]
        
    def _visualizeCostFunction(self, param=None):
        linkage = self._getLinkageMatrix()
        y = linkage[:,2]
        x = np.arange(len(y), 0, -1)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_ylabel('cost')
        ax.set_xlabel('class count')
        
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
        