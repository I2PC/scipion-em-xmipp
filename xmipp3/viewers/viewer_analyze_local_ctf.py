# **************************************************************************
# *
# * Authors:     Carlos Óscar Sánchez Sorzano
# *              Estrella Fernández Giménez
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

import matplotlib.pyplot as plt

from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, LabelParam

from protocols import XmippProtAnalyzeLocalCTF
from protocols.nma.data import Point


class XmippAnalyzeLocalCTFViewer(ProtocolViewer):
    """ Visualization of the output of analyze_local_ctf protocol
    """
    _label = 'viewer analyze local defocus'
    _targets = [XmippProtAnalyzeLocalCTF]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayLocalDefocus', StringParam, default='1',
                      label='Display local defocus',
                      help='Type the ID of the micrograph to see particle local defocus of that micrograph')

    def _getVisualizeDict(self):
        return {'displayLocalDefocus': self._viewLocalDefocus}

    def _viewLocalDefocus(self, paramName):
        components = self.displayLocalDefocus.get()
        return self._doViewLocalDefocus(components)

    def _doViewLocalDefocus(self, components):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        data = self.loadData()
        x = data.xbyId
        y = data.ybyId
        defocus = data.meanDefocusbyId
        c = self._getExtraPath("micrographCoef.xmd")
        pol = c(0) + c(1)*x + c(2)*y

        for c, m in [('r','o'), ('b','^')]:
            self.particlesDef = ax.scatter(x,y,defocus, c=c, marker=m)
            self.plane = ax.scatter(x,y,pol, c=c, marker=m)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (defocus)')

        return (self.particlesDef, self.plane)

    def loadData(self):
        """ Iterate over the images and create a Data object with theirs Points.
        """
        particles = self.protocol.getInputParticles()
        mics = self.protocol.getInputMics()

        data = self._getExtraPath("micrographDefoci.xmd")
        for i, particle in enumerate(particles):
            data.addPoint(Point(pointId=particle.getObjId()))

        for i, mic in enumerate(mics):
            data.addPoint(Point(pointId=mic.getObjId()))

        return data

