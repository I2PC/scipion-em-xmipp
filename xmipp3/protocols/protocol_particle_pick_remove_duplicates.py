# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
# *             Tomas Majtner (tmajtner@cnb.csic.es)  -- streaming version
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************
"""
Consensus picking protocol
"""

import os

from math import sqrt
import numpy as np

import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtParticlePicking
from pyworkflow.em.data import SetOfCoordinates, Coordinate

class XmippProtPickingRemoveDuplicates(ProtParticlePicking):
    """
    This protocol removes coordinates that are closer than a given threshold.
    The remaining coordinate is the average of the previous ones.
    """

    _label = 'remove duplicates'

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates", important=True,
                      help='Select the set of coordinates to compare')
        form.addParam('consensusRadius', params.IntParam, default=10,
                      label="Radius",
                      help="All coordinates within this radius (in pixels) "
                           "are presumed to correspond to the same particle")

#--------------------------- INSERT steps functions ---------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('removeDuplicates')
        self._insertFunctionStep('createOutputStep')

    def removeDuplicates(self):
        coords = self.inputCoordinates.get()
        consensusRadius = self.consensusRadius.get()

        self.outputSet = self._createSetOfCoordinates(coords.getMicrographs())
        self.outputSet.copyInfo(coords)

        for mic in coords.iterMicrographs():
            coordArray = np.asarray([x.getPosition() for x in coords.iterCoordinates(mic)], dtype=float)

            Ncoords = coordArray.shape[0]
            allCoords = np.zeros([Ncoords, 2])
            votes = np.zeros(Ncoords)

            Ncurrent = 0
            for n in range(Ncoords):
                coord = coordArray[n,:]
                if Ncurrent > 0:
                    dist = np.sum((coord - allCoords[0:Ncurrent]) ** 2, axis=1)
                    imin = np.argmin(dist)
                    if sqrt(dist[imin]) < consensusRadius:
                        newCoord = (votes[imin] * allCoords[imin,] + coord) / (
                                votes[imin] + 1)
                        allCoords[imin,] = newCoord
                        votes[imin] += 1
                    else:
                        allCoords[Ncurrent, :] = coord
                        votes[Ncurrent] = 1
                        Ncurrent += 1
                else:
                    allCoords[Ncurrent, :] = coord
                    votes[Ncurrent] = 1
                    Ncurrent += 1

            for n in range(Ncurrent):
                c = Coordinate()
                c.setPosition(allCoords[n,0],allCoords[n,1])
                c.setMicrograph(mic)
                self.outputSet.append(c)

    def createOutputStep(self):
        self._defineOutputs(outputCoordinates=self.outputSet)
        self._defineSourceRelation(self.inputCoordinates.get(), self.outputSet)

    def _summary(self):
        message = []
        message.append("Radius = %d" % self.consensusRadius)
        return message