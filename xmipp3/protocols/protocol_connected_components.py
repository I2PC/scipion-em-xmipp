# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez
# *              Carlos Oscar Sanchez Sorzano
# *
# *  BCU, Centro Nacional de Biotecnologia, CSIC
# *
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
import numpy as np
from pyworkflow.em.protocol import EMProtocol
from pyworkflow.protocol.params import PointerParam, FloatParam
from tomo.objects import SetOfCoordinates3D


class XmippProtConnectedComponents(EMProtocol):
    """ This protocol takes a set of coordinates and detects the ones which are located along the membrane, removing
    those ones which are not located in the membrane."""

    _label = 'connected components'

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input subtomograms')
        form.addParam('inputCoordinates', PointerParam, label="Input Coordinates", important=True,
                      pointerClass='SetOfCoordinates3D', help='Select the SetOfCoordinates3D.')
        form.addParam('distance', FloatParam, label='Distance', help='Maximum radial distance')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('computeConnectedComponents')
        self._insertFunctionStep('createOutput')

    # --------------------------- STEPS functions -------------------------------
    def computeConnectedComponents(self):
        inputCoor = self.inputCoordinates.get()
        d = self.distance.get()
        matrix = np.zeros([inputCoor.getSize(), inputCoor.getSize()])
        coorlist = []
        for i, coor in enumerate(inputCoor.iterItems()):
            coorlist.append([coor.getX(), coor.getY(), coor.getZ()])
        for j, coor1 in enumerate(coorlist):
            for k, _ in enumerate(coorlist, start=j):
                if k == len(coorlist):
                    break
                else:
                    coor2 = coorlist[k]
                    if abs(coor1[0]-coor2[0]) <= d and abs(coor1[1]-coor2[1]) <= d and abs(coor1[2]-coor2[2]) <= d:
                        matrix[j, k] = 1
                    else:
                        matrix[j, k] = 0
        np.savetxt(self._getExtraPath('connection_matrix'), matrix)

    def createOutput(self):
        pass
        # inputSet = self.inputCoordinates.get()
        # outputSet = SetOfCoordinates3D()
        # # outputSet.copyInfo(inputSet)
        # # outputSet.copyItems(inputSet, updateItemCallback=self._updateItem)
        # self._defineOutputs(outputSetOfCoordinates3D=outputSet)
        # self._defineSourceRelation(self.inputCoordinates, outputSet)

    # --------------------------- INFO functions --------------------------------
    def _validate(self):
        validateMsgs = []
        return validateMsgs

    def _summary(self):
        summary = []
        summary.append("")
        return summary

    def _methods(self):
        methods = []
        methods.append("")
        return methods
