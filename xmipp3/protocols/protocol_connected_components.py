# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez
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
from pyworkflow.utils import importFromPlugin
ProtTomoBase = importFromPlugin("tomo.protocols", "ProtTomoBase")

class XmippProtConnectedComponents(EMProtocol, ProtTomoBase):
    """ This protocol takes a set of coordinates and identifies connected components among the picked particles."""

    _label = 'connected components'

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input subtomograms')
        form.addParam('inputCoordinates', PointerParam, label="Input Coordinates",
                      pointerClass='SetOfCoordinates3D', help='Select the SetOfCoordinates3D.')
        form.addParam('distance', FloatParam, label='Distance', help='Maximum radial distance (in pixels) between '
                                                                     'particles to consider that they are in the same '
                                                                     'connected component. Wizard returns three times '
                                                                     'the box size of the input coordinates.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('computeConnectedComponents')
        self._insertFunctionStep('createOutput')

    # --------------------------- STEPS functions -------------------------------
    def computeConnectedComponents(self):
        inputCoor = self.inputCoordinates.get()
        minDist = self.distance.get()
        A = np.zeros([inputCoor.getSize(), inputCoor.getSize()])
        coorlist = []
        for i, coor in enumerate(inputCoor.iterItems()):
            coorlist.append([coor.getX(), coor.getY(), coor.getZ()])
        for j, coor1 in enumerate(coorlist):
            for k, _ in enumerate(coorlist, start=j+1):
                if k == len(coorlist):
                    break
                else:
                    coor2 = coorlist[k]
                    if abs(coor1[0]-coor2[0]) <= minDist and abs(coor1[1]-coor2[1]) <= minDist \
                            and abs(coor1[2]-coor2[2]) <= minDist:
                        A[j, k] = 1
                        A[k, j] = 1
        np.savetxt(self._getExtraPath('adjacency_matrix'), A)
        D = np.diag(A.sum(axis=1))
        np.savetxt(self._getExtraPath('degree_matrix'), D)
        L = D - A
        np.savetxt(self._getExtraPath('laplacian_matrix'), L)
        vals, vecs = np.linalg.eig(L)
        vals = vals.real
        vecs = vecs.real
        np.savetxt(self._getExtraPath('eigenvecs_matrix'), vecs)
        np.savetxt(self._getExtraPath('eigenvalues_matrix'), vals)
        vals0list = [i for i, x in enumerate(vals.real) if abs(x) < (1/(np.sqrt(len(inputCoor)))*1e-3)]
        self.listOfSets = [[] for x in xrange(len(vals0list))]
        for k in range(len(inputCoor)):
            row = []
            for j in vals0list:
                row.append(vecs[k, j])
            row = [abs(number)for number in row]
            ixMax = np.argmax(row)
            self.listOfSets[ixMax].append(k)

    def createOutput(self):
        inputSet = self.inputCoordinates.get()
        if len(self.listOfSets) == 0:
            self._defineOutputs(output3DCoordinates=inputSet)
            self._defineSourceRelation(inputSet, inputSet)
        else:
            for ix, coorInd in enumerate(self.listOfSets):
                outputSet = self._createSetOfCoordinates3D(inputSet.getVolumes(), ix+1)
                outputSet.copyInfo(inputSet)
                outputSet.setBoxSize(inputSet.getBoxSize())
                for coor3D in inputSet.iterItems():
                    if (coor3D.getObjId()-1) in coorInd:
                        outputSet.append(coor3D)
                name = 'output3DCoordinates%s' % str(ix+1)
                args = {}
                args[name] = outputSet
                self._defineOutputs(**args)
                self._defineSourceRelation(inputSet, outputSet)

    # --------------------------- INFO functions --------------------------------

    def _validate(self):
        validateMsgs = []
        return validateMsgs

    def _summary(self):
        summary = []
        summary.append("Maximum radial distance between particles in the same connected component: %d pixels"
                       % (self.distance.get()))
        return summary

    def _methods(self):
        methods = []
        methods.append("%d connected component identified, with a maximum radial distance of %d pixels."
                       % (len(self._outputs), self.distance.get()))
        return methods

