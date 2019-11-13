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
from scipy import stats as s
from sklearn.cluster import KMeans
from pyworkflow.em.protocol import EMProtocol
from pyworkflow.protocol.params import PointerParam, FloatParam
from tomo.protocols import ProtTomoBase


class XmippProtConnectedComponents(EMProtocol, ProtTomoBase):
    """ This protocol takes a set of coordinates and identifies connected components ("clusters") among the picked
    particles, keeping just the coordinates in the biggest cluster. This is performed in order to ideally keep  the
    "real" particles, which are supposed to be in a region of interest, and removing the coordinates picked in spread
    areas and background."""

    _label = 'tomo picking cleaner'

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input subtomograms')
        form.addParam('inputCoordinates', PointerParam, label="Input Coordinates",
                      pointerClass='SetOfCoordinates3D', help='Select the SetOfCoordinates3D.')
        form.addParam('distance', FloatParam, label='Distance', default=100, help='Maximum radial distance (in pixels) '
                                                                                  'between picked particles to consider'
                                                                                  ' that they are in the same cluster.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('computeConnectedComponents')
        self._insertFunctionStep('createOutput')

    # --------------------------- STEPS functions -------------------------------
    def computeConnectedComponents(self):
        inputCoor = self.inputCoordinates.get()
        minDist = self.distance.get()

        # Construct the adjacency matrix (A)
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
                    else:
                        A[j, k] = 0
        np.savetxt(self._getExtraPath('adjacency_matrix'), A)

        # Construct the degree matrix (D)
        D = np.diag(A.sum(axis=1))
        np.savetxt(self._getExtraPath('degree_matrix'), D)

        # Compute the Laplacian (L) and its eigenvalues and eigenvectors to perform spectral clustering
        L = D - A
        np.savetxt(self._getExtraPath('laplacian_matrix'), L)
        vals, vecs = np.linalg.eig(L)
        vecs = vecs[:, np.argsort(vals)]
        vals = vals[np.argsort(vals)]
        # print("vals:", vals)

        # The number of eigenvalues = 0 => number of connected components (n)
        nonzeros = np.count_nonzero(vals)
        n = len(vals) - nonzeros
        print("# clusters: ", n)

        # Kmeans on first three vectors with nonzero eigenvalues: kmeans = KMeans(n_clusters=4); kmeans.fit(vecs[:,1:4])
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(vecs[:, 1:n])  # ???
        labels = kmeans.labels_
        print("Particle labels: ", labels)
        # it seems that the index of the label corresponds with the id of the coordinate (but it is not demonstrated!!)
        # count which is the biggest cluster and get the index of the labels belonging to the biggest cluster
        # if there are more than 1 cluster with "maximun size" it takes the one with the lowest label
        self.coorIndx = [i for i, x in enumerate(labels) if x == int(s.mode(labels)[0])]
        print("biggest cluster: ", int(s.mode(labels)[0]))
        # print("idxs:", self.coorIndx)

    def createOutput(self):
        # output are coordinates belonging to the biggest cc
        inputSet = self.inputCoordinates.get()
        outputSet = self._createSetOfCoordinates3D(inputSet.getVolumes())
        outputSet.copyInfo(inputSet)
        outputSet.setBoxSize(inputSet.getBoxSize())
        for coor3D in inputSet.iterItems():
            if (coor3D.getObjId()-1) in self.coorIndx:
                outputSet.append(coor3D)
        self._defineOutputs(outputSetOfCoordinates3D=outputSet)
        self._defineSourceRelation(inputSet, outputSet)

    # --------------------------- INFO functions --------------------------------
    def _validate(self):
        validateMsgs = []
        return validateMsgs

    def _summary(self):
        summary = []
        summary.append("Maximum radial distance between particles in the same cluster: %d pixels\nParticles removed: %d"
                       % (self.distance.get(), self.inputCoordinates.get().getSize() -
                          self.outputSetOfCoordinates3D.getSize()))
        return summary

    def _methods(self):
        methods = []
        methods.append("The biggest connected component identified, with a maximum radial distance of %d pixels, "
                       "contains %d particles, which have been kept, removing %d false picked particles."
                       % (self.distance.get(), self.outputSetOfCoordinates3D.getSize(),
                          self.inputCoordinates.get().getSize() - self.outputSetOfCoordinates3D.getSize()))
        return methods

