# *****************************************************************************
# *
# * Authors:     David Herreros Calero     (dherreros@cnb.csic.es)
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
# *****************************************************************************

import os
from sklearn.cluster import KMeans
import numpy as np

import pyworkflow.protocol.params as param
from pyworkflow import VERSION_2_0
import pwem.emlib.metadata as md
from pyworkflow.utils.properties import Message
from pwem.protocols import ProtClassify2D, pwobj

from xmipp3.convert import getXmippAttribute


class XmippProtKmeansSPH(ProtClassify2D):
    """ Reduce the number of deformation coefficients based on k-means vectors """

    _label = 'kmeans clustering'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        self.stepsExecutionMode = param.STEPS_PARALLEL

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputParts', param.PointerParam, pointerClass='SetOfParticles',
                      label="Input particles",
                      important=True)
        form.addParam('column', param.EnumParam, choices=['SPH', 'NMA'],
                      label='Column to cluster', default=0,
                      help='Selecto from the type of coefficients to be used for the '
                           'clustering.')
        form.addParam('clusters', param.IntParam, default=50,
                      label='Number of clusters',
                      help='Number of clusters to initialize kmeans algorithm.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('findClustersStep')
        self._insertFunctionStep('createOutputStep')

    def findClustersStep(self):
        self.kDict = []
        particles = self.inputParts.get()
        colData = self.column.get()
        coeffs = []
        if colData == 0:
            mdLabel = md.MDL_SPH_COEFFICIENTS
        elif colData == 1:
            mdLabel = md.MDL_NMA
        for particle in particles.iterItems():
            coeffs.append(np.fromstring(getXmippAttribute(particle, mdLabel).get(), sep=','))
        self.kmeans = KMeans(n_clusters=self.clusters.get()).fit(np.asarray(coeffs))

    def createOutputStep(self):
        classes2DSet = self._createSetOfClasses2D(self.inputParts.get())
        classes2DSet.classifyItems(updateItemCallback=self._updateParticle,
                                   updateClassCallback=self._updateClass,
                                   itemDataIterator=iter(self.kmeans.labels_))
        result = {'outputClasses': classes2DSet}
        self._defineOutputs(**result)
        self._defineSourceRelation(self.inputParts, classes2DSet)

    # ------------------------------- UTILS functions -------------------------------
    def _updateParticle(self, item, idc):
        item.setClassId(idc)

    def _updateClass(self, item):
        coeff = self.kmeans.cluster_centers_[item.getObjId()-1]
        coeff = ','.join(['%f' % num for num in coeff])
        representative = pwobj.CsvList()
        representative.set(coeff)
        # setattr(representative, '_classID', pwobj.Integer(item.getObjId()))
        # setattr(representative, '_objDoStore', True)
        item.setRepresentative(representative)
