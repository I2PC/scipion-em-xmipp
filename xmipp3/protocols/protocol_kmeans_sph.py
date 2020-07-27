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

from pwem.protocols import ProtClassify2D


class XmippProtKmeansSPH(ProtClassify2D):
    """ Reduce the number of deformation coefficients based on k-means vectors """

    _label = 'kmeans reduction sph'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        self.stepsExecutionMode = param.STEPS_PARALLEL

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)

        form.addParam('inputCoeffs', param.PathParam,
                      label="Input coefficients",
                      important=True,
                      help='Output metadata file from "sph angular align" program.')
        form.addParam('clusters', param.IntParam, default=50,
                      label='Number of clusters',
                      help='Number of clusters to initialize kmeans algorithm.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('findClustersStep')
        self._insertFunctionStep('createOutputStep')

    def findClustersStep(self):
        coeffs = []
        mdCoeffs = md.MetaData(self.inputCoeffs.get())
        for row in md.iterRows(mdCoeffs):
            coeffs.append(mdCoeffs.getValue(md.MDL_SPH_COEFFICIENTS, row.getObjId()))
        self.kmeans = KMeans(n_clusters=self.clusters.get()).fit(np.asarray(coeffs))


    def createOutputStep(self):
        mdOut = md.MetaData()
        fnOut = self._getExtraPath('sph_cluster.xmd')
        for label, coeff in enumerate(self.kmeans.cluster_centers_):
            row = md.Row()
            row.setValue(md.MDL_SPH_COEFFICIENTS, coeff.tolist())
            row.setValue(md.MDL_WEIGHT, float(np.sum(self.kmeans.labels_ == label).tolist()))
            row.addToMd(mdOut)
            mdOut.write(fnOut)
