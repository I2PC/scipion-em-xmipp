# # *************************************************
# # * This protocol will be offered in future releases (more testing is needed)
# # *************************************************

# # *****************************************************************************
# # *
# # * Authors:     David Herreros Calero     (dherreros@cnb.csic.es)
# # *
# # * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# # *
# # * This program is free software; you can redistribute it and/or modify
# # * it under the terms of the GNU General Public License as published by
# # * the Free Software Foundation; either version 2 of the License, or
# # * (at your option) any later version.
# # *
# # * This program is distributed in the hope that it will be useful,
# # * but WITHOUT ANY WARRANTY; without even the implied warranty of
# # * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # * GNU General Public License for more details.
# # *
# # * You should have received a copy of the GNU General Public License
# # * along with this program; if not, write to the Free Software
# # * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# # * 02111-1307  USA
# # *
# # *  All comments concerning this program package may be sent to the
# # *  e-mail address 'scipion@cnb.csic.es'
# # *
# # *****************************************************************************
#
# import os
# from sklearn.cluster import KMeans
# import numpy as np
#
# from pyworkflow.object import Integer
# import pyworkflow.protocol.params as param
# from pyworkflow import VERSION_2_0
# import pwem.emlib.metadata as md
# from pyworkflow.utils.properties import Message
# from pwem.protocols import ProtClassify2D, pwobj
#
# from xmipp3.convert import getXmippAttribute
#
#
# class XmippProtKmeansSPH(ProtClassify2D):
#     """ Reduce the number of deformation coefficients based on k-means vectors """
#
#     _label = 'kmeans clustering'
#     _lastUpdateVersion = VERSION_2_0
#
#     def __init__(self, **args):
#         ProtClassify2D.__init__(self, **args)
#         self.stepsExecutionMode = param.STEPS_PARALLEL
#
#     # --------------------------- DEFINE param functions -----------------------
#     def _defineParams(self, form):
#         form.addSection(label=Message.LABEL_INPUT)
#         form.addParam('inputParts', param.MultiPointerParam, pointerClass='SetOfParticles',
#                       label="Input particles",
#                       important=True)
#         form.addParam('column', param.EnumParam, choices=['SPH', 'NMA'],
#                       label='Column to cluster', default=0,
#                       help='Selecto from the type of coefficients to be used for the '
#                            'clustering.')
#
#     # --------------------------- INSERT steps functions -----------------------
#     def _insertAllSteps(self):
#         self._insertFunctionStep('findClustersStep')
#         self._insertFunctionStep('createOutputStep')
#
#     def findClustersStep(self):
#         self.kDict = []
#         coeffs = []
#         for inputParts in self.inputParts:
#             particles = inputParts.get()
#             colData = self.column.get()
#             if colData == 0:
#                 mdLabel = md.MDL_SPH_COEFFICIENTS
#             elif colData == 1:
#                 mdLabel = md.MDL_NMA
#             for particle in particles.iterItems():
#                 coeffs.append(np.fromstring(getXmippAttribute(particle, mdLabel).get(), sep=','))
#         maxClusters = 101 if len(coeffs) > 101 else len(coeffs) + 1
#         rmse = np.zeros((maxClusters - 1))
#         coeffs = np.asarray(coeffs)[:, :-8]
#         for nClusters in range(1, maxClusters):
#             kmeans = KMeans(n_clusters=nClusters).fit(coeffs)
#             rmse[nClusters-1] = np.sqrt(kmeans.inertia_ / len(coeffs))
#         p1 = np.array((1, rmse[0]))
#         p2 = np.array((maxClusters - 1, rmse[-1]))
#         d = np.zeros((maxClusters - 1))
#         for nClusters in range(2, maxClusters - 1):
#             p3 = np.array((nClusters, rmse[nClusters - 1]))
#             d[nClusters-1] = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
#         nClusters = np.argmax(d) + 1
#         self.kmeans = KMeans(n_clusters=nClusters).fit(np.asarray(coeffs))
#
#     def createOutputStep(self):
#         classes2DSet = self._createSetOfClasses2D(self.inputParts[0].get())
#         classes2DSet.classifyItems(updateItemCallback=self._updateParticle,
#                                    updateClassCallback=self._updateClass,
#                                    itemDataIterator=iter(self.kmeans.labels_))
#         result = {'outputClasses': classes2DSet}
#         classes2DSet.L1 = Integer(self.inputParts[0].get().L1)
#         classes2DSet.L2 = Integer(self.inputParts[0].get().L2)
#         classes2DSet.Rmax = Integer(self.inputParts[0].get().Rmax)
#         self._defineOutputs(**result)
#         for inputParts in self.inputParts:
#             self._defineSourceRelation(inputParts, classes2DSet)
#
#     # ------------------------------- UTILS functions -------------------------------
#     def _updateParticle(self, item, idc):
#         item.setClassId(idc)
#
#     def _updateClass(self, item):
#         coeff = self.kmeans.cluster_centers_[item.getObjId()-1]
#         coeff = ','.join(['%f' % num for num in coeff])
#         representative = pwobj.CsvList()
#         representative.set(coeff)
#         # setattr(representative, '_classID', pwobj.Integer(item.getObjId()))
#         # setattr(representative, '_objDoStore', True)
#         item.setRepresentative(representative)
