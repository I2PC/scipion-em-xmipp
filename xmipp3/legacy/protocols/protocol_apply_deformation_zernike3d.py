# # *************************************************
# # * This protocol will be offered in future releases (more testing is needed)
# # * TODO: If this protocol is added, check that sampling rate is properly set in header of mrc
# # *************************************************

# # **************************************************************************
# # *
# # * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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
# # **************************************************************************
#
# import os, glob
# import numpy as np
#
# from pwem.protocols import ProtAnalysis3D
# from pwem.objects import Volume, Integer
# import pwem.emlib.metadata as md
# import pyworkflow.protocol.params as params
# import pyworkflow.utils as pwutils
#
#
# class XmippProtApplyZernike3D(ProtAnalysis3D):
#     """ Protocol to apply the deformation computed through spherical harmonics to
#     EM maps. """
#     _label = 'apply deformation - Zernike3D'
#
#     # --------------------------- DEFINE param functions --------------------------------------------
#     def _defineParams(self, form):
#         form.addSection(label='Input')
#         form.addParam('inputVol', params.PointerParam, label="Input volume",
#                       pointerClass='Volume', important=True,
#                       help='Select a Volume to be deformed.')
#         form.addParam('inputClasses', params.PointerParam, pointerClass='SetOfClasses2D',
#                       label="Input Coefficients", important=True,
#                       help='Specify a path to the deformation coefficients metadata.')
#
#     # --------------------------- INSERT steps functions -----------------------
#     def _insertAllSteps(self):
#         self._insertFunctionStep("deformStep")
#         self._insertFunctionStep("createOutputStep")
#
#     # --------------------------- STEPS functions ------------------------------
#     def deformStep(self):
#         self.samplingRate_Volume = self.inputVol.get().getSamplingRate()
#         self.outParams = []
#         classes = self.inputClasses.get()
#         self.samplingRate_Coefficients = classes.getSamplingRate()
#         correctionFactor = self.samplingRate_Coefficients / self.samplingRate_Volume
#         for rep in classes.iterItems():
#             basisParams = [rep.L1, rep.L2, rep.Rmax]
#             coeffs = rep.getRepresentative().get()
#             coeffs = np.fromstring(coeffs, sep=',')
#             coeffs = correctionFactor * coeffs
#             idx = rep.getObjId() + 1
#             file = self._getTmpPath('coeffs.txt')
#             # np.savetxt(file, coeffs)
#             with open(file, 'w') as fid:
#                 fid.write(' '.join(map(str, basisParams)) + "\n")
#                 fid.write(' '.join(map(str, coeffs)) + "\n")
#             outFile = pwutils.removeBaseExt(self.inputVol.get().getFileName()) + '_%d_deformed.mrc' % idx
#             params = ' -i %s --clnm %s -o %s' % \
#                      (self.inputVol.get().getFileName(), file, self._getExtraPath(outFile))
#             self.runJob("xmipp_volume_apply_deform_sph", params)
#             params = ' -i %s --sampling_rate %f' % (self._getExtraPath(outFile), self.samplingRate_Volume)
#             self.runJob("xmipp_image_header", params)
#             self.outParams.append((self._getExtraPath(outFile), rep.getSize()))
#
#     def createOutputStep(self):
#         classes3DSet = self._createSetOfClasses3D(self.inputClasses.get().getImages())
#         classes3DSet.classifyItems(updateItemCallback=self._updateParticle,
#                                    updateClassCallback=self._updateClass,
#                                    itemDataIterator=self.iterClassesId())
#         result = {'outputClasses': classes3DSet}
#         self._defineOutputs(**result)
#         self._defineSourceRelation(self.inputClasses, classes3DSet)
#         self._defineOutputs(**result)
#         self._defineSourceRelation(self.inputVol, classes3DSet)
#
#     # ------------------------------- UTILS functions -------------------------------
#     def _updateParticle(self, item, idc):
#         item.setClassId(idc)
#
#     def _updateClass(self, item):
#         representative = item.getRepresentative()
#         volumeFile = pwutils.removeBaseExt(self.inputVol.get().getFileName()) + '_%d_deformed.mrc' \
#                      % (item.getObjId() + 1)
#         volumeFile = self._getExtraPath(volumeFile)
#         representative.setSamplingRate(self.samplingRate_Volume)
#         representative.setLocation(volumeFile)
#
#     def iterClassesId(self):
#         for img in self.inputClasses.get().iterClassItems():
#             yield img.getClassId()
