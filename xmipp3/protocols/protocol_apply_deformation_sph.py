# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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

import os, glob
import numpy as np

from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume, Integer
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils


class XmippProtApplySPH(ProtAnalysis3D):
    """ Protocol to apply the deformation computed through spherical harmonics to
    EM maps. """
    _label = 'apply deformation sph'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVol', params.PointerParam, label="Input volume",
                      pointerClass='Volume', important=True,
                      help='Select a Volume to be deformed.')
        form.addParam('inputClasses', params.PointerParam, pointerClass='SetOfClasses2D',
                      label="Input Coefficients", important=True,
                      help='Specify a path to the deformation coefficients metadata.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        self.outParams = []
        classes = self.inputClasses.get()
        for rep in classes.iterItems():
            coeffs = rep.getRepresentative().get()
            coeffs = np.fromstring(coeffs, sep=',')
            idx = rep.getObjId()
            file = self._getTmpPath('coeffs.txt')
            np.savetxt(file, coeffs)
            outFile = pwutils.removeBaseExt(self.inputVol.get().getFileName()) + '_%d_deformed.vol' % idx
            params = ' -i %s --clnm %s -o %s' % \
                     (self.inputVol.get().getFileName(), file, self._getExtraPath(outFile))
            self.runJob("xmipp_volume_apply_deform_sph", params)
            self.outParams.append((self._getExtraPath(outFile), rep.getSize()))

    def createOutputStep(self):
        self.samplingRate = self.inputVol.get().getSamplingRate()
        classes3DSet = self._createSetOfClasses3D(self.inputClasses.get().getImages())
        classes3DSet.classifyItems(updateItemCallback=self._updateParticle,
                                   updateClassCallback=self._updateClass,
                                   itemDataIterator=self.iterClassesId())
        result = {'outputClasses': classes3DSet}
        self._defineOutputs(**result)
        self._defineSourceRelation(self.inputClasses, classes3DSet)
        # if len(self.outParams) == 1:
        #     outVol = Volume()
        #     outVol.setLocation(self.outParams[0][0])
        #     outVol.setSamplingRate(samplingRate)
        #     self._defineOutputs(outputVolume=outVol)
        #     self._defineSourceRelation(self.inputVol, outVol)
        # else:
        #     outVolumes = self._createSetOfVolumes()
        #     outVolumes.setSamplingRate(samplingRate)
        #     for outTuple in self.outParams:
        #         outVol = Volume()
        #         outVol.setLocation(outTuple[0])
        #         outVol.setSamplingRate(samplingRate)
        #         setattr(outVol, '_numberImages', Integer(outTuple[1]))
        #         outVolumes.append(outVol)
        self._defineOutputs(**result)
        self._defineSourceRelation(self.inputVol, classes3DSet)

    # ------------------------------- UTILS functions -------------------------------
    def _updateParticle(self, item, idc):
        item.setClassId(idc)

    def _updateClass(self, item):
        representative = item.getRepresentative()
        volumeFile = self._getExtraPath('volume_state%02d.mrc' % item.getObjId())
        representative.setSamplingRate(self.samplingRate)
        representative.setLocation(item.getObjId(), volumeFile)

    def iterClassesId(self, set):
        for img in set.iterClassItems():
            yield img.getClassId()
