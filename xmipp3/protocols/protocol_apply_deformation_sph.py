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

from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume
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
        form.addParam('inputCoeff', params.PathParam, label="Input Coefficients",
                      important=True,
                      help='Specify a path to the deformation coefficients file.')
        form.addParam('filesPattern', params.StringParam,
                      label='Pattern',
                      help="Pattern of the files to be imported.\n\n"
                           "The pattern can contain standard wildcards such as\n"
                           "*, ?, etc, or special ones like ### to mark some\n"
                           "digits in the filename as ID.\n\n"
                           "NOTE: wildcards and special characters "
                           "('*', '?', '#', ':', '%') cannot appear in the "
                           "actual path.")

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        for idx, file in enumerate(self._iterFiles(self.filesPattern.get())):
            outFile = pwutils.removeBaseExt(self.inputVol.get().getFileName()) + '_%d_deformed.vol' % idx
            params = ' -i %s --clnm %s -o %s' % \
                     (self.inputVol.get().getFileName(), file, self._getExtraPath(outFile))
            self.runJob("xmipp_volume_apply_deform_sph", params)

    def createOutputStep(self):
        outFiles = sorted(glob.glob(self._getExtraPath('*.vol')))
        samplingRate = self.inputVol.get().getSamplingRate()
        if len(outFiles) == 1:
            outVol = Volume()
            outVol.setLocation(outFiles[0])
            outVol.setSamplingRate(samplingRate)
            self._defineOutputs(outputVolume=outVol)
            self._defineSourceRelation(self.inputVol, outVol)
        else:
            outVolumes = self._createSetOfVolumes()
            outVolumes.setSamplingRate(samplingRate)
            for file in outFiles:
                outVol = Volume()
                outVol.setLocation(file)
                outVol.setSamplingRate(samplingRate)
                outVolumes.append(outVol)
            self._defineOutputs(outputVolumes=outVolumes)
            self._defineSourceRelation(self.inputVol, outVolumes)


    # --------------------------- UTILS functions ------------------------------
    def _iterFiles(self, pattern):
        filePath = self.inputCoeff.get()
        filePaths = sorted(glob.glob(os.path.join(filePath, pattern)))
        for fn in filePaths:
            yield fn