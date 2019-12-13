# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *              Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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
import numpy as np

from pyworkflow.em import ImageHandler
from pyworkflow.em.data import Image, Volume
from pyworkflow.em.protocol import ProtAnalysis3D
from pyworkflow.protocol.params import PointerParam, EnumParam, IntParam
from pyworkflow.utils import importFromPlugin
SetOfTomograms = importFromPlugin("tomo.objects", "SetOfTomograms")
import xmippLib


class XmippProtProjectZ(ProtAnalysis3D):
    """
    Project a set of volumes or subtomograms to obtain their X, Y or Z projection of the desired range of slices.
    """
    _label = 'Projection'

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='General parameters')
        form.addParam('input', PointerParam, pointerClass="SetOfSubTomograms, SetOfVolumes",
                      label='Input Volumes')
        form.addParam('dirParam', EnumParam, choices=['X', 'Y', 'Z'], default=2, display=EnumParam.DISPLAY_HLIST,
                      label='Projection direction')
        form.addParam('rangeParam', EnumParam, choices=['All', 'Range'], default=0, display=EnumParam.DISPLAY_HLIST,
                      label='Range of slices', help='Range of slices used to compute the projection, where 0 is the '
                                                    'central slice.')
        form.addParam('cropParam', IntParam, default=10, label='Voxels', condition="rangeParam == 1",
                      help='Crop this amount of voxels in each side of the selected direction.')

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('projectZStep')
        self._insertFunctionStep('createOutputStep')
    
    # --------------------------- STEPS functions -------------------------------
    def projectZStep(self):
        input = self.input.get()
        x, y, z = input.getDim()
        n = input.getSize()

        if self.rangeParam.get() == 1:
            cropParam = self.cropParam.get()
        else:
            cropParam = 0

        dir = self.dirParam.get()

        for item in input.iterItems():
            vol = Volume()
            idx = item.getObjId()
            vol.setLocation('%d@%s' % (idx, item.getFileName()))
            vol = ImageHandler().read(vol.getLocation())
            volData = vol.getData()
            proj = np.empty([x, y])

            np.savetxt(self._getExtraPath('volData'), volData)

            if dir == 0:  # X
                volData = volData[:, :, :, x/2-cropParam:x/2+cropParam]
                for zi in range(z):
                    for yi in range(y):
                        proj[zi][yi] = sum(volData[:, zi, yi, :])

            elif dir == 1:  # Y
                volData = volData[:, :, x / 2 - cropParam:x / 2 + cropParam, :]
                for zi in range(z):
                    for xi in range(x):
                        proj[zi][xi] = sum(volData[:, zi, :, xi])

            else:       # Z
                volData = volData[:, x / 2 - cropParam:x / 2 + cropParam, :, :]
                for xi in range(x):
                    for yi in range(y):
                        proj[xi][yi] = np.sum(volData[:, :, yi, xi], axis=1)

            print(proj.size())

    def createOutputStep(self):
        pass

# --------------------------- INFO functions ------------------------------
    def _methods(self):
        vols = self.input.get()
        return ["Projection of %d volumes with dimensions %s obtained with xmipp_phantom_project"
                % (vols.getSize(), vols.getDimensions())]

    def _summary(self):
        summary = []
        if not self.isFinished():
            summary.append("Output views not ready yet.")

        if self.getOutputsSize() >= 1:
            for key, output in self.iterOutputAttributes():
                summary.append("*%s:* \n %s " % (key, output.getObjComment()))
        return summary

    def getSummary(self, imgSetOut):
        summary = []
        summary.append("\n   -Number of projections generated: %s" % imgSetOut.getSize())
        return "\n".join(summary)
