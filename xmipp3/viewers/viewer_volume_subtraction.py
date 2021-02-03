# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
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

import os
import tempfile
from distutils.spawn import find_executable
from os.path import exists, join

import pyworkflow.protocol.params as params
from pwem.emlib.image import ImageHandler
from pwem.objects import (SetOfVolumes)
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO
from pwem.viewers import Chimera, ChimeraView, EmProtocolViewer
from xmipp3.protocols.protocol_volume_adjust_sub import XmippProtVolSubtraction, XmippProtVolAdjust

VOLUME_SLICES = 1
VOLUME_CHIMERA = 0


class XmippProtVolSubtractionViewer(EmProtocolViewer):
    """ Visualize the input and output volumes of protocol XmippProtVolumeSubtraction
        by choosing Chimera (3D) or Xmipp visualizer (2D).
        The axes of coordinates x, y, z will be shown by choosing Chimera"""

    _label = 'viewer volumes subtraction'
    _targets = [XmippProtVolSubtraction, XmippProtVolAdjust]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization volumes subtraction output')
        form.addParam('displayVol', params.EnumParam,
                      choices=['chimerax', 'slices'], default=VOLUME_CHIMERA,
                      display=params.EnumParam.DISPLAY_HLIST,
                      label='Display volume with',
                      help='*chimerax*: display volumes as surface with '
                           'ChimeraX.\n*slices*: display volumes as 2D slices '
                           'along z axis.\n')

    def _getVisualizeDict(self):
        return {
            'displayVol': self._showVolumes,
        }

    def _validate(self):
        if find_executable(Chimera.getProgram()) is None:
            return ["chimerax is not available. Either install it or choose"
                    " option 'slices'. "]
        return []

    # =========================================================================
    # Show Volumes
    # =========================================================================

    def _showVolumes(self, paramName=None):
        if self.displayVol == VOLUME_CHIMERA:
            return self._showVolumesChimera()
        elif self.displayVol == VOLUME_SLICES:
            return self._showVolumesXmipp()

    def _createSetOfVolumes(self):
        tmpFileName = join(tempfile.gettempdir(), 'tmpVolumes_adjust.sqlite')
        if exists(tmpFileName):
            os.remove(tmpFileName)
        _outputVol = self.protocol.outputVolume
        setOfVolumes = SetOfVolumes(filename=tmpFileName)
        setOfVolumes.append(_outputVol)
        setOfVolumes.write()
        return setOfVolumes

    def _showVolumesChimera(self):
        tmpFileNameCMD = join(tempfile.gettempdir(), "vol_adjust_chimera.cxc")
        f = open(tmpFileNameCMD, "w")
        dim = self.protocol.outputVolume.getDim()[0]
        sampling = self.protocol.outputVolume.getSamplingRate()
        tmpFileName = os.path.abspath(join(tempfile.gettempdir(), "axis_vol_adjust.bild"))
        Chimera.createCoordinateAxisFile(dim,
                                         bildFileName=tmpFileName,
                                         sampling=sampling)
        f.write("open %s\n" % tmpFileName)
        f.write("cofr 0,0,0\n")  # set center of coordinates
        _outputVol = self.protocol.outputVolume
        outputVolFileName = os.path.abspath(ImageHandler.removeFileType(
            _outputVol.getFileName()))

        # output vol origin coordinates
        x_output, y_output, z_output = self.protocol.outputVolume.getShiftsFromOrigin()
        f.write("open %s\n" % outputVolFileName)
        f.write("volume #2 voxelSize %f origin "
                "%0.2f,%0.2f,%0.2f\n"
                % (_outputVol.getSamplingRate(), x_output/sampling, y_output/sampling, z_output/sampling))
        f.close()
        return [ChimeraView(tmpFileNameCMD)]

    def _showVolumesXmipp(self):
        setOfVolumes = self._createSetOfVolumes()
        return [self.objectView(setOfVolumes)]
