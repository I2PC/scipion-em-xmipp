# **************************************************************************
# *
# * Authors:  Roberto Marabini (roberto@cnb.csic.es), May 2013
# *           Marta Martinez (mmmtnez@cnb.csic.es)
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
from distutils.spawn import find_executable
from os.path import exists

import pyworkflow.protocol.params as params
from pwem.constants import SYM_I222
from pwem.emlib.image import ImageHandler
from pwem.objects import (SetOfVolumes)
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO
from pwem.viewers import Chimera, ChimeraView, EmProtocolViewer
from xmipp3.protocols.protocol_extract_asymmetric_unit import XmippProtExtractUnit
from xmipp3.constants import (XMIPP_TO_SCIPION, XMIPP_I222)

VOLUME_SLICES = 1
VOLUME_CHIMERA = 0


class viewerXmippProtExtractUnit(EmProtocolViewer):
    """ Visualize the input and output volumes of protocol XmippProtExtractUnit
        by choosing Chimera (3D) or Xmipp visualizer (2D).
        The axes of coordinates x, y, z will be shown by choosing Chimera"""
    _label = 'viewer extract asymmetric unit'
    _targets = [XmippProtExtractUnit]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    # ROB: I know that there is a nice chimera interface but it does not work
    # in this case since I am interested in reading the MRC header. So I will
    # use chimera as an external program

    def _defineParams(self, form):
        form.addSection(label='Visualization of input volume and extracted '
                              'asymmetric unit')
        form.addParam('displayVol', params.EnumParam,
                      choices=['chimerax', 'slices'], default=VOLUME_CHIMERA,
                      display=params.EnumParam.DISPLAY_HLIST,
                      label='Display volume with',
                      help='*chimerax*: display volumes as surface with '
                           'ChimeraX.\n*slices*: display volumes as 2D slices '
                           'along z axis.\n')

    def _getVisualizeDict(self):
        return{
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
        if not exists(self.protocol._getExtraPath('tmpVolumes.sqlite')):
            tmpFileName = self.protocol._getExtraPath("tmpVolumes.sqlite")
            _inputVol = self.protocol.inputVolumes.get()
            _outputVol = self.protocol.outputVolume
            setOfVolumes = SetOfVolumes(filename=tmpFileName)
            setOfVolumes.append(_inputVol)
            setOfVolumes.append(_outputVol)
            setOfVolumes.write()
        else:
            tmpFileName = self.protocol._getExtraPath('tmpVolumes.sqlite')
            setOfVolumes = SetOfVolumes(filename=tmpFileName)
        return setOfVolumes

    def _showVolumesChimera(self):
        tmpFileNameCMD = self.protocol._getExtraPath("chimera.cxc")
        f = open(tmpFileNameCMD, "w")
        dim = self.protocol.inputVolumes.get().getDim()[0]
        sampling = self.protocol.inputVolumes.get().getSamplingRate()
        tmpFileName = os.path.abspath(self.protocol._getExtraPath("axis.bild"))
        Chimera.createCoordinateAxisFile(dim,
                                 bildFileName=tmpFileName,
                                 sampling=sampling)
        f.write("open %s\n" % tmpFileName)
        f.write("cofr 0,0,0\n")  # set center of coordinates
        
        _inputVol = self.protocol.inputVolumes.get()
        _outputVol = self.protocol.outputVolume
        inputVolFileName = os.path.abspath(ImageHandler.removeFileType(
            _inputVol.getFileName()))

        # input vol origin coordinates
        x_input, y_input, z_input = _inputVol.getShiftsFromOrigin()
        f.write("open %s\n" % inputVolFileName)
        f.write("volume #2 style mesh level 0.001 voxelSize %f origin "
                "%0.2f,%0.2f,%0.2f\n"
                % (_inputVol.getSamplingRate(), x_input, y_input, z_input))

        outputVolFileName = os.path.abspath(ImageHandler.removeFileType(
            _outputVol.getFileName()))

        # output vol origin coordinates
        x_output, y_output, z_output = _outputVol.getShiftsFromOrigin()
        f.write("open %s\n" % outputVolFileName)
        f.write("volume #3 style surface level 0.001 voxelSize %f origin "
                "%0.2f,%0.2f,%0.2f\n"
                % (_outputVol.getSamplingRate(), x_output, y_output, z_output))

        cMap = ['red', 'yellow', 'green', 'cyan', 'blue']
        d = {}
        innerRadius = self.protocol.innerRadius.get()
        d['outerRadius'] = self.protocol.outerRadius.get() * sampling
        if innerRadius < 0:
           innerRadius = 0
        d['innerRadius'] = innerRadius * sampling
        d['symmetry'] = Chimera.getSymmetry(XMIPP_TO_SCIPION[self.protocol.symmetryGroup.get()])

        if self.protocol.symmetryGroup >= XMIPP_I222:
            f.write("shape icosahedron mesh true radius %(outerRadius)d "
                    "orientation %(symmetry)s\n" % d)
        step = (d['outerRadius'] - d['innerRadius']) / float(len(cMap) - 1)
        f.write("color radial #3 center 0,0,0 palette -")
        counter = 0
        s = ""
        for color in cMap:
            s += "%d,%s:" % (d['innerRadius'] + counter * step, color)
            counter += 1
        f.write(s[:-1] + '\n')

        f.close()

        return [ChimeraView(tmpFileNameCMD)]

    def _showVolumesXmipp(self):

        setOfVolumes = self._createSetOfVolumes()

        return [self.objectView(setOfVolumes)]
