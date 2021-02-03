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

from pwem.convert import Ccp4Header
from pwem.objects import Transform
from pwem.objects import Volume
from pwem.emlib.image import ImageHandler
from pwem.viewers import Chimera, ChimeraView, EmProtocolViewer
from xmipp3.protocols.protocol_postProcessing_deepPostProcessing import \
    (XmippProtDeepVolPostProc)
from pwem.viewers.viewer_chimera import (Chimera, sessionFile)
from pyworkflow.viewer import DESKTOP_TKINTER, Viewer

class viewerXmippProtDeepVolPostProc(Viewer):
    """ Visualize the input and output volumes of protocol XmippProtDeepVolPostProc
        with ChimeraX (3D).
        The axes of coordinates x, y, z will be shown"""
    _label = 'viewer deepEMhancer'
    _targets = [XmippProtDeepVolPostProc]
    _environments = [DESKTOP_TKINTER]

    # =========================================================================
    # Show Volumes
    # =========================================================================

    def _visualize(self, obj, **args):
        # Input map(s) are a parameter from the protocol
        dim = 150.
        sampling = 1.
        _inputVol = None
        _inputHalf1 = None
        _inputHalf2 = None
        listVol = []
        directory = self.protocol._getExtraPath()

        if self.protocol.useHalfMapsInsteadVol.get():
            if self.protocol.halfMapsAttached.get():
                if self.protocol.inputVolume.get() is not None:
                    _inputVol = self.protocol.inputVolume.get()
                    dim = _inputVol.getDim()[0]
                    sampling = _inputVol.getSamplingRate()
                    listVol.append(_inputVol)
            else:
                if self.protocol.inputHalf1.get() is not None:
                    _inputHalf1 = self.protocol.inputHalf1.get()
                    _inputHalf2 = self.protocol.inputHalf2.get()
                    dim = _inputHalf1.getDim()[0]
                    sampling = _inputHalf1.getSamplingRate()
                    listVol.append(_inputHalf1)
                    listVol.append(_inputHalf2)

        else:
            if self.protocol.inputVolume.get() is not None:
                _inputVol = self.protocol.inputVolume.get()
                dim = _inputVol.getDim()[0]
                sampling = _inputVol.getSamplingRate()
                listVol.append(_inputVol)
        bildFileName = self.protocol._getExtraPath("axis_output.bild")
        Chimera.createCoordinateAxisFile(dim,
                                         bildFileName=bildFileName,
                                         sampling=sampling)
        fnCxc = self.protocol._getExtraPath("chimera_output.cxc")
        f = open(fnCxc, 'w')
        # change to workingDir
        # If we do not use cd and the project name has an space
        # the protocol fails even if we pass absolute paths
        f.write('cd %s\n' % os.getcwd())
        f.write("open %s\n" % bildFileName)
        f.write("cofr 0,0,0\n")  # set center of coordinates
        counter = 1
        for vol in listVol:
            counter += 1
            self._visInputVolume(f, vol, counter)
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".mrc"):
                counter += 1
                volFileName = os.path.join(directory, filename)
                vol = Volume()
                vol.setFileName(volFileName)
                if self.protocol.useHalfMapsInsteadVol.get():
                    if self.protocol.halfMapsAttached.get():
                        inVol = self.protocol.inputVolume.get()
                    else:
                        inVol = self.protocol.inputHalf1.get()
                else:
                    inVol = self.protocol.inputVolume.get()

                vol.setSamplingRate(inVol.getSamplingRate())
                vol.setOrigin(inVol.getOrigin(True))

                self._visInputVolume(f, vol, counter)
                f.write("volume #%d level %0.3f\n"
                        % (counter, 0.001))

        f.close()

        return [ChimeraView(fnCxc)]

    def _visInputVolume(self, f, vol, counter):
        inputVolFileName = ImageHandler.removeFileType(vol.getFileName())
        f.write("open %s\n" % inputVolFileName)
        if vol.hasOrigin():
            x, y, z = vol.getOrigin().getShifts()
        else:
            x, y, z = vol.getOrigin(force=True).getShifts()
        f.write("volume #%d style surface voxelSize %f\n"
                "volume #%d origin %0.2f,%0.2f,%0.2f\n"
                % (counter, vol.getSamplingRate(), counter, x, y, z))