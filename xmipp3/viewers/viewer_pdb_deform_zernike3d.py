# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
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

import pyworkflow.viewer as pwviewer
import pwem.viewers.views as vi
import pyworkflow.protocol.params as params
from pwem.viewers import ChimeraView
from pyworkflow.object import Set

from xmipp3.protocols.protocol_apply_zernike3d import XmippApplyZernike3D


class XmippPDBDeformViewer(pwviewer.ProtocolViewer):
    """ Visualize the deformation applied to the PDB file """
    _label = 'viewer pdb deform sph'
    _targets = [XmippApplyZernike3D]
    _environments = [pwviewer.DESKTOP_TKINTER, pwviewer.WEB_DJANGO]
    OPEN_FILE = "open %s\n"

    def _defineParams(self, form):
        self.deformed = self.protocol.deformed.get()
        self.have_set = isinstance(self.deformed, Set)

        form.addSection(label='Show deformation')
        form.addParam('pdbIdChoice', params.EnumParam,
                      condition='self.have_set==True',
                      choices=list(self.deformed.getIdSet()),
                      default=0,
                      label='Structure to display', display=params.EnumParam.DISPLAY_COMBO,
                      help='Select which structure to display from the IDs of the set')
        form.addParam('doShowPDB', params.LabelParam,
                      label="Display original and deformed PDB or volume")
        form.addParam('doShowMorph', params.LabelParam,
                      label="Display a morphing between the original and deformed PDB or volume")

    def _getVisualizeDict(self):
        if self.have_set==True:
            self.chosen = self.deformed[list(self.deformed.getIdSet())[self.pdbIdChoice]]
        else:
            self.chosen = self.deformed

        return {'doShowPDB': self._doShowPDB,
                'doShowMorph': self._doShowMorph}

    def _doShowPDB(self, obj, **kwargs):
        if self.protocol.applyPDB.get() == True:
            scriptFile = self.protocol._getPath('pdb_deform_chimera.cxc')
            fhCmd = open(scriptFile, 'w')
            inputFile = os.path.abspath(self.protocol.inputPDB.get().getFileName())
            outputFile = os.path.abspath(self.chosen.getFileName())

            fhCmd.write(self.OPEN_FILE % inputFile)
            fhCmd.write(self.OPEN_FILE % outputFile)
            # fhCmd.write("start Model Panel\n")
            fhCmd.write("show cartoons\n")
            fhCmd.write("cartoon style width 1.5 thick 1.5\n")
            fhCmd.write("style stick\n")
            fhCmd.write("color bymodel\n")
            fhCmd.close()

            view = ChimeraView(scriptFile)
            return [view]
        else:
            raise ValueError("This viewer is only for atomic structures")

    def _doShowMorph(self, obj, **kwargs):
        if self.protocol.applyPDB.get() == True:
            scriptFile = self.protocol._getPath('pdb_deform_chimera.cxc')
            fhCmd = open(scriptFile, 'w')
            inputFile = os.path.abspath(self.protocol.inputPDB.get().getFileName())
            outputFile = os.path.abspath(self.protocol.chosen.getFileName())

            fhCmd.write(self.OPEN_FILE % inputFile)
            fhCmd.write(self.OPEN_FILE % outputFile)
            fhCmd.write("hide models\n")
            fhCmd.write("morph #1,2 frames 50 play false\n")
            fhCmd.write("coordset #3 1,\n")
            fhCmd.write("wait 50\n")
            fhCmd.write("coordset #3 50,1\n")
            fhCmd.close()

            view = ChimeraView(scriptFile)
            return [view]
        else:
            raise ValueError("This viewer is only for atomic structures")

