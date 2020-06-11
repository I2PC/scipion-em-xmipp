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
from xmipp3.protocols.protocol_pdb_deform_sph import XmippProtDeformPDB


class XmippPDBDeformViewer(pwviewer.ProtocolViewer):
    """ Visualize the deformation applied to the PDB file """
    _label = 'viewer pdb deform sph'
    _targets = [XmippProtDeformPDB]
    _environments = [pwviewer.DESKTOP_TKINTER, pwviewer.WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('doShowPDB', params.LabelParam,
                      label="Display original and deformed PDB")
        form.addParam('doShowMorph', params.LabelParam,
                      label="Display a morphing between the original and deformed PDB")

    def _getVisualizeDict(self):
        return {'doShowPDB': self._doShowPDB,
                'doShowMorph': self._doShowMorph}

    def _doShowPDB(self, obj, **kwargs):
        scriptFile = self.protocol._getPath('pdb_deform_chimera.cmd')
        fhCmd = open(scriptFile, 'w')
        inputFile = os.path.abspath(self.protocol.inputPDB.get().getFileName())
        outputFile = os.path.abspath(self.protocol.outputPDB.getFileName())

        fhCmd.write("open %s\n" % inputFile)
        fhCmd.write("open %s\n" % outputFile)
        fhCmd.write("start Model Panel\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowMorph(self, obj, **kwargs):
        scriptFile = self.protocol._getPath('pdb_deform_chimera.cmd')
        fhCmd = open(scriptFile, 'w')
        inputFile = os.path.abspath(self.protocol.inputPDB.get().getFileName())
        outputFile = os.path.abspath(self.protocol.outputPDB.getFileName())

        fhCmd.write("open %s\n" % inputFile)
        fhCmd.write("open %s\n" % outputFile)
        fhCmd.write("~modeldisp #0,1\n")
        fhCmd.write("morph  start #0 frames 100\n")
        fhCmd.write("morph interpolate #1\n")
        fhCmd.write("morph movie\n")
        fhCmd.write("start Model Panel\n")
        fhCmd.write("coordset #2 1,100\n")
        fhCmd.write("wait 100\n")
        fhCmd.write("coordset #2 100,1,-1\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

