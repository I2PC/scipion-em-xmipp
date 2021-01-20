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

from os.path import abspath
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pwem.viewers import ChimeraView
import pyworkflow.protocol.params as params
from pyworkflow.utils import getExt, removeExt
from os.path import abspath
from xmipp3.protocols.protocol_pdb_deform_sph import XmippProtPDBDeformSPH


class XmippPDBDeformSphViewer(ProtocolViewer):
    """ Visualize the output of protocol pdb strain """
    _label = 'viewer pdb deform sph'
    _targets = [XmippProtPDBDeformSPH]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('pdb', params.EnumParam, label='PDB to show',
                      choices=['First', 'Second'], default=0,
                      display=params.EnumParam.DISPLAY_HLIST)
        form.addParam('doShowStrain', params.LabelParam,
                      label="Display the strain deformation")
        form.addParam('doShowRotation', params.LabelParam,
                      label="Display the rotation deformation")
        form.addParam('doShowMorphOrig', params.LabelParam,
                      label="Display the morphing between first and second structures")
        form.addParam('doShowMorphDeformedRef', params.LabelParam,
                      label="Display the morphing between deformed and target structures")

    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        return {'doShowStrain': self._doShowStrain,
                'doShowRotation': self._doShowRotation,
                'doShowMorphOrig': self._doShowMorphOrigRef,
                'doShowMorphDeformedRef': self._doShowDeformedOrigRef,
                }


    def _doShowStrain(self, param=None):
        if self.pdb.get() == 0:
            pdb = self.protocol._getFileName('fnStruct_1')
        else:
            pdb = self.protocol._getFileName('fnStruct_2')
        scriptFile = self.protocol._getPath('strain_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fninput = removeExt(pdb) + "_strain.pdb"
        fninput = abspath(fninput)

        fhCmd.write("open %s\n" % fninput)
        fhCmd.write("show cartoons\n")
        fhCmd.write("cartoon style width 1.5 thick 1.5\n")
        fhCmd.write("style stick\n")
        fhCmd.write('color by occupancy palette rainbow\n')
        fhCmd.write("view\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]


    def _doShowRotation(self, param=None):
        if self.pdb.get() == 0:
            pdb = self.protocol._getFileName('fnStruct_1')
        else:
            pdb = self.protocol._getFileName('fnStruct_2')
        scriptFile = self.protocol._getPath('strain_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fninput = removeExt(pdb) + "_rotation.pdb"
        fninput = abspath(fninput)

        fhCmd.write("open %s\n" % fninput)
        fhCmd.write("show cartoons\n")
        fhCmd.write("cartoon style width 1.5 thick 1.5\n")
        fhCmd.write("style stick\n")
        fhCmd.write('color by occupancy palette rainbow\n')
        fhCmd.write("view\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowMorphOrigRef(self, param=None):
        scriptFile = self.protocol._getPath('morph_orig_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        inputFile = abspath(self.protocol._getFileName('fnStruct_1'))
        outputFile = abspath(self.protocol._getFileName('fnStruct_2'))

        fhCmd.write("open %s\n" % inputFile)
        fhCmd.write("open %s\n" % outputFile)
        fhCmd.write("hide models\n")
        fhCmd.write("morph #1,2 frames 50 play false\n")
        fhCmd.write("coordset #3 1,\n")
        fhCmd.write("wait 50\n")
        fhCmd.write("coordset #3 50,1\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowDeformedOrigRef(self, param=None):
        scriptFile = self.protocol._getPath('morph_deformed_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        if self.pdb.get() == 0:
            inputFile = removeExt(self.protocol._getFileName('fnStruct_1')) + "_deformed.pdb"
            outputFile = self.protocol._getFileName('fnStruct_2')
        else:
            inputFile = removeExt(self.protocol._getFileName('fnStruct_2')) + "_deformed.pdb"
            outputFile = self.protocol._getFileName('fnStruct_1')

        inputFile = abspath(inputFile)
        outputFile = abspath(outputFile)
        fhCmd.write("open %s\n" % inputFile)
        fhCmd.write("open %s\n" % outputFile)
        fhCmd.write("hide models\n")
        fhCmd.write("morph #1,2 frames 50 play false\n")
        fhCmd.write("coordset #3 1,\n")
        fhCmd.write("wait 50\n")
        fhCmd.write("coordset #3 50,1\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]
