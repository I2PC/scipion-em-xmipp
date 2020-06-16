# **************************************************************************
# *
# * Authors:  Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pwem.viewers import ChimeraView
import pyworkflow.protocol.params as params
from pyworkflow.utils import getExt, removeExt
from os.path import abspath
from xmipp3.protocols.protocol_volume_deform_sph import XmippProtVolumeDeformSPH


class XmippVolumeDeformSphViewer(ProtocolViewer):
    """ Visualize the output of protocol volume strain """
    _label = 'viewer volume deform sph'
    _targets = [XmippProtVolumeDeformSPH]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    

    def _defineParams(self, form):
        form.addSection(label='Show deformation')
        form.addParam('doShowStrain', params.LabelParam,
                      label="Display the strain deformation")
        form.addParam('doShowRotation', params.LabelParam,
                      label="Display the rotation deformation")
        form.addParam('doShowMorphOrigRef', params.LabelParam,
                      label="Display the morphing between original and reference volumes")
        form.addParam('doShowMorphDeformedRef', params.LabelParam,
                      label="Display the morphing between deformed and reference volumes")

    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        return {'doShowStrain': self._doShowStrain,
                'doShowRotation': self._doShowRotation,
                'doShowMorphOrigRef': self._doShowMorphOrigRef,
                'doShowMorphDeformedRef': self._doShowDeformedOrigRef
                }


    def _doShowStrain(self, param=None):

        scriptFile = self.protocol._getPath('strain_chimera.cmd')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnInputVol'))
        ext = getExt(self.protocol._getFileName('fnInputVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnOutVol'))
        fnStrain = abspath(fnbase2)

        fhCmd.write("open %s\n" % fninput)
        fhCmd.write("open %s\n" % (fnStrain+"_strain.mrc"))

        fhCmd.write("volume #0 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #1 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("focus\n")
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("scolor #0 volume #1 cmap rainbow reverseColors True\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]


    def _doShowRotation(self, param=None):

        scriptFile = self.protocol._getPath('rotation_chimera.cmd')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnInputVol'))
        ext = getExt(self.protocol._getFileName('fnInputVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnOutVol'))
        fnStrain = abspath(fnbase2)

        fhCmd.write("open %s\n" % fninput)
        fhCmd.write("open %s\n" % (fnStrain+"_rotation.mrc"))

        fhCmd.write("volume #0 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #1 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("focus\n")
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("scolor #0 volume #1 cmap rainbow reverseColors True\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowMorphOrigRef(self, param=None):

        scriptFile = self.protocol._getPath('morph_orig_ref_chimera.cmd')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnInputVol'))
        ext = getExt(self.protocol._getFileName('fnInputVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnRefVol'))
        ext2 = getExt(self.protocol._getFileName('fnRefVol'))
        fnref = abspath(fnbase2 + ext2[0:4])

        fhCmd.write("open %s\n" % fninput)
        fhCmd.write("open %s\n" % fnref)

        fhCmd.write("volume #0 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #1 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("focus\n")
        fhCmd.write("vol #0 hide\n")
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("vop morph #0,1 frames 500\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowDeformedOrigRef(self, param=None):

        scriptFile = self.protocol._getPath('morph_deformed_ref_chimera.cmd')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnOutVol'))
        ext = getExt(self.protocol._getFileName('fnOutVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnRefVol'))
        ext2 = getExt(self.protocol._getFileName('fnRefVol'))
        fnref = abspath(fnbase2 + ext2[0:4])

        fhCmd.write("open %s\n" % fninput)
        fhCmd.write("open %s\n" % fnref)

        fhCmd.write("volume #0 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #1 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("focus\n")
        fhCmd.write("vol #0 hide\n")
        fhCmd.write("vol #1 hide\n")
        fhCmd.write("vop morph #0,1 frames 500\n")
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]
