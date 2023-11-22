# **************************************************************************
# *
# * Authors:  Amaya Jimenez Moreno  (ajimenez@cnb.csic.es)
# *           David Herreros Calero (dherreros@cnb.csic.es)
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
from pyworkflow.object import Set
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pwem.viewers import ChimeraView
import pyworkflow.protocol.params as params
from pyworkflow.utils import getExt, removeExt
from os.path import abspath
from xmipp3.protocols.protocol_volume_deform_zernike3d import XmippProtVolumeDeformZernike3D
from xmipp3.protocols.protocol_apply_zernike3d import XmippApplyZernike3D


class XmippVolumeDeformZernike3DViewer(ProtocolViewer):
    """ Visualize the output of protocol volume strain """
    _label = 'viewer volume deform sph'
    _targets = [XmippProtVolumeDeformZernike3D, XmippApplyZernike3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    OPEN_FILE = "open %s\n"
    VOXEL_SIZE = "volume #%d voxelSize %s\n"
    VOL_HIDE = "vol #%d hide\n"
    VIEW = "view\n"

    def _defineParams(self, form):
        if isinstance(self.protocol, XmippApplyZernike3D):
            self.only_apply = True
            self.deformed = self.protocol.deformed.get()
            self.have_set = isinstance(self.deformed, Set)
        else:
            self.have_set = False

        form.addSection(label='Show deformation')
        form.addParam('idChoice', params.EnumParam,
                      condition='self.have_set==True',
                      choices=list(self.deformed.getIdSet()), default=0,
                      label='Structure to display', display=params.EnumParam.DISPLAY_COMBO,
                      help='Select which volume to display from the IDs of the set')
        form.addParam('doShowStrain', params.LabelParam,
                      condition='self.only_apply==False',
                      label="Display the strain deformation")
        form.addParam('doShowRotation', params.LabelParam,
                      condition='self.only_apply==False',
                      label="Display the rotation deformation")
        form.addParam('doShowMorphOrigRef', params.LabelParam,
                      condition='self.only_apply==False',
                      label="Display the morphing between original and reference volumes")
        form.addParam('doShowMorphDeformedRef', params.LabelParam,
                      label="Display the morphing between deformed and reference volumes")

    def _getVisualizeDict(self):
        if isinstance(self.protocol, XmippApplyZernike3D):
            if self.have_set==True:
                self.inputVol = self.protocol.volume[list(self.deformed.getIdSet())[self.pdbIdChoice]]
                self.chosen = self.deformed[list(self.deformed.getIdSet())[self.pdbIdChoice]]
            else:
                self.inputVol = self.protocol.volume
                self.chosen = self.deformed

            myDict = {
                'fnRefVol': self.inputVol.getFileName(),
                'fnOutVol': self.chosen.getFileName()
                    }
            self.protocol._updateFilenamesDict(myDict)

        self.protocol._createFilenameTemplates()
        return {'doShowStrain': self._doShowStrain,
                'doShowRotation': self._doShowRotation,
                'doShowMorphOrigRef': self._doShowMorphOrigRef,
                'doShowMorphDeformedRef': self._doShowDeformedOrigRef
                }


    def _doShowStrain(self, param=None):

        scriptFile = self.protocol._getPath('strain_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnInputVol'))
        ext = getExt(self.protocol._getFileName('fnInputVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnOutVol'))
        fnStrain = abspath(fnbase2)

        fhCmd.write(self.OPEN_FILE % fninput)
        fhCmd.write(self.OPEN_FILE % (fnStrain+"_strain.mrc"))
        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % counter)
        fhCmd.write('color sample #%d map #%d palette rainbow\n' % (counter - 1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]


    def _doShowRotation(self, param=None):

        scriptFile = self.protocol._getPath('rotation_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnInputVol'))
        ext = getExt(self.protocol._getFileName('fnInputVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnOutVol'))
        fnStrain = abspath(fnbase2)

        fhCmd.write(self.OPEN_FILE % fninput)
        fhCmd.write(self.OPEN_FILE % (fnStrain+"_rotation.mrc"))
        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % (counter))
        fhCmd.write('color sample #%d map #%d palette rainbow\n' % (counter - 1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowMorphOrigRef(self, param=None):

        scriptFile = self.protocol._getPath('morph_orig_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnInputVol'))
        ext = getExt(self.protocol._getFileName('fnInputVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnRefVol'))
        ext2 = getExt(self.protocol._getFileName('fnRefVol'))
        fnref = abspath(fnbase2 + ext2[0:4])

        fhCmd.write(self.OPEN_FILE % fninput)
        fhCmd.write(self.OPEN_FILE % fnref)

        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % (counter))
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % (counter))
        fhCmd.write("volume morph #%d,%d frames 500\n" % (counter - 1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]

    def _doShowDeformedOrigRef(self, param=None):
        if self.apply_only == True and self.protocol.applyPDB.get() == True:
            raise ValueError("This viewer is only for volumes, not atomic structures")

        scriptFile = self.protocol._getPath('morph_deformed_ref_chimera.cxc')
        fhCmd = open(scriptFile, 'w')
        fnbase = removeExt(self.protocol._getFileName('fnOutVol'))
        ext = getExt(self.protocol._getFileName('fnOutVol'))
        fninput = abspath(fnbase + ext[0:4])
        smprt = self.protocol.outputVolume.getSamplingRate()

        fnbase2 = removeExt(self.protocol._getFileName('fnRefVol'))
        ext2 = getExt(self.protocol._getFileName('fnRefVol'))
        fnref = abspath(fnbase2 + ext2[0:4])

        fhCmd.write(self.OPEN_FILE % fninput)
        fhCmd.write(self.OPEN_FILE % fnref)
        counter = 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        fhCmd.write(self.VOL_HIDE % counter)
        counter += 1
        fhCmd.write(self.VOXEL_SIZE % (counter, str(smprt)))
        # fhCmd.write("focus\n")
        fhCmd.write(self.VOL_HIDE % counter)
        fhCmd.write("volume morph #%d,%d frames 500\n" %  (counter-1, counter))
        fhCmd.write(self.VIEW)
        fhCmd.close()

        view = ChimeraView(scriptFile)
        return [view]
