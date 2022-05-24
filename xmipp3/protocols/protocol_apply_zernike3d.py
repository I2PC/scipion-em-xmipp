
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


from pwem.protocols import ProtAnalysis3D
from pwem.objects import AtomStruct, Volume

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Float


class XmippApplyZernike3D(ProtAnalysis3D):
    """ Protocol for PDB deformation based on Zernike3D basis. """
    _label = 'apply deformation field - Zernike3D'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('volume', params.PointerParam, label="Zernike3D volume",
                      important=True, pointerClass="Volume",
                      help='Volume with a Zernike3D coefficients assign. For better results, '
                           'the volume and the PDB should be aligned')
        form.addParam('applyPDB', params.BooleanParam, label="Apply to structure?",
                      default=False,
                      help="If True, you will be able to provide an atomic structure to deformed "
                           "based on the Zernike3D coefficients associated to the input volume")
        form.addParam('inputPDB', params.PointerParam, label="Input PDB",
                      pointerClass='AtomStruct', allowsNull=True, condition="applyPDB==True",
                      help='Atomic structure to apply the deformation fields defined by the '
                           'Zernike3D coefficients associated to the input volume')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        # Write coefficient to file
        z_clnm_file = self._getExtraPath("z_clnm.txt")
        self.writeZernikeFile(z_clnm_file)

        if self.applyPDB.get():
            outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed.pdb'
            params = ' --pdb %s --clnm %s -o %s' % \
                     (self.inputPDB.get().getFileName(), z_clnm_file, self._getExtraPath(outFile))
            self.runJob("xmipp_pdb_sph_deform", params)
        else:
            outFile = self._getExtraPath("deformed_volume.mrc")
            volume = self.volume.get()
            volume_file = volume.getFileName()
            if pwutils.getExt(volume_file) == ".mrc":
                volume_file += ":mrc"
            params = "-i %s  --step 1 --blobr 2 -o %s --clnm %s" % \
                     (volume_file, outFile, z_clnm_file)
            if volume.refMask:
                mask_file = volume.refMask.get()
                if pwutils.getExt(mask_file) == ".mrc":
                    volume_file += ":mrc"
                params += " --mask %s" % mask_file
            self.runJob("xmipp_volume_apply_coefficient_zernike3d", params)

    def createOutputStep(self):
        volume = self.volume.get()

        L1 = volume.L1
        L2 = volume.L2
        Rmax = volume.Rmax
        refMap = volume.refMap
        refMask = volume.refMask

        if self.applyPDB.get():
            outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed.pdb'
            pdb = AtomStruct(self._getExtraPath(outFile))
            pdb.L1 = L1
            pdb.L2 = L2
            pdb.Rmax = Float(volume.getSamplingRate() * Rmax.get())
            pdb.refMap = refMap
            pdb.refMask = refMask
            self._defineOutputs(deformedStructure=pdb)
            self._defineSourceRelation(self.inputPDB, pdb)
            self._defineSourceRelation(volume, pdb)
        else:
            vol = Volume()
            vol.setSamplingRate(volume.getSamplingRate())
            vol.setFileName(self._getExtraPath("deformed_volume.mrc"))
            vol.L1 = L1
            vol.L2 = L2
            vol.Rmax = Rmax
            vol.refMap = refMap
            vol.refMask = refMask
            self._defineOutputs(deformedVolume=vol)
            self._defineSourceRelation(volume, vol)

    # --------------------------- UTILS functions ------------------------------
    def writeZernikeFile(self, file):
        volume = self.volume.get()
        L1 = volume.L1.get()
        L2 = volume.L2.get()
        Rmax = volume.getSamplingRate() * volume.Rmax.get() if self.applyPDB.get() else volume.Rmax.get()
        z_clnm = volume._xmipp_sphCoefficients.get()
        with open(file, 'w') as fid:
            fid.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
            fid.write(z_clnm.replace(",", " ") + "\n")
