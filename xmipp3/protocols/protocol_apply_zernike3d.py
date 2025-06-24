
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *              James Krieger (jmkrieger@cnb.csic.es)
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
from pwem.objects import AtomStruct, SetOfAtomStructs
from pwem.objects import Volume, SetOfVolumes

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.object import Float, Integer


class XmippApplyZernike3D(ProtAnalysis3D):
    """ Applies deformation to atomic structures using Zernike3D basis functions. This allows flexible modeling of structural variations in atomic models to better fit experimental density maps.
 """
    _label = 'apply deformation field - Zernike3D'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('volume', params.PointerParam, label="Zernike3D volume(s)",
                      important=True, pointerClass="SetOfVolumes,Volume",
                      help='Volume(s) with Zernike3D coefficients assigned.')
        form.addParam('inputVolumeMask', params.PointerParam, label="Input volume mask", pointerClass='VolumeMask',
                      condition="volume and not hasattr(volume,'refMask')")
        form.addParam('L1', params.IntParam,
                      label='Zernike Degree',
                      condition="volume and not hasattr(volume,'L1')",
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('L2', params.IntParam,
                      label='Harmonical Degree',
                      condition="volume and not hasattr(volume,'L2')",
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('applyPDB', params.BooleanParam, label="Apply to structure?",
                      default=False,
                      help="If True, you will be able to provide an atomic structure to be deformed "
                           "based on the Zernike3D coefficients associated to the input volume(s). "
                           "If False, the coefficients will be applied to the volume(s) directly.")
        form.addParam('inputPDB', params.PointerParam, label="Input PDB",
                      pointerClass='AtomStruct', allowsNull=True, condition="applyPDB==True",
                      help='Atomic structure to apply the deformation fields defined by the '
                           'Zernike3D coefficients associated to the input volume. '
                           'For better results, the volume(s) and structure should be aligned')
        form.addParam('moveBoxOrigin', params.BooleanParam, default=False, condition="applyPDB==True",
                      label="Move structure to box origin?",
                      help="If PDB has been aligned inside Scipion, set to False. Otherwise, this option will "
                           "correctly place the PDB in the origin of the volume.")

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("deformStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions ------------------------------
    def deformStep(self):
        if isinstance(self.volume.get(), Volume):
            self.volumes = [self.volume.get()]
        else:
            self.volumes = self.volume.get()

        num_vols = len(self.volumes)
        self.len_num_vols = len(str(num_vols))

        for i, volume in enumerate(self.volumes):
            i_pad = str(i).zfill(self.len_num_vols)

            # Write coefficients to file
            z_clnm_file = self._getExtraPath("z_clnm_{0}.txt".format(i_pad))
            z_clnm = volume._xmipp_sphCoefficients.get()
            self.writeZernikeFile(z_clnm, z_clnm_file)

            if self.applyPDB.get():
                boxSize = self.volume.get().getXDim()
                samplingRate = self.volume.get().getSamplingRate()
                outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed_{0}.pdb'.format(i_pad)
                params = ' --pdb %s --clnm %s -o %s --sr %f' % \
                         (self.inputPDB.get().getFileName(), z_clnm_file, self._getExtraPath(outFile),
                          samplingRate)
                if self.moveBoxOrigin.get():
                    params += " --boxsize %d" % boxSize
                self.runJob("xmipp_pdb_sph_deform", params)
            else:
                outFile = self._getExtraPath("deformed_volume_{0}.mrc".format(i_pad))
                volume_file = volume.getFileName()
                if pwutils.getExt(volume_file) == ".mrc":
                    volume_file += ":mrc"
                params = "-i %s  --step 1 --blobr 2 -o %s --clnm %s" % \
                         (volume_file, outFile, z_clnm_file)
                if volume.refMask:
                    mask_file = volume.refMask.get() if hasattr(volume, 'refMask') \
                                else self.inputVolumeMask.get()
                    if pwutils.getExt(mask_file) == ".mrc":
                        volume_file += ":mrc"
                    params += " --mask %s" % mask_file
                self.runJob("xmipp_volume_apply_coefficient_zernike3d", params)

    def createOutputStep(self):
        L1 = self.volume.get().L1 if hasattr(self.volume.get(), 'L1') \
                                     else Integer(self.L1.get())
        L2 = self.volume.get().L2 if hasattr(self.volume.get(), 'L2') \
            else Integer(self.L2.get())
        Rmax = Float(int(0.5 * self.volume.get().getXDim()))
        if isinstance(self.volumes, list):
            volume = self.volume.get()

            refMap = volume.refMap
            refMask = volume.refMask
            z_clnm = volume._xmipp_sphCoefficients

            if self.applyPDB.get():
                outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed.pdb'
                pdb = AtomStruct(self._getExtraPath(outFile))
                pdb.L1 = L1
                pdb.L2 = L2
                pdb.Rmax = Float(volume.getSamplingRate() * Rmax.get())
                pdb.refMap = refMap
                pdb.refMask = refMask
                pdb._xmipp_sphCoefficients = z_clnm
                self._defineOutputs(deformed=pdb)
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
                vol._xmipp_sphCoefficients = z_clnm
                self._defineOutputs(deformed=vol)
                self._defineSourceRelation(volume, vol)
        else:
            if self.applyPDB.get():
                pdbs = SetOfAtomStructs().create(self._getPath())
            else:
                vols = self._createSetOfVolumes()
                vols.setSamplingRate(self.volumes.getSamplingRate())

            for i, volume in enumerate(self.volumes):
                i_pad = str(i).zfill(self.len_num_vols)

                refMap = volume.refMap
                refMask = volume.refMask
                z_clnm = volume._xmipp_sphCoefficients

                if self.applyPDB.get():
                    outFile = pwutils.removeBaseExt(self.inputPDB.get().getFileName()) + '_deformed_{0}.pdb'.format(i_pad)
                    pdb = AtomStruct(self._getExtraPath(outFile))
                    pdb.L1 = L1
                    pdb.L2 = L2
                    pdb.Rmax = Float(volume.getSamplingRate() * Rmax.get())
                    pdb.refMap = refMap
                    pdb.refMask = refMask
                    pdb._xmipp_sphCoefficients = z_clnm

                    pdbs.append(pdb)
                else:
                    vol = Volume()
                    vol.setSamplingRate(volume.getSamplingRate())
                    vol.setFileName(self._getExtraPath("deformed_volume_{0}.mrc".format(i_pad)))
                    vol.L1 = L1
                    vol.L2 = L2
                    vol.Rmax = Rmax
                    vol.refMap = refMap
                    vol.refMask = refMask
                    vol._xmipp_sphCoefficients = z_clnm

                    vols.append(vol)

            if self.applyPDB.get():
                self._defineOutputs(deformed=pdbs)
                self._defineSourceRelation(self.inputPDB, pdbs)
                self._defineSourceRelation(self.volume, pdbs)
            else:
                self._defineOutputs(deformed=vols)
                self._defineSourceRelation(self.volume, vols)

    # --------------------------- UTILS functions ------------------------------
    def writeZernikeFile(self, z_clnm, file):
        volume = self.volume.get()
        L1 = volume.L1.get() if hasattr(volume, 'L1') \
             else self.L1.get()
        L2 = volume.L2.get() if hasattr(volume, 'L2') \
             else self.L2.get()
        Rmax = int(0.5 * volume.getXDim())
        Rmax = volume.getSamplingRate() * Rmax if self.applyPDB.get() else Rmax
        with open(file, 'w') as fid:
            fid.write(' '.join(map(str, [L1, L2, Rmax])) + "\n")
            fid.write(z_clnm.replace(",", " ") + "\n")
