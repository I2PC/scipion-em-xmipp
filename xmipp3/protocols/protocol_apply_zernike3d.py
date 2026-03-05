
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
    """ Applies deformation to atomic structures using Zernike3D basis
    functions. This allows flexible modeling of structural variations in atomic
    models to better fit experimental density maps.

    AI Generated:

    What this protocol is for

    Apply deformation field – Zernike3D takes a deformation field encoded as
    Zernike3D coefficients and applies it either to a 3D volume or to an atomic
    structure (PDB). The biological motivation is flexible modeling: many
    macromolecules are not perfectly rigid, and when you estimate continuous
    deformations (for example, from Zernike3D-based heterogeneity analysis),
    you often want to materialize those deformations to see what they mean
    structurally. This protocol is the “application” step: it turns coefficients
    into an actual deformed map or a deformed structure that you can visualize,
    compare, fit, or analyze downstream.

    A typical use case is that you have one or more volumes that already carry
    Zernike3D coefficients (for instance, volumes representing different
    deformation states), and you want to generate the corresponding deformed
    outputs. If you work with atomic models, you can also warp a PDB according
    to those same coefficients to obtain an interpretable, atom-level
    representation of the deformation.

    Inputs: what you provide and what they represent

    The main input is one or more Zernike3D volume(s). In practice, this means
    a Volume or SetOfVolumes that has Zernike3D coefficients associated with
    it. Many workflows store these coefficients as part of the volume object
    produced by an upstream Zernike3D estimation step; this protocol expects
    that association.

    Depending on how the input volume(s) were generated, the protocol may also
    ask you for Zernike parameters explicitly:

    A volume mask (if the input volume does not already carry the reference
    mask internally). This mask defines the region where the deformation is
    considered meaningful and prevents deformations from being driven by or
    applied to irrelevant solvent regions.

    The Zernike Degree (L1) and Harmonical Degree (L2) (if they are not
    already stored in the volume). These control the complexity of the
    deformation basis: low degrees capture smooth, global deformations; higher
    degrees allow more localized, higher-frequency warping.

    Finally, you choose whether you want to apply the deformation to a structure:

    If Apply to structure? is set to No, the protocol will apply the
    coefficients directly to the input volume(s), producing deformed map(s).

    If Apply to structure? is set to Yes, you provide an input PDB (atomic
    structure) that will be deformed according to the coefficients.

    Applying deformation to volumes versus applying deformation to a PDB

    If you apply the deformation to volumes, the protocol produces a new
    deformed volume for each input coefficient set. This is often the most
    direct way to see the deformation in the same representation as the
    experimental map. Biologically, this is convenient for comparing states,
    generating movies, interpreting which regions move, or preparing deformed
    maps for segmentation or visualization.

    If you apply the deformation to a PDB, the protocol outputs a deformed
    atomic structure for each coefficient set. This is particularly useful when
    you want an atom-level interpretation of the motion: which domains shift,
    how secondary structure is displaced, whether loops move coherently, and
    so on. It is also useful as a starting point for further real-space
    refinement or as an interpretable model for downstream biological
    discussion.

    A key practical point is that the volume(s) and the input structure should
    already be in the same coordinate frame. If the PDB is not aligned to the
    volume reference frame, the deformation will still be applied
    mathematically, but the result will not correspond to the intended
    biological motion.

    The “Move structure to box origin?” option

    When deforming a PDB, the protocol offers Move structure to box origin?
    This is a common source of confusion in practice, so it helps to interpret
    it in workflow terms.

    If your PDB has been aligned and positioned inside Scipion with correct
    origin conventions (i.e., it already matches the volume box coordinate
    system used by the Zernike3D field), you typically keep this option
    disabled.

    If, on the other hand, your PDB comes from an external source and its
    coordinate system is not centered or not expressed in the same origin
    convention as the volume box, enabling this option helps place the structure
    correctly relative to the volume by using the volume box size. Biologically,
    this matters because Zernike3D deformation fields are defined over the
    volume box; if the model is shifted relative to that box, the applied
    deformation will act on the wrong spatial region.

    A practical rule is: if you visualize the PDB and the reference map together
    and they overlap correctly before deformation, you usually do not need to
    move to the box origin. If they do not, you likely need either an explicit
    alignment step or this origin correction (depending on the situation).

    Mask usage and biological implications

    When applying coefficients to volumes, the protocol can use a mask (either
    stored inside the input volume object or provided as an input). Biologically,
    the mask is important because deformations outside the molecular region are
    not meaningful and can introduce edge artifacts. A good mask typically
    includes the molecular density and excludes large solvent regions. If your
    upstream Zernike3D estimation was performed with a particular mask, you
    generally want to use the same one when applying the coefficients, to
    preserve consistency.

    Understanding L1 and L2 (in practical biological terms)

    The Zernike Degree (L1) and Harmonical Degree (L2) define how rich the
    deformation basis is. You do not usually “tune” these at the application
    stage (they come from how the coefficients were generated), but it helps to
    know what they imply when interpreting the results:

    Lower degrees correspond to smoother, more global motions (domain-level
    shifts, overall bending, gentle warps).

    Higher degrees allow finer spatial variation (more localized changes), which
    can capture subtler heterogeneity but can also look less physically
    plausible if pushed beyond what the data supports.

    If you see deformations that look overly wiggly or non-biological, that is
    often a sign that the underlying coefficient estimation used an overly
    flexible basis relative to the data quality—something you would address
    upstream. At the application stage, you mainly want to ensure the degrees
    you apply match the coefficients you have.

    Outputs: what you get

    The protocol produces an output object named deformed. Depending on your
    inputs and choices, this output can be:

    A single deformed Volume, if you provided one input volume and chose to
    deform volumes.

    A SetOfVolumes with one deformed volume per input coefficient set, if you
    provided a set of input volumes.

    A single deformed AtomStruct (PDB), if you provided one coefficient set and
    chose to deform a structure.

    A SetOfAtomStructs, if you provided multiple coefficient sets and chose to
    deform a structure.

    Each output carries the relevant Zernike3D parameters (L1, L2, and an
    effective Rmax scale), and it keeps references to the associated map/mask
    metadata when available, which helps maintain provenance in a Scipion
    workflow.

    Typical biological processing scenarios

    A common scenario is to take several Zernike3D coefficient states (for
    example, different conformations along a continuous trajectory) and
    generate a corresponding series of deformed maps. You can then visualize
    them as a morph or movie to communicate the motion clearly.

    Another common scenario is to deform an atomic model across those same
    states. This is particularly powerful when you want to describe motion in
    terms of domains and residues, or when you want to interpret how
    flexibility could relate to function (opening/closing, ligand access,
    allostery).

    In both cases, the main success factor is coordinate consistency: ensure
    your coefficient-bearing volumes, reference maps/masks, and any PDB you
    deform are all aligned in the same frame before applying deformation.

    Practical checks after running

    After producing deformed outputs, it is good practice to visually inspect
    them in the same viewer as the reference map. For deformed volumes, overlay
    them with the original reference to see whether motions are plausible and
    localized to expected regions. For deformed PDBs, check for obvious geometry
    issues (extreme distortions, broken-looking regions) and confirm that the
    global placement still makes sense relative to the map.

    In short, this protocol is the “make it real” step for Zernike3D deformation
    fields, producing deformed maps and/or deformed atomic structures that you
    can directly interpret biologically.
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
