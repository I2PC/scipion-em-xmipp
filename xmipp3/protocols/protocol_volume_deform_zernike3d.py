
# **************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
# *              David Herreros Calero (dherreros@cnb.csic.es)
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

import numpy as np

from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pwem.emlib.image import ImageHandler
from pwem.objects import Volume
from pyworkflow import VERSION_2_0


class XmippProtVolumeDeformZernike3D(ProtAnalysis3D):
    """  Performs volume deformation based on Zernike3D functions, allowing
    flexible adjustments of 3D maps. This protocol aids in modeling
    conformational changes or correcting structural distortions in volumes.

    AI Generated

    ## Overview

    The Volume Deform - Zernike3D protocol deforms one 3D volume so that it better
    matches another reference volume.

    The protocol uses Zernike3D and spherical-harmonic deformation functions to
    estimate a smooth deformation field between two maps. This is useful when two
    volumes represent related conformations of the same structure and the user
    wants to describe or model the transformation from one state to another.

    The protocol first brings the two volumes to a common working sampling rate and
    box size. It then locally aligns the input volume to the reference volume,
    estimates a Zernike3D deformation, and outputs the deformed volume.

    The main output is a volume corresponding to **Volume 2** after alignment and
    deformation toward **Volume 1**.

    ## Inputs and General Workflow

    The protocol requires two volumes:

    - **Volume 1**, the reference volume;
    - **Volume 2**, the volume to be deformed.

    The workflow has four main stages.

    First, the two maps are converted to MRC format, resampled according to the
    target resolution, and cropped to a common size. Second, the protocol performs
    a local rigid alignment of Volume 2 to Volume 1. Third, the aligned Volume 2 is
    deformed toward Volume 1 using Zernike3D deformation functions. Finally, the
    output volume header is updated with the working sampling rate and the result
    is registered in Scipion.

    ## Volume 1

    The **Volume 1** parameter defines the reference map.

    This is the volume that Volume 2 will be aligned and deformed toward. It should
    represent the target conformation or target structural state.

    The reference volume should be reasonably clean and should correspond to the
    same molecular object or comparable molecular region as Volume 2. If the two
    volumes represent unrelated structures, the deformation may be mathematically
    computed but biologically meaningless.

    ## Volume 2

    The **Volume 2** parameter defines the map to be modified.

    This volume is first locally aligned to Volume 1 and then deformed toward it.
    The output volume should therefore be interpreted as a deformed version of
    Volume 2 in the coordinate frame of Volume 1.

    Volume 2 should already be roughly comparable to Volume 1. The protocol can
    perform local alignment, but it is not intended to solve arbitrary large
    misalignments between unrelated maps.

    ## Target Resolution

    The **Target resolution** parameter defines the resolution scale used to
    prepare the maps for deformation analysis.

    The protocol computes a working sampling rate as one third of the target
    resolution, but never finer than the sampling rates of the input volumes. In
    other words, the maps are resampled so that the selected target resolution is
    placed at approximately two thirds of the Fourier spectrum.

    The default value is 8 Å, which is appropriate for smooth, medium-resolution
    deformation analysis.

    A lower numerical target resolution includes finer structural details but may
    make the deformation more sensitive to noise. A higher value focuses the
    deformation on coarser global shape changes.

    ## Resampling and Cropping

    Before deformation, both volumes are converted and resampled to a common
    working sampling rate.

    If the two resampled volumes have different box sizes, the larger one is
    cropped so that both maps have the same working dimension.

    This preprocessing ensures that the deformation program compares compatible
    volumes on the same grid.

    Users should make sure that important density is not lost during cropping. If
    one volume has a much smaller box than the other, the final comparison may
    ignore density present only in the larger map.

    ## Initial Local Alignment

    Before estimating the deformation, the protocol performs a local rigid
    alignment.

    Both volumes are first smoothed with a real-space Gaussian filter to improve
    the robustness of the alignment. The protocol then estimates the local
    transformation needed to align Volume 2 to Volume 1 and applies that
    transformation to the original Volume 2.

    This step places Volume 2 into the coordinate frame of Volume 1 before the
    non-rigid deformation is estimated.

    ## Multiresolution

    The **Multiresolution** parameter controls the filtered versions of the volumes
    used during Zernike3D deformation.

    The values define the filtering levels used by the deformation program. The
    default values provide a simple multiresolution strategy, allowing the fit to
    use information at more than one spatial scale.

    This helps the deformation focus on robust structural differences rather than
    being driven by a single frequency band.

    ## Sphere Radius

    The **Sphere radius** parameter defines the radius of the sphere where the
    spherical harmonics are computed.

    If the value is 0, the deformation program uses its default behavior.

    When a positive value is provided, the radius is internally adapted to the
    working sampling rate. This parameter is advanced and should normally be
    changed only when the user wants to restrict the deformation support to a
    specific molecular radius.

    ## Zernike Degree

    The **Zernike Degree** parameter controls the degree of the Zernike polynomials
    used to describe the deformation.

    Lower values allow only smoother and simpler deformations. Higher values allow
    more complex deformation fields.

    The default value is 3, which corresponds to a relatively smooth deformation
    model.

    Increasing this value may help represent more complex conformational changes,
    but it can also make the deformation more flexible and more sensitive to noise
    or local artifacts.

    ## Harmonic Degree

    The **Harmonical Degree** parameter controls the degree of the spherical
    harmonics used in the deformation basis.

    Together with the Zernike degree, it defines the complexity of the deformation
    field.

    The protocol validates that the Zernike degree must be greater than or equal
    to the harmonic degree. If the harmonic degree is larger, the protocol reports
    an error.

    ## Regularization

    The **Regularization** parameter penalizes large deformations.

    Higher values penalize deformation more strongly, producing smoother and more
    conservative transformations. Lower values allow more flexible deformation.

    Regularization is important because an overly flexible deformation may fit
    noise or artifacts rather than meaningful conformational differences.

    The default value is intended as a practical compromise.

    ## GPU Execution

    The protocol supports both GPU and CPU execution.

    When GPU execution is enabled, the CUDA implementation of the Zernike3D volume
    deformation program is used. When GPU execution is disabled, the CPU
    implementation is used and the number of threads can be applied.

    GPU execution is recommended when available, especially for larger volumes or
    more complex deformation settings.

    ## Strain Analysis

    The deformation step is run with strain analysis enabled.

    This means that, in addition to producing the deformed output volume, the
    underlying Zernike3D program can generate deformation-related files describing
    the estimated transformation and associated deformation quantities.

    These files may be useful for advanced analysis or visualization of the
    deformation field.

    ## Output Volume

    The main output is **outputVolume**.

    This output is written as:

    `vol1DeformedTo2.mrc`

    Despite the filename, the protocol logic aligns and deforms the input volume
    toward the reference volume. The output should therefore be interpreted as the
    deformed version of **Volume 2** in relation to **Volume 1**.

    The output volume is assigned the working sampling rate computed from the
    target-resolution and input-sampling parameters.

    ## Deformation Coefficients

    The protocol writes Zernike3D deformation information to files with the
    `Volumes` root name.

    One important file is `Volumes_clnm.txt`, which contains the basis parameters
    and deformation coefficients. When the target resolution differs from 3 Å, the
    protocol rescales these coefficients to account for the target-resolution
    change.

    These files are mainly intended for advanced users who want to inspect or reuse
    the deformation model.

    ## Interpreting the Result

    The output volume should be interpreted as a deformation-based mapping between
    two related structures.

    A meaningful result suggests that the differences between the two input maps
    can be represented by a smooth deformation. This may correspond to domain
    motions, conformational transitions, flexible fitting, or structural
    rearrangements.

    However, the deformation is not proof of a physical pathway. It is a
    mathematical transformation that makes one map resemble another under the
    chosen Zernike3D basis and regularization.

    The result should be interpreted together with the original maps, local
    resolution, map quality, and biological context.

    ## Practical Recommendations

    Use this protocol with two related volumes representing the same molecule or
    comparable molecular regions.

    Make sure the maps are roughly aligned before running the protocol, even though
    the protocol performs local alignment internally.

    Start with the default target resolution of 8 Å for global conformational
    differences.

    Use conservative Zernike and harmonic degrees at first. Increase them only
    when smoother deformation models cannot represent the expected change.

    Keep regularization enabled and avoid very small values unless the maps are
    clean and the expected deformation is complex.

    Use GPU execution when available.

    Inspect the deformed output together with both original volumes. Check whether
    the deformation improves agreement without introducing unrealistic distortions.

    ## Final Perspective

    Volume Deform - Zernike3D is a deformable map-registration protocol.

    For biological users, its main value is that it provides a way to model smooth
    structural changes between two related cryo-EM maps. It can help compare
    conformational states, study flexible regions, or generate a deformed map that
    matches a reference state more closely.

    The protocol should be treated as a structural-analysis and modeling tool. Its
    output is useful for exploring map-to-map differences, but biological
    interpretation requires validation against the original density and the
    underlying experimental context.
     """
    _label = 'volume deform - Zernike3D'
    _lastUpdateVersion = VERSION_2_0

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                           Select the one you want to use.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParam('refVolume', params.PointerParam, label="Volume 1",
                      pointerClass='Volume')
        form.addParam('inputVolume', params.PointerParam, label="Volume 2",
                      pointerClass='Volume')
        form.addParam('sigma', params.NumericListParam, label="Multiresolution", default="1 2",
                      help="Perform the analysys comparing different filtered versions of the volumes")
        form.addParam('targetResolution', params.FloatParam, label="Target resolution",
                      default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('Rmax', params.IntParam, default=0,
                      label='Sphere radius',
                      experLevel=params.LEVEL_ADVANCED,
                      help='Radius of the sphere where the spherical harmonics will be computed.')
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('penalization', params.FloatParam, default=0.00025, label='Regularization',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Penalization to deformations (higher values penalize more the deformation).')
        form.addParallelSection(threads=4, mpi=0)


    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            'fnRefVol': self._getExtraPath('ref_volume.mrc'),
            'fnInputVol': self._getExtraPath('input_volume.mrc'),
            'fnInputFilt': self._getExtraPath('input_volume_filt.mrc'),
            'fnRefFilt': self._getExtraPath('ref_volume_filt.mrc'),
            'fnOutVol': self._getExtraPath('vol1DeformedTo2.mrc')
                 }
        self._updateFilenamesDict(myDict)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._createFilenameTemplates()
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.deformStep)
        self._insertFunctionStep(self.convertOutputStep)
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ------------------------------
    def convertInputStep(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnRefVol = self._getFileName('fnRefVol')

        XdimI = self.inputVolume.get().getDim()[0]
        TsI = self.inputVolume.get().getSamplingRate()
        XdimR = self.refVolume.get().getDim()[0]
        TsR = self.refVolume.get().getSamplingRate()
        self.newTs = self.targetResolution.get() * 1.0 / 3.0
        self.newTs = max(TsI, TsR, self.newTs)
        newXdimI = XdimI * TsI / self.newTs
        newXdimR = XdimR * TsR / self.newTs
        newRmax = self.Rmax.get() * TsI / self.newTs
        self.newXdim = min(newXdimI, newXdimR)
        self.newRmax = min(newRmax, self.Rmax.get())

        ih = ImageHandler()
        ih.convert(self.inputVolume.get(), fnInputVol)
        if XdimI != newXdimI:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (fnInputVol, newXdimI))
        if newXdimI > self.newXdim:
            self.runJob("xmipp_transform_window", " -i %s --crop %d" %
                        (fnInputVol, (newXdimI-self.newXdim)))


        ih.convert(self.refVolume.get(), fnRefVol)

        if XdimR != newXdimR:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (fnRefVol, newXdimR))
        if newXdimR > self.newXdim:
            self.runJob("xmipp_transform_window", " -i %s --crop %d" %
                        (fnRefVol, (newXdimR-self.newXdim)))


    def deformStep(self):
        fnRefVol = self._getFileName('fnRefVol')
        fnOutVol = self._getFileName('fnOutVol')

        self.alignMaps()

        params = ' -i %s -r %s -o %s --analyzeStrain --l1 %d --l2 %d --sigma "%s" --oroot %s --regularization %f' % \
                 (fnOutVol, fnRefVol, fnOutVol, self.l1.get(), self.l2.get(), self.sigma.get(),
                  self._getExtraPath('Volumes'), self.penalization.get())
        if self.newRmax != 0:
            params = params + ' --Rmax %d' % self.newRmax

        if self.useGpu.get():
            self.runJob("xmipp_cuda_volume_deform_sph", params)
        else:
            if self.numberOfThreads.get() != 0:
                params = params + ' --thr %d' % self.numberOfThreads.get()
            self.runJob("xmipp_volume_deform_sph", params)

    def convertOutputStep(self):
        fnOutVol = self._getFileName('fnOutVol')
        params = ' -i %s --sampling_rate %f' % (fnOutVol, self.newTs)
        self.runJob("xmipp_image_header", params)

    def createOutputStep(self):
        if round(self.targetResolution.get(), 2) != 3.0:
            correctionFactor = self.targetResolution.get() / 3.0
            with open(self._getExtraPath('Volumes_clnm.txt'), 'r') as fid:
                lines = fid.readlines()
                basisParams = np.fromstring(lines[0], sep=' ')
                if self.Rmax.get() != 0:
                    basisParams[2] = self.Rmax.get()
                else:
                    basisParams[2] = self.refVolume.get().getDim()[0] / 2
                clnm = np.fromstring(lines[1], sep=' ') * correctionFactor
            with open(self._getExtraPath('Volumes_clnm.txt'), 'w') as fid:
                fid.write(' '.join(map(str, basisParams)) + "\n")
                fid.write(' '.join(map(str, clnm)) + "\n")
        vol = Volume()
        vol.setLocation(self._getFileName('fnOutVol'))
        vol.setSamplingRate(self.newTs)
        self._defineOutputs(outputVolume=vol)
        self._defineSourceRelation(self.inputVolume, vol)

    def alignMaps(self):
        fnInputVol = self._getFileName('fnInputVol')
        fnInputFilt = self._getFileName('fnInputFilt')
        fnRefVol = self._getFileName('fnRefVol')
        fnRefFilt = self._getFileName('fnRefFilt')
        fnOutVol = self._getFileName('fnOutVol')

        # Filter the volumes to improve alignment quality
        params = " -i %s -o %s --fourier real_gaussian 2" % (fnInputVol, fnInputFilt)
        self.runJob("xmipp_transform_filter", params)
        params = " -i %s -o %s --fourier real_gaussian 2" % (fnRefVol, fnRefFilt)
        self.runJob("xmipp_transform_filter", params)

        # Find transformation needed to align the volumes
        params = ' --i1 %s --i2 %s --local --dontScale ' \
                 '--copyGeo %s' % \
                 (fnRefFilt, fnInputFilt, self._getExtraPath("geo.txt"))
        self.runJob("xmipp_volume_align", params)

        # Apply transformation of filtered volume to original volume
        with open(self._getExtraPath("geo.txt"), 'r') as file:
            geo_str = file.read().replace('\n', ',')
        params = " -i %s -o %s --matrix %s" % (fnInputVol, fnOutVol, geo_str)
        self.runJob("xmipp_transform_geometry", params)

    # ------------------------- VALIDATE functions -----------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        l1 = self.l1.get()
        l2 = self.l2.get()
        if (l1 - l2) < 0:
            errors.append('Zernike degree must be higher than '
                          'SPH degree.')
        return errors