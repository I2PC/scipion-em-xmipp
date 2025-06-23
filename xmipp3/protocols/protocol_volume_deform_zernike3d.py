
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
    """ Protocol for volume deformation based on Zernike3D.

    (AI Generated).

    Purpose and Scope. This protocol analyzes the deformation required to transform one 3D volume into another, using a
    spherical harmonics and Zernike polynomials framework. It quantifies shape changes between two reconstructions,
    potentially representing different conformational or functional states. It is particularly suited for identifying
    local and global deformations in flexible macromolecular complexes.

    Inputs. The user must provide two 3D volumes: a reference volume and a volume to be deformed. The input volumes
    must have their physical dimensions properly defined via the sampling rate. Additionally, the user may specify
    parameters to control the level of multiresolution analysis, the resolution target, the radius of the spherical
    region to be analyzed, and the degrees of the Zernike and spherical harmonic expansions. A regularization term
    controls how smooth or flexible the deformation is allowed to be.

    Protocol Behavior. The protocol begins by resampling both input volumes to match a target resolution and voxel
    size, ensuring compatibility for harmonic analysis. It applies Gaussian filtering to facilitate alignment and then
    computes the geometric transformation needed to align the two filtered volumes. This transformation is applied to
    the original input volume.

    Next, the protocol performs a spherical harmonic decomposition of the aligned volume using the specified degrees
    and within the specified spherical region. A multiresolution comparison is optionally conducted to assess the
    deformation at different spatial scales. The deformation parameters are regularized to prevent overfitting, with
    stronger regularization producing smoother deformations.

    The output is a new volume that represents the input volume geometrically deformed to match the reference.
    Additionally, the protocol stores quantitative deformation parameters derived from the Zernike3D basis in a
    results file.

    Outputs. The primary output is a 3D volume that has been aligned and deformed to best match the reference
    structure. This volume can be visualized to assess the extent of the transformation or used as input for further
    structural analysis. The deformation parameters (encoded in the Zernike3D space) are saved in a text file, which
    can be used for advanced analysis or visualization of shape change modes.

    User Workflow. The user selects the two volumes for comparison and adjusts advanced parameters such as Zernike
    and harmonic degrees or the regularization term only if necessary. After execution, the user can inspect the
    deformed volume and retrieve the file containing deformation coefficients. If discrepancies in sampling or box
    size exist, the protocol automatically resizes and crops the inputs to ensure proper alignment and analysis.

    Interpretation and Best Practices. Larger Zernike and harmonic degrees allow the protocol to capture finer
    deformations but increase computational cost and may lead to overfitting if not properly regularized. The
    regularization parameter should be set low enough to allow real deformations but high enough to avoid capturing
    noise. The target resolution should be chosen to balance detail and reliability, with typical values between 5 and
    10 Å.

    The deformation results should be interpreted in light of biological expectations, particularly for symmetric or
    flexible assemblies where motions may be collective or domain-specific. Users may visualize the deformation field
    or compare deformation parameters across datasets to identify consistent shape change patterns.
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
        if self.targetResolution.get() != 3.0:
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