# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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

from pyworkflow import VERSION_1_1
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED, EnumParam, DeprecatedParam)
from pwem.protocols import ProtAnalysis3D
from pyworkflow.object import Float
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import getExt
from pwem.objects import Volume
import pwem.emlib.metadata as md

from pwem.convert.headers import Ccp4Header
from pyworkflow import BETA, UPDATED, NEW, PROD

MONORES_METHOD_URL = 'https://github.com/I2PC/scipion/wiki/XmippProtMonoRes'
OUTPUT_RESOLUTION_FILE = 'monoresResolutionMap.mrc'
OUTPUT_RESOLUTION_FILE_CHIMERA = 'monoresResolutionChimera.mrc'
OUTPUT_MASK_FILE = 'refinedMask.mrc'
FN_MEAN_VOL = 'meanvol'
METADATA_MASK_FILE = 'metadataresolutions'
FN_METADATA_HISTOGRAM = 'hist.xmd'
BINARY_MASK = 'binarymask'
FN_GAUSSIAN_MAP = 'gaussianfilter'


class XmippProtMonoRes(ProtAnalysis3D):
    """    
    Assigns local resolution values to each voxel within a given 3D map,
    providing detailed insight into regional map quality. This aids in
    interpreting structural data by highlighting areas of varying resolution.

    AI Generated

    ## Overview

    The Local MonoRes protocol estimates the local resolution of a 3D cryo-EM map
    using the monogenic-signal approach.

    Global resolution values summarize the reconstruction with a single number, but
    cryo-EM maps often have regions with different levels of detail. A rigid core
    may be well resolved, while flexible domains, peripheral regions, or poorly
    ordered parts may be resolved at lower resolution. Local MonoRes addresses this
    by assigning a local-resolution value to each voxel inside a mask.

    The result is a 3D local-resolution map. Each voxel value represents the
    estimated resolution, in angstroms, at that position. Lower numerical values
    indicate better local resolution, while higher numerical values indicate poorer
    local resolution.

    The protocol can estimate local resolution from a single full map or from two
    half maps. When half maps are used, the noise estimation can be performed from
    the difference between the two halves.

    ## Inputs and General Workflow

    The protocol can work in two main modes:

    - using a single input volume;
    - using two half volumes.

    The user may provide a binary mask. If no mask is provided, the protocol
    generates an approximate mask automatically by smoothing and thresholding the
    input volume.

    The protocol then runs the Xmipp monogenic-signal local-resolution program over
    the selected resolution range. The output local-resolution map is registered in
    Scipion as a volume. A Chimera-oriented resolution map and a histogram of
    resolution values are also generated.

    ## Use Half Volumes

    The **Would you like to use half volumes?** option controls whether the
    protocol uses one full map or two half maps.

    If this option is disabled, the protocol estimates local resolution from the
    selected full map.

    If this option is enabled, the protocol uses two half maps. This allows the
    noise estimation to be based on the difference between independently
    reconstructed halves, which is often preferable for validation-oriented local
    resolution analysis.

    Use half maps when they are available and correctly associated with the
    reconstruction.

    ## Half Maps Stored with the Input Volume

    When half-volume mode is enabled, the **Are the half volumes stored with the
    input volume?** option controls how the half maps are provided.

    If this option is enabled, the user selects **Input Half Maps**, which is a
    volume object containing associated half-map file names. The protocol reads the
    two half maps from that volume.

    If this option is disabled, the user must explicitly provide **Volume Half 1**
    and **Volume Half 2**.

    In both cases, the two half maps should have the same box size, sampling rate,
    origin, orientation, and processing state.

    ## Input Volume

    When half-volume mode is disabled, the **Input Volume** parameter defines the
    map to be analyzed.

    This map should be a reconstruction with a correct sampling rate and origin.
    The sampling rate is required to express local-resolution values in angstroms.

    The map should be reasonably preprocessed for local-resolution analysis.
    Extremely noisy maps, maps with strong artifacts, or maps with inappropriate
    masking can produce unreliable local-resolution estimates.

    ## Binary Mask

    The **Binary Mask** parameter defines the region in which local resolution is
    estimated.

    The mask separates specimen from background. If the user provides a mask, the
    protocol binarizes it using the selected mask threshold. If no mask is
    provided, the protocol creates an approximate mask automatically by applying a
    Gaussian filter to the input volume and thresholding the result.

    A good mask should include the molecular density while excluding solvent and
    irrelevant background. If the mask is too tight, it may remove real density. If
    it is too loose, background noise may affect the local-resolution estimate.

    ## Mask Threshold

    The **Mask threshold** parameter is used when the provided mask is not already
    binary.

    Mask values below the threshold are converted to 0, and values above the
    threshold are converted to 1. This creates the binary mask used by MonoRes.

    The default threshold is 0.5, which is appropriate for many masks whose values
    range between 0 and 1.

    If the input mask has a different intensity convention, the threshold may need
    to be adjusted.

    ## Automatic Mask Generation

    If no mask is provided, the protocol generates one automatically.

    It applies a real-space Gaussian-like low-pass filter to the input volume, then
    uses a threshold based on a fraction of the maximum filtered value. The result
    is binarized and used as the analysis mask.

    This automatic mask is useful for quick analysis, but a carefully prepared user
    mask is usually preferable for final reporting.

    Users should inspect the generated mask when possible, especially for maps with
    weak density, flexible regions, or strong background artifacts.

    ## Exclude Area

    The **Exclude Area** parameter allows the user to provide an additional mask
    for a region that should be excluded from local-resolution estimation.

    This is an advanced option. It may be useful when a particular part of the map,
    such as an artifact, contaminant, or region outside the biological molecule,
    should not influence the analysis.

    The exclusion mask should be used carefully, because excluding biologically
    relevant density may bias the local-resolution interpretation.

    ## Resolution Range

    The **Resolution Range** parameters define the range of resolutions tested by
    MonoRes.

    The range is given in angstroms:

    - **High** defines the high-resolution end of the search;
    - **Low** defines the low-resolution limit;
    - **Step** defines the spacing between tested resolution values.

    The protocol validates that a low-resolution limit is provided.

    The step size controls how finely the resolution range is sampled. A smaller
    step gives finer resolution estimates but increases computation time. If no
    step is provided, the protocol uses a default value of 0.5 Å.

    ## Significance

    The **Significance** parameter controls the statistical hypothesis tests used
    by MonoRes.

    The default value is 0.95. This value determines how strict the method is when
    deciding whether signal at a given resolution is significant.

    A higher significance level is more stringent. A lower value is more
    permissive, but may be more sensitive to noise.

    Most users should start with the default value.

    ## Noise Estimation from Half Maps

    When half maps are used, the **Use noise inside protein?** option controls how
    the noise distribution is estimated.

    When enabled, the noise distribution is estimated inside the protein region by
    using the difference between the two half maps. This is recommended because it
    uses the independent half-map information to characterize noise in the region
    of interest.

    If disabled, noise estimation follows the alternative behavior of the
    underlying MonoRes program.

    For most half-map workflows, the recommended setting is to keep this option
    enabled.

    ## Gaussian Noise Assumption

    The **Consider noise gaussian?** option tells the protocol to assume that the
    noise in the map follows a Gaussian distribution.

    This assumption is often approximately reasonable, but it may not always hold.
    Noise in cryo-EM maps can be affected by reconstruction, masking, filtering,
    preferred orientation, and other processing effects.

    This is an advanced option. Users should enable it only when they want the
    Gaussian-noise assumption used explicitly by the MonoRes calculation.

    ## Output Resolution Volume

    The main output is **resolution_Volume**.

    This output is a 3D volume where voxel values represent local resolution in
    angstroms. It is assigned the sampling rate and origin of the input map or half
    maps.

    The volume can be visualized directly or used to color a cryo-EM map according
    to local resolution.

    Lower values indicate regions estimated to have better local resolution.
    Higher values indicate regions estimated to have poorer local resolution.

    ## Chimera-Oriented Resolution Map

    The protocol also produces a Chimera-oriented local-resolution map internally.

    This version is used to compute the minimum and maximum resolution values
    reported in the protocol summary. It may also be useful for visualization
    depending on the workflow.

    The summary reports the highest and lowest estimated local resolution values.
    Here, “highest resolution” corresponds to the smallest angstrom value.

    ## Histogram

    The protocol creates a histogram of local-resolution values inside the output
    mask.

    The histogram summarizes the distribution of resolution values across the map.
    This is useful because two maps may have the same global resolution but very
    different local-resolution distributions.

    For example, one map may have a small well-resolved core and large poorly
    resolved regions, while another may have more uniform resolution throughout
    the structure.

    ## Interpreting the Local-Resolution Map

    The local-resolution map should be interpreted as a spatial map-quality
    estimate.

    Regions with better local resolution usually correspond to more rigid,
    well-ordered, and better-supported parts of the structure. Regions with worse
    local resolution may correspond to flexible domains, low occupancy, weaker
    alignment signal, preferred-orientation effects, or lower local particle
    support.

    Local resolution should not be interpreted in isolation. It should be compared
    with the original density, half-map agreement, global FSC, map-model fit, and
    biological knowledge.

    ## Practical Recommendations

    Use half maps when available, especially for validation-oriented analysis.

    Provide a good binary mask for final analyses. Automatic masks are convenient,
    but user-defined masks are usually more reliable.

    Set the low-resolution limit to cover the expected poorest local resolution in
    the map.

    Use a step size of 0.5 Å as a practical starting point. Use smaller steps only
    when finer sampling is needed.

    Keep the default significance value at first.

    Inspect the output resolution map together with the original density. Check
    whether flexible, peripheral, or poorly ordered regions show worse local
    resolution, as expected.

    Be cautious near mask boundaries, weak density, and regions affected by
    post-processing artifacts.

    ## Final Perspective

    Local MonoRes is a local-resolution estimation protocol based on the monogenic
    signal.

    For biological users, its main value is that it shows how map quality changes
    across the reconstruction rather than reducing the entire map to a single
    global resolution number.

    The output local-resolution map is useful for interpreting structural
    features, identifying flexible or poorly resolved regions, guiding model
    building, and communicating regional confidence in a cryo-EM map.
    """
    _label = 'local MonoRes'
    _lastUpdateVersion = VERSION_1_1
    _devStatus = PROD

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.min_res_init = Float()
        self.max_res_init = Float()

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        self.halfVolumes = DeprecatedParam('useHalfVolumes', self)
        form.addParam('useHalfVolumes', BooleanParam, default=False,
                      label="Would you like to use half volumes?",
                      help='The noise estimation for determining the '
                           'local resolution is performed via half volumes.')

        self.halfVolumesFile = DeprecatedParam('hasHalfVolumesFile', self)
        form.addParam('hasHalfVolumesFile', BooleanParam, default=True,
                      condition='useHalfVolumes', allowsNull=True,
                      label="Are the half volumes stored with the input volume?",
                      help='Usually, the half volumes are stored as properties of '
                           'the input volume. If this is not the case, set this to '
                           'False and specify the two halves you want to use.')

        self.inputVolumes = DeprecatedParam('fullMap', self)
        form.addParam('fullMap', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True,
                      condition='not useHalfVolumes',
                      help='Select a volume for determining its '
                           'local resolution.')

        form.addParam('associatedHalves', PointerParam, pointerClass='Volume',
                      label="Input Half Maps", important=True,
                      condition='useHalfVolumes and hasHalfVolumesFile',
                      help='Select a volume for determining its '
                           'local resolution.')

        self.inputVolume = DeprecatedParam('halfMap1', self)
        form.addParam('halfMap1', PointerParam, pointerClass='Volume',
                      label="Volume Half 1", important=True,
                      condition='useHalfVolumes and not hasHalfVolumesFile',
                      help='Select the first half of a volume for determining its '
                           'local resolution.')

        self.inputVolume2 = DeprecatedParam('halfMap2', self)
        form.addParam('halfMap2', PointerParam, pointerClass='Volume',
                      label="Volume Half 2", important=True,
                      condition='useHalfVolumes and not hasHalfVolumesFile',
                      help='Select the second half of a volume for determining a '
                           'local resolution.')

        self.Mask = DeprecatedParam('mask', self)
        form.addParam('mask', PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Binary Mask",
                      help='The mask determines which points are specimen'
                           ' and which are not')

        form.addParam('maskExcl', PointerParam, pointerClass='VolumeMask',
                      expertLevel=LEVEL_ADVANCED,
                      allowsNull=True,
                      label="Exclude Area",
                      help='The mask determines the area of the protein to be'
                           'excluded in the estimation of the local resolution')

        group = form.addGroup('Extra parameters')
        line = group.addLine('Resolution Range (Å)',
                             help="The range of resolutions to be analysed."
                                  "The local resolution will be estimated from low to high in "
                                  "steps given by the step box (advanced parameter)")

        group.addParam('significance', FloatParam, default=0.95,
                       expertLevel=LEVEL_ADVANCED,
                       label="Significance",
                       help='Resolution is computed using hypothesis tests, '
                            'this value determines the significance of that test')

        self.maskthreshold = DeprecatedParam('maskThreshold', self)
        group.addParam('maskThreshold', FloatParam, default=0.5,
                       expertLevel=LEVEL_ADVANCED,
                       label="Mask threshold",
                       help='If the provided mask is not binary. Then, MonoRes'
                            'will try to binarize it. Mask values below the threshold'
                            'will be change to 0 and above the thresthol will be 1')

        self.noiseonlyinhalves = DeprecatedParam('noiseOnlyInHalves', self)
        form.addParam('noiseOnlyInHalves', BooleanParam, expertLevel=LEVEL_ADVANCED,
                      default=True,
                      label="Use noise inside protein?",
                      condition='useHalfVolumes',
                      help='(Yes recommended) the noise distribution will be estimated'
                           ' in the protein region (inside the mask) by means of the '
                           'difference of both half maps.')

        form.addParam('gaussianNoise', BooleanParam, expertLevel=LEVEL_ADVANCED,
                      default=False,
                      label="Consider noise gaussian?",
                      help='It assumes the noise in the map as gaussian. '
                           'Note that this assumption might not be true, despite ,it is '
                           ' in general.')

        line.addParam('minRes', FloatParam, default=0, label='High')
        line.addParam('maxRes', FloatParam, allowsNull=True, label='Low')
        line.addParam('stepSize', FloatParam, allowsNull=True,
                      expertLevel=LEVEL_ADVANCED, label='Step')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {
            FN_MEAN_VOL: self._getExtraPath('mean_volume.mrc'),
            METADATA_MASK_FILE: self._getExtraPath('mask_data.xmd'),
            BINARY_MASK: self._getExtraPath('binarized_mask.mrc'),
            FN_GAUSSIAN_MAP: self._getExtraPath('gaussianfilted.mrc')
        }
        self._updateFilenamesDict(myDict)

    def _insertAllSteps(self):
        # Convert input into xmipp Metadata format
        self._createFilenameTemplates()
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('resolutionMonogenicSignalStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep("createHistogram")

    def convertInputStep(self):
        """ Read the input volume.
        """
        if self.useHalfVolumes:
            if not self.hasHalfVolumesFile:
                self.vol1Fn = self.halfMap1.get().getFileName()
                self.vol2Fn = self.halfMap2.get().getFileName()
            else:
                self.vol1Fn, self.vol2Fn = self.associatedHalves.get().getHalfMaps().split(',')
            extVol1 = getExt(self.vol1Fn)
            extVol2 = getExt(self.vol2Fn)
            if (extVol1 == '.mrc') or (extVol1 == '.map'):
                self.vol1Fn = self.vol1Fn + ':mrc'
            if (extVol2 == '.mrc') or (extVol2 == '.map'):
                self.vol2Fn = self.vol2Fn + ':mrc'

            if not self.mask.hasValue():
                self.ifNomask(self.vol1Fn)
            else:
                self.maskFn = self.mask.get().getFileName()
        else:
            self.vol0Fn = self.fullMap.get().getFileName()
            extVol0 = getExt(self.vol0Fn)
            if (extVol0 == '.mrc') or (extVol0 == '.map'):
                self.vol0Fn = self.vol0Fn + ':mrc'

            if not self.mask.hasValue():
                self.ifNomask(self.vol0Fn)
            else:
                self.maskFn = self.mask.get().getFileName()

        extMask = getExt(self.maskFn)
        if (extMask == '.mrc') or (extMask == '.map'):
            self.maskFn = self.maskFn + ':mrc'

        if self.mask.hasValue():
            params = ' -i %s' % self.maskFn
            params += ' -o %s' % self._getFileName(BINARY_MASK)
            params += ' --select below %f' % self.maskThreshold.get()
            params += ' --substitute binarize'

            self.runJob('xmipp_transform_threshold', params)

    def ifNomask(self, fnVol):
        if self.useHalfVolumes:
            if not self.hasHalfVolumesFile:
                xdim, _ydim, _zdim = self.halfMap1.get().getDim()
                params = ' -i %s' % fnVol
            else:
                xdim, _ydim, _zdim = self.associatedHalves.get().getDim()
                params = ' -i %s' % fnVol
        else:
            xdim, _ydim, _zdim = self.fullMap.get().getDim()
            params = ' -i %s' % fnVol
        params += ' -o %s' % self._getFileName(FN_GAUSSIAN_MAP)
        setsize = 0.02 * xdim
        params += ' --fourier real_gaussian %f' % setsize

        self.runJob('xmipp_transform_filter', params)
        img = ImageHandler().read(self._getFileName(FN_GAUSSIAN_MAP))
        imgData = img.getData()
        max_val = np.amax(imgData) * 0.05

        params = ' -i %s' % self._getFileName(FN_GAUSSIAN_MAP)
        params += ' --select below %f' % max_val
        params += ' --substitute binarize'
        params += ' -o %s' % self._getFileName(BINARY_MASK)

        self.runJob('xmipp_transform_threshold', params)

        self.maskFn = self._getFileName(BINARY_MASK)

    def resolutionMonogenicSignalStep(self):
        if self.stepSize.hasValue():
            freq_step = self.stepSize.get()
        else:
            freq_step = 0.5
        
        samplingRate = -1

        if self.useHalfVolumes:
            params = ' --vol %s' % self.vol1Fn
            params += ' --vol2 %s' % self.vol2Fn
            if self.hasHalfVolumesFile:
                samplingRate = self.associatedHalves.get().getSamplingRate()
            else:
                samplingRate = self.halfMap1.get().getSamplingRate()
            if self.noiseOnlyInHalves.get() is True:
                params += ' --noiseonlyinhalves'
        else:
            params = ' --vol %s' % self.vol0Fn
            samplingRate = self.fullMap.get().getSamplingRate()

        params += ' --sampling_rate %f' % samplingRate

        params += ' --mask %s' % self._getFileName(BINARY_MASK)

        if self.maskExcl.hasValue():
            params += ' --maskExcl %s' % self.maskExcl.get().getFileName()

        params += ' --minRes %f' % self.minRes.get()
        params += ' --maxRes %f' % self.maxRes.get()
        params += ' --step %f' % freq_step
        params += ' -o %s' % self._getExtraPath()
        if self.gaussianNoise.get() is True:
            params += ' --gaussian'
        params += ' --significance %f' % self.significance.get()
        params += ' --threads %i' % self.numberOfThreads.get()

        self.runJob('xmipp_resolution_monogenic_signal', params)

        ccp4header = Ccp4Header(self._getExtraPath(OUTPUT_RESOLUTION_FILE), readHeader=True)
        ccp4header.setSampling(samplingRate)
        ccp4header.writeHeader()
        ccp4header = Ccp4Header(self._getExtraPath(OUTPUT_RESOLUTION_FILE_CHIMERA), readHeader=True)
        ccp4header.setSampling(samplingRate)
        ccp4header.writeHeader()

    def createHistogram(self):

        M = float(self.max_res_init)
        m = float(self.min_res_init)
        range_res = round((M - m) * 4.0)

        params = ' -i %s' % self._getExtraPath(OUTPUT_RESOLUTION_FILE)
        params += ' --mask binary_file %s' % self._getExtraPath(OUTPUT_MASK_FILE)
        params += ' --steps %f' % (range_res)
        params += ' --range %f %f' % (self.min_res_init, self.max_res_init)
        params += ' -o %s' % self._getExtraPath(FN_METADATA_HISTOGRAM)

        self.runJob('xmipp_image_histogram', params)

    def readMetaDataOutput(self):
        mData = md.MetaData(self._getFileName(METADATA_MASK_FILE))
        NvoxelsOriginalMask = float(mData.getValue(md.MDL_COUNT, mData.firstObject()))
        NvoxelsOutputMask = float(mData.getValue(md.MDL_COUNT2, mData.firstObject()))
        nvox = int(round(
            ((NvoxelsOriginalMask - NvoxelsOutputMask) / NvoxelsOriginalMask) * 100))
        return nvox

    def getMinMax(self, imageFile):
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        min_res = round(np.amin(imgData) * 100) / 100
        max_res = round(np.amax(imgData) * 100) / 100
        return min_res, max_res

    def createOutputStep(self):
        volume = Volume()
        volume.setFileName(self._getExtraPath(OUTPUT_RESOLUTION_FILE))
        if self.useHalfVolumes:
            if not self.hasHalfVolumesFile:
                volume.setSamplingRate(self.halfMap1.get().getSamplingRate())
                volume.setOrigin(self.halfMap1.get().getOrigin(True))
                self._defineOutputs(resolution_Volume=volume)
                self._defineSourceRelation(self.halfMap1, volume)
            else:
                volume.setSamplingRate(self.associatedHalves.get().getSamplingRate())
                volume.setOrigin(self.associatedHalves.get().getOrigin(True))
                self._defineOutputs(resolution_Volume=volume)
                self._defineSourceRelation(self.associatedHalves, volume)
        else:
            volume.setSamplingRate(self.fullMap.get().getSamplingRate())
            volume.setOrigin(self.fullMap.get().getOrigin(True))
            self._defineOutputs(resolution_Volume=volume)
            self._defineSourceRelation(self.fullMap, volume)

        # Setting the min max for the summary
        imageFile = self._getExtraPath(OUTPUT_RESOLUTION_FILE_CHIMERA)
        min_, max_ = self.getMinMax(imageFile)
        self.min_res_init.set(round(min_ * 100) / 100)
        self.max_res_init.set(round(max_ * 100) / 100)
        self._store(self.min_res_init)
        self._store(self.max_res_init)

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'resolution_Volume'):
            messages.append(
                'Information about the method/article in ' + MONORES_METHOD_URL)
        return messages

    def _summary(self):
        summary = []
        summary.append("Highest resolution %.2f Å,   "
                       "Lowest resolution %.2f Å. \n" % (self.min_res_init,
                                                         self.max_res_init))
        return summary

    def _validate(self):
        errors = []

        if not self.maxRes.get():
            errors.append("You must provide a low resolution limit")

        if self.useHalfVolumes.get():
            if self.hasHalfVolumesFile.get():
                if not self.associatedHalves.get():
                    errors.append("You need to select the Associated halves")
            else:
                if not self.halfMap1.get() or not self.halfMap2.get():
                    errors.append("You need to select the volumes half")
        else:
            if not self.fullMap.get():
                errors.append("You need to select an input volume")

        return errors

    def _citations(self):
        return ['Vilas2018']
