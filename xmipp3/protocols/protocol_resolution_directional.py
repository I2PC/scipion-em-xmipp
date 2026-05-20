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

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume
from pyworkflow.utils import getExt
import pwem.emlib.metadata as md

from pwem.emlib import (MDL_XCOOR, MDL_YCOOR, MDL_ZCOOR,
                      MDL_ANGLE_ROT, MDL_ANGLE_TILT,
                      MDL_MAX, MDL_MIN, MDL_INTSCALE)


MONORES_METHOD_URL = 'https://github.com/I2PC/scipion/wiki/XmippProtMonoDir'


OUTPUT_RADIAL_AVERAGES = 'Radial_averages.xmd'
OUTPUT_RESOLUTION_FILE = 'monoresResolutionMap.mrc'
OUTPUT_RESOLUTION_FILE_CHIMERA = 'monoresResolutionChimera.mrc'
OUTPUT_MASK_FILE = 'output_Mask.vol'
FN_MEAN_VOL = 'mean_volume.vol'
METADATA_ANGLES_FILE = 'angles_md.xmd'
OUTPUT_DOA_FILE = 'local_anisotropy.vol'
OUTPUT_VARIANCE_FILE = 'resolution_variance.vol'
OUTPUT_DIRECTIONS_FILE = 'ellipsoids.xmd'
OUTPUT_MD_RADIAL_FILE = 'radial_resolution.xmd'
OUTPUT_MD_AZIMUTHAL_FILE = 'azimuthal_resolution.xmd'
OUTPUT_DESCR = 'ellipsoid.descr'
OUTPUT_ELLIP = 'ellipsoid.vol'
OUTPUT_RADIAL_FILE = 'radial_resolution.vol'
OUTPUT_AZIMUTHAL_FILE = 'azimuthal_resolution.vol'
OUTPUT_MEANRES_FILE = 'mean_resolution.vol'
OUTPUT_HIGHESTRES_FILE = 'highestResolution.vol'
OUTPUT_DOA1_FILE = 'doaMetric.vol'
OUTPUT_DOA2_FILE = 'meanResdoa.vol'
OUTPUT_LOWESTRES_FILE = 'lowestResolution.vol'
OUTPUT_THRESHOLDS_FILE = 'thresholds.xmd'
OUTPUT_MD_MINDIRECTIONAL_FILE = 'hist_prefdir.xmd'
OUTPUT_ZSCOREMAP_FILE = 'zscoreMap.vol'


class XmippProtMonoDir(ProtAnalysis3D):
    """    
    Asseses directional local resolution values of a 3D map. Enables
    identifying angular assignment errors and possible preferential directions.
    This method uses monores local resolution algorithm in a directional manner.

    AI Generated

    ## Overview

    The Directional Resolution MonoDir protocol estimates directional local
    resolution in a 3D cryo-EM map.

    Standard local-resolution methods assign one resolution value to each voxel or
    region of the map. Directional local resolution goes further by asking whether
    the resolution is the same in all directions. This is important because cryo-EM
    maps can be anisotropic: the map may be better resolved along some directions
    than others.

    Directional anisotropy can arise from preferred particle orientations,
    incomplete angular coverage, angular-assignment errors, or uneven information
    content in Fourier space. This protocol uses the MonoRes local-resolution
    framework in a directional way to estimate radial and azimuthal resolution
    components, together with degree-of-anisotropy maps.

    The main outputs are volumes describing directional resolution and anisotropy.

    ## Inputs and General Workflow

    The protocol requires:

    - an input volume;
    - a binary mask.

    The input volume and mask are converted to the format expected by Xmipp. The
    protocol first computes a standard MonoRes local-resolution map. It then runs
    the directional-resolution analysis, using the volume, mask, sampling rate,
    significance level, resolution step, and volume radius.

    The directional analysis produces several output files, including radial and
    azimuthal resolution maps, highest- and lowest-resolution maps, degree-of-
    anisotropy maps, threshold metadata, radial averages, preferred-direction
    metadata, and a z-score map.

    The Scipion outputs exposed by the protocol are the degree-of-anisotropy
    volume, the radial-resolution volume, and the azimuthal-resolution volume.

    ## Input Volume

    The **Input Volume** parameter defines the map whose directional resolution
    will be estimated.

    The volume should be a 3D reconstruction with a correct sampling rate. The
    sampling rate is required to express resolution values in angstroms and to set
    the frequency scale of the analysis.

    The map should be aligned with its corresponding mask and should represent the
    final or intermediate reconstruction to be analyzed.

    Directional resolution is especially informative when the user suspects
    anisotropy, preferred orientation, or uneven map quality in different spatial
    directions.

    ## Binary Mask

    The **Binary Mask** parameter defines the specimen region used for the
    analysis.

    The mask tells the protocol which voxels belong to the specimen and which
    belong to the background. It is used both in the MonoRes calculation and in the
    directional-resolution step.

    A good mask should include the molecular density while excluding solvent and
    irrelevant background. If the mask is too tight, it may remove real density and
    distort the local-resolution analysis. If it is too loose, background noise may
    influence the estimates.

    Although the form allows the mask parameter to be nullable, the protocol
    expects a mask file during execution. In practice, users should provide a
    binary mask.

    ## Significance

    The **Significance** parameter controls the statistical hypothesis tests used
    to determine resolution.

    The default value is 0.95. Higher values make the test more stringent, whereas
    lower values make it more permissive.

    This parameter affects the sensitivity of the local and directional resolution
    estimates. A very strict setting may report poorer resolution or fewer
    significant features, while a permissive setting may be more sensitive to weak
    signal but also more vulnerable to noise.

    Most users should start with the default value.

    ## Resolution Step

    The **Resolution Step** parameter defines the spacing between tested resolution
    values.

    For example, with a step of 0.5 Å, the protocol searches resolution values in
    increments of 0.5 Å. A smaller step gives a finer resolution scale, but may
    increase computation time. A larger step is faster but produces coarser
    resolution estimates.

    The default value provides a practical balance between precision and speed.

    ## Fast Computation

    The **Fast Computation** option enables a faster version of the directional
    resolution calculation.

    This option is recommended for large volumes, where the full directional
    analysis may be computationally expensive.

    Fast computation can reduce runtime, but users should be aware that speedups
    may come with approximations. For final reporting or borderline cases, it may
    be useful to compare fast and non-fast results on a representative map.

    ## Premasked Volumes

    The **Is the original premasked?** option should be enabled when the input
    volume has already been masked inside a spherical mask before this protocol is
    run.

    This affects how the protocol estimates noise. If the original map is
    premasked, the noise should be estimated inside the premask but outside the
    provided specimen mask.

    When this option is enabled, the user can provide a **Spherical mask radius**.
    If the radius is set to -1, the protocol uses half of the volume size as the
    radius.

    This setting is important because incorrect noise estimation can affect the
    resolution and anisotropy results.

    ## Spherical Mask Radius

    The **Spherical mask radius** parameter is used only when the original volume
    is marked as premasked.

    It defines the radius, in pixels, of the spherical premask. If the value is
    -1, the protocol uses half the volume size.

    This radius should correspond to the mask originally used to premask the map.
    If it is too small or too large, the noise region used by the protocol may not
    match the actual background available in the volume.

    ## MonoRes Step

    Before the directional analysis, the protocol runs a standard MonoRes local
    resolution calculation.

    This step estimates local resolution using the monogenic-signal approach. It
    uses the input volume, mask, sampling rate, significance level, and a resolution
    search range.

    The resulting MonoRes map is passed to the directional-resolution program as
    part of the analysis.

    This provides the baseline local-resolution information from which directional
    components are evaluated.

    ## Directional Resolution Step

    The directional-resolution step analyzes the map in different directions.

    The protocol computes several outputs, including radial and azimuthal
    resolution maps, highest- and lowest-resolution maps, degree-of-anisotropy
    metrics, thresholds, radial averages, preferred-direction information, and a
    z-score map.

    The goal is to characterize not only how well each region is resolved, but also
    whether that resolution depends strongly on direction.

    Regions with strong directional anisotropy may correspond to parts of the map
    where structural information is missing or poorly supported in some directions.

    ## Radial Resolution Volume

    The **radialVolume** output contains the radial-resolution map.

    This map describes the directional resolution component associated with radial
    directions. It can help identify regions where resolution behaves differently
    along radial directions compared with other directions.

    The values should be interpreted in angstroms, with lower values corresponding
    to better local resolution.

    ## Azimuthal Resolution Volume

    The **azimuthalVolume** output contains the azimuthal-resolution map.

    This map describes the directional resolution component associated with
    azimuthal directions.

    Comparison between radial and azimuthal maps can help reveal whether certain
    regions are preferentially resolved along one type of direction.

    As with other resolution maps, lower angstrom values indicate better estimated
    resolution.

    ## Degree-of-Anisotropy Volume

    The **outputVolume_doa** output contains a degree-of-anisotropy map.

    This map summarizes how directional the local resolution is. Values closer to
    isotropic behavior indicate that resolution is similar in different directions,
    whereas stronger anisotropy indicates that resolution varies depending on
    direction.

    This output is useful for detecting regions where map interpretation may be
    directionally biased.

    ## Histograms and Diagnostic Files

    The protocol creates histograms for several quantities, including:

    - degree of anisotropy;
    - a second anisotropy-related metric;
    - radial resolution;
    - azimuthal resolution.

    These histograms summarize the distribution of values inside the mask and can
    help assess whether anisotropy is widespread or localized.

    The protocol also writes several diagnostic files, such as threshold metadata,
    radial averages, preferred-direction metadata, z-score maps, and highest- and
    lowest-resolution volumes. These files may be useful for advanced inspection or
    method-development workflows.

    ## Interpreting Directional Resolution

    Directional local resolution should be interpreted as a map-quality diagnostic.

    A map may have a good global FSC but still show directional weakness in some
    regions. This can happen when particles have preferred orientations, when the
    angular distribution is incomplete, or when alignment is less reliable for some
    views.

    Regions with strong anisotropy should be interpreted cautiously. Apparent
    features may be better supported in one direction than another, and model
    building may be less reliable in anisotropic regions.

    Directional resolution complements, rather than replaces, global FSC and
    standard local-resolution analysis.

    ## Practical Recommendations

    Use this protocol when you suspect anisotropy, preferred orientation, or uneven
    directional information in a map.

    Provide a good binary mask. The directional analysis depends strongly on
    separating specimen from background.

    Start with the default significance and resolution step. Modify them only when
    there is a clear reason.

    Enable fast computation for large volumes or exploratory analysis. For final
    analysis, consider running the full calculation if computationally feasible.

    Use the premasked option only when the input volume was already masked before
    the protocol. Provide the correct spherical radius when known.

    Compare the radial and azimuthal resolution maps with the degree-of-anisotropy
    map. Regions where these differ strongly deserve careful inspection.

    Interpret directional-resolution results together with angular-distribution
    plots, FSC curves, local-resolution maps, and visual map quality.

    ## Final Perspective

    Directional Resolution MonoDir is a map-validation protocol designed to reveal
    anisotropy in local resolution.

    For biological users, its main value is that it can identify regions of a map
    where resolution depends on direction. This is important because such regions
    may be less reliable for model building or biological interpretation, even if
    the global resolution appears good.

    The protocol is especially useful for diagnosing preferred-orientation effects,
    angular-assignment problems, and direction-dependent loss of structural
    information.
    """
    _label = 'directional resolution MonoDir'
    _lastUpdateVersion = VERSION_2_0
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputVolumes', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True,
                      help='Select a volume for determining its local resolution.')

        form.addParam('Mask', PointerParam, pointerClass='VolumeMask',
                      label="Binary Mask", allowsNull=True,
                      help='The mask determines which points are specimen and which ones not')

        group = form.addGroup('Extra parameters')
#         group.addParam('angularsampling', FloatParam, default=15, expertLevel=LEVEL_ADVANCED,
#                       label="Angular Sampling",
#                       help='Angular sampling to cover the projection sphere')
        
        group.addParam('significance', FloatParam, default=0.95, expertLevel=LEVEL_ADVANCED,
                      label="Significance",
                      help='Relution is computed using hipothesis tests, this value determines'
                      'the significance of that test')
        
        group.addParam('resstep', FloatParam, default=0.5, expertLevel=LEVEL_ADVANCED,
                      label="Resolution Step",
                      help='The resolution will be sought in steps of this values, '
                      'with step = 0.3, then 1A, 1.3A, 1.6A,...')
        
        group.addParam('fast', BooleanParam, default=False, 
                      label="Fast Computation",
                      help='Fast computation is recommended for large volumes.')
        
        group.addParam('isPremasked', BooleanParam, default=False,
                      label="Is the original premasked?",
                      help='Sometimes the original volume is masked inside a spherical mask. In this case'
                      'please select yes')
        
        group.addParam('volumeRadius', FloatParam, default=-1,
                      label="Spherical mask radius (px)",
                      condition = 'isPremasked', 
                      help='When the original volume is originally premasked, the noise estimation ought'
                      'to be performed inside that premask, and out of the provieded mask asked in the previus'
                      'box. The radius value, determines the radius of the spherical premask. By default'
                      'radius = -1 use the half of the volume size as radius')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        
        self.micsFn = self._getPath()

        self.vol0Fn = self.inputVolumes.get().getFileName()
        self.maskFn = self.Mask.get().getFileName()

        # Convert input into xmipp Metadata format
        convertId = self._insertFunctionStep('convertInputStep')

        self._insertFunctionStep('MonoResStep')      
          
        self._insertFunctionStep('directionalResolutionStep',
                                      prerequisites=[convertId])

        self._insertFunctionStep('createOutputStep')

#         self._insertFunctionStep("createEllipsoid")
        self._insertFunctionStep("createHistrogramStep")
        
        
    def convertInputStep(self):
        """ Read the input volume.
        """
        extVol0 = getExt(self.vol0Fn)
        if (extVol0 == '.mrc') or (extVol0 == '.map'):
            self.vol0Fn = self.vol0Fn + ':mrc'

        extMask = getExt(self.maskFn)
        if ((extMask == '.mrc') or (extMask == '.map')):
            self.maskFn = self.maskFn + ':mrc'


    def directionalResolutionStep(self):

        if self.isPremasked:
            if self.volumeRadius == -1:
                xdim, _ydim, _zdim = self.inputVolumes.get().getDim()
                xdim = xdim*0.5
            else:
                xdim = self.volumeRadius.get()
        else:
            xdim, _ydim, _zdim = self.inputVolumes.get().getDim()
            xdim = xdim*0.5

        # Number of frequencies
        #Nfreqs = xdim
                
        params = ' --vol %s' % self.vol0Fn
        params += ' --mask %s' % self.maskFn
        params += ' -o %s' % self._getExtraPath(OUTPUT_RESOLUTION_FILE)
        params += ' --sampling_rate %f' % self.inputVolumes.get().getSamplingRate()
        params += ' --volumeRadius %f' % xdim
        params += ' --significance %f' % self.significance.get()
        params += ' --resStep %f' % self.resstep.get()
        params += ' --radialRes %s' % self._getExtraPath(OUTPUT_RADIAL_FILE)
        params += ' --azimuthalRes %s' % self._getExtraPath(OUTPUT_AZIMUTHAL_FILE)
        params += ' --highestResolutionVol %s' % self._getExtraPath(OUTPUT_HIGHESTRES_FILE)
        params += ' --lowestResolutionVol %s' % self._getExtraPath(OUTPUT_LOWESTRES_FILE)
        params += ' --doa1 %s' % self._getExtraPath(OUTPUT_DOA1_FILE)
        params += ' --doa2 %s' % self._getExtraPath(OUTPUT_DOA2_FILE)
        params += ' --radialAzimuthalThresholds %s' % self._getExtraPath(OUTPUT_THRESHOLDS_FILE)
        params += ' --radialAvG %s' % self._getExtraPath(OUTPUT_RADIAL_AVERAGES)
        params += ' --prefMin %s' % self._getExtraPath(OUTPUT_MD_MINDIRECTIONAL_FILE)
        params += ' --zScoremap %s' % self._getExtraPath(OUTPUT_ZSCOREMAP_FILE)
        params += ' --threads %i' % self.numberOfThreads.get()
        params += ' --monores %s' % self._getExtraPath(OUTPUT_RESOLUTION_FILE)
        if (self.fast.get() is True):
            params += ' --fast'

        self.runJob('xmipp_resolution_directional', params)
        
        #TODO: Take a metadata and set maxRes minRes, idem with azimuthal and tangencial


    def MonoResStep(self):

        params = ' --vol %s' % self.vol0Fn
        params += ' --mask %s' % self.maskFn
        params += ' -o %s' % self._getExtraPath()
        params += ' --sampling_rate %f' % self.inputVolumes.get().getSamplingRate()
        params += ' --step %f' % 0.25
        params += ' --minRes %f' % (2.0*self.inputVolumes.get().getSamplingRate())
        params += ' --maxRes %f' % 18.0
        params += ' --significance %f' % self.significance.get()
        params += ' --threads %i' % self.numberOfThreads.get()  
        self.runJob('xmipp_resolution_monogenic_signal', params)

    def createEllipsoid(self):

        xdim, ydim, zdim = self.inputVolumes.get().getDim()
        f = open(self._getExtraPath(OUTPUT_DESCR),'w') 
        str_ = '%i %i %i 0\n' %(xdim, ydim, zdim)     
        f.write(str_)
        
        mtd = md.MetaData()
        mtd.read(self._getExtraPath(OUTPUT_DIRECTIONS_FILE))
        for objId in mtd:
            xcoor = mtd.getValue(MDL_XCOOR, objId)
            ycoor = mtd.getValue(MDL_YCOOR, objId)
            zcoor = mtd.getValue(MDL_ZCOOR, objId)
            rot = mtd.getValue(MDL_ANGLE_ROT, objId)
            tilt = mtd.getValue(MDL_ANGLE_TILT, objId)
            len_max = mtd.getValue(MDL_MAX, objId)
            len_min = mtd.getValue(MDL_MIN, objId)
            doa = mtd.getValue(MDL_INTSCALE, objId)
            str_ = 'ell = %f %i %i %i %f %f %f %f %f 0\n' %(doa, xcoor, ycoor, zcoor, 
                                            len_max, len_min, len_min, rot, tilt)
            f.write(str_)
        
        f.close()

        params = ' -i %s' % self._getExtraPath(OUTPUT_DESCR)
        params += ' -o %s' % self._getExtraPath(OUTPUT_ELLIP)
        
        self.runJob('xmipp_phantom_create', params)


    def createHistrogram(self, fnVol, fnOut, doa):

        params = ' -i %s' % fnVol
        params += ' --mask binary_file %s' % self.maskFn
        params += ' --steps %f' % 30
        params += ' -o %s' % fnOut
        if doa is True:
            params += ' --range %f %f' % (0, 1)#(self.minRes.get(), self.maxRes.get())
        else:
            params += ' --range %f %f' % (0, 30)
        self.runJob('xmipp_image_histogram', params)
        
    
    def createHistrogramStep(self):
        self.createHistrogram(self._getExtraPath(OUTPUT_DOA1_FILE), self._getExtraPath('hist_DoA.xmd'), True)
        self.createHistrogram(self._getExtraPath(OUTPUT_DOA2_FILE), self._getExtraPath('hist_DoA2.xmd'), False)
        self.createHistrogram(self._getExtraPath(OUTPUT_RADIAL_FILE), self._getExtraPath('hist_radial.xmd'), False)
        self.createHistrogram(self._getExtraPath(OUTPUT_AZIMUTHAL_FILE), self._getExtraPath('hist_azimuthal.xmd'), False)
        
    def createOutputStep(self):
        
        volume=Volume()
        volume.setFileName(self._getExtraPath(OUTPUT_DOA1_FILE))
        volume.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(outputVolume_doa=volume)
        self._defineSourceRelation(self.inputVolumes, volume)
        
        volume.setFileName(self._getExtraPath(OUTPUT_AZIMUTHAL_FILE))
        volume.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(azimuthalVolume=volume)
        self._defineSourceRelation(self.inputVolumes, volume)

        volume.setFileName(self._getExtraPath(OUTPUT_RADIAL_FILE))
        volume.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(radialVolume=volume)
        self._defineSourceRelation(self.inputVolumes, volume)        

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'doaVol'):
            messages.append(
                'Information about the method/article in ' + MONORES_METHOD_URL)
        return messages
    
    def _summary(self):
        summary = []

        return summary

    def _citations(self):
        return ['Not yet']
