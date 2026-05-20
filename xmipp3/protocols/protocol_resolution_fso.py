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
from pyworkflow.object import Float
from pyworkflow.utils import getExt
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)

from pyworkflow import BETA, UPDATED, NEW, PROD
from pwem.objects import Volume
from pwem.protocols import ProtAnalysis3D
from pyworkflow import BETA, UPDATED, NEW, PROD

OUTPUT_3DFSC = '3dFSC.mrc'
OUTPUT_DIRECTIONAL_FILTER = 'filteredMap.mrc'
OUTPUT_DIRECTIONAL_DISTRIBUTION = 'Resolution_Distribution.xmd'


class XmippProtFSO(ProtAnalysis3D):
    """    
    Given two half maps the protocol estimates Fourier Shell Occupancy to
    determine the global anisotropy of the map.
    See more information here:
    https://github.com/I2PC/xmipp/wiki/FSO---Fourier-Shell-Occupancy

    AI Generated

    ## Overview

    The Resolution FSO protocol estimates the directional resolution anisotropy of
    a cryo-EM reconstruction from two half maps.

    FSO stands for Fourier Shell Occupancy. The method evaluates how resolution is
    distributed across directions in Fourier space. This is useful because a map
    may have a reasonable global FSC resolution while still being anisotropic: the
    data may support high resolution in some directions but poorer resolution in
    others.

    Directional anisotropy is often associated with preferred particle
    orientations, incomplete angular coverage, missing views, or direction-dependent
    weakness in the reconstruction. By analyzing the two half maps directionally,
    this protocol helps the user assess whether the map resolution is globally
    uniform or directionally biased.

    The protocol can also estimate a 3D FSC map and a directionally filtered map
    when requested.

    ## Inputs and General Workflow

    The protocol requires two half maps.

    The half maps can be provided in two ways:

    - as half maps associated with an input volume;
    - as two explicitly selected half-map volumes.

    Optionally, the user can provide a mask to restrict the analysis to the
    specimen region.

    The protocol converts the input file names to the format expected by Xmipp,
    then runs the Xmipp FSO resolution program. The calculation uses the half maps,
    sampling rate, cone angle, optional mask, FSC threshold, and number of threads.

    The results are written as files in the protocol output directory. The protocol
    does not currently create a dedicated Scipion output object for the FSO result,
    because Scipion does not have a specific object type for this kind of
    directional FSC/FSO plot.

    ## Half Maps Stored with the Input Volume

    The **Are the half volumes stored with the input volume?** option controls how
    the half maps are provided.

    If this option is enabled, the user selects **Input Half Maps**, which is a
    volume object that has associated half maps. The protocol reads the two
    half-map file names from that volume.

    This is the usual option when a reconstruction protocol produced a volume with
    half maps stored as associated metadata.

    In this mode, the sampling rate is taken from the selected input volume.

    ## Explicit Half Maps

    If **Are the half volumes stored with the input volume?** is disabled, the user
    must provide the two half maps explicitly:

    - **Half Map 1**;
    - **Half Map 2**.

    This option is useful when the half maps exist as independent volume objects
    rather than being attached to a single reconstructed volume.

    The two half maps should come from independent halves of the same particle
    dataset. They should have the same box size, sampling rate, orientation,
    origin, and preprocessing state.

    In this mode, the sampling rate is taken from the first half map.

    ## Mask

    The **Mask** parameter is optional.

    If provided, the mask restricts the FSO calculation to the specimen region.
    This can reduce the influence of solvent and background noise.

    The mask should include the molecular density while avoiding unnecessary
    background. Smooth masks are generally preferable for Fourier-space
    calculations, because sharp masks can introduce artifacts.

    The same mask is applied consistently to the directional resolution analysis.

    ## Cone Angle

    The **Cone Angle** parameter controls the angular aperture used to compute
    directional FSC curves.

    The cone angle is the angle between the axis of the cone and its generatrix.
    For each direction in Fourier space, the method considers information inside a
    cone around that direction. The directional FSC is then estimated from that
    restricted region.

    The default value is 17 degrees. This value is recommended by the method
    implementation as a suitable cone angle for measuring directional FSCs.

    A smaller cone angle gives a more direction-specific estimate but may use fewer
    Fourier samples and therefore become noisier. A larger cone angle gives a
    smoother estimate but may average over more directions and reduce sensitivity
    to anisotropy.

    ## Estimate 3DFSC

    The **Estimate 3DFSC** option controls whether the protocol estimates a 3D FSC
    map and applies directional filtering.

    The 3D FSC is a function defined in Fourier space. The profile of this function
    along a given direction corresponds to the directional FSC.

    When this option is enabled, the protocol requests the additional 3D FSC
    filtering output from the Xmipp FSO program. This can produce files such as a
    3D FSC map and a directionally filtered map.

    These outputs are useful for advanced analysis of anisotropy and for inspecting
    how directional resolution affects the reconstructed map.

    ## FSC Threshold

    The **FSC Threshold** parameter defines the FSC value used to determine the
    directional resolution limit.

    The default is 0.143, a standard threshold commonly used in cryo-EM FSC
    analysis. Other possible thresholds include 0.5 or 0.3, depending on the
    validation convention used by the user.

    Changing the threshold changes the reported directional resolution values. A
    higher threshold is stricter and usually reports lower resolution. A lower
    threshold is more permissive.

    Users should report the threshold used when presenting FSO or directional
    resolution results.

    ## Output Files

    The protocol writes its results to the protocol output directory.

    Depending on the selected options, these files may include:

    - FSO or directional FSC information;
    - resolution distribution metadata;
    - a 3D FSC map;
    - a directionally filtered map;
    - intermediate FSC files.

    The code defines file names such as `3dFSC.mrc`, `filteredMap.mrc`, and
    `Resolution_Distribution.xmd`.

    At present, the protocol does not expose these results as a specific Scipion
    output object. Users should inspect the generated files and plots through the
    protocol results or file viewers.

    ## Interpreting FSO Results

    FSO results describe how uniformly Fourier information supports the
    reconstruction in different directions.

    If the directional resolution distribution is fairly uniform, the map is more
    isotropic. If some directions show substantially worse resolution, the map is
    anisotropic.

    Strong anisotropy may indicate preferred orientation, missing angular
    information, or limited reconstruction support in specific directions. This can
    affect map interpretation, model building, and confidence in structural
    features.

    FSO should be interpreted together with angular-distribution plots, half-map
    FSC, local resolution, map appearance, and biological context.

    ## Practical Recommendations

    Use this protocol with two independent half maps from the same reconstruction.

    Use the associated-half-map option when the half maps are already attached to a
    Scipion volume. Use explicit half maps when they are stored as separate volume
    objects.

    Provide a mask when possible, especially for maps with large solvent regions.

    Start with the default cone angle of 17 degrees and FSC threshold of 0.143.

    Enable 3DFSC estimation when you want additional directional filtering or
    advanced anisotropy analysis.

    Inspect the directional resolution distribution rather than relying only on the
    global FSC resolution. A map can have a good global resolution but still be
    directionally limited.

    Be cautious when interpreting regions affected by strong anisotropy, because
    features may be better supported in some directions than others.

    ## Final Perspective

    Resolution FSO is a directional map-validation protocol.

    For biological users, its main value is that it detects and summarizes
    resolution anisotropy using two independently reconstructed half maps. This can
    reveal preferred-orientation effects or directional weaknesses that are not
    fully captured by a single global FSC number.

    The protocol is especially useful when assessing final reconstructions,
    diagnosing anisotropy, or deciding whether a map requires further data
    collection, better angular coverage, or cautious interpretation in
    directionally weak regions.
    """
    _label = 'resolution fso'
    _lastUpdateVersion = VERSION_2_0
    _devStatus = PROD

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('halfVolumesFile', BooleanParam, default=False,
                      label="Are the half volumes stored with the input volume?",
                      help='Usually, the half volumes are stored as properties of '
                      'the input volume. If this is not the case, set this to '
                      'False and specify the two halves you want to use.')

        form.addParam('inputHalves', PointerParam, pointerClass='Volume',
                      label="Input Half Maps",
                      condition = 'halfVolumesFile',
                      help='Select a half maps for determining its '
                      ' resolution anisotropy and resolution.')

        form.addParam('half1', PointerParam, pointerClass='Volume',
                      condition = "not halfVolumesFile",
                      label="Half Map 1", important=True,
                      help='Select one map for determining the directional FSC resolution.')

        form.addParam('half2', PointerParam, pointerClass='Volume',
                      condition = "not halfVolumesFile",
                      label="Half Map 2", important=True,
                      help='Select the second map for determining the '
                      'directional FSC resolution.')

        form.addParam('mask', PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Mask",
                      help='The mask determines which points are specimen'
                      ' and which are not')

        form.addParam('coneAngle', FloatParam, default=17.0,
                      expertLevel=LEVEL_ADVANCED,
                      label="Cone Angle",
                      help='Angle between the axis of the cone and the generatrix. '
                           'An angle of 17 degrees is the best angle (see Nat Methods'
                           'JL Vilas 2023) to measuare the directional FSCs')

        form.addParam('estimate3DFSC', BooleanParam, default=True,
                      label="Estimate 3DFSC ",
                      help='Set to estimate the 3DFSCD map. This is a 3D function that depends of the resolution.'
                           'The profile of the 3DFSC along a given direction is the directiontal FSC')

        form.addParam('threshold', FloatParam, expertLevel=LEVEL_ADVANCED,
                      default=0.143,
                      label="FSC Threshold",
                      help='Threshold for the fsc. By default the standard 0.143. '
                           'Other common thresholds are 0.5 and 0.3.')

        form.addParallelSection(threads = 4, mpi = 0)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _createFilenameTemplates(self):
        """ Centralize how files are called """
        myDict = {OUTPUT_3DFSC: self._getExtraPath("3dFSC.mrc"),
                  OUTPUT_DIRECTIONAL_FILTER: self._getExtraPath("filteredMap.mrc"),
                  }
        self._updateFilenamesDict(myDict)


    def _insertAllSteps(self):
        self._createFilenameTemplates()
        # Convert input into xmipp Metadata format
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('FSOestimationStep')
        self._insertFunctionStep('createOutputStep')

    def mrc_convert(self, fileName, outputFileName):
        """Check if the extension is .mrc, if not then uses xmipp to convert it
        """
        ext = getExt(fileName)
        if (ext != '.mrc') and (ext != '.map'):
            params = ' -i "%s"' % fileName
            params += ' -o "%s"' % outputFileName
            self.runJob('xmipp_image_convert', params)
            return outputFileName+':mrc'
        else:
            return fileName+':mrc'

    def convertInputStep(self):
        """ Read the input volume.
        """

        if self.halfVolumesFile:
            self.vol1Fn, self.vol2Fn = self.inputHalves.get().getHalfMaps().split(',')
        else:
            self.vol1Fn = self.half1.get().getFileName()
            self.vol2Fn = self.half2.get().getFileName()

        extVol1 = getExt(self.vol1Fn)
        extVol2 = getExt(self.vol2Fn)

        if (extVol1 == '.mrc') or (extVol1 == '.map'):
            self.vol1Fn = self.vol1Fn + ':mrc'
        if (extVol2 == '.mrc') or (extVol2 == '.map'):
            self.vol2Fn = self.vol2Fn + ':mrc'

        if self.mask.hasValue():
            self.maskFn = self.mask.get().getFileName()
            extMask = getExt(self.maskFn)
            if (extMask == '.mrc') or (extMask == '.map'):
                self.maskFn = self.maskFn + ':mrc'


    def FSOestimationStep(self):
        import os
        fndir = self._getExtraPath("fsc")

        os.mkdir(fndir)

        params = ' --half1 "%s"' % self.vol1Fn
        params += ' --half2 "%s"' % self.vol2Fn
        params += ' -o %s' % self._getExtraPath()
        if self.halfVolumesFile:
            params += ' --sampling %f' % self.inputHalves.get().getSamplingRate()
        else:
            params += ' --sampling %f' % self.half1.get().getSamplingRate()

        if self.mask.hasValue():
            params += ' --mask "%s"' % self.maskFn

        params += ' --anglecone %f' % self.coneAngle.get()

        if self.estimate3DFSC.get():
            params += ' --threedfsc_filter'

        params += ' --threshold %s' % self.threshold.get()
        params += ' --threads %s' % self.numberOfThreads.get()

        self.runJob('xmipp_resolution_fso', params)


    def createOutputStep(self):
        """
        There is no output for this method. The result is a plot similar to the FSC, but Scipion has no object for it
        This method is left with a pass to leave flexible enought in a possible future
        """
        pass

    # --------------------------- INFO functions ------------------------------
    def _methods(self):
        messages = []
        messages.append('Information about the method/article in ')
        return messages

    def _validate(self):
        errors = []

        if self.halfVolumesFile.get():
            if not self.inputHalves.get():
                errors.append("You need to select the Associated halves")
        else:
            if not self.half1.get():
                errors.append("You need to select the half1")
            if not self.half2.get():
                errors.append("You need to select the half2")

        return errors

    def _summary(self):
        summary = []
        summary.append(" ")
        return summary

    def _citations(self):
        return ['Vilas2023']
