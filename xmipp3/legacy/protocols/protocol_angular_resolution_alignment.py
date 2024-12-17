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


import os
from pyworkflow import VERSION_2_0
from pyworkflow.utils import getExt
from pyworkflow.protocol.params import (PointerParam, BooleanParam, FloatParam,
                                        LEVEL_ADVANCED)
from pwem.protocols import ProtAnalysis3D

RADIAL_RESOLUTION_FN = 'radial_FSC_resolution.xmd'

class XmippProtResolutionAlignment(ProtAnalysis3D):
    """    
    Given two half maps the protocol estimates if the reconstruction presents angular
    alignment errors. To do that, a set of directional FSC along all possible directions
    are estimated. The result is a curve Resolution-radius. If this curve presents a slope
    then the map present angular assignment errors, but it the graph is flat (horizontal), the map
    is error free. Note that this protocol generates a plot, not a Scipion object. Its result
    can only be visualized.
    """
    _label = 'resolution alignment'
    _lastUpdateVersion = VERSION_2_0

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
                      condition='halfVolumesFile',
                      help='Select a half maps for determining its '
                           ' resolution anisotropy and resolution.')

        form.addParam('half1', PointerParam, pointerClass='Volume',
                      condition="not halfVolumesFile",
                      label="Half Map 1", important=True,
                      help='Select one map for determining the '
                           'directional FSC resolution.')

        form.addParam('half2', PointerParam, pointerClass='Volume',
                      condition="not halfVolumesFile",
                      label="Half Map 2", important=True,
                      help='Select the second map for determining the '
                           'directional FSC resolution.')

        form.addParam('mask', PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      label="Mask",
                      help='The mask determines which points are specimen'
                           ' and which are not')

        form.addParam('helicalReconstruction', BooleanParam, default=False,
                      label="Is a the protein a helix",
                      help='blablabla')

        form.addParam('limRadius', BooleanParam, default=True,
                      expertLevel=LEVEL_ADVANCED,
                      label="Limit the protein radius",
                      help='blablabla')

        form.addParam('usedirectionalfsc', BooleanParam, default=True,
                      expertLevel=LEVEL_ADVANCED,
                      label="use directional fsc",
                      help='blablabla')

        form.addParam('coneAngle', FloatParam, default=17.0,
                      expertLevel=LEVEL_ADVANCED,
                      label="Cone Angle",
                      help='Angle between the axis of the cone and the generatrix. '
                           'An angle of 17 degrees is the best angle (see publication'
                           'Vilas 2021) to measuare directional FSCs')

        form.addParam('threshold', FloatParam, expertLevel=LEVEL_ADVANCED,
                      default=0.143,
                      label="FSC Threshold",
                      help='Threshold for the fsc. By default the standard 0.143. '
                           'Other common thresholds are 0.5 and 0.3.')

        form.addParallelSection(threads=4, mpi=0)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertInputStep)
        self._insertFunctionStep(self.angularResolutionAlignmentStep)

    def convertInputStep(self):
        """ This function sets the maps properly into mrc
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

    def angularResolutionAlignmentStep(self):
        """
        This function runs the algorithm to detect misalignment
        """
        fndir = self._getExtraPath("fsc")

        os.mkdir(fndir)

        params = ' --half1 "%s"' % self.vol1Fn
        params += ' --half2 "%s"' % self.vol2Fn
        params += ' -o %s' % self._getExtraPath(RADIAL_RESOLUTION_FN)
        if self.halfVolumesFile:
            params += ' --sampling %f' % self.inputHalves.get().getSamplingRate()
        else:
            params += ' --sampling %f' % self.half1.get().getSamplingRate()

        if self.helicalReconstruction.get():
            params += ' --helix '

        if self.mask.hasValue():
            params += ' --mask "%s"' % self.maskFn
        if self.limRadius.get():
            params += ' --limit_radius '
        if self.usedirectionalfsc.get():
            params += ' --directional_resolution '
            params += ' --anglecone %f' % self.coneAngle.get()

        params += ' --threshold %s' % self.threshold.get()
        params += ' --threads %s' % self.numberOfThreads.get()

        self.runJob('xmipp_angular_resolution_alignment', params)

    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        ARTICLE_URL = 'Pending on publication'
        messages.append('Information about the method/article in ' + ARTICLE_URL)

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
        summary.append("This protocol does not produce Scipion Objects as output. Click on Analyze results to visualize the results")
        return summary

