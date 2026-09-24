# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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

from os.path import exists
from pyworkflow.protocol.params import MultiPointerParam
from pwem.objects import Volume, Transform
from pwem.protocols import ProtInitialVolume

CITE = 'Fernandez-Gimenez2021'

class XmippProtVolConsensus(ProtInitialVolume):
    """ This protocol performs a fusion of all the input volumes, which should
    be preprocessed with protocol 'volume substraction' saving volume 2, in
    order to be as similar as possible before the fusion. The output of
    this protocol is the consensus volume and another volume which indicates
    the maximun difference between input volumes in each voxel.

    AI Generated

    ## Overview

    The Volume Consensus protocol fuses several input volumes into a single
    consensus volume.

    When several maps represent the same structure or related reconstructions, it
    can be useful to combine them into one consensus map. This protocol takes a set
    of input volumes and computes a consensus volume from them. It also produces a
    second volume that represents the maximum difference between the input volumes
    at each voxel.

    The consensus volume summarizes the common density shared by the input maps.
    The difference volume helps identify regions where the input volumes disagree.

    This protocol is intended for volumes that have already been made comparable.
    The code help specifically indicates that the input volumes should preferably
    be preprocessed with the **Volume subtraction** workflow, saving the adjusted
    second volume, so that the volumes are as similar as possible before fusion.

    ## Inputs and General Workflow

    The protocol requires a list of input volumes.

    It writes the file names of all selected volumes to an input list and then runs
    the Xmipp consensus-volume program. The program creates two output maps:

    - a consensus volume;
    - a difference volume.

    The protocol registers both maps as Scipion output volumes. It also creates a
    ChimeraX command script that opens both volumes and colors the consensus volume
    according to the difference map.

    ## Input Volumes

    The **Volumes** parameter defines the set of volumes to be fused.

    All input volumes should represent the same structure or comparable structural
    states. They should be aligned, have the same box size, have the same voxel
    size, and share a consistent origin.

    The protocol validates that all input volumes have the same pixel size. If the
    pixel sizes differ, it reports an error.

    If the box size, voxel size, or origin are not compatible, the consensus volume
    may not be generated correctly. In that case, the protocol raises an error
    asking the user to check that the input volumes have equal box size, voxel
    size, and origin.

    ## Preparing Volumes for Consensus

    The input volumes should be as comparable as possible before running this
    protocol.

    In practice, this means that they should usually be:

    - aligned to the same coordinate frame;
    - sampled at the same voxel size;
    - placed in the same box size;
    - normalized or adjusted to comparable amplitudes;
    - filtered or processed consistently.

    The protocol documentation in the code notes that the volumes are expected to
    have been preprocessed with **Volume subtraction**, saving Volume 2, so that
    they are as similar as possible before the consensus step.

    This preparation is important because the consensus operation assumes that
    differences between maps are meaningful, not caused by mismatched scale,
    origin, orientation, or box size.

    ## Consensus Volume

    The main output is **outputVolume**.

    This volume is written as:

    `consensus_volume.mrc`

    It represents the fused consensus map obtained from the selected input
    volumes. The output volume is assigned the sampling rate of the first input
    volume.

    The consensus volume can be used for visualization, comparison, or downstream
    processing when the user wants a single representative map derived from
    several related input maps.

    ## Difference Volume

    The second output is **outputVolumeDiff**.

    This volume is written as:

    `consensus_volume_diff.mrc`

    It represents the maximum difference between the input volumes at each voxel.
    This map is useful for identifying regions where the input volumes disagree.

    High values in the difference volume indicate regions with stronger variation
    among the input maps. Low values indicate regions where the input volumes are
    more similar.

    ## ChimeraX Visualization Script

    The protocol creates a ChimeraX command script named:

    `result_fusion_chimera.cxc`

    This script opens the consensus volume and the difference volume. It hides the
    difference volume and colors the consensus volume by sampling values from the
    difference map using a rainbow palette.

    This visualization helps the user inspect where the consensus map is stable
    and where the input volumes differ.

    ## Output Sampling Rate

    Both output volumes use the sampling rate of the first input volume.

    This is appropriate when all input volumes have the same sampling rate, which
    is required by the validation step.

    The output should therefore be interpreted in the same voxel-size units as the
    input maps.

    ## Interpreting the Consensus Result

    The consensus volume should be interpreted as a fused representation of the
    input maps.

    Regions that are consistent across the inputs are expected to appear more
    stable in the consensus. Regions where the inputs differ may still appear in
    the consensus, but their uncertainty or variability should be assessed using
    the difference volume.

    The difference volume is not a local-resolution map. It is a voxel-wise
    disagreement map derived from the input volumes. It should be interpreted as a
    measure of variability between the maps being fused.

    ## Practical Recommendations

    Use this protocol only after the input volumes have been aligned and brought to
    the same box size, voxel size, and origin.

    Check that the volumes are comparable in filtering, normalization, and
    amplitude scale before fusion.

    Use the difference volume to inspect where the input maps disagree.

    Open the generated ChimeraX script to visualize the consensus map colored by
    local disagreement.

    Do not interpret the consensus volume alone. Always compare it with the
    individual input volumes and with the difference map.

    If the protocol does not generate an output, check that all input volumes have
    equal box size, voxel size, and origin.

    ## Final Perspective

    Volume Consensus is a map-fusion protocol.

    For biological users, its main value is that it creates a representative
    consensus map from several comparable input volumes and provides a difference
    map showing where those volumes disagree.

    The protocol is most useful when the input maps have already been carefully
    prepared and adjusted. The consensus map summarizes common density, while the
    difference map highlights regions that may correspond to structural
    variability, processing differences, or remaining inconsistencies among the
    inputs.
    """

    _label = 'volume consensus'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vols', MultiPointerParam, pointerClass='Volume', label="Volumes",
                      help='Select the volumes for the consensus.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('fusionStep')
        self._insertFunctionStep('createOutputStep')
        self._insertFunctionStep("createChimeraScript")

    # --------------------------- STEPS functions ---------------------------------------------------
    def fusionStep(self):
        inputVols = self._getExtraPath("input_volumes.txt")
        fhInputVols = open(inputVols, 'w')
        for i, vol in enumerate(self.vols):
            fileName = vol.get().getFileName()
            if fileName.endswith(':mrc'):
                fileName = fileName[:-4]
            fhInputVols.write(fileName + '\n')
        fhInputVols.close()
        outVolFn = self._getExtraPath("consensus_volume.mrc")
        args = " -i %s -o %s" % (inputVols, outVolFn)
        self.runJob("xmipp_volume_consensus", args)

    def createOutputStep(self):
        outVol = Volume()
        outVol.setSamplingRate(self.vols[0].get().getSamplingRate())
        outVol.setFileName(self._getExtraPath("consensus_volume.mrc"))
        if not exists(self._getExtraPath("consensus_volume.mrc")):
            raise NoOutputGenerated("Consensus volume NOT generated, please check input volumes to ensure they have "
                                    "equal box size, voxel size and origin.")
        else:
            outVol2 = Volume()
            outVol2.setSamplingRate(self.vols[0].get().getSamplingRate())
            outVol2.setFileName(self._getExtraPath("consensus_volume_diff.mrc"))
            self._defineOutputs(outputVolume=outVol)
            self._defineOutputs(outputVolumeDiff=outVol2)

    def createChimeraScript(self):
        fnRoot = "extra/"
        scriptFile = self._getPath('result') + '_fusion_chimera.cxc'
        fhCmd = open(scriptFile, 'w')
        fhCmd.write("open %s\n" % (fnRoot+"consensus_volume.mrc"))
        fhCmd.write("open %s\n" % (fnRoot+"consensus_volume_diff.mrc"))
        fhCmd.write("vol #2 hide\n")
        fhCmd.write("color sample #1 map #2 palette rainbow\n")
        fhCmd.close()

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            for i, vol in enumerate(self.vols):
                summary.append("Volume %d: %s" % (i+1, vol.get().getFileName()))
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            methods.append("We compute a consensus volume from %d input volumes at %f A/px" %
                           (self.vols.getSize(), self.vols[0].get().getSamplingRate()))
        return methods

    def _validate(self):
        errors = []
        voxel_size = []
        for i, vol in enumerate(self.vols):
            voxel_size.append(round(vol.get().getSamplingRate(), 2))
        result = all(element == voxel_size[0] for element in voxel_size)
        if not result:
            errors.append('Pixel size should be the same for all input volumes.')
        return errors

    def _citations(self):
        return ['Fernandez-Gimenez2021']


class NoOutputGenerated(Exception):
    """No output generation error"""
    pass
