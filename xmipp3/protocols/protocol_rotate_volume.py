# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
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

from pyworkflow.protocol.params import PointerParam, EnumParam, FloatParam
from pwem.convert.headers import setMRCSamplingRate
from pwem.objects.data import Volume
from pwem.protocols import EMProtocol


class XmippProtRotateVolume(EMProtocol):
    """ Rotates a 3D volume around the x, y, and z axes by specified angles.
    This protocol allows flexible repositioning of volumes for alignment,
    comparison, or visualization.

    AI Generated

    ## Overview

    The Rotate Volume protocol creates a rotated copy of a 3D volume.

    This protocol is useful when a map needs to be reoriented for visualization,
    comparison, presentation, or compatibility with another workflow. It does not
    perform a full volume alignment against another reference. Instead, it applies
    a user-defined geometrical transformation to a single input volume.

    The protocol supports two rotation modes:

    - aligning a selected axis with the Z axis;
    - rotating the volume by a specified number of degrees around a selected axis.

    The output is a new volume containing the rotated map. The original input
    volume is not modified.

    ## Inputs and General Workflow

    The input is a single 3D volume.

    The protocol reads the input volume file and runs the Xmipp geometry
    transformation program. Depending on the selected mode, it either aligns the
    chosen axis with the Z direction or rotates the volume around the chosen axis
    by the requested angle.

    The resulting rotated volume is written as an MRC file and registered in
    Scipion as a new output volume.

    The output volume copies the metadata from the input volume, including the
    sampling rate.

    ## Input Volume

    The **Volume** parameter defines the map to be rotated.

    This volume can be any Scipion volume object. The protocol uses the file name
    of the selected volume and creates a new rotated file.

    The input volume should already be the map that the user wants to reorient.
    The protocol does not search for an optimal alignment and does not compare the
    volume with another map.

    ## Rotation Mode

    The **Rotation mode** parameter controls the type of transformation.

    There are two options:

    **Align with Z** rotates the volume so that the selected axis is aligned with
    the Z axis.

    **rotate** applies a rotation by a user-defined number of degrees around the
    selected axis.

    The first option is useful when the user wants a principal direction or known
    axis to be oriented along Z. The second option is useful for explicit manual
    rotations.

    ## Axis

    The **Axis** parameter defines the axis used by the selected rotation mode.

    The available options are:

    - X axis;
    - Y axis;
    - Z axis.

    In **Align with Z** mode, the selected axis is the direction that will be
    aligned with the Z axis.

    In **rotate** mode, the selected axis is the axis around which the volume will
    be rotated.

    ## Degrees

    The **Degrees** parameter is used only when the rotation mode is **rotate**.

    It defines the rotation angle, in degrees, applied around the selected axis.
    For example, selecting the Z axis and 90 degrees rotates the volume by 90
    degrees around Z.

    Positive and negative angles can be used to rotate in opposite directions,
    depending on the convention of the underlying geometry transformation.

    ## Output Volume

    The main output is **outputVolume**.

    This output is the rotated version of the input volume. It is written as
    `rotated_vol.mrc` in the protocol working directory.

    The output volume copies the input volume metadata and keeps the same sampling
    rate. This means that the physical pixel size of the map is preserved after
    rotation.

    The output can be used in later protocols just like any other volume.

    ## Interpretation of the Result

    The output should be interpreted as the same density map in a different
    orientation.

    No new structural information is created. No refinement, filtering, masking,
    or validation is performed. The transformation only changes the spatial
    orientation of the volume.

    Because interpolation may be involved during rotation, repeated rotations
    should be avoided when possible. If several rotations are needed, it is better
    to apply the intended final rotation once rather than repeatedly rotating the
    same volume.

    ## Practical Recommendations

    Use this protocol when you need to reorient a volume manually.

    Use **Align with Z** when a known axis should be placed along the Z direction.

    Use **rotate** when you know the explicit angle and axis required.

    Check the output visually after rotation to confirm that the map is oriented as
    expected.

    Avoid using this protocol as a substitute for volume alignment. If the goal is
    to align two maps to each other, use a dedicated volume-alignment protocol.

    Keep the original input volume unchanged and use the rotated output for
    downstream steps that require the new orientation.

    ## Final Perspective

    Rotate Volume is a simple geometry utility for reorienting 3D maps.

    For biological users, its value is practical: it helps prepare maps for
    visualization, comparison, figure generation, or workflows that expect a
    particular orientation.

    The protocol should be understood as a manual transformation tool. It preserves
    the map content and sampling rate while producing a new volume in the selected
    orientation.
    """

    _label = 'rotate volume'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vol', PointerParam, pointerClass='Volume', label="Volume ", help='Specify a volume.')
        form.addParam('rotType', EnumParam, choices=['Align with Z', 'rotate'], display=EnumParam.DISPLAY_HLIST,
                      default=1, label='Rotation mode: ', help='Align (x,y,z) with Z axis')
        form.addParam('dirParam', EnumParam, choices=['X', 'Y', 'Z'], default=1, display=EnumParam.DISPLAY_HLIST,
                      label='Axis: ', help='Align (x,y,z) with Z axis')
        form.addParam('deg', FloatParam, label='Degrees: ', default=90, condition='rotType == 1',
                      help='degrees of rotation in selected axis')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('rotateStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def rotateStep(self):
        vol = self.vol.get()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        args = '-i %s -o %s --rotate_volume' % (fnVol, self._getExtraPath('rotated_vol.mrc'))
        rotType = self.rotType.get()
        if rotType == 0:
            args += ' alignZ'
        if rotType == 1:
            args += ' axis %d' % self.deg.get()
        dirParam = self.dirParam.get()
        if dirParam == 0:
            args += ' 1 0 0'
        if dirParam == 1:
            args += ' 0 1 0'
        if dirParam == 2:
            args += ' 0 0 1'
        program = "xmipp_transform_geometry"
        self.runJob(program, args)

    def createOutputStep(self):
        outputVol = Volume()
        inputVol = self.vol.get()
        outputVol.copyInfo(inputVol)
        outputVol.setLocation(self._getExtraPath('rotated_vol.mrc'))
        setMRCSamplingRate(outputVol.getFileName(), inputVol.getSamplingRate())
        self._defineOutputs(outputVolume=outputVol)
        self._defineSourceRelation(inputVol, outputVol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = 'Volume'
        rotType = self.rotType.get()
        if rotType == 0:
            summary += ' aligned with'
        if rotType == 1:
            summary += ' rotated %d degrees around' % self.deg.get()
        dirParam = self.dirParam.get()
        if dirParam == 0:
            summary += ' X axis'
        if dirParam == 1:
            summary += ' Y axis'
        if dirParam == 2:
            summary += ' Z axis'
        return summary
