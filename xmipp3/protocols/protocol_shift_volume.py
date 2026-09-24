# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *
# *  BCU, Centro Nacional de Biotecnologia, CSIC
# *
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

from pyworkflow.protocol.params import PointerParam, FloatParam, BooleanParam, IntParam
import pyworkflow.object as pwobj
from pwem.protocols import EMProtocol
from pwem.objects import Volume
from pwem.emlib.image import ImageHandler as ih

import numpy as np


class XmippProtShiftVolume(EMProtocol):
    """ Shifts a 3D volume spatially according to user-provided parameters.

    AI Generated

    ## Overview

    The Shift Volume protocol translates a 3D volume by a user-defined or
    automatically determined shift.

    This protocol is useful when a map needs to be recentered, moved consistently
    with a shifted particle set, or translated so that its center of mass is placed
    at the volume center. It does not perform rotational alignment or refinement.
    It only applies a spatial shift to a single input volume.

    The shift can be obtained in three ways:

    - from a previous **Shift particles** protocol;
    - from user-provided X, Y, and Z shift values;
    - from the center of mass of the input volume.

    The main output is a shifted volume. The protocol also outputs the X, Y, and Z
    shift values used.

    ## Inputs and General Workflow

    The input is a single 3D volume.

    The protocol first determines the shift values to apply. If requested, it reads
    the shift values from a previous Shift particles protocol. Otherwise, it uses
    manual X, Y, and Z values or computes a shift from the volume center of mass.

    Optionally, the protocol first windows the input volume to a new box size. It
    then applies the selected shift using the Xmipp geometry transformation
    program.

    Finally, it creates a new output volume and records the applied shift values.

    ## Input Volume

    The **Volume** parameter defines the map to be shifted.

    The input volume should be the volume that the user wants to translate. The
    original volume is not modified; the protocol writes a new shifted volume.

    The output volume keeps the same sampling rate as the input volume.

    ## Use the Same Shifts as for the Particles

    The **Use the same shifts as for the particles?** option tells the protocol to
    reuse the shift values produced by a previous **Shift particles** protocol.

    This is useful when particles and volumes must be moved consistently. For
    example, if particles were recentered on a specific domain or region, the
    corresponding volume can be shifted by the same X, Y, and Z values so that both
    remain in the same coordinate convention.

    When this option is enabled, the user must select the previous Shift particles
    protocol.

    ## Shift Particles Protocol

    The **Shift particles protocol** parameter is used when the shift values are
    taken from a previous particle-shifting run.

    The protocol reads the scalar outputs:

    - shiftX;
    - shiftY;
    - shiftZ.

    These values are then applied to the input volume.

    This option helps maintain consistency between a shifted particle set and a
    shifted reference or reconstruction volume.

    ## User-Defined Shift

    If **Use the same shifts as for the particles?** is disabled and **Use center
    of mass?** is also disabled, the user provides the shift values manually.

    The parameters are:

    - **x**;
    - **y**;
    - **z**.

    These values define the translation applied to the volume.

    Manual shifts are useful when the user already knows the desired displacement,
    for example from visual inspection, previous calculations, or external
    coordinate conventions.

    ## Use Center of Mass

    The **Use center of mass?** option computes the shift automatically from the
    input volume.

    In this mode, the protocol reads the volume, sets negative density values to
    zero, computes the center of mass of the remaining positive density, and
    chooses the shift that moves this center of mass toward the center of the box.

    This is useful when the volume is off-center and the user wants to recenter it
    based on its density distribution.

    This option assumes that the positive density corresponds to the structure of
    interest. If the map contains strong positive artifacts, disconnected density,
    or large background features, the center-of-mass shift may not represent the
    desired biological center.

    ## Box Size

    The **Use original box size for the shifted volume?** option controls whether
    the output volume keeps the same box size as the input.

    If enabled, the original box size is used.

    If disabled, the protocol first applies a windowing operation to create a
    volume with the selected **Final box size**. The shift is then applied to that
    windowed volume.

    Changing the box size can be useful when recentering a smaller region or when
    preparing a volume for a workflow that requires a specific box size.

    ## Final Box Size

    The **Final box size** parameter is used when the original box size is not
    kept.

    It defines the cubic box size of the intermediate windowed volume and therefore
    the size of the final shifted volume.

    The value should be large enough to contain the shifted density. If the box is
    too small, relevant density may be cropped.

    ## Shift Application

    The protocol applies the shift using `xmipp_transform_geometry` with the
    `--shift` option.

    The transformation is performed with the `--dont_wrap` option. This means that
    density shifted outside the box is not wrapped around to the opposite side.

    This behavior is usually appropriate for recentering, because wrapped density
    would create artificial features at the box boundaries.

    ## Output Volume

    The main output is **outputVolume**.

    This output is the shifted map written as `shift_volume.mrc` in the protocol
    working directory. It is registered as a Scipion volume and assigned the same
    sampling rate as the input volume.

    The output can be used for visualization, comparison, refinement setup, or
    other workflows that require the map to be centered or shifted consistently.

    ## Shift Outputs

    The protocol also produces three scalar outputs:

    - **shiftX**;
    - **shiftY**;
    - **shiftZ**.

    These values record the translation applied to the volume.

    They are useful for documentation, reproducibility, and for applying the same
    shift to related objects in later workflows.

    ## Interpretation of the Result

    The output volume should be interpreted as the same density map translated in
    space.

    No new structural information is created. The protocol does not refine,
    filter, mask, align, or validate the map. It only shifts the position of the
    density within the box.

    If the selected shift is correct, the volume will be better centered or better
    matched to the coordinate convention required by downstream protocols. If the
    shift is incorrect, relevant density may become off-center or cropped.

    ## Practical Recommendations

    Use the same shift as the particles when you need consistency between shifted
    particles and a corresponding volume.

    Use manual X, Y, and Z values when the desired displacement is already known.

    Use center-of-mass shifting for simple recentering, but inspect the result
    afterwards.

    Be careful with maps containing large artifacts, disconnected components, or
    negative/positive density imbalance when using center of mass.

    Keep the original box size unless there is a clear reason to change it.

    If changing the box size, make sure the new box is large enough to contain the
    shifted structure.

    Compare the shifted output with the original volume to confirm that the density
    has moved as expected.

    ## Final Perspective

    Shift Volume is a simple volume-translation utility.

    For biological users, its main value is practical coordinate control. It helps
    recenter maps, apply the same shifts used for particles, or prepare volumes for
    downstream protocols that expect a particular origin or density position.

    The protocol should be understood as a geometrical transformation step. Its
    correctness depends on choosing shift values that match the intended coordinate
    frame and biological region.
    """

    _label = 'shift volume'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVol', PointerParam, pointerClass='Volume', label="Volume", help='Volume to shift')
        form.addParam('shiftBool', BooleanParam, label='Use the same shifts as for the particles?', default='True',
                      help='Use output shifts of protocol "shift particles" which should be executed previously')
        form.addParam('inputProtocol', PointerParam, pointerClass='XmippProtShiftParticles', allowsNull=True,
                      label="Shift particles protocol", condition='shiftBool')
        form.addParam('useCM', BooleanParam, label='Use center of mass?', default='False',
                      help='Select the position where the particles will be shifted in a volume displayed in a wizard.')
        COND = 'not shiftBool and not useCM'
        form.addParam('x', FloatParam, label="x", condition=COND, allowsNull=True)
        form.addParam('y', FloatParam, label="y", condition=COND, allowsNull=True)
        form.addParam('z', FloatParam, label="z", condition=COND, allowsNull=True)
        form.addParam('boxSizeBool', BooleanParam, label='Use original box size for the shifted volume?',
                      default='True', help='Use input volume box size for the shifted volume.')
        form.addParam('boxSize', IntParam, label='Final box size', condition='not boxSizeBool',
                      help='Box size of the shifted volume.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('shiftStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def shiftStep(self):
        fnVol = self.inputVol.get().getFileName()
        if not self.boxSizeBool:
            box = self.boxSize.get()
            self.runJob('xmipp_transform_window', '-i "%s" -o "%s" --size %d %d %d' %
                        (fnVol, self._getExtraPath("resized_volume.mrc"), box, box, box))
            fnVol = self._getExtraPath("resized_volume.mrc")
        if self.shiftBool:
            shiftprot = self.inputProtocol.get()
            self.shiftx = shiftprot.shiftX.get()
            self.shifty = shiftprot.shiftY.get()
            self.shiftz = shiftprot.shiftZ.get()
        else:
            if not self.useCM:
                self.shiftx = self.x.get()
                self.shifty = self.y.get()
                self.shiftz = self.z.get()
            else:
                if fnVol.endswith('.mrc'):
                    fnVol += ':mrc'
                vol = ih().read(fnVol).getData()
                vol[vol < 0] = 0
                xs = np.linspace(-vol.shape[2] / 2, vol.shape[2] / 2, vol.shape[2])
                ys = np.linspace(-vol.shape[1] / 2, vol.shape[1] / 2, vol.shape[1])
                zs = np.linspace(-vol.shape[0] / 2, vol.shape[0] / 2, vol.shape[0])
                xs, ys, zs = np.meshgrid(xs, ys, zs, indexing='ij')
                totalMass = vol.sum()
                self.shiftx = -(xs * vol).sum() / totalMass
                self.shifty = -(ys * vol).sum() / totalMass
                self.shiftz = -(zs * vol).sum() / totalMass
        program = "xmipp_transform_geometry"
        args = '-i %s -o %s --shift %f %f %f --dont_wrap' % \
               (fnVol, self._getExtraPath("shift_volume.mrc"), self.shiftx, self.shifty, self.shiftz)
        self.runJob(program, args)

    def createOutputStep(self):
        out_vol = Volume()
        in_vol = self.inputVol.get()
        out_vol.setSamplingRate(in_vol.getSamplingRate())
        out_vol.setFileName(self._getExtraPath("shift_volume.mrc"))
        self._defineOutputs(outputVolume=out_vol)
        self._defineOutputs(shiftX=pwobj.Float(self.shiftx),
                            shiftY=pwobj.Float(self.shifty),
                            shiftZ=pwobj.Float(self.shiftz))
        self._defineSourceRelation(in_vol, out_vol)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            if self.shiftBool:
                summary.append("Volume shift as particles in %s" % self.inputProtocol.get())
            else:
                summary.append("User defined shift")
        return summary
