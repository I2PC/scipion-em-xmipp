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

import numpy as np

from pwem.emlib.image import ImageHandler as ih
from pwem.objects import Volume
from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, FloatParam
import pyworkflow.object as pwobj
from pwem.protocols import EMProtocol
from xmipp3.convert import writeSetOfParticles
from xmipp3.utils import applyTransform, readImage, writeImageFromArray


class XmippProtShiftParticles(EMProtocol):
    """ This protocol shifts particles to center them into a point selected in
    a volume. To do so, it generates new shifted images and modify the
    transformation matrix according to the shift performed.

    AI Generated

    ## Overview

    The Shift Particles protocol recenters a set of particles according to a
    selected 3D position.

    In single-particle workflows, particle images may need to be shifted so that a
    particular structural point becomes the new center. This can be useful when the
    particles were extracted around a suboptimal center, when the user wants to
    focus on a specific region of a larger complex, or when a volume-derived point
    should define the new particle origin.

    This protocol shifts particles using either:

    - a user-selected point in a volume;
    - the center of mass of a 3D mask.

    The protocol can either apply the shift directly to the particle images or
    store the shift in the particle transformation matrices for later application.

    The main output is a new particle set with updated alignment information. The
    protocol also outputs the X, Y, and Z shift values used.

    ## Inputs and General Workflow

    The input is a set of particles with projection-alignment transformation
    matrices.

    The protocol first converts the particle set to Xmipp metadata format. It then
    determines the 3D point that should become the new center. This point can come
    from the coordinates selected by the user in a volume, or from the center of
    mass of an input 3D mask.

    The protocol then calls the Xmipp geometry transformation program to shift the
    particles. Depending on the selected options, the shift is applied to the
    particle images or stored in the metadata. If requested, the shifted particles
    are also cropped or windowed to a new box size.

    Finally, the protocol creates an output particle set and stores the shift
    values as scalar outputs.

    ## Input Particles

    The **Particles** parameter defines the particle set to be shifted.

    The particles must contain transformation matrices. These transformations are
    needed because the protocol updates the particle alignment information
    according to the selected 3D shift.

    If the input particles do not have transformation matrices, the protocol
    reports a validation error.

    This protocol is therefore intended for particles that have already been
    assigned projection-alignment information, for example after refinement,
    classification, or alignment.

    ## Select Position in Volume

    The **Select position in volume?** option controls how the new center is
    defined.

    If this option is enabled, the user provides a volume and selects a point in
    that volume using the wizard. The selected point defines the 3D position to
    which the particles should be shifted.

    This is useful when the desired center is a specific structural feature, such
    as a domain, binding site, symmetry center, or region of interest.

    If this option is disabled, the protocol uses a 3D mask and computes its center
    of mass.

    ## Volume for Selecting the New Center

    The **Volume** parameter is used when the user selects the new center manually.

    The volume is displayed in the wizard, where the user can choose the point that
    should become the center of the shifted particles.

    The selected coordinates are stored in the **x**, **y**, and **z** parameters.

    The volume should be in the same coordinate frame as the particles. If the
    volume and particles are not consistent, the selected point may not correspond
    to the intended structural location.

    ## X, Y, and Z Coordinates

    The **x**, **y**, and **z** parameters store the selected shift target.

    These values define the 3D point used by the Xmipp geometry transformation.

    When the point is selected through the wizard, these parameters are filled with
    the selected coordinates. The protocol then uses them to shift the particles so
    that this point becomes the new center.

    Users should verify that the selected point is meaningful in relation to the
    particle alignment and the reference volume.

    ## Volume Mask and Center of Mass

    When **Select position in volume?** is disabled, the **Volume mask** parameter
    is used.

    The protocol reads the mask and computes its center of mass. This center of
    mass becomes the target point for shifting the particles.

    This option is useful when the desired center corresponds to a segmented region
    of the volume, such as a domain or subunit represented by a mask.

    The mask should be defined in the same coordinate frame as the particles and
    should isolate the region whose center should be used. If the mask contains
    multiple disconnected regions or unwanted density, the center of mass may not
    represent the intended point.

    ## Apply Shift to Particles

    The **Apply shift to particles?** option controls whether the particle images
    are actually shifted.

    If set to **Yes**, the protocol applies the shift to the particle images. The
    output images are newly transformed, and the metadata are updated accordingly.

    If set to **No**, the particle image files are not transformed. Instead, the
    shift is stored in the transformation matrix metadata. This is faster because
    the images do not need to be rewritten. The shift can later be applied using a
    2D alignment-application protocol or by re-extracting particles.

    Use **Yes** when you need physically shifted particle images immediately. Use
    **No** when you only want to update the alignment metadata and defer image
    resampling to a later step.

    ## Box Size

    The **Use original box size for the shifted particles?** option controls
    whether the output particles keep the same box size as the input particles.

    If enabled, the shifted particles keep the original box size.

    If disabled, the user provides a **Final box size**, and the protocol windows
    or crops the shifted particles to that size.

    Changing the box size can be useful when recentering on a smaller region of a
    larger particle, but it should be done carefully. A box that is too small may
    cut off relevant density. A larger or unchanged box may be safer when the
    appropriate final box size is uncertain.

    ## Final Box Size

    The **Final box size** parameter is used when the original box size is not
    kept.

    It defines the box size of the output shifted particles.

    The value should be large enough to contain the recentered structure or region
    of interest, including enough surrounding background for downstream processing.

    After shifting, the protocol uses a windowing operation to create particles
    with the requested final size.

    ## Inverse Transformation

    The **Inverse** option controls whether the inverse transformation matrix is
    used.

    This is an advanced option related to the convention used for applying the
    particle transformation. The default enables inverse transformation.

    Most users should keep the default unless they know that their workflow
    requires the opposite transformation convention.

    If the particles appear shifted in the wrong direction, this option may be one
    of the settings to check.

    ## Interpolation

    The **Interpolation** parameter defines how pixel values are interpolated when
    the shift is applied to the particle images.

    There are two options:

    **Linear** interpolation is faster and usually sufficient for many workflows.

    **Spline** interpolation may provide smoother interpolation but can be more
    computationally expensive.

    This option matters only when the shift is applied to the particle images. If
    the shift is only stored in metadata, image interpolation is not performed at
    this stage.

    ## Output Particles

    The main output is **outputParticles**.

    This output contains the shifted particle set. It copies the information from
    the input particle set and reads the updated Xmipp metadata with projection
    alignment information.

    If the shift was applied, the output points to the transformed particle images.
    If the shift was not applied, the output particles keep image data unchanged
    but carry updated transformation matrices.

    The output particle set can be used in downstream protocols such as alignment
    application, reconstruction, classification, or re-extraction workflows.

    ## Shift Outputs

    The protocol also produces three scalar outputs:

    - **shiftX**;
    - **shiftY**;
    - **shiftZ**.

    These values record the 3D shift target used by the protocol.

    They are useful for documentation, reproducibility, and checking that the
    selected center or mask-derived center of mass was interpreted as expected.

    ## Interpretation of the Result

    The output particles should be interpreted as the same particles recentered
    according to the selected 3D point.

    No particle quality assessment, classification, or refinement is performed.
    The protocol only changes the particle centering and alignment metadata.

    If the selected point is biologically meaningful and correctly related to the
    particle alignment, the output can help focus downstream analysis on a desired
    region. If the point is wrong, the particles may become poorly centered.

    ## Practical Recommendations

    Use this protocol only with particles that already have valid projection
    transformation matrices.

    Use manual point selection when the desired center is a recognizable structural
    feature in a reference volume.

    Use the mask center of mass when the desired center corresponds to a segmented
    domain or region.

    Make sure that the volume or mask is in the same coordinate frame as the input
    particles.

    Use **Apply shift to particles = No** when you want a fast metadata-only update
    and plan to apply the transformation later.

    Use **Apply shift to particles = Yes** when downstream protocols require the
    particle images to be physically shifted.

    Inspect a subset of output particles to confirm that the recentering worked as
    expected.

    Be cautious when reducing the box size. Ensure that the recentered density is
    not cropped.

    ## Final Perspective

    Shift Particles is a recentering utility for aligned particle sets.

    For biological users, it is useful when a dataset should be shifted toward a
    specific structural point, such as a domain, subunit, or mask-defined center of
    mass. The protocol can either rewrite the particle images or store the shift in
    the alignment metadata, giving flexibility between immediate image
    transformation and faster metadata-only workflows.

    The most important requirement is that the selected center, reference volume,
    mask, and particle transformations all share the same coordinate frame.
    """

    _label = 'shift particles'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, pointerClass='SetOfParticles', label="Particles",
                      help='Select the SetOfParticles with transformation matrix to be shifted.')
        form.addParam('option', BooleanParam, label='Select position in volume?', default='True',
                      help='Select the position where the particles will be shifted in a volume displayed in a wizard.')
        form.addParam('inputVol', PointerParam, pointerClass='Volume', label="Volume", allowsNull=True,
                      condition='option', help='Volume to select the point (by clicking in the wizard for selecting the'
                                               ' new center) that will be the new center of the particles.')
        form.addParam('xin', FloatParam, label="x", condition='option',
                      help='Use the wizard to select the new center for the shifted particles by shift+click on the '
                           'blue point a drag it to the desired location while pressing shift.')
        form.addParam('yin', FloatParam, label="y", condition='option')
        form.addParam('zin', FloatParam, label="z", condition='option')
        form.addParam('inputMask', PointerParam, pointerClass='VolumeMask', label="Volume mask", allowsNull=True,
                      condition='not option', help='3D mask to compute the center of mass, the particles will be '
                                                   'shifted to the computed center of mass')
        form.addParam('applyShift', BooleanParam, label='Apply shift to particles?', default='True',
                      help='Yes: The shift is applied to particle images and zero shift is stored in the metadata. No: '
                           'The shift is stored in the transformation matrix in the metadata, but not applied to the '
                           'particle image (i.e. the output images are the same of input images). This option takes '
                           'less time and the shift could be applied later using protocol "xmipp3 - apply alignment 2d"'
                           ' or by re-extracting the particles.')
        form.addParam('boxSizeBool', BooleanParam, label='Use original box size for the shifted particles?',
                      default='True', help='Use input particles box size for the shifted particles.')
        form.addParam('boxSize', IntParam, label='Final box size', condition='not boxSizeBool',
                      help='Box size for the shifted particles.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('shiftStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        """convert input particles into .xmd file """
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath("input_particles.xmd"))

    def shiftStep(self):
        """call xmipp program to shift the particles"""
        if self.option:
            self.x = self.xin.get()
            self.y = self.yin.get()
            self.z = self.zin.get()
        else:
            fnvol = self.inputMask.get().getFileName()
            if fnvol.endswith('.mrc'):
                fnvol += ':mrc'
            vol = Volume()
            vol.setFileName(fnvol)
            vol = ih().read(vol.getFileName())
            masscenter = vol.centerOfMass()
            self.x = masscenter[0] - 0.5 * vol.getDimensions()[0]
            self.y = masscenter[1] - 0.5 * vol.getDimensions()[1]
            self.z = masscenter[2] - 0.5 * vol.getDimensions()[2]

        # Shift matrix
        self.tr_shift = np.eye(4)
        self.tr_shift[0, -1] = self.x
        self.tr_shift[1, -1] = self.y
        self.tr_shift[2, -1] = self.z

    def createOutputStep(self):
        """create output with the new particles"""
        self.ix = 0
        inputParticles = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputParticles)

        # Prepare images
        if self.applyShift.get():
            images = np.squeeze(readImage(inputParticles.getFirstItem().getFileName()).getData())
            transformed_images = []
            transformed_images_fn = self._getExtraPath("transformed_images.mrcs")

        for particle in inputParticles.iterItems():
            tr = particle.getTransform().getMatrix()
            tr = self.tr_shift @ tr

            if self.applyShift.get():
                # Transformation matrix to be applied to the image
                tr_inv = np.linalg.inv(tr)
                tr_to_be_applied = np.eye(3)
                tr_to_be_applied[0, -1] = -tr_inv[0, -1]
                tr_to_be_applied[1, -1] = -tr_inv[1, -1]

                # Transformation matrix to be saved in the metadata
                tr_inv[0, -1] = 0.0
                tr_inv[1, -1] = 0.0
                tr = np.linalg.inv(tr_inv)

                # Apply transformation to image
                image = images[particle.getObjId() - 1]
                transformed_images.append(applyTransform(image, tr_to_be_applied, image.shape))

                # Change particle location
                particle.setLocation((particle.getObjId(), transformed_images_fn))

            particle.getTransform().setMatrix(tr)
            outputSet.append(particle.clone())

        # Save transformed images if needed
        if self.applyShift.get():
            writeImageFromArray(np.stack(transformed_images, axis=0)[:, None, ...], transformed_images_fn)

        self._defineOutputs(outputParticles=outputSet)
        self._defineOutputs(shiftX=pwobj.Float(self.x),
                            shiftY=pwobj.Float(self.y),
                            shiftZ=pwobj.Float(self.z))
        self._defineSourceRelation(inputParticles, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputParticles'):
            summary.append("Output particles not ready yet.")
        else:
            if self.option:
                option = "User defined shift"
            else:
                option = "Shift to center of mass"
            if not self.interp.get():
                interp = 'linear'
            else:
                interp = 'spline'
            summary.append("%s\ninterpolation: %s" % (option, interp))
            if self.inv.get():
                summary.append("inverse matrix applied")
        return summary

    def _validate(self):
        for part in self.inputParticles.get().iterItems():
            if not part.hasTransform():
                validatemsg = ['Please provide particles which have transformation matrix.']
                return validatemsg
