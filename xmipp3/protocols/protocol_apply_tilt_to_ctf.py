# **************************************************************************
# *
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

from typing import Optional

import pyworkflow.protocol.params as params
from pyworkflow.utils.properties import Message

from pwem.objects import (Particle, Coordinate, Micrograph, CTFModel,
                          SetOfParticles, SetOfMicrographs)
from pwem.protocols import EMProtocol
from pyworkflow import BETA, UPDATED, NEW, PROD

import math

class XmippProtApplyTiltToCtf(EMProtocol):
    """ Applies a local deviation correction to the particle’s contrast
    transfer function (CTF) estimation based on the tilt angle of the
    micrograph. This adjustment improves reconstruction quality, especially for
    tilted samples.

    AI Generated:

    What this protocol is for

    Apply tilt to CTF is a practical correction step for datasets acquired with
    a known stage tilt (or effective specimen tilt) where defocus varies
    systematically across the micrograph. In a tilted acquisition, particles
    located on one side of the image are physically closer to the objective
    lens than particles on the opposite side, so they experience a different
    defocus. If you treat the whole micrograph as having a single defocus
    value, that assumption becomes increasingly inaccurate as the tilt grows,
    and this can degrade downstream refinement and resolution.

    This protocol applies a simple, physically motivated correction: it updates
    each particle’s defocus by adding an offset that depends on the particle’s
    position along the tilt axis and the specified tilt angle. The result is a
    new particle set whose per-particle CTF defocus values are adjusted to
    reflect the linear defocus gradient induced by tilt. For a biological user,
    the goal is straightforward: better CTF modeling for tilted data, which
    often translates into improved consistency in refinement and better
    high-resolution signal.

    This is not a full “local CTF refinement” algorithm; it is an explicit
    geometric correction based on known tilt geometry. It is therefore
    particularly useful when you know the acquisition tilt angle (for example
    from the microscope settings) and you want a fast, deterministic way to
    incorporate that information into per-particle defocus.

    What you need to provide

    You provide a set of particles and the set of micrographs from which they
    were extracted. The particles must already have two types of metadata: a
    CTF model (so there are defocus values to correct) and particle coordinates
    (so the protocol knows where each particle sits within its micrograph). The
    micrographs are needed because the correction depends on the micrograph
    dimensions and sampling rate, and because particles need to be associated
    with their parent micrograph.

    In practice, you run this after particle extraction and after a standard
    CTF estimation step that has assigned defocus values to particles
    (typically inherited from micrograph CTF, or already particle-level). Then
    you apply the tilt-induced correction to produce a particle set with
    updated defoci.

    How the correction works conceptually

    The protocol assumes that tilt produces a linear defocus ramp across the
    micrograph along a chosen axis. You tell the protocol which axis
    corresponds to the tilt direction (X or Y), the tilt angle in degrees, and
    whether defocus increases or decreases along that axis. For each particle,
    the protocol measures how far the particle is from the micrograph center
    along the chosen axis, converts that distance into ångströms using the
    micrograph pixel size, and multiplies it by the sine of the tilt angle
    (with the appropriate sign). That quantity is the defocus offset added to
    both DefocusU and DefocusV.

    Biologically, this is exactly what you expect from simple tilted-plane
    geometry: particles on one side of the micrograph are effectively at a
    different height relative to the beam, and thus show a systematic defocus
    shift. This protocol encodes that correction in the particle CTF parameters.

    Tilt axis: choosing X vs Y

    The tilt axis parameter specifies the direction along which defocus changes
    across the micrograph. The protocol offers X and Y. The correct choice
    depends on the microscope acquisition geometry and on how your micrographs
    are oriented in your processing pipeline.

    As a practical guideline, in many tomography-related conventions the tilt
    axis is often treated as Y, but for single-particle tilted data it can vary
    depending on acquisition settings and any rotations introduced during
    motion correction or preprocessing. If you choose the wrong axis, the
    protocol will apply the gradient in the wrong direction, which can make CTF
    modeling worse rather than better. When in doubt, it is useful to confirm
    tilt direction from acquisition metadata or by inspecting whether defocus
    appears to vary predominantly along one direction in diagnostic tools.

    Tilt angle: what value to use

    The tilt angle is the acquisition tilt in degrees and is restricted to the
    range 0–90°. A tilt angle of 0 means no correction (no gradient). As tilt
    increases, the defocus gradient across the micrograph becomes stronger, and
    this correction becomes more important.

    For a biological user, the main decision is simply to use the tilt value
    that matches your acquisition. If your dataset includes multiple tilts (for
    example, intentionally varied tilts), you should apply this protocol
    separately per subset or use appropriate metadata-driven approaches; this
    particular protocol applies one global tilt angle setting to all particles
    you feed into it.

    Tilt sign: “Increasing” vs “Decreasing”

    Even with the correct axis and angle, you must specify whether defocus
    increases or decreases as you move along the chosen axis. This is
    essentially the directionality of the gradient. The protocol provides two
    options: “Increasing” and “Decreasing”.

    In biological practice, the easiest way to determine the correct sign is
    to compare a few particles from opposite sides of a micrograph (along the
    tilt axis) using any tool that can estimate local defocus or by inspecting
    whether refinement improves when using one sign versus the other. If you
    pick the wrong sign, you will invert the defocus ramp and degrade CTF
    consistency.

    Output: what you get

    The protocol outputs a new SetOfParticles with updated CTF parameters.
    The only intended change is the per-particle defocus values: both DefocusU
    and DefocusV are shifted by the computed tilt-induced offset. Everything
    else about the particle set—images, coordinates, general metadata—remains
    the same.

    This makes the output easy to integrate downstream: you simply use the
    output particle set in subsequent steps (2D classification, refinement,
    polishing strategies that rely on particle CTF, etc.) so that those steps
    see a more realistic defocus model for each particle.

    When this helps most (biological perspective)

    This protocol is most useful when you have moderate to high tilt and a
    fairly large field of view, because that combination produces a sizable
    defocus difference across the micrograph. It can be beneficial for
    specimens that require tilt to overcome preferred orientation, and for
    workflows where high-resolution consistency depends strongly on accurate
    CTF modeling.

    It is also useful as a lightweight alternative when you do not want to
    run a full local CTF refinement but you still want to account for the
    dominant, predictable component of defocus variability introduced by tilt.

    What this protocol does not replace

    This correction is intentionally simple: it models defocus variation as a
    linear gradient. Real micrographs may also exhibit non-linear defocus
    behavior due to uneven ice thickness, charging, local bending, or
    higher-order optical effects. If those effects dominate, you may still
    need true local CTF refinement tools. Nonetheless, even in those cases,
    applying the known tilt geometry can remove a large systematic component
    and make subsequent local refinement more stable.

    Practical recommendation

    A good practical workflow is to apply this correction when you know the
    acquisition tilt and then evaluate whether downstream refinement metrics
    (for example, map sharpening behavior, FSC, or visual high-frequency detail)
    improve. If you see improvement, keep it; if not, re-check axis and sign
    first before concluding that tilt correction is unnecessary.
    """
    _devStatus = PROD

    _label = 'apply tilt to ctf'
    _tilt_axes = ['X', 'Y']
    _tilt_signs = ['Increasing', 'Decreasing']

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam, label=Message.LABEL_INPUT_PART,
                      pointerClass=SetOfParticles, pointerCondition='hasCTF and hasCoordinates',
                      important=True,
                      help='Select the particles that you want to apply the'
                           'local CTF correction.')
        form.addParam('inputMicrographs', params.PointerParam, label=Message.LABEL_INPUT_MIC,
                      pointerClass=SetOfMicrographs,
                      important=True,
                      help='The micrographs from which particles were extracted.')
        form.addParam('tiltAxis', params.EnumParam, label='Tilt axis', 
                      choices=self._tilt_axes, default=1,
                      help='The tilt axis. In tomography this is Y by convention.')
        form.addParam('tiltAngle', params.FloatParam, label='Tilt angle',
                      default=0, validators=[params.Range(0, 90)],
                      help='The angle at which the acquisition is tilted. '
                      'In degrees.')
        form.addParam('tiltSign', params.EnumParam, label='Tilt sign',
                      choices=self._tilt_signs, default=0,
                      help='Wether defocus increases or decreases in terms '
                      'of the selected tilt axis.')
        
    #--------------------------- INSERT steps functions --------------------------------------------
    
    def _insertAllSteps(self):
        self._insertFunctionStep('createOutputStep')        

    #--------------------------- STEPS functions --------------------------------------------        
    def createOutputStep(self):
        inputParticles: SetOfParticles = self.inputParticles.get()
        outputParticles: SetOfParticles = self._createSetOfParticles()
        
        TILT_INDICES = [1, 0]
        SIGNS = [+1, -1]
        sign = SIGNS[self.tiltSign.get()]
        self.tiltIndex = TILT_INDICES[self.tiltAxis.get()]
        self.sineFactor = sign*math.sin(math.radians(self.tiltAngle.get()))
        
        outputParticles.copyInfo(inputParticles)
        outputParticles.copyItems(inputParticles,
                                  updateItemCallback=self._updateItem )
     
        self._defineOutputs(outputParticles=outputParticles)
        self._defineSourceRelation(self.inputParticles, outputParticles)
    
    #--------------------------- UTILS functions --------------------------------------------
    def _updateItem(self, particle: Particle, _):
        # Obtain necessary objects
        coordinate: Coordinate = particle.getCoordinate()
        micrograph: Optional[Micrograph] = coordinate.getMicrograph()
        if micrograph is None:
            micrographId = coordinate.getMicId()
            micrograph = self.inputMicrographs.get()[micrographId]
        
        # Compute the CTF offset
        dimensions = micrograph.getXDim(), micrograph.getYDim()
        position = coordinate.getPosition()
        r = position[self.tiltIndex] - (dimensions[self.tiltIndex] / 2)
        r *= micrograph.getSamplingRate() # Convert to angstroms.
        dy = self.sineFactor*r
        
        # Write to output
        ctf: CTFModel = particle.getCTF()
        ctf.setDefocusU(ctf.getDefocusU() + dy)
        ctf.setDefocusV(ctf.getDefocusV() + dy)
