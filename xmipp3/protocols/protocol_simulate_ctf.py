# **************************************************************************
# *
# * Authors:     C.O.S. Sorzano (coss@cnb.csic.es)
# *              Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
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
import random

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_3_0
from pwem.protocols import Prot2D
from pwem.objects import CTFModel
from pyworkflow import BETA, UPDATED, NEW, PROD

import xmippLib


class XmippProtSimulateCTF(Prot2D):
    """
    Simulate the effect of the CTF (no amplitude decay).
    A random defocus is chosen between the lower and upper defocus for each
    projection.

    AI Generated

    ## Overview

    The Simulate CTF protocol applies a simulated contrast transfer function, or
    CTF, to an existing set of particle images.

    In cryo-EM, the microscope CTF modifies the recorded image depending on
    parameters such as defocus, voltage, spherical aberration, and amplitude
    contrast. This protocol creates synthetic CTF-affected particles by filtering
    each input particle with a randomly selected defocus value within a user-defined
    range.

    The protocol can simulate either non-astigmatic or astigmatic CTF. It can also
    add Gaussian noise before and/or after the CTF is applied.

    The main output is a new set of particles with simulated CTF effects and
    corresponding CTF metadata assigned to each particle.

    ## Inputs and General Workflow

    The input is a set of particles.

    The protocol creates an output image stack with the same image dimensions and
    number of particles as the input set. Then, for each particle, it selects a
    random defocus value between the lower and upper defocus limits. If astigmatism
    is enabled, it also generates a second defocus value and a defocus angle.

    The particle is filtered with the corresponding CTF. Optional noise can be
    added before the CTF filtering, after the CTF filtering, or both.

    For each output particle, the protocol stores the simulated CTF parameters in
    the particle metadata.

    ## Input Particles

    The **Input particles** parameter defines the particle set to which the
    simulated CTF will be applied.

    These particles may be synthetic projections, clean particles, previously
    processed particles, or any Scipion-compatible SetOfParticles.

    The protocol does not generate particle projections by itself. It modifies the
    existing particle images by applying a CTF filter and optional noise.

    This makes the protocol useful for simulation experiments, testing algorithms,
    training workflows, and controlled validation of CTF-aware processing methods.

    ## Voltage

    The **Voltage** parameter defines the microscope acceleration voltage, in kV.

    Typical cryo-EM values are 200 kV or 300 kV. The voltage affects the electron
    wavelength and therefore the shape of the simulated CTF.

    The selected voltage is also stored in the acquisition metadata of the output
    particle set and each output particle.

    ## Spherical Aberration Cs

    The **Spherical aberration Cs** parameter defines the microscope spherical
    aberration coefficient, in millimeters.

    This value affects the phase shift introduced by the microscope optics and
    therefore the CTF oscillations.

    The default value is 2.7 mm, a common value for many microscope configurations.
    Users should modify it if they want to simulate a specific microscope setup.

    ## Fraction Inelastic Scattering

    The **Fraction inelastic scattering** parameter corresponds to the amplitude
    contrast, also called Q0 in the protocol.

    The value should be between 0 and 1. It controls the amplitude-contrast
    contribution to the simulated CTF.

    A typical cryo-EM value is around 0.07, which is the default used by the
    protocol.

    This value is stored as the amplitude contrast in the output acquisition
    metadata.

    ## Defocus Range

    The **Lower defocus** and **Upper defocus** parameters define the range from
    which the defocus value is randomly sampled for each particle.

    For every input particle, the protocol selects one defocus value uniformly at
    random between these two limits.

    The values are expressed in angstroms. Negative values represent overfocus.

    Using a range of defocus values allows the output particle set to mimic a
    dataset acquired at different defoci.

    ## Non-Astigmatic CTF

    When **Simulate astigmatic CTF?** is disabled, the protocol simulates a
    non-astigmatic CTF.

    In this case, the same defocus value is used for both defocus directions:

    - defocusU = defocusV;
    - defocusAngle = 0.

    This is the simpler simulation mode and is useful when the user wants to study
    the effect of defocus without astigmatism.

    ## Astigmatic CTF

    When **Simulate astigmatic CTF?** is enabled, the protocol simulates an
    astigmatic CTF.

    For each particle, the protocol first samples defocusU from the main defocus
    range. It then samples a defocus difference between the lower and upper
    defocus-difference values and adds it to defocusU to obtain defocusV.

    It also samples a defocus angle between the lower and upper angle limits.

    This produces particle-specific astigmatic CTF metadata:

    - defocusU;
    - defocusV;
    - defocusAngle.

    Astigmatic simulation is useful for testing algorithms under more realistic
    microscope conditions.

    ## Defocus Angle Range

    The **Lower defocus angle** and **Upper defocus angle** parameters are used
    when astigmatism is enabled.

    They define the range, in degrees, from which the astigmatism angle is randomly
    sampled for each particle.

    The help text indicates that values should be between 0 and 90 degrees.

    ## Defocus Difference Range

    The **Lower defocus difference between defocusU and defocusV** and **Upper
    defocus difference between defocusU and defocusV** parameters control the
    amount of astigmatism.

    For each particle, a random value is sampled from this range and added to
    defocusU to obtain defocusV.

    Small values produce weak astigmatism. Larger positive or negative values
    produce stronger differences between the two defocus directions.

    ## Noise Before CTF

    The **Noise before CTF** parameter controls Gaussian noise added before the CTF
    filter is applied.

    The value is the standard deviation, or sigma, of the Gaussian noise. If the
    value is 0, no pre-CTF noise is added.

    Adding noise before CTF simulates noise that is affected by the same CTF
    filtering as the particle signal.

    This can be useful for simulation studies where the user wants to distinguish
    between noise introduced before and after image formation.

    ## Noise After CTF

    The **Noise after CTF** parameter controls Gaussian noise added after the CTF
    filter is applied.

    The value is the standard deviation, or sigma, of the Gaussian noise. If the
    value is 0, no post-CTF noise is added.

    Adding noise after CTF simulates detector or image-level noise that is not
    filtered by the CTF in the same way as the signal.

    The protocol uses a fixed seed for this post-CTF random generator, making this
    part of the noise generation reproducible.

    ## Output Particles

    The main output is **outputParticles**.

    This output contains the CTF-simulated particle images. The images are written
    to a new MRC stack called `images.mrc`.

    Each output particle keeps the original particle information when possible,
    but its image location points to the new simulated stack. The output particle
    set is marked as having CTF information.

    The output acquisition metadata include the selected voltage, spherical
    aberration, amplitude contrast, and magnification.

    ## Output CTF Metadata

    Each output particle receives a CTF model containing the simulated parameters.

    For non-astigmatic simulations, defocusU and defocusV are identical and the
    defocus angle is 0.

    For astigmatic simulations, defocusU, defocusV, and defocus angle vary
    according to the user-defined random ranges.

    This metadata allows downstream protocols to treat the output as a particle set
    with known simulated CTF parameters.

    ## Interpretation of the Result

    The output particles should be interpreted as synthetic CTF-affected versions
    of the input particles.

    The protocol does not simulate all microscope effects. In particular, the code
    description notes that the simulated CTF has no amplitude decay. Therefore, the
    output should be understood as a controlled CTF simulation rather than a full
    realistic microscope simulation.

    The optional noise parameters can make the data more realistic, but the result
    still depends on simplified assumptions.

    ## Practical Recommendations

    Use clean or synthetic input particles when testing how algorithms respond to
    known CTF parameters.

    Use a realistic voltage, spherical aberration, and amplitude contrast for the
    microscope conditions you want to simulate.

    Set the defocus range to match the scenario being tested. A wide range creates
    more heterogeneous CTF conditions.

    Enable astigmatism when testing algorithms that must handle defocusU,
    defocusV, and defocus-angle variation.

    Use noise before and after CTF to create more challenging synthetic datasets,
    but start with zero noise when debugging.

    Remember that negative defocus values represent overfocus.

    Inspect the output particles and metadata before using them in downstream
    validation or training workflows.

    ## Final Perspective

    Simulate CTF is a synthetic-data protocol for applying controlled CTF effects
    to an existing particle set.

    For biological users and method developers, its value is that it creates
    particle images with known CTF parameters. This makes it useful for testing
    CTF correction, reconstruction, classification, training, and validation
    workflows under controlled conditions.

    The protocol should be understood as a simplified CTF simulation tool, not as a
    complete physical simulator of cryo-EM image formation.
    """

    _label = 'simulate ctf'
    _devStatus = PROD

    _lastUpdateVersion = VERSION_3_0

    def __init__(self, *args, **kwargs):
        Prot2D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', params.PointerParam,
                      allowsNull=False,
                      pointerClass='SetOfParticles',
                      label="Input particles")
        form.addParam('voltage', params.FloatParam, default=300,
                      label="Voltage (kV)")
        form.addParam('cs', params.FloatParam, default=2.7,
                      label="Spherical aberration Cs (mm)")
        form.addParam('Q0', params.FloatParam, default=0.07,
                      label="Fraction inelastic scattering",
                      help="Between 0 and 1")
        form.addParam('Defocus0', params.FloatParam, default=5000,
                      label="Lower defocus (A)",
                      help="Negative value is overfocus")
        form.addParam('DefocusF', params.FloatParam, default=25000,
                      label="Upper defocus (A)",
                      help="Negative value is overfocus")
        form.addParam('astig', params.BooleanParam, default=False,
                      label="Simulate astigmatic CTF?",
                      help="If yes, defocusU and defocusV will have different values with a difference determined by"
                           " the user, and there will be a value for angle")
        form.addParam('angle0', params.FloatParam, default=40, condition='astig',
                      label="Lower defocus angle (degrees)",
                      help="Between 0 and 90")
        form.addParam('angleF', params.FloatParam, default=50, condition='astig',
                      label="Upper defocus angle (degrees)",
                      help="Between 0 and 90")
        form.addParam('Defocus0diff', params.FloatParam, default=-500, condition='astig',
                      label="Lower defocus difference between defocusU and defocusV (A)")
        form.addParam('DefocusFdiff', params.FloatParam, default=500, condition='astig',
                      label="Upper defocus difference between defocusU and defocusV(A)")
        form.addParam('noiseBefore', params.FloatParam, default=0, label='Noise before CTF', help='Sigma')
        form.addParam('noiseAfter', params.FloatParam, default=0, label='Noise after CTF', help='Sigma')

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('simulateStep')

    # --------------------------- STEPS functions -------------------------------
    def convertInputStep(self):
        x, y, _ = self.inputParticles.get().getDimensions()
        n = self.inputParticles.get().getSize()
        xmippLib.createEmptyFile(self._getPath("images.mrc"), x, y, 1, n)

    def simulateStep(self):
        n = 1
        fnStk = self._getPath("images.mrc")
        Ts = self.inputParticles.get().getSamplingRate()

        imgSetOut = self._createSetOfParticles()
        imgSetOut.copyInfo(self.inputParticles.get())
        imgSetOut.setHasCTF(True)
        acquisition = imgSetOut.getAcquisition()
        acquisition.setVoltage(self.voltage.get())
        acquisition.setAmplitudeContrast(self.Q0.get())
        acquisition.setSphericalAberration(self.cs.get())
        acquisition.setMagnification(1)

        for particle in self.inputParticles.get():
            location = particle.getLocation()
            fnIn = str(location[0]) + "@" + location[1]
            fnOut = str(n) + "@" + fnStk

            if self.noiseBefore>0:
                I=xmippLib.Image(fnIn)
                Idata = I.getData()
                generator = np.random.default_rng()
                I.setData(Idata+self.noiseBefore.get()*generator.normal(size=Idata.shape))
                I.write(fnOut)
                fnIn=fnOut

            defocusU = random.uniform(self.Defocus0.get(), self.DefocusF.get())
            args = "-i %s -o %s" % (fnIn, fnOut)
            if self.astig:
                defocusV = defocusU + random.uniform(self.Defocus0diff.get(), self.DefocusFdiff.get())
                defocusAngle = random.uniform(self.angle0.get(), self.angleF.get())
                args += " --fourier ctfdefastig %f %f %f %f %f %f --sampling %f -v 0" % \
                        (self.voltage, self.cs, self.Q0, defocusU, defocusV, defocusAngle, Ts)
            else:
                defocusV = defocusU
                defocusAngle = 0
                args += " --fourier ctfdef %f %f %f %f  --sampling %f -v 0" % \
                        (self.voltage, self.cs, self.Q0, defocusU, Ts)
            self.runJob("xmipp_transform_filter", args)

            if self.noiseAfter>0:
                I=xmippLib.Image(fnOut)
                Idata = I.getData()
                generator = np.random.default_rng(42) #Provided a seed for this random generator.
                I.setData(Idata+self.noiseAfter.get()*generator.normal(size=Idata.shape))
                I.write(fnOut)

            newCTF = CTFModel()
            newCTF.setDefocusU(defocusU)
            newCTF.setDefocusV(defocusV)
            newCTF.setDefocusAngle(defocusAngle)
            newParticle = particle.clone()
            newParticle.setLocation((n, fnStk))
            acquisition = newParticle.getAcquisition()
            acquisition.setVoltage(self.voltage.get())
            acquisition.setAmplitudeContrast(self.Q0.get())
            acquisition.setSphericalAberration(self.cs.get())
            acquisition.setMagnification(1)
            newParticle.setCTF(newCTF)
            imgSetOut.append(newParticle)
            n += 1

        self._defineOutputs(outputParticles=imgSetOut)
        self._defineSourceRelation(self.inputParticles.get(), imgSetOut)

    # --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []
        summary.append("Voltage=%f kV" % self.voltage)
        summary.append("Cs=%f mm" % self.cs)
        summary.append("Q0=%f" % self.Q0)
        summary.append("Defocus range=[%f,%f] A" % (self.Defocus0, self.DefocusF))
        summary.append('Noise before=%f' %self.noiseBefore)
        summary.append('Noise after=%f' %self.noiseAfter)
        return summary

