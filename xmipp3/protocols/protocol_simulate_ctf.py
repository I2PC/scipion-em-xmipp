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
    A random defocus is chosen between the lower and upper defocus for each projection.
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
                generator = np.random.default_rng()
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

