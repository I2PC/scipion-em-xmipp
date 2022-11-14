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

from os.path import basename
from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, FloatParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pwem import emlib
from pwem.protocols import EMProtocol
from xmipp3.convert import writeSetOfParticles, readSetOfParticles


class XmippProtSubtractProjectionBase(EMProtocol):
    """ Helper class that contains some Protocol utilities methods
    used by both  XmippProtSubtractProjection and XmippProtBoostParticles."""

    # --------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineParams(cls, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, pointerClass='SetOfParticles', label="Particles: ",
                      help='Specify a SetOfParticles')
        form.addParam('vol', PointerParam, pointerClass='Volume', label="Reference volume ",
                      help='Specify a volume.')
        form.addParam('cirmaskrad', FloatParam, label="Circular mask radius: ", default=-1, expertLevel=LEVEL_ADVANCED,
                      help='Radius of the circular mask to avoid edge artifacts. '
                           'If -1 it is half the X dimension of the input particles')
        form.addParam('resol', FloatParam, label="Maximum resolution: ", default=3, expertLevel=LEVEL_ADVANCED,
                      help='Maximum resolution (in A) of the data ')
        form.addParam('nonNegative', BooleanParam, label="Ignore particles with negative beta0 or R2?: ",
                      default=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Particles with negative beta0 or R2 will not appear in the output set as they are '
                           'considered bad particles. Moreover, negative betas will not contribute to mean beta if '
                           '"mean" option is selected')
        form.addParam('limit_freq', BooleanParam, label="Limit frequency?: ", default=False,
                      expertLevel=LEVEL_ADVANCED,
                      help='Limit frequency in the adjustment process to the frequency correspondent to the resolution'
                           ' indicated in "Maximum resolution" field above')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3,
                      expertLevel=LEVEL_ADVANCED,
                      help='Decay of the filter (sigma) to smooth the mask transition')
        form.addParam('pad', IntParam, label="Fourier padding factor: ", default=2, expertLevel=LEVEL_ADVANCED,
                      help='The volume is zero padded by this factor to produce projections')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertSubSteps()
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputSet)
        readSetOfParticles(self._getExtraPath("output_particles.xmd"), outputSet,
                           extraLabels=[emlib.MDL_SUBTRACTION_R2, emlib.MDL_SUBTRACTION_BETA0,
                                        emlib.MDL_SUBTRACTION_BETA1])
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputSet, outputSet)


class XmippProtSubtractProjection(XmippProtSubtractProjectionBase):
    """ This protocol computes the subtraction between particles and a reference volume, by computing its projections with the same angles that input particles have. Then, each particle and the correspondent projection of the reference volume are numerically adjusted and subtracted using a mask which denotes the region to keep. """

    _label = 'subtract projection'
    INPUT_PARTICLES = "input_particles.xmd"

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        XmippProtSubtractProjectionBase._defineParams(form)
        form.addParam('mask', PointerParam, pointerClass='VolumeMask', label='Mask for region to keep', allowsNull=True,
                      help='Specify a 3D mask for the region of the input volume that you want to keep. '
                           'If no mask is given, the subtraction is performed in whole images.')
        form.addParam('mwidth', FloatParam, label="Extra width in final mask: ", default=40, expertLevel=LEVEL_ADVANCED,
                      help='Length (in A) to add for each side to the final mask. -1 means no mask.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertSubSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('subtractionStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath(self.INPUT_PARTICLES))

    def subtractionStep(self):
        vol = self.vol.get().clone()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        args = '-i %s --ref %s -o %s --sampling %f --max_resolution %f --fmask_width %f --padding %f ' \
               '--sigma %d --limit_freq %d --cirmaskrad %d --save %s' % \
               (self._getExtraPath(self.INPUT_PARTICLES), fnVol, self._getExtraPath("output_particles"),
                vol.getSamplingRate(), self.resol.get(), self.mwidth.get(), self.pad.get(), self.sigma.get(),
                int(self.limit_freq.get()), self.cirmaskrad.get(), self._getExtraPath())
        mask = self.mask.get()
        if mask is not None:
            args += ' --mask %s' % mask.getFileName()
        if self.nonNegative.get():
            args += ' --nonNegative'
        self.runJob("xmipp_subtract_projection", args)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.inputParticles.get().getFirstItem()
        vol = self.vol.get()
        mask = self.mask.get()
        if part.getDim()[0] != vol.getDim()[0]:
            errors.append("Input particles and volume should have same X and Y dimensions")
        if part.getSamplingRate() != vol.getSamplingRate():
            errors.append("Input particles and volume should have same sampling rate")
        if mask:
            if vol.getSamplingRate() != mask.getSamplingRate():
                errors.append("Input volume and mask should have same sampling rate")
            if vol.getDim() != mask.getDim():
                errors.append("Input volume and mask should have same dimensions")
        if self.resol.get() == 0:
            errors.append("Resolution (angstroms) should be bigger than 0")
        return errors

    def _summary(self):
        summary = ["Volume: %s\nSet of particles: %s\nMask: %s" %
                   (self.vol.get().getFileName(), self.inputParticles.get(), self.mask.get().getFileName())]
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append("Volume projections subtracted to particles keeping the region in %s"
                           % basename(self.mask.get().getFileName()))
        return methods


class XmippProtBoostParticles(XmippProtSubtractProjectionBase):
    """ This protocol tries to boost the frequencies of the particles to imporve them, based on an adjustment on its correspondent projections from a reference volume. """

    _label = 'boost particles'
    INPUT_PARTICLES = "input_particles.xmd"

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        XmippProtSubtractProjectionBase._defineParams(form)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertSubSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('boostingStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath(self.INPUT_PARTICLES))

    def boostingStep(self):
        vol = self.vol.get().clone()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        args = '-i %s --ref %s -o %s --sampling %f --max_resolution %f --padding %f --sigma %d --limit_freq %d ' \
               '--cirmaskrad %d --boost --save %s'\
               % (self._getExtraPath(self.INPUT_PARTICLES), fnVol, self._getExtraPath("output_particles"),
                  vol.getSamplingRate(), self.resol.get(), self.pad.get(), self.sigma.get(), int(self.limit_freq.get()),
                  self.cirmaskrad.get(), self._getExtraPath())

        if self.nonNegative.get():
            args += ' --nonNegative'
        self.runJob("xmipp_subtract_projection", args)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.inputParticles.get().getFirstItem()
        vol = self.vol.get()
        if part.getDim()[0] != vol.getDim()[0]:
            errors.append("Input particles and volume should have same X and Y dimensions")
        if part.getSamplingRate() != vol.getSamplingRate():
            errors.append("Input particles and volume should have same sampling rate")
        if self.resol.get() == 0:
            errors.append("Resolution (angstroms) should be bigger than 0")
        return errors

    def _summary(self):
        summary = ["Volume: %s\nSet of particles: %s\n" %
                   (self.vol.get().getFileName(), self.inputParticles.get())]
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append("Particles boosted according to their equivalent projections from a reference volume.")
        return methods