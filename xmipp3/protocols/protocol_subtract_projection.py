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


class XmippProtSubtractProjection(EMProtocol):
    """ This protocol computes the subtraction between particles and a reference volume, by computing its projections with the same angles that input particles have. Then, each particle and the correspondent projection of the reference volume are numerically adjusted and subtracted using a mask which denotes the region to keep. """

    _label = 'subtract projection'
    INPUT_PARTICLES = "input_particles.xmd"

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('particles', PointerParam, pointerClass='SetOfParticles', label="Particles: ",
                      help='Specify a SetOfParticles')
        form.addParam('vol', PointerParam, pointerClass='Volume', label="Reference volume ", help='Specify a volume.')
        form.addParam('mask', PointerParam, pointerClass='VolumeMask', label='Mask for region to keep', allowsNull=True,
                      help='Specify a 3D mask for the region of the input volume that you want to keep. '
                           'If no mask is given, the subtraction is performed in whole images.')
        form.addParam('mwidth', FloatParam, label="Extra width in final mask: ", default=40, expertLevel=LEVEL_ADVANCED,
                      help='Length (in A) to add for each side to the final result mask. -1 means no mask.')
        form.addParam('resol', FloatParam, label="Maximum resolution: ", default=3,  expertLevel=LEVEL_ADVANCED,
                      help='Maximum resolution (in A) of the data ')
        form.addParam('limit_freq', BooleanParam, label="Limit frequency?: ", default=False, expertLevel=LEVEL_ADVANCED,
                      help='Limit frequency in the adjustment process to the frequency correspondent to the resolution'
                           ' indicated in "Maximum resolution" field above')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3, expertLevel=LEVEL_ADVANCED,
                      help='Decay of the filter (sigma) to smooth the mask transition')
        form.addParam('pad', IntParam, label="Fourier padding factor: ", default=2, expertLevel=LEVEL_ADVANCED,
                      help='The volume is zero padded by this factor to produce projections')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('subtractionStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.particles.get(), self._getExtraPath(self.INPUT_PARTICLES))

    def subtractionStep(self):
        vol = self.vol.get().clone()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        args = '-i %s --ref %s -o %s --sampling %f --max_resolution %f --fmask_width %f --padding %f ' \
               '--sigma %d --limit_freq %d' % \
               (self._getExtraPath(self.INPUT_PARTICLES), fnVol, self._getExtraPath("output_particles"),
                vol.getSamplingRate(), self.resol.get(), self.mwidth.get(), self.pad.get(), self.sigma.get(),
                int(self.limit_freq.get()))
        mask = self.mask.get()
        if mask is not None:
            args += ' --mask %s' % mask.getFileName()
        args += ' --save %s' % self._getExtraPath()
        self.runJob("xmipp_subtract_projection", args)

    def createOutputStep(self):
        inputSet = self.particles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputSet)
        readSetOfParticles(self._getExtraPath('input_particles.xmd'), outputSet, extraLabels=[emlib.MDL_SUBTRACTION_R2])
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputSet, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.particles.get().getFirstItem()
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
                   (self.vol.get().getFileName(), self.particles.get(), self.mask.get().getFileName())]
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append("Volume projections subtracted to particles keeping the region in %s"
                           % basename(self.mask.get().getFileName()))
        return methods
