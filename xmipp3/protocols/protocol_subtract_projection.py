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
import pwem.emlib.metadata as md
from pwem.protocols import EMProtocol
from xmipp3.convert import writeSetOfParticles


class XmippProtSubtractProjection(EMProtocol):
    """ This protocol computes the subtraction between particles and a initial volume, by computing its projections
    with the same angles that input particles have. Then, each particle and the correspondent projection of the initial
    volume are numerically adjusted and subtracted using a mask which denotes the region to keep. """

    _label = 'subtract projection'
    INPUT_PARTICLES = "input_particles.xmd"

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('particles', PointerParam, pointerClass='SetOfParticles', label="Particles: ",
                      help='Specify a SetOfParticles.')
        form.addParam('vol', PointerParam, pointerClass='Volume', label="Initial volume ", help='Specify a volume.')
        form.addParam('maskVol', PointerParam, pointerClass='VolumeMask', label='Volume mask', allowsNull=True,
                      help='3D mask for the input volume. This mask is not mandatory but advisable.')
        form.addParam('mask', PointerParam, pointerClass='VolumeMask', label="Mask for region to keep", allowsNull=True,
                      help='Specify a 3D mask for the region of the input volume that you want to keep.')
        form.addParam('resol', FloatParam, label="Filter at resolution: ", default=3, allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Resolution (A) at which subtraction will be performed, filtering the volume projections.'
                           'Value 0 implies no filtering.')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3, condition='resol',
                      help='Decay of the filter (sigma parameter) to smooth the mask transition',
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('iter', IntParam, label="Number of iterations: ", default=5, expertLevel=LEVEL_ADVANCED,
                      help='Number of iterations for the adjustment process of the images before the subtraction itself'
                           'several iterations are recommended to improve the adjustment.')
        form.addParam('rfactor', FloatParam, label="Relaxation factor (lambda): ", default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Relaxation factor for Fourier amplitude projector (POCS), it should be between 0 and 1, '
                           'being 1 no relaxation and 0 no modification of volume 2 amplitudes')

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
        resol = self.resol.get()
        iters = self.iter.get()
        program = "xmipp_subtract_projection"
        args = '-i %s --ref %s -o %s --iter %s --lambda %s' % (self._getExtraPath(self.INPUT_PARTICLES), fnVol,
                                                               self._getExtraPath("output_particles"), iters,
                                                               self.rfactor.get())
        args += ' --saveProj %s' % self._getExtraPath('')
        if resol:
            fc = vol.getSamplingRate()/resol
            args += ' --cutFreq %f --sigma %d' % (fc, self.sigma.get())
        if self.maskVol.get() is not None:
            args += ' --maskVol %s' % self.maskVol.get().getFileName()
        if self.mask.get() is not None:
            args += ' --mask %s' % self.mask.get().getFileName()
        self.runJob(program, args)

    def createOutputStep(self):
        self.ix = 0  # initiate counter for particle file name
        inputSet = self.particles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputSet)
        outputSet.copyItems(inputSet, updateItemCallback=self._updateItem,
                            itemDataIterator=md.iterRows(self._getExtraPath(self.INPUT_PARTICLES)))
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputSet, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = ["Volume: %s" % self.vol.get().getFileName()]
        summary.append("Set of particles: %s" % self.particles.get())
        if self.mask:
            summary.append("Mask: %s" % self.mask.get().getFileName())
        if self.resol.get() != 0:
            summary.append("Subtraction at resolution %f A" % self.resol.get())
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append("Volume projections from %s subtracted from particles" %
                           basename(self.vol.get().getFileName()))
            if self.mask:
                methods.append("with mask %s" % basename(self.mask.get().getFileName()))
            if self.resol.get() != 0:
                methods.append(" at resolution %f A" % self.resol.get())
        return methods

    def _validate(self):
        errors = []
        rfactor = self.rfactor.get()
        if rfactor < 0 or rfactor > 1:
            errors.append('Relaxation factor (lambda) must be between 0 and 1')
        return errors

    # --------------------------- UTLIS functions --------------------------------------------
    def _updateItem(self, item, row):
        newFn = row.getValue(md.MDL_IMAGE)
        self.ix = self.ix + 1
        newFn = newFn.split('@')[1]
        item.setLocation(self.ix, newFn)
