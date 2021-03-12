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
from pwem import ALIGN_3D
from pwem.emlib import lib
import pwem.emlib.metadata as md
from pwem.protocols import EMProtocol

from xmipp3.convert import alignmentToRow, ctfModelToRow


class XmippProtSubtractProjection(EMProtocol):
    """ This protocol computes the projection subtraction between particles and initial volume. To achive this, it
    computes projections with the same angles of input particles from an input initial volume and
    then, each particle is adjusted and subtracted to its correspondent projection. """

    _label = 'subtract projection'

    # --------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('particles', PointerParam, pointerClass='SetOfParticles', label="Particles: ",
                      help='Specify a SetOfParticles.')
        form.addParam('vol', PointerParam, pointerClass='Volume', label="Initial volume ", help='Specify a volume.')
        form.addParam('maskBool', BooleanParam, label='Subtract in a region?', default=True,
                      help='The mask is not mandatory but highly recommendable.')
        form.addParam('mask', PointerParam, pointerClass='VolumeMask', label="3D mask for subtraction region",
                      condition='maskBool', help='Specify a 3D mask for volume 1.')
        form.addParam('resol', FloatParam, label="Filter at resolution: ", default=3, allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Resolution (A) at which subtraction will be performed, filtering the volume projections.'
                           'Value 0 implies no filtering.')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3, condition='resol',
                      help='Decay of the filter (sigma parameter) to smooth the mask transition',
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('iter', IntParam, label="Number of iterations: ", default=5, expertLevel=LEVEL_ADVANCED)
        form.addParam('rfactor', FloatParam, label="Relaxation factor (lambda): ", default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Relaxation factor for Fourier amplitude projector (POCS), it should be between 0 and 1, '
                           'being 1 no relaxation and 0 no modification of volume 2 amplitudes')
        form.addParam('saveFiles', BooleanParam, label='Save intermediate files?', default=False,
                      expertLevel=LEVEL_ADVANCED, help='Save input particle filtered and computed projection adjusted, '
                                                       'which are the volumes that are really subtracted.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('subtractionStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        # convert input particles into .xmd file
        mdParticles = lib.MetaData()
        for part in self.particles.get():
            id = part.getObjId()
            ix = part.getIndex()
            fn = "%s@%s" % (ix, part.getFileName())
            nRow = md.Row()
            nRow.setValue(lib.MDL_ITEM_ID, int(id))
            nRow.setValue(lib.MDL_IMAGE, fn)
            alignmentToRow(part.getTransform(), nRow, ALIGN_3D)
            ctfModelToRow(part.getCTF(), nRow)
            nRow.addToMd(mdParticles)
        mdParticles.write(self._getExtraPath("input_particles.xmd"))
        
    def subtractionStep(self):
        vol = self.vol.get().clone()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        resol = self.resol.get()
        iter = self.iter.get()
        program = "xmipp_subtract_projection"
        args = '-i %s --ref %s -o %s --iter %s --lambda %s' % (self._getExtraPath("input_particles.xmd"), fnVol,
                                                               self._getExtraPath("output_particles"), iter,
                                                               self.rfactor.get())
        if resol:
            fc = vol.getSamplingRate()/resol
            args += ' --cutFreq %f --sigma %d' % (fc, self.sigma.get())

        if self.maskBool:
            args += ' --mask %s' % self.mask.get().getFileName()
        if self.saveFiles:
            args += ' --savePart %s --saveProj %s' % (self._getExtraPath('particle_filtered.mrc'),
                                                      self._getExtraPath('projection_adjusted.mrc'))
        self.runJob(program, args)

    def createOutputStep(self):
        self.ix = 0
        inputSet = self.particles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputSet)

        outputSet.copyItems(inputSet, updateItemCallback=self._updateItem,
                            itemDataIterator=md.iterRows(self._getExtraPath("input_particles.xmd")))

        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputSet, outputSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = ["Volume: %s" % self.vol.get().getFileName()]
        summary.append("Set of particles: %s" % self.particles.get())
        if self.maskBool:
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
            if self.maskBool:
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
