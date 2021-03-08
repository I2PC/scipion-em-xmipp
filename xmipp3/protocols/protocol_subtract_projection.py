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

from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, FloatParam, EnumParam, StringParam, FileParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem.convert import headers, downloadPdb, cifToPdb
from pwem.objects import Volume, Transform
from pwem.protocols import EMProtocol


class XmippProtSubtractProjection(EMProtocol):
    """ This protocol computes the projection subtraction between particles and initial volume. To achive this, it
    computes projections with the same angles of input particles from an input initial volume and
    then, each particle is adjusted and subtracted to its correspondent projection. """

    _label = 'projection subtraction'

    # --------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('vol1', PointerParam, pointerClass='Volume', label="Volume 1 ", help='Specify a volume.')
        form.addParam('masks', BooleanParam, label='Mask volumes?', default=True,
                      help='The masks are not mandatory but highly recommendable.')
        form.addParam('mask1', PointerParam, pointerClass='VolumeMask', label="Mask for volume 1",
                      condition='masks', help='Specify a mask for volume 1.')
        form.addParam('resol', FloatParam, label="Filter at resolution: ", default=3, allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Resolution (A) at which subtraction will be performed, filtering the input volumes.'
                           'Value 0 implies no filtering.')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3, condition='resol',
                      help='Decay of the filter (sigma parameter) to smooth the mask transition',
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('iter', IntParam, label="Number of iterations: ", default=5, expertLevel=LEVEL_ADVANCED)
        form.addParam('rfactor', FloatParam, label="Relaxation factor (lambda): ", default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Relaxation factor for Fourier amplitude projector (POCS), it should be between 0 and 1, '
                           'being 1 no relaxation and 0 no modification of volume 2 amplitudes')

        form.addParam('pdb', BooleanParam, label='Is the second input a PDB?', default=False,
                      help='If yes, the protocol will generate and store in folder "extra" of this protocol '
                           'a volume and a mask from the pdb. This is not the recommended option, as the automatic '
                           'conversion of the PDB into a density map may not be successful due to origin mismatches. '
                           'We recommend to convert previously the PDB, inspect the converted map and use the map as '
                           'input. If not, a second volume has to be input and optionally (but highly recommendable), '
                           'a mask for it.')
        form.addParam('inputPdbData', EnumParam, choices=['object', 'file'], condition='pdb',
                      label="Retrieve PDB from", default=self.IMPORT_OBJ,
                      display=EnumParam.DISPLAY_HLIST,
                      help='Retrieve PDB data from server, use a pdb Object, or a local file')
        form.addParam('pdbObj', PointerParam, pointerClass='AtomStruct',
                      label="Input pdb ", condition='inputPdbData == IMPORT_OBJ and pdb', allowsNull=True,
                      help='Specify a pdb object. This is not the recommended option, as the automatic conversion of '
                           'the PDB into a density map may not be successful due to origin mismatches. We recommend'
                           'to convert previously the PDB, inspect the converted map and use the map as input.')
        form.addParam('pdbFile', FileParam,
                      label="File path", condition='inputPdbData == IMPORT_FROM_FILES and pdb', allowsNull=True,
                      help='Specify a path to desired PDB structure.')
        form.addParam('vol2', PointerParam, pointerClass='Volume', label="Volume 2 ", condition='pdb == False',
                      help='Specify a volume.')
        form.addParam('mask2', PointerParam, pointerClass='VolumeMask', label="Mask for volume 2",
                      condition='masks and pdb==False', help='Specify a mask for volume 1.')
        form.addParam('saveFiles', BooleanParam, label='Save intermediate files?', default=False,
                      expertLevel=LEVEL_ADVANCED, help='Save input volume 1 filtered and input volume 2 adjusted, which'
                                                       'are the volumes that are really subtracted.')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('subtractionStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        # convert input particles into .xmd file
        pass

    def subtractionStep(self):
        vol1 = self.vol1.get().clone()
        fileName1 = vol1.getFileName()
        if fileName1.endswith('.mrc'):
            fileName1 += ':mrc'
        if self.pdb:
            vol2 = self.outFile
            mask2 = self._getExtraPath("mask2.mrc")
        else:
            vol2 = self.vol2.get().getFileName()
            if vol2.endswith('.mrc'):
                vol2 += ':mrc'
            if self.masks:
                mask2 = self.mask2.get().getFileName()
        resol = self.resol.get()
        iter = self.iter.get()
        program = "xmipp_volume_subtraction"
        args = '--i1 %s --i2 %s -o %s --iter %s --lambda %s --sub' % \
               (vol1.getFileName(), vol2, self._getExtraPath("output_volume.mrc"), iter, self.rfactor.get())
        if resol:
            fc = vol1.getSamplingRate()/resol
            args += ' --cutFreq %f --sigma %d' % (fc, self.sigma.get())

        if self.masks:
            args += ' --mask1 %s --mask2 %s' % (self.mask1.get().getFileName(), mask2)
        if self.saveFiles:
            args += ' --saveV1 %s --saveV2 %s' % (self._getExtraPath('vol1_filtered.mrc'),
                                                  self._getExtraPath('vol2_adjusted.mrc'))
        self.runJob(program, args)

    def createOutputStep(self):
        vol1 = self.vol1.get()
        volume = Volume()
        volume.setSamplingRate(vol1.getSamplingRate())
        if vol1.getFileName().endswith('mrc'):
            origin = Transform()
            ccp4header = headers.Ccp4Header(vol1.getFileName(), readHeader=True)
            shifts = ccp4header.getOrigin()
            origin.setShiftsTuple(shifts)
            volume.setOrigin(origin)
        volume.setFileName(self._getExtraPath("output_volume.mrc"))
        filename = volume.getFileName()
        if filename.endswith('.mrc') or filename.endswith('.map'):
            volume.setFileName(filename + ':mrc')
        self._defineOutputs(outputVolume=volume)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = ["Volume 1: %s" % self.vol1.get().getFileName()]
        summary.append("Volume 2: %s" % self.vol2.get().getFileName())
        if self.masks:
            summary.append("Input mask 1: %s" % self.mask1.get().getFileName())
        if self.resol.get() != 0:
            summary.append("Subtraction at resolution %f A" % self.resol.get())
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            methods.append("Volume %s subtracted from volume %s" % (self.vol2.get().getFileName(),
                                                                    self.vol1.get().getFileName()))
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
