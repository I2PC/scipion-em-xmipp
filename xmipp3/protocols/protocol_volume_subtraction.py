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

from os.path import join
from shutil import move
from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, FloatParam, EnumParam, StringParam, FileParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.utils import replaceBaseExt, removeExt, getExt

from pwem.convert import headers, downloadPdb, cifToPdb
from pwem.objects import Volume, Transform
from pwem.protocols import ProtInitialVolume



class XmippProtVolSubtraction(ProtInitialVolume):
    """ This protocol scales a volume in order to assimilate it to another one. Then, it can calculate the subtraction
    of the two volumes. Second input can be a pdb. The volumes should be aligned previously and they have to
    be equal in size"""

    _label = 'volumes subtraction'
    IMPORT_OBJ = 0
    IMPORT_FROM_FILES = 1

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        form.addSection(label='Input')
        form.addParam('vol1', PointerParam, pointerClass='Volume', label="Volume 1 ", help='Specify a volume.')
        form.addParam('pdb', BooleanParam, label='Is the second input a PDB?', default=False)
        form.addParam('inputPdbData', EnumParam, choices=['object', 'file'], condition='pdb',
                      label="Retrieve PDB from", default=self.IMPORT_OBJ,
                      display=EnumParam.DISPLAY_HLIST,
                      help='Retrieve PDB data from server, use a pdb Object, or a local file')
        form.addParam('pdbObj', PointerParam, pointerClass='AtomStruct',
                      label="Input pdb ", condition='inputPdbData == IMPORT_OBJ and pdb', allowsNull=True,
                      help='Specify a pdb object.')
        form.addParam('pdbFile', FileParam,
                      label="File path", condition='inputPdbData == IMPORT_FROM_FILES and pdb', allowsNull=True,
                      help='Specify a path to desired PDB structure.')
        form.addParam('vol2', PointerParam, pointerClass='Volume', label="Volume 2 ", condition='pdb == False',
                      help='Specify a volume.')
        form.addParam('masks', BooleanParam, label='Mask volumes?', default=True,
                      help='The masks are not mandatory but highly recommendable.')
        form.addParam('mask1', PointerParam, pointerClass='VolumeMask', label="Mask for volume 1",
                      condition='masks', help='Specify a mask for volume 1.')
        form.addParam('mask2', PointerParam, pointerClass='VolumeMask', label="Mask for volume 2",
                      condition='masks and pdb==False', help='Specify a mask for volume 1.')
        form.addParam('sub', BooleanParam, label='Save difference volume:', default=True,
                      help='"Yes": the output of the protocol is the difference between input volumes. '
                           '"No": the output of the protocol is the second volume modified in order to assimilate it to'
                           'the first volume.')
        form.addParam('resol', FloatParam, label="Subtraction at resolution: ", default=5, allowsNull=True,
                      help='Resolution (A) at which subtraction will be performed, filtering the input volumes.'
                           'Value 0 implies no filtering.')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3, condition='resol',
                      help='Decay of the filter (sigma parameter) to smooth the mask transition',
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('iter', IntParam, label="Number of iterations: ", default=5, expertLevel=LEVEL_ADVANCED)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        if self.pdb:
            self._insertFunctionStep('convertPdbStep')
            self._insertFunctionStep('generateMask2Step')
        self._insertFunctionStep('subtractionStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertPdbStep(self):
        vol1 = self.vol1.get()
        pdbFn = self._getPdbFileName()
        self.outFile = self._getVolName()
        if getExt(pdbFn)==".cif":
            pdbFn2=replaceBaseExt(pdbFn, 'pdb')
            cifToPdb(pdbFn, pdbFn2)
            pdbFn = pdbFn2
        samplingR = vol1.getSamplingRate()
        size = vol1.getDim()
        ccp4header = headers.Ccp4Header(vol1.getFileName(), readHeader=True)
        self.shifts = ccp4header.getOrigin()
        args = ' -i %s --sampling %f -o %s --size %d %d %d --orig %d %d %d' % \
                (pdbFn, samplingR, removeExt(self.outFile), size[2], size[1], size[0], self.shifts[0]/samplingR,
                 self.shifts[1]/samplingR, self.shifts[2]/samplingR)
        program = "xmipp_volume_from_pdb"
        self.runJob(program, args)

    def generateMask2Step(self):
        args = ' -i %s -o %s --select below 0.010000 --substitute binarize' % (self.outFile, self._getExtraPath("mask2.mrc"))
        program = "xmipp_transform_threshold"
        self.runJob(program, args)

        args2 = ' -i %s --binaryOperation dilation --size 1' % (self._getExtraPath("mask2.mrc"))
        program2 = "xmipp_transform_morphology"
        self.runJob(program2, args2)

    def subtractionStep(self):
        vol1 = self.vol1.get()
        if self.pdb:
            vol2 = self.outFile
            mask2 = self._getExtraPath("mask2.mrc")
        else:
            vol2 = self.vol2.get().getFileName()
            if self.masks:
                mask2 = self.mask2.get().getFileName()
        resol = self.resol.get()
        sub = self.sub.get()
        iter = self.iter.get()
        program = "xmipp_volume_subtraction"
        args = '-i1 %s -i2 %s -o %s --iter %s' % (vol1.getFileName(), vol2,
                                                  self._getExtraPath("output_volume.mrc"), iter)
        if resol:
            fc = vol1.getSamplingRate()/resol
            args += ' --cutFreq %f --sigma %d' % (fc, self.sigma.get())
        if sub:
            args += ' --sub'
        if self.masks:
            args += ' --mask1 %s --mask2 %s' % (self.mask1.get().getFileName(), mask2)
        self.runJob(program, args)

        # move('commonmask.mrc', join(self._getExtraPath(), 'common_mask.mrc'))
        # move('V1masked.mrc', join(self._getExtraPath(), 'V1_masked.mrc'))
        # move('V2masked.mrc', join(self._getExtraPath(), 'V2_masked.mrc'))
        # for n in range(iter):
        #     move('V2masked_Amp1_%d.mrc' % n, join(self._getExtraPath(), 'V2_Amp1_%d.mrc' % n))
        #     move('V2masked_Amp1_ph2_%d.mrc' % n, join(self._getExtraPath(), 'V2_Amp1_ph2_%d.mrc' % n))
        #     move('V2masked_Amp1_ph2_nonneg_%d.mrc' % n, join(self._getExtraPath(), 'V2_Amp1_ph2_nonneg_%d.mrc' % n))
        # move('maskfilter.mrc', join(self._getExtraPath(), 'maskfilter.mrc'))
        # if resol:
        #     move('V1filter.mrc', join(self._getExtraPath(), 'V1filter.mrc'))
        #     move('V2filter.mrc', join(self._getExtraPath(), 'V2filter.mrc'))

    def createOutputStep(self):
        vol1 = self.vol1.get()
        volume = Volume()
        volume.setSamplingRate(vol1.getSamplingRate())
        origin = Transform()
        ccp4header = headers.Ccp4Header(vol1.getFileName(), readHeader=True)
        shifts = ccp4header.getOrigin()
        origin.setShiftsTuple(shifts)
        volume.setOrigin(origin)
        volume.setFileName(self._getExtraPath("output_volume.mrc"))
        self._defineOutputs(outputVolume=volume)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        if not hasattr(self, 'outputVolume'):
            summary.append("Output volume not ready yet.")
        else:
            summary.append("Input vol 1: %s" % self.vol1.get().getFileName())
            if self.pdb:
                if self.inputPdbData == self.IMPORT_OBJ:
                    summary.append("Input PDB File: %s" % self.pdbObj.get().getFileName())
                else:
                    summary.append("Input PDB File: %s" % self.pdbFile.get())
                summary.append("Mask 2 generated")
            else:
                summary.append("Input 2: volume %s" % self.vol2.get().getFileName())
            if self.masks:
                summary.append("Input mask 1: %s" % self.mask1.get().getFileName())
                summary.append("Input mask 2: %s" % self.mask2.get().getFileName())
            if self.resol.get() != 0:
                summary.append("Subtraction at resolution %d A" % self.resol.get())
            if self.sub:
                summary.append("Output volume: difference")
            else:
                summary.append("Output volume: volume 2 modified")

        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputVolume'):
            methods.append("Output volume not ready yet.")
        else:
            if self.sub:
                if self.pdb:
                    vol2 = ' generated from pdb'
                else:
                    vol2 = self.vol2.get().getFileName()
                methods.append("Volume %s subtracted from volume %s" % (vol2, self.vol1.get().getFileName()))
            else:
                methods.append("Volume %s modified to assimilate it to volume %s" % (self.vol2.get().getFileName(),
                                                                                     self.vol1.get().getFileName()))
            if self.resol.get() != 0:
                methods.append(" at resolution %d A" % self.resol.get())

        return methods

    # --------------------------- UTLIS functions --------------------------------------------
    def _getPdbFileName(self):
        if self.inputPdbData == self.IMPORT_OBJ:
            return self.pdbObj.get().getFileName()
        else:
            return self.pdbFile.get()

    def _getVolName(self):
        return self._getExtraPath(replaceBaseExt(self._getPdbFileName(), "vol"))
