# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *           Federico P. de Isidro-Gomez
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
from pyworkflow.protocol.params import PointerParam, BooleanParam, IntParam, FloatParam, EnumParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pwem import emlib
from pwem.protocols import EMProtocol
from xmipp3.convert import writeSetOfParticles, readSetOfParticles
from pyworkflow import PROD, UPDATED

OUTPUT_MRCS = "output_particles.mrcs"
OUTPUT_XMD = "output_particles.xmd"
CITE = 'Fernandez-Gimenez2023a'

class XmippProtSubtractProjectionBase(EMProtocol):
    """ Helper class that contains some Protocol utilities methods
    used by both  XmippProtSubtractProjection and XmippProtBoostParticles."""
    _devStatus = PROD

    # --------------------------- DEFINE param functions --------------------------------------------
    @classmethod
    def _defineParams(cls, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', 
                      PointerParam, 
                      pointerClass='SetOfParticles', 
                      label="Particles ",
                      help='Specify a SetOfParticles')
        form.addParam('vol', 
                      PointerParam, 
                      pointerClass='Volume', 
                      label="Reference volume ",
                      help='Specify a volume.')
        form.addParam('maskOption',
                EnumParam,
                choices=['Circular mask', 'Protein mask'],
                default=0,
                label='Mask',
                help='Mask to be applied to particles before subtraction. Pixels out to mask are ignored in the analysis:\n'
                     '- Circular mask: circular mask is applied to every particle.\n'
                     '- Protein mask: specimen mask indicating the region of the map that must be conisdered in the analysis.')
        form.addParam('cirmaskrad', 
                      FloatParam, 
                      label="Circular mask radius: ", 
                      default=-1, 
                      expertLevel=LEVEL_ADVANCED,
                      help='Radius of the circular mask to avoid edge artifacts. '
                           'If -1 it is half the X dimension of the input particles',
                      condition='maskOption==0')
        form.addParam('mask', 
                      PointerParam, 
                      pointerClass='VolumeMask', 
                      label='Mask ', 
                      allowsNull=True,
                      help='Specify a 3D mask for the region of the input volume that must be considered in the analysis.',
                      condition='maskOption==1')
        form.addParam('ignoreCTF',
                      BooleanParam,
                      label="Ignore CTF",
                      default=False,
                      help='Do not consider CTF in the subtraction. Use if particles have been CTF corrected.')
        form.addParam('resol', 
                      FloatParam, 
                      label="Maximum resolution (A)", 
                      default=-1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Maximum resolution in Angtroms up to which the substraction is calculated. By default (-1) it is '
                           ' set to sampling/sqrt(2).')
        form.addParam('nonNegative',
                      BooleanParam,
                      label="Ignore particles with negative beta0 or R2?: ",
                      default=True,
                      expertLevel=LEVEL_ADVANCED,
                      help='Particles with negative beta0 or R2 will not appear in the output set as they are '
                           'considered bad particles. Moreover, negative betas will not contribute to mean beta if '
                           '"mean" option is selected')
        form.addParam('realSpaceProjection', 
                      EnumParam,
                      choices=['Fourier ', 'Real space'],
                      default=0,
                      label='Projector',
                      help='Projector for the input volume (mask is always projected in real space):\n'
                           '- Fourier: faster but more artifact prone.\n'
                           '- Real space: slower but more accurate.')
        form.addParam('sigma', 
                      FloatParam, 
                      label="Decay of the filter (sigma): ", 
                      default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help='Decay of the filter (sigma) to smooth the mask transition',
                      condition='realSpaceProjection==0')
        form.addParam('pad', 
                      IntParam, 
                      label="Fourier padding factor: ", 
                      default=2, 
                      expertLevel=LEVEL_ADVANCED,
                      help='The volume is zero padded by this factor to produce projections',
                      condition='realSpaceProjection==0')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertSubSteps()
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputSet)
        readSetOfParticles(self._getExtraPath(OUTPUT_XMD), outputSet,
                           extraLabels=[emlib.MDL_SUBTRACTION_R2,
                                        emlib.MDL_SUBTRACTION_BETA0,
                                        emlib.MDL_SUBTRACTION_BETA1,
                                        emlib.MDL_SUBTRACTION_B])
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputSet, outputSet)


class XmippProtSubtractProjection(XmippProtSubtractProjectionBase):
    """ This protocol computes the subtraction between particles and a reference volume, by computing its projections with the same angles that input particles have. Then, each particle and the correspondent projection of the reference volume are numerically adjusted and subtracted using a mask which denotes the region to keep. """

    _label = 'subtract projection'
    INPUT_PARTICLES = "input_particles.xmd"

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        XmippProtSubtractProjectionBase._defineParams(form)
        form.addParam('maskRoi', PointerParam, pointerClass='VolumeMask', label='ROI mask ', allowsNull=True,
                      help='Specify a 3D mask for the region of the input volume that you want to keep or subtract, '
                           'avoiding masks with 1s in background. If no mask is given, the subtraction is performed in'
                           ' whole images.')
        form.addParam('subtract', EnumParam, default=0, choices=["Keep", "Subtract"], display=EnumParam.DISPLAY_HLIST,
                      label="Mask contains the part to ")
        form.addParallelSection(threads=0, mpi=4)

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
        args = '-i %s --ref %s -o %s --sampling %f --max_resolution %f --padding %f ' \
               '--sigma %d --save %s --save_metadata_stack %s --keep_input_columns ' % \
               (self._getExtraPath(self.INPUT_PARTICLES), fnVol, self._getExtraPath(OUTPUT_MRCS),
                vol.getSamplingRate(), self.resol.get(), self.pad.get(), self.sigma.get(),
                self._getExtraPath(), self._getExtraPath(OUTPUT_XMD))
        
        if self.maskOption.get() == 0:
            args += " --cirmaskrad %d " % self.cirmaskrad.get()
        else:
            fnMask = self.mask.get().getFileName()
            if fnMask.endswith('.mrc'):
                fnMask += ':mrc'
            args += " --mask %s" % fnMask

        maskRoi = self.maskRoi.get()
        if maskRoi is not None:
            fnMaskRoi = maskRoi.getFileName()
            if fnMaskRoi.endswith('.mrc'):
                fnMaskRoi += ':mrc'
            args += ' --mask_roi %s' % fnMaskRoi
        
        if self.nonNegative.get():
            args += ' --nonNegative'
        
        if self.subtract.get():
            args += ' --subtract'
        
        if self.realSpaceProjection.get() == 1:
            args += ' --realSpaceProjection'

        if self.ignoreCTF.get():
            args += ' --ignoreCTF'
        
        self.runJob("xmipp_subtract_projection", args)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.inputParticles.get().getFirstItem()
        vol = self.vol.get()
        mask = self.mask.get()
        if part.getDim()[0] != vol.getDim()[0]:
            errors.append("Input particles and volume should have same X and Y dimensions")
        if round(part.getSamplingRate(), 2) != round(vol.getSamplingRate(), 2):
            errors.append("Input particles and volume should have same sampling rate")
        if mask:
            if round(vol.getSamplingRate(), 2) != round(mask.getSamplingRate(), 2):
                errors.append("Input volume and mask should have same sampling rate")
            if vol.getDim() != mask.getDim():
                errors.append("Input volume and mask should have same dimensions")
        if self.resol.get() == 0:
            errors.append("Resolution (angstroms) should be bigger than 0")
        return errors

    def _warnings(self):
        part = self.inputParticles.get().getFirstItem()
        if part.getDim()[0] > 750 or part.getDim()[1] > 750:
            return ["Particles are quite big, consider to change 'pad=1' (advanced parameter) in order to save RAM "
                    "(even if your RAM is big)."]

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
        form.addParallelSection(threads=0, mpi=4)

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
        args = '-i %s --ref %s -o %s --sampling %f --max_resolution %f --padding %f --sigma %d ' \
               '--cirmaskrad %d --boost --save %s --save_metadata_stack %s --keep_input_columns '\
               % (self._getExtraPath(self.INPUT_PARTICLES), fnVol, self._getExtraPath(OUTPUT_MRCS),
                  vol.getSamplingRate(), self.resol.get(), self.pad.get(), self.sigma.get(),
                  self.cirmaskrad.get(), self._getExtraPath(), self._getExtraPath(OUTPUT_XMD))

        if self.nonNegative.get():
            args += ' --nonNegative'
            
        if self.ignoreCTF.get():
            args += ' --ignoreCTF'
            
        self.runJob("xmipp_subtract_projection", args)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.inputParticles.get().getFirstItem()
        vol = self.vol.get()
        if part.getDim()[0] != vol.getDim()[0]:
            errors.append("Input particles and volume should have same X and Y dimensions")
        if round(part.getSamplingRate(), 2) != round(vol.getSamplingRate(), 2):
            errors.append("Input particles and volume should have same sampling rate")
        if self.resol.get() == 0:
            errors.append("Resolution (angstroms) should be bigger than 0")
        return errors

    def _warnings(self):
        part = self.inputParticles.get().getFirstItem()
        if part.getDim()[0] > 750 or part.getDim()[1] > 750:
            return ["Particles are quite big, consider to change 'pad=1' (advanced parameter) in order to save RAM "
                    "(even if your RAM is big)."]

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
