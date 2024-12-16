# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Federico P. de Isidro-Gomez
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
from pyworkflow.protocol.params import PointerParam, IntParam, EnumParam
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pwem.protocols import EMProtocol
from xmipp3.convert import writeSetOfParticles, readSetOfParticles
from pyworkflow import NEW


class XmippProtClassifyPartialOccupancy(EMProtocol):
    """ This protocol classify a set of particles based on the local density in the region defined by the provided mask. """

    _label = 'classify partial occupancy'
    INPUT_PARTICLES = "input_particles.xmd"
    OUTPUT_PARTICLES = "output_particles.xmd"
    OROOT_PREFIX = "subtracted_part"
    _devStatus = NEW

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputParticles',
                      PointerParam,
                      pointerClass='SetOfParticles',
                      label="Particles ",
                      help='Specify a SetOfParticles')
        
        form.addParam('vol',
                      PointerParam,
                      pointerClass='Volume',
                      label="Weight volume ",
                      help='Volume wheighting the local consistency of information of the sample reconstruction.')
        
        form.addParam('realSpaceProjection',
                      EnumParam,
                      choices=['Fourier ', 'Real space'],
                      default=0,
                      label='Projector',
                      help='Projector for the input volume (mask is always projected in real space):\n'
                           '- Fourier: faster but more artifact prone.\n'
                           '- Real space: slower but more accurate.')
        
        form.addParam('pad',
                      IntParam,
                      label="Fourier padding factor: ",
                      default=2,
                      expertLevel=LEVEL_ADVANCED,
                      help='The volume is zero padded by this factor to produce projections',
                      condition='realSpaceProjection==0')
        
        form.addParam('maskRoi',
                      PointerParam,
                      pointerClass='VolumeMask',
                      label='ROI mask ',
                      help='Specify a 3D mask for the region of the input volume ')
        
        form.addParallelSection(threads=0,
                                mpi=4)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertStep')
        self._insertFunctionStep('subtractionStep')
        self._insertFunctionStep('createOutputStep')


    # --------------------------- STEPS functions --------------------------------------------
    def convertStep(self):
        writeSetOfParticles(self.inputParticles.get(), self._getExtraPath(self.INPUT_PARTICLES))

    def subtractionStep(self):
        vol = self.vol.get().clone()
        fnVol = vol.getFileName()
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'

        maskRoi = self.maskRoi.get()
        fnMaskRoi = maskRoi.getFileName()
        if fnMaskRoi.endswith('.mrc'):
            fnMaskRoi += ':mrc'

        params = {
           "-i":  self._getExtraPath(self.INPUT_PARTICLES),
           "--ref": fnVol,
           "-o": self._getExtraPath(self.OUTPUT),
           "--padding": self.pad.get(),
           "--save": self._getExtraPath(),
           "--oroot": self._getExtraPath(self.OROOT_PREFIX),
           "--mask_roi": fnMaskRoi,
        }

        if self.realSpaceProjection.get() == 1:
            params["--realSpaceProjection"] = ' '

        args = ' '.join(['%s %s' % (k, str(v)) for k, v in params.items()])   
        
        self.runJob("xmipp_subtract_projection", args)

    def createOutputStep(self):
        inputSet = self.inputParticles.get()
        outputSet = self._createSetOfParticles()
        outputSet.copyInfo(inputSet)
        readSetOfParticles(self._getExtraPath(self.OUTPUT), outputSet)
        self._defineOutputs(outputParticles=outputSet)
        self._defineSourceRelation(inputSet, outputSet)


    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
        part = self.inputParticles.get().getFirstItem()
        vol = self.vol.get()
        maskRoi = self.maskRoi.get()
        if part.getDim()[0] != vol.getDim()[0]:
            errors.append("Input particles and volume should have same X and Y dimensions")
        if round(part.getSamplingRate(), 2) != round(vol.getSamplingRate(), 2):
            errors.append("Input particles and volume should have same sampling rate")
        if round(vol.getSamplingRate(), 2) != round(maskRoi.getSamplingRate(), 2):
            errors.append("Input volume and mask should have same sampling rate")
        if vol.getDim() != maskRoi.getDim():
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
                   (self.vol.get().getFileName(), self.inputParticles.get(), self.maskRoi.get().getFileName())]
        return summary

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputParticles'):
            methods.append("Output particles not ready yet.")
        else:
            methods.append("Input particles splited based on partial occupancy successfully!")
        return methods
