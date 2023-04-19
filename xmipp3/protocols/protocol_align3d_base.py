# **************************************************************************
# *
# * Authors:     Erney Ramirez (eramirez@cnb.csic.es)
# *              Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
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

import os
from math import floor

from pyworkflow.object import Float
from pyworkflow.utils.path import cleanPath, copyFile
from pyworkflow.protocol.params import (Form, PointerParam, FloatParam, BooleanParam,
                                        IntParam, StringParam,
                                        STEPS_PARALLEL, LEVEL_ADVANCED, USE_GPU, GPU_LIST)

from pwem.emlib.image import ImageHandler
from pwem.protocols import ProtReconstruct3D
from pwem.objects import SetOfClasses2D

from pwem import emlib
from xmipp3.convert import (writeSetOfClasses2D, readSetOfVolumes,
                            readSetOfParticles, writeSetOfParticles)
from xmipp3.base import isMdEmpty, isXmippCudaPresent

class Alignment3DModelBase:
    def __init__(self, baseDir: str) -> None:
        self._baseDir = baseDir
    
    def setVolume(self, volumeFn: str):
        self._volumeFn = volumeFn

    def setExperimental(self, experimentalFn: str):
        self._experimentalFn = experimentalFn

    def createGallery(self, step: float, sym: str):
        perturb = math.sin(math.radians(step)) / 4
        
        args = []
        args += ['-i', self._volumeFn]
        args += ['-o', self._getClassGalleryStackFilename(iteration, cls, repetition)]
        args += ['--sampling_rate', step]
        args += ['--perturb', perturb]
        args += ['--sym', sym]
        
        if False: # TODO speak with coss
            args += ['--compute_neighbors']
            args += ['--angular_distance', -1]    
            args += ['--experimental_images', self._getIterationInputParticleMdFilename(iteration)]

        self.runJob('xmipp_angular_project_library', args)
    
    def train():
        pass

    def align():
        pass

class XmippProtAlign3DBase(ProtReconstruct3D):
    def __init__(self, model: Alignment3DModelBase, hasTraining=False, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._hasTraining = hasTraining
        
    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form: Form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label='Particles', important=True,
                      pointerClass='SetOfParticles',
                      help='Input particle set')
        form.addParam('considerInputAlignment', BooleanParam, label='Consider previous alignment',
                      default=True,
                      help='Consider the alignment of input particles')
        form.addParam('correctCtf', BooleanParam, label='Correct CTF',
                      default=True,
                      help='Correct the CTf of the particles using a Wiener filter')
        form.addParam('inputVolumes', PointerParam, label='Initial volumes', important=True,
                      pointerClass='Volume', minNumObjects=1,
                      help='Provide a volume for each class of interest')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        
        form.addSection(label='Training', condition='_hasTraining')

        form.addSection(label='Align')
        
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('correctCtfStep')
        self._insertFunctionStep('createOutputStep')
    
    #--------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        writeSetOfParticles(self.inputParticles.get(), 
                            self._getInputParticleMdFilename())
            
    def correctCtfStep(self):
        particles = self.inputParticles.get()
        
        if particles.correctCtf:
            args = []
            args += ['-i', self._getInputParticleMdFilename()]
            args += ['-o', self._getWienerParticleStackFilename()]
            args += ['--save_metadata_stack', self._getWienerParticleMdFilename()]
            args += ['--keep_input_columns']
            args += ['--sampling_rate', particles.getSamplingRate()]
            args += ['--pad', '2']
            args += ['--wc', '-1.0']
            if particles.isPhaseFlipped():
                args +=  ['--phase_flipped']

            self.runJob('xmipp_ctf_correct_wiener2d', args)
            
        else:
            # TODO When the stack is already in MRC format, simply link
            args = []
            args += ['-i', self._getInputParticleMdFilename()]
            args += ['-o', self._getWienerParticleStackFilename()]
            args += ['--save_metadata_stack', self._getWienerParticleMdFilename()]
            args += ['--keep_input_columns']

            self.runJob('xmipp_image_convert', args, numberOfMpi=1)

    def trainStep(self):
        self._model.train()

    def alignStep(self):
        self._model.align()
        
    def createOutputStep(self):
        partSet = self._createSetOfParticles()             
        EXTRA_LABELS = [
            #emlib.MDL_COST
        ]
         # Fill
        readSetOfParticles(
            self._getAlignedParticlesFilename(),
            partSet,
            extraLabels=EXTRA_LABELS
        )
        partSet.setSamplingRate(self.inputParticles.get().getSamplingRate())       
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(self.inputParticles, partSet)
        
    
    #--------------------------- UTILS functions --------------------------------------------        
    def _getInputParticleMdFilename(self):
        return self._getExtraPath('input_particles.xmd')

    def _getWienerParticleMdFilename(self):
        return self._getExtraPath('input_particles_wiener.xmd')
    
    def _getWienerParticleStackFilename(self):
        return self._getExtraPath('input_particles_wiener.mrcs')
    
    def _getAlignedParticlesFilename(self):
        return self._getExtraPath('aligned_paricles.xmd')
        