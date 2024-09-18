# ******************************************************************************
# *
# * Authors:     Daniel Marchan Torres (da.marchan@cnb.csic.es)
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
# ******************************************************************************
import os.path
import numpy as np
import matplotlib.pyplot as plt

from pwem.emlib.image import ImageHandler
from pwem import emlib
from pwem.protocols import ProtAnalysis2D
from pwem.objects import SetOfParticles, SetOfClasses2D, SetOfMicrographs
from pwem.emlib.metadata import getFirstRow

import pyworkflow.protocol.params as params
from pyworkflow import BETA, NEW

from xmipp3.convert import readSetOfParticles, writeSetOfParticles




OUTPUT_PARTICLES = "outputParticles"
OUTPUT_CLASSES = "outputClasses"
OUTPUT_MICROGRAPHS = "outputMicrographs"

class XmippProtDeepCL2(ProtAnalysis2D):
    """ Estimate the 2D average resolution ."""

    _label = 'deep 2D classification'
    _devStatus = NEW
    _possibleOutputs = {OUTPUT_PARTICLES:SetOfParticles,
                        OUTPUT_CLASSES:SetOfClasses2D,
                        OUTPUT_MICROGRAPHS:SetOfMicrographs}

    def __init__(self, **args):
        ProtAnalysis2D.__init__(self, **args)
        self._classesParticles = {}
        self._imh = ImageHandler()
        self.stepsExecutionMode = params.STEPS_PARALLEL

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', params.PointerParam,
                      label="Input 2D classes",
                      important=True, pointerClass='SetOfClasses2D',
                      help='Select the input classes to be mapped.')
        form.addParam('proportion', params.FloatParam,
                      label="Proportion particles",
                      default=1,
                      help='This option allows to compute the 2D Average resolution with only a percentage of the'
                           ' particles. Write the proportion you want to use. If 1 it will use all the particles.')
        form.addParam('useFilter', params.BooleanParam, default=True,
                      label='Apply Low-Pass Filter?',
                      help='By setting "yes" your particles will be ctf corrected.'
                           'If you choose "no" your particles would not be modified.')

        # Defining parallel arguments
        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        deps = []
        inputClasses = self.inputClasses.get()
        proportion = self.proportion.get()

        for clazz in inputClasses.iterItems(orderBy="id", direction="ASC"):
            numberParticles = len(clazz)
            extractLimitLen = self.calculateLimitLen(numberParticles, proportion)
            extractLimitLen = extractLimitLen - 1 if extractLimitLen % 2 != 0 else extractLimitLen

            classId = clazz.getObjId()
            classParticles = self._loadEmptyParticleSet(clazz)

            if extractLimitLen:
                for particle in clazz.iterItems(orderBy='id', direction='ASC', limit=extractLimitLen):
                    newParticle = particle.clone()
                    classParticles.append(newParticle)

                resolStep = self._insertFunctionStep(self.createTrainingDataset, classParticles, classId, prerequisites=[])
                deps.append(resolStep)

        self._insertFunctionStep(self.createOutputStep, prerequisities=[deps])

    # --------------------------- STEPS functions -------------------------------
    def _loadEmptyParticleSet(self, classParticles):
        classParticles.loadAllProperties()
        acquisition = classParticles.getAcquisition()
        copyPartSet = self._createSetOfParticles()
        copyPartSet.copyInfo(classParticles)
        copyPartSet.setAcquisition(acquisition)

        return copyPartSet

    def calculateLimitLen(self, numberParticles, proportion):
        if numberParticles < 2:
            print('Skipping this class')
            extractLimitLen = None
        else:
            extractLimitLen = int(numberParticles * proportion)
            if extractLimitLen < 2:
                extractLimitLen = 2

        return extractLimitLen

    def createTrainingDataset(self, classParticles, classId):
        directory_ref = self._getExtraPath("results_class_%d" % classId)
        os.mkdir(directory_ref)
        particles_training_fn = os.path.join(directory_ref, 'particles_training.mrcs') # maybe change to .stk
        pixel_size = classParticles.getFirstItem().getSamplingRate()
        self.proprocessParticles(classParticles, pixel_size, directory_ref, particles_training_fn)


    def proprocessParticles(self, particles, pixel_size, directory_ref, particles_training_fn):
        self.info('Preprocess particles in progress ...')
        input_stk = os.path.join(directory_ref, 'imagesInput.xmd')
        writeSetOfParticles(particles, input_stk)
        # Low pass filter images
        output_lpf_fn = os.path.join(directory_ref, 'lpf_particles.mrcs')
        output_lpf_md = os.path.join(directory_ref, 'lpf_particles.xmd')
        self.lowPassFilter(input_stk, output_lpf_fn, output_lpf_md, pixel_size)
        print('using weiner filter')
        # Resize images
        output_lpf_ds_fn = os.path.join(directory_ref, 'lpf_ds_particles.mrcs')
        newSize = 128
        self.resizeParticles(output_lpf_fn, output_lpf_ds_fn, newSize)

    def correctWiener(self, particles, pixel_size, directory_ref):
        self.info('Preprocess particles in progress ...')
        inputStk = os.path.join(directory_ref, 'imagesInput.xmd')
        fnCorrectedStk = os.path.join(directory_ref, 'corrected_particles.stk')
        fnCorrected = os.path.join(directory_ref, 'corrected_particles.xmd')
        writeSetOfParticles(particles, inputStk)

        row = getFirstRow(inputStk)
        hasCTF = row.containsLabel(emlib.MDL_CTF_MODEL) or row.containsLabel(emlib.MDL_CTF_DEFOCUSU)

        if hasCTF:
            args = (" -i %s -o %s --save_metadata_stack %s --pad 2 --wc -1.0 --sampling_rate %f --keep_input_columns" %
                    (inputStk, fnCorrectedStk, fnCorrected, pixel_size))

            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=1)

            tmpDir = self._getTmpPath(os.path.basename(directory_ref))
            os.mkdir(tmpDir)
            fnTmpSqlite = os.path.join(tmpDir, "particlesTmp.sqlite")
            correctedParticles = SetOfParticles(filename=fnTmpSqlite)
            correctedParticles.copyInfo(particles)
            readSetOfParticles(fnCorrected, correctedParticles)
        else:
            self.info('Cannot do the wiener ctf correction, the input particles do not have a ctf associated.')
            correctedParticles = particles

        return correctedParticles

    def lowPassFilter(self, inputFn, outputFn, outPutmdFn, pixel_size):
        #highFreq = pixel_size / (pixel_size * 5) # five times bigger than current one or maybe 15 A is enough
        highFreq = pixel_size / 6 # the max resolution 2D classification algorithms normally use
        freqDecay = pixel_size / 100 # default parameter in filter particles protocol

        args = (" -i %s --fourier low_pass %f %f -o %s --save_metadata_stack %s --keep_input_columns" %
                    (inputFn, highFreq, freqDecay, outputFn, outPutmdFn))
        # trainingFn has not the metadata is just the images, use the md if you want to convert metadata
        self.runJob("xmipp_transform_filter", args, numberOfMpi=1)

    def resizeParticles(self, inputFn, outputFn, newSize):
        args = (" -i %s -o %s --fourier %d" %
                (inputFn, outputFn, newSize))
        self.runJob("xmipp_image_resize", args, numberOfMpi=1)

    def createOutputStep(self):
        pass