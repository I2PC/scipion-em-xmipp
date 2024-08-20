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
import pyworkflow.protocol.params as param
from networkx.algorithms.isomorphism.matchhelpers import tmpdoc
from pwem.emlib.image import ImageHandler
import pwem.emlib.metadata as md
from pwem import emlib
from pwem.protocols import ProtAnalysis2D
from pwem.objects import SetOfParticles, SetOfAverages, SetOfClasses2D, SetOfMicrographs
from pyworkflow import BETA, UPDATED, NEW, PROD
import pyworkflow.protocol.params as params
from pwem.emlib.metadata import iterRows, getFirstRow

from xmipp3.convert import readSetOfParticles, writeSetOfParticles
import matplotlib.pyplot as plt



OUTPUT_PARTICLES = "outputParticles"
OUTPUT_CLASSES = "outputClasses"
OUTPUT_MICROGRAPHS = "outputMicrographs"

class XmippProtCL2DResolution(ProtAnalysis2D):
    """ Estimate the 2D average resolution ."""

    _label = '2D classes resolution'
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
        form.addParam('inputClasses', param.PointerParam,
                      label="Input 2D classes",
                      important=True, pointerClass='SetOfClasses2D',
                      help='Select the input classes to be mapped.')
        form.addParam('proportion', param.FloatParam,
                      label="Proportion particles",
                      default=1,
                      help='This option allows to compute the 2D Average resolution with only a percentage of the'
                           ' particles. Write the proportion you want to use. If 1 it will use all the particles.')
        form.addParam('useWiener', param.BooleanParam, default=True,
                      label='Apply Wiener filter (ctf correction)?',
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

                resolStep = self._insertFunctionStep(self.estimateClassResolution, classParticles, classId, prerequisites=[])
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

    def estimateClassResolution(self, particles, classId):
        print("For class %d the particles to be used to estimate resolution %d" % (classId, len(particles)))
        directory_ref = self._getExtraPath("results_class_%d" %classId)
        pixel_size = particles.getFirstItem().getSamplingRate()
        os.mkdir(directory_ref)

        if self.useWiener.get():
            corrected_particles = self.correctWiener(particles, pixel_size, directory_ref)
        else:
            corrected_particles = particles

        fnStackEven = os.path.join(directory_ref, "even_aligned_particles_ref%d.mrcs" % classId)
        fnStackOdd = os.path.join(directory_ref, "odd_aligned_particles_ref%d.mrcs" % classId)
        counter_even = 1
        counter_odd = 1

        index = 1
        for corrected_particle in corrected_particles.iterItems(orderBy='id', direction='ASC'):
            transform = corrected_particle.getTransform()

            if index%2 == 0:
                self._imh.convert(corrected_particle, (counter_even, fnStackEven), transform=transform)
                counter_even += 1
            else:
                self._imh.convert(corrected_particle, (counter_odd, fnStackOdd), transform=transform)
                counter_odd += 1

            index += 1

        average_even = self._imh.computeAverage(fnStackEven)
        average_even.write(os.path.join(directory_ref, "even_average_ref%d.mrc" % classId))
        average_odd = self._imh.computeAverage(fnStackOdd)
        average_odd.write(os.path.join(directory_ref, "odd_average_ref%d.mrc" % classId))

        fsc_2d = self.computeFSC(average_even, average_odd, directory_ref, classId)
        fsc_1d = self.radial_average_2d(fsc_2d)

        self.plot_fsc(fsc=fsc_1d, directory=directory_ref, pixel_size=pixel_size)

    def correctWiener(self, particles, pixel_size, directory_ref):
        self.info('CTF correction in progress ...')
        inputStk = os.path.join(directory_ref, 'imagesInput.xmd')
        fnCorrectedStk = os.path.join(directory_ref, 'corrected_particles.stk')
        fnCorrected = os.path.join(directory_ref, 'corrected_particles.xmd')
        writeSetOfParticles(particles, inputStk)

        row = getFirstRow(inputStk)
        hasCTF = row.containsLabel(emlib.MDL_CTF_MODEL) or row.containsLabel(emlib.MDL_CTF_DEFOCUSU)

        if hasCTF:
            # args = (" -i %s -o %s --save_metadata_stack %s --sampling_rate %f --keep_input_columns --correct_envelope" %
            #         (inputStk, fnCorrectedStk, fnCorrected, pixel_size))
            args = (" -i %s -o %s --save_metadata_stack %s --sampling_rate %f --keep_input_columns" %
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

    def computeFSC(self, even_image, odd_image, directory, class_id):
        # Compute the Fourier Transform
        from scipy.fftpack import fft2, ifft2, fftshift # Scipy or numpy
        ft1 = fft2(even_image.getData())
        ft2 = fft2(odd_image.getData())
        # Compute the Fourier Transforms
        #ft1 = np.fft.fft2(even_image.getData())
        #ft2 = np.fft.fft2(odd_image.getData())

        # Compute the cross - correlation
        # cross_correlation = fftshift(np.real(ifft2(ft1 * np.conj(ft2))))
        cross_correlation = fftshift(np.real(ft1 * np.conj(ft2)))

        # Compute the auto-correlations
        # auto_corr1 = fftshift(np.real(ifft2(ft1 * np.conj(ft1))))
        # auto_corr2 = fftshift(np.real(ifft2(ft2 * np.conj(ft2))))
        auto_corr1 = fftshift(np.real(ft1 * np.conj(ft1)))
        auto_corr2 = fftshift(np.real(ft2 * np.conj(ft2)))

        # Compute FSC in 2D
        fsc_2d = np.abs(cross_correlation) / np.sqrt(np.abs(auto_corr1) * np.abs(auto_corr2)) # Point to point

        return fsc_2d

    def radial_average_2d(self, image):
        y, x = np.indices((image.shape))
        center = np.array([x.max() / 2, y.max() / 2])
        r = np.hypot(x - center[0], y - center[1])

        # Radial profile
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), image.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    def createOutputStep(self):
        # todo: put a wait here cause it is ending before the steps
        print('FIIINISHIIIIING.-------------------------------------')

    def plot_fsc(self, fsc, directory, pixel_size, threshold=0.143):
        # Handle NaN and Inf values in FSC
        fsc = np.nan_to_num(fsc, nan=0.0, posinf=0.0, neginf=0.0)

        spatial_frequencies = np.linspace(0, 0.5, len(fsc))  # Nyquist frequency is 0.5 cycles/pixel
        resolution_angstroms = 1 / (spatial_frequencies * 2 * pixel_size)  # Convert to 1/Å

        # Handle NaN and Inf values in spatial frequencies
        valid_mask = np.isfinite(resolution_angstroms)
        resolution_angstroms = resolution_angstroms[valid_mask]
        fsc = fsc[valid_mask]

        plt.figure(figsize=(10, 6))
        plt.plot(resolution_angstroms, fsc, label='FSC Curve')
        plt.axhline(y=threshold, color='r', linestyle='--', label='0.143 Threshold')
        plt.xlabel('Spatial Frequency (1/Å)')
        plt.ylabel('FSC')
        plt.title('Fourier Shell Correlation (FSC) Plot')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, resolution_angstroms.max()-5)
        plt.gca().invert_xaxis()  # Invert x-axis to show higher resolution (smaller Å) on the left
        plt.savefig(os.path.join(directory, "fsc_plot.png"))

    def smooth_curve(self, fsc, window_size=5):
        window = np.ones(window_size) / window_size
        return np.convolve(fsc, window, mode='same')


# x_dim, y_dim, _, _ = even_image.getDimensions()
        # even_psd = even_image.computePSD(0.4, x_dim, y_dim, 1)
        # even_psd.convertPSD()
        # even_psd.write(os.path.join(directory, "even_psd_ref%d.mrc" % class_id))
        #
        # odd_psd = odd_image.computePSD(0.4, x_dim, y_dim, 1)
        # odd_psd.convertPSD()
        # odd_psd.write(os.path.join(directory, "odd_psd_ref%d.mrc" % class_id))




