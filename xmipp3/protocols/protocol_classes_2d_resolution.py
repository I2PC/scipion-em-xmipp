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
from skimage.metrics import structural_similarity as ssim



OUTPUT_PARTICLES = "outputParticles"
OUTPUT_CLASSES = "outputClasses"
OUTPUT_CLASSES_DISCARDED = 'outputClasses_discarded'
OUTPUT_MICROGRAPHS = "outputMicrographs"
DISCARDED = 'discarded'

class XmippProtCL2DResolution(ProtAnalysis2D):
    """ Estimate 2D average resolution ."""

    # Todo: - Particles to have a resolution value
    #   - Micrographs to have an average resolution value

    _label = '2D classes resolution'
    _devStatus = NEW
    _possibleOutputs = {
        OUTPUT_CLASSES: SetOfClasses2D,
        OUTPUT_CLASSES_DISCARDED: SetOfClasses2D
    }

    outputDict = {}

    # _possibleOutputs = {OUTPUT_PARTICLES:SetOfParticles,
    #                     OUTPUT_CLASSES:SetOfClasses2D,
    #                     OUTPUT_MICROGRAPHS:SetOfMicrographs}

    def __init__(self, **args):
        ProtAnalysis2D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL
        self.resolDict = {}
        self._imh = ImageHandler()
        self.goodList = []
        self.badList = []

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', params.PointerParam,
                      label="Input 2D classes",
                      important=True, pointerClass='SetOfClasses2D',
                      help='Select the input classes to be mapped.')
        form.addParam('limitResol', params.FloatParam,
                      label="Limit resolution (A)",
                      default=10,
                      help='This value will reject those classes that are above the limit resolution.')
        form.addParam('proportion', params.FloatParam,
                      label="Proportion particles",
                      default=1,
                      help='This option allows to compute the 2D Average resolution with only a percentage of the'
                           ' particles. Write the proportion you want to use. If 1 it will use all the particles.')
        form.addParam('useWiener', params.BooleanParam, default=True,
                      label='Apply Wiener filter (ctf correction)?',
                      help='By setting "yes" your particles will be ctf corrected.'
                           'If you choose "no" your particles would not be modified.')
        form.addParam('useLPfilter', params.BooleanParam, default=True,
                      label='Apply low-pass filter?',
                      help='By setting "yes" your particles will be low pass filtered by a limit resolution (A).'
                           'If you choose "no" your particles would not be modified.')
        form.addParam('limitLPF', params.FloatParam,
                      label="Limit resolution low-pass filter (A)",
                      default=6,
                      help='This value will low-pass filter your particles to this resolution.')
        form.addParam('useMask', params.BooleanParam, default=True,
                      label='Apply mask?',
                      help='By setting "yes" your particles will be low pass filtered by a limit resolution (A).'
                           'If you choose "no" your particles would not be modified.')

        # Defining parallel arguments
        form.addParallelSection(threads=4, mpi=1)


    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        resolSteps = []
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
                resolSteps.append(resolStep)

        self._insertFunctionStep(self.createOutputStep, prerequisites=resolSteps)


    # --------------------------- STEPS functions -------------------------------
    def calculateLimitLen(self, numberParticles, proportion):
        if numberParticles < 2:
            self.info('Skipping this class')
            extractLimitLen = None
        else:
            extractLimitLen = int(numberParticles * proportion)
            if extractLimitLen < 2:
                extractLimitLen = 2

        return extractLimitLen


    def _loadEmptyParticleSet(self, classParticles):
        classParticles.loadAllProperties()
        acquisition = classParticles.getAcquisition()
        copyPartSet = self._createSetOfParticles()
        copyPartSet.copyInfo(classParticles)
        copyPartSet.setAcquisition(acquisition)

        return copyPartSet


    def estimateClassResolution(self, particles, classId):
        # TODO: when particles come from .mrcs xmipp fails to work with this format, convert step
        self.info("For class %d the particles to be used to estimate resolution %d" % (classId, len(particles)))
        tmp_path = self._getTmpPath("results_class_%d" %classId)
        output_path = self._getExtraPath("results_class_%d" %classId)

        pixel_size = particles.getFirstItem().getSamplingRate()
        os.mkdir(tmp_path)
        os.mkdir(output_path)

        processed_particles = self.preprocessClassParticles(particles, pixel_size, tmp_path)

        bestIds = self.calculate_best_particles_ssim_scores(processed_particles, classId)

        fnStackEven = os.path.join(tmp_path, "even_aligned_particles_ref%d.mrcs" % classId)
        fnStackOdd = os.path.join(tmp_path, "odd_aligned_particles_ref%d.mrcs" % classId)
        counter_even = 1
        counter_odd = 1
        index = 1

        #best_particle = masked_particles_normalized.getItem("id", bestIds[0]).getImage().getData()
        #plot_average(best_particle)

        for particleId in bestIds:
            masked_particle = processed_particles.getItem("id", particleId)
            if not self.useMask.get():
                transform = masked_particle.getTransform()
            else:
                transform = None

            if index%2 == 0:
                self._imh.convert(masked_particle, (counter_even, fnStackEven), transform=transform)
                counter_even += 1
            else:
                self._imh.convert(masked_particle, (counter_odd, fnStackOdd), transform=transform)
                counter_odd += 1

            index += 1

        average_even = self._imh.computeAverage(fnStackEven)
        average_even.write(os.path.join(output_path, "even_average_ref%d.mrc" % classId))
        average_odd = self._imh.computeAverage(fnStackOdd)
        average_odd.write(os.path.join(output_path, "odd_average_ref%d.mrc" % classId))

        frc, resolutions, digFreq = compute_fsc_and_resolutions(average_even.getData(), average_odd.getData(), pixel_size)

        resolution_limit = estimate_resolution_frc(resolutions, frc, threshold=0.143)

        plot_frc(fscglob=frc, digFreq=digFreq, sampling=pixel_size, threshold=0.143,
                 resolution_limit=resolution_limit, directory=output_path)

        self.resolDict[classId] = resolution_limit


    def preprocessClassParticles(self, particles, pixel_size, output_dir):
        inputStk = os.path.join(output_dir, 'imagesInput.xmd')
        writeSetOfParticles(particles, inputStk)
        row = getFirstRow(inputStk)
        hasCTF = row.containsLabel(emlib.MDL_CTF_MODEL) or row.containsLabel(emlib.MDL_CTF_DEFOCUSU)

        if self.useWiener.get():
            if hasCTF:
                corrected_particles_fn = self.correctWiener(inputStk, pixel_size, output_dir)
            else:
                self.info('Cannot do the wiener ctf correction, the input particles do not have a ctf associated.')
        else:
            corrected_particles_fn = inputStk

        if self.useLPfilter.get():
            lpf_fn = self.lowPassFilter(corrected_particles_fn, pixel_size, output_dir)
        else:
            lpf_fn = corrected_particles_fn

        if self.useMask.get():
            dim = particles.getDimensions()[0]
            masked_particles_fn = self.applyMask(lpf_fn, output_dir, dim)
        else:
            masked_particles_fn = lpf_fn

        normalized_particles_fn = self.normalized_particles(masked_particles_fn, output_dir, dim)

        tmpDir = self._getTmpPath(os.path.basename(output_dir))
        os.mkdir(tmpDir)
        fnTmpSqlite = os.path.join(tmpDir, "particlesTmp.sqlite")
        processedParticles = SetOfParticles(filename=fnTmpSqlite)
        processedParticles.copyInfo(particles)
        readSetOfParticles(normalized_particles_fn, processedParticles)

        return processedParticles


    def correctWiener(self, inputStk, pixel_size, directory_ref):
        self.info('CTF correction in progress ...')
        fnCorrectedStk = os.path.join(directory_ref, 'corrected_particles.stk')
        fnCorrected = os.path.join(directory_ref, 'corrected_particles.xmd')
        args = (" -i %s -o %s --save_metadata_stack %s --pad 2 --wc -1.0 --sampling_rate %f --keep_input_columns" %
               (inputStk, fnCorrectedStk, fnCorrected, pixel_size))

        self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())

        return fnCorrected


    def lowPassFilter(self, inputFn, pixel_size, directory_ref):
        self.info('Low pass filter in progress ...')
        fnLPFStk = os.path.join(directory_ref, 'lpf_particles.stk')
        fnLPF = os.path.join(directory_ref, 'lpf_particles.xmd')

        highFreq = pixel_size / self.limitLPF.get()  # the max resolution 2D classification algorithms normally use
        freqDecay = pixel_size / 100  # default parameter in filter particles protocol

        args = (" -i %s --fourier low_pass %f %f -o %s --save_metadata_stack %s --keep_input_columns" %
                (inputFn, highFreq, freqDecay, fnLPFStk, fnLPF))
        # trainingFn has not the metadata is just the images, use the md if you want to keep metadata
        self.runJob("xmipp_transform_filter", args, numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())

        return fnLPF


    def applyMask(self, inputFn, directory_ref, dim):
        self.info('Applying mask in progress ...')
        fnMaskedStk = os.path.join(directory_ref, 'masked_particles.stk')
        fnMasked = os.path.join(directory_ref, 'masked_particles.xmd')

        args = (" -i %s --mask gaussian %f --substitute 0 -o %s --save_metadata_stack %s --keep_input_columns" %
                (inputFn, int(dim/2), fnMaskedStk, fnMasked))

        self.runJob("xmipp_transform_mask", args, numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())

        return fnMaskedStk  # When the mask is applied the transformation matrix is applied


    def normalized_particles(self, inputFn, directory_ref, dim):
        self.info('Normalizing particles in progress ...')
        fnNormalizedStk = os.path.join(directory_ref, 'normalized_particles.stk')
        fnNormalized = os.path.join(directory_ref, 'nromalized_particles.xmd')

        args = (" -i %s -o %s --save_metadata_stack %s --keep_input_columns --method NewXmipp --background circle %d" %
               (inputFn, fnNormalizedStk, fnNormalized, int(dim / 2)))

        self.runJob("xmipp_transform_normalize", args, numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())

        return fnNormalizedStk


    def calculate_best_particles_ssim_scores(self, particles, classId):
        classRepresentative = self.inputClasses.get().getItem("id", classId).getRepresentative().getImage().getData()
        classRepresentative_normalized = z_normalize(classRepresentative)

        particles_scores = {}
        for particle in particles.iterItems(orderBy='id', direction='ASC'):
            img_particle = particle.clone().getImage().getData()
            # plot_average(img_particle)
            ssim_score = compute_ssim(classRepresentative_normalized, img_particle)
            particles_scores[particle.getObjId()] = ssim_score

        # plot_histogram(masked_particles_scores.values())
        bestIds = getBestSsimScoresIds(particles_scores)

        return bestIds


    def arrangeFRC_and_frcGlobal(self, FT1, FT2, mapSize, sampling):

        def defineFrequencies(mapSize):
            '''defines the spatial frequencies present in the Fourier transforms of the 2D images.'''
            freq = np.fft.fftfreq(mapSize) # Generates the frequency components for the given image size.
            fx, fy = np.meshgrid(freq, freq)
            freqMap = np.sqrt(np.multiply(fx, fx) + np.multiply(fy, fy)) # The combined frequency map, representing the magnitude of the spatial frequencies.
            candidates = freqMap <= 0.5 # A boolean mask identifying frequency components below the Nyquist limit

            return freqMap, candidates, fx[candidates], fy[candidates]

        freqMap, candidates, fx, fy = defineFrequencies(mapSize)
        # filters the Fourier transforms and the frequency map based on the valid frequency candidates identified earlier.
        idxFreq = np.round(freqMap * mapSize)  ##.astype(int)
        FT1_vec = FT1[candidates]
        FT2_vec = FT2[candidates]
        freqMap = freqMap[candidates]
        idxFreq = idxFreq[candidates]
        # Computes the correlation numerator and denominators required for the FSC/FRC calculation
        Nfreqs = round(mapSize / 2) # The number of unique frequencies up to the Nyquist limit.
        num = np.real(np.multiply(FT1_vec, np.conjugate(FT2_vec))) # Numerator of the FRC

        den1 = np.absolute(FT1_vec) ** 2  # denominator auto-correlation FT1
        den2 = np.absolute(FT2_vec) ** 2  # denominator auto-correlation FT2

        # Summing over frequency bins
        num_dirfsc = np.zeros(Nfreqs)
        den1_dirfsc = np.zeros(Nfreqs)
        den2_dirfsc = np.zeros(Nfreqs)
        fourierIdx = np.arange(0, Nfreqs)
        for i in fourierIdx: # Group the computed values (num, den1, den2) by frequency bin
            auxIdx = (idxFreq == i)
            num_aux = num[auxIdx]
            den1_aux = den1[auxIdx]
            den2_aux = den2[auxIdx]
            num_dirfsc[i] = np.sum(num_aux)
            den1_dirfsc[i] = np.sum(den1_aux)
            den2_dirfsc[i] = np.sum(den2_aux)

        # FRC Calculation
        fscglob = np.divide(num_dirfsc, np.sqrt(np.multiply(den1_dirfsc, den2_dirfsc) + 1e-38))  # final FRC curve
        fscglob[0] = 1  # corresponds to the lowest frequency (DC component).
        digFreq = np.divide(fourierIdx + 1.0, mapSize)  # Represents the discrete frequency bins normalized by the map size.
        resolutions = np.divide(sampling, digFreq)  # Converts these frequency bins to real-space resolution

        return fscglob, resolutions, digFreq


    def createOutputStep(self):
        self.info('The resolution limit for your classes is: %s' %self.resolDict)
        inputClasses = self.inputClasses.get()
        limitResol = self.limitResol.get()

        outClasses = SetOfClasses2D.create(self._getPath())
        outBadClasses = SetOfClasses2D.create(self._getPath(), suffix=DISCARDED)
        outClasses.copyInfo(inputClasses)
        outBadClasses.copyInfo(inputClasses)

        for classId, classResol in self.resolDict.items():
            if classResol <= limitResol:
                self.goodList.append(classId)
            else:
                self.badList.append(classId)

        if len(self.goodList):
            outClasses.appendFromClasses(inputClasses,
                                         filterClassFunc=self._addGoodClass)
            self.outputDict[OUTPUT_CLASSES] = outClasses

        if len(self.badList):
            outBadClasses.appendFromClasses(inputClasses,
                                            filterClassFunc=self._addBadClass)
            self.outputDict[OUTPUT_CLASSES_DISCARDED] = outBadClasses

        self._defineOutputs(**self.outputDict)

        if len(self.goodList):
            self._defineSourceRelation(self.inputClasses, outClasses)

        if len(self.badList):
            self._defineSourceRelation(self.inputClasses, outBadClasses)


    # --------------------------- UTILS functions -----------------------------
    def plot_fsc(self, fsc, directory, pixel_size, threshold=0.143):
        # Handle NaN and Inf values in FSC
        fsc = np.nan_to_num(fsc, nan=0.0, posinf=0.0, neginf=0.0)

        spatial_frequencies = np.linspace(0, 0.5, len(fsc))  # Nyquist frequency is 0.5 cycles/pixel
        resolution_angstroms = 1 / (spatial_frequencies * 2 * pixel_size)  # Convert to 1/?

        # Handle NaN and Inf values in spatial frequencies
        valid_mask = np.isfinite(resolution_angstroms)
        resolution_angstroms = resolution_angstroms[valid_mask]
        fsc = fsc[valid_mask]

        plt.figure(figsize=(10, 6))
        plt.plot(resolution_angstroms, fsc, label='FSC Curve')
        plt.axhline(y=threshold, color='r', linestyle='--', label='0.143 Threshold')
        plt.xlabel('Spatial Frequency (1/?)')
        plt.ylabel('FSC')
        plt.title('Fourier Shell Correlation (FSC) Plot')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, resolution_angstroms.max()-5)
        plt.gca().invert_xaxis()  # Invert x-axis to show higher resolution (smaller ?) on the left
        plt.savefig(os.path.join(directory, "fsc_plot.png"))


    def _addGoodClass(self, item):
        """ Callback function to append only good classes. """
        return False if item.getObjId() not in self.goodList else True


    def _addBadClass(self, item):
        """ Callback function to append only bad classes. """
        return False if item.getObjId() not in self.badList else True


# ------------------------- STATIC METHODS ----------------------------------
def define_frequencies(map_size):
    """
    Define spatial frequencies and filter out those beyond the Nyquist frequency.

    Parameters:
    map_size (int): Size of the 2D Fourier space.

    Returns:
    tuple: A tuple containing:
        - freq_map (ndarray): The frequency map.
        - valid_freqs (ndarray): Boolean array indicating valid frequencies.
        - idx_freq (ndarray): The index of each frequency in the frequency map.
    """
    freq = np.fft.fftfreq(map_size)
    fx, fy = np.meshgrid(freq, freq)
    freq_map = np.sqrt(fx ** 2 + fy ** 2)
    valid_freqs = freq_map <= 0.5  # Nyquist limit
    idx_freq = np.round(freq_map * map_size).astype(int)

    return freq_map, valid_freqs, idx_freq


def filter_fourier_components(FT1, FT2, valid_freqs):
    """
    Filter Fourier transforms by valid frequency components.

    Parameters:
    FT1 (ndarray): Fourier transform of the first image.
    FT2 (ndarray): Fourier transform of the second image.
    valid_freqs (ndarray): Boolean array indicating valid frequencies.

    Returns:
    tuple: Filtered Fourier transform components.
    """
    FT1_vec = FT1[valid_freqs]
    FT2_vec = FT2[valid_freqs]
    return FT1_vec, FT2_vec


def compute_frc_components(FT1_vec, FT2_vec, idx_freq, Nfreqs):
    """
    Compute the numerator and denominators needed for FRC calculation.

    Parameters:
    FT1_vec (ndarray): Filtered Fourier transform of the first image.
    FT2_vec (ndarray): Filtered Fourier transform of the second image.
    idx_freq (ndarray): The index of each frequency in the frequency map.
    Nfreqs (int): Number of unique frequencies up to the Nyquist limit.

    Returns:
    tuple: Numerator and denominators for the FRC calculation.
    """
    num = np.real(FT1_vec * np.conjugate(FT2_vec))
    den1 = np.abs(FT1_vec) ** 2
    den2 = np.abs(FT2_vec) ** 2

    num_dirfsc = np.zeros(Nfreqs)
    den1_dirfsc = np.zeros(Nfreqs)
    den2_dirfsc = np.zeros(Nfreqs)

    for i in range(Nfreqs):
        freq_mask = (idx_freq == i)
        num_dirfsc[i] = np.sum(num[freq_mask])
        den1_dirfsc[i] = np.sum(den1[freq_mask])
        den2_dirfsc[i] = np.sum(den2[freq_mask])

    return num_dirfsc, den1_dirfsc, den2_dirfsc


def calculate_frc(num_dirfsc, den1_dirfsc, den2_dirfsc):
    """
    Calculate the final FSC/FRC values.

    Parameters:
    num_dirfsc (ndarray): Numerator for the FRC calculation.
    den1_dirfsc (ndarray): Denominator from the first image.
    den2_dirfsc (ndarray): Denominator from the second image.

    Returns:
    ndarray: The calculated FSC/FRC values.
    """
    frc = num_dirfsc / np.sqrt(den1_dirfsc * den2_dirfsc + 1e-38)
    frc[0] = 1  # Ensure the first value is set to 1
    return frc


def compute_fsc_and_resolutions(image1, image2, sampling):
    """
    Main function to compute the FSC/FRC and corresponding resolutions.

    Parameters:
    image1 (ndarray): First image.
    image2 (ndarray): Second image.
    sampling (float): Sampling rate in Angstroms per pixel.

    Returns:
    tuple: A tuple containing:
        - frc (ndarray): The calculated FSC/FRC values.
        - resolutions (ndarray): Corresponding resolutions in Angstroms.
        - digFreq (ndarray): The digital frequencies.
    """
    # Compute the Fourier Transforms with numpy
    ft1 = np.fft.fft2(image1)
    ft2 = np.fft.fft2(image2)
    map_size = np.shape(image1)[0]

    # Define frequencies and filter valid ones
    freq_map, valid_freqs, idx_freq = define_frequencies(map_size)

    # Filter Fourier components by valid frequencies
    FT1_vec, FT2_vec = filter_fourier_components(ft1, ft2, valid_freqs)
    idx_freq = idx_freq[valid_freqs]

    # Number of unique frequencies
    Nfreqs = map_size // 2

    # Compute FRC components (numerator and denominators)
    num_dirfsc, den1_dirfsc, den2_dirfsc = compute_frc_components(FT1_vec, FT2_vec, idx_freq, Nfreqs)

    # Calculate FRC
    frc = calculate_frc(num_dirfsc, den1_dirfsc, den2_dirfsc)

    # Calculate corresponding resolutions and digital frequencies
    digFreq = (np.arange(Nfreqs) + 1) / map_size
    resolutions = sampling / digFreq

    return frc, resolutions, digFreq


def estimate_resolution_frc(resolutions, frc, threshold):
    # Find the intersection point with the threshold
    cross_point = np.where(frc <= threshold)[0]
    if len(cross_point) > 0:
        cross_index = cross_point[0]

        # Interpolate to find a more precise intersection if needed
        if cross_index > 0:
            x1, x2 = resolutions[cross_index - 1], resolutions[cross_index]
            y1, y2 = frc[cross_index - 1], frc[cross_index]
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            resolution_at_threshold = (threshold - intercept) / slope
        else:
            resolution_at_threshold = resolutions[cross_index]

        print(f"The resolution at FRC = {threshold} is approximately {resolution_at_threshold:.2f} ?")
    else:
        resolution_at_threshold = None
        print(f"The FSC curve does not cross the threshold of {threshold}.")

    return resolution_at_threshold


def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=img1.max() - img1.min())


def plot_frc(fscglob, digFreq, sampling, threshold, resolution_limit, directory):
    def formatFreq(value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999.
        if value:
            inv = 1 / value
        return "1/%0.2f" % inv

    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatFreq))
    ax.set_ylim([-0.1, 1.1])
    ax.plot(digFreq / sampling, fscglob, label='FRC Curve')
    plt.xlabel('Resolution ($A^{-1}$)')
    plt.ylabel('FRC (a.u)')
    plt.title('FRC curve')

    # Plot the threshold line
    plt.axhline(y=threshold, color='k', linestyle='dashed', label=f'Threshold = {threshold}')

    if resolution_limit:
        plt.axvline(x=1 / resolution_limit, color='r', linestyle='dotted',
                    label=f'Resolution = {resolution_limit:.2f} ?')

    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(directory, "frc_plot.png"))
    # plt.show()


def plot_average(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()


def plot_histogram(scores):
    plt.figure()
    plt.hist(scores)
    plt.show()


def getBestSsimScoresIds(resultDict):
    # Step 1: Sort the dictionary by SSIM scores (values) in descending order (biggest scores first)
    sorted_particles = sorted(resultDict.items(), key=lambda x: x[1], reverse=True)

    # Step 2: Keep the best 50% of particles
    half_count = int(len(sorted_particles) * 0.7)
    best_particles = sorted_particles[:half_count]

    # Step 3: Extract the particle IDs and sort them in ascending order
    # best_particle_ids = sorted([particle_id for particle_id, score in best_particles])
    best_particle_ids = [particle_id for particle_id, score in best_particles]

    return best_particle_ids


def z_normalize(image):
    """Normalize the image using z-normalization."""
    return (image - np.mean(image)) / np.std(image)