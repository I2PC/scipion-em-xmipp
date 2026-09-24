# **************************************************************************
# *
# * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
# *              Pablo Conesa (pconesa@cnb.csic.es)
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

from pyworkflow import VERSION_1_1
from pyworkflow.utils import removeExt
from pyworkflow.protocol.params import (PointerParam, EnumParam, FloatParam,
                                        LEVEL_ADVANCED)

from pwem.protocols import ProtRefine3D
from pwem.objects import Volume
import pwem.emlib.metadata as md

from xmipp3.convert import writeSetOfParticles, xmippToLocation


class XmippProtAddNoise(ProtRefine3D):
    """    
    Given a sets of volumes or particles the protocol adds noise to them 
    The types of noise are Uniform, Student and Gaussian.
    """
    GAUSSIAN_NOISE = 0
    STUDENT_NOISE = 1
    UNIFORM_NOISE = 2
    _lastUpdateVersion = VERSION_1_1
    # --------------------------- DEFINE param functions ------------------------

    def _defineParams(self, form):
        
        form.addParam('noiseType', EnumParam,
                      choices=['Gaussian', 'Student', 'Uniform'],
                      default = 0,
                      label="Noise Type")
        
        form.addParam('gaussianStd', FloatParam, default=0.08, 
                      condition='noiseType == %d' % self.GAUSSIAN_NOISE,
                      label="Standard Deviation", 
                      help='Please, introduce the standard deviation value.'
                      'Mean value can be changed in advanced mode.')
        
        form.addParam('gaussianMean', FloatParam, default=0,
                      expertLevel=LEVEL_ADVANCED,
                      condition='noiseType == %d' % self.GAUSSIAN_NOISE,
                      label="Mean", 
                      help='Please, introduce the mean value (default = 0).')
        
        form.addParam('studentDf', FloatParam, default=1, 
                      condition='noiseType == %d' % self.STUDENT_NOISE,
                      label="Degree of Freedom", 
                      help='Please, introduce the Degree of Freedom.'
                      'Mean value can be changed in advanced mode.')
        
        form.addParam('studentStd', FloatParam, default=0.08, 
                      condition='noiseType == %d' % self.STUDENT_NOISE,
                      label="Standard Deviation", 
                      help='Please, introduce the standard deviation value.'
                      'Mean value can be changed in advanced mode.')
        
        form.addParam('studentMean', FloatParam, default=0,
                      expertLevel=LEVEL_ADVANCED,
                      condition='noiseType == %d' % self.STUDENT_NOISE,
                      label="Mean", 
                      help='Please, introduce the mean value (default = 0).')
        
        form.addParam('uniformMin', FloatParam, default=0, 
                      condition='noiseType == %d' % self.UNIFORM_NOISE,
                      label="Minimum Value", 
                      help='Please, introduce the minimum value. (default = 0)')
        
        form.addParam('uniformMax', FloatParam, default=1, 
                      condition='noiseType == %d' % self.UNIFORM_NOISE,
                      label="Maximum Value", 
                      help='Please, introduce the maximum value (default = 1).')
        form.addParallelSection(threads=1, mpi=1)

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):        
        self.micsFn = self._getPath()
        # Convert input into xmipp Metadata format
        convertId = self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('addNoiseStep')
        self._insertFunctionStep('createOutputStep')

    def _getTypeOfNoise(self):
        if self.noiseType == self.GAUSSIAN_NOISE:
            kindNoise = 'gaussian'
            noiseParams = '%f %f' % (self.gaussianStd, self.gaussianMean)
        if self.noiseType == self.STUDENT_NOISE:
            kindNoise = 'student'
            noiseParams = '%f %f %f' % (self.studentDf, self.studentStd
                                         , self.studentMean)
        if self.noiseType == self.UNIFORM_NOISE:
            kindNoise = 'uniform'
            noiseParams = '%f %f' % (self.uniformMin, self.uniformMax)
        return kindNoise, noiseParams

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        validateMsgs = []
        if self.input and not self.input.hasValue():
            validateMsgs.append('Please provide input volume.')  
        return validateMsgs

    def _summary(self):
        summary = []
        if hasattr(self, 'outputVolume'):
            summary.append("Volume with %s noise has been obtained" % (self.getEnumText("noiseType")))
        elif hasattr(self, 'outputParticles'):
            summary.append("Particles with %s noise has been obtained" % (self.getEnumText("noiseType")))
        elif not hasattr(self, 'outputVolume') or not hasattr(self, 'outputParticles'):
                summary.append("Output not ready yet.")
        return summary
    
    def _methods(self):
        messages = []
        if hasattr(self, 'outputVolume'):
            messages.append('Noisy volume has been obtained')
        elif hasattr(self, 'outputParticles'):
            messages.append('Noisy particles have been obtained')
        return messages
    
    def _citations(self):
        return ['Do not apply']
    
    def getSummary(self):
        summary = []
        summary.append("Particles analyzed:")
        #summary.append("Particles picked: %d" %coordsSet.getSize())
        return "\n"#.join(summary)
    
    
class XmippProtAddNoiseVolumes(XmippProtAddNoise):
    """    
    Given a set of volumes, or a volume the protocol will add noise to them 
    The types of noise are Uniform, Student and Gaussian.

    AI Generated

    ## Overview

    The Add Noise Volume/s protocol creates noisy versions of one or more input
    volumes.

    This protocol is useful for simulation, robustness testing, method
    development, teaching, and validation workflows. By adding controlled noise to
    a map, the user can test how downstream protocols behave when the signal is
    degraded.

    The protocol supports three types of noise:

    - Gaussian noise;
    - Student noise;
    - Uniform noise.

    The output is either a single noisy volume or a set of noisy volumes, depending
    on whether the input is one volume or a set of volumes.

    ## Inputs and General Workflow

    The input can be either a single volume or a set of volumes.

    For each input volume, the protocol runs the Xmipp noise-addition program with
    the selected noise model and parameters. The noisy map is written to a new MRC
    file whose name is derived from the input file name and ends with
    `_Noisy.mrc`.

    The output volume or volume set copies the input metadata and points to the new
    noisy files.

    ## Input Volume/s

    The **Input Volume/s** parameter defines the map or maps to which noise will be
    added.

    If the input is a single volume, the protocol creates one noisy output volume.

    If the input is a set of volumes, the protocol creates a new set containing one
    noisy version of each input volume.

    The protocol does not align, filter, normalize, or otherwise modify the maps
    except for adding the selected noise.

    ## Noise Type

    The **Noise Type** parameter selects the statistical distribution of the noise.

    The available options are:

    **Gaussian** adds normally distributed noise.

    **Student** adds noise following a Student distribution.

    **Uniform** adds uniformly distributed noise between a minimum and maximum
    value.

    The choice depends on the simulation or testing scenario. Gaussian noise is the
    most common general-purpose option. Student noise can generate heavier-tailed
    perturbations. Uniform noise is useful when the user wants bounded random
    values.

    ## Gaussian Noise

    When **Gaussian** noise is selected, the user can define:

    - **Standard Deviation**;
    - **Mean**.

    The standard deviation controls the amplitude of the added noise. Larger values
    produce noisier maps.

    The mean defines the central value of the Gaussian distribution. The default
    mean is 0, meaning that the noise has no average offset.

    Gaussian noise is useful for testing how robust downstream processing is to
    random additive fluctuations.

    ## Student Noise

    When **Student** noise is selected, the user can define:

    - **Degree of Freedom**;
    - **Standard Deviation**;
    - **Mean**.

    The degree of freedom controls the shape of the Student distribution. Low
    values can produce heavier tails, meaning that stronger outlier noise values
    may occur more often than in a Gaussian distribution.

    This option is useful when the user wants to test behavior under more
    outlier-prone noise.

    ## Uniform Noise

    When **Uniform** noise is selected, the user can define:

    - **Minimum Value**;
    - **Maximum Value**.

    Noise values are sampled uniformly between these two limits.

    Uniform noise is useful when the user wants bounded perturbations rather than a
    distribution with long tails.

    ## Output Volume

    If the input is a single volume, the main output is **outputVolume**.

    This output is the noisy version of the input volume. It copies the input
    volume information and points to the generated `_Noisy.mrc` file.

    The output can be used in downstream protocols exactly like any other volume.

    ## Output Volume Set

    If the input is a set of volumes, the protocol creates an output volume set.

    Each output item corresponds to one input volume, but its file name is replaced
    by the generated noisy file.

    The output set preserves the input set information while replacing the volume
    locations with the noisy versions.

    ## Interpreting the Result

    The output should be interpreted as the same input map with additive random
    noise.

    No biological or structural information is added. The protocol only degrades or
    perturbs the input volume according to the selected noise distribution.

    Because the result depends on random noise generation, different runs may
    produce different noisy maps unless the underlying program uses a fixed random
    seed.

    ## Practical Recommendations

    Use Gaussian noise for standard robustness tests.

    Use Student noise when you want heavier-tailed noise with occasional stronger
    perturbations.

    Use Uniform noise when you want bounded random perturbations.

    Start with small noise amplitudes and increase them gradually when testing
    downstream stability.

    Inspect the noisy output visually before using it in later analysis.

    Keep the original input volumes unchanged and use the noisy outputs only for
    simulation or validation workflows.

    ## Final Perspective

    Add Noise Volume/s is a simple simulation protocol for perturbing 3D maps.

    For biological users and method developers, its main value is that it provides
    controlled noisy inputs for testing reconstruction, filtering, sharpening,
    classification, validation, or visualization workflows.

    The protocol should be understood as a synthetic-data utility. It does not
    model the full physical process of cryo-EM image formation; it only adds
    statistical noise to existing volumes.
    """
    _label = 'add noise volume/s'
    
    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
       
        form.addParam('input', PointerParam,
                      pointerClass='SetOfVolumes, Volume',
                      label="Input Volume/s", 
                      help='Select a volume or Set of volumes.')
        
        XmippProtAddNoise._defineParams(self, form)
        
    def convertInputStep(self):
        pass
        
    def _getNoisyOutputPath(self, fnvol):
        fnNoisy = self._getExtraPath(removeExt(basename(fnvol)) + '_Noisy.mrc')
        return fnNoisy

    def _addNoisetoVolumeStep(self, kindNoise, noiseParams, vol):
        fnvol = vol.getFileName()
        fnNoisy = self._getNoisyOutputPath(fnvol)
        params = " -i %s --type %s %s -o %s" % (fnvol, kindNoise, noiseParams, 
                                                    fnNoisy)
        self.runJob('xmipp_transform_add_noise', params, numberOfMpi=1)
        self.runJob('xmipp_image_header', '-i %s --sampling_rate %f'%(fnvol, vol.getSamplingRate()), numberOfMpi=1)

    def addNoiseStep(self):
        kindNoise, noiseParams = self._getTypeOfNoise()
        
        inputSet = self.input.get()
        if isinstance(inputSet, Volume):
            self._addNoisetoVolumeStep(kindNoise, noiseParams, inputSet)
        else:
            for vol in self.input.get():
                self._addNoisetoVolumeStep(kindNoise, noiseParams, vol)

    def createOutputStep(self):
        #Output Volume/SetOfVolumes
        volInput = self.input.get()
        if self._isSingleVolume():
            # Create the output with the same class as
            # the input, that should be Volume or a subclass
            # of Volume like VolumeMask
            fnvol = volInput.getFileName()
            fnOutVol = self._getNoisyOutputPath(fnvol)
            
            volClass = volInput.getClass()
            vol = volClass() # Create an instance with the same class of input 
            vol.copyInfo(volInput)
            vol.setFileName(fnOutVol)
            self._defineOutputs(outputVolume=vol)
            self._defineSourceRelation(self.input.get(), vol)
        else:
            volumes = self._createSetOfVolumes()
            volumes.copyInfo(volInput)
            volumes.copyItems(volInput, updateItemCallback=self._updateNoisyPath)
            self._defineOutputs(outputVol=volumes)
            self._defineSourceRelation(self.input.get(), volumes)
            
#         self._defineTransformRelation(self.inputVolumes, self.outputVol)
    def _updateNoisyPath(self, vol, row):
        fnvol = vol.getFileName()
        fnOutVol = self._getNoisyOutputPath(fnvol)
        vol.setFileName(fnOutVol)
        
    def _isSingleVolume(self):
        return isinstance(self.input.get(), Volume)


class XmippProtAddNoiseParticles(XmippProtAddNoise):
    """    
    Given a set of particles, the protocol will add noise to them 
    The types of noise are Uniform, Student and Gaussian.

    AI Generated

    ## Overview

    The Add Noise Particles protocol creates a noisy version of an input particle
    set.

    This protocol is useful for simulation, robustness testing, algorithm
    development, and validation. By adding controlled noise to particle images, the
    user can test how downstream steps such as 2D classification, alignment,
    screening, reconstruction, or neural-network methods behave under degraded
    signal-to-noise conditions.

    The protocol supports three noise models:

    - Gaussian noise;
    - Student noise;
    - Uniform noise.

    The main output is a new particle set whose images contain the added noise.

    ## Inputs and General Workflow

    The input is a set of particles.

    The protocol first writes the input particle set to Xmipp metadata format.
    Then it runs the Xmipp noise-addition program on the particle metadata and
    saves the noisy particles into a new stack.

    Finally, it creates an output particle set by copying the input particle
    metadata and updating the image locations so that each particle points to its
    corresponding noisy image.

    ## Input Particles

    The **Input particles** parameter defines the particle set to which noise will
    be added.

    The protocol does not change the particle alignment, CTF metadata, sampling
    rate, or other particle information. It only creates new noisy images and
    updates the particle image locations.

    This makes the output suitable for controlled tests where the metadata should
    remain the same while the image content becomes noisier.

    ## Noise Type

    The **Noise Type** parameter selects the statistical distribution of the added
    noise.

    The available options are:

    **Gaussian** adds normally distributed noise.

    **Student** adds noise following a Student distribution.

    **Uniform** adds uniformly distributed noise.

    The choice depends on the kind of perturbation the user wants to simulate.

    ## Gaussian Noise

    When **Gaussian** noise is selected, the user can define:

    - **Standard Deviation**;
    - **Mean**.

    The standard deviation controls the noise strength. Larger values make the
    particles noisier.

    The mean controls the average value of the noise. The default is 0, meaning
    that the noise does not introduce a systematic intensity offset.

    Gaussian noise is the most common option for general signal-to-noise robustness
    tests.

    ## Student Noise

    When **Student** noise is selected, the user can define:

    - **Degree of Freedom**;
    - **Standard Deviation**;
    - **Mean**.

    The Student distribution can have heavier tails than the Gaussian
    distribution. This means that extreme noise values may occur more often,
    especially for low degrees of freedom.

    This option is useful when testing algorithms against outlier-prone or
    non-Gaussian noise.

    ## Uniform Noise

    When **Uniform** noise is selected, the user can define:

    - **Minimum Value**;
    - **Maximum Value**.

    The protocol samples noise values uniformly between these two limits.

    Uniform noise is useful when the user wants noise values bounded within a
    specific interval.

    ## Output Particles

    The main output is **outputParticles**.

    This output particle set copies the input particle information and replaces
    the image locations with the generated noisy images.

    The noisy particle images are stored in a new stack, and the metadata file
    created by Xmipp is used to update the particle locations.

    The output can be used in downstream protocols like any other particle set.

    ## Preserved Metadata

    The output particles preserve the metadata of the input particles.

    This means that alignment parameters, CTF information, sampling rate, and other
    stored attributes remain available, while the underlying image data are changed
    to the noisy versions.

    This is useful for experiments where the user wants to isolate the effect of
    image noise while keeping all other processing information unchanged.

    ## Interpreting the Result

    The output particles should be interpreted as synthetic noisy versions of the
    input particles.

    The protocol does not simulate full microscope image formation, radiation
    damage, CTF effects, detector effects, or realistic background correlations.
    It only adds random noise according to the selected distribution.

    Therefore, the protocol is best understood as a controlled perturbation tool
    rather than a complete physical simulator.

    ## Practical Recommendations

    Use this protocol when testing the robustness of classification, alignment,
    screening, or reconstruction protocols.

    Start with modest noise levels and increase them gradually.

    Use Gaussian noise for general tests, Student noise for heavier-tailed
    perturbations, and Uniform noise for bounded noise.

    Inspect a representative subset of noisy particles before running expensive
    downstream processing.

    Use the same input metadata and compare downstream results between the original
    and noisy particle sets to assess sensitivity to noise.

    Keep the original particle set unchanged and use the output only for testing,
    simulation, or validation.

    ## Final Perspective

    Add Noise Particles is a synthetic-data utility for perturbing particle images
    with controlled random noise.

    For biological users and method developers, its main value is that it allows
    controlled experiments on how particle-processing workflows behave as image
    noise increases.

    The protocol is simple by design: it preserves the particle metadata and
    generates new noisy image data, making it useful for reproducible comparisons
    between clean and degraded particle sets.
    """
    _label = 'add noise particles'
    
    # --------------------------- DEFINE param functions --------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
       
        form.addParam('input', PointerParam, pointerClass='SetOfParticles',
                      label="Input particles", 
                      help='Select a set of particles.')
        
        XmippProtAddNoise._defineParams(self, form)

    def convertInputStep(self):
        """ Read the input metadatata.
        """
        # Get the converted input micrographs in Xmipp format
        inputSet = self.input.get()
        inputPath = self._getExtraPath('inputSet')
        fnSet = inputPath+'.xmd'
        writeSetOfParticles(inputSet, fnSet)
        
    def addNoiseStep(self):
        kindNoise, noiseParams = self._getTypeOfNoise()
        params ='--save_metadata_stack'
        params += " -i %s --type %s %s -o %s" % (self._getExtraPath('inputSet.xmd'),
                                      kindNoise, noiseParams,
                                  self.getFileNameNoisyStk())
        
        self.runJob('xmipp_transform_add_noise', params, numberOfMpi=1)
       
    def createOutputStep(self):
        #Output Volume/SetOfVolumes
        particlesSet = self._createSetOfParticles()
        particlesSet.copyInfo(self.input.get())
        inputMd = self._getExtraPath('Noisy.xmd')
        particlesSet.copyItems(self.input.get(),
                               updateItemCallback=self._updateParticle,
                               itemDataIterator=md.iterRows(inputMd))
        self._defineOutputs(outputParticles=particlesSet)
        self._defineSourceRelation(self.input.get(), particlesSet)      

    def getFileNameNoisyStk(self):
        return self._getExtraPath('Noisy.stk')

    def _updateParticle(self, particle, row):
        #fn = particle.getFileName()
        # fnOut = self.getFileNameNoisyStk()
        # particle.setFileName(fnOut)
        
        index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
        particle.setLocation(index, filename)

