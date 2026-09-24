# **************************************************************************
# *
# * Authors:         Javier Vargas (jvargas@cnb.csic.es) (2016)
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

from os.path import join, isfile
from shutil import copyfile
import os

from pyworkflow.object import Float, String
from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        STEPS_PARALLEL,
                                        StringParam, BooleanParam, IntParam,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST)

from pyworkflow.utils.path import moveFile, makePath, cleanPattern
from pyworkflow.gui.plotter import Plotter

from pwem.objects import Volume
import pwem.emlib.metadata as md
from pwem.protocols import ProtAnalysis3D

from xmipp3.constants import CUDA_ALIGN_SIGNIFICANT
from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes, \
    getImageLocation
from xmipp3.base import isXmippCudaPresent

class XmippProtMultiRefAlignability(ProtAnalysis3D):
    """    
    Performs soft alignment validation of a set of particles confronting them
    against a given 3DEM map. This protocol produces particle alignment
    precision and accuracy parameters.

    AI Generated

    ## Overview

    The Multireference Alignability protocol evaluates how reliably a set of
    particles can be aligned against one or more 3D reference volumes.

    In single-particle cryo-EM, each particle image must be assigned an orientation
    relative to a 3D map. Some particles are easy to align because their projection
    contains distinctive features. Other particles are ambiguous because many
    orientations produce similar projections, because the particle is noisy, or
    because the structure has symmetry or pseudo-symmetry.

    This protocol performs a soft alignment-validation analysis. It compares the
    experimental particles with projection libraries generated from the reference
    volume and estimates alignability-related scores for each particle. These
    scores describe how precise and how accurate the angular assignment is expected
    to be.

    The protocol also reports global alignability quality parameters for each
    input volume and produces a 2D validation plot showing the relationship between
    angular precision and angular accuracy.

    ## Inputs and General Workflow

    The protocol requires:

    - one input volume, or a set of input volumes;
    - a set of input particles with alignment information.

    The particles are first converted to Xmipp metadata format. Optionally, the
    particles are CTF-corrected by Wiener filtering. They are then resized or
    filtered according to the target resolution used for the validation.

    For each input volume, the protocol generates a projection library. It then
    compares the experimental particles with the reference projections and keeps a
    set of the most similar candidate orientations for each particle. A second
    comparison is performed using projections generated from the reference volume
    itself, so that experimental particles and reference projections can be
    evaluated under comparable conditions.

    Finally, the protocol computes alignability scores and produces output
    particles annotated with precision, accuracy, and mirror-related scores. It
    also outputs the input volumes enriched with global alignability parameters.

    ## Input Volume or Volumes

    The **Input volume** parameter provides the 3D map used as reference for the
    alignability analysis.

    Although the interface presents this as an input volume, the protocol can
    internally handle either a single volume or a set of volumes. Each volume is
    processed separately, and a different alignability analysis is produced for
    each one.

    The reference volume should represent the structure that the particles are
    expected to contain. If the volume is wrong, too low quality, or corresponds to
    a different conformation, the alignability scores may reflect model mismatch
    rather than true orientation uncertainty.

    This protocol is especially useful when comparing alternative references or
    when evaluating whether some volumes provide more reliable particle alignment
    than others.

    ## Input Particles

    The **Input particles** parameter should point to a particle set with alignment
    information.

    The protocol uses these particles as the experimental images to be validated.
    The particles should correspond to the same specimen represented by the input
    volume. They should also have appropriate CTF metadata if CTF correction is
    enabled.

    Poorly centered particles, contaminants, strong heterogeneity, or incorrect
    preprocessing can reduce apparent alignability. Therefore, the scores should
    always be interpreted in relation to the quality and composition of the input
    particle set.

    ## Symmetry Group

    The **Symmetry group** parameter defines the symmetry used when generating
    projections and evaluating orientations.

    For asymmetric particles, this is usually **c1**. For symmetric structures, the
    appropriate symmetry group should be specified using Xmipp symmetry notation.

    Correct symmetry is important because equivalent orientations should not be
    treated as different biological views. If the wrong symmetry is used, the
    protocol may overestimate or underestimate orientation ambiguity.

    For example, a symmetric map may appear difficult to align if symmetry
    equivalences are not properly taken into account.

    ## Pseudo-Symmetry Group

    The **Pseudo symmetry group** parameter is an advanced option used when the map
    is close to a more restrictive symmetry than the one reported in the main
    symmetry parameter.

    Pseudo-symmetry can make alignment ambiguous. A particle may match several
    orientations almost equally well because the structure has repeated or nearly
    repeated features.

    Providing a pseudo-symmetry group allows the protocol to account for this type
    of ambiguity in the validation step.

    This option should only be used when there is a clear structural reason to
    suspect pseudo-symmetry.

    ## Angular Sampling

    The **Angular Sampling** parameter defines the angular distance, in degrees,
    between neighboring projection directions in the generated projection library.

    A smaller angular sampling gives a denser projection library and can detect
    orientation ambiguity more precisely, but it increases computation time. A
    larger angular sampling is faster, but may miss fine differences between nearby
    orientations.

    The default value is a practical compromise for many datasets. Advanced users
    may refine it depending on particle size, expected resolution, and the degree
    of angular ambiguity.

    ## Number of Orientations for Particle

    The **Number of Orientations for particle** parameter controls how many of the
    best matching projection directions are kept for each particle during the
    validation.

    Keeping several candidate orientations is essential for alignability analysis.
    If only one orientation fits well, the particle is more likely to be
    unambiguous. If several orientations fit almost equally well, the particle may
    be difficult to align precisely or accurately.

    A larger value explores more alternative orientations but increases
    computation. A smaller value focuses only on the most competitive candidates.

    ## Minimum and Maximum Tilt Angles

    The **Minimum allowed tilt angle** and **Maximum allowed tilt angle without
    mirror check** parameters restrict the tilt-angle range considered during
    alignment.

    These advanced parameters can be useful when the user knows that certain views
    should not be considered, or when the acquisition geometry imposes limits on
    expected orientations.

    Restricting the tilt range can reduce ambiguity and computation, but it should
    be done carefully. If the true particle orientations fall outside the allowed
    range, the alignability analysis will be biased.

    ## CTF Correction

    The **CTF correction** option performs CTF correction by Wiener filtering
    before the alignability analysis.

    CTF effects can make particles harder to compare with ideal projections of a
    volume. Wiener filtering attempts to compensate for these effects and can make
    the alignment-validation comparison more meaningful.

    This option requires reliable CTF information in the input particles. If the
    particles have already been phase-flipped, the protocol passes that information
    to the correction step.

    CTF correction is usually helpful when the goal is to compare particles with
    reference projections at a meaningful resolution, but it should be interpreted
    in relation to the quality of the CTF estimates.

    ## Isotropic Correction

    The **Isotropic Correction** option is used when CTF correction is enabled.

    If selected, the correction assumes that there is no astigmatism and applies an
    isotropic CTF correction. This simplifies the correction by treating the CTF as
    radially symmetric.

    This may be appropriate when astigmatism is negligible or when a simpler
    correction is desired. If significant astigmatism is present, an isotropic
    correction may be less accurate.

    ## Padding Factor and Wiener Constant

    The **Padding factor** and **Wiener constant** are advanced parameters for the
    Wiener CTF correction.

    The padding factor controls the amount of padding used during correction. The
    Wiener constant controls the strength of the Wiener filtering. If the Wiener
    constant is negative, the default behavior of the underlying program is used.

    Most users should leave these parameters at their default values unless they
    have experience with CTF correction and a specific reason to tune them.

    ## Correct for CTF Envelope

    The **Correct for CTF envelope** option is an advanced CTF-correction setting.

    It should only be used when the envelope function has been well estimated. If
    the envelope is inaccurate, correcting for it may introduce artifacts or make
    the particle comparison less reliable.

    For most routine workflows, this option should remain disabled unless there is
    a clear reason to enable it.

    ## Target Resolution

    The **Target resolution** parameter controls the resolution to which particles
    are effectively low-pass filtered or resized for the alignability analysis.

    The protocol modifies the working sampling and image size according to this
    target. The default value is intended to focus the analysis on robust
    medium-resolution information. Values around 8 to 10 Å are often useful because
    they preserve enough structural signal for alignment while reducing the
    influence of high-frequency noise.

    Changing this value should be done carefully. A very high-resolution target may
    make the analysis too sensitive to noise. A very low-resolution target may hide
    orientation-specific features and underestimate alignability.

    ## GPU Execution

    The protocol can use a GPU implementation for the significant-orientation
    search. GPU execution is enabled by default through hidden execution
    parameters.

    If GPU execution is requested but the required Xmipp CUDA program is not
    available, the protocol reports a validation error.

    GPU execution is useful because the projection-search steps can be
    computationally demanding.

    ## Alignability Precision

    The **alignability precision** score describes how sharply the particle
    orientation is determined.

    A particle with high precision has a well-defined best orientation: nearby or
    alternative orientations fit worse. A particle with low precision has several
    similar candidate orientations, meaning its exact angular assignment is
    uncertain.

    Low precision may occur because of low signal-to-noise ratio, small particle
    size, lack of distinctive features, preferred views, or symmetry-related
    ambiguity.

    ## Alignability Accuracy

    The **alignability accuracy** score describes whether the assigned orientation
    is expected to be correct relative to the reference.

    A particle may be precise but inaccurate if the algorithm confidently selects a
    wrong orientation. Conversely, a particle may be approximately accurate but not
    very precise if several nearby orientations fit similarly.

    Accuracy and precision should therefore be interpreted together. The protocol
    produces both particle-level scores and global volume-level parameters.

    ## Mirror Score

    The protocol also computes a mirror-related score.

    Mirror ambiguity is important in cryo-EM because some projections may be
    difficult to distinguish from their mirrored counterparts, especially for
    certain views, symmetries, or low-resolution particles.

    A high mirror ambiguity may indicate that the particle view is not reliable for
    determining handedness or that the reference has features that make mirrored
    orientations difficult to separate.

    ## Output Particles

    For each analyzed volume, the protocol produces an output particle set.

    The output particles preserve the input particle information and are annotated
    with Xmipp alignability scores, including:

    - alignability precision score;
    - alignability accuracy score;
    - mirror score;
    - a combined weight derived from accuracy and precision.

    These particle-level scores can be used to inspect which particles are
    reliably alignable and which ones are more ambiguous.

    They may also help identify subsets of particles that contribute more strongly
    to reliable angular assignment.

    ## Output Volumes

    The protocol also produces an output volume set.

    Each output volume corresponds to one input volume and is annotated with global
    alignability parameters, including precision, accuracy, and mirror-related
    weights.

    These global values summarize how well the particle set can be aligned against
    each reference volume.

    When several reference volumes are analyzed, these values can help compare
    which reference provides more reliable alignment.

    ## Soft Alignment Validation Plot

    For each input volume, the protocol creates a 2D plot of particle-level
    alignability.

    The plot shows angular precision on one axis and angular accuracy on the other.
    Each point corresponds to a particle.

    This plot provides a visual summary of the alignment-validation landscape. A
    cluster of particles with high precision and high accuracy suggests reliable
    alignment. A broad distribution or many particles with low values suggests
    substantial angular ambiguity.

    The plot is useful for diagnosing whether poor reconstruction quality may be
    related to ambiguous particle orientations.

    ## Practical Recommendations

    Use this protocol when you want to evaluate whether particles can be reliably
    aligned against a given reference volume.

    Use the correct symmetry group. Incorrect symmetry can strongly affect the
    interpretation of orientation ambiguity.

    Keep the target resolution near the default range unless there is a clear
    reason to change it. Medium-resolution information is often more reliable for
    alignment validation than very high-frequency detail.

    Enable CTF correction when reliable CTF metadata are available and when the
    particles have not already been corrected in a way that would make the setting
    inconsistent.

    Interpret precision, accuracy, and mirror scores together. A particle may fail
    one aspect of alignability while still looking acceptable in another.

    When comparing several reference volumes, examine both the global volume scores
    and the particle-level score distributions.

    Use the soft alignment validation plot to identify whether the dataset contains
    a large population of ambiguous particles.

    ## Final Perspective

    Multireference Alignability is a validation protocol for angular assignment.
    It estimates how reliably particles can be oriented with respect to one or more
    3D reference volumes.

    For biological users, this is useful because not all particles contribute
    equally to a reliable reconstruction. Some views or particles may be inherently
    ambiguous, especially in the presence of symmetry, pseudo-symmetry, noise, or
    weak structural features.

    By providing particle-level and volume-level alignability scores, the protocol
    helps assess whether a reconstruction problem may come from poor angular
    information rather than only from particle number, refinement settings, or
    sample quality.
    """
    _label = 'multireference alignability'

    INPUTARG = "-i %s"
    OUTPUTARG = " -o %s"

    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')
        form.addParam('inputVolumes', PointerParam, pointerClass='Volume',
                      label="Input volume",
                      help='Select the input volume(s).')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignment',
                      label="Input particles", important=True,
                      help='Select the input projection images.')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp')
        form.addParam('angularSampling', FloatParam, default=5,
                      expertLevel=LEVEL_ADVANCED,
                      label="Angular Sampling (degrees)",
                      help='Angular distance (in degrees) between neighboring projection points ')
        form.addParam('numOrientations', FloatParam, default=7,
                      expertLevel=LEVEL_ADVANCED,
                      label="Number of Orientations for particle",
                      help='Parameter to define the number of most similar volume \n'
                           '    projected images for each projection image')
        form.addParam('doNotUseWeights', BooleanParam, default=False,
                      expertLevel=LEVEL_ADVANCED,
                      label="Do not use the weights",
                      help='Do not use the weights in the clustering calculation')
        form.addParam('pseudoSymmetryGroup', StringParam, default='',
                      expertLevel=LEVEL_ADVANCED,
                      label="Pseudo symmetry group",
                      help='Add only in case the map is close to a symmetry different and more restrict than the one reported in the parameter Symmetry group.'
                           'See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp')
        form.addParam('minTilt', FloatParam, default=0,
                      expertLevel=LEVEL_ADVANCED,
                      label="Minimum allowed tilt angle",
                      help='Tilts below this value will not be considered for the alignment')
        form.addParam('maxTilt', FloatParam, default=180,
                      expertLevel=LEVEL_ADVANCED,
                      label="Maximum allowed tilt angle without mirror check",
                      help='Tilts above this value will not be considered for the alignment without mirror check')

        form.addSection(label='Preprocess')
        form.addParam('doWiener', BooleanParam, default='True',
                      label="CTF correction",
                      help='Perform CTF correction by Wiener filtering.')
        form.addParam('isIsotropic', BooleanParam, default='True',
                      label="Isotropic Correction", condition='doWiener',
                      help='If true, Consider that there is not astigmatism and then it is performed an isotropic correction.')
        form.addParam('padding_factor', IntParam, default=2,
                      expertLevel=LEVEL_ADVANCED,
                      label="Padding factor", condition='doWiener',
                      help='Padding factor for Wiener correction ')
        form.addParam('wiener_constant', FloatParam, default=-1,
                      expertLevel=LEVEL_ADVANCED,
                      label="Wiener constant", condition='doWiener',
                      help=' Wiener-filter constant (if < 0: use FREALIGN default)')
        form.addParam('correctEnvelope', BooleanParam, default='False',
                      expertLevel=LEVEL_ADVANCED,
                      label="Correct for CTF envelope", condition='doWiener',
                      help=' Only in cases where the envelope is well estimated correct for it')
        form.addParam('targetResolution', FloatParam, default=8,
                      label='Target resolution (A)',
                      help='Low pass filter the particles to this resolution. This usually helps a lot obtaining good alignment. You should have a good'
                           ' reason to modify this value outside the range  [8-10] A')

        form.addParallelSection(threads=1, mpi=1)

    def _getFileName(self, key, **kwargs):
        if key=="volume":
            return self._getExtraPath("volume.mrc")
        elif key=="reference_particles":
            return self._getPath("reference_particles.xmd")
        else:
            return ""

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        convertId = self._insertFunctionStep('convertInputStep',
                                             self.inputParticles.get().getObjId())
        deps = []  # store volumes steps id to use as dependencies for last step
        commonParams = self._getCommonParams()
        commonParamsRef = self._getCommonParamsRef()

        sym = self.symmetryGroup.get()

        for i, vol in enumerate(self._iterInputVols()):

            volName = getImageLocation(vol)
            volDir = self._getVolDir(i + 1)

            pmStepId = self._insertFunctionStep('projectionLibraryStep',
                                                volDir,
                                                self.angularSampling.get(),
                                                prerequisites=[convertId])

            sigStepId1 = self._insertFunctionStep('significantStep',
                                                  volName, volDir,
                                                  'exp_particles.xmd',
                                                  commonParams,
                                                  prerequisites=[pmStepId])

            phanProjStepId = self._insertFunctionStep('phantomProject',
                                                      prerequisites=[
                                                          sigStepId1])

            sigStepId2 = self._insertFunctionStep('significantStep',
                                                  volName, volDir,
                                                  'ref_particles.xmd',
                                                  commonParamsRef,
                                                  prerequisites=[
                                                      phanProjStepId])

            if (not (self.pseudoSymmetryGroup.get() == '')):
                sym = self.pseudoSymmetryGroup.get()

            volStepId = self._insertFunctionStep('alignabilityStep',
                                                 volName, volDir,
                                                 sym,
                                                 prerequisites=[sigStepId2])

            deps.append(volStepId)

        self._insertFunctionStep('createOutputStep',
                                 prerequisites=deps)

    def convertInputStep(self, particlesId):
        """ Write the input images as a Xmipp metadata file.
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """

        writeSetOfParticles(self.inputParticles.get(),
                            self._getPath('input_particles.xmd'))

        if self.doWiener.get():
            params = self.INPUTARG % self._getPath('input_particles.xmd')
            params += self.OUTPUTARG % self._getExtraPath(
                'corrected_ctf_particles.stk')
            params += '  --save_metadata_stack %s' % self._getExtraPath(
                'corrected_ctf_particles.xmd')
            params += '  --pad %s' % self.padding_factor.get()
            params += '  --wc %s' % self.wiener_constant.get()
            params += '  --sampling_rate %s' % self.inputParticles.get().getSamplingRate()

            if self.inputParticles.get().isPhaseFlipped():
                params += '  --phase_flipped '

            if self.correctEnvelope:
                params += '  --correct_envelope '

            nproc = self.numberOfMpi.get()
            nT = self.numberOfThreads.get()

            self.runJob('xmipp_ctf_correct_wiener2d',
                        params)

        newTs, newXdim = self._getModifiedSizeAndSampling()

        if self.doWiener.get():
            params =  self.INPUTARG % self._getExtraPath('corrected_ctf_particles.xmd')
        else :
            params =  self.INPUTARG % self._getPath('input_particles.xmd')
            
        params +=  self.OUTPUTARG % self._getExtraPath('scaled_particles.stk')
        params +=  '  --save_metadata_stack %s' % self._getExtraPath('scaled_particles.xmd')
        params +=  '  --fourier %d' % newXdim
        
        self.runJob('xmipp_image_resize',params)
        
        from pwem.emlib.image import ImageHandler
        img = ImageHandler()
        img.convert(self.inputVolumes.get(), self._getFileName("volume"))
        Xdim = self.inputVolumes.get().getDim()[0]
        if Xdim != newXdim:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" % \
                        (self._getFileName("volume"), newXdim), numberOfMpi=1)

    def _getCommonParams(self):
        params = self.INPUTARG % self._getExtraPath('scaled_particles.xmd')
        params += ' --sym %s' % self.symmetryGroup.get()
        params += ' --dontReconstruct'
        params += ' --useForValidation %0.3f' % (self.numOrientations.get() - 1)
        params += ' --dontCheckMirrors'
        return params

    def _getCommonParamsRef(self):
        params = self.INPUTARG % self._getFileName("reference_particles")
        params += ' --sym %s' % self.symmetryGroup.get()
        params += ' --dontReconstruct'
        params += ' --useForValidation %0.3f' % (self.numOrientations.get() - 1)
        params += ' --dontCheckMirrors'
        return params

    def _getModifiedSizeAndSampling(self):
        Xdim = self.inputParticles.get().getDimensions()[0]
        Ts = self.inputParticles.get().getSamplingRate()
        newTs = self.targetResolution.get() * 0.4
        newTs = max(Ts, newTs)
        newXdim = Xdim * Ts / newTs
        return newTs, newXdim

    def phantomProject(self):
        nproc = self.numberOfMpi.get()
        nT = self.numberOfThreads.get()

        newTs, newXdim = self._getModifiedSizeAndSampling()

        pathParticles = self._getExtraPath('scaled_particles.xmd')
        R = -int(newXdim / 2)
        f = open(self._getExtraPath('params.txt'), 'w')
        f.write("""# XMIPP_STAR_1 *
#
data_block1
_dimensions2D '%d %d'
_projAngleFile %s
_ctfPhaseFlipped %d
_ctfCorrected %d
_applyShift 0
_noisePixelLevel   '0 0'""" % (newXdim, newXdim, pathParticles,
                               self.inputParticles.get().isPhaseFlipped(),
                               self.doWiener.get()))
        f.close()
        param = self.INPUTARG % self._getFileName("volume")
        param += ' --params %s' % self._getExtraPath('params.txt')
        param += self.OUTPUTARG % self._getFileName("reference_particles")
        param += ' --sampling_rate % 0.3f' % newTs
        param += ' --method fourier'
                
        #while (~isfile(self._getExtraPath('params'))):
        #    print('No created')
        
        self.runJob('xmipp_phantom_project', 
                    param, numberOfMpi=1,numberOfThreads=1)

        param = self.INPUTARG % self._getPath('reference_particles.stk')
        param += ' --mask circular %d' % R
        self.runJob('xmipp_transform_mask', param, numberOfMpi=nproc,
                    numberOfThreads=nT)

    def projectionLibraryStep(self, volDir, angularSampling):

        # Generate projections from this reconstruction
        nproc = self.numberOfMpi.get()
        nT = self.numberOfThreads.get()
        volName = self._getFileName("volume")
        makePath(volDir)
        fnGallery = (volDir + '/gallery.stk')
        params = '-i %s -o %s --sampling_rate %f --sym %s --method fourier 1 0.25 bspline --compute_neighbors --angular_distance %f --experimental_images %s --max_tilt_angle %f --min_tilt_angle %f' \
                 % (
                 volName, fnGallery, angularSampling, self.symmetryGroup.get(),
                 -1, self._getExtraPath('scaled_particles.xmd'),
                 self.maxTilt.get(), self.minTilt.get())

        self.runJob("xmipp_angular_project_library", params, numberOfMpi=nproc,
                    numberOfThreads=nT)

    def significantStep(self, volName, volDir, anglesPath, params):
        nproc = self.numberOfMpi.get()
        nT = self.numberOfThreads.get()
        fnGallery = (volDir + '/gallery.doc')

        if not self.useGpu.get():
            params += ' --initgallery  %s' % fnGallery
            params += ' --odir %s' % volDir
            params += ' --iter %d' % 1
            self.runJob('xmipp_reconstruct_significant',
                        params, numberOfMpi=nproc, numberOfThreads=nT)
            copyfile(volDir + '/angles_iter001_00.xmd',
                     self._getTmpPath(anglesPath))
        else:
            count=0
            GpuListCuda=''
            if self.useQueueForSteps() or self.useQueue():
                GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                GpuList = GpuList.split(",")
                for elem in GpuList:
                    GpuListCuda = GpuListCuda+str(count)+' '
                    count+=1
            else:
                GpuList = ' '.join([str(elem) for elem in self.getGpuList()])
                GpuListAux = ''
                for elem in self.getGpuList():
                    GpuListCuda = GpuListCuda+str(count)+' '
                    GpuListAux = GpuListAux+str(elem)+','
                    count+=1
                os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux

            if anglesPath == 'exp_particles.xmd':
                params = self.INPUTARG % self._getExtraPath('scaled_particles.xmd')
            elif anglesPath == 'ref_particles.xmd':
                params = self.INPUTARG % self._getFileName("reference_particles")
            params += ' --keepBestN %f' % (self.numOrientations.get() - 1)
            params += ' -r  %s' % fnGallery
            params += ' -o  %s' % self._getTmpPath(anglesPath)
            params += ' --dev %s ' % GpuListCuda
            self.runJob(CUDA_ALIGN_SIGNIFICANT, params, numberOfMpi=1)

    def alignabilityStep(self, volName, volDir, sym):

        nproc = self.numberOfMpi.get()
        nT = self.numberOfThreads.get()

        makePath(volDir)
        inputFile = self._getPath('input_particles.xmd')
        inputFileRef = self._getFileName("reference_particles")
        aFile = self._getTmpPath('exp_particles.xmd')
        aFileRef = self._getTmpPath('ref_particles.xmd')
        aFileGallery = (volDir + '/gallery.doc')

        params = self.INPUTARG % inputFile
        params += ' -i2 %s' % inputFileRef
        params += '  --volume %s' % volName
        params += '  --angles_file %s' % aFile
        params += '  --angles_file_ref %s' % aFileRef
        params += '  --gallery %s' % aFileGallery
        params += ' --odir %s' % volDir
        params += ' --sym %s' % sym
        params += ' --check_mirrors'

        if self.doNotUseWeights:
            params += ' --dontUseWeights'

        self.runJob('xmipp_multireference_aligneability', params,
                    numberOfMpi=nproc, numberOfThreads=nT)

    def neighbourhoodDirectionStep(self, volName, volDir, sym):

        aFileGallery = (volDir + '/gallery.doc')
        neighbours = (volDir + '/neighbours.xmd')

        params = '  --i1 %s' % self._getPath('input_particles.xmd')
        params += ' --i2 %s' % aFileGallery
        params += self.OUTPUTARG % neighbours
        params += ' --dist %s' % (self.angDist.get() + 1)
        params += ' --sym %s' % sym

        self.runJob('xmipp_angular_neighbourhood', params, numberOfMpi=1,
                    numberOfThreads=1)

    def angularAccuracyStep(self, volName, volDir, indx):

        nproc = self.numberOfMpi.get()
        nT = self.numberOfThreads.get()

        neighbours = (volDir + '/neighbours.xmd')

        params = self.INPUTARG % volName
        params += ' --i2 %s' % neighbours
        params += self.OUTPUTARG % (
                volDir + '/pruned_particles_alignability_accuracy.xmd')

        self.runJob('xmipp_angular_accuracy_pca', params, numberOfMpi=nproc,
                    numberOfThreads=nT)

    def createOutputStep(self):

        outputVols = self._createSetOfVolumes()

        for i, vol in enumerate(self._iterInputVols()):
            volDir = self._getVolDir(i + 1)
            volume = vol.clone()
            volPrefix = 'vol%03d_' % (i + 1)

            m_pruned = md.MetaData()
            m_pruned.read(volDir + '/pruned_particles_alignability.xmd')
            prunedMd = self._getExtraPath(
                volPrefix + 'pruned_particles_alignability.xmd')

            moveFile(join(volDir, 'pruned_particles_alignability.xmd'),
                     prunedMd)
            m_volScore = md.MetaData()
            m_volScore.read(volDir + '/validationAlignability.xmd')
            validationMd = self._getExtraPath(
                volPrefix + 'validation_alignability.xmd')
            moveFile(join(volDir, 'validationAlignability.xmd'), validationMd)

            imgSet = self.inputParticles.get()

            outImgSet = self._createSetOfParticles(volPrefix)
            outImgSet.copyInfo(imgSet)

            outImgSet.copyItems(imgSet,
                                updateItemCallback=self._setWeight,
                                itemDataIterator=md.iterRows(prunedMd,
                                                             sortByLabel=md.MDL_ITEM_ID))

            mdValidatoin = md.getFirstRow(validationMd)

            weight = mdValidatoin.getValue(md.MDL_WEIGHT_PRECISION_ALIGNABILITY)
            volume.weightAlignabilityPrecision = Float(weight)

            weight = mdValidatoin.getValue(md.MDL_WEIGHT_ACCURACY_ALIGNABILITY)
            volume.weightAlignabilityAccuracy = Float(weight)

            weight = mdValidatoin.getValue(md.MDL_WEIGHT_PRECISION_MIRROR)
            volume.weightMirror = Float(weight)

            volume.cleanObjId()  # clean objects id to assign new ones inside the set
            outputVols.append(volume)
            self._defineOutputs(outputParticles=outImgSet)

            self.createPlot2D(volPrefix, m_pruned)

        outputVols.setSamplingRate(volume.getSamplingRate())
        self._defineOutputs(outputVolumes=outputVols)

        cleanPattern(self._getPath("reference_particles.*"))
        cleanPattern(self._getExtraPath("scaled_particles.*"))
        cleanPattern(self._getExtraPath("reference_particles.*"))
        cleanPattern(self._getExtraPath("corrected_ctf_particles.*"))
        cleanPattern(self._getFileName("volume"))
        cleanPattern(self._getExtraPath("params.txt"))

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        validateMsgs = []
        # if there are Volume references, it cannot be empty.
        if self.inputVolumes.get() and not self.inputVolumes.hasValue():
            validateMsgs.append('Please provide an input reference volume.')
        if self.inputParticles.get() and not self.inputParticles.hasValue():
            validateMsgs.append('Please provide input particles.')
        if self.useGpu and not isXmippCudaPresent(CUDA_ALIGN_SIGNIFICANT):
            validateMsgs.append("You have asked to use GPU, but I cannot find the Xmipp GPU programs in the path")
        return validateMsgs

    def _summary(self):
        summary = [
            "Input particles:  %s" % self.inputParticles.get().getNameId()]
        summary.append("-----------------")

        if self.inputVolumes.get():
            for i, vol in enumerate(self._iterInputVols()):
                summary.append("Input volume(s)_%d: [%s]" % (i + 1, vol))
        summary.append("-----------------")

        if (not hasattr(self, 'outputVolumes')):
            summary.append("Output volumes not ready yet.")
        else:
            for i, vol in enumerate(self._iterInputVols()):
                VolPrefix = 'vol%03d_' % (i + 1)
                mdVal = md.MetaData(self._getExtraPath(
                    VolPrefix + 'validation_alignability.xmd'))
                weightAccuracy = mdVal.getValue(
                    md.MDL_WEIGHT_ACCURACY_ALIGNABILITY, mdVal.firstObject())
                weightPrecision = mdVal.getValue(
                    md.MDL_WEIGHT_PRECISION_ALIGNABILITY, mdVal.firstObject())
                weightAlignability = mdVal.getValue(md.MDL_WEIGHT_ALIGNABILITY,
                                                    mdVal.firstObject())

                summary.append("ALIGNABILITY ACCURACY parameter_%d : %f" % (
                i + 1, weightAccuracy))
                summary.append("ALIGNABILITY PRECISION parameter_%d : %f" % (
                i + 1, weightPrecision))
                summary.append(
                    "ALIGNABILITY ACCURACY & PRECISION parameter_%d : %f" % (
                    i + 1, weightAlignability))

                summary.append("-----------------")
        return summary

    def _methods(self):
        messages = []
        if (hasattr(self, 'outputVolumes')):
            messages.append('The quality parameter(s) has been obtained using '
                            'the approach [Vargas2014a] with angular sampling '
                            'of %f and number of orientations of %f' % (
                            self.angularSampling.get(),
                            self.numOrientations.get()))
        return messages

    def _citations(self):
        return ['Vargas2014a']

    # --------------------------- UTILS functions --------------------------------------------
    def _getVolDir(self, volIndex):
        return self._getTmpPath('vol%03d' % volIndex)

    def _iterInputVols(self):
        """ In this function we will encapsulate the logic
        to iterate through the input volumes.
        This give the flexibility of having Volumes, SetOfVolumes or
        a combination of them as input and the protocol code
        remain the same.
        """
        inputVols = self.inputVolumes.get()

        if isinstance(inputVols, Volume):
            yield inputVols
        else:
            for vol in inputVols:
                yield vol

    def _defineMetadataRootName(self, mdrootname, volId):

        if mdrootname == 'P':
            VolPrefix = 'vol%03d_' % (volId)
            return self._getExtraPath(VolPrefix + 'clusteringTendency.xmd')
        if mdrootname == 'Volume':
            VolPrefix = 'vol%03d_' % (volId)
            return self._getExtraPath(VolPrefix + 'validation.xmd')

    def _defineVolumeName(self, volId):
        fscFn = self._defineMetadataRootName('Volume', volId)
        return fscFn

    def _setWeight(self, item, row):
        item._xmipp_scoreAlignabilityPrecision = Float(
            row.getValue(md.MDL_SCORE_BY_ALIGNABILITY_PRECISION))
        item._xmipp_scoreAlignabilityAccuracy = Float(
            row.getValue(md.MDL_SCORE_BY_ALIGNABILITY_ACCURACY))
        item._xmipp_scoreMirror = Float(row.getValue(md.MDL_SCORE_BY_MIRROR))
        item._xmipp_weight = Float( float(item._xmipp_scoreAlignabilityAccuracy)*float(item._xmipp_scoreAlignabilityPrecision))
        
    def createPlot2D(self,volPrefix,md):
        
        from pwem import emlib
        
        figurePath = self._getExtraPath(volPrefix + 'softAlignmentValidation2D.png')
        figureSize = (8, 6)

        # alignedMovie = mic.alignMetaData
        plotter = Plotter(*figureSize)
        figure = plotter.getFigure()

        ax = figure.add_subplot(111)
        ax.grid()
        ax.set_title('Soft alignment validation map')
        ax.set_xlabel('Angular Precision')
        ax.set_ylabel('Angular Accuracy')

        for objId in md:
            x = md.getValue(emlib.MDL_SCORE_BY_ALIGNABILITY_PRECISION, objId)
            y = md.getValue(emlib.MDL_SCORE_BY_ALIGNABILITY_ACCURACY, objId)
            ax.plot(x, y, 'r.',markersize=1)

        ax.grid(True, which='both')
        ax.autoscale_view(True, True, True)

        plotter.savefig(figurePath)
        plotter.show()
        return plotter
