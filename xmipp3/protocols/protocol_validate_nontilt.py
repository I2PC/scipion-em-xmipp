# **************************************************************************
# *
# * Authors:     Javier Vargas (jvargas@cnb.csic.es)
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
import os

from pyworkflow.object import Float, String
from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        STEPS_PARALLEL,
                                        StringParam, EnumParam, LEVEL_ADVANCED,
                                        BooleanParam, USE_GPU, GPU_LIST)
from pwem.objects import Volume
from pwem.protocols import ProtAnalysis3D
from pyworkflow.utils.path import moveFile, makePath
import pwem.emlib.metadata as md

from xmipp3.constants import CUDA_ALIGN_SIGNIFICANT
from xmipp3.convert import writeSetOfParticles
from xmipp3.base import isXmippCudaPresent

PROJECTION_MATCHING = 0
SIGNIFICANT = 1


class XmippProtValidateNonTilt(ProtAnalysis3D):
    """    
    Ranks a set of volumes based on alignment reliability using a
    clusterability test. This validation helps identify well-aligned
    structures and discard poorly aligned or inconsistent reconstructions,
    improving final data quality.

    AI Generated

    ## Overview

    The Validate Nontilt protocol evaluates one or more candidate 3D volumes using
    a set of experimental particle images.

    This protocol is intended for validation situations where tilted-pair data are
    not available. Instead of using tilt geometry, it tests how reliably the input
    particles can be assigned to projections of each candidate volume. A volume
    that produces stable, meaningful, and clusterable angular assignments is
    considered more consistent with the particle data than a volume that produces
    ambiguous or unreliable assignments.

    The protocol generates projection galleries from each input volume, aligns the
    input particles against those galleries, and then applies a non-tilt validation
    test. The result is an output set of volumes annotated with a quality weight.
    This weight can be used to compare or rank the candidate volumes.

    ## Inputs and General Workflow

    The protocol requires:

    - one volume or a set of volumes to validate;
    - a set of particles or 2D classes used as projection images.

    For each input volume, the protocol performs the following steps:

    1. writes the input particles to Xmipp metadata;
    2. band-pass filters the volume to the selected resolution range;
    3. generates a projection gallery from the filtered volume;
    4. assigns possible orientations to the particles using either projection
       matching or significant alignment;
    5. runs the non-tilt validation test;
    6. stores the validation score and clustering-tendency information.

    The final output is a set of volumes copied from the input volumes and
    annotated with the validation weight.

    ## Input Volumes

    The **Input volumes** parameter defines the candidate volume or volumes to be
    validated.

    The input can be a single **Volume** or a **SetOfVolumes**. Each volume is
    processed independently and receives its own validation metadata.

    This protocol is useful when several possible initial models, class volumes, or
    alternative reconstructions are available and the user wants to assess which
    ones are most consistent with the particle images.

    The volumes should correspond to the same specimen and should be comparable to
    the input particle set. Unrelated volumes may still produce a numerical result,
    but it will not be biologically meaningful.

    ## Input Particles

    The **Input particles** parameter provides the projection images used for
    validation.

    The input can be a **SetOfParticles** or a **SetOfClasses2D**. These images are
    written to Xmipp metadata and compared against projections of each candidate
    volume.

    The quality and representativeness of these particles are critical. If the
    particle set contains many contaminants, junk particles, strong heterogeneity,
    or missing views, the validation result may be difficult to interpret.

    ## Symmetry Group

    The **Symmetry group** parameter defines the symmetry used when generating
    projections and during validation.

    For asymmetric particles, use **c1**. If the structure has known symmetry, the
    corresponding Xmipp symmetry group can be provided.

    Correct symmetry can make angular assignment more stable and biologically
    consistent. Incorrect symmetry can make a wrong volume appear artificially
    better or can hide important asymmetric differences.

    Use symmetry only when it is justified by the specimen.

    ## Image Alignment Method

    The **Image alignment** parameter controls how particles are assigned to
    possible projections of the candidate volume.

    There are two options:

    **Projection_Matching** uses standard angular projection matching against the
    projection gallery.

    **Significant** uses significant angular assignment, keeping a selected number
    of plausible orientations per particle. This is the default mode.

    The significant mode is often useful for validation because it considers
    alignment reliability and ambiguity rather than forcing only one orientation
    too early.

    ## Resolution to Filter

    The **Resolution to filter** parameters define a band-pass filter applied to
    each candidate volume before projections are generated.

    The parameters are:

    - **High**, the high-pass filtering resolution in angstroms;
    - **Low**, the low-pass filtering resolution in angstroms.

    The protocol converts these values into digital frequencies using the sampling
    rate of the particle set.

    Filtering helps focus the validation on a controlled resolution range. It can
    remove very low-frequency background and high-frequency noise that may not be
    reliable for angular assignment.

    The default values define a broad band-pass range suitable for many validation
    workflows.

    ## Angular Sampling

    The **Angular Sampling** parameter defines the angular spacing, in degrees, of
    the projection gallery.

    A smaller value generates a denser gallery and allows more precise orientation
    assignment, but increases computation time.

    A larger value is faster but may make alignment less accurate and may reduce
    the reliability of the validation.

    The default value is 5 degrees, which provides a practical compromise for many
    datasets.

    ## Number of Orientations per Particle

    The **Number of orientations per particle** parameter defines how many possible
    orientations are retained for each particle during alignment.

    Keeping several orientations is important for assessing ambiguity. A particle
    that can be explained equally well by many unrelated orientations provides
    weaker evidence for the volume than a particle with a stable, well-defined
    orientation.

    The default value is 10.

    Increasing this number may provide a richer description of angular ambiguity
    but also increases computation and metadata size.

    ## Significance

    The **Significance** parameter controls the validation test against a reference
    distribution of uniformly distributed random points.

    The default value is 0.95.

    This value affects how strictly the protocol evaluates whether the angular
    assignments show meaningful clusterability or reliability. A higher
    significance is more stringent; a lower value is more permissive.

    Most users should start with the default value.

    ## Projection Gallery

    For each candidate volume, the protocol creates a projection gallery.

    The gallery is generated from the filtered volume using the selected angular
    sampling and symmetry group. The gallery also stores neighbor information and
    angular-distance information needed by later alignment and validation steps.

    The experimental particles are then compared against this gallery.

    ## Significant Alignment Mode

    In significant alignment mode, the protocol assigns particles to projections
    using a significant angular-assignment procedure.

    When GPU execution is enabled, the protocol uses the Xmipp CUDA significant
    alignment program. When GPU execution is disabled, it uses the CPU significant
    reconstruction/alignment machinery without reconstruction.

    This mode keeps the best selected number of orientations per particle and uses
    them for validation.

    GPU execution can substantially speed up this step.

    ## Projection Matching Mode

    In projection-matching mode, the protocol uses standard projection matching.

    The search uses the generated projection gallery, an outer radius based on half
    the particle box size, and a maximum shift/search range based on one tenth of
    the particle box size.

    This mode is more classical and direct. It may be useful for comparison with
    older workflows or when the user wants a standard projection-matching
    validation path.

    ## Non-Tilt Validation Step

    After angular assignment, the protocol runs the Xmipp non-tilt validation
    program.

    The validation uses:

    - the assigned orientations;
    - the filtered candidate volume;
    - the selected symmetry;
    - the significance value;
    - whether significant alignment was used.

    The validation produces metadata files describing the quality of the volume and
    the clustering tendency of the angular assignments.

    These files are stored for each input volume.

    ## Output Volumes

    The main output is **outputVolumes**.

    This output contains copies of the input volumes, each annotated with an Xmipp
    quality weight. The weight is read from the validation metadata and stored as
    an attribute of the output volume.

    The output set sampling rate is set to the sampling rate of the input particle
    set.

    The output volumes can be inspected, compared, or ranked according to their
    validation weights.

    ## Validation Metadata

    For each volume, the protocol writes validation metadata files in the protocol
    output directory.

    The files include:

    - a `validation.xmd` file containing the main validation result;
    - a `clusteringTendency.xmd` file containing clustering-tendency information.

    The protocol renames these files with a volume-specific prefix, such as
    `vol001_validation.xmd` and `vol001_clusteringTendency.xmd`.

    These files are useful for advanced inspection of the validation results.

    ## Interpreting the Weight

    The output weight is a validation score derived from the non-tilt validation
    analysis.

    It should be used as a relative indicator when comparing candidate volumes
    against the same particle set and with the same protocol parameters.

    A better-scoring volume is more consistent with the particle alignment
    reliability under this test. However, the score should not be interpreted
    alone. It should be considered together with visual inspection, refinement
    behavior, FSC, class averages, and biological plausibility.

    ## GPU Execution

    The protocol supports GPU execution for the significant-alignment mode.

    If GPU execution is requested but the required Xmipp CUDA programs are not
    available, the protocol reports a validation error.

    GPU execution is recommended when available, especially for large particle sets
    or dense angular sampling.

    ## Practical Recommendations

    Use this protocol when you have several possible initial volumes and want to
    rank them without tilted-pair validation.

    Use the same particle set and the same parameters for all candidate volumes so
    that scores are comparable.

    Start with the default significant alignment mode, angular sampling, and
    significance level.

    Use projection matching when you want a standard angular-assignment comparison
    or when significant alignment is not desired.

    Apply symmetry only when it is biologically justified.

    Choose filtering limits that remove irrelevant noise but preserve the
    structural features needed for alignment.

    Inspect both the output weights and the original volumes. A high score does
    not automatically guarantee that the volume is biologically correct.

    ## Final Perspective

    Validate Nontilt is a reference-validation protocol for single-particle data
    when tilted-pair validation is not available.

    For biological users, its main value is that it provides a way to compare
    candidate 3D volumes based on how reliably experimental particles align to
    their projection galleries. This can help select among alternative initial
    models or reconstructions before committing to further refinement.

    The protocol should be used as part of a broader validation strategy, together
    with visual inspection, 2D class consistency, FSC analysis, refinement
    behavior, and biological knowledge.
    """

    _label = 'validate_nontilt'
    WEB = 0

    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

        if (self.WEB == 1):
            self.stepsExecutionMode = STEPS_PARALLEL

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

        form.addParam('inputVolumes', PointerParam,
                      pointerClass='SetOfVolumes, Volume',
                      label="Input volumes",
                      help='Select the input volumes.')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles, SetOfClasses2D',
                      label="Input particles",
                      help='Select the input projection images .')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp')
        form.addParam('alignmentMethod', EnumParam, label='Image alignment',
                      choices=['Projection_Matching', 'Significant'],
                      default=SIGNIFICANT)
        line = form.addLine('Resolution to filter (A)')
        line.addParam('highPassFilter', FloatParam, default=150, label='High')
        line.addParam('lowPassFilter', FloatParam, default=15, label='Low')

        form.addParam('angularSampling', FloatParam, default=5,
                      expertLevel=LEVEL_ADVANCED,
                      label="Angular Sampling (degrees)",
                      help='Angular distance (in degrees) between neighboring projection points ')
        form.addParam('numOrientations', FloatParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label="Number of orientations per particle",
                      help='Number of possible orientations in which a particle can be \n')
        form.addParam('significanceNoise', FloatParam, default=0.95,
                      expertLevel=LEVEL_ADVANCED,
                      label="Significance",
                      help='Significance of the aligniability with respect'
                           ' to a a set of uniformly distributed random points \n')

        form.addParallelSection(threads=0, mpi=4)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        deps = []  # store volumes steps id to use as dependencies for last step
        self.partSet = self.inputParticles.get()

        convertId = self._insertFunctionStep('convertInputStep',
                                             self.partSet.getObjId())

        for vol in self._iterInputVols():
            filterId = self._insertFunctionStep('filterVolumeStep',
                                                vol.getObjId(),
                                                vol.getFileName(),
                                                prerequisites=[convertId])
            pmStepId = self._insertFunctionStep('projectionLibraryStep',
                                                vol.getObjId(),
                                                prerequisites=[filterId])

            if self.alignmentMethod == SIGNIFICANT:
                sigStepId = self._insertFunctionStep('significantStep',
                                                     vol.getObjId(),
                                                     prerequisites=[pmStepId])
            else:
                sigStepId = self._insertFunctionStep('projectionMatchingStep',
                                                     vol.getObjId(),
                                                     prerequisites=[pmStepId])

            volStepId = self._insertFunctionStep('validationStep',
                                                 vol.getObjId(),
                                                 prerequisites=[sigStepId])
            deps.append(volStepId)

        self._insertFunctionStep('createOutputStep', prerequisites=deps)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self, particlesId):
        """ Write the input images as a Xmipp metadata file.
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        writeSetOfParticles(self.partSet,
                            self._getMdParticles())

    def filterVolumeStep(self, volId, volFn):
        params = {"inputVol": volFn,
                  "filtVol": self._getVolFiltered(volId),
                  "highPass": self.partSet.getSamplingRate() / self.highPassFilter.get(),
                  "lowPass": self.partSet.getSamplingRate() / self.lowPassFilter.get()
                  }

        args = ' -i %(inputVol)s -o %(filtVol)s --fourier band_pass %(highPass)f %(lowPass)f'
        self.runJob('xmipp_transform_filter', args % params, numberOfMpi=1,
                    numberOfThreads=1)

    def projectionLibraryStep(self, volId):
        # Generate projections from this reconstruction
        volDir = self._getVolDir(volId)
        makePath(volDir)

        params = {"inputVol": self._getVolFiltered(volId),
                  "gallery": self._getGalleryStack(volId),
                  "sampling": self.partSet.getSamplingRate(),
                  "symmetry": self.symmetryGroup.get(),
                  "angSampling": self.angularSampling.get(),
                  "expParticles": self._getMdParticles()
                  }

        args = '-i %(inputVol)s -o %(gallery)s --sampling_rate %(angSampling)f --sym %(symmetry)s'
        args += ' --method fourier 1 0.25 bspline --compute_neighbors --angular_distance -1'
        args += ' --experimental_images %(expParticles)s --max_tilt_angle 180'

        self.runJob("xmipp_angular_project_library", args % params)

    def significantStep(self, volId):
        count=0
        GpuListCuda=''
        if self.useGpu.get():
            if self.useQueueForSteps() or self.useQueue():
                GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                GpuList = GpuList.split(",")
                for elem in GpuList:
                    GpuListCuda = GpuListCuda+str(count)+' '
                    count+=1
            else:
                GpuListAux = ''
                for elem in self.getGpuList():
                    GpuListCuda = GpuListCuda+str(count)+' '
                    GpuListAux = GpuListAux+str(elem)+','
                    count+=1
                os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux

        params = {"inputParts": self._getMdParticles(),
                  "symmetry": self.symmetryGroup.get(),
                  "angSampling": self.angularSampling.get(),
                  "orientations": self.numOrientations.get(),
                  "gallery": self._getGalleryMd(volId),
                  "outDir": self._getVolDir(volId),
                  "output": "angles_iter001_00.xmd",
                  "device": GpuListCuda,
                  }

        if not self.useGpu.get():
            args = ' -i %(inputParts)s --sym %(symmetry)s --angularSampling %(angSampling)0.3f --dontReconstruct'
            args += ' --useForValidation %(orientations)0.3f --initgallery  %(gallery)s --odir %(outDir)s --iter 1 --dontCheckMirrors'
            self.runJob('xmipp_reconstruct_significant', args % params)
        else:
            args = '-i %(inputParts)s -r %(gallery)s -o %(output)s --keepBestN %(orientations)f '
            args += '--odir %(outDir)s --dev %(device)s '
            self.runJob(CUDA_ALIGN_SIGNIFICANT, args % params, numberOfMpi=1)

    def projectionMatchingStep(self, volId):
        params = {"inputParts": self._getMdParticles(),
                  "outerRadius": self.partSet.getDimensions()[0] / 2,
                  "shift": self.partSet.getDimensions()[0] / 10,
                  "search5D": self.partSet.getDimensions()[0] / 10,
                  "gallery": self._getGalleryStack(volId),
                  "orientations": self.numOrientations.get(),
                  "output": self._getAnglesMd(volId)
                  }

        args = ' -i %(inputParts)s --Ri 0.0 --Ro %(outerRadius)0.3f --max_shift %(shift)0.3f --append'
        args += ' --search5d_shift %(search5D)0.3f --number_orientations %(orientations)0.3f -o %(output)s --ref %(gallery)s'

        self.runJob('xmipp_angular_projection_matching', args % params)

    def validationStep(self, volId):
        params = {"inputAngles": self._getAnglesMd(volId),
                  "filtVol": self._getVolFiltered(volId),
                  "symmetry": self.symmetryGroup.get(),
                  "significance": self.significanceNoise.get(),
                  "outDir": self._getVolDir(volId)
                  }

        args = ' --i %(inputAngles)s --volume %(filtVol)s --odir %(outDir)s'
        args += ' --significance_noise %(significance)0.2f --sym %(symmetry)s'

        if (self.alignmentMethod == SIGNIFICANT):
            args += ' --useSignificant '

        self.runJob('xmipp_validation_nontilt', args % params)

    def createOutputStep(self):
        outputVols = self._createSetOfVolumes()

        for vol in self._iterInputVols():
            volume = vol.clone()
            volDir = self._getVolDir(vol.getObjId())
            volPrefix = 'vol%03d_' % (vol.getObjId())
            validationMd = self._getExtraPath(volPrefix + 'validation.xmd')
            moveFile(join(volDir, 'validation.xmd'),
                     validationMd)
            clusterMd = self._getExtraPath(volPrefix + 'clusteringTendency.xmd')
            moveFile(join(volDir, 'clusteringTendency.xmd'), clusterMd)

            mData = md.MetaData(validationMd)
            weight = mData.getValue(md.MDL_WEIGHT, mData.firstObject())
            volume._xmipp_weight = Float(weight)
            volume.clusterMd = String(clusterMd)
            volume.cleanObjId()  # clean objects id to assign new ones inside the set
            outputVols.append(volume)

        outputVols.setSamplingRate(self.partSet.getSamplingRate())
        self._defineOutputs(outputVolumes=outputVols)
        self._defineTransformRelation(self.inputVolumes, outputVols)

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        validateMsgs = []
        # if there are Volume references, it cannot be empty.
        if self.inputVolumes.get() and not self.inputVolumes.hasValue():
            validateMsgs.append('Please provide an input reference volume.')
        if self.inputParticles.get() and not self.inputParticles.hasValue():
            validateMsgs.append('Please provide input particles.')
        if self.useGpu and not isXmippCudaPresent(CUDA_ALIGN_SIGNIFICANT):
            validateMsgs.append("You have asked to use GPU, but I cannot find the Xmipp GPU programs")
        return validateMsgs

    def _summary(self):
        summary = []

        if (not hasattr(self, 'outputVolumes')):
            summary.append("Output volumes not ready yet.")
        else:
            size = 0
            for i, vol in enumerate(self._iterInputVols()):
                size += 1
            summary.append("Volumes to validate: *%d* " % size)
            summary.append("Angular sampling: %s" % self.angularSampling.get())
            summary.append(
                "Significance value: %s" % self.significanceNoise.get())

        return summary

    def _methods(self):
        messages = []
        if (hasattr(self, 'outputVolumes')):
            messages.append(
                'The quality parameter(s) has been obtained using the approach [Vargas2014a] with angular sampling of %f and significant value of %f' % (
                self.angularSampling.get(), self.alpha.get()))
        return messages

    def _citations(self):
        return ['Vargas2014a']

    # --------------------------- UTILS functions --------------------------------------------
    def _getVolDir(self, volIndex):
        return self._getExtraPath('vol%03d' % volIndex)

    def _getVolFiltered(self, volIndex):
        return self._getVolDir(volIndex) + "_filt.vol"

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

    def _getMdParticles(self):
        return self._getPath('input_particles.xmd')

    def _getGalleryStack(self, volIndex):
        return join(self._getVolDir(volIndex), 'gallery.stk')

    def _getGalleryMd(self, volIndex):
        return join(self._getVolDir(volIndex), 'gallery.doc')

    def _getAnglesMd(self, volIndex):
        return join(self._getVolDir(volIndex), 'angles_iter001_00.xmd')

