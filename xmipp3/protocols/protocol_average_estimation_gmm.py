# ******************************************************************************
# *
# * Authors: Andres Contreras Santos (andres.contreras@cnb.csic.es)
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
from pathlib import Path
from shlex import join, quote
from typing import Union, Dict, List
from enum import IntEnum


from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_2D
from pwem.objects import SetOfClasses2D


from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam, EnumParam, FloatParam
from pyworkflow.constants import BETA
from xmipp3.base import XmippProtocol


from xmipp3.convert import particleToRow


class EstimatorType(IntEnum):
    IRLS = 0
    FOURIER_IRLS = 1
    FOURIER_MASKED = 2
    ADMM = 3
    NOISE_CORRECTED_COSINE = 4

    @property
    def label(self) -> str:
        """String representation used for CLI flags and protocol form choices"""
        return self.name.lower()

ROBUST_WEIGHT_COL = "wRobust"
STD_ROBUST_WEIGHT_COL = "wRobustStd"
GMM_WEIGHT_COL = "wRobustGmm"

WEIGHT_COLUMN_TO_ATTRIBUTE = {
    ROBUST_WEIGHT_COL: "_xmippRobustWeight",
    STD_ROBUST_WEIGHT_COL: "_xmippRobustWeightStandardized",
    GMM_WEIGHT_COL: "_xmippRobustWeightGmm",
}


def add_estimator_section(form):
    form.addSection(label="Estimator")
    form.addParam(
        "estimatorType",
        EnumParam,
        default=EstimatorType.IRLS.value,
        choices=[e.label for e in EstimatorType],
        help=("Type of robust estimator to use to compute the new class averages."),
        label="Estimator type",
    )
    form.addParam(
        "gmmReweighting",
        BooleanParam,
        default=True,
        help=(
            "Apply GMM reweighting to the results of the estimator. "
            "For noise-corrected cosine, use its native GMM weighting; "
            "disable this option for direct cosine weights. "
            "GMM reweighting makes the estimator more aggressive in "
            "rejecting possibly misaligned or corrupted particles. This means "
            "it can slightly improve performance on more contaminated datasets."
        ),
        label="GMM Reweighting",
    )
    estimator_condition: Dict[EstimatorType, str] = {
        estimator_type: f"bool(estimatorType == {estimator_type.value})" 
        for estimator_type in EstimatorType
    }
    gmm_condition = "bool(gmmReweighting)"
    
    form.addParam(
        "estimatorIterations",
        IntParam,
        condition="not(" + gmm_condition + ")",
        default=30,
        help="Number of estimator iterations",
        label="Estimator iterations"
    )
    form.addParam(
        "lowpassCutoff",
        FloatParam,
        condition=estimator_condition[EstimatorType.FOURIER_MASKED],
        default=0.25,
        help="Normalized frequency cutoff for the estimator's low pass mask",
        label="Lowpass Cutoff",
    )
    
    form.addParam(
        "internalEstimatorIterations",
        IntParam,
        condition=f"{gmm_condition} and estimatorType != {EstimatorType.NOISE_CORRECTED_COSINE.value}",
        default=1,
        help="Number of iterations for the internal estimator",
        label="Internal iterations",
    )
    form.addParam(
        "gmmIterations",
        IntParam,
        condition=gmm_condition,
        default=10,
        help="Number of GMM iterations",
        label="GMM iterations",
    )
    form.addParam(
        "checkDegenerateGmm",
        BooleanParam,
        condition=gmm_condition,
        default=True,
        help=(
            "If using a GMM-type estimator, this option makes sure the GMM model "
            "is checked for degeneracy at each iteration in each class. "
            "The model is considered degenerate if the two GMM components are too "
            "close together, or if the component associated with good particles "
            "has too little weight."
        ),
        label="Check GMM degeneracy?",
        expertLevel=LEVEL_ADVANCED,
    )
    form.addParam(
        "gmmMinSep",
        FloatParam,
        condition=gmm_condition,
        default=0.05,
        help="Minimum relative separation between GMM components.",
        label="Minimum GMM separation",
        expertLevel=LEVEL_ADVANCED,
    )
    form.addParam(
        "gmmMinWeight",
        FloatParam,
        condition=gmm_condition,
        default=0.6,
        help="Minimum weight for the good component of the GMM.",
        label="Minimum GMM good weight",
        expertLevel=LEVEL_ADVANCED,
    )
    form.addParam(
        "gmmInitialBadWeight",
        FloatParam,
        condition=gmm_condition,
        default=0.05,
        help="Initial weight for the GMM component with a lower mean weight",
        expertLevel=LEVEL_ADVANCED,
        label="Initial bad component weight"
    )
    form.addParam(
        "gmmInitialBadQuantile",
        FloatParam,
        condition=gmm_condition,
        default=0.05,
        help=(
            "Quantile used to calculate the mean for the GMM component with a "
            "lower mean weight. Because this is the component with a lower mean, "
            "the provided quantile should be between 0 and 0.5."
        ),
        expertLevel=LEVEL_ADVANCED,
        label="Initial bad component mean"
    )


    noise_condition = estimator_condition[EstimatorType.NOISE_CORRECTED_COSINE]
    form.addParam(
        "estimateNoiseVariance", BooleanParam, condition=noise_condition,
        default=True, label="Estimate noise variance?",
        help="Estimate input pixel noise variance with checkerboard MAD inside the mask. "
             "CTF correction and alignment interpolation can make the white-noise model approximate.",
    )
    form.addParam(
        "noiseVariance", FloatParam, condition=f"{noise_condition} and not estimateNoiseVariance",
        default=1.0, label="Input noise variance",
        help="Known per-pixel variance in the supplied image units, before score preprocessing. "
             "This is a variance, not a standard deviation.",
    )
    form.addParam(
        "poolNoiseVariance", BooleanParam, condition=f"{noise_condition} and estimateNoiseVariance",
        default=False, label="Pool noise estimates?",
        help="Use the median automatic variance for every image. Enable only for common noise variance.",
        expertLevel=LEVEL_ADVANCED,
    )
    for name, default, label, help_text in [
        ("noiseFilterSigma", 0.0, "Score smoothing width (pixels)",
         "Gaussian Fourier smoothing for scores only. Zero disables smoothing; averages use original images."),
        ("noiseMinFrequency", 0.0, "Minimum score frequency",
         "Radial frequency in cycles/pixel. Zero disables the lower cutoff."),
        ("noiseMaxFrequency", 0.0, "Maximum score frequency",
         "Radial frequency in cycles/pixel. Zero disables the upper cutoff. 0.25 is half axial Nyquist."),
        ("noiseMinSignalFraction", 0.05, "Minimum resolved signal fraction",
         "Require corrected signal energy above this fraction of expected noise energy. "
         "Unresolved images receive cosine zero."),
    ]:
        form.addParam(name, FloatParam, condition=noise_condition, default=default,
                      label=label, help=help_text, expertLevel=LEVEL_ADVANCED)
    form.addParam(
        "clipNoiseCosine", BooleanParam, condition=noise_condition, default=False,
        label="Clip corrected cosine?", expertLevel=LEVEL_ADVANCED,
        help="Clip scores to [-1,1]. Disabled by default to avoid point masses in the GMM. "
             "Direct weights always clip to [0,1].",
    )
    form.addParam(
        "noiseWeightPower", FloatParam, condition=f"{noise_condition} and not gmmReweighting",
        default=1.0, label="Cosine weight power", expertLevel=LEVEL_ADVANCED,
        help="Positive exponent applied to direct cosine weights; larger values increase contrast.",
    )


def build_estimator_args(protocol):
    """Build shared flags before the subcommand, then estimator-specific flags.

    Native noise-corrected GMM owns its outer loop: it must not be wrapped in
    --gmm, and gmmIterations controls --estimator-max-iter in that mode.
    """
    estimator = protocol._getEstimatorType()
    native = estimator == EstimatorType.NOISE_CORRECTED_COSINE
    use_gmm = protocol.gmmReweighting.get()
    args = []
    if use_gmm:
        if native:
            args += ["--estimator-max-iter", str(protocol.gmmIterations.get())]
        else:
            args += ["--gmm", "--estimator-max-iter", str(protocol.internalEstimatorIterations.get()),
                     "--gmm-external-max-iter", str(protocol.gmmIterations.get())]
        args += ["--gmm-initial-bad-weight", str(protocol.gmmInitialBadWeight.get()),
                 "--gmm-initial-bad-quantile", str(protocol.gmmInitialBadQuantile.get())]
        if protocol.checkDegenerateGmm.get():
            args += ["--gmm-check-degenerate", "--gmm-min-component-sep", str(protocol.gmmMinSep.get()),
                     "--gmm-min-good-weight", str(protocol.gmmMinWeight.get())]
        else:
            args += ["--no-gmm-check-degenerate"]
        if protocol.saveGmmFits.get():
            args += ["--out-gmm-diagnostics", protocol._getGmmDiagnosticsPath()]
    else:
        args += ["--no-gmm", "--estimator-max-iter", str(protocol.estimatorIterations.get())]

    if native:
        args += [estimator.label, "--weighting", "gmm" if use_gmm else "cosine",
                 "--filter-sigma", str(protocol.noiseFilterSigma.get()),
                 "--min-frequency", str(protocol.noiseMinFrequency.get()),
                 "--min-signal-fraction", str(protocol.noiseMinSignalFraction.get()),
                 "--clip-cosine" if protocol.clipNoiseCosine.get() else "--no-clip-cosine"]
        maximum = protocol.noiseMaxFrequency.get()
        if maximum != 0:
            args += ["--max-frequency", str(maximum)]
        if protocol.estimateNoiseVariance.get():
            args += ["--pool-noise" if protocol.poolNoiseVariance.get() else "--no-pool-noise"]
        else:
            args += ["--noise-variance", str(protocol.noiseVariance.get())]
        if not use_gmm:
            args += ["--weight-power", str(protocol.noiseWeightPower.get())]
    elif estimator == EstimatorType.FOURIER_MASKED:
        args += ["fourier_irls", "--weight-approach", "per-image", "--lowpass-mask",
                 "--lowpass-mask-cutoff", str(protocol.lowpassCutoff.get())]
    else:
        args += [estimator.label]
    return join(args)


class XmippProtAverageEstimationGmm(ProtClassify2D, XmippProtocol):
    """
    Improves class averages by using robust estimation. Different estimation
    techniques can be chosen, including a Gaussian Mixture Model (GMM) on
    the distances of each particle to a given reference (the class average).
    """

    _label = "estimation_gmm"
    _lastUpdateVersion = VERSION_3_0
    _conda_env = "xmipp_pyTorch"
    _devStatus = BETA

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label="Input")
        form.addParam(
            "inputClasses",
            PointerParam,
            pointerClass="SetOfClasses2D",
            label="Input Classes",
            help="Set of classes to be read",
        )
        form.addParam(
            "correctCtf",
            BooleanParam,
            default=True,
            label="Correct CTF?",
            help="If you set to *Yes*, the CTF of the experimental particles will be corrected",
        )
        form.addParam(
            "classId",
            IntParam,
            default=-1,
            label="Class ID",
            help="Class to select for average estimation. "
            "Zero or any negative value means the estimation "
            "will be applied to all classes",
            expertLevel=LEVEL_ADVANCED,
        )
        form.addParam(
            "saveGmmFits",
            BooleanParam,
            default=False,
            help="Save extra files with information about GMM fits.",
            label="Save GMM fits?",
            expertLevel=LEVEL_ADVANCED,
        )

        add_estimator_section(form)

        form.addSection(label="Compute")
        form.addParam(
            "useGpu",
            BooleanParam,
            default=True,
            label="Use GPU?",
            help="If you set to *Yes*, the estimation process will try to use the GPU "
            "for hardware acceleration. This might speed up the process if CUDA is available.",
        )

        form.addParallelSection(threads=0, mpi=4)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputStep")
        self._insertFunctionStep("preprocessStep")
        self._insertFunctionStep("averageEstimationStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- UTILS functions -----------------------------
    def _getSelectedClassIds(self) -> List[int]:
        """Return the class identifiers selected for processing."""
        available_ids = sorted(int(cls.getObjId()) for cls in self.inputClasses.get())

        requested_id = self.classId.get()

        if requested_id <= 0:
            return available_ids

        if requested_id not in available_ids:
            raise ValueError(
                "Requested class ID is unavailable " "in the input classes object."
            )

        return [requested_id]

    def _buildMetadataStack(
        self,
        classes2D: SetOfClasses2D,
        classIds: List[int],
        outputMdPath: Union[str, Path],
    ):
        """
        Write particles from selected 2D classes to a single Xmipp metadata file.

        The output metadata contains the particle locations, CTF parameters,
        2D alignment parameters, and class identifiers.

        Parameters
        ----------
        classes2D : SetOfClasses2D
            Input set containing the particle classes.
        classIds : list of int
            Identifiers of the classes to include in the output metadata.
        outputMdPath : str or pathlib.Path
            Path where the Xmipp metadata file will be written.
        """
        particlesMd = md.MetaData()

        for cls in classes2D:
            classId = int(cls.getObjId())

            if classId not in classIds:
                continue

            for particle in cls:
                row = md.Row()

                particleToRow(
                    particle,
                    row,
                    alignType=ALIGN_2D,
                    writeCtf=True,
                    writeAcquisition=True,
                )

                row.setValue(md.MDL_REF, classId)

                row.addToMd(particlesMd)

        outputMdPath = str(Path(outputMdPath))
        particlesMd.write(outputMdPath)

    def _prepareParticleStack(
        self,
        inputMetadataPath: Union[str, Path],
        outputParticlesPath: Union[str, Path],
        outputMetadataPath: Union[str, Path],
    ):
        """
        If CTF correction is enabled, the images are corrected with
        ``xmipp_ctf_correct_wiener2d``. Otherwise, they are copied to a standalone
        stack with ``xmipp_image_convert``. In both cases, an accompanying metadata
        file referencing the newly generated stack is written.

        Parameters
        ----------
        inputMetadataPath : str or pathlib.Path
            Metadata file containing the particles to process.
        outputParticlesPath : str or pathlib.Path
            Path of the output particle stack.
        outputMetadataPath : str or pathlib.Path
            Path of the metadata file referencing the output stack.

        Notes
        -----
        This method does not apply the class alignment. Alignment is handled
        separately by ``_applyAlignment``.
        """
        inputMetadataPath = str(Path(inputMetadataPath))
        outputParticlesPath = str(Path(outputParticlesPath))
        outputMetadataPath = str(Path(outputMetadataPath))

        args = (
            f"-i {inputMetadataPath} "
            f"-o {outputParticlesPath} "
            f"--save_metadata_stack {outputMetadataPath} "
            f"--keep_input_columns "
        )

        if self.correctCtf.get():
            args += f"--sampling_rate {self.inputClasses.get().getSamplingRate()}"

            if self.inputClasses.get().getImages().isPhaseFlipped():
                args += "--phase_flipped "

            self.runJob(
                "xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get()
            )
        else:
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)

    def _applyAlignment(
        self,
        inputMetadataPath: Union[str, Path],
        outputParticlesPath: Union[str, Path],
        outputMetadataPath: Union[str, Path],
    ):
        """
        Apply the 2D transforms stored in a particle metadata file.

        The transformed images are written to a new stack using
        ``xmipp_transform_geometry --apply_transform``. A corresponding metadata
        file referencing the aligned stack is also generated.

        Parameters
        ----------
        inputMetadataPath : str or pathlib.Path
            Metadata containing the images and their 2D alignment parameters.
        outputParticlesPath : str or pathlib.Path
            Path of the aligned particle stack.
        outputMetadataPath : str or pathlib.Path
            Path of the metadata file referencing the aligned images.
        """
        inputMetadataPath = str(Path(inputMetadataPath))
        outputParticlesPath = str(Path(outputParticlesPath))
        outputMetadataPath = str(Path(outputMetadataPath))

        args = (
            f"-i {inputMetadataPath} "
            f"-o {outputParticlesPath} "
            f"--save_metadata_stack {outputMetadataPath} "
            f"--keep_input_columns "
            f"--apply_transform"
        )
        self.runJob(
            "xmipp_transform_geometry", args, numberOfMpi=self.numberOfMpi.get()
        )

    def _getInputParticlesPath(self):
        return self._getExtraPath("inputParticles.xmd")

    def _getPreprocessedParticlesPath(self):
        return self._getExtraPath("preprocessed.mrcs")

    def _getPreprocessedMetadataPath(self):
        return self._getExtraPath("preprocessed.xmd")

    def _getGmmDiagnosticsPath(self):
        return self._getExtraPath("gmmDiagnostics")

    def _getEstimatorType(self) -> EstimatorType:
        return EstimatorType(self.estimatorType.get())

    def _getEstimatorWeightColumns(self):
        base_weight_columns = [ROBUST_WEIGHT_COL, STD_ROBUST_WEIGHT_COL]
        if self.gmmReweighting.get():
            return base_weight_columns + [GMM_WEIGHT_COL]
        return base_weight_columns

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        """
        Convert the selected input classes to a single Xmipp particle metadata file.

        The generated metadata contains the image location, CTF parameters,
        2D alignment parameters, and class identifier for every selected particle.
        """
        inputClasses = self.inputClasses.get()

        # Build metadata file with all the requested class particles
        self._buildMetadataStack(
            classes2D=inputClasses,
            classIds=self._getSelectedClassIds(),
            outputMdPath=self._getInputParticlesPath(),
        )

    def preprocessStep(self):
        """
        Preprocess all selected particles before average estimation.

        The particles are optionally CTF-corrected, collected into a single
        image stack, and transformed according to their stored 2D alignments.
        """
        particlesMdPath = self._getInputParticlesPath()

        # Correct CTF (if requested), align particles and extract particle stack
        self._prepareParticleStack(
            inputMetadataPath=particlesMdPath,
            outputMetadataPath=self._getTmpPath("corrected.xmd"),
            outputParticlesPath=self._getTmpPath("corrected.mrcs"),
        )

        # Apply alignment to all images in the extracted stack
        self._applyAlignment(
            inputMetadataPath=self._getTmpPath("corrected.xmd"),
            outputMetadataPath=self._getPreprocessedMetadataPath(),
            outputParticlesPath=self._getPreprocessedParticlesPath(),
        )

    def averageEstimationStep(self):
        """
        Estimate robust and conventional averages for all selected classes.

        The external robust estimator processes the preprocessed particles grouped
        by class and writes particle weights together with the resulting class
        average stacks.
        """
        env = self.getCondaEnv()
        device = "cuda" if self.useGpu.get() else "cpu"

        # Prepare output paths for the star file with weights and averages
        outputStarPath = self._getExtraPath("particles.star")
        correctedAveragePath = self._getExtraPath("corrected_avgs.mrcs")
        originalAveragePath = self._getExtraPath("original_avgs.mrcs")

        # Run the GMM average estimation script for all classes
        args = (
            f"--input-xmd {quote(str(self._getPreprocessedMetadataPath()))} "
            f"--base-xmd {quote(str(self._getInputParticlesPath()))} "
            f"--out-star {quote(str(outputStarPath))} "
            f"--out-corrected-avgs {quote(str(correctedAveragePath))} "
            f"--out-original-avgs {quote(str(originalAveragePath))} "
            f"--device {device} "
        )

        args += build_estimator_args(self)

        self.runJob("xmipp_gmm_average_estimation", args, env=env, numberOfMpi=1)

    def createOutputStep(self):
        """
        Create the particle and class outputs of the protocol.

        Particle robust weights and class assignments are restored from the
        estimator metadata. Two sets of 2D classes are created using the robust
        and conventional class averages as representatives.
        """
        outputParticlesMd = md.MetaData(self._getExtraPath("particles.star"))

        weightColumns = self._getEstimatorWeightColumns()

        weightsById: Dict[int, Dict[str, float]] = {}
        for row in md.iterRows(outputParticlesMd):
            itemId = row.getValue(md.MDL_ITEM_ID)

            if itemId in weightsById:
                raise RuntimeError(
                    f"Duplicated itemId={itemId} in robust averaging output metadata."
                )

            weightsById[itemId] = {col: row.getValue(col) for col in weightColumns}

        outputParticles = self._createSetOfParticles()
        inputClasses = self.inputClasses.get()
        outputParticles.copyInfo(inputClasses.getImages())

        for cl in inputClasses:
            classId = cl.getObjId()

            for particle in cl:
                itemId = particle.getObjId()

                try:
                    weightsDict = weightsById[itemId]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Weights were not found for particle with itemId={itemId}."
                    ) from exc

                outputParticle = particle.clone()
                outputParticle.setClassId(classId)

                for col in weightColumns:
                    outputParticle.__setattr__(
                        WEIGHT_COLUMN_TO_ATTRIBUTE[col], Float(weightsDict[col])
                    )

                outputParticles.append(outputParticle)

        # The estimation script writes averages following sorted class IDs.
        classIds = self._getSelectedClassIds()
        classIndex = {classId: index for index, classId in enumerate(classIds, start=1)}

        # Create classes with robust averages as representatives
        robustClasses = self._createOutputClasses(
            particles=outputParticles,
            classIndex=classIndex,
            averagesPath=self._getExtraPath("corrected_avgs.mrcs"),
            suffix="_robust",
        )

        # Create classes with ordinary averages as representatives
        standardClasses = self._createOutputClasses(
            particles=outputParticles,
            classIndex=classIndex,
            averagesPath=self._getExtraPath("original_avgs.mrcs"),
            suffix="_standard",
        )

        # Define protocol outputs
        self._defineOutputs(outputParticles=outputParticles)
        self._defineSourceRelation(self.inputClasses, outputParticles)

        self._defineOutputs(outputClasses_robust=robustClasses)
        self._defineSourceRelation(outputParticles, robustClasses)

        self._defineOutputs(outputClasses_standard=standardClasses)
        self._defineSourceRelation(outputParticles, standardClasses)

    def _createOutputClasses(
        self,
        particles,
        classIndex: Dict[int, int],
        averagesPath: Union[str, Path],
        suffix: str,
    ) -> SetOfClasses2D:
        """
        Create a set of 2D classes using a stack of class averages as representatives.

        Parameters
        ----------
        particles : SetOfParticles
            Particles to classify according to their stored class identifiers.
        classIndex : dict of int to int
            Mapping from class identifiers to 1-based image indices in the
            average stack.
        averagesPath : str or pathlib.Path
            Path to the stack containing the class representative images.
        suffix : str
            Suffix used to identify the generated Scipion output set.

        Returns
        -------
        SetOfClasses2D
            Set of 2D classes with the requested averages as representatives.
        """
        outputClasses = self._createSetOfClasses2D(particles, suffix)

        samplingRate = particles.getSamplingRate()
        averagesPath = str(Path(averagesPath))

        def updateClass(classItem):
            classId = classItem.getObjId()

            representative = classItem.getRepresentative()
            representative.setLocation(classIndex[classId], averagesPath)
            representative.setSamplingRate(samplingRate)

        outputClasses.classifyItems(updateClassCallback=updateClass)

        return outputClasses
