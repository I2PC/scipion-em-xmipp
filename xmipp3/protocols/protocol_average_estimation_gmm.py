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
from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from shlex import quote

import pwem.emlib.metadata as md
from pwem.constants import ALIGN_2D
from pwem.objects import SetOfClasses2D
from pwem.protocols import ProtClassify2D
from pyworkflow import VERSION_3_0
from pyworkflow.constants import BETA
from pyworkflow.object import Float
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (
    BooleanParam,
    EnumParam,
    FloatParam,
    IntParam,
    PointerParam,
)

from xmipp3.base import XmippProtocol
from xmipp3.convert import particleToRow


class EstimatorType(IntEnum):
    IRLS = 0
    FOURIER_IRLS = 1
    FOURIER_MASKED = 2
    ADMM = 3

    @property
    def label(self) -> str:
        """String representation used for CLI flags and protocol form choices"""
        return self.name.lower()


# Column names assigned by xmipp_robust_estimation_gmm
ROBUST_WEIGHT_COL = "wRobust"
STD_ROBUST_WEIGHT_COL = "wRobustStd"
GMM_WEIGHT_COL = "wRobustGmm"

# Map every xmipp column to the name this protocol should assign to it
WEIGHT_COLUMN_TO_ATTRIBUTE = {
    ROBUST_WEIGHT_COL: "_xmippRobustWeight",
    STD_ROBUST_WEIGHT_COL: "_xmippRobustWeightStandardized",
    GMM_WEIGHT_COL: "_xmippRobustWeightGmm",
}


def add_estimator_section(form):
    """
    Add all parameters related to estimator selection and configuration to the protocol form.
    """
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
            "Apply GMM reweighting to the results of the estimator."
            "GMM reweighting makes the estimator more aggressive in "
            "rejecting possibly misaligned or corrupted particles. This means "
            "it can slightly improve performance on more contaminated datasets."
        ),
        label="GMM Reweighting",
    )
    estimator_condition: dict[EstimatorType, str] = {
        estimator_type: f"bool(estimatorType == {estimator_type.value})"
        for estimator_type in EstimatorType
    }
    gmm_condition = "bool(gmmReweighting)"

    # Non-GMM parameters: number of iterations and low pass filter cutoff
    form.addParam(
        "estimatorIterations",
        IntParam,
        condition="not(" + gmm_condition + ")",
        default=30,
        help="Number of estimator iterations",
        label="Estimator iterations",
    )
    form.addParam(
        "lowpassCutoff",
        FloatParam,
        condition=estimator_condition[EstimatorType.FOURIER_MASKED],
        default=0.25,
        help="Normalized frequency cutoff for the estimator's low pass mask",
        label="Lowpass Cutoff",
    )

    # GMM parameters
    form.addParam(
        "internalEstimatorIterations",
        IntParam,
        condition=gmm_condition,
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
    # Degeneracy conditions
    form.addParam(
        "checkDegenerateGmm",
        BooleanParam,
        condition=gmm_condition,
        default=True,
        help=(
            "If using a GMM-type estimator, this option makes sure the GMM model "
            "is checked for degeneracy after the last iteration in each class. "
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
    # Initialization parameters
    form.addParam(
        "gmmInitialBadWeight",
        FloatParam,
        condition=gmm_condition,
        default=0.05,
        help="Initial weight for the GMM component with a lower mean weight",
        expertLevel=LEVEL_ADVANCED,
        label="Initial bad component weight",
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
        label="Initial bad component mean",
    )


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

        # Saving parameters
        form.addParam(
            "saveGmmFits",
            BooleanParam,
            default=False,
            help="Save extra files with information about GMM fits.",
            label="Save GMM fits?",
            expertLevel=LEVEL_ADVANCED,
        )
        form.addParam(
            "saveLowHighWeightAverages",
            BooleanParam,
            default=False,
            help="For each class, save two extra averages: the average of a certain proportion of the particles with the lowest weights and the same proportion with the highest weights.",
            label="Save averages of particles with lowest/highest weights?",
            expertLevel=LEVEL_ADVANCED,
        )
        form.addParam(
            "saveAverageDifferences",
            BooleanParam,
            default=False,
            help="For each class, save the difference between the regular average and the robust average.",
            label="Save average differences?",
            expertLevel=LEVEL_ADVANCED,
        )

        add_estimator_section(form)

        # TODO: instead of this, use the standard way to choose GPU ID in xmipp protocols
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
    def _getSelectedClassIds(self) -> list[int]:
        """Return the class identifiers selected for processing."""
        available_ids = sorted(int(cls.getObjId()) for cls in self.inputClasses.get())

        requested_id = self.classId.get()

        if requested_id <= 0:
            return available_ids

        if requested_id not in available_ids:
            raise ValueError(
                "Requested class ID is unavailable in the input classes object."
            )

        return [requested_id]

    def _buildMetadataStack(
        self,
        classes2D: SetOfClasses2D,
        classIds: list[int],
        outputMdPath: str | Path,
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

    def _correctCtf(
        self,
        inputMetadataPath: str | Path,
        outputParticlesPath: str | Path,
        outputMetadataPath: str | Path,
    ):
        """
        Apply Wiener CTF correction and write metadata for the corrected stack.

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
        inputClasses = self.inputClasses.get()
        args = [
            f"-i {quote(str(inputMetadataPath))}",
            f"-o {quote(str(outputParticlesPath))}",
            f"--save_metadata_stack {quote(str(outputMetadataPath))}",
            "--keep_input_columns",
            f"--sampling_rate {inputClasses.getSamplingRate()}",
        ]
        if inputClasses.getImages().isPhaseFlipped():
            args.append("--phase_flipped")

        self.runJob(
            "xmipp_ctf_correct_wiener2d",
            " ".join(args),
            numberOfMpi=self.numberOfMpi.get(),
        )

    def _applyAlignment(
        self,
        inputMetadataPath: str | Path,
        outputParticlesPath: str | Path,
        outputMetadataPath: str | Path,
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

        args = [
            f"-i {quote(inputMetadataPath)}",
            f"-o {quote(outputParticlesPath)}",
            f"--save_metadata_stack {quote(outputMetadataPath)}",
            "--keep_input_columns",
            "--apply_transform",
        ]
        self.runJob(
            "xmipp_transform_geometry",
            " ".join(args),
            numberOfMpi=self.numberOfMpi.get(),
        )

    def _getInputParticlesPath(self):
        return self._getExtraPath("inputParticles.xmd")

    def _getPreprocessedParticlesPath(self):
        return self._getExtraPath("preprocessed.mrcs")

    def _getPreprocessedMetadataPath(self):
        return self._getExtraPath("preprocessed.xmd")

    def _getGmmDiagnosticsPath(self):
        return self._getExtraPath("gmmDiagnostics")

    def _getLowWeightAveragesPath(self):
        return self._getExtraPath("lowWeightAverages.mrcs")

    def _getHighWeightAveragesPath(self):
        return self._getExtraPath("highWeightAverages.mrcs")

    def _getAverageDifferencesPath(self):
        return self._getExtraPath("averageDifferences.mrcs")

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
        alignmentInputPath = self._getInputParticlesPath()

        if self.correctCtf.get():
            correctedMetadataPath = self._getTmpPath("corrected.xmd")
            self._correctCtf(
                inputMetadataPath=alignmentInputPath,
                outputMetadataPath=correctedMetadataPath,
                outputParticlesPath=self._getTmpPath("corrected.mrcs"),
            )
            alignmentInputPath = correctedMetadataPath

        # Apply alignment to all images in the extracted stack
        self._applyAlignment(
            inputMetadataPath=alignmentInputPath,
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

        args = [
            f"--input-xmd {quote(self._getPreprocessedMetadataPath())}",
            f"--base-xmd {quote(self._getInputParticlesPath())}",
            f"--out-star {quote(outputStarPath)}",
            f"--out-corrected-avgs {quote(correctedAveragePath)}",
            f"--out-original-avgs {quote(originalAveragePath)}",
            f"--device {device}",
            "--overwrite",  # allow retries to replace any average stacks written by a previous attempt
        ]

        if self.saveLowHighWeightAverages.get():
            args.extend(
                [
                    f"--out-low-weight-avgs {quote(self._getLowWeightAveragesPath())}",
                    f"--out-high-weight-avgs {quote(self._getHighWeightAveragesPath())}",
                ]
            )

        if self.saveAverageDifferences.get():
            args.append(
                f"--out-difference-avgs {quote(self._getAverageDifferencesPath())}"
            )

        if self.gmmReweighting.get():
            args.extend(
                [
                    "--gmm",
                    f"--estimator-max-iter {self.internalEstimatorIterations.get()}",
                    f"--gmm-external-max-iter {self.gmmIterations.get()}",
                    f"--gmm-initial-bad-weight {self.gmmInitialBadWeight.get()}",
                    f"--gmm-initial-bad-quantile {self.gmmInitialBadQuantile.get()}",
                ]
            )

            if self.checkDegenerateGmm.get():
                args.extend(
                    [
                        "--gmm-check-degenerate",
                        f"--gmm-min-component-sep {self.gmmMinSep.get()}",
                        f"--gmm-min-good-weight {self.gmmMinWeight.get()}",
                    ]
                )
            else:
                args.append("--no-gmm-check-degenerate")

            if self.saveGmmFits.get():
                args.append(
                    f"--out-gmm-diagnostics {quote(self._getGmmDiagnosticsPath())}"
                )
        else:
            args.extend(
                [
                    "--no-gmm",
                    f"--estimator-max-iter {self.estimatorIterations.get()}",
                ]
            )

        estimatorType = self._getEstimatorType()
        if estimatorType == EstimatorType.FOURIER_MASKED:
            args.extend(
                [
                    "fourier_irls",
                    "--weight-approach per-image",
                    "--lowpass-mask",
                    f"--lowpass-mask-cutoff {self.lowpassCutoff.get()}",
                ]
            )
        else:
            args.append(estimatorType.label)

        self.runJob(
            "xmipp_gmm_average_estimation", " ".join(args), env=env, numberOfMpi=1
        )

    def createOutputStep(self):
        """
        Create the particle and class outputs of the protocol.

        Particle robust weights and class assignments are restored from the
        estimator metadata. Two sets of 2D classes are created using the robust
        and conventional class averages as representatives.
        """
        outputParticlesMd = md.MetaData(self._getExtraPath("particles.star"))

        weightColumns = self._getEstimatorWeightColumns()

        weightsById: dict[int, dict[str, float]] = {}
        for row in md.iterRows(outputParticlesMd):
            itemId = row.getValue(md.MDL_ITEM_ID)

            if itemId in weightsById:
                raise RuntimeError(
                    f"Duplicated itemId={itemId} in robust averaging output metadata."
                )

            weightsById[itemId] = {col: row.getValue(col) for col in weightColumns}

        # The estimation script writes averages following sorted class IDs
        classIds = self._getSelectedClassIds()
        classIndex = {classId: index for index, classId in enumerate(classIds, start=1)}

        outputParticles = self._createSetOfParticles()
        inputClasses = self.inputClasses.get()
        outputParticles.copyInfo(inputClasses.getImages())

        for cl in inputClasses:
            classId = cl.getObjId()

            # Skip classes that were not requested by the user
            if classId not in classIndex:
                continue

            for particle in cl:
                itemId = particle.getObjId()

                try:
                    weightsdict = weightsById[itemId]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Weights were not found for particle with itemId={itemId}."
                    ) from exc

                outputParticle = particle.clone()
                outputParticle.setClassId(classId)

                for col in weightColumns:
                    outputParticle.__setattr__(
                        WEIGHT_COLUMN_TO_ATTRIBUTE[col], Float(weightsdict[col])
                    )

                outputParticles.append(outputParticle)

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
        classIndex: dict[int, int],
        averagesPath: str | Path,
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
