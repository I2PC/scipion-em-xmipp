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
from shlex import join

import pwem.emlib.metadata as md
from pwem.constants import ALIGN_2D
from pwem.objects import Class2D, Particle, SetOfClasses2D
from pwem.protocols import ProtClassify2D
from pyworkflow import VERSION_3_0
from pyworkflow.constants import BETA
from pyworkflow.object import Float
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (
    GPU_LIST,
    USE_GPU,
    BooleanParam,
    EnumParam,
    FloatParam,
    IntParam,
    PointerParam,
    StringParam,
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
        help="Type of robust estimator to use to compute the new class averages.",
        label="Estimator type",
    )
    form.addParam(
        "gmmReweighting",
        BooleanParam,
        default=True,
        help=(
            "Apply GMM reweighting to the results of the estimator. "
            "GMM reweighting makes the estimator more aggressive in "
            "rejecting possibly misaligned or corrupted particles. This means "
            "it can slightly improve performance on more contaminated datasets."
        ),
        label="GMM Reweighting",
    )

    form.addParam(
        "estimatorIterations",
        IntParam,
        condition="not gmmReweighting",
        default=30,
        help="Number of estimator iterations",
        label="Estimator iterations",
    )
    form.addParam(
        "lowpassCutoff",
        FloatParam,
        condition=f"estimatorType == {EstimatorType.FOURIER_MASKED.value}",
        default=0.25,
        help="Normalized frequency cutoff for the estimator's low pass mask",
        label="Lowpass Cutoff",
    )

    # GMM parameters
    form.addParam(
        "internalEstimatorIterations",
        IntParam,
        condition="gmmReweighting",
        default=1,
        help="Number of iterations for the internal estimator",
        label="Internal iterations",
    )
    form.addParam(
        "gmmIterations",
        IntParam,
        condition="gmmReweighting",
        default=10,
        help="Number of GMM iterations",
        label="GMM iterations",
    )
    # Degeneracy conditions
    form.addParam(
        "checkDegenerateGmm",
        BooleanParam,
        condition="gmmReweighting",
        default=True,
        help=(
            "Check the GMM model for degeneracy during fitting. "
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
        condition="gmmReweighting and checkDegenerateGmm",
        default=0.05,
        help="Minimum relative separation between GMM components.",
        label="Minimum GMM separation",
        expertLevel=LEVEL_ADVANCED,
    )
    form.addParam(
        "gmmMinWeight",
        FloatParam,
        condition="gmmReweighting and checkDegenerateGmm",
        default=0.6,
        help="Minimum weight for the good component of the GMM.",
        label="Minimum GMM good weight",
        expertLevel=LEVEL_ADVANCED,
    )
    # Initialization parameters
    form.addParam(
        "gmmInitialBadWeight",
        FloatParam,
        condition="gmmReweighting",
        default=0.05,
        help="Initial weight for the GMM component with a lower mean weight",
        expertLevel=LEVEL_ADVANCED,
        label="Initial bad component weight",
    )
    form.addParam(
        "gmmInitialBadQuantile",
        FloatParam,
        condition="gmmReweighting",
        default=0.05,
        help=(
            "Quantile used to calculate the mean for the GMM component with a "
            "lower mean weight. Because this is the component with a lower mean, "
            "the provided quantile should be between 0 and 0.5."
        ),
        expertLevel=LEVEL_ADVANCED,
        label="Initial bad component mean",
    )


def build_estimator_args(protocol) -> list[str]:
    """Return shared CLI tokens for the configured estimator."""
    if protocol.gmmReweighting.get():
        args = [
            "--gmm",
            "--estimator-max-iter",
            str(protocol.internalEstimatorIterations.get()),
            "--gmm-external-max-iter",
            str(protocol.gmmIterations.get()),
            "--gmm-initial-bad-weight",
            str(protocol.gmmInitialBadWeight.get()),
            "--gmm-initial-bad-quantile",
            str(protocol.gmmInitialBadQuantile.get()),
        ]
        if protocol.checkDegenerateGmm.get():
            args.extend(
                [
                    "--gmm-check-degenerate",
                    "--gmm-min-component-sep",
                    str(protocol.gmmMinSep.get()),
                    "--gmm-min-good-weight",
                    str(protocol.gmmMinWeight.get()),
                ]
            )
        else:
            args.append("--no-gmm-check-degenerate")
    else:
        args = [
            "--no-gmm",
            "--estimator-max-iter",
            str(protocol.estimatorIterations.get()),
        ]

    estimator = protocol._getEstimatorType()
    if estimator == EstimatorType.FOURIER_MASKED:
        args.extend(
            [
                "fourier_irls",
                "--weight-approach",
                "per-image",
                "--lowpass-mask",
                "--lowpass-mask-cutoff",
                str(protocol.lowpassCutoff.get()),
            ]
        )
    else:
        args.append(estimator.label)

    return args


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
        form.addHidden(
            USE_GPU,
            BooleanParam,
            default=True,
            label="Use GPU for execution",
            help="Select *Yes* to run the averaging estimator on a GPU or *No* to run it on the CPU.",
        )
        form.addHidden(
            GPU_LIST,
            StringParam,
            default="0",
            expertLevel=LEVEL_ADVANCED,
            label="Choose GPU IDs",
            help=(
                "GPU IDs available for local execution. The estimator uses the "
                "first visible GPU. For queued jobs, GPU allocation is handled "
                "by the scheduler."
            ),
        )

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
            "saveLowHighWeightAverages",
            BooleanParam,
            default=False,
            help="For each class, save two extra averages: the average of a certain proportion of the particles with the lowest weights and the same proportion with the highest weights.",
            label="Save averages of particles with lowest/highest weights?",
            expertLevel=LEVEL_ADVANCED,
        )

        add_estimator_section(form)
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

        particlesMd.write(str(outputMdPath))

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
            "-i",
            str(inputMetadataPath),
            "-o",
            str(outputParticlesPath),
            "--save_metadata_stack",
            str(outputMetadataPath),
            "--keep_input_columns",
            "--sampling_rate",
            str(inputClasses.getSamplingRate()),
        ]
        if inputClasses.getImages().isPhaseFlipped():
            args.append("--phase_flipped")

        self.runJob(
            "xmipp_ctf_correct_wiener2d", join(args), numberOfMpi=self.numberOfMpi.get()
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
        args = [
            "-i",
            str(inputMetadataPath),
            "-o",
            str(outputParticlesPath),
            "--save_metadata_stack",
            str(outputMetadataPath),
            "--keep_input_columns",
            "--apply_transform",
        ]
        self.runJob(
            "xmipp_transform_geometry", join(args), numberOfMpi=self.numberOfMpi.get()
        )

    def _getInputParticlesPath(self):
        return self._getTmpPath("inputParticles.xmd")

    def _getPreprocessedParticlesPath(self):
        return self._getTmpPath("preprocessed.mrcs")

    def _getPreprocessedMetadataPath(self):
        return self._getTmpPath("preprocessed.xmd")

    def _getLowWeightAveragesPath(self):
        return self._getExtraPath("lowWeightAverages.mrcs")

    def _getHighWeightAveragesPath(self):
        return self._getExtraPath("highWeightAverages.mrcs")

    def _getOutputStarPath(self):
        return self._getTmpPath("particles.star")

    def _getCorrectedAveragesPath(self):
        return self._getExtraPath("corrected_avgs.mrcs")

    def _getOriginalAveragesPath(self):
        return self._getExtraPath("original_avgs.mrcs")

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
        use_gpu = self.usesGpu()
        device = "cuda" if use_gpu else "cpu"
        if use_gpu and not self.useQueue():
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.getGpuList()))

        # Prepare output paths for the star file with weights and averages
        outputStarPath = self._getOutputStarPath()
        correctedAveragePath = self._getCorrectedAveragesPath()
        originalAveragePath = self._getOriginalAveragesPath()

        args = [
            "--input-xmd",
            str(self._getPreprocessedMetadataPath()),
            "--out-star",
            str(outputStarPath),
            "--out-corrected-avgs",
            str(correctedAveragePath),
            "--out-original-avgs",
            str(originalAveragePath),
            "--device",
            device,
            "--overwrite",  # allow retries to replace any average stacks written by a previous attempt
        ]

        if self.saveLowHighWeightAverages.get():
            args.extend(
                [
                    "--out-low-weight-avgs",
                    str(self._getLowWeightAveragesPath()),
                    "--out-high-weight-avgs",
                    str(self._getHighWeightAveragesPath()),
                ]
            )

        args.extend(build_estimator_args(self))

        self.runJob("xmipp_gmm_average_estimation", join(args), env=env, numberOfMpi=1)

    def _readOutputWeights(self, weightColumns):
        """Return weights keyed by (item ID, class ID) and sorted fitted class IDs."""
        outputParticlesMd = md.MetaData(self._getOutputStarPath())
        columnNames = ["itemId", "ref", *weightColumns]
        labels = [
            md.MDL_ITEM_ID,
            md.MDL_REF,
            *(md.str2Label(col) for col in weightColumns),
        ]
        for name, label in zip(columnNames, labels):
            if label < 0 or not outputParticlesMd.containsLabel(label):
                raise RuntimeError(
                    f"Missing column {name} in robust averaging output metadata."
                )

        columns = [outputParticlesMd.getColumnValues(label) for label in labels]

        weightsById = {}
        fittedClassIds = set()
        for ids, weights in zip(zip(*columns[:2]), zip(*columns[2:])):
            if ids in weightsById:
                raise RuntimeError(
                    f"Duplicated pair itemId={ids[0]}, classId={ids[1]} in "
                    "robust averaging output metadata."
                )
            weightsById[ids] = weights
            fittedClassIds.add(ids[1])

        return weightsById, sorted(fittedClassIds)

    def createOutputStep(self):
        """Create weighted 2D classes with robust and conventional representatives."""
        weightColumns = self._getEstimatorWeightColumns()
        weightsById, fittedClassIds = self._readOutputWeights(weightColumns)
        weightAttributes = [WEIGHT_COLUMN_TO_ATTRIBUTE[col] for col in weightColumns]

        # The estimation script writes averages following sorted class IDs
        selectedIds = set(self._getSelectedClassIds())
        classIndex = {
            classId: index for index, classId in enumerate(fittedClassIds, start=1)
        }

        inputClasses = self.inputClasses.get()
        imagesPointer = inputClasses.getImagesPointer()
        robustClasses = self._createSetOfClasses2D(imagesPointer, suffix="_robust")
        standardClasses = self._createSetOfClasses2D(imagesPointer, suffix="_standard")

        for inputClass in inputClasses:
            classId = inputClass.getObjId()
            if classId not in selectedIds or inputClass.getSize() == 0:
                continue
            if classId not in classIndex:
                raise RuntimeError(
                    f"No fitted particles were found for classId={classId}."
                )

            stackIndex = classIndex[classId]
            robustClass = self._addOutputClass(
                robustClasses, inputClass, stackIndex, self._getCorrectedAveragesPath()
            )
            standardClass = self._addOutputClass(
                standardClasses, inputClass, stackIndex, self._getOriginalAveragesPath()
            )

            for particle in inputClass:
                itemId = particle.getObjId()
                try:
                    weights = weightsById[(itemId, classId)]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Weights were not found for itemId={itemId}, classId={classId}."
                    ) from exc

                weighted = particle.clone()
                weighted.setClassId(classId)
                for attribute, weight in zip(weightAttributes, weights):
                    setattr(weighted, attribute, Float(weight))

                robustClass.append(weighted)
                standardClass.append(weighted.clone())

            robustClasses.update(robustClass)
            standardClasses.update(standardClass)

        robustClasses.write()
        standardClasses.write()

        self._defineOutputs(
            outputClasses_robust=robustClasses, outputClasses_standard=standardClasses
        )
        self._defineSourceRelation(self.inputClasses, robustClasses)
        self._defineSourceRelation(self.inputClasses, standardClasses)

    def _addOutputClass(
        self,
        outputClasses: SetOfClasses2D,
        inputClass: Class2D,
        stackIndex: int,
        averagesPath: str | Path,
    ) -> Class2D:
        outputClass = Class2D(objId=inputClass.getObjId())
        outputClass.copyInfo(inputClass)

        representative = Particle()
        representative.setLocation(stackIndex, str(averagesPath))
        representative.setSamplingRate(inputClass.getSamplingRate())
        outputClass.setRepresentative(representative)

        outputClasses.append(outputClass)
        return outputClass
