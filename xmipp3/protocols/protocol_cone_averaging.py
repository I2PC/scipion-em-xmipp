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
from shlex import quote
from typing import Dict, Union, Set

from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
from pwem.objects import SetOfClasses2D

from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (
    PointerParam,
    IntParam,
    BooleanParam,
    StringParam,
)
from pyworkflow.constants import BETA

from xmipp3.base import XmippProtocol
from xmipp3.convert import writeSetOfParticles

from .protocol_average_estimation_gmm import (
    ROBUST_WEIGHT_COL,
    STD_ROBUST_WEIGHT_COL,
    GMM_WEIGHT_COL,
    WEIGHT_COLUMN_TO_ATTRIBUTE,
    EstimatorType,
    add_estimator_section,
    build_estimator_args,
)


class XmippProtConeAveraging(ProtClassify2D, XmippProtocol):
    _label = "cone_averaging"
    _lastUpdateVersion = VERSION_3_0
    _conda_env = "xmipp_pyTorch"
    _devStatus = BETA

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label="Input")
        form.addParam(
            "inputParticles",
            PointerParam,
            pointerClass="SetOfParticles",
            label="Input Particles",
            help="Set of particles to be read",
        )
        form.addParam(
            "correctCtf",
            BooleanParam,
            default=True,
            # expertLevel=LEVEL_ADVANCED,
            label="Correct CTF?",
            help="If you set to *Yes*, the CTF of the experimental particles will be corrected",
        )
        form.addParam(
            "symmetryGroup",
            StringParam,
            default="c1",
            label="Symmetry group",
            help=(
                "Symmetry group of the volume. Only 'cn' with n >= 1, "
                "and 'dn' with n >= 2 are currently supported."
            ),
        )
        form.addParam(
            "numberOfGroups",
            IntParam,
            label="Number of groups",
            help=(
                "The particles will be split into groups according to their "
                "orientations, grouping together those with similar viewing "
                "directions. This parameter determines the number of such groups "
                "that will be created."
            ),
            default=100,
        )
        form.addParam(
            "deduplicateReferences",
            BooleanParam,
            default=True,
            help=(
                "For volumes that have a non-trivial symmetry, the generated set of "
                "reference viewing directions that are used to group the particles "
                "may contain redundant directions. "
                "If you set this to 'yes', these redundancies will be eliminated. "
                "As a result, the final number of groups might be smaller "
                "than initially requested."
            ),
            label="Deduplicate references?",
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
        form.addParam(
            "groupingBatchSize",
            IntParam,
            label="Grouping Batch Size",
            help=(
                "Batch size used to process the particles when grouping them "
                "by viewing direction."
            ),
            default=1024,
            expertLevel=LEVEL_ADVANCED,
        )

        form.addParallelSection(threads=0, mpi=4)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputStep")
        self._insertFunctionStep("groupIntoConesStep")
        self._insertFunctionStep("prepareParticlesStep")
        self._insertFunctionStep("coneAveragingStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- UTILS functions -----------------------------
    def _getGroupByColumn(self):
        return "cone_group"

    def _getInputMdPath(self):
        return self._getTmpPath("inputParticles.xmd")

    def _getGroupingOutputStarPath(self):
        return self._getTmpPath("groupedParticles.star")

    def _getGroupingReferenceMetadataPath(self):
        return self._getExtraPath("coneReferences.star")

    def _getAveragingOutputStarPath(self):
        return self._getTmpPath("particlesWithWeights.star")

    def _getParticleStackPath(self):
        return self._getTmpPath("preparedParticles.mrcs")

    def _getCorrectedConeAveragesPath(self):
        return self._getExtraPath("correctedConeAverages.mrcs")

    def _getRawConeAveragesPath(self):
        return self._getExtraPath("rawConeAverages.mrcs")

    def _getParticleMdPath(self):
        return self._getTmpPath("preparedParticles.xmd")

    def _getCtfCorrectedStackPath(self):
        return self._getTmpPath("ctfCorrectedParticles.mrcs")

    def _getCtfCorrectedMdPath(self):
        return self._getTmpPath("ctfCorrectedParticles.xmd")

    def _getGmmDiagnosticsPath(self):
        return self._getExtraPath("gmmDiagnostics")

    def _getEstimatorType(self):
        return EstimatorType(self.estimatorType.get())

    def _getEstimatorWeightColumns(self):
        base_weight_columns = [ROBUST_WEIGHT_COL, STD_ROBUST_WEIGHT_COL]
        if self.gmmReweighting.get():
            return base_weight_columns + [GMM_WEIGHT_COL]
        return base_weight_columns

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        writeSetOfParticles(
            imgSet=self.inputParticles.get(),
            filename=self._getInputMdPath(),
        )

    def groupIntoConesStep(self):
        env = self.getCondaEnv()

        groupingArgs = (
            f"--input-xmd {self._getInputMdPath()} "
            f"--out-star {self._getGroupingOutputStarPath()} "
            f"--out-group-column {quote(str(self._getGroupByColumn()))} "
            f"--n-groups {self.numberOfGroups.get()} "
            f"--grouping-batch-size {self.groupingBatchSize.get()} "
            f"--symmetry-group {self.symmetryGroup.get()} "
            f"--out-reference-md {self._getGroupingReferenceMetadataPath()} "
        )

        if self.deduplicateReferences.get():
            groupingArgs += "--deduplicate-references "

        self.runJob("xmipp_cone_grouping", groupingArgs, env=env, numberOfMpi=1)

    def prepareParticlesStep(self):
        """
        Create the CTF-corrected, cone-aligned stack used by the GMM.

        The grouping script writes a temporary 2D geometry (psi, shifts and
        flip) into the grouped metadata. If requested, CTF correction is
        performed first in the original image coordinates. The geometric
        transform is then applied physically to the images, so the GMM does not
        need to read any alignment parameters.
        """
        geometryInput = self._getGroupingOutputStarPath()

        if self.correctCtf.get():
            samplingRate = self.inputParticles.get().getSamplingRate()
            ctfArgs = (
                f"-i '{geometryInput}' "
                f"-o '{self._getCtfCorrectedStackPath()}' "
                f"--save_metadata_stack '{self._getCtfCorrectedMdPath()}' "
                f"--keep_input_columns "
                f"--sampling_rate {samplingRate} "
            )

            if self.inputParticles.get().isPhaseFlipped():
                ctfArgs += "--phase_flipped "

            self.runJob(
                "xmipp_ctf_correct_wiener2d",
                ctfArgs,
                numberOfMpi=self.numberOfMpi.get(),
            )
            geometryInput = self._getCtfCorrectedMdPath()

        geometryArgs = (
            f"-i '{geometryInput}' "
            f"-o '{self._getParticleStackPath()}' "
            f"--save_metadata_stack {quote(str(self._getParticleMdPath()))} "
            f"--keep_input_columns "
            f"--apply_transform "
        )

        self.runJob(
            "xmipp_transform_geometry",
            geometryArgs,
            numberOfMpi=self.numberOfMpi.get(),
        )

    def coneAveragingStep(self):
        env = self.getCondaEnv()
        device = "cuda" if self.useGpu.get() else "cpu"

        args = (
            f"--input-xmd {quote(str(self._getParticleMdPath()))} "
            f"--base-xmd {quote(str(self._getInputMdPath()))} "
            f"--out-star {quote(str(self._getAveragingOutputStarPath()))} "
            f"--device {device} "
            f"--group-by-column {quote(str(self._getGroupByColumn()))} "
            f"--out-corrected-avgs {quote(str(self._getCorrectedConeAveragesPath()))} "
            f"--out-original-avgs {quote(str(self._getRawConeAveragesPath()))} "
        )

        args += build_estimator_args(self)

        self.runJob(
            "xmipp_gmm_average_estimation", args, env=env, numberOfMpi=1
        )

    def createOutputStep(self):
        outputMd = md.MetaData(self._getAveragingOutputStarPath())

        weightColumns = self._getEstimatorWeightColumns()

        weightsById: Dict[int, Dict[str, float]] = {}
        groupById: Dict[int, int] = {}
        nonEmptyGroups: Set[int] = set()
        for row in md.iterRows(outputMd):
            itemId = row.getValue(md.MDL_ITEM_ID)

            if itemId in weightsById:
                raise RuntimeError(
                    f"Duplicated itemId={itemId} in GMM output metadata."
                )

            weightsById[itemId] = {col: row.getValue(col) for col in weightColumns}

            group = int(row.getValue(self._getGroupByColumn()))
            groupById[itemId] = group
            nonEmptyGroups.add(group)

        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(self.inputParticles.get())
        inputParticles = self.inputParticles.get()

        for particle in inputParticles:
            itemId = particle.getObjId()

            try:
                weightsDict = weightsById[itemId]
            except KeyError as exc:
                raise RuntimeError(
                    f"No GMM weights found for particle " f"with itemId={itemId}."
                ) from exc

            outputParticle = particle.clone()

            for col in weightColumns:
                outputParticle.__setattr__(
                    WEIGHT_COLUMN_TO_ATTRIBUTE[col], Float(weightsDict[col])
                )

            outputParticle.setClassId(groupById[itemId])

            outputParticles.append(outputParticle)

        classValues = sorted(nonEmptyGroups)
        classIndex = {value: classValues.index(value) + 1 for value in classValues}

        standardClasses = self._createOutputClasses(
            particles=outputParticles,
            classIndex=classIndex,
            averagesPath=self._getRawConeAveragesPath(),
            suffix="_standard",
        )

        robustClasses = self._createOutputClasses(
            particles=outputParticles,
            classIndex=classIndex,
            averagesPath=self._getCorrectedConeAveragesPath(),
            suffix="_robust",
        )

        self._defineOutputs(outputParticles=outputParticles)
        self._defineSourceRelation(self.inputParticles, outputParticles)

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
