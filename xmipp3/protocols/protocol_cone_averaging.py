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
from typing import Dict, Union

from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
from pwem.objects import SetOfClasses2D

from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam, EnumParam
from pyworkflow.constants import BETA

from xmipp3.base import XmippProtocol
from xmipp3.convert import writeSetOfParticles, rowToParticle

from .protocol_average_estimation_gmm import ESTIMATORS


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
            "useGpu",
            BooleanParam,
            default=True,
            # expertLevel=LEVEL_ADVANCED,
            label="Use GPU?",
            help="If you set to *Yes*, the estimation process will try to use the GPU "
            "for hardware acceleration. This might speed up the process if CUDA is available.",
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
            expertLevel=LEVEL_ADVANCED,
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
        form.addParam(
            "estimatorType",
            EnumParam,
            default=0,
            # ESTIMATORS is dict[int, str], this ensures the list keeps correct order
            choices=[ESTIMATORS[i] for i in range(len(ESTIMATORS))],
            help=(
                "Type of robust estimator to use to compute the new class averages. "
                "As a rule of thumb, the 'gmm' estimator should be more aggressive in "
                "rejecting possibly misaligned or corrupted particles. This means "
                "its performance can be better for more contaminated datasets, and "
                "slightly worse in very clean datasets."
                "'irls' and 'fourier_irls' should both be relatively fast and less "
                "aggresive in particle rejection. 'admm' combines both 'irls' and "
                "'fourier_irls', and it can improve their results at the cost of "
                "more computation time."
            ),
            label="Estimator type",
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
        return self._getExtraPath("inputParticles.xmd")

    def _getGroupingOutputStarPath(self):
        return self._getExtraPath("groupedParticles.star")

    def _getAveragingOutputStarPath(self):
        return self._getExtraPath("particlesWithWeights.star")

    def _getParticleStackPath(self):
        return self._getTmpPath("preparedParticles.mrcs")

    def _getCorrectedConeAveragesPath(self):
        return self._getExtraPath("correctedConeAverages.mrcs")

    def _getRawConeAveragesPath(self):
        return self._getExtraPath("rawConeAverages.mrcs")

    def _getParticleMdPath(self):
        return self._getExtraPath("preparedParticles.xmd")

    def _getCtfCorrectedStackPath(self):
        return self._getTmpPath("ctfCorrectedParticles.mrcs")

    def _getCtfCorrectedMdPath(self):
        return self._getExtraPath("ctfCorrectedParticles.xmd")

    def _getEstimatorType(self):
        return ESTIMATORS[self.estimatorType.get()]

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        writeSetOfParticles(
            imgSet=self.inputParticles.get(),
            filename=self._getInputMdPath(),
        )

    def groupIntoConesStep(self):
        env = self.getCondaEnv()

        args = (
            f"--input-xmd {self._getInputMdPath()} "
            f"--out-star {self._getGroupingOutputStarPath()} "
            f"--out-group-column '{self._getGroupByColumn()}' "
            f"--n-groups {self.numberOfGroups.get()} "
            f"--grouping-batch-size {self.groupingBatchSize.get()} "
        )

        self.runJob("xmipp_cone_grouping", args, env=env, numberOfMpi=1)

    def prepareParticlesStep(self):
        """
        Create the CTF-corrected, cone-aligned stack used by the GMM.

        The grouping script writes a temporary 2D geometry (psi, shifts and
        flip) into the grouped metadata. If requested, CTF correction is
        performed first in the original image coordinates. The geometric
        transform is then applied physically to the images, so the GMM does not
        need to read any alignment parameters.
        """
        geometry_input = self._getGroupingOutputStarPath()

        if self.correctCtf.get():
            sampling_rate = self.inputParticles.get().getSamplingRate()
            ctf_args = (
                f"-i '{geometry_input}' "
                f"-o '{self._getCtfCorrectedStackPath()}' "
                f"--save_metadata_stack '{self._getCtfCorrectedMdPath()}' "
                f"--keep_input_columns "
                f"--sampling_rate {sampling_rate} "
            )

            if self.inputParticles.get().isPhaseFlipped():
                ctf_args += "--phase_flipped "

            self.runJob(
                "xmipp_ctf_correct_wiener2d",
                ctf_args,
                numberOfMpi=self.numberOfMpi.get(),
            )
            geometry_input = self._getCtfCorrectedMdPath()

        geometry_args = (
            f"-i '{geometry_input}' "
            f"-o '{self._getParticleStackPath()}' "
            f"--save_metadata_stack '{self._getParticleMdPath()}' "
            f"--keep_input_columns "
            f"--apply_transform "
        )

        self.runJob(
            "xmipp_transform_geometry",
            geometry_args,
            numberOfMpi=self.numberOfMpi.get(),
        )

    def coneAveragingStep(self):
        env = self.getCondaEnv()
        device = "cuda" if self.useGpu.get() else "cpu"

        script_args = (
            f"--input-xmd '{self._getParticleMdPath()}' "
            f"--base-xmd '{self._getInputMdPath()}' "
            f"--out-star '{self._getAveragingOutputStarPath()}' "
            f"--device {device} "
            f"--group-by-column '{self._getGroupByColumn()}' "
            f"--estimator-type '{self._getEstimatorType()}' "
            f"--out-corrected-avgs '{self._getCorrectedConeAveragesPath()}' "
            f"--out-original-avgs '{self._getRawConeAveragesPath()}' "
        )
        self.runJob("xmipp_gmm_average_estimation", script_args, env=env, numberOfMpi=1)

    def createOutputStep(self):
        outputMd = md.MetaData(self._getAveragingOutputStarPath())

        weights_by_id = {}
        group_by_id = {}
        nonEmptyGroups = set()
        for row in md.iterRows(outputMd):
            itemId = row.getValue(md.MDL_ITEM_ID)

            if itemId in weights_by_id:
                raise RuntimeError(
                    f"Duplicated itemId={itemId} in GMM output metadata."
                )

            weights_by_id[itemId] = (
                row.getValue("wRobust"),
                row.getValue("wRobustGmm"),
            )

            group = int(row.getValue(self._getGroupByColumn()))
            group_by_id[itemId] = group
            nonEmptyGroups.add(group)

        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(self.inputParticles.get())
        inputParticles = self.inputParticles.get()

        for particle in inputParticles:
            itemId = particle.getObjId()

            try:
                weight, weightGmm = weights_by_id[itemId]
            except KeyError as exc:
                raise RuntimeError(
                    f"No GMM weights found for particle " f"with itemId={itemId}."
                ) from exc

            outputParticle = particle.clone()

            outputParticle._xmippRobustWeight = Float(weight)
            outputParticle._xmippRobustWeightGmm = Float(weightGmm)
            outputParticle.setClassId(group_by_id[itemId])

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
