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

from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md

from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam
from pyworkflow.constants import BETA

from xmipp3.base import XmippProtocol
from xmipp3.convert import writeSetOfParticles, rowToParticle


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
        return self._getExtraPath("preparedParticles.mrcs")

    def _getParticleMdPath(self):
        return self._getExtraPath("preparedParticles.xmd")

    def _getCtfCorrectedStackPath(self):
        return self._getExtraPath("ctfCorrectedParticles.mrcs")

    def _getCtfCorrectedMdPath(self):
        return self._getExtraPath("ctfCorrectedParticles.xmd")

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
                numberOfMpi=1,
            )
            geometry_input = self._getCtfCorrectedMdPath()

        geometry_args = (
            f"-i '{geometry_input}' "
            f"-o '{self._getParticleStackPath()}' "
            f"--save_metadata_stack '{self._getParticleMdPath()}' "
            f"--keep_input_columns "
            f"--apply_transform "
        )

        self.runJob("xmipp_transform_geometry", geometry_args, numberOfMpi=1)

    def coneAveragingStep(self):
        env = self.getCondaEnv()
        device = "cuda" if self.useGpu.get() else "cpu"

        script_args = (
            f"--input-xmd '{self._getParticleMdPath()}' "
            f"--base-xmd '{self._getInputMdPath()}' "
            f"--out-star '{self._getAveragingOutputStarPath()}' "
            f"--device {device} "
            f"--group-by-column '{self._getGroupByColumn()}' "
        )
        self.runJob("xmipp_gmm_average_estimation", script_args, env=env, numberOfMpi=1)

    def createOutputStep(self):
        outputMd = md.MetaData(self._getAveragingOutputStarPath())

        weights_by_id = {}
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

            outputParticles.append(outputParticle)

        self._defineOutputs(outputParticles=outputParticles)
        self._defineSourceRelation(self.inputParticles, outputParticles)
