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
from typing import Union, Dict, List


from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
from pwem.constants import ALIGN_2D
from pwem.objects import SetOfClasses2D


from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam, EnumParam
from pyworkflow.constants import BETA
from xmipp3.base import XmippProtocol


from xmipp3.convert import particleToRow

ESTIMATORS = {0: "gmm", 1: "irls", 2: "fourier_irls"}

ESTIMATOR_WEIGHT_COLUMNS = {
    "gmm": ["wRobust", "wRobustGmm"],
    "irls": ["wRobust"],
    "fourier_irls": ["wRobust"],
}

WEIGHT_COLUMN_TO_ATTRIBUTE = {
    "wRobust": "_xmippRobustWeight",
    "wRobustGmm": "_xmippRobustWeightGmm",
}


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
            "classId",
            IntParam,
            default=-1,
            # expertLevel=LEVEL_ADVANCED,
            label="Class ID",
            help="Class to select for average estimation. "
            "Zero or any negative value means the estimation "
            "will be applied to all classes",
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
            "estimatorType",
            EnumParam,
            default=0,
            choices=[ESTIMATORS[i] for i in range(len(ESTIMATORS))],
            help=(
                "Type of robust estimator to use to compute the new class averages. "
                "As a rule of thumb, the 'gmm' estimator should be more aggressive in "
                "rejecting possibly misaligned or corrupted particles. This means "
                "its performance can be better for more contaminated datasets, and "
                "slightly worse in very clean datasets."
            ),
            label="Estimator type",
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
        self.runJob("xmipp_transform_geometry", args, numberOfMpi=self.numberOfMpi.get())

    def _getInputParticlesPath(self):
        return self._getExtraPath("inputParticles.xmd")

    def _getPreprocessedParticlesPath(self):
        return self._getExtraPath("preprocessed.mrcs")

    def _getPreprocessedMetadataPath(self):
        return self._getExtraPath("preprocessed.xmd")

    def _getEstimatorType(self):
        return ESTIMATORS[self.estimatorType.get()]

    def _getEstimatorWeightColumns(self):
        return ESTIMATOR_WEIGHT_COLUMNS[self._getEstimatorType()]

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
        scriptArgs = (
            f"--input-xmd {self._getPreprocessedMetadataPath()} "
            f"--base-xmd {self._getInputParticlesPath()} "
            f"--out-star {outputStarPath} "
            f"--out-corrected-avg {correctedAveragePath} "
            f"--out-original-avg {originalAveragePath} "
            f"--device {device} "
            f"--estimator-type {self._getEstimatorType()} "
        )
        self.runJob("xmipp_gmm_average_estimation", scriptArgs, env=env, numberOfMpi=1)

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
