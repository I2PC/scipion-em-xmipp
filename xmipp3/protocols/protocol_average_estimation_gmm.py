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
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam
from pyworkflow.constants import BETA
from xmipp3.base import XmippProtocol


from xmipp3.convert import particleToRow, rowToParticle


class XmippProtAverageEstimationGmm(ProtClassify2D, XmippProtocol):
    """
    Improves class averages by estimating a robust average using a Gaussian Mixture
    Model (GMM) on the distances of the particles to the average.
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

        form.addParallelSection(threads=0, mpi=4)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputStep")
        self._insertFunctionStep("preprocessStep")
        self._insertFunctionStep("averageEstimationStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- UTILS functions -----------------------------
    def _buildMetadataStack(
        self,
        classes2D: SetOfClasses2D,
        classIds: List[int],
        outputMdPath: Union[str, Path],
    ):
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
        inputParticlesPath : str or pathlib.Path
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
        self.runJob("xmipp_transform_geometry", args, numberOfMpi=1)

    def _getInputParticlesPath(self):
        return self._getExtraPath("inputParticles.xmd")

    def _getPreprocessedParticlesPath(self):
        return self._getExtraPath("preprocessed.mrcs")

    def _getPreprocessedMetadataPath(self):
        return self._getExtraPath("preprocessed.xmd")

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        """
        Convert the input classes to Xmipp metadata and select classes to process.

        The complete input ``SetOfClasses2D`` is written to a temporary Xmipp
        metadata file, including the particles assigned to each class. The method
        then validates the optional user-supplied class identifier and builds the
        ordered mapping between selected class IDs and their metadata block names.

        The input sampling rate is also stored for use during CTF correction.
        """
        inputClasses = self.inputClasses.get()

        # Get all class ids in input classes object
        classIds = set()
        for cls in inputClasses:
            classIds.add(int(cls.getObjId()))

        # Identify class IDs to work with: all classes if classId input is <= 0,
        # otherwise the user-requested class.
        classId = self.classId.get()
        if classId > 0:
            if not classId in classIds:
                raise ValueError(
                    "Requested class ID is unavailable in the input classes object"
                )
            self.classIds = [classId]
            self.classNames = {classId: "class%06d_images" % classId}
        else:
            self.classIds = sorted(classIds)
            self.classNames = {i: "class%06d_images" % i for i in self.classIds}

        # Build metadata file with all the requested class particles
        self._buildMetadataStack(
            classes2D=inputClasses,
            classIds=self.classIds,
            outputMdPath=self._getInputParticlesPath(),
        )

    def preprocessStep(self):
        particlesMdPath = self._getInputParticlesPath()

        # Correct CTF (if requested) and extract particle stack
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
        env = self.getCondaEnv()
        device = "cuda" if self.useGpu.get() else "cpu"

        # Prepare output paths for the star file with weights and averages
        outputStarPath = self._getExtraPath("particles.star")
        correctedAveragePath = self._getExtraPath("corrected_avgs.mrcs")
        originalAveragePath = self._getExtraPath("original_avgs.mrcs")

        # Run the GMM average estimation script for all classes
        script_args = (
            f"--input-xmd {self._getPreprocessedMetadataPath()} "
            f"--base-xmd {self._getInputParticlesPath()} "
            f"--out-star {outputStarPath} "
            f"--out-corrected-avg {correctedAveragePath} "
            f"--out-original-avg {originalAveragePath} "
            f"--device {device} "
        )
        self.runJob("xmipp_gmm_average_estimation", script_args, env=env, numberOfMpi=1)

    def createOutputStep(self):
        outputParticlesMd = md.MetaData(self._getExtraPath("particles.star"))
        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(self.inputClasses.get().getImages())

        for row in md.iterRows(outputParticlesMd):
            particle = rowToParticle(row, alignType=ALIGN_2D)

            classId = row.getValue(md.MDL_REF)
            weight = row.getValue("wRobust")
            weightGmm = row.getValue("wRobustGmm")

            # Add the robust weights and the class ID as particle attributes
            particle.setClassId(classId)
            particle._xmippRobustWeight = Float(weight)
            particle._xmippRobustWeightGmm = Float(weightGmm)

            outputParticles.append(particle)

        # The estimation script writes averages following sorted class IDs.
        classIndex = {
            classId: index for index, classId in enumerate(self.classIds, start=1)
        }

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
