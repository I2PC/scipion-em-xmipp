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
from typing import Tuple, Union, Dict, List


from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
from pwem.emlib import MD_APPEND
from pwem.constants import ALIGN_2D
from pwem.objects import SetOfClasses2D, Particle


from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam
from pyworkflow.constants import BETA
from xmipp3.base import XmippProtocol


from xmipp3.convert import (
    writeSetOfClasses2D,
    readSetOfClasses2D,
    particleToRow,
    readSetOfParticles,
    rowToParticle,
)


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

    def _convertAlignmentConvention(
        self,
        metadataPath: Union[str, Path],
    ) -> None:
        """
        Adapt class alignment angles to xmipp_transform_geometry conventions.

        ``writeSetOfClasses2D`` stores the in-plane angle in ``angleRot`` for
        these inputs, whereas ``xmipp_transform_geometry`` expects it in
        ``anglePsi`` when applying a 2D transform.

        Parameters
        ----------
        metadataPath : str or pathlib.Path
            Metadata file whose angle columns will be modified in place.

        Notes
        -----
        This conversion is specific to the current Scipion/Xmipp metadata
        conventions and should be revisited if either side changes how 2D
        alignments are represented.
        """
        metadataPath = str(Path(metadataPath))

        self.runJob(
            "xmipp_metadata_utilities",
            f'-i {metadataPath} --operate drop_column "anglePsi"',
            numberOfMpi=1,
        )
        self.runJob(
            "xmipp_metadata_utilities",
            f'-i {metadataPath} --operate rename_column "angleRot anglePsi"',
            numberOfMpi=1,
        )

    def _preprocessClass(self, classId: int, className: str) -> Tuple[str, str]:
        """
        Prepare the input files required to estimate one class average.

        The particle block corresponding to the requested class is extracted from
        the global input metadata and written to a temporary metadata file. Its
        images are then materialized in a standalone stack, optionally corrected
        for the CTF, converted to the alignment convention expected by Xmipp, and
        geometrically transformed to produce an aligned stack.

        Parameters
        ----------
        classId : int
            Identifier of the class being processed. It is used to generate unique
            temporary file names.
        className : str
            Name of the metadata block containing the class particles in the input
            metadata file.

        Returns
        -------
        alignedClassMdPath : pathlib.Path
            Metadata file referencing the aligned particle stack used as input by
            the GMM average-estimation program.
        classParticlesMdPath : str
            Metadata file containing the original particle information for the
            selected class. The estimation program uses it as the base metadata to
            which the calculated weights are added.

        Notes
        -----
        All generated files are temporary protocol files. This method prepares the
        inputs but does not run the robust average estimator.
        """
        # Read metadata block corresponding to the requested class
        classMdBlockName = className + "@" + self._getInputMdName()
        classParticlesMd = md.MetaData(classMdBlockName)

        # Save the selected block to a temporary file
        classParticlesMdPath = self._getTmpPath(f"selected_particles_{classId}.xmd")
        classParticlesMd.write(classParticlesMdPath)

        # Preprocess the particles (CTF correction if requested, and alignment) and save to a temporary file
        preparedStackPath = self._getTmpPath(f"preprocessed_particles_{classId}.mrcs")
        preparedStackMdPath = Path(preparedStackPath).with_suffix(".xmd")
        self._prepareParticleStack(
            inputMetadataPath=classParticlesMdPath,
            outputParticlesPath=preparedStackPath,
            outputMetadataPath=preparedStackMdPath,
        )

        # Rename angle columns to deal with different alignment conventions
        self._convertAlignmentConvention(preparedStackMdPath)

        # Apply alignment to the stack
        alignedPath = self._getTmpPath(f"aligned_particles_{classId}.mrcs")
        alignedClassMdPath = str(Path(alignedPath).with_suffix(".xmd"))
        self._applyAlignment(
            inputMetadataPath=preparedStackMdPath,
            outputParticlesPath=alignedPath,
            outputMetadataPath=alignedClassMdPath,
        )

        return alignedClassMdPath, classParticlesMdPath

    def _runAverageEstimation(
        self,
        classId: int,
        inputMdPath: Union[str, Path],
        baseMdPath: Union[str, Path],
        device: str,
        env,
    ) -> Tuple[str, str, str]:
        """
        Run the GMM-based robust average estimator for one class.

        The external ``xmipp_gmm_average_estimation`` program reads the aligned
        particle stack referenced by ``inputMdPath``, iteratively estimates robust
        particle weights, and computes both a weighted and an unweighted class
        average. The calculated weights are appended to a copy of the metadata
        supplied through ``baseMdPath``.

        Parameters
        ----------
        classId : int
            Identifier of the class being processed. It is used only to generate
            unique output file names.
        inputMdPath : str or pathlib.Path
            Metadata referencing the aligned, and optionally CTF-corrected,
            particle stack used by the estimator.
        baseMdPath : str or pathlib.Path
            Original metadata for the selected particles. The output weight columns
            are added to a copy of this metadata.
        device : {"cpu", "cuda"}
            PyTorch device requested for the estimation program.
        env
            Environment returned by :meth:`getCondaEnv`, used to run the external
            estimator with its required Python dependencies.

        Returns
        -------
        outputStarPath : str
            Metadata file containing the original particle information and the
            calculated robust-weight columns.
        correctedAveragePath : str
            Path of the robust GMM-weighted average image.
        originalAveragePath : str
            Path of the corresponding unweighted average image.
        """
        # Prepare output paths for the star file with weights and averages
        outputStarPath = self._getTmpPath(f"class_particles_{classId}.star")
        correctedAveragePath = self._getTmpPath(f"corrected_avg_{classId}.mrc")
        originalAveragePath = self._getTmpPath(f"original_avg_{classId}.mrc")

        # Run the GMM average estimation script for the current class
        script_args = (
            f"--input-xmd {str(inputMdPath)} "
            f"--out-star {outputStarPath} "
            f"--base-xmd {str(baseMdPath)} "
            f"--out-corrected-avg {correctedAveragePath} "
            f"--out-original-avg {originalAveragePath} "
            f"--device {device} "
        )
        self.runJob("xmipp_gmm_average_estimation", script_args, env=env, numberOfMpi=1)

        return outputStarPath, correctedAveragePath, originalAveragePath

    def _addImageToMd(self, imagePath: Union[Path, str], metadata: md.MetaData) -> None:
        row = md.Row()
        row.setValue(md.MDL_IMAGE, f"1@{imagePath}")
        row.addToMd(metadata)

    def _saveToImageStack(
        self,
        metadata: md.MetaData,
        outputPath: Union[str, Path],
        writeMdName: str = "tmpStackMd.xmd",
    ) -> None:
        outputPath = str(Path(outputPath))

        writeMdPath = self._getTmpPath(writeMdName)
        metadata.write(writeMdPath)
        self.runJob(
            "xmipp_image_convert",
            f"-i {writeMdPath} -o {outputPath}",
            numberOfMpi=1,
        )

    def _getInputMdName(self):
        return self._getTmpPath("inputClasses.xmd")

    def _getInputParticlesPath(self):
        return self._getExtraPath("inputParticles.xmd")

    def _getPreprocessedParticlesPath(self):
        return self._getExtraPath("preprocessed.mrcs")

    def _getPreprocessedMetadataPath(self):
        return self._getExtraPath("preprocessed.xmd")

    def _getOutputClassesMdPath(self):
        return self._getExtraPath("outputClasses.xmd")

    def _getRawClassesMdPath(self):
        return self._getExtraPath("rawClasses.xmd")

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

    def averageEstimationStepOld(self):
        """
        Preprocess, estimate and collect the averages for all selected classes.

        For each selected class, this step:

        1. extracts and preprocesses its particles;
        2. applies their stored 2D alignment;
        3. runs the external GMM-based average estimator;
        4. stores the resulting particle weights in the output metadata;
        5. accumulates the robust and unweighted class representatives.

        After all classes have been processed, the individual average images are
        combined into two output stacks and the corresponding ``classes`` metadata
        blocks are written with updated representative-image references.
        """

        env = self.getCondaEnv()
        device = "cuda" if self.useGpu.get() else "cpu"

        outputClassesMdPath = self._getOutputClassesMdPath()
        rawClassesMdPath = self._getRawClassesMdPath()
        correctedAveragesStackPath = self._getExtraPath("correctedAverages.mrcs")
        originalAveragesStackPath = self._getExtraPath("originalAverages.mrcs")

        mdNewClassesBlock = md.MetaData()
        mdOriginalClassesBlock = md.MetaData()
        mdCorrectedAveragesToStack = md.MetaData()
        mdOriginalAveragesToStack = md.MetaData()

        # Copy old classes block to later update the image field with the new average
        oldClassesBlock = md.MetaData("classes@" + self._getInputMdName())
        oldClassRows = {}
        for row in md.iterRows(oldClassesBlock):
            oldClassRows[row.getValue(md.MDL_REF)] = row.clone()

        for index, (classId, className) in enumerate(self.classNames.items(), start=1):
            # Preprocessing: save stack, correct ctf and apply alignment
            alignedClassMetadataPath, baseMdPath = self._preprocessClass(
                classId=classId, className=className
            )

            # Run estimation method
            resultStarPath, correctedAveragePath, originalAveragePath = (
                self._runAverageEstimation(
                    classId=classId,
                    inputMdPath=alignedClassMetadataPath,
                    baseMdPath=baseMdPath,
                    device=device,
                    env=env,
                )
            )

            # Save the results for this class in the metadata files for the output
            class_metadata = md.MetaData(resultStarPath)
            class_metadata.write(className + "@" + outputClassesMdPath, MD_APPEND)
            class_metadata.write(className + "@" + rawClassesMdPath, MD_APPEND)

            # Add the corrected and original averages to the list of averages to be stacked
            self._addImageToMd(correctedAveragePath, mdCorrectedAveragesToStack)
            self._addImageToMd(originalAveragePath, mdOriginalAveragesToStack)

            # Update the image field of the old class row and add it to the new metadata block
            if classId in oldClassRows:
                row = oldClassRows[classId].clone()
                row.setValue(md.MDL_IMAGE, f"{index}@{correctedAveragesStackPath}")
                row.addToMd(mdNewClassesBlock)

                row2 = oldClassRows[classId].clone()
                row2.setValue(md.MDL_IMAGE, f"{index}@{originalAveragesStackPath}")
                row2.addToMd(mdOriginalClassesBlock)

        # Save corrected and original averages as image stacks
        self._saveToImageStack(
            mdCorrectedAveragesToStack,
            correctedAveragesStackPath,
            writeMdName="correctedAveragesToStack.xmd",
        )
        self._saveToImageStack(
            mdOriginalAveragesToStack,
            originalAveragesStackPath,
            writeMdName="originalAveragesToStack.xmd",
        )

        # Add the classes block with the updated image field to the new metadata file
        mdNewClassesBlock.write("classes@" + outputClassesMdPath, MD_APPEND)
        mdOriginalClassesBlock.write("classes@" + rawClassesMdPath, MD_APPEND)

    def createOutputStepOld(self):
        imagesPointer = self.inputClasses.get().getImagesPointer()

        # Create output classes based on the new metadata file with weights,
        # with the corrected representatives
        outputClassesMd = self._getOutputClassesMdPath()
        outputClasses = self._createSetOfClasses2D(imagesPointer, "corrected")
        readSetOfClasses2D(outputClasses, outputClassesMd)

        # Create raw output classes based on the new metadata file with weights,
        # with the original representatives
        rawClassesMd = self._getRawClassesMdPath()
        rawClasses = self._createSetOfClasses2D(imagesPointer, "raw")
        readSetOfClasses2D(rawClasses, rawClassesMd)

        # Join the particles from all classes into a single output set of particles
        finalParticles = self._createSetOfParticles()
        finalParticles.setSamplingRate(self.inputClasses.get().getSamplingRate())

        # Iterate over every class
        for cl in outputClasses.iterItems():
            classId = cl.getObjId()

            # Extract class block and iterate over its rows (particles)
            blockName = f"class{classId:06d}_images"
            classParticlesMd = md.MetaData(blockName + "@" + outputClassesMd)
            for particle, row in zip(cl.iterItems(), md.iterRows(classParticlesMd)):
                weight = row.getValue("wRobust")
                weightGmm = row.getValue("wRobustGmm")
                particle.setClassId(classId)

                # Add the robust weights as particle attributes
                particle._xmippRobustWeight = Float(weight)
                particle._xmippRobustWeightGmm = Float(weightGmm)

                finalParticles.append(particle)

        # Define protocol outputs: the new sets of classes and the joined set of particles
        self._defineOutputs(outputParticles=finalParticles)
        self._defineSourceRelation(self.inputClasses, finalParticles)

        self._defineOutputs(outputClasses_corrected=outputClasses)
        self._defineSourceRelation(self.inputClasses, outputClasses)

        self._defineOutputs(outputClasses_raw=rawClasses)
        self._defineSourceRelation(self.inputClasses, rawClasses)
