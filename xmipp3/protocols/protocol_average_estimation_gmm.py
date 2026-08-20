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
import json
import time
from pathlib import Path
from typing import Tuple, Union, Dict, List
from dataclasses import dataclass, asdict


from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
from pwem.emlib import MD_APPEND


from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam
from pyworkflow.constants import BETA
from xmipp3.base import XmippProtocol


from xmipp3.convert import writeSetOfClasses2D, readSetOfClasses2D, writeSetOfParticles


@dataclass
class ClassConfig:
    index: int
    class_id: int
    class_name: str
    input_xmd: str
    base_xmd: str
    out_star: str
    out_corrected_avg: str
    out_original_avg: str


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
        self._insertFunctionStep("averageEstimationStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- UTILS functions -----------------------------
    def _prepareParticleStack(
        self,
        inputParticlesPath: Union[str, Path],
        outputParticlesPath: Union[str, Path],
        outputMetadataPath: Union[str, Path],
    ):
        """
        Materialize a particle stack for subsequent class preprocessing.

        The input metadata identifies the particles belonging to one class. If CTF
        correction is enabled, the images are corrected with
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
        inputParticlesPath = str(Path(inputParticlesPath))
        outputParticlesPath = str(Path(outputParticlesPath))
        outputMetadataPath = str(Path(outputMetadataPath))

        if self.correctCtf.get():
            args = (
                f"-i {inputParticlesPath} "
                f"-o {outputParticlesPath} "
                f"--save_metadata_stack {outputMetadataPath} "
                f"--sampling_rate {self.sampling_rate}"
            )
            self.runJob(
                "xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get()
            )
        else:
            args = (
                f"-i  {inputParticlesPath} "
                f"-o {outputParticlesPath} "
                f"--save_metadata_stack {outputMetadataPath}"
            )
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
        alignedClassMdPath : str
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
            inputParticlesPath=classParticlesMdPath,
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
        batchConfigPath: Union[str, Path],
        timingsPath: Union[str, Path],
    ) -> None:
        """
        Run the GMM-based robust average estimator for all classes in a single
        external Python process.
        """
        env = self.getCondaEnv()
        device = "cuda" if self.useGpu.get() else "cpu"

        scriptArgs = (
            f'--batch-config "{batchConfigPath}" '
            f"--device {device} "
            f'--timings-file "{timingsPath}"'
        )
        self.runJob(
            "xmipp_gmm_average_estimation",
            scriptArgs,
            env=env,
            numberOfMpi=1,
        )

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
        return self._getExtraPath("inputClasses.xmd")

    def _getOutputClassesMdPath(self):
        return self._getExtraPath("outputClasses.xmd")

    def _getRawClassesMdPath(self):
        return self._getExtraPath("rawClasses.xmd")

    def _getCorrectedAveragesStackPath(self):
        return self._getExtraPath("correctedAverages.mrcs")

    def _getOriginalAveragesStackPath(self):
        return self._getExtraPath("originalAverages.mrcs")

    def _getAverageEstimationPaths(
        self,
        classId: int,
    ) -> Tuple[str, str, str]:
        return (
            self._getTmpPath(f"class_particles_{classId}.star"),
            self._getTmpPath(f"corrected_avg_{classId}.mrc"),
            self._getTmpPath(f"original_avg_{classId}.mrc"),
        )

    def _writeAverageEstimationBatchConfig(self, classConfigDicts: List[Dict]) -> str:
        batchConfigPath = self._getTmpPath("gmm_average_batch.json")

        with open(batchConfigPath, "w", encoding="utf-8") as file:
            json.dump(
                {"format_version": 1, "classes": classConfigDicts},
                file,
                indent=2,
            )

        return batchConfigPath

    def _prepareAverageEstimationBatch(self) -> Tuple[str, List[ClassConfig]]:
        """Preprocess all selected classes and prepare the batch description."""
        classConfigs: List[ClassConfig] = []

        for index, (classId, className) in enumerate(self.classNames.items(), start=1):
            alignedClassMetadataPath, baseMdPath = self._preprocessClass(
                classId=classId, className=className
            )
            resultStarPath, correctedAveragePath, originalAveragePath = (
                self._getAverageEstimationPaths(classId)
            )

            config = ClassConfig(
                index=index,
                class_id=classId,
                class_name=className,
                input_xmd=str(alignedClassMetadataPath),
                base_xmd=str(baseMdPath),
                out_star=str(resultStarPath),
                out_corrected_avg=str(correctedAveragePath),
                out_original_avg=str(originalAveragePath),
            )
            classConfigs.append(config)

        batchConfigDicts = [asdict(c) for c in classConfigs]
        batchConfigPath = self._writeAverageEstimationBatchConfig(batchConfigDicts)

        return batchConfigPath, classConfigs

    def _collectAverageEstimationResults(self, classConfigs: List[ClassConfig]) -> None:
        """Write class metadata and combine per-class averages into output stacks."""
        outputClassesMdPath = self._getOutputClassesMdPath()
        rawClassesMdPath = self._getRawClassesMdPath()
        correctedAveragesStackPath = self._getCorrectedAveragesStackPath()
        originalAveragesStackPath = self._getOriginalAveragesStackPath()

        mdNewClassesBlock = md.MetaData()
        mdOriginalClassesBlock = md.MetaData()
        mdCorrectedAveragesToStack = md.MetaData()
        mdOriginalAveragesToStack = md.MetaData()

        oldClassesBlock = md.MetaData("classes@" + self._getInputMdName())
        oldClassRows = {
            row.getValue(md.MDL_REF): row.clone()
            for row in md.iterRows(oldClassesBlock)
        }

        for result in classConfigs:
            classId = result.class_id
            className = result.class_name
            index = result.index

            classMetadata = md.MetaData(result.out_star)
            classMetadata.write(className + "@" + outputClassesMdPath, MD_APPEND)
            classMetadata.write(className + "@" + rawClassesMdPath, MD_APPEND)

            self._addImageToMd(result.out_corrected_avg, mdCorrectedAveragesToStack)
            self._addImageToMd(result.out_original_avg, mdOriginalAveragesToStack)

            if classId in oldClassRows:
                correctedRow = oldClassRows[classId].clone()
                correctedRow.setValue(
                    md.MDL_IMAGE, f"{index}@{correctedAveragesStackPath}"
                )
                correctedRow.addToMd(mdNewClassesBlock)

                originalRow = oldClassRows[classId].clone()
                originalRow.setValue(
                    md.MDL_IMAGE, f"{index}@{originalAveragesStackPath}"
                )
                originalRow.addToMd(mdOriginalClassesBlock)

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

        mdNewClassesBlock.write("classes@" + outputClassesMdPath, MD_APPEND)
        mdOriginalClassesBlock.write("classes@" + rawClassesMdPath, MD_APPEND)

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
        stepStart = time.perf_counter()

        inputMdName = self._getInputMdName()

        # Write the input data as a set of 2D classes
        writeSetOfClasses2D(self.inputClasses.get(), inputMdName, writeParticles=True)

        # Get all class ids in input classes file
        classesBlock = md.MetaData("classes@" + inputMdName)
        classIdsSet = set()
        for row in md.iterRows(classesBlock):
            classIdsSet.add(row.getValue(md.MDL_REF))

        # Identify class IDs to work with: all classes if classId input is <= 0,
        # otherwise the user-requested class.
        classId = self.classId.get()
        if classId > 0:
            if not classId in classIdsSet:
                raise ValueError(
                    "Requested class ID is unavailable in the input classes object"
                )
            self.classNames = {classId: "class%06d_images" % classId}
        else:
            self.classNames = {i: "class%06d_images" % i for i in sorted(classIdsSet)}

        self.sampling_rate = self.inputClasses.get().getSamplingRate()

        timings = {
            "preprocessing": 0.0,
            "estimation": 0.0,
            "saving_outputs": 0.0,
            "other": time.perf_counter() - stepStart,
        }
        self._saveTimings(timings)

    def averageEstimationStep(self):
        """Preprocess classes, run one batch estimation, and save its results."""
        timingsPath = Path(self._getEstimationTimingsPath())
        if timingsPath.exists():
            timingsPath.unlink()

        stepStart = time.perf_counter()
        timings = self._loadTimings()

        start = time.perf_counter()
        batchConfigPath, classResults = self._prepareAverageEstimationBatch()
        preprocessingTime = time.perf_counter() - start

        start = time.perf_counter()
        self._runAverageEstimation(
            batchConfigPath=batchConfigPath, timingsPath=timingsPath
        )
        estimationTime = time.perf_counter() - start

        start = time.perf_counter()
        self._collectAverageEstimationResults(classResults)
        savingResultsTime = time.perf_counter() - start

        measuredTime = preprocessingTime + estimationTime + savingResultsTime
        timings["preprocessing"] += preprocessingTime
        timings["estimation"] += estimationTime
        timings["saving_outputs"] += savingResultsTime
        timings["other"] += max(0.0, time.perf_counter() - stepStart - measuredTime)

        self._saveTimings(timings)
        self._printEstimationTimings()

    def createOutputStep(self):
        stepStart = time.perf_counter()

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

        timings = self._loadTimings()
        timings["saving_outputs"] += time.perf_counter() - stepStart
        self._saveTimings(timings)

        self._printTimings(timings)

    def _getTimingsPath(self) -> str:
        """Return the path used to persist protocol timing information."""
        return self._getExtraPath("timings.json")

    def _loadTimings(self) -> Dict[str, float]:
        """Load timing information saved by previous protocol steps."""
        timingsPath = Path(self._getTimingsPath())

        if not timingsPath.exists():
            return {
                "preprocessing": 0.0,
                "estimation": 0.0,
                "saving_outputs": 0.0,
                "other": 0.0,
            }

        with timingsPath.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _saveTimings(self, timings: Dict[str, float]) -> None:
        """Persist timing information for subsequent protocol steps."""
        with open(self._getTimingsPath(), "w", encoding="utf-8") as file:
            json.dump(timings, file, indent=2)

    def _printTimings(self, timings: Dict[str, float]) -> None:
        """Print a summary of the measured protocol execution times."""
        totalTime = sum(timings.values())

        print("\n" + "=" * 60)
        print("GMM average estimation timing")
        print("=" * 60)

        for name, elapsed in timings.items():
            percentage = 100.0 * elapsed / totalTime if totalTime > 0.0 else 0.0
            label = name.replace("_", " ").capitalize()
            print(f"{label:<20}: {elapsed:10.3f} s ({percentage:5.1f} %)")

        print("-" * 60)
        print(f"{'Total':<20}: {totalTime:10.3f} s")
        print("=" * 60 + "\n")

    def _getEstimationTimingsPath(self) -> str:
        """Return the path containing accumulated script timings."""
        return self._getExtraPath("estimationTimings.json")

    def _printEstimationTimings(self) -> None:
        """Print accumulated timings from all estimation script calls."""
        timingsPath = Path(self._getEstimationTimingsPath())

        if not timingsPath.exists():
            print("No estimation timing information was generated.")
            return

        with timingsPath.open("r", encoding="utf-8") as file:
            timings = json.load(file)

        totalTime = timings["total"]

        labels = {
            "imports_and_startup": "Imports and script startup",
            "argument_parsing": "Argument parsing",
            "device_setup": "Device setup",
            "read_images": "Read images",
            "distance_setup": "Distance setup",
            "masking": "Mask creation/application",
            "estimator_setup": "Estimator setup",
            "estimator_fit": "Estimator fit",
            "result_conversion": "Result conversion",
            "write_corrected_average": "Write corrected averages",
            "write_original_average": "Write original averages",
            "write_metadata": "Write metadata",
            "write_optional_arrays": "Write optional arrays",
            "other": "Other",
        }

        print("\n" + "=" * 72)
        print("Accumulated GMM estimation script timings")
        print("=" * 72)
        print(f"{'Script calls':<38}: {timings['n_calls']:10d}")
        print(f"{'Images processed':<38}: {timings['n_images']:10d}")
        print("-" * 72)

        for key, label in labels.items():
            elapsed = timings[key]
            percentage = 100.0 * elapsed / totalTime if totalTime > 0.0 else 0.0
            print(f"{label:<38}: " f"{elapsed:10.3f} s " f"({percentage:5.1f} %)")

        print("-" * 72)
        print(f"{'Total':<38}: {totalTime:10.3f} s")
        print("=" * 72 + "\n")
