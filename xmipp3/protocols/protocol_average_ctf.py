# ******************************************************************************
# *
# * Authors: Erney Ramirez Aportela (eramirez@cnb.csic.es)
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
import enum
import sys
import emtable
import os
from datetime import datetime
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
from pwem.emlib import MD_APPEND
from pwem.constants import ALIGN_2D
from pwem.objects import SetOfParticles, SetOfClasses2D

from pyworkflow import VERSION_3_0
from pyworkflow.object import Float
from pyworkflow.protocol.params import PointerParam, IntParam, BooleanParam
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.constants import BETA
from xmipp3.base import XmippProtocol


from xmipp3.convert import writeSetOfClasses2D, readSetOfParticles, readSetOfClasses2D


class XmippProtAverageCtf(ProtClassify2D, XmippProtocol):
    """
    Apply CTF to class average.
    """

    _label = "class_average_ctf"
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
            label="Class ID",
            help="Class to select for average estimation. "
            "Zero or any negative value means the estimation "
            "will be applied to all classes",
        )
        form.addParam(
            "correctCtf",
            BooleanParam,
            default=True,
            label="Correct CTF?",
            help="If you set to *Yes*, the CTF of the experimental particles will be corrected",
        )
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
        Reads the selected particles, CTF-corrects them if requested, and saves the
        preprocessed images to a temporary file.
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
        Prepare the inputs required by the GMM average estimator.

        Parameters
        ----------
        classId : int
            ID of the class of images to be processed. Used only for identifying
            the generated files.
        className : str
            Name of the metadata block where the class information is contained
            within the input metadata file.

        Returns
        -------
        alignedClassMdPath : str
            Metadata referencing the CTF-corrected (if CTF correction was requested)
            and aligned particle stack.
        classParticlesMdPath : str
            Original class metadata used to preserve particle information.
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
        # alignedPath = self._getTmpPath(f"aligned_particles_{classId}.mrcs")
        # alignedClassMdPath = Path(alignedPath).with_suffix(".xmd")
        # self._applyAlignment(
        #     inputMetadataPath=preparedStackMdPath,
        #     outputParticlesPath=alignedPath,
        #     outputMetadataPath=alignedClassMdPath,
        # )
        alignedClassMdPath = preparedStackMdPath
        return alignedClassMdPath, classParticlesMdPath

    def _runAverageEstimation(
        self,
        classId: int,
        inputMdPath: Union[str, Path],
        sampling: float,
        env,
    ):

        # Prepare output paths for the star file with weights and averages
        # outputStarPath = self._getTmpPath(f"class_particles_{classId}.star")
        # correctedAveragePath = self._getTmpPath(f"corrected_avg_{classId}.mrc")
        originalAveragePath = self._getTmpPath(f"original_avg_{classId}.mrc")

        # Run the GMM average estimation script for the current class
        script_args = (
            f" {str(inputMdPath)} "
            f"--out {originalAveragePath} "
            f" --sampling_rate {sampling} "
        )
        self.runJob("xmipp_average_ctf", script_args, env=env, numberOfMpi=1)

        return originalAveragePath

    def _addImageToMd(
        self, imagePath: Union[Path, str], mdPath: Union[Path, str]
    ) -> None:
        row = md.Row()
        row.setValue(md.MDL_IMAGE, f"1@{imagePath}")
        row.addToMd(mdPath)

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
        return self._getTmpPath("inputClasses.xmd")

    def _getOutputClassesMdPath(self):
        return self._getExtraPath("outputClasses.xmd")

    def _getRawClassesMdPath(self):
        return self._getExtraPath("rawClasses.xmd")

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        """Selects the requested class and saves its metadata file."""
        inputMdName = self._getInputMdName()

        # Write the input data as a set of 2D classes
        writeSetOfClasses2D(self.inputClasses.get(), inputMdName, writeParticles=True)

        # Get all class ids in input classes file
        classesBlock = md.MetaData("classes@" + inputMdName)
        class_ids = set()
        for row in md.iterRows(classesBlock):
            class_ids.add(row.getValue(md.MDL_REF))

        # Identify class IDs to work with: all classes if classId input is <= 0,
        # otherwise the user-requested class.
        classId = self.classId.get()
        if classId > 0:
            if not classId in class_ids:
                raise ValueError(
                    "Requested class ID is unavailable in the input classes object"
                )
            self.class_names = {classId: "class%06d_images" % classId}
        else:
            self.class_names = {i: "class%06d_images" % i for i in sorted(class_ids)}

        self.sampling_rate = self.inputClasses.get().getSamplingRate()

    def averageEstimationStep(self):
        """
        For each requested class, reads the preprocessed particles, runs
        the robust estimation method and writes:
        - a metadata file that contains an extra field with each image's score
        - a .mrcs file with the corrected averages calculated by the estimation method
        - a .mrcs file with the original, uncorrected averages
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

        for index, (classId, className) in enumerate(self.class_names.items(), start=1):
            # Preprocessing: save stack, correct ctf and apply alignment
            alignedClassMetadataPath, baseMdPath = self._preprocessClass(
                classId=classId, className=className
            )

            # Run estimation method
            # resultStarPath, correctedAveragePath, originalAveragePath = (
            originalAveragePath =  self._runAverageEstimation(
                    classId=classId,
                    inputMdPath=alignedClassMetadataPath,
                    sampling=self.inputClasses.get().getSamplingRate(),
                    env=env,
                )
            #)

            # Save the results for this class in the metadata files for the output
            # class_metadata = md.MetaData(resultStarPath)
            # class_metadata.write(className + "@" + outputClassesMdPath, MD_APPEND)
            # class_metadata.write(className + "@" + rawClassesMdPath, MD_APPEND)

            # Add the corrected and original averages to the list of averages to be stacked
            # self._addImageToMd(correctedAveragePath, mdCorrectedAveragesToStack)
            self._addImageToMd(originalAveragePath, mdOriginalAveragesToStack)

            # Update the image field of the old class row and add it to the new metadata block
            if classId in oldClassRows:
                # row = oldClassRows[classId].clone()
                # row.setValue(md.MDL_IMAGE, f"{index}@{correctedAveragesStackPath}")
                # row.addToMd(mdNewClassesBlock)

                row2 = oldClassRows[classId].clone()
                row2.setValue(md.MDL_IMAGE, f"{index}@{originalAveragesStackPath}")
                row2.addToMd(mdOriginalClassesBlock)

        # Save corrected and original averages as image stacks
        # self._saveToImageStack(mdCorrectedAveragesToStack, correctedAveragesStackPath)
        self._saveToImageStack(mdOriginalAveragesToStack, originalAveragesStackPath)

        # Add the classes block with the updated image field to the new metadata file
        # mdNewClassesBlock.write("classes@" + outputClassesMdPath, MD_APPEND)
        mdOriginalClassesBlock.write("classes@" + rawClassesMdPath, MD_APPEND)

    def createOutputStep(self):
        imagesPointer = self.inputClasses.get().getImagesPointer()

        # Create output classes based on the new metadata file with weights,
        # with the corrected representatives
        # outputClassesMd = self._getOutputClassesMdPath()
        # outputClasses = self._createSetOfClasses2D(imagesPointer, "corrected")
        # readSetOfClasses2D(outputClasses, outputClassesMd)

        # Create raw output classes based on the new metadata file with weights,
        # with the original representatives
        rawClassesMd = self._getRawClassesMdPath()
        rawClasses = self._createSetOfClasses2D(imagesPointer, "raw")
        readSetOfClasses2D(rawClasses, rawClassesMd)

        # Join the particles from all classes into a single output set of particles
        # finalParticles = self._createSetOfParticles()
        # finalParticles.setSamplingRate(self.inputClasses.get().getSamplingRate())
        #
        # # Iterate over every class
        # for cl in outputClasses.iterItems():
        #     classId = cl.getObjId()
        #
        #     # Extract class block and iterate over its rows (particles)
        #     blockName = f"class{classId:06d}_images"
        #     classParticlesMd = md.MetaData(blockName + "@" + outputClassesMd)
        #     for particle, row in zip(cl.iterItems(), md.iterRows(classParticlesMd)):
        #         weight = row.getValue("wRobust")
        #         weightGmm = row.getValue("wRobustGmm")
        #         particle.setClassId(classId)
        #
        #         # Add the robust weights as particle attributes
        #         particle._xmippRobustWeight = Float(weight)
        #         particle._xmippRobustWeightGmm = Float(weightGmm)
        #
        #         finalParticles.append(particle)

        # Define protocol outputs: the new sets of classes and the joined set of particles
        # self._defineOutputs(outputParticles=finalParticles)
        # self._defineSourceRelation(self.inputClasses, finalParticles)
        #
        # self._defineOutputs(outputClasses_corrected=outputClasses)
        # self._defineSourceRelation(self.inputClasses, outputClasses)

        self._defineOutputs(outputClasses_raw=rawClasses)
        self._defineSourceRelation(self.inputClasses, rawClasses)
    
    