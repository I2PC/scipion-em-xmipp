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
import enum
import sys
import emtable
import os
from datetime import datetime
import time
from pathlib import Path
from typing import Union

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
        # self._insertFunctionStep("preprocessStep")
        self._insertFunctionStep("averageEstimationStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- UTILS functions -----------------------------
    def _preprocessParticles(
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

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        """Selects the requested class and saves its metadata file."""
        self.inputMdName = self._getExtraPath("inputClasses.xmd")

        # Write the input data as a set of 2D classes
        writeSetOfClasses2D(
            self.inputClasses.get(), self.inputMdName, writeParticles=True
        )
        self.selectedParticlesPaths = []

        # Get all class ids in input classes file
        classesBlock = md.MetaData("classes@" + self.inputMdName)
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

        new_metadata_path = self._getExtraPath("outputClasses.xmd")
        original_metadata_path = self._getExtraPath("rawClasses.xmd")
        output_corrected_stack_path = self._getExtraPath("corrected_averages.mrcs")
        output_original_stack_path = self._getExtraPath("original_averages.mrcs")

        mdNewClassesBlock = md.MetaData()
        mdOriginalClassesBlock = md.MetaData()
        mdCorrectedAveragesToStack = md.MetaData()
        mdOriginalAveragesToStack = md.MetaData()

        # Copy old classes block to later update the image field with the new average
        oldClassesBlock = md.MetaData("classes@" + self.inputMdName)
        old_class_rows = {}
        for row in md.iterRows(oldClassesBlock):
            old_class_rows[row.getValue(md.MDL_REF)] = row.clone()

        for index, (classId, className) in enumerate(self.class_names.items(), start=1):
            # Read metadata block corresponding to the requested class
            particles_metadata_name = className + "@" + self.inputMdName
            particles_md = md.MetaData(particles_metadata_name)

            # Save the selected block to a temporary file
            particles_name = f"selected_particles_{classId}.xmd"
            particles_path = self._getTmpPath(particles_name)
            particles_md.write(particles_path)

            # Preprocess the particles (CTF correction if requested, and alignment) and save to a temporary file
            preprocessed_particles_path = self._getTmpPath(
                f"preprocessed_particles_{classId}.mrcs"
            )
            preprocessedMetadataPath = Path(preprocessed_particles_path).with_suffix(
                ".xmd"
            )
            self._preprocessParticles(
                inputParticlesPath=particles_path,
                outputParticlesPath=preprocessed_particles_path,
                outputMetadataPath=preprocessedMetadataPath,
            )

            # Rename angle columns to deal with different alignment conventions
            args = f'-i {preprocessedMetadataPath} --operate drop_column "anglePsi"'
            self.runJob("xmipp_metadata_utilities", args, numberOfMpi=1)
            args = f'-i {preprocessedMetadataPath} --operate rename_column "angleRot anglePsi"'
            self.runJob("xmipp_metadata_utilities", args, numberOfMpi=1)

            # Apply alignment to images
            alignedPath = self._getTmpPath(f"aligned_particles_{classId}.mrcs")
            alignedClassMetadataPath = Path(alignedPath).with_suffix(".xmd")
            self._applyAlignment(
                inputMetadataPath=preprocessedMetadataPath,
                outputParticlesPath=alignedPath,
                outputMetadataPath=alignedClassMetadataPath,
            )

            # Prepare output path for the star file with weights
            output_star_name = f"class_particles_{classId}.star"
            output_star_path = self._getTmpPath(output_star_name)
            tmp_corrected_avg_path = self._getTmpPath(f"corrected_avg_{classId}.mrc")
            tmp_original_avg_path = self._getTmpPath(f"original_avg_{classId}.mrc")

            # Run the GMM average estimation script for the current class
            device = "cuda" if self.useGpu.get() else "cpu"
            script_args = (
                f"--input-xmd {alignedClassMetadataPath} "
                f"--out-star {output_star_path} "
                f"--base-xmd {particles_path} "
                f"--out-corrected-avg {tmp_corrected_avg_path} "
                f"--out-original-avg {tmp_original_avg_path} "
                f"--device {device}"
            )
            self.runJob(
                "xmipp_gmm_average_estimation", script_args, env=env, numberOfMpi=1
            )

            class_metadata = md.MetaData(output_star_path)
            class_metadata.write(className + "@" + new_metadata_path, MD_APPEND)
            class_metadata.write(className + "@" + original_metadata_path, MD_APPEND)

            # Add the corrected and original averages to the list of averages to be stacked
            row_corrected_avg = md.Row()
            row_corrected_avg.setValue(md.MDL_IMAGE, f"1@{tmp_corrected_avg_path}")
            row_corrected_avg.addToMd(mdCorrectedAveragesToStack)

            row_original_avg = md.Row()
            row_original_avg.setValue(md.MDL_IMAGE, f"1@{tmp_original_avg_path}")
            row_original_avg.addToMd(mdOriginalAveragesToStack)

            # Update the image field of the old class row with the new average and add it to the new metadata block
            if classId in old_class_rows:
                row = old_class_rows[classId].clone()
                row.setValue(md.MDL_IMAGE, f"{index}@{output_corrected_stack_path}")
                row.addToMd(mdNewClassesBlock)

                row2 = old_class_rows[classId].clone()
                row2.setValue(md.MDL_IMAGE, f"{index}@{output_original_stack_path}")
                row2.addToMd(mdOriginalClassesBlock)

        # Write a metadata file with the corrected averages and convert it to a stack
        tmp_corrected_averages_list_xmd = self._getTmpPath(
            "corrected_averages_list.xmd"
        )
        mdCorrectedAveragesToStack.write(tmp_corrected_averages_list_xmd)
        self.runJob(
            "xmipp_image_convert",
            f"-i {tmp_corrected_averages_list_xmd} -o {output_corrected_stack_path}",
            numberOfMpi=1,
        )

        # Write a metadata file with the original averages
        tmp_original_averages_list_xmd = self._getTmpPath("original_averages_list.xmd")
        mdOriginalAveragesToStack.write(tmp_original_averages_list_xmd)
        self.runJob(
            "xmipp_image_convert",
            f"-i {tmp_original_averages_list_xmd} -o {output_original_stack_path}",
            numberOfMpi=1,
        )

        # Add the classes block with the updated image field to the new metadata file
        mdNewClassesBlock.write("classes@" + new_metadata_path, MD_APPEND)
        mdOriginalClassesBlock.write("classes@" + original_metadata_path, MD_APPEND)

    def createOutputStep(self):
        imagesPointer = self.inputClasses.get().getImagesPointer()

        # Create output classes based on the new metadata file with weights,
        # with the corrected representatives
        outputClasses = self._createSetOfClasses2D(imagesPointer, "corrected")
        readSetOfClasses2D(outputClasses, self._getExtraPath("outputClasses.xmd"))

        # Create raw output classes based on the new metadata file with weights,
        # with the original representatives
        rawClasses = self._createSetOfClasses2D(imagesPointer, "raw")
        readSetOfClasses2D(rawClasses, self._getExtraPath("rawClasses.xmd"))

        # Join the particles from all classes into a single output set of particles
        outputParticles = self._createSetOfParticles()
        particlesMd = md.utils.joinBlocks(
            self._getExtraPath("outputClasses.xmd"), "class0"
        )
        particlesMd.write(self._getExtraPath("outputParticles.xmd"))
        readSetOfParticles(self._getExtraPath("outputParticles.xmd"), outputParticles)
        outputParticles.setSamplingRate(self.inputClasses.get().getSamplingRate())

        finalParticles = self._createSetOfParticles()
        finalParticles.copyInfo(outputParticles)

        for cl in outputClasses.iterItems():
            classId = cl.getObjId()
            blockName = f"class{classId:06d}_images"
            classParticlesMd = md.MetaData(
                blockName + "@" + self._getExtraPath("outputClasses.xmd")
            )
            for particle, row in zip(cl.iterItems(), md.iterRows(classParticlesMd)):
                weight = row.getValue("wRobust")
                weightGmm = row.getValue("wRobustGmm")
                particle.setClassId(classId)
                particle._xmippRobustWeight = Float(weight)
                particle._xmippRobustWeightGmm = Float(weightGmm)
                finalParticles.append(particle)

        # Define protocol outputs: the new sets of classes and the joined set of particles
        self._defineOutputs(outputParticles=outputParticles)
        self._defineSourceRelation(self.inputClasses, outputParticles)

        self._defineOutputs(outputClasses_corrected=outputClasses)
        self._defineSourceRelation(self.inputClasses, outputClasses)

        self._defineOutputs(outputClasses_raw=rawClasses)
        self._defineSourceRelation(self.inputClasses, rawClasses)
