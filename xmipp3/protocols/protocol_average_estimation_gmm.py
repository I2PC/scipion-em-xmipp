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

from pyworkflow import VERSION_3_0
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
            default=1,
            # expertLevel=LEVEL_ADVANCED,
            label="Class ID",
            help="Class to select for average estimation. ",
            # "Zero or any negative value means the estimation "
            # "will be applied to all classes",
        )
        form.addParam(
            "correctCtf",
            BooleanParam,
            default=True,
            # expertLevel=LEVEL_ADVANCED,
            label="Correct CTF?",
            help="If you set to *Yes*, the CTF of the experimental particles will be corrected",
        )

        form.addParallelSection(threads=1, mpi=4)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputStep")
        # self._insertFunctionStep("preprocessStep")
        self._insertFunctionStep("averageEstimationStep")
        self._insertFunctionStep("createOutputStep")

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        """Selects the requested class and saves its metadata file."""
        self.inputMdName = self._getTmpPath("inputClasses.xmd")

        # Write the input data as a set of 2D classes
        writeSetOfClasses2D(
            self.inputClasses.get(), self.inputMdName, writeParticles=True
        )
        self.selectedParticlesPaths = []
        mdBlocks = md.getBlocksInMetaDataFile(self.inputMdName)

        # Identify class IDs to work with: all classes if classId input is <= 0,
        # otherwise the user-requested class.
        classId = self.classId.get()
        if classId > 0:
            if classId >= len(mdBlocks):
                raise ValueError(
                    "Requested class ID is higher than the number of classes"
                )
            self.class_names = {classId: mdBlocks[classId]}
        else:
            self.class_names = {i: mdBlocks[i] for i in range(1, len(mdBlocks))}

        self.sampling_rate = self.inputClasses.get().getSamplingRate()

    # --------------------------- UTILS functions -----------------------------
    def _preprocessParticles(
        self, particlesPath: Union[str, Path], output_path: Union[str, Path]
    ) -> str:
        """
        Reads the selected particles, CTF-corrects them if requested, and saves the
        preprocessed images to a temporary file.

        Returns
        -------
        str
            Path to the temporary file with the preprocessed particles
        """
        output_path = str(Path(output_path))

        if self.correctCtf.get():
            args = " -i  %s -o %s --sampling_rate %s " % (
                particlesPath,
                output_path,
                self.sampling_rate,
            )
            self.runJob(
                "xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get()
            )
        else:
            args = " -i  %s -o %s --save_metadata_stack " % (
                particlesPath,
                output_path,
            )
            self.runJob("xmipp_image_convert", args)

    def averageEstimationStep(self):
        """
        For each requested class, reads the preprocessed particles, runs
        the robust estimation method and writes:
        - a metadata file that contains an extra field with each image's score
        """
        env = self.getCondaEnv()

        new_metadata_path = self._getExtraPath("outputClasses.xmd")
        output_stack_path = self._getExtraPath("corrected_averages.mrcs")

        mdNewClassesBlock = md.MetaData()
        mdAveragesToStack = md.MetaData()

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

            # Preprocess the particles (CTF correction if requested) and save to a temporary file
            preprocessed_particles_path = self._getTmpPath(
                f"preprocessed_particles_{classId}.mrc"
            )
            self._preprocessParticles(particles_path, preprocessed_particles_path)
            class_metadata_path = (
                Path(preprocessed_particles_path).with_suffix(".xmd").resolve()
            )

            # Prepare output path for the star file with weights
            output_star_name = f"class_particles_{classId}.star"
            output_star_path = self._getTmpPath(output_star_name)
            tmp_corrected_avg_path = self._getTmpPath(f"corrected_avg_{classId}.mrc")
            # output_original_avg_path = self._getExtraPath(f"original_avg_{classId}.mrc")

            # Run the GMM average estimation script for the current class
            # "--out-corrected-avg %s --out-original-avg %s "
            # --out-weights %s --out-distances %s 
            script_args = "--input-xmd %s --out-star %s --base-xmd %s --out-corrected-avg %s --rotate-first" % (
                str(class_metadata_path),
                str(output_star_path),
                str(particles_path),
                str(tmp_corrected_avg_path),
                # str(self._getTmpPath(f"gmm_weights_{classId}.npy")),
                # str(self._getTmpPath(f"original_distances_{classId}.npy")),
                # str(output_original_avg_path),
            )
            self.runJob("xmipp_gmm_average_estimation", script_args, env=env)

            class_metadata = md.MetaData(output_star_path)
            class_metadata.write(className + "@" + new_metadata_path, MD_APPEND)
            
            # # Add the weights to the metadata of the current class
            # class_metadata = md.MetaData()
            # gmm_weights = np.load(self._getTmpPath(f"gmm_weights_{classId}.npy"))
            # original_weights = np.load(self._getTmpPath(f"original_distances_{classId}.npy"))
            # for row, weight, original_weight in zip(md.iterRows(particles_md), gmm_weights, original_weights):
            #     new_row = row.clone()
            #     new_row.setValue(md.MDL_ROBUST_GMM_WEIGHT, float(weight))
            #     new_row.setValue(md.MDL_ROBUST_ORIGINAL_WEIGHT, float(original_weight))
            #     new_row.addToMd(class_metadata)

            # Add the corrected average to the list of averages to be stacked
            row_avg = md.Row()
            row_avg.setValue(md.MDL_IMAGE, f"1@{tmp_corrected_avg_path}")
            row_avg.addToMd(mdAveragesToStack)

            # Update the image field of the old class row with the new average and add it to the new metadata block
            if classId in old_class_rows:
                row = old_class_rows[classId]
                row.setValue(md.MDL_IMAGE, f"{index}@{output_stack_path}")
                row.addToMd(mdNewClassesBlock)
        
        # Write a metadata file with the corrected averages and convert it to a stack
        tmp_averages_list_xmd = self._getTmpPath("averages_list.xmd")
        mdAveragesToStack.write(tmp_averages_list_xmd)
        self.runJob("xmipp_image_convert", f"-i {tmp_averages_list_xmd} -o {output_stack_path}")

        # Add the classes block with the updated image field to the new metadata file
        mdNewClassesBlock.write("classes@" + new_metadata_path, MD_APPEND)


    def createOutputStep(self):
        # Create output classes based on the new metadata file with weights
        outputClasses = self._createSetOfClasses2D(
            self.inputClasses.get().getImagesPointer()
        )
        readSetOfClasses2D(outputClasses, self._getExtraPath("outputClasses.xmd"))

        # Join the particles from all classes into a single output set of particles
        outputParticles = self._createSetOfParticles()
        particlesMd = md.utils.joinBlocks(
            self._getExtraPath("outputClasses.xmd"), "class0"
        )
        particlesMd.write(self._getExtraPath("outputParticles.xmd"))
        readSetOfParticles(self._getExtraPath("outputParticles.xmd"), outputParticles)
        outputParticles.setSamplingRate(self.inputClasses.get().getSamplingRate())

        # Define protocol outputs: the new set of classes and the joined set of particles
        self._defineOutputs(
            outputClasses=outputClasses,
            outputParticles=outputParticles,
        )
        self._defineSourceRelation(self.inputClasses, outputClasses)
        self._defineSourceRelation(self.inputClasses, outputParticles)
