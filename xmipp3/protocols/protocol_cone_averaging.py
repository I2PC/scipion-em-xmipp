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
from __future__ import annotations

from enum import IntEnum
from math import isfinite
from shlex import join

import pwem.emlib.metadata as md
from pwem.constants import ALIGN_2D
from pwem.objects import Class2D, SetOfClasses2D
from pwem.protocols import ProtClassify2D
from pyworkflow import VERSION_3_0
from pyworkflow.constants import BETA
from pyworkflow.protocol import LEVEL_ADVANCED
from pyworkflow.protocol.params import (
    BooleanParam,
    EnumParam,
    FloatParam,
    IntParam,
    PointerParam,
    StringParam,
)

from xmipp3.base import XmippProtocol
from xmipp3.convert import rowToAlignment, writeSetOfParticles

_CONE_GEOMETRY_LABELS = (md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y, md.MDL_FLIP)


class GroupingModes(IntEnum):
    NEAREST = 0
    THRESHOLD = 1

    @property
    def label(self) -> str:
        """String representation used for CLI flags and protocol form choices"""
        return self.name.lower()


class XmippProtConeGrouping(ProtClassify2D, XmippProtocol):
    """Group projection-aligned particles by viewing direction into 2D classes."""

    _label = "cone grouping"
    _possibleOutputs = {"outputClasses": SetOfClasses2D}
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
            "symmetryGroup",
            StringParam,
            default="c1",
            label="Symmetry group",
            help=(
                "Symmetry group of the volume. Supported groups are "
                "Cn and Dn (n >= 1), T, O, I, and I1–I4."
            ),
        )
        form.addParam(
            "numberOfGroups",
            IntParam,
            label="Number of groups",
            help=(
                "Number of reference viewing directions to generate. "
                "Only groups with assigned particles are published as classes."
            ),
            default=100,
        )
        form.addParam(
            "groupingMode",
            EnumParam,
            default=GroupingModes.NEAREST.value,
            choices=[mode.label for mode in GroupingModes],
            help=(
                "Grouping mode used for clustering particles with similar viewing "
                "directions. 'nearest' means each particle is assigned to the group "
                "whose center is closest to the particle's viewing direction. "
                "'threshold' means a particle gets assigned to every group where "
                "the angular distance between its own viewing direction and the "
                "group's center is less than or equal to a given threshold, chosen "
                "with the 'Angular threshold' parameter. A particle can appear "
                "in multiple classes, each with its own 2D alignment."
            ),
            label="Grouping Mode",
        )
        form.addParam(
            "angularThresholdDeg",
            FloatParam,
            default=10.0,
            help=(
                "In threshold mode, each particle is grouped to every reference viewing direction "
                "whose angular distance to the particle's viewing direction is smaller than or equal "
                "to this threshold. "
                "The threshold ranges from 0 to 90 degrees. "
                "The protocol fails if any particle has no reference within this radius."
            ),
            condition=f"groupingMode == {GroupingModes.THRESHOLD.value}",
            label="Angular Threshold",
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

        form.addSection(label="Compute")
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

    def _validate(self):
        errors = super()._validate()
        particles = self.inputParticles.get()
        if particles is not None:
            if particles.isEmpty():
                errors.append("Input particles must contain at least one particle.")
            if not particles.hasAlignmentProj():
                errors.append("Input particles must have projection alignment.")

        for value, label in (
            (self.numberOfGroups.get(), "Number of groups"),
            (self.groupingBatchSize.get(), "Grouping batch size"),
        ):
            if value is None or value <= 0:
                errors.append(f"{label} must be a positive integer.")

        if self.groupingMode.get() == GroupingModes.THRESHOLD.value:
            radius = self.angularThresholdDeg.get()
            if not isfinite(radius) or not 0 <= radius <= 90:
                errors.append(
                    "Angular threshold must be finite and between 0 and 90 degrees."
                )

        return errors

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputStep")
        self._insertFunctionStep("groupIntoConesStep")
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

    def _getGroupingMode(self):
        return GroupingModes(self.groupingMode.get())

    # --------------------------- STEPS functions --------------------------
    def convertInputStep(self):
        writeSetOfParticles(
            imgSet=self.inputParticles.get(),
            filename=self._getInputMdPath(),
        )

    def groupIntoConesStep(self):
        env = self.getCondaEnv()
        groupingMode = self._getGroupingMode()

        groupingArgs = [
            "--input-xmd",
            self._getInputMdPath(),
            "--out-star",
            self._getGroupingOutputStarPath(),
            "--out-group-column",
            self._getGroupByColumn(),
            "--n-groups",
            str(self.numberOfGroups.get()),
            "--grouping-batch-size",
            str(self.groupingBatchSize.get()),
            "--symmetry-group",
            self.symmetryGroup.get(),
            "--out-reference-md",
            self._getGroupingReferenceMetadataPath(),
            "--grouping-mode",
            groupingMode.label,
        ]

        if groupingMode == GroupingModes.THRESHOLD:
            groupingArgs.extend(
                [
                    "--angular-radius-deg",
                    str(self.angularThresholdDeg.get()),
                    "--unassigned-policy",
                    "error",
                ]
            )

        if self.deduplicateReferences.get():
            groupingArgs.append("--deduplicate-references")

        self.runJob("xmipp_cone_grouping", join(groupingArgs), env=env, numberOfMpi=1)

    def createOutputStep(self):
        coneGeometryByParticle = self._readMemberships()
        inputParticles = self.inputParticles.get()
        outputClasses = self._createSetOfClasses2D(self.inputParticles)
        classesByGroup = {}
        geometryRow = md.Row()

        for particle in inputParticles:
            coneGeometryInfo = coneGeometryByParticle.pop(particle.getObjId(), {})
            for groupId, geometry in coneGeometryInfo.items():
                if groupId not in classesByGroup:
                    cone = Class2D(objId=groupId)
                    cone.copyInfo(inputParticles)
                    cone.setAlignment2D()
                    cone.setRepresentative(None)
                    outputClasses.append(cone)
                    classesByGroup[groupId] = cone

                member = particle.clone()
                member.setClassId(groupId)
                for label, value in zip(_CONE_GEOMETRY_LABELS, geometry):
                    geometryRow.setValue(label, value)
                member.setTransform(rowToAlignment(geometryRow, ALIGN_2D))
                classesByGroup[groupId].append(member)

        if coneGeometryByParticle:
            itemId = next(iter(coneGeometryByParticle))
            outputClasses.close()
            raise RuntimeError(
                f"Cone grouping output refers to unknown itemId={itemId}."
            )

        for cone in classesByGroup.values():
            outputClasses.update(cone)
        outputClasses.write()
        self._defineOutputs(outputClasses=outputClasses)
        self._defineSourceRelation(self.inputParticles, outputClasses)

    def _readMemberships(self):
        """Index cone-specific 2D geometry by particle and group ID."""
        path = self._getGroupingOutputStarPath()
        groupingMetadata = md.MetaData(path)
        groupColumn = self._getGroupByColumn()
        groupLabel = md.str2Label(groupColumn)
        for label in (md.MDL_ITEM_ID, groupLabel, *_CONE_GEOMETRY_LABELS):
            if label < 0 or not groupingMetadata.containsLabel(label):
                column = groupColumn if label == groupLabel else md.label2Str(label)
                raise RuntimeError(
                    f"Missing column {column} in cone grouping output {path}."
                )

        coneGeometryByParticle = {}
        for row in md.iterRows(groupingMetadata):
            itemId = row.getValue(md.MDL_ITEM_ID)
            groupId = int(row.getValue(groupLabel))
            memberships = coneGeometryByParticle.setdefault(itemId, {})
            if groupId in memberships:
                raise RuntimeError(
                    f"Duplicate membership itemId={itemId}, {groupColumn}={groupId} "
                    f"in cone grouping output {path}."
                )
            memberships[groupId] = tuple(
                row.getValue(label) for label in _CONE_GEOMETRY_LABELS
            )

        return coneGeometryByParticle
