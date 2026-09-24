# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
# *             Tomas Majtner (tmajtner@cnb.csic.es)  -- streaming version
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************

import numpy as np

import pyworkflow.protocol.params as params
from pyworkflow.utils import getFiles, removeBaseExt

from .protocol_particle_pick_consensus import (XmippProtConsensusPicking,
                                               consensusWorker, getReadyMics)

FACTOR_RADIUS = 0.6

class XmippProtPickingRemoveDuplicates(XmippProtConsensusPicking):
    """
    This protocol removes coordinates that are closer than a given threshold.
    The remaining coordinate is the average of the previous ones.

    AI Generated

    ## Overview

    The Remove Duplicates protocol cleans a coordinate set by merging particle
    coordinates that are closer than a selected distance threshold.

    Particle-picking protocols may sometimes produce repeated coordinates for the
    same particle. This can happen when two picks are placed very close to each
    other, when an automatic picker detects the same particle more than once, or
    when manual corrections leave duplicated positions in the coordinate set.

    If duplicates are not removed, the same physical particle may be extracted
    several times. This can bias later processing, overrepresent some particles,
    and introduce unnecessary redundancy into classification or reconstruction.

    This protocol identifies coordinates that are close enough to be considered
    the same particle and replaces them with a single averaged coordinate.

    ## Inputs and General Workflow

    The input is one **SetOfCoordinates**.

    For each micrograph, the protocol reads all coordinates associated with that
    micrograph. It then groups coordinates that fall within the selected radius.
    Each group is treated as one particle, and the output coordinate is computed as
    the average position of the grouped coordinates.

    The protocol writes a new cleaned coordinate set containing the non-duplicated
    coordinates.

    The output can be used directly for particle extraction.

    ## Input Coordinates

    The **Input coordinates** parameter should point to the coordinate set to be
    cleaned.

    This coordinate set may come from manual picking, automatic picking, consensus
    picking, or any other Scipion-compatible particle-picking protocol.

    The protocol processes the coordinates micrograph by micrograph. It preserves
    the association with the original micrographs and creates an output coordinate
    set linked to the input coordinate set.

    ## Radius

    The **Radius** parameter defines the maximum distance, in pixels, within which
    two coordinates are considered duplicates.

    If two or more coordinates are closer than this radius, they are assumed to
    represent the same particle and are merged.

    A small radius removes only nearly identical picks. A larger radius removes
    coordinates that are farther apart, but may accidentally merge nearby distinct
    particles if particles are densely packed.

    The radius should be chosen according to particle size, picking accuracy, and
    particle density.

    ## Automatic Radius from Box Size

    If the radius is set to **-1**, the protocol automatically defines the radius
    as:

    \[
    0.6 \times \text{box size}
    \]

    This default is based on the idea that two coordinates separated by much less
    than the particle box size are likely to refer to the same particle.

    This automatic option is convenient when the coordinate set has a meaningful
    box size. However, users should still consider whether the default radius is
    appropriate for their specimen. Very crowded particles, elongated particles, or
    particles with unusual shapes may require manual adjustment.

    ## Coordinate Merging

    When several coordinates are considered duplicates, the protocol keeps one
    representative coordinate.

    This representative position is computed as the average of the duplicated
    coordinates. Therefore, the output coordinate may be slightly different from
    any of the original positions.

    Averaging is useful when several picks are clustered around the same particle
    center. It provides a central position for extraction rather than arbitrarily
    choosing one of the duplicate picks.

    ## Output Coordinates

    The main output is **outputCoordinates**.

    This output contains the cleaned coordinate set after duplicate removal. It
    uses the same micrographs as the input coordinate set and preserves the
    coordinate-set box size information.

    The output can be passed to extraction protocols in the same way as any other
    coordinate set.

    The number of output coordinates is usually smaller than or equal to the number
    of input coordinates.

    ## Streaming Behavior

    The protocol supports streaming coordinate input.

    As coordinates become available for each micrograph, the protocol can process
    that micrograph and append cleaned coordinates to the output set. The output
    stream remains open until the input coordinate stream is closed and all
    available micrographs have been processed.

    This makes the protocol useful in automated picking pipelines, where
    coordinates may be produced progressively.

    ## Practical Recommendations

    Use this protocol after automatic picking if the picker tends to place several
    coordinates on the same particle.

    Use it after consensus or merged picking workflows when several sources may
    contribute overlapping coordinates.

    Start with the automatic radius if the coordinate box size is reliable. Then
    inspect the cleaned coordinates visually on representative micrographs.

    Decrease the radius if nearby distinct particles are being merged.

    Increase the radius if obvious duplicates remain after cleaning.

    Be especially careful with crowded samples, filaments, aggregates, or particles
    that are very close to each other. In such cases, an overly large radius can
    remove valid neighboring particles.

    ## Final Perspective

    Remove Duplicates is a coordinate-cleaning protocol. It does not pick new
    particles and does not evaluate particle quality. Instead, it ensures that the
    same physical particle is not represented by several nearby coordinates.

    For biological users, this is important because duplicate coordinates can
    propagate into duplicate particle images, biased class averages, and
    unnecessary computational cost.

    The protocol is most useful as a simple quality-control step between particle
    picking and particle extraction.
    """

    _label = 'remove duplicates'
    outputName = 'outputCoordinates'
    FN_PREFIX = 'purgedCoords_'

    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates", important=True,
                      help='Select the set of coordinates to compare')
        form.addParam('consensusRadius', params.IntParam, default=-1,
                      label="Radius",
                      allowsPointers=True,
                      help="All coordinates within this radius (in pixels) "
                           "are presumed to correspond to the same particle.\n "
                           "If -1 then 0.6 * box size is going to be assigned as radius.")

        # FIXME: It's not using more than one since
        #         self.stepsExecutionMode = STEPS_SERIAL
        # form.addParallelSection(threads=4, mpi=0)

#--------------------------- INSERT steps functions ----------------------------
    def insertNewCoorsSteps(self, mics):
        deps = []
        for micrograph in mics:
            stepId = self._insertFunctionStep("removeDuplicatesStep",
                                              micrograph.getObjId(),
                                              micrograph.getMicName(),
                                              prerequisites=[])
            deps.append(stepId)
        return deps

    def _checkNewInput(self):
        # If continue from an stopped run, don't repeat what is done
        if not self.checkedMics:
            for fn in getFiles(self._getExtraPath()):
                fn = removeBaseExt(fn)
                if fn.startswith(self.FN_PREFIX):
                    self.checkedMics.update([self.getMicId(fn)])
                    self.processedMics.update([self.getMicId(fn)])

        readyMics, self.streamClosed = getReadyMics(self.inputCoordinates.get())

        newMicsIds = readyMics.difference(self.checkedMics)

        if newMicsIds:
            self.checkedMics.update(newMicsIds)

            inMics = self.getMainInput().getMicrographs()
            newMics = [inMics[micId].clone() for micId in newMicsIds]

            fDeps = self.insertNewCoorsSteps(newMics)
            outputStep = self._getFirstJoinStep()
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()

    def getMainInput(self):
        return self.inputCoordinates.get()

    def getConsensusRadius(self):
        if self.consensusRadius.get() == -1:
            consensusRadius = int(self.inputCoordinates.get().getBoxSize()*FACTOR_RADIUS)
        else:
            consensusRadius = self.consensusRadius.get()
        return consensusRadius

    def defineRelations(self, outputSet):
        self._defineTransformRelation(self.getMainInput(), outputSet)

    def removeDuplicatesStep(self, micId, micName):
        print("Removing duplicates for micrograph %d: '%s'"
              % (micId, micName))

        coordArray = np.asarray([x.getPosition() for x in
                                 self.getMainInput().iterCoordinates(micId)],
                                dtype=int)

        consensusWorker([coordArray], 1, self.getConsensusRadius(),
                        self._getTmpPath('%s%s.txt' % (self.FN_PREFIX, micId)))

        self.processedMics.update([micId])

    def _validate(self):
        errors = []
        return errors

    def _summary(self):
        return ["Radius = %d" % self.getConsensusRadius()]
