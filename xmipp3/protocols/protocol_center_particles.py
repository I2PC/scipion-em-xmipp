# ******************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
# *              Alberto García Mena (alberto.garcia@cnb.csic.es)
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

import numpy as np
import os
from pyworkflow import VERSION_2_0
from pwem.constants import ALIGN_2D
from pwem.objects import Class2D, Particle, Coordinate, Transform, SetOfClasses2D, SetOfParticles
from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params

from pwem.emlib import MD_APPEND
from xmipp3.convert import (rowToAlignment, alignmentToRow,
                            rowToParticle, writeSetOfClasses2D, xmippToLocation)

OUTPUT_CLASSES = 'outputClasses'
OUTPUT_PARTICLES = 'outputParticles'

class XmippProtCenterParticles(ProtClassify2D):
    """ Recenters particles that are initially un-centered by computing
    alignment shifts relative to a reference point. Proper centering enhances
    the accuracy of subsequent classification and refinement steps.

    AI Generated:

    What this protocol is for

    Center particles is a corrective protocol aimed at a very common practical
    problem in cryo-EM workflows: particles that are systematically off-center
    after picking/extraction or after an early alignment/classification step.
    Even small centering errors can reduce the quality of 2D class averages,
    bias alignment, and ultimately harm 3D refinement—especially when particles
    are small, have weak signal, or when you are trying to push resolution.

    This protocol takes an existing SetOfClasses2D (i.e., 2D classes with their
    member particles) and recenters the particles by computing and applying a
    consistent centering transform derived from the class information. The
    outcome is a new set of 2D classes and a new particle set in which the
    particles have been repositioned so that their centers are better aligned,
    improving downstream classification and refinement stability.

    A biological way to think about it is: you already have a meaningful set of
    class averages, but the particles contributing to them are not properly
    centered; this protocol uses the class centering information to “pull”
    particles toward the correct center, producing cleaner averages and
    better-behaved alignment in later steps.

    What you need to provide

    You provide two inputs:

    First, Input Classes: a SetOfClasses2D. This should be a set of classes
    that already contains the particle members and their current 2D alignment
    parameters. In practice, it is usually the output of a 2D classification
    step (or any step that produces classes with assigned shifts/angles for
    their particles).

    Second, a Set of micrographs associated with those classes. This is
    required because the protocol also updates particle coordinates
    consistently with the new centering, and those coordinates are defined
    relative to the micrographs. Conceptually, this input tells Scipion how the
    particles relate back to the original images once their centering offsets
    are adjusted.

    What the protocol does, in practical terms

    The protocol first computes a centering transform for each class
    representative (the class average). Using those class-level centering
    transforms, it then propagates the correction to all particles in each
    class by updating their 2D alignment parameters and adjusting their
    coordinates accordingly. The protocol is careful to keep the particle
    identity and class membership consistent, but it modifies the metadata so
    that the particle is effectively re-expressed as “more centered”.

    Importantly, this is not “another 2D classification”; it is a centering and
    bookkeeping step that uses the class-derived shifts to correct particle
    placement. You typically run it when you have evidence that the dataset is
    systematically shifted—e.g., class averages look sharp but consistently
    off-center, or you see that the highest-density region is not around the
    box center.

    Outputs: what you get and how to use them

    The protocol produces two outputs:

    outputClasses: a new SetOfClasses2D whose class representatives correspond
    to the centered results, and whose class members carry the updated 2D
    alignment metadata. This is the natural output to inspect if you want to
    see whether class averages look better centered and whether the classes are
    more interpretable.

    outputParticles: a new SetOfParticles containing the recentered particles.
    This is the output you would typically feed into subsequent steps such as
    another round of 2D classification, 3D initial model workflows, or
    refinement pipelines, because many downstream algorithms behave better when
    particles are well centered.

    A key point for biological processing is that the protocol gives you both a
    class-level and a particle-level product. If your next step is another 2D
    classification, you would normally use outputParticles. If you want to
    visually validate the effect quickly, you look at outputClasses and confirm
    that the averages are now centered.

    How to interpret the summary information

    After running, the protocol writes a short summary that includes how many
    particles required meaningful displacement and an estimate of the average
    displacement magnitude (in pixels). Biologically, this helps you understand
    whether you are correcting a subtle nuisance (small average shifts) or a
    major systematic problem (large shifts affecting a significant fraction of
    particles). If a large percentage of particles needed strong displacement,
    it may also indicate upstream issues such as mis-centered picking
    coordinates, extraction offsets, or a mismatch between picking and
    extraction box definitions.

    When this protocol is most useful (typical scenarios)

    This protocol is especially useful when:

    Your particles were extracted with a box center that does not coincide with
    the true particle center (common when using coarse picking or when the
    coordinate reference is inconsistent between programs).

    You have decent 2D classes but the averages are consistently displaced from
    the box center, suggesting systematic centering bias.

    You want to “clean up” centering before starting 3D, because poor centering
    can make initial model generation unstable and can worsen angular
    assignment.

    It is also often helpful as a stabilizing step between early 2D
    classification rounds: run 2D classification, center particles using this
    protocol, then run a second 2D classification that benefits from
    better-centered inputs.

    What this protocol does not replace

    Centering improves the geometry of the dataset, but it does not solve
    problems such as wrong picks, severe heterogeneity, strong contamination,
    or poor CTF correction. If class averages are bad because the particles are
    not the same object or are too noisy, centering will not “fix” that. Its
    role is more specific: it improves consistency by making sure that the
    object of interest sits near the box center across particles.

    Practical advice for biological users

    A simple way to decide whether to use this protocol is to visually inspect
    2D class averages: if many good-looking classes have their main density
    displaced from the center, centering is likely worth doing. After running,
    inspect the new class averages and, if they look more consistently centered,
    continue downstream with outputParticles to benefit from improved alignment
    behavior.
    """
    _label = 'center particles'
    _lastUpdateVersion = VERSION_2_0
    _possibleOutputs = {OUTPUT_CLASSES:SetOfClasses2D, OUTPUT_PARTICLES:SetOfParticles}

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', params.PointerParam,
                      pointerClass='SetOfClasses2D',
                      important=True,
                      label="Input Classes",
                      help='Set of classes to be read')
        form.addParam('inputMics', params.PointerParam,
                      pointerClass='SetOfMicrographs',
                      important=True,
                      label="Set of micrographs",
                      help='Set of micrographs related to the selected input '
                           'classes')

        form.addParallelSection(threads=0, mpi=0)
    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('realignStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------

    def realignStep(self):

        inputMdName = self._getExtraPath('inputClasses.xmd')
        writeSetOfClasses2D(self.inputClasses.get(), inputMdName,
                            writeParticles=True)

        centeredStackName = self._getExtraPath('centeredStack.stk')
        self._params = {'input': inputMdName,
                        'output': centeredStackName}
        args = ('-i %(input)s -o %(output)s --save_metadata_transform')
        self.runJob("xmipp_transform_center_image", args % self._params,
                    numberOfMpi=1)

        centeredMdName = centeredStackName.replace('stk', 'xmd')
        centeredMd = md.MetaData(centeredMdName)
        centeredStack = md.MetaData(centeredStackName)

        listName = []
        listTransform = []
        for rowStk in md.iterRows(centeredStack):
            listName.append(rowStk.getValue(md.MDL_IMAGE))
        for rowMd in md.iterRows(centeredMd):
            listTransform.append(rowToAlignment(rowMd, ALIGN_2D))

        mdNewClasses = md.MetaData()
        for i, row in enumerate(md.iterRows(inputMdName)):
            newRow = md.Row()
            newRow.setValue(md.MDL_IMAGE, listName[i])
            refNum = row.getValue(md.MDL_REF)
            newRow.setValue(md.MDL_REF, refNum)
            classCount = row.getValue(md.MDL_CLASS_COUNT)
            newRow.setValue(md.MDL_CLASS_COUNT, classCount)
            newRow.addToMd(mdNewClasses)
        mdNewClasses.write('classes@' + self._getExtraPath('final_classes.xmd'),
                           MD_APPEND)

        mdImages = md.MetaData()
        i = 0
        mdBlocks = md.getBlocksInMetaDataFile(inputMdName)
        resultMat = Transform()
        listDisplacament = []
        centerSummary = self._getPath("summary.txt")
        centerSummary = open(centerSummary, "w")
        particlesCentered = 0
        totalParticles = 0

        for block in mdBlocks:
            if block.startswith('class00'):
                newMat = listTransform[i]
                newMatrix = newMat.getMatrix()
                mdClass = md.MetaData(block + "@" + inputMdName)
                mdNewClass = md.MetaData()
                i += 1
                flag_psi = True
                for rowIn in md.iterRows(mdClass):
                    #To create the transformation matrix (and its parameters)
                    #  for the realigned particles
                    totalParticles += 1
                    if rowIn.getValue(md.MDL_ANGLE_PSI)!=0:
                        flag_psi=True
                    if rowIn.getValue(md.MDL_ANGLE_ROT)!=0:
                        flag_psi=False
                    inMat = rowToAlignment(rowIn, ALIGN_2D)
                    inMatrix = inMat.getMatrix()
                    resultMatrix = np.dot(newMatrix,inMatrix)
                    resultMat.setMatrix(resultMatrix)
                    rowOut=md.Row()
                    rowOut.copyFromRow(rowIn)
                    alignmentToRow(resultMat, rowOut, ALIGN_2D)
                    if flag_psi==False:
                        newAngle = rowOut.getValue(md.MDL_ANGLE_PSI)
                        rowOut.setValue(md.MDL_ANGLE_PSI, 0.)
                        rowOut.setValue(md.MDL_ANGLE_ROT, newAngle)

                    #To create the new coordinates for the realigned particles
                    inPoint = np.array([[0.],[0.],[0.],[1.]])
                    invResultMat = np.linalg.inv(resultMatrix)
                    centerPoint = np.dot(invResultMat,inPoint)

                    if int(centerPoint[0]) > 1 or int(centerPoint[1]) > 1:
                        particlesCentered += 1
                        listDisplacament.append([int(centerPoint[0]), int(centerPoint[1])])

                    rowOut.setValue(md.MDL_XCOOR, rowOut.getValue(
                        md.MDL_XCOOR)+int(centerPoint[0]))
                    rowOut.setValue(md.MDL_YCOOR, rowOut.getValue(
                        md.MDL_YCOOR)+int(centerPoint[1]))

                    rowOut.addToMd(mdNewClass)

                mdNewClass.write(block + "@" + self._getExtraPath(
                    'final_classes.xmd'), MD_APPEND)
                mdImages.unionAll(mdNewClass)

        listModule = [(np.sqrt((x[0] * x[0]) + (x[1] * x[1]))) for x in listDisplacament]
        moduleDisp = round(sum(listModule) / len(listModule), 1)
        centerSummary.write("Particles centered: {} \t({}%) "
                            "\nAverage module displacement: {}px\n".
                            format(particlesCentered,
                            round((100 * particlesCentered/totalParticles ), 1),
                            moduleDisp))
        centerSummary.close()
        mdImages.write(self._getExtraPath('final_images.xmd'))


    def createOutputStep(self):
        inputParticles = self.inputClasses.get().getImages()
        outputClasses = self._createSetOfClasses2D(self.inputClasses.get().getImagesPointer())
        self._fillClasses(outputClasses)
        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(inputParticles)
        self._fillParticles(outputParticles)

        result = {OUTPUT_CLASSES: outputClasses, OUTPUT_PARTICLES: outputParticles}
        self._defineOutputs(**result)
        self._defineSourceRelation(self.inputClasses, outputClasses)
        self._defineSourceRelation(self.inputClasses, outputParticles)

    # --------------------------- UTILS functions ------------------------------
    def _fillClasses(self, outputClasses):
        """ Create the SetOfClasses2D """
        inputSet = self.inputClasses.get().getImages()
        myRep = md.MetaData('classes@' + self._getExtraPath(
            'final_classes.xmd'))

        for row in md.iterRows(myRep):
            fn = row.getValue(md.MDL_IMAGE)
            rep = Particle()
            rep.setLocation(xmippToLocation(fn))
            repId = row.getObjId()
            newClass = Class2D(objId=repId)
            newClass.setAlignment2D()
            newClass.copyInfo(inputSet)
            newClass.setAcquisition(inputSet.getAcquisition())
            newClass.setRepresentative(rep)
            outputClasses.append(newClass)

        i = 1
        mdBlocks = md.getBlocksInMetaDataFile(self._getExtraPath(
            'final_classes.xmd'))
        for block in mdBlocks:
            if block.startswith('class00'):
                mdClass = md.MetaData(block + "@" + self._getExtraPath(
                    'final_classes.xmd'))
                imgClassId = i
                newClass = outputClasses[imgClassId]
                newClass.enableAppend()
                for row in md.iterRows(mdClass):
                    part = rowToParticle(row)
                    newClass.append(part)
                i += 1
                newClass.setAlignment2D()
                outputClasses.update(newClass)

    def _fillParticles(self, outputParticles):
        """ Create the SetOfParticles"""
        myParticles = md.MetaData(self._getExtraPath('final_images.xmd'))
        outputParticles.enableAppend()
        for row in md.iterRows(myParticles):
            #To create the new particle
            p = rowToParticle(row)
            outputParticles.append(p)

    # --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []
        summary.append("Realignment of %s classes."
                       % self.inputClasses.get().getSize())

        centerSummary = self._getPath("summary.txt")
        if not os.path.exists(centerSummary):
            summary.append("No summary file yet.")
        else:
            centerSummary = open(centerSummary, "r")
            for line in centerSummary.readlines():
                summary.append(line.rstrip())
            centerSummary.close()
        return summary

    def _validate(self):
        errors = []
        try:
            self.inputClasses.get().getImages()
        except AttributeError:
            errors.append('Try and catch InputClasses has no particles.')

        return errors

