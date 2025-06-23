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
    """ Recenters particles that are initially un-centered by computing alignment shifts relative to a reference point. Proper centering enhances the accuracy of subsequent classification and refinement steps. """
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

