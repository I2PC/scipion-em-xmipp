# ******************************************************************************
# *
# * Authors:     Erney Ramirez-Aportela (eramnirez@cnb.csic.es)
# *             
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
import random

from pyworkflow import VERSION_3_0
from pwem import emlib
from pwem.constants import ALIGN_2D
from pwem.objects import Class2D, Particle, Coordinate, Transform
from pwem.protocols import ProtClassify2D
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
import pickle
import xmipp3


from pwem.emlib import MD_APPEND
from xmipp3.convert import (rowToAlignment, alignmentToRow,
                            rowToParticle, writeSetOfClasses2D, xmippToLocation)


class XmippProtDeepClassify(ProtClassify2D, xmipp3.XmippProtocol):
    """ Realignment of un-centered particles. """
    _label = 'deep classify'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_DLTK_v0.3'

    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', params.PointerParam,
                      pointerClass='SetOfClasses2D',
                      important=True,
                      label="Input Classes",
                      help='Set of classes to be realing')

        form.addParam('inputSet', params.PointerParam, label="Input images to predict",
                      pointerClass='SetOfParticles')

        form.addParallelSection(threads=0, mpi=0)
    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('readImageStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------

    def readImageStep(self):

        inputMdName = self._getExtraPath('inputClasses.xmd')
        writeSetOfClasses2D(self.inputClasses.get(), inputMdName,
                            writeParticles=True)

        setImage = self.inputSet.get()

        dim = int(self.inputClasses.get().getDimensions()[0])
        print("dimensiones = ", dim)

        cl = []
        cl2 = []
        listCount = []
        
        for row in md.iterRows(inputMdName):
        
            classFile = row.getValue(md.MDL_IMAGE)
            classCount = row.getValue(md.MDL_CLASS_COUNT)
            refNum = row.getValue(md.MDL_REF)

            classImage = emlib.Image(classFile).getData()
            # normalization
            classImage = (classImage - np.mean(classImage)) / np.std(classImage)

            #classImage_array = np.array(classImage)

            print(classImage)

            print(classCount)
            
            listCount.append(classCount)
            cl.append(classImage)
            cl2.append(classFile)
        np.save(self._getTmpPath('clases_array.npy'), cl)

        print(refNum)
        print(listCount)

        #save list of image for class
        countFile = '%s' %self._getTmpPath('countList')
        with open(countFile, "wb") as fp:
            pickle.dump(listCount, fp)

        count=0
        mdBlocks = md.getBlocksInMetaDataFile(inputMdName)

        listImage = []
        for block in mdBlocks:
                    
            # listImage.append([])

                       
            if block.startswith('class' + cl2[count].split("@")[0]):
                
                mdClass = md.MetaData(block + "@" + inputMdName)
                
                for rowIn in md.iterRows(mdClass):
                    image = rowIn.getValue(md.MDL_IMAGE)

                    image_array = emlib.Image(image).getData()
                    # normalization
                    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

                    listImage.append(image_array)

                count+=1

        # listImageF = np.array([elem for singleList in listImage for elem in singleList])
        #listImageF = np.array(listImage)

        np.save(self._getTmpPath('images_array.npy'), listImage)

        
        #select class-image pair randomly

        # n = np.random.randint(0,refNum-1)
        # print("random class = ",n)
        # clTest = cl[n]
        # imTest = np.random.choice(listImage[n])
        # print(clTest, imTest)

        # training mode
        #args = "%s" %self._getTmpPath('images_array.npy')
        args = "%s  %s  %s  %d %d" %(self._getTmpPath('clases_array.npy'),  self._getTmpPath('images_array.npy') ,self._getTmpPath('countList'), refNum, dim)
        self.runJob("xmipp_deep_classify", args, env=self.getCondaEnv())

        # predict mode
        #args = "%s" %self._getTmpPath('images_array.npy')
        model_save = self._getTmpPath('model_train.h5')
        args = "%s  %d %s" %(self._getTmpPath('images_predict_array.npy'), dim, model_save)
        self.runJob("xmipp_deep_classify_predict", args, env=self.getCondaEnv())
        











    def createOutputStep(self):
        inputParticles = self.inputClasses.get().getImages()

        outputClasses = self._createSetOfClasses2D(inputParticles)
        self._fillClasses(outputClasses)
        result = {'outputClasses': outputClasses}
        self._defineOutputs(**result)
        self._defineSourceRelation(self.inputClasses, outputClasses)

        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(inputParticles)
        self._fillParticles(outputParticles)
        result = {'outputParticles': outputParticles}
        self._defineOutputs(**result)
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
        """ Create the SetOfParticles and SetOfCoordinates"""
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
        return summary

    # def _validate(self):
    #     errors = []
    #     try:
    #         self.inputClasses.get().getImages().getAcquisition()
    #     except AttributeError:
    #         errors.append('InputClasses has not clases')
    #
    #     return errors

