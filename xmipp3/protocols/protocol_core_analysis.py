# ******************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Carlos Oscar Sánchez Sorzano (coss@cnb.csic.es)
# *              Daniel Marchán Torres (da.marchan@cnb.csic.es)
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

from os.path import join, exists
from glob import glob


import pyworkflow.protocol.params as param
from pyworkflow.utils.path import makePath

import pwem.emlib.metadata as md
from pwem.protocols import ProtClassify2D
from pwem.constants import ALIGN_2D

from xmipp3.convert import (writeSetOfClasses2D, xmippToLocation,
                            rowToAlignment)


CLASSES_CORE = '_core'


class XmippProtCoreAnalysis(ProtClassify2D):
    """ Analyzes the core of a 2D classification. The core is calculated through
    the Mahalanobis distance from each image to the center of the class. """
    
    _label = 'core analysis'
    
    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        if self.numberOfMpi.get() < 2:
            self.numberOfMpi.set(2)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputClasses', param.PointerParam,
                      label="Input classes",
                      pointerClass='SetOfClasses2D',
                      help='Set of input classes to be analyzed')
        form.addParam('thZscore', param.FloatParam, default=3,
                      label='Junk Zscore',
                      help='Which is the average Z-score to be considered as '
                           'junk. Typical values go from 1.5 to 3. For the '
                           'Gaussian distribution 99.5% of the data is '
                           'within a Z-score of 3. Lower Z-scores reject more '
                           'images. Higher Z-scores accept more images.')
        form.addParam('thPCAZscore', param.FloatParam, default=3,
                      label='PCA Zscore',
                      help='Which is the PCA Z-score to be considered as junk. '
                           'Typical values go from 1.5 to 3. For the Gaussian '
                           'distribution 99.5% of the data is within a '
                           'Z-score of 3. Lower Z-scores reject more images. '
                           'Higher Z-scores accept more images.')
        form.addParallelSection(threads=0, mpi=4)

    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self._defineFileNames()
        self._insertFunctionStep('analyzeCore')
        self._insertFunctionStep('createOutputStep')

    def analyzeCore(self):
        # Put in a function convertInputStep
        fnLevel = self._getExtraPath('level_00')
        makePath(fnLevel)
        inputMdName = join(fnLevel, 'level_classes.xmd')
        writeSetOfClasses2D(self.inputClasses.get(), inputMdName, writeParticles=True)

        args = " --dir %s --root level --computeCore %f %f" % (self._getExtraPath(),
                                                               self.thZscore, self.thPCAZscore)
        self.runJob('xmipp_classify_CL2D_core_analysis', args)
        self.runJob("xmipp_classify_evaluate_classes", "-i %s"%\
                    self._getExtraPath(join("level_00", "level_classes_core.xmd")), numberOfMpi=1)

    #--------------------------- STEPS functions -------------------------------
    def _defineFileNames(self):
        """ Centralize how files are called within the protocol. """
        self.levelPath = self._getExtraPath('level_%(level)02d/')
        myDict = {
                  'final_classes': self._getPath('classes2D%(sub)s.sqlite'),
                  'output_particles': self._getExtraPath('images.xmd'),
                  'level_classes': self.levelPath + 'level_classes%(sub)s.xmd',
                  'level_images': self.levelPath + 'level_images%(sub)s.xmd',
                  'classes_scipion': (self.levelPath + 'classes_scipion_level_'
                                                   '%(level)02d%(sub)s.sqlite'),
                  }
        self._updateFilenamesDict(myDict)

    def createOutputStep(self):
        """ Store the SetOfClasses2D object
        resulting from the protocol execution.
        """
        inputParticles = self.inputClasses.get().getImagesPointer()
        level = self._getLastLevel()
        subset = CLASSES_CORE

        subsetFn = self._getFileName("level_classes", level=level, sub=subset)

        if exists(subsetFn):
            classes2DSet = self._createSetOfClasses2D(inputParticles, subset)
            self._fillClassesFromLevel(classes2DSet, 'last', subset)
            result = {'outputClasses' + subset: classes2DSet}
            self._defineOutputs(**result)
            self._defineSourceRelation(inputParticles, classes2DSet)

    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        validateMsgs = []
        if self.numberOfMpi <= 1:
            validateMsgs.append('Mpi needs to be greater than 1.')
        return validateMsgs
    
    def _citations(self):
        citations=['Sorzano2014']
        return citations

    def _methods(self):
        strline ='We calculated the class cores %s. [Sorzano2014]' % self.getObjectTag('outputClasses_core')
        return [strline]
    
    # --------------------------- UTILS functions -------------------------------
    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(md.MDL_REF))
        item.setTransform(rowToAlignment(row, ALIGN_2D))

    def _updateClass(self, item):
        classId = item.getObjId()

        if classId in self._classesInfo:
            index, fn, _ = self._classesInfo[classId]
            item.setAlignment2D()
            rep = item.getRepresentative()
            rep.setLocation(index, fn)
            rep.setSamplingRate(self.inputClasses.get().getImages().getSamplingRate())

    def _loadClassesInfo(self, filename):
        """ Read some information about the produced 2D classes
        from the metadata file.
        """
        self._classesInfo = {}  # store classes info, indexed by class id

        mdClasses = md.MetaData(filename)

        for classNumber, row in enumerate(md.iterRows(mdClasses)):
            index, fn = xmippToLocation(row.getValue(md.MDL_IMAGE))
            # Store info indexed by id, we need to store the row.clone() since
            # the same reference is used for iteration
            self._classesInfo[index] = (index, fn, row.clone())

    def _fillClassesFromLevel(self, clsSet, level, subset):
        """ Create the SetOfClasses2D from a given iteration. """
        classRf = ''
        self._loadClassesInfo(self._getLevelMdClasses(lev=level, subset=classRf))
        if subset == '' and level == "last":
            xmpMd = self._getFileName('output_particles')
            if not exists(xmpMd):
                xmpMd = self._getLevelMdImages(level, subset)
        else:
            xmpMd = self._getLevelMdImages(level, subset)
        iterator = md.SetMdIterator(xmpMd, sortByLabel=md.MDL_ITEM_ID,
                                    updateItemCallback=self._updateParticle,
                                    skipDisabled=True)
        # itemDataIterator is not neccesary because, the class SetMdIterator
        # contain all the information about the metadata
        clsSet.classifyItems(updateItemCallback=iterator.updateItem,
                             updateClassCallback=self._updateClass)

    def _getLevelMdClasses(self, lev=0, block="classes", subset=""):
        """ Return the classes metadata for this iteration.
        block parameter can be 'info' or 'classes'."""
        if lev == "last":
            lev = self._getLastLevel()
        mdFile = self._getFileName('level_classes', level=lev, sub=subset)
        if block:
            mdFile = block + '@' + mdFile

        return mdFile

    def _getLevelMdImages(self, level, subset):
        if level == "last":
            level = self._getLastLevel()

        xmpMd = self._getFileName('level_images', level=level, sub=subset)
        if not exists(xmpMd):
            self._createLevelMdImages(level, subset)

        return xmpMd

    def _createLevelMdImages(self, level, sub):
        if level == "last":
            level = self._getLastLevel()
        mdClassesFn = self._getLevelMdClasses(lev=level, block="", subset=sub)
        mdImgs = md.joinBlocks(mdClassesFn, "class0")
        mdImgs.write(self._getFileName('level_images', level=level, sub=sub))

    def _getLastLevel(self):
        """ Find the last Level number """
        clsFn = self._getFileName('level_classes', level=0, sub="")
        levelTemplate = clsFn.replace('level_00', 'level_??')
        lev = len(glob(levelTemplate)) - 1
        return lev
