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
    the Mahalanobis distance from each image to the center of the class.

    AI Generated:

    ## Overview

    The Core Analysis protocol evaluates the internal consistency of 2D classes
    by identifying the most representative particles within each class. It is
    based on statistical distances—specifically the Mahalanobis distance—between
    each particle and the center of its assigned class. The goal is to
    distinguish well-aligned, structurally consistent particles (the *core*)
    from outliers or poorly aligned images (often referred to as *junk*).

    In practical cryo-EM workflows, this protocol is particularly useful after
    a 2D classification step. Even when classes appear visually clean, they
    often contain particles that do not fully conform to the dominant structure.
    Removing these particles improves the quality of downstream steps such as
    3D reconstruction, refinement, or heterogeneity analysis.

    For a biological user, this protocol provides a principled way to “clean”
    classes without relying purely on visual inspection, which can be
    subjective and time-consuming.

    ## Inputs and General Workflow

    The protocol takes as input a **set of 2D classes**, typically produced by
    a previous classification method. Each class contains particles that are
    assumed to represent similar projections of the underlying structure.

    The analysis proceeds by modeling the statistical distribution of particles
    within each class. Using this model, the protocol computes how far each
    particle deviates from the class center. Particles that deviate strongly
    are considered less reliable and may be excluded from the class core.

    The result is a refined set of classes that contains only the most
    representative particles, while outliers are effectively discarded.

    ## Understanding the Concept of “Core”

    The *core* of a class can be understood as the subset of particles that
    best represent the underlying signal. These particles are mutually
    consistent in both appearance and alignment.

    In contrast, particles outside the core may arise from several sources:

    * Misalignment during classification
    * Structural heterogeneity (different conformations)
    * Noise-dominated images
    * Contaminants or artifacts

    By focusing on the core, the protocol enhances the structural signal and
    reduces variability that is not biologically meaningful.

    ## Z-score Thresholds: Controlling Particle Selection

    The main parameters of this protocol are two Z-score thresholds, which
    control how strictly particles are filtered.

    ### Junk Z-score

    This parameter defines how far a particle can deviate from the class center
    before being considered an outlier. Lower values make the selection
    stricter, removing more particles. Higher values are more permissive and
    retain more images.

    From a practical perspective, values around 2–3 are commonly used. A value
    near 3 corresponds roughly to keeping particles within the main body of a
    Gaussian distribution. Reducing this threshold is useful when classes are
    noisy or suspected to contain significant contamination.

    ### PCA Z-score

    This parameter performs a similar filtering but in a reduced feature space
    obtained through principal component analysis (PCA). It captures
    variability in the main modes of variation within the class.

    Biologically, this is particularly relevant when subtle structural
    differences or alignment inconsistencies exist. The PCA-based filtering can
    detect outliers that are not obvious in the original space.

    As with the Junk Z-score, lower values enforce stricter filtering.

    ## Outputs and Their Interpretation

    The protocol produces a new set of 2D classes containing only the particles
    that belong to the *core* of each class.

    Importantly, the class identities are preserved, but their composition
    changes: each class now contains fewer particles, ideally those that are
    more consistent and better aligned.

    From a biological standpoint, these refined classes typically show:

    * Sharper structural features
    * Reduced noise
    * Improved interpretability

    However, users should be aware that aggressive filtering may remove rare
    but biologically relevant states. This is particularly important in systems
    with continuous heterogeneity or multiple conformations.

    ## Practical Recommendations

    In routine workflows, this protocol is best used after an initial 2D
    classification, especially when preparing data for high-resolution
    reconstruction.

    A good strategy is to start with moderate Z-score thresholds (around 3)
    and visually inspect the resulting classes. If classes still appear noisy
    or blurred, lowering the thresholds can improve consistency.

    For datasets with suspected heterogeneity, caution is advised.
    Over-filtering may eliminate particles corresponding to minor
    conformational states, which could be biologically important.

    It is also useful to compare results before and after core analysis.
    Improvements in class sharpness and downstream reconstruction quality are
    good indicators that the protocol has been beneficial.

    ## Final Perspective

    Core Analysis is a statistically grounded alternative to manual class
    cleaning. By identifying the most representative particles within each
    class, it enhances data quality in a reproducible and objective manner.

    For most cryo-EM users, this protocol serves as a bridge between initial
    classification and high-quality structural interpretation, helping to
    ensure that subsequent analyses are based on the most reliable subset of
    the data.

    """
    
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
