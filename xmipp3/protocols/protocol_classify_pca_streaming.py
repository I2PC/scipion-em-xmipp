# ******************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
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
import sys
import timeit
from os.path import join, dirname, exists
import os
from datetime import datetime
import time
from pyworkflow.utils import prettyTime
# import emtable

from pyworkflow import VERSION_3_0
from pyworkflow.object import Set, Float, String
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, EnumParam, IntParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STATUS_NEW
from pyworkflow.utils.path import cleanPath, makePath
from pyworkflow.protocol import ProtStreamingBase, STEPS_PARALLEL

import pwem.emlib.metadata as md
from pwem.protocols import ProtClassify2D
from pwem.objects import SetOfClasses2D, SetOfParticles
from pwem.constants import ALIGN_NONE, ALIGN_2D
import xmipp3

from xmipp3.convert import (writeSetOfParticles, createItemMatrix,
                            writeSetOfClasses2D, xmippToLocation,
                            rowToAlignment)
from threading import Event

OUTPUT = "outputClasses"
BATCH_CLASSIFICATION = 5000
BATCH_UPDATE = 1000


def updateEnviron(gpuNum):
    """ Create the needed environment for pytorch programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)


class XmippProtClassifyPcaStreaming(ProtClassify2D, ProtStreamingBase, xmipp3.XmippProtocol):
    """ Classifies a set of images. """

    _label = '2D classification pca streaming'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_pyTorch'

    # Mode
    CREATE_CLASSES = 0
    UPDATE_CLASSES = 1

    _possibleOutputs = {OUTPUT: SetOfClasses2D}

    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL
        self._initialStep()
    # if self.numberOfMpi.get() < 2:
    #     self.numberOfMpi.set(2)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addHidden(GPU_LIST, StringParam, default='0',
                       label="Choose GPU ID",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")

        form.addSection(label='Input')

        form.addParam('inputParticles', PointerParam,
                      label="Input images",
                      important=True, pointerClass='SetOfParticles',
                      help='Select the input images to be classified.')
        form.addParam('numberOfClasses', IntParam, default=50,
                      label='Number of classes:',
                      help='Number of classes (or references) to be generated.')
        form.addParam('mode', EnumParam, choices=['create_classes', 'update_classes'],
                      label="Create or update 2D classes?", default=self.CREATE_CLASSES,
                      display=EnumParam.DISPLAY_HLIST,
                      help='This option allows for either global refinement from an initial volume '
                           ' or just alignment of particles. If the reference volume is at a high resolution, '
                           ' it is advisable to only align the particles and reconstruct at the end of the iterative process.')
        form.addParam('initialClasses', PointerParam,
                      label="Initial classes",
                      condition="mode",
                      pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Set of initial classes to start the classification')
        form.addParam('mask', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
                      label='Use Gaussian Mask?',
                      help='If you set to *Yes*, a gaussian mask is applied to the images.')
        form.addParam('sigma', IntParam, default=-1, expertLevel=LEVEL_ADVANCED,
                      label='sigma:', condition="mask",
                      help='Sigma is the parameter that controls the dispersion or "width" of the curve..')

        form.addSection(label='Pca training')

        form.addParam('resolution', FloatParam, label="max resolution", default=10,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef', FloatParam, label="% variance", default=0.4, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of coefficients to be considers (between 0-1).'
                           ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('training', IntParam, default=40000,
                      label="particles for training",
                      help='Number of particles for PCA training')

        # form.addParallelSection(threads=1, mpi=4)
        form.addParallelSection(threads=3, mpi=1)  # Poner 4 mpi?

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        """
        This step should be implemented by any streaming protocol.
        It should check its input and when ready conditions are met
        call the self._insertFunctionStep method.
        """
        newParticlesSet = self._loadEmptyParticleSet()
        self._initFnStep()
        self._insertFunctionStep(self.closeOutputStep, wait=True)
        self.newDeps = []

        while not self.finish:
            if not self._newParticlesToProcess():
                self.info('No new particles')
            else:
                closeStep = self._getFirstJoinStep()
                particlesSet = self._loadInputParticleSet()
                self.isStreamClosed = particlesSet.getStreamState()

                for particle in particlesSet.iterItems(where="id > %s" % self.lastParticleId):
                    newParticlesSet.append(particle.clone())

                self.info(f'%d new particles' % len(newParticlesSet))
                # ------------------------------------ PCA TRAINING ----------------------------------------------
                if self.doPcaTraining(newParticlesSet):
                    self._insertPCASteps(newParticlesSet)
                # ------------------------------------ CLASSIFICATION ----------------------------------------------
                if self.doClassification(newParticlesSet):
                    # TODO: va a haber un tamaño minimo para la primera clasificacion en el form?
                    self._insertClassificationSteps(newParticlesSet)
                    newParticlesSet = self._loadEmptyParticleSet()
                # ------------------------------------ Update ----------------------------------------------
                self.lastParticleId = max(particlesSet.getIdSet())
                print('Last particle input id', self.lastParticleId)
                closeStep.addPrerequisites(*self.newDeps)

            if self.isStreamClosed == Set.STREAM_CLOSED:
                print('Stream closed')
                # Finish everything and close output sets
                if len(newParticlesSet):
                    print(f'Finish processing with last batch %d' % len(newParticlesSet))
                    self.lastRound = True
                else:
                    outputStep = self._getFirstJoinStep()
                    if outputStep and outputStep.isWaiting():
                        outputStep.setStatus(STATUS_NEW)
                        self.finish = True

            sys.stdout.flush()
            time.sleep(10)

    # --------------------------- STEPS functions -------------------------------
    def _initialStep(self):
        updateEnviron(self.gpuList.get())
        self.finish = False
        self.lastParticleId = 0
        self.lastParticleProcessedId = 0
        self.flagPcaDone = False
        self.isStreamClosed = False
        self.lastRound = False
        self.particlesProcessed = []
        self.classificationRound = 0
        if self.mode == self.UPDATE_CLASSES:
            self.classificationDone = True
        else:
            self.classificationDone = False

    def _initFnStep(self):
        self.imgsPcaXmd = self._getExtraPath('images_pca.xmd')
        self.imgsPcaFn = self._getTmpPath('images_pca.mrc')
        self.imgsOrigXmd = self._getExtraPath('imagesInput_.xmd')
        self.imgsXmd = self._getTmpPath('images_.xmd')
        self.imgsFn = self._getTmpPath('images_.mrc')
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getTmpPath('references.mrcs')

    def _insertPCASteps(self, newParticlesSet):
        # Process data
        self.convertStep = self._insertFunctionStep(self.convertInputStep,
                                                    # self.inputParticles.get(), self.imgsOrigXmd, self.imgsXmd) #wiener
                                                    newParticlesSet, self.imgsPcaXmd, self.imgsPcaFn,
                                                    prerequisites=[])
        numTrain = min(len(newParticlesSet), self.training.get())
        self.pcaStep = self._insertFunctionStep(self.pcaTraining, self.imgsPcaFn, self.resolution.get(),
                                                numTrain, prerequisites=self.convertStep)
        self.flagPcaDone = True  # Activate flag so this step is only done once

    def _insertClassificationSteps(self, newParticlesSet):
        self.updateFnClassification()
        self.convertStep2 = self._insertFunctionStep(self.convertInputStep,
                                                     # self.inputParticles.get(), self.imgsOrigXmd, self.imgsXmd) #wiener
                                                     newParticlesSet, self.imgsOrigXmd, self.imgsFn,
                                                     prerequisites=self.pcaStep)
        self.classStep = self._insertFunctionStep(self.classification, self.imgsFn, self.numberOfClasses.get(),
                                                  self.imgsOrigXmd, self.mask.get(), self.sigma.get(),
                                                  prerequisites=self.convertStep2)
        self.updateStep = self._insertFunctionStep(self._updateOutputSetOfClasses, newParticlesSet,
                                                   Set.STREAM_OPEN, prerequisites=self.classStep)
        self.newDeps.append(self.updateStep)

    def convertInputStep(self, input, outputOrig, outputMRC):
        writeSetOfParticles(input, outputOrig)
        if self.mode == self.UPDATE_CLASSES:
            initial2DClasses = self.initialClasses.get()
            if isinstance(initial2DClasses, SetOfClasses2D):
                writeSetOfClasses2D(initial2DClasses,
                                    self.refXmd, writeParticles=False)
            else:
                writeSetOfParticles(initial2DClasses,
                                    self.refXmd)

            args = ' -i  %s -o %s  ' % (self.refXmd, self.ref)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)
        # args = ' -i %s  -o %s --pixel_size %s --spherical_aberration %s --voltage %s --batch 1024 --device cuda:0'% \
        #         (outputOrig, outputMRC, self.sampling, self.acquisition.getSphericalAberration(), self.acquisition.getVoltage())
        # env = self.getCondaEnv()
        # env['LD_LIBRARY_PATH'] = ''
        # self.runJob("xmipp_swiftalign_wiener_2d", args, numberOfMpi=1, env=env)
        args = ' -i  %s -o %s  ' % (outputOrig, outputMRC)
        self.runJob("xmipp_image_convert", args, numberOfMpi=1)

    def pcaTraining(self, inputIm, resolutionTrain, numTrain):
        args = ' -i %s  -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca  --batchPCA' % \
               (inputIm, self.sampling, resolutionTrain, self.coef.get(), numTrain, self._getExtraPath())

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca_train", args, numberOfMpi=1, env=env)

    def classification(self, inputIm, numClass, stfile, mask, sigma):
        args = ' -i %s -s %s -c %s -n 18 -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt -o %s/classes -stExp %s' % \
               (inputIm, self.sampling, numClass, self._getExtraPath(), self._getExtraPath(), self._getExtraPath(),
                stfile)
        if mask:
            args += ' --mask --sigma %s ' % (sigma)

        if self.mode == self.UPDATE_CLASSES or self.classificationDone:
            args += ' -r %s ' % self.ref

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca", args, numberOfMpi=1, env=env)
        # Wait until the classification is finished
        self.classificationDone = True

    def _updateOutputSetOfClasses(self, particles, streamMode):
        outputName = 'outputClasses'
        outputClasses, update = self._loadOutputSet(outputName)
        self._fillClassesFromLevel(outputClasses, update)
        self._updateOutputSet(outputName, outputClasses, streamMode)

        if not update: # First time
            self._defineSourceRelation(self.getInputPointer(), outputClasses)
            # For further updating
            writeSetOfClasses2D(outputClasses,
                                self.refXmd, writeParticles=False)

            args = ' -i  %s -o %s  ' % (self.refXmd, self.ref)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)

        self.particlesProcessed.extend(particles.getIdSet())
        self.lastParticleProcessedId = max(particles.getIdSet())

    def closeOutputStep(self):
        self._closeOutputSet()

    # --------------------------- UTILS functions -----------------------------
    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all micrographs
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'closeOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

    def _loadInputParticleSet(self):
        """ Returns te input set of particles"""
        partSet = self.inputParticles.get()
        partSet.loadAllProperties()

        return partSet

    def getInputPointer(self):
        return self.inputParticles

    def _loadEmptyParticleSet(self):
        partSet = self.inputParticles.get()
        partSet.loadAllProperties()
        self.sampling = partSet.getSamplingRate()
        self.acquisition = partSet.getAcquisition()
        copyPartSet = self._createSetOfParticles()
        # copyPartSet = SetOfParticles()
        copyPartSet.copyInfo(partSet)
        copyPartSet.setSamplingRate(self.sampling)
        copyPartSet.setAcquisition(self.acquisition)

        return copyPartSet

    def updateFnClassification(self):
        """Update input based on the iteration it is"""
        self.imgsOrigXmd = updateFileName(self.imgsOrigXmd, self.classificationRound)
        self.imgsXmd = updateFileName(self.imgsXmd, self.classificationRound)
        self.imgsFn = updateFileName(self.imgsFn, self.classificationRound)
        self.classificationRound += 1

    def _newParticlesToProcess(self):
        particlesFile = self.inputParticles.get().getFileName()
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(os.path.getmtime(particlesFile))
        self.debug('Last check: %s, modification: %s'
                   % (prettyTime(self.lastCheck),
                      prettyTime(mTime)))
        # If the input have not changed since our last check,
        # it does not make sense to check for new input data
        if (self.lastCheck > mTime and self.lastParticleId>0) and not self.lastRound:
            newParticlesBool = False
        else:
            newParticlesBool = True

        self.lastCheck = now
        return newParticlesBool

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
            rep.setSamplingRate(self.inputParticles.get().getSamplingRate())

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
            self._classesInfo[classNumber + 1] = (index, fn, row.clone())

    def _fillClassesFromLevel(self, clsSet, update = False):
        """ Create the SetOfClasses2D from a given iteration. """
        self._loadClassesInfo(self._getExtraPath('classes_classes.star'))

        xmpMd = self._getExtraPath('classes_images.star')

        iterator = md.SetMdIterator(xmpMd, sortByLabel=md.MDL_ITEM_ID,
                                    updateItemCallback=self._updateParticle,
                                    skipDisabled=True)
        params = {}
        if update:
            print(r'Last particle processed id is %s' %self.lastParticleProcessedId)
            params = {"where":"id > %s" %self.lastParticleProcessedId}

        with self._lock:
            clsSet.classifyItems(updateItemCallback=iterator.updateItem,
                             updateClassCallback=self._updateClass, iterParams=params)

    def _loadOutputSet(self, outputName):
        """
        Load the output set if it exists or create a new one.
        """
        outputSet = getattr(self, outputName, None)
        update = False
        if outputSet is None:
            outputSet = self._createSetOfClasses2D(self.getInputPointer())
            outputSet.setStreamState(Set.STREAM_OPEN)
        else:
            update = True

        return outputSet, update

    def doPcaTraining(self, newParticlesSet):
        return (len(newParticlesSet) >= self.training.get() and not self.flagPcaDone) or \
                (self.lastRound and not self.flagPcaDone)

    def doClassification(self, newParticlesSet):
        return (len(newParticlesSet) >= BATCH_CLASSIFICATION and self.flagPcaDone and not self.classificationDone) \
                        or (len(newParticlesSet) >= BATCH_UPDATE and self.classificationDone) or self.lastRound

def updateFileName(filepath, round):
    filename = os.path.basename(filepath)
    new_filename = f"{filename[:filename.find('_')]}_{round}{filename[filename.rfind('.'):]}"
    return os.path.join(os.path.dirname(filepath), new_filename)