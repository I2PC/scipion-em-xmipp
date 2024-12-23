# ******************************************************************************
# *
# * Authors: Erney Ramirez Aportela (eramirez@cnb.csic.es)
# *          Daniel Marchan Torres (da.marchan@cnb.csic.es)
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
import emtable
import os
from datetime import datetime
import time

from pyworkflow.utils import prettyTime
from pyworkflow import VERSION_3_0
from pyworkflow.object import Set
from pyworkflow.protocol.params import IntParam
from pyworkflow.constants import BETA

from pwem.objects import SetOfClasses2D, SetOfAverages, SetOfParticles
from pwem.constants import ALIGN_NONE, ALIGN_2D

from xmipp3.convert import (readSetOfParticles, writeSetOfParticles,
                            writeSetOfClasses2D)
from xmipp3.protocols.protocol_classify_pca import updateEnviron, XmippProtClassifyPca

OUTPUT_CLASSES = "outputClasses"
OUTPUT_AVERAGES = "outputAverages"
PCA_FILE = "pca_done.txt"
CLASSIFICATION_FILE = "classification_done.txt"
LAST_DONE_FILE = "last_done.txt"
BATCH_UPDATE = 5000


class XmippProtClassifyPcaStreaming(XmippProtClassifyPca):
    """ Classifies a set of images. """

    _label = '2D classification pca streaming'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_pyTorch'
    _devStatus = BETA

    # Mode
    CREATE_CLASSES = 0
    UPDATE_CLASSES = 1

    _possibleOutputs = {OUTPUT_CLASSES: SetOfClasses2D,
                        OUTPUT_AVERAGES: SetOfAverages}

    def __init__(self, **args):
        XmippProtClassifyPca.__init__(self, **args)
        #self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form = self._defineCommonParams(form)
        form.addSection(label='Classification')
        form.addParam('classificationBatch', IntParam, default=50000,
                      condition="not mode",
                      label="particles for initial classification",
                      help='Number of particles for an initial classification to compute the 2D references')

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self.initializeParams()
        self._insertFunctionStep(self.createOutputStep,
                                 prerequisites=[], wait=True, needsGPU=False)

    def createOutputStep(self):
        self._closeOutputSet()

    def initializeParams(self):
        # Streaming variavles
        self.finished = False
        self.newDeps = []
        # Important to have both:
        self.insertedIds = []  # Contains images that have been inserted in a Step (checkNewInput).
        self.processedIds = []  # Ids to be output
        self.isStreamClosed = self.inputParticles.get().isStreamClosed()
        self.inputFn = self.inputParticles.get().getFileName()

        self.staticRun = True if self.isStreamClosed else False
        if self.staticRun:
            self.info('Static Run')

        #  Program variables
        self._initialStep()

    def _getFirstJoinStepName(self):
        # This function will be used for streaming, to check which is
        # the first function that need to wait for all ctfs
        # to have completed, this can be overriden in subclasses
        # (e.g., in Xmipp 'sortPSDStep')
        return 'createOutputStep'

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == self._getFirstJoinStepName():
                return s
        return None

    def _stepsCheck(self):
        self._checkNewInput()
        #self._checkNewOutput()

    def _checkNewInput(self):
        # Check if there are new images to process from the input set
        self.lastCheck = getattr(self, 'lastCheck', datetime.now())
        mTime = datetime.fromtimestamp(os.path.getmtime(self.inputFn))
        self.debug('Last check: %s, modification: %s'
                    % (prettyTime(self.lastCheck),
                       prettyTime(mTime)))
        # If the input.sqlite have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and self.insertedIds:  # If this is empty it is due to a static "continue" action or it is the first round
            return None

        inputSet = self._loadInputSet(self.inputFn)
        inputSetIds = inputSet.getIdSet()

        if self.insertedIds:
            newIds = [idImage for idImage in inputSetIds if idImage not in self.insertedIds]
        else:
            newIds = list(inputSetIds)

        self.lastCheck = datetime.now()
        self.isStreamClosed = inputSet.isStreamClosed()
        inputSet.close()

        outputStep = self._getFirstJoinStep()

        if self.isContinued() and not self.insertedIds:  # For "Continue" action and the first round
            doneIds, _ = self._getAllDoneIds()
            skipIds = list(set(newIds).intersection(set(doneIds)))
            newIds = list(set(newIds).difference(set(doneIds)))
            self.info("Skipping Images with ID: %s, seems to be done" % skipIds)
            self.insertedIds = doneIds  # During the first round of "Continue" action it has to be filled

        if newIds and (self._doPcaTraining(newIds) or self._doClassification(newIds)):
            fDeps = self._insertProcessingSteps(newIds)

            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)

            self.updateSteps()

    def _loadInputSet(self, inputFn):
        self.debug("Loading input db: %s" % inputFn)
        inputSet = SetOfParticles(filename=inputFn)
        inputSet.loadAllProperties()
        return inputSet

    # ------------------------------------------ Utils ---------------------------------------

    def _getAllDoneIds(self):
        doneIds = []
        sizeOutput = 0

        if hasattr(self, OUTPUT_CLASSES):
            sizeOutput = self.outputClasses.getImages().getSize()
            doneIds.extend(list(self.outputClasses.getImages().getIdSet()))

        return doneIds, sizeOutput

    # --------------------------- STEPS functions -------------------------------
    def _initialStep(self):
        #self.lastRound = False
        #self.pcaLaunch = False
        self.classificationLaunch = False
        self.classificationRound = 0
        self.firstTimeDone = False
        # Initialize variables
        self.sigmaProt = self.sigma.get()
        if self.sigmaProt == -1:
            self.sigmaProt = self.inputParticles.get().getDimensions()[0] / 3

        self.sampling = self.inputParticles.get().getSamplingRate()
        self.acquisition = self.inputParticles.get().getAcquisition()
        resolution = self.resolution.get()
        if resolution < 2 * self.sampling:
            resolution = (2 * self.sampling) + 0.5
        self.resolutionPca = resolution

        if self.mode == self.UPDATE_CLASSES:
            self.numberClasses = len(self.initialClasses.get())
            self.classificationBatch.set(BATCH_UPDATE)
        else:
            self.numberClasses = self.numberOfClasses.get()

        # Initialize files
        self._initFnStep()

    def _initFnStep(self):
        updateEnviron(self.gpuList.get())
        self.inputFn = self.inputParticles.get().getFileName()
        self.imgsPcaXmd = self._getExtraPath('images_pca.xmd')
        self.imgsPcaXmdOut = self._getTmpPath('images_pca.xmd')  # Wiener
        self.imgsPcaFn = self._getTmpPath('images_pca.mrc')
        self.imgsOrigXmd = self._getExtraPath('imagesInput_.xmd')
        self.imgsXmd = self._getTmpPath('images_.xmd')  # Wiener
        self.imgsFn = self._getTmpPath('images_.mrc')
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getExtraPath('classes.mrcs')


    def _insertProcessingSteps(self, newIds):
        deps = []

        inputParticles = self._loadInputSet(self.inputFn)

        if not self.staticRun:
            newParticlesSet = self._loadEmptyParticleSet()  # Esto hay que moverlo fuera y pasarle como atributo para streaming
            for partId in newIds:
                particle = inputParticles.getItem("id", partId).clone()
                newParticlesSet.append(particle)
        else:
            newParticlesSet = inputParticles

        # ------------------------------------ PCA TRAINING ----------------------------------------------
        if self._doPcaTraining(newIds):
            self.pcaStep = self._insertPCASteps(newParticlesSet)
            # fDeps.append(self.pcaStep) We dont need to store it as it is a dependecy of the other
        # ------------------------------------ CLASSIFICATION ----------------------------------------------
        if self._doClassification(newIds):
            print('DO classification steeeeeep')
            classifyStep = self._insertClassificationSteps(newParticlesSet)
            self.insertedIds.extend(newIds)
            deps.append(classifyStep)

        return deps

    def _insertPCASteps(self, newParticlesSet):
        pcaStep = self._insertFunctionStep(self.runPCASteps, newParticlesSet,
                                                        prerequisites=[], needsGPU=True)
        return pcaStep

    def runPCASteps(self, newParticlesSet):
        # Run PCA steps
        # self.pcaLaunch = True
        if self.correctCtf:
            self.convertInputStep(newParticlesSet, self.imgsPcaXmd, self.imgsPcaXmdOut)  # Wiener filter
        else:
            self.convertInputStep(newParticlesSet, self.imgsPcaXmd, self.imgsPcaFn)
        numTrain = min(len(newParticlesSet), self.training.get())
        self.pcaTraining(self.imgsPcaFn, self.resolutionPca, numTrain)
        # self.pcaLaunch = False
        self._setPcaDone()


    def _insertClassificationSteps(self, newParticlesSet):
        self._updateFnClassification()
        classStep = self._insertFunctionStep(self.runClassificationSteps,
                                             newParticlesSet, prerequisites=self.pcaStep, needsGPU=True)
        #updateStep = self._insertFunctionStep(self.updateOutputSetOfClasses,
        #                                      lastCreationTime, Set.STREAM_OPEN, prerequisites=classStep,
        #                                      needsGPU=False)
        #self.newDeps.append(updateStep)
        return classStep

    def runClassificationSteps(self, newParticlesSet):
        self.classificationLaunch = True
        if self.correctCtf:
            self.convertInputStep(newParticlesSet, self.imgsOrigXmd, self.imgsXmd) # Wiener filter
        else:
            self.convertInputStep(newParticlesSet, self.imgsOrigXmd, self.imgsFn)

        self.classification(self.imgsFn, self.numberClasses,
                            self.imgsOrigXmd, self.mask.get(), self.sigmaProt)
        self.classificationLaunch = False

    def convertInputStep(self, input, outputOrig, outputMRC):
        if self._isPcaDone() and self.staticRun:  # Static run: reuse files to avoid doing this twice
            print('static run motherfucker')
            self.imgsOrigXmd = self.imgsPcaXmd
            self.imgsFn = self.imgsPcaFn
        else:
            writeSetOfParticles(input, outputOrig)

            if self.correctCtf:
                # ------------ WIENER -----------------------
                args = (' -i %s -o %s --pixel_size %s --spherical_aberration %s '
                       '--voltage %s --q0 %s --batch 512 --padding 2 --device cuda:%d') % \
                      (outputOrig, outputMRC, self.sampling, self.acquisition.getSphericalAberration(),
                       self.acquisition.getVoltage(), self.acquisition.getAmplitudeContrast(), int(self.gpuList.get()))
                env = self.getCondaEnv()
                env = self._setEnvVariables(env)
                self.runJob("xmipp_swiftalign_wiener_2d", args, numberOfMpi=1, env=env)
            else:
                args = ' -i  %s -o %s  ' % (outputOrig, outputMRC)
                self.runJob("xmipp_image_convert", args, numberOfMpi=1)

        # For classification update
        if self.mode == self.UPDATE_CLASSES and not self.firstTimeDone:
            initial2DClasses = self.initialClasses.get()
            self.info('Write classes')
            if isinstance(initial2DClasses, SetOfClasses2D):
                writeSetOfClasses2D(initial2DClasses,
                                    self.refXmd, writeParticles=False)
            else:
                writeSetOfParticles(initial2DClasses,
                                    self.refXmd,
                                    alignType=ALIGN_NONE)

            args = ' -i  %s -o %s  ' % (self.refXmd, self.ref)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)
            self.firstTimeDone = True

    def pcaTraining(self, inputIm, resolutionTrain, numTrain):
        args = ' -i %s  -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca  --batchPCA' % \
               (inputIm, self.sampling, resolutionTrain, self.coef.get(), numTrain, self._getExtraPath())

        env = self.getCondaEnv()
        env = self._setEnvVariables(env)
        self.runJob("xmipp_classify_pca_train", args, numberOfMpi=1, env=env)

    def classification(self, inputIm, numClass, stfile, mask, sigma):
        args = ' -i %s -c %s -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt -o %s/classes -stExp %s' % \
               (inputIm, numClass, self._getExtraPath(), self._getExtraPath(), self._getExtraPath(),
                stfile)
        if mask:
            args += ' --mask --sigma %s ' % (sigma)

        if self.mode == self.UPDATE_CLASSES or self._isClassificationDone():
            args += ' -r %s ' % self.ref

        env = self.getCondaEnv()
        env = self._setEnvVariables(env)
        self.runJob("xmipp_classify_pca", args, numberOfMpi=1, env=env)

    def updateOutputSetOfClasses(self, lastCreationTime, streamMode):
        outputName = OUTPUT_CLASSES
        outputClasses, update = self._loadOutputSet(outputName)

        self._fillClassesFromLevel(outputClasses, update)
        self._updateOutputSet(outputName, outputClasses, streamMode)
        self._updateOutputAverages(update)

        if not update:  # First time
            self._defineSourceRelation(self._getInputPointer(), outputClasses)
            self._setClassificationDone()
            self.numberClasses = len(outputClasses)  # In case the original number of classes is not reached

        self.lastCreationTimeProcessed = lastCreationTime
        self.info(r'Last creation time processed UPDATED is %s' % str(self.lastCreationTimeProcessed))
        self._writeLastDone(str(self.lastCreationTimeProcessed))
        self._writeLastClassificationRound(self.classificationRound)

        self.info(r'Last classification round processed is %d' % self.classificationRound)
        self.classificationRound += 1

    def _updateOutputAverages(self, update):
        outRefs = self._loadOutputAverageSet()
        readSetOfParticles(self.ref, outRefs)
        self._updateOutputSet(OUTPUT_AVERAGES, outRefs, Set.STREAM_CLOSED)
        if not update:  # First Time
            self._defineSourceRelation(self._getInputPointer(), outRefs)

    def closeOutputStep(self):
        self._closeOutputSet()

    # --------------------------- UTILS functions -----------------------------
    def _loadInputParticleSet(self):
        """ Returns te input set of particles"""
        self.debug("Loading input db: %s" % self.inputFn)
        partSet = SetOfParticles(filename=self.inputFn)
        partSet.loadAllProperties()

        return partSet

    def _getInputPointer(self):
        return self.inputParticles

    def _loadEmptyParticleSet(self):
        partSet = SetOfParticles(filename=self.inputFn)
        partSet.loadAllProperties()
        copyPartSet = self._createSetOfParticles()
        copyPartSet.copyInfo(partSet)

        return copyPartSet

    def _setEnvVariables(self, env):
        """ Method to set all the environment variables needed to run PCA program """
        env['LD_LIBRARY_PATH'] = ''
        # Limit the number of threads
        env['OMP_NUM_THREADS'] = '12'
        env['MKL_NUM_THREADS'] = '12'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        return env

    def _updateFnClassification(self):
        """Update input based on the iteration it is"""
        self.imgsOrigXmd = updateFileName(self.imgsOrigXmd, self.classificationRound)
        self.imgsXmd = updateFileName(self.imgsXmd, self.classificationRound)
        self.imgsFn = updateFileName(self.imgsFn, self.classificationRound)
        self.info('Starts classification round: %d' % self.classificationRound)

    def _newParticlesToProcess(self):
        particlesFile = self.inputFn
        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(os.path.getmtime(particlesFile))
        self.debug('Last check: %s, modification: %s'
                   % (self.lastCheck,
                      prettyTime(mTime)))
        # If the input have not changed since our last check,
        # it does not make sense to check for new input data
        if (self.lastCheck > mTime and self.lastCreationTime) and not self.lastRound:
            newParticlesBool = False
        else:
            newParticlesBool = True

        self.lastCheck = now
        return newParticlesBool

    def _fillClassesFromLevel(self, clsSet, update=False):
        """ Create the SetOfClasses2D from a given iteration. """
        self._createModelFile()
        self._loadClassesInfo(self._getExtraPath('classes_contrast_classes.star'))
        mdIter = emtable.Table.iterRows('particles@' + self._getExtraPath('classes_images.star'))

        params = {}
        if update:
            self.info(r'Last creation time processed is %s' % str(self.lastCreationTimeProcessed))
            params = {"where": 'creation>"' + str(self.lastCreationTimeProcessed) + '"'}

        with self._lock:
            clsSet.classifyItems(updateItemCallback=self._updateParticle,
                                 updateClassCallback=self._updateClass,
                                 itemDataIterator=mdIter,  # relion style
                                 iterParams=params,
                                 doClone=False,  # So the creation time is maintained
                                 raiseOnNextFailure=False)  # So streaming can happen

    def _loadOutputSet(self, outputName):
        """
        Load the output set if it exists or create a new one.
        """
        outputSet = getattr(self, outputName, None)
        update = False
        if outputSet is None:
            outputSet = self._createSetOfClasses2D(self._getInputPointer())
            outputSet.setStreamState(Set.STREAM_OPEN)
        else:
            update = True

        return outputSet, update

    def _loadOutputAverageSet(self):
        """
        Load an empty output setOfAverages
        """
        outputRefs = self._createSetOfAverages()  # We need to create always an empty set since we need to rebuild it
        partSet = SetOfParticles(filename=self.inputFn)
        partSet.loadAllProperties()
        outputRefs.copyInfo(partSet)
        outputRefs.setSamplingRate(self.sampling)
        outputRefs.setAlignment(ALIGN_2D)

        return outputRefs

    def _doPcaTraining(self, newParticlesIds):
        """ Two cases for launching PCA steps:
            - If there are enough particles and PCA has not been launched or finished.
            - Launch if it's the last round of new particles and PCA has not been launched or finished. """
        return ((len(newParticlesIds) >= self.training.get() or self.isStreamClosed)
                and (not self._isPcaDone() and not hasattr(self, "pcaStep")))
            #(self.lastRound and not self._isPcaDone() and not

    def _doClassification(self, newParticlesIds):
        """ Three cases for launching Classification
            - First round of classification: enough particles and PCA done
            - Update classification: classification done, PCA done and enough batch size
            - First or update classification with a smaller batch: last round of particles and not classification launch
        """
        #return (len(newParticlesSet) >= self.classificationBatch.get() and self._isPcaDone()
        #        and not self._isClassificationDone() and not self.classificationLaunch) \
        #    or (len(newParticlesSet) >= BATCH_UPDATE and self._isClassificationDone() and self._isPcaDone()
        #        and not self.classificationLaunch) \
        #    or (self.lastRound and not self.classificationLaunch)
        return (((len(newParticlesIds) >= self.classificationBatch.get() and not self._isClassificationDone())
                or (len(newParticlesIds) >= BATCH_UPDATE and self._isClassificationDone())
                 and (self._isPcaDone() and not self.classificationLaunch)) or self.staticRun and not self.classificationLaunch)

    def _isPcaDone(self):
        done = False
        if os.path.exists(self._getExtraPath(PCA_FILE)):
            done = True
        return done

    def _setPcaDone(self):
        with open(self._getExtraPath(PCA_FILE), "w") as file:
            file.write('%d' % self.pcaStep)

    def _getPcaStep(self):
        " If continue then use this to collect the code for the PCA step"
        with open(self._getExtraPath(PCA_FILE), "r") as file:
            content = file.read()
            return int(content)

    def _isClassificationDone(self):
        done = False
        if os.path.exists(self._getExtraPath(CLASSIFICATION_FILE)):
            done = True

        if self.mode == self.UPDATE_CLASSES:
            done = True

        return done

    def _setClassificationDone(self):
        with open(self._getExtraPath(CLASSIFICATION_FILE), "w"):
            self.debug("Creating Classification DONE file")

    def _writeLastClassificationRound(self, classificationRound):
        with open(self._getExtraPath(CLASSIFICATION_FILE), "w") as file:
            file.write('%d' % classificationRound)

    def _getLastClassificationRound(self):
        with open(self._getExtraPath(CLASSIFICATION_FILE), "r") as file:
            content = file.read()
            return int(content)

    def _writeLastDone(self, creationTime):
        """ Write to a text file the last item creation time done. """
        with open(self._getExtraPath(LAST_DONE_FILE), 'w') as file:
            file.write('%s' % creationTime)

    def _getLastDone(self):
        # Open the file in read mode and read the number
        with open(self._getExtraPath(LAST_DONE_FILE), "r") as file:
            content = file.read()
        return str(content)

    def _updateVarsToContinue(self):
        """ Method to if needed and the protocol is set to continue then it will see in which state it was stopped """

        self.pcaStep = []

        if self._isClassificationDone():
            self.lastCreationTime = self._getLastDone()
            self.classificationRound = self._getLastClassificationRound() + 1  # Since this is the last processed
        else:
            self.lastCreationTime = ''
            self.classificationRound = 0

        self.lastCreationTimeProcessed = self.lastCreationTime
        # Convert the string to a datetime object
        self.lastCheck = datetime.strptime(self.lastCreationTime, '%Y-%m-%d %H:%M:%S')

# --------------------------- Static functions --------------------------------
def updateFileName(filepath, round):
    filename = os.path.basename(filepath)
    newFilename = f"{filename[:filename.find('_')]}_{round}{filename[filename.rfind('.'):]}"
    return os.path.join(os.path.dirname(filepath), newFilename)
