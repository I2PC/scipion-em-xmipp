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
from pyworkflow.protocol import ProtStreamingBase, STEPS_PARALLEL
from pyworkflow.constants import BETA

from pwem.objects import SetOfClasses2D, SetOfAverages, SetOfParticles
from pwem.constants import ALIGN_NONE, ALIGN_2D

from xmipp3.convert import (readSetOfParticles, writeSetOfParticles,
                            writeSetOfClasses2D)
from xmipp3.protocols.protocol_classify_pca import XmippProtClassifyPca

OUTPUT_CLASSES = "outputClasses"
OUTPUT_AVERAGES = "outputAverages"
PCA_FILE = "pca_done.txt"
CLASSIFICATION_FILE = "classification_done.txt"
LAST_DONE_FILE = "last_done.txt"
BATCH_UPDATE = 5000


class XmippProtClassifyPcaStreaming(ProtStreamingBase, XmippProtClassifyPca):
    """ Performs a 2D classification of particles using PCA. This method is optimized to run in streaming, enabling efficient processing of large datasets.  """

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
        self.stepsExecutionMode = STEPS_PARALLEL

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form = self._defineCommonParams(form)
        form.addSection(label='Classification')
        form.addParam('classificationBatch', IntParam, default=50000,
                      condition="not mode",
                      label="particles for initial classification",
                      help='Number of particles for an initial classification to compute the 2D references')

        form.addParallelSection(threads=3)

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        """
        This step should be implemented by any streaming protocol.
        It should check its input and when ready conditions are met
        call the self._insertFunctionStep method.
        """
        self._initialStep()
        self.newDeps = []
        newParticlesSet = self._loadEmptyParticleSet()

        if self.isContinued() and self._isPcaDone():
            self.info('Continue protocol')
            self._updateVarsToContinue()

        while not self.finish:
            if not self._newParticlesToProcess():
                self.info('No new particles')
            else:
                particlesSet = self._loadInputParticleSet()
                self.streamState = particlesSet.getStreamState()
                where = None
                if self.lastCreationTime:
                    where = 'creation>"' + str(self.lastCreationTime) + '"'

                for particle in particlesSet.iterItems(orderBy='creation', direction='ASC', where=where):
                    tmp = particle.getObjCreation()
                    newParticlesSet.append(particle.clone())

                particlesSet.close()
                self.lastCreationTime = tmp
                self.info('%d new particles' % len(newParticlesSet))
                self.info('Last creation time REGISTER %s' % self.lastCreationTime)

                # ------------------------------------ PCA TRAINING ----------------------------------------------
                if self._doPcaTraining(newParticlesSet):
                    self.pcaStep = self._insertFunctionStep(self.runPCASteps, newParticlesSet,
                                                            prerequisites=[], needsGPU=True)
                    if len(newParticlesSet) == len(particlesSet) and self.streamState == Set.STREAM_CLOSED:
                        self.staticRun = True
                        self.info('Static Run')
                # ------------------------------------ CLASSIFICATION ----------------------------------------------
                if self._doClassification(newParticlesSet):
                    self._insertClassificationSteps(newParticlesSet, self.lastCreationTime)
                    newParticlesSet = self._loadEmptyParticleSet()

            if self.streamState == Set.STREAM_CLOSED:
                self.info('Stream closed')
                # Finish everything and close output sets
                if len(newParticlesSet):
                    self.info('Finish processing with last batch %d' % len(newParticlesSet))
                    self.lastRound = True
                else:
                    self._insertFunctionStep(self.closeOutputStep, prerequisites=self.newDeps, needsGPU=False)
                    self.finish = True
                    continue  # To avoid waiting

            with self._lock: # Add this lock so it will not block the iterItems of the classify method
                self.inputParticles.get().close() # If this is not close then it blocks the input protocol
            sys.stdout.flush()
            time.sleep(30)

        sys.stdout.flush()  # One last flush

    # --------------------------- STEPS functions -------------------------------
    def _initialStep(self):
        self.finish = False
        self.lastCreationTime = 0
        self.lastCreationTimeProcessed = 0
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.pcaLaunch = False
        self.classificationLaunch = False
        self.classificationRound = 0
        self.firstTimeDone = False
        self.staticRun = False
        # Initialize files
        self._initFnStep()

    def _initFnStep(self):
        self.inputFn = self.inputParticles.get().getFileName()
        self.imgsPcaXmd = self._getExtraPath('images_pca.xmd')
        self.imgsPcaXmdOut = self._getTmpPath('images_pca.xmd')  # Wiener
        self.imgsPcaFn = self._getTmpPath('images_pca.mrc')
        self.imgsOrigXmd = self._getExtraPath('imagesInput_.xmd')
        self.imgsXmd = self._getTmpPath('images_.xmd')  # Wiener
        self.imgsFn = self._getTmpPath('images_.mrc')
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getExtraPath('classes.mrcs')
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

    def getGpusList(self, separator):
        strGpus = ""
        for elem in self._stepsExecutor.getGpuList():
            strGpus = strGpus + str(elem) + separator
        return strGpus[:-1]

    def setGPU(self, oneGPU=False):
        if oneGPU:
            gpus = self.getGpusList(",")[0]
        else:
            gpus = self.getGpusList(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.info(f'Visible GPUS: {gpus}')
        return gpus


    def runPCASteps(self, newParticlesSet):
        # Run PCA steps
        self.pcaLaunch = True
        if self.correctCtf:
            self.convertInputStep(newParticlesSet, self.imgsPcaXmd, self.imgsPcaXmdOut)  # Wiener filter
        else:
            self.convertInputStep(newParticlesSet, self.imgsPcaXmd, self.imgsPcaFn)
        numTrain = min(len(newParticlesSet), self.training.get())
        self.pcaTraining(self.imgsPcaFn, self.resolutionPca, numTrain)
        self.pcaLaunch = False
        self._setPcaDone()

    def _insertClassificationSteps(self, newParticlesSet, lastCreationTime):
        self._updateFnClassification()
        classStep = self._insertFunctionStep(self.runClassificationSteps,
                                             newParticlesSet, prerequisites=self.pcaStep, needsGPU=True)
        updateStep = self._insertFunctionStep(self.updateOutputSetOfClasses,
                                              lastCreationTime, Set.STREAM_OPEN, prerequisites=classStep,
                                              needsGPU=False)
        self.newDeps.append(updateStep)

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
            self.imgsOrigXmd = self.imgsPcaXmd
            self.imgsFn = self.imgsPcaFn
        else:
            writeSetOfParticles(input, outputOrig)

            if self.correctCtf:
                # ------------ WIENER -----------------------
                args = (' -i %s -o %s --pixel_size %s --spherical_aberration %s '
                       '--voltage %s --q0 %s --batch 512 --padding 2 --device cuda:%d') % \
                      (outputOrig, outputMRC, self.sampling, self.acquisition.getSphericalAberration(),
                       self.acquisition.getVoltage(), self.acquisition.getAmplitudeContrast(), 0) # CUDA_VISIBLE_DEVICES is set then only id "0" available
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

            args = ' -i  %s -o %s ' % (self.refXmd, self.ref)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)
            self.firstTimeDone = True

    def pcaTraining(self, inputIm, resolutionTrain, numTrain):
        gpuID = self.setGPU(oneGPU=True)
        args = ' -i %s  -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca  --batchPCA -g %s' % \
               (inputIm, self.sampling, resolutionTrain, self.coef.get(), numTrain, self._getExtraPath(), gpuID)

        env = self.getCondaEnv()
        env = self._setEnvVariables(env)
        self.runJob("xmipp_classify_pca_train", args, numberOfMpi=1, env=env)

    def classification(self, inputIm, numClass, stfile, mask, sigma):
        gpuID = self.setGPU(oneGPU=True)
        args = ' -i %s -c %s -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt -o %s/classes -stExp %s -g %s' % \
               (inputIm, numClass, self._getExtraPath(), self._getExtraPath(), self._getExtraPath(),
                stfile, gpuID)
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

    def _doPcaTraining(self, newParticlesSet):
        """ Two cases for launching PCA steps:
            - If there are enough particles and PCA has not been launched or finished.
            - Launch if it's the last round of new particles and PCA has not been launched or finished. """
        return (len(newParticlesSet) >= self.training.get() and not self._isPcaDone() and not self.pcaLaunch) or \
            (self.lastRound and not self._isPcaDone() and not self.pcaLaunch)

    def _doClassification(self, newParticlesSet):
        """ Three cases for launching Classification
            - First round of classification: enough particles and PCA done
            - Update classification: classification done, PCA done and enough batch size
            - First or update classification with a smaller batch: last round of particles and not classification launch
        """
        return (len(newParticlesSet) >= self.classificationBatch.get() and self._isPcaDone()
                and not self._isClassificationDone() and not self.classificationLaunch) \
            or (len(newParticlesSet) >= BATCH_UPDATE and self._isClassificationDone() and self._isPcaDone()
                and not self.classificationLaunch) \
            or (self.lastRound and not self.classificationLaunch)

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

    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are not errors. If some errors are found, a list with
        the error messages will be returned.
        """
        error=self.validateDLtoolkit()
        return error

# --------------------------- Static functions --------------------------------
def updateFileName(filepath, round):
    filename = os.path.basename(filepath)
    newFilename = f"{filename[:filename.find('_')]}_{round}{filename[filename.rfind('.'):]}"
    return os.path.join(os.path.dirname(filepath), newFilename)
