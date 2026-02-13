# ******************************************************************************
# *
# * Authors: Erney Ramirez Aportela (eramirez@cnb.csic.es)
# *          Daniel Marchan Torres (da.marchan@cnb.csic.es)
# *          Yunior C. Fonseca Reyna (cfonseca@cnb.csic.es)
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
import enum
import sys
import emtable
import os
from datetime import datetime
import time
import numpy as np

from pwem.protocols import ProtClassify2D
from pyworkflow.utils import prettyTime
from pyworkflow import VERSION_3_0
from pyworkflow.object import Set
from pyworkflow.protocol.params import IntParam, StringParam, PointerParam, EnumParam, BooleanParam, FloatParam
from pyworkflow.protocol import ProtStreamingBase, STEPS_PARALLEL, GPU_LIST, LEVEL_ADVANCED
from pyworkflow.constants import BETA

from pwem.objects import SetOfClasses2D, SetOfAverages, SetOfParticles, Transform
from pwem.constants import ALIGN_NONE, ALIGN_2D, ALIGN_PROJ, ALIGN_3D
from xmipp3 import XmippProtocol

from xmipp3.convert import (readSetOfParticles, writeSetOfParticles,
                            writeSetOfClasses2D, xmippToLocation, matrixFromGeometry)

OUTPUT_CLASSES = "outputClasses"
OUTPUT_AVERAGES = "outputAverages"
PCA_FILE = "pca_done.txt"
CLASSIFICATION_FILE = "classification_done.txt"
LAST_DONE_FILE = "last_done.txt"


class XMIPPCOLUMNS(enum.Enum):
    # PARTICLES CONSTANTS
    ctfVoltage = "ctfVoltage"  # 1
    ctfDefocusU = "ctfDefocusU"  # 2
    ctfDefocusV = "ctfDefocusV"  # 3
    ctfDefocusAngle = "ctfDefocusAngle"  # 4
    ctfSphericalAberration = "ctfSphericalAberration"  # 5
    ctfQ0 = "ctfQ0"  # 6
    ctfCritMaxFreq = "ctfCritMaxFreq"  # 7
    ctfCritFitting = "ctfCritFitting"  # 8
    enabled = "enabled"  # 9
    image = "image"  # 10
    itemId = "itemId"  # 11
    micrograph = "micrograph"  # 12
    micrographId = "micrographId"  # 13
    scoreByVariance = "scoreByVariance"  # 14
    scoreByGiniCoeff = "scoreByGiniCoeff"  # 15
    xcoor = "xcoor"  # 16
    ycoor = "ycoor"  # 17
    ref = "ref"  # 18
    anglePsi = "anglePsi"  # 19
    angleRot = "angleRot"  # 20
    angleTilt = "angleTilt"  # 21
    shiftX = "shiftX"  # 22
    shiftY = "shiftY"  # 23
    shiftZ = "shiftZ"  # 24
    flip = "flip"

    # CLASSES CONSTANTS
    classCount = "classCount"  # 3


ALIGNMENT_DICT = {"shiftX": XMIPPCOLUMNS.shiftX.value,
                  "shiftY": XMIPPCOLUMNS.shiftY.value,
                  "shiftZ": XMIPPCOLUMNS.shiftZ.value,
                  "flip": XMIPPCOLUMNS.flip.value,
                  "anglePsi": XMIPPCOLUMNS.anglePsi.value,
                  "angleRot": XMIPPCOLUMNS.angleRot.value,
                  "angleTilt": XMIPPCOLUMNS.angleTilt.value
                  }


# def updateEnviron(gpuNum):
#     """ Create the needed environment for pytorch programs. """
#     print("updating environ to select gpu %s" % (gpuNum))
#     if gpuNum == '':
#         os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     else:
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)

CONTRAST_AVERAGES_FILE = 'classes_classes.star'
AVERAGES_IMAGES_FILE = 'classes_images.star'


class XmippProtClassifyPcaStreaming(ProtStreamingBase, ProtClassify2D, XmippProtocol):
    """ Performs a 2D classification of particles using PCA. This method is optimized to run in streaming,
        enabling efficient processing of large datasets.  """

    _label = 'alignPCA-2D'
    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_pyTorch'
    _devStatus = BETA

    # Mode
    CREATE_CLASSES = 0
    UPDATE_CLASSES = 1

    _possibleOutputs = {OUTPUT_CLASSES: SetOfClasses2D,
                        OUTPUT_AVERAGES: SetOfAverages}

    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)

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
        form.addParam('mode', EnumParam, choices=['create_classes', 'update_classes'],
                      label="Create or update 2D classes?", default=self.CREATE_CLASSES,
                      display=EnumParam.DISPLAY_HLIST,
                      help='This option allows for the classification '
                           'or simply alignment of particles into previously created classes.')
        form.addParam('numberOfClasses', IntParam, default=50,
                      condition="not mode",
                      label='Number of classes:',
                      help='Number of classes (or references) to be generated.')
        form.addParam('initialClasses', PointerParam,
                      label="Initial classes",
                      condition="mode",
                      pointerClass='SetOfClasses2D, SetOfAverages',
                      help='Set of initial classes to start the classification')
        form.addParam('correctCtf', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
                      label='Correct CTF?',
                      help='If you set to *Yes*, the CTF of the experimental particles will be corrected')
        form.addParam('mask', BooleanParam, default=True, expertLevel=LEVEL_ADVANCED,
                      label='Use Gaussian Mask?',
                      help='If you set to *Yes*, a gaussian mask is applied to the images.')
        form.addParam('sigma', IntParam, default=-1, expertLevel=LEVEL_ADVANCED,
                      label='sigma:', condition="mask",
                      help='Sigma is the parameter that controls the dispersion or "width" of the curve.'
                           ' If the parameter is set to -1, sigma = dim/3.')

        form.addSection(label='Pca training')
        form.addParam('resolution', FloatParam, label="max resolution", default=8,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef', FloatParam, label="% variance", default=0.85, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of variance to determine the number of PCA components (between 0-1).'
                           ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('training', IntParam, default=100000, expertLevel=LEVEL_ADVANCED,
                      label="particles for training",
                      help='Number of particles for PCA training')
        form.addSection(label='Classification')
        form.addParam('classificationBatch', IntParam, default=75000,
                      label="particles for initial classification in streaming",
                      help='Number of particles for an initial classification to compute the 2D references in streaming')

        form.addSection(label='Compute')
        form.addParam('classificationMPIs', IntParam, default=4,
                      label="MPIs",
                      help= 'MPI is used to parallelize CTF correction.' 
                            ' The CTF is corrected using the xmipp_wiener_2d function.' 
                            ' If multiple processors are available, it is recommended'
                            ' to set this value as high as possible (e.g., 24, 32...).')

        form.addParallelSection(threads=3, mpi=1)

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        self._initialStep()
        self.newDeps = []
        newParticlesSet = self._loadEmptyParticleSet()

        if self.isContinued() and False:
            self.info('Continue protocol')
            self._updateVarsToContinue()

        checkInterval = 5

        while not self.finish:
            particlesSet = self._loadInputParticleSet()
            self.streamState = particlesSet.getStreamState()

            where = None
            if self.lastCreationTime:
                where = 'creation>"' + str(self.lastCreationTime) + '"'
            tmp = None
            newCount = 0

            for particle in particlesSet.iterItems(orderBy='creation', direction='ASC', where=where):
                tmp = particle.getObjCreation()
                newParticlesSet.append(particle.clone())
                newCount += 1

            particlesSet.close()

            if tmp is not None:
                self.lastCreationTime = tmp

            if newCount == 0:
                self.info('No new particles')
            else:
                self.info('%d new particles in batch' % len(newParticlesSet))
                self.info('Last creation time REGISTER %s' % self.lastCreationTime)

                if self._doClassification(newParticlesSet):
                    self._insertClassificationSteps(newParticlesSet, self.lastCreationTime)
                    newParticlesSet = self._loadEmptyParticleSet()

            if self.streamState == Set.STREAM_CLOSED:
                self.info('Stream closed')
                if len(newParticlesSet):
                    self.info('Finish processing with last batch %d' % len(newParticlesSet))
                    self.lastRound = True

                    # Force one more iteration to flush last batch
                    if self._doClassification(newParticlesSet):
                        self._insertClassificationSteps(newParticlesSet, self.lastCreationTime)
                        newParticlesSet = self._loadEmptyParticleSet()
                else:
                    self._insertFunctionStep(self.closeOutputStep,
                                             prerequisites=self.newDeps,
                                             needsGPU=False)
                    self.finish = True
                    continue

            with self._lock:  # Add this lock so it will not block the iterItems of the classify method
                self.inputParticles.get().close()  # If this is not close then it blocks the input protocol
            time.sleep(checkInterval)
            sys.stdout.flush()

        sys.stdout.flush()

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
        self.imgsOrigXmd = self._getExtraPath('imagesInput_.xmd')
        self.imgsXmd = self._getTmpPath('images_.xmd')  # Wiener
        self.imgsFn = self._getTmpPath('images_.mrc') # Wiener
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getExtraPath('classes.mrcs')
        self.sigmaProt = self.sigma.get()
        if self.sigmaProt == -1:
            self.sigmaProt = self.inputParticles.get().getDimensions()[0] / 3

        self.sampling = self.inputParticles.get().getSamplingRate()
        resolution = self.resolution.get()
        if resolution < 2 * self.sampling:
            resolution = (2 * self.sampling) + 0.5
        self.resolutionPca = resolution

        if self.mode.get() == self.UPDATE_CLASSES:
            self.numberClasses = len(self.initialClasses.get())
        else:
            self.numberClasses = self.numberOfClasses.get()

    def getGpusList(self, separator):
        strGpus = ""
        for elem in self._stepsExecutor.getGpuList():
            strGpus = strGpus + str(elem) + separator
        return strGpus[:-1]

    def setGPU(self, oneGPU=False):
        print("aqui va la gpu")
        print(self.getGpusList)
        exit()
        if oneGPU:
            gpus = self.getGpusList(",")[0]
        else:
            gpus = self.getGpusList(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.info(f'Visible GPUS: {gpus}')
        return gpus

    def _insertClassificationSteps(self, newParticlesSet, lastCreationTime):
        self._updateFnClassification()
        classStep = self._insertFunctionStep(self.runClassificationSteps,
                                             newParticlesSet,
                                             prerequisites=[],
                                             needsGPU=True)
        updateStep = self._insertFunctionStep(self.updateOutputSetOfClasses,
                                              lastCreationTime, Set.STREAM_OPEN, prerequisites=classStep,
                                              needsGPU=False)
        self.newDeps.append(updateStep)

    def runClassificationSteps(self, newParticlesSet):
        
        self.convertInputStep(newParticlesSet, self.imgsOrigXmd, self.imgsFn)
        
        numTrain = min(len(newParticlesSet), self.training.get())
        self.classification(self.imgsFn, self.numberClasses, self.imgsOrigXmd,
                            self.mask.get(), self.sigmaProt, numTrain, self.resolutionPca)
        # self.classificationLaunch = False

    def convertInputStep(self, input, outputOrig, outputMRC):
        writeSetOfParticles(input, outputOrig)

        if self.correctCtf.get():
            # ------------ WIENER -----------------------
            args = ' -i %s  --operate  randomize'%(outputOrig)
            self.runJob("xmipp_metadata_utilities", args, numberOfMpi=1)
            
            args = ' -i  %s -o %s --sampling_rate %s '%(outputOrig, outputMRC, self.sampling)
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.classificationMPIs.get())
            
        else:
            args = ' -i  %s -o %s  ' % (outputOrig, outputMRC)
            self.runJob("xmipp_image_convert", args)

        # For classification update
        if self.mode.get() == self.UPDATE_CLASSES and not self.firstTimeDone:
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
            self.runJob("xmipp_image_convert", args)
            self.firstTimeDone = True

    def classification(self, inputIm, numClass, stfile, mask, sigma, numTrain, resolutionTrain):
        args = ' -i %s -s %s -c %s -t %s -hr %s -p %s  -o %s/classes -stExp %s' % \
               (inputIm, self.sampling, numClass, numTrain, resolutionTrain, self.coef.get(), self._getExtraPath(),
                stfile)
        if mask:
            args += ' --mask --sigma %s ' % (sigma)

        if self.mode.get() == self.UPDATE_CLASSES or self._isClassificationDone():
            args += ' -r %s ' % self.ref

        env = self.getCondaEnv()
        env = self._setEnvVariables(env)
        self.runJob("xmipp_alignPCA_2D", args, env=env)
        
        args = ' -i %s  --operate  sort itemId'%(self._getExtraPath(AVERAGES_IMAGES_FILE))
        self.runJob("xmipp_metadata_utilities", args, numberOfMpi=1)

    def updateOutputSetOfClasses(self, lastCreationTime, streamMode):
        outputName = OUTPUT_CLASSES
        outputClasses, update = self._loadOutputSet(outputName)

        self._fillClassesFromLevel(outputClasses, update)
        self._updateOutputSet(outputName, outputClasses, streamMode)
        # self._updateOutputAverages(update)

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
        self.classificationRound += 1

    def _newParticlesToProcess(self):
        particlesFile = self.inputFn
        now = datetime.now()

        lastCheck = getattr(self, "lastCheck", now)
        self.lastCheck = lastCheck

        mTime = datetime.fromtimestamp(os.path.getmtime(particlesFile))
        self.debug("Last check: %s, modification: %s"
                   % (lastCheck, prettyTime(mTime)))

        fileUnchanged = lastCheck > mTime
        alreadyProcessedSomething = bool(getattr(self, "lastCreationTime", None))
        isLastRound = bool(getattr(self, "lastRound", False))

        hasNewParticles = not (fileUnchanged and alreadyProcessedSomething and not isLastRound)

        self.lastCheck = now
        return hasNewParticles

    def _fillClassesFromLevel(self, clsSet, update=False):
        """ Create the SetOfClasses2D from a given iteration. """
        self._createModelFile()
        
        self._loadClassesInfo(self._getExtraPath(CONTRAST_AVERAGES_FILE))
        mdIter = emtable.Table.iterRows('particles@' + self._getExtraPath(AVERAGES_IMAGES_FILE))

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

    def _doClassification(self, newParticlesSet):
        """ Three cases for launching Classification
            - First round of classification: enough particles and PCA done
            - Update classification: classification done, PCA done and enough batch size
            - First or update classification with a smaller batch: last round of particles and not classification launch
        """
        n = len(newParticlesSet)
        b = self.classificationBatch.get()
        if self.classificationLaunch:
            return False

        batchReady = n >= b
        lastRoundReady = self.lastRound

        return batchReady or lastRoundReady


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

        if self._isClassificationDone():
            self.lastCreationTime = self._getLastDone()
            self.classificationRound = self._getLastClassificationRound() + 1  # Since this is the last processed
        else:
            self.lastCreationTime = ''
            self.classificationRound = 1

        self.lastCreationTimeProcessed = self.lastCreationTime
        # Convert the string to a datetime object
        self.lastCheck = datetime.strptime(self.lastCreationTime, '%Y-%m-%d %H:%M:%S')

    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are no errors. If some errors are found, a list with
        the error messages will be returned.
        """
        errors = []
        if self.inputParticles.get().getDimensions()[0] > 256:
            errors.append("You should resize the particles."
                          " Sizes smaller than 128 pixels are recommended.")
        er = self.validateDLtoolkit()
        if not isinstance(er, list):
            er = [er]
        if er:
            errors+=er
        return errors
    
    def _warnings(self):
        validateMsgs = []
        if self.inputParticles.get().getDimensions()[0] > 128:
            validateMsgs.append("Particle sizes equal to or less"
                                " than 128 pixels are recommended.")
        if self.inputParticles.get().getDimensions()[0] > 256:
            validateMsgs.append("Particle sizes bigger than 256 may"
                                " saturate the GPU memory.")
        return validateMsgs
    
    def _summary(self):
        summary = []

        if not hasattr(self, 'outputClasses'):
            summary.append("Output classes not ready yet.")
        else:
            summary.append('2D clasification using AlignPCA')

        return summary

    # #--------------------------- UTILS functions -------------------------------
    # EMTABLE IMPLEMENTATION
    def _updateParticle(self, item, row):
        if row is None:
            self.info('Row is none finish updating particle')
            setattr(item, "_appendItem", False)
        else:
            if item.getObjId() == row.get(XMIPPCOLUMNS.itemId.value):
                item.setClassId(row.get(XMIPPCOLUMNS.ref.value))
                item.setTransform(rowToAlignmentEmtable(row, ALIGN_2D))
            else:
                self.error('The particles ids are not synchronized')
                setattr(item, "_appendItem", False)

    def _updateClass(self, item):
        classId = item.getObjId()
        if classId in self._classesInfo:
            index, fn, _ = self._classesInfo[classId]
            item.setAlignment2D()
            rep = item.getRepresentative()
            rep.setLocation(index, fn)
            rep.setSamplingRate(self.inputParticles.get().getSamplingRate())

    def _createModelFile(self):
        with open(self._getExtraPath(CONTRAST_AVERAGES_FILE), 'r') as file:
            # Read the lines of the file
            lines = file.readlines()
        # Open the file for writing
        with open(self._getExtraPath(CONTRAST_AVERAGES_FILE), "w") as file:
            # Iterate through the lines
            for line in lines:
                # Replace "data_" with "data_particles" if found
                modifiedLine = line.replace("data_", "data_particles")
                # Write the modified line to the file
                file.write(modifiedLine)

        with open(self._getExtraPath(AVERAGES_IMAGES_FILE), 'r') as file:
            lines = file.readlines()

            # Find the index of the last non-empty line
        lastNonEmptyInd = len(lines) - 1
        while lastNonEmptyInd >= 0 and lines[lastNonEmptyInd].strip() == "":
            lastNonEmptyInd -= 1

        # Modify the lines
        modifiedLines = []
        for line in lines[:lastNonEmptyInd + 1]:
            # Replace "data_" with "data_particles" if found
            # modifiedLine = line.replace("data_Particles", "data_particles")
            modifiedLine = line.replace("data_noname", "data_particles")
            modifiedLines.append(modifiedLine)

        # Write the modified lines back to the file
        with open(self._getExtraPath(AVERAGES_IMAGES_FILE), "w") as file:
            file.writelines(modifiedLines)

    def _loadClassesInfo(self, filename):
        """ Read some information about the produced 2D classes
        from the metadata file.
        """
        self._classesInfo = {}  # store classes info, indexed by class id

        mdFileName = '%s@%s' % ('particles', filename)
        table = emtable.Table(fileName=filename)

        for classNumber, row in enumerate(table.iterRows(mdFileName)):
            index, fn = xmippToLocation(row.get(XMIPPCOLUMNS.image.value))
            # Store info indexed by id, we need to store the row.clone() since
            # the same reference is used for iteration
            self._classesInfo[classNumber + 1] = (index, fn, row)
        self._numClass = index

# --------------------------- Static functions --------------------------------
def rowToAlignmentEmtable(alignmentRow, alignType):
    """
    is2D == True-> matrix is 2D (2D images alignment)
            otherwise matrix is 3D (3D volume alignment or projection)
    invTransform == True  -> for xmipp implies projection
    """

    is2D = alignType == ALIGN_2D
    inverseTransform = alignType == ALIGN_PROJ

    if alignmentRow.hasAnyColumn(ALIGNMENT_DICT.values()):
        alignment = Transform()
        angles = np.zeros(3)
        shifts = np.zeros(3)
        flip = alignmentRow.get(XMIPPCOLUMNS.flip.value, default=0.)

        shifts[0] = alignmentRow.get(XMIPPCOLUMNS.shiftX.value, default=0.)
        shifts[1] = alignmentRow.get(XMIPPCOLUMNS.shiftY.value, default=0.)

        if not is2D:
            angles[0] = alignmentRow.get(XMIPPCOLUMNS.angleRot.value, default=0.)
            angles[1] = alignmentRow.get(XMIPPCOLUMNS.angleTilt.value, default=0.)
            angles[2] = alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.)
            shifts[2] = alignmentRow.get(XMIPPCOLUMNS.shiftZ.value, default=0.)
            if flip:
                angles[1] = angles[1] + 180  # tilt + 180
                angles[2] = - angles[2]  # - psi, COSS: this is mirroring X
                shifts[0] = - shifts[0]  # -x
        else:
            psi = alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.)
            rot = alignmentRow.get(XMIPPCOLUMNS.angleRot.value, default=0.)
            if not np.isclose(rot, 0., atol=1e-6) and not np.isclose(psi, 0., atol=1e-6):
                print("HORROR rot and psi are different from zero in 2D case")

            angles[0] = psi + rot

        M = matrixFromGeometry(shifts, angles, inverseTransform)

        if flip:
            if alignType == ALIGN_2D:
                M[0, :2] *= -1.  # invert only the first two columns
                # keep x
                M[2, 2] = -1.  # set 3D rot
            elif alignType == ALIGN_3D:
                M[0, :3] *= -1.  # now, invert first line excluding x
                M[3, 3] *= -1.
            elif alignType == ALIGN_PROJ:
                pass

        alignment.setMatrix(M)

    else:
        alignment = None

    return alignment


# --------------------------- Static functions --------------------------------
def updateFileName(filepath, classRound):
    filename = os.path.basename(filepath)
    newFilename = f"{filename[:filename.find('_')]}_{classRound}{filename[filename.rfind('.'):]}"
    return os.path.join(os.path.dirname(filepath), newFilename)