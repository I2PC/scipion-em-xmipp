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
import enum
import os
from datetime import datetime
import time
import numpy as np
from pyworkflow.utils import prettyTime

from pyworkflow import VERSION_3_0
from pyworkflow.object import Set
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, EnumParam, IntParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol import ProtStreamingBase, STEPS_PARALLEL

from pwem.protocols import ProtClassify2D
from pwem.objects import SetOfClasses2D, Transform
from xmipp3 import XmippProtocol
from pwem.constants import ALIGN_NONE, ALIGN_PROJ, ALIGN_2D, ALIGN_3D

from xmipp3.convert import (readSetOfParticles, writeSetOfParticles,
                            writeSetOfClasses2D, xmippToLocation, matrixFromGeometry)

OUTPUT = "outputClasses"
PCA_FILE = "pca_done.txt"
CLASSIFICATION_FILE = "classification_done.txt"
LAST_DONE_FILE = "last_done.txt"
BATCH_UPDATE = 5000

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


def updateEnviron(gpuNum):
    """ Create the needed environment for pytorch programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)


class XmippProtClassifyPcaStreaming(ProtClassify2D, ProtStreamingBase, XmippProtocol):
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
                      help='This option allows for either global refinement from an initial volume '
                           ' or just alignment of particles. If the reference volume is at a high resolution, '
                           ' it is advisable to only align the particles and reconstruct at the end of the iterative process.')
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
                      help='Sigma is the parameter that controls the dispersion or "width" of the curve..')

        form.addSection(label='Pca training')

        form.addParam('resolution', FloatParam, label="max resolution", default=10,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef', FloatParam, label="% variance", default=0.5, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of coefficients to be considers (between 0-1).'
                           ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('training', IntParam, default=40000,
                      label="particles for training",
                      help='Number of particles for PCA training')

        form.addSection(label='Classification')
        form.addParam('classificationBatch', IntParam, default=20000,
                      condition="not mode",
                      label="particles for initial classification",
                      help='Number of particles for an initial classification to compute the 2D references')

        # form.addParallelSection(threads=1, mpi=4)
        form.addParallelSection(threads=3, mpi=1)

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        """
        This step should be implemented by any streaming protocol.
        It should check its input and when ready conditions are met
        call the self._insertFunctionStep method.
        """
        self._initFnStep()
        self.newDeps = []
        newParticlesSet = self._loadEmptyParticleSet()

        while not self.finish:
            if not self._newParticlesToProcess():
                self.info('No new particles')
            else:
                particlesSet = self._loadInputParticleSet()
                self.isStreamClosed = particlesSet.getStreamState()

                where = None
                if self.lastCreationTime:
                    where = 'creation>"' + str(self.lastCreationTime) + '"'

                for particle in particlesSet.iterItems(orderBy='creation', direction='ASC', where=where):
                    newParticlesSet.append(particle.clone())
                    tmp = particle.getObjCreation()

                self.lastCreationTime = tmp
                self.info(f'%d new particles' % len(newParticlesSet))
                self.info('Last creation time %s' % self.lastCreationTime)

                # ------------------------------------ PCA TRAINING ----------------------------------------------
                if self._doPcaTraining(newParticlesSet):
                    self.pcaStep = self._insertFunctionStep(self.runPCASteps, newParticlesSet, prerequisites=[])
                # ------------------------------------ CLASSIFICATION ----------------------------------------------
                if self._doClassification(newParticlesSet):
                    self._insertClassificationSteps(newParticlesSet)
                    newParticlesSet = self._loadEmptyParticleSet()

            if self.isStreamClosed == Set.STREAM_CLOSED:
                self.info('Stream closed')
                # Finish everything and close output sets
                if len(newParticlesSet):
                    self.info(f'Finish processing with last batch %d' % len(newParticlesSet))
                    self.lastRound = True
                else:
                    self._insertFunctionStep(self.closeOutputStep, prerequisites=self.newDeps)
                    self.finish = True

            sys.stdout.flush()
            time.sleep(30)

    # --------------------------- STEPS functions -------------------------------
    def _initialStep(self):
        self.finish = False
        self.lastCreationTime = 0
        self.lastCreationTimeProcessed = 0
        self.isStreamClosed = False
        self.lastRound = False
        self.pcaLaunch = False
        self.classificationLaunch = False
        self.classificationRound = 0  # todo: if continue catch this classificationRound
        self.firstTimeDone = False

    def _initFnStep(self):
        updateEnviron(self.gpuList.get())
        self.imgsPcaXmd = self._getExtraPath('images_pca.xmd')
        self.imgsPcaFn = self._getTmpPath('images_pca.mrc')
        self.imgsOrigXmd = self._getExtraPath('imagesInput_.xmd')
        self.imgsXmd = self._getTmpPath('images_.xmd')
        self.imgsFn = self._getTmpPath('images_.mrc')
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getExtraPath('classes.mrcs')
        self.sigmaProt = self.sigma.get()
        if self.sigmaProt == -1:
            self.sigmaProt = self.inputParticles.get().getDimensions()[0] / 3

        self.sampling = self.inputParticles.get().getSamplingRate()
        resolution = self.resolution.get()
        if resolution < 2 * self.sampling:
            resolution = (2 * self.sampling) + 0.5

        if self.mode == self.UPDATE_CLASSES:
            self.numberClasses = len(self.initialClasses.get())
            self.classificationBatch.set(BATCH_UPDATE)
        else:
            self.numberClasses = self.numberOfClasses.get()

        self.resolutionPca = resolution

    def runPCASteps(self, newParticlesSet):
        # Run PCA steps
        self.pcaLaunch = True
        self.convertInputStep(newParticlesSet, self.imgsPcaXmd, self.imgsPcaFn)
        numTrain = min(len(newParticlesSet), self.training.get())
        self.pcaTraining(self.imgsPcaFn, self.resolutionPca, numTrain)
        self.pcaLaunch = False
        self._setPcaDone()

    def _insertClassificationSteps(self, newParticlesSet):
        self._updateFnClassification()
        self.classStep = self._insertFunctionStep(self.runClassificationSteps,
                                                  newParticlesSet, prerequisites=self.pcaStep)
        self.updateStep = self._insertFunctionStep(self.updateOutputSetOfClasses, newParticlesSet,
                                                   Set.STREAM_OPEN, prerequisites=self.classStep)
        self.newDeps.append(self.updateStep)

    def runClassificationSteps(self, newParticlesSet):
        self.classificationLaunch = True
        self.convertInputStep(  # self.inputParticles.get(), self.imgsOrigXmd, self.imgsXmd) #wiener
            newParticlesSet, self.imgsOrigXmd, self.imgsFn)
        self.classification(self.imgsFn, self.numberClasses,
                            self.imgsOrigXmd, self.mask.get(), self.sigmaProt)
        self.classificationLaunch = False

    def convertInputStep(self, input, outputOrig, outputMRC):
        writeSetOfParticles(input, outputOrig)
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

        if self.correctCtf:
            args = ' -i  %s -o %s --sampling_rate %s ' % (outputOrig, outputMRC, self.sampling)
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get())
            # WIENER Oier
            # args = ' -i %s  -o %s --pixel_size %s --spherical_aberration %s --voltage %s --batch 1024 --device cuda:0'% \
            #         (outputOrig, outputMRC, self.sampling, self.acquisition.getSphericalAberration(), self.acquisition.getVoltage())
            # env = self.getCondaEnv()
            # env['LD_LIBRARY_PATH'] = ''
            # self.runJob("xmipp_swiftalign_wiener_2d", args, numberOfMpi=1, env=env)
        else:
            args = ' -i  %s -o %s  ' % (outputOrig, outputMRC)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)

    def pcaTraining(self, inputIm, resolutionTrain, numTrain):
        args = ' -i %s  -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca  --batchPCA' % \
               (inputIm, self.sampling, resolutionTrain, self.coef.get(), numTrain, self._getExtraPath())

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca_train", args, numberOfMpi=1, env=env)

    def classification(self, inputIm, numClass, stfile, mask, sigma):
        args = ' -i %s -s %s -c %s -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt -o %s/classes -stExp %s' % \
               (inputIm, self.sampling, numClass, self._getExtraPath(), self._getExtraPath(), self._getExtraPath(),
                stfile)
        if mask:
            args += ' --mask --sigma %s ' % (sigma)

        if self.mode == self.UPDATE_CLASSES or self._isClassificationDone():
            args += ' -r %s ' % self.ref

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca", args, numberOfMpi=1, env=env)

    def updateOutputSetOfClasses(self, particles, streamMode):
        outputName = 'outputClasses'
        outputClasses, update = self._loadOutputSet(outputName)

        self._fillClassesFromLevel(outputClasses, update)
        self._updateOutputSet(outputName, outputClasses, streamMode)

        if not update:  # First time
            self._defineSourceRelation(self._getInputPointer(), outputClasses)
            self._updateOutputAverages(update)
            self._setClassificationDone()

        self._writeLastDone(str(self.lastCreationTimeProcessed))

        for particle in particles.iterItems(orderBy='creation', direction='DESC'):
            self.lastCreationTimeProcessed = particle.getObjCreation()
            self.info(r'Last creation time processed UPDATED is %s' % str(self.lastCreationTimeProcessed))
            print(particle.getObjId())
            break

    def _updateOutputAverages(self, update):
        outputName = 'outputAverages'
        outRefs = self._loadOutputAverageSet(outputName)
        readSetOfParticles(self.ref, outRefs)
        self._updateOutputSet(outputName, outRefs, Set.STREAM_CLOSED)
        if not update:  # First Time
            self._defineSourceRelation(self._getInputPointer(), outRefs)

    def closeOutputStep(self):
        self._closeOutputSet()

    # --------------------------- UTILS functions -----------------------------
    def _loadInputParticleSet(self):
        """ Returns te input set of particles"""
        partSet = self.inputParticles.get()
        partSet.loadAllProperties()

        return partSet

    def _getInputPointer(self):
        return self.inputParticles

    def _loadEmptyParticleSet(self):
        partSet = self.inputParticles.get()
        partSet.loadAllProperties()
        self.acquisition = partSet.getAcquisition()
        copyPartSet = self._createSetOfParticles()
        copyPartSet.copyInfo(partSet)
        copyPartSet.setSamplingRate(self.sampling)
        copyPartSet.setAcquisition(self.acquisition)

        return copyPartSet

    def _updateFnClassification(self):
        """Update input based on the iteration it is"""
        self.imgsOrigXmd = updateFileName(self.imgsOrigXmd, self.classificationRound)
        self.imgsXmd = updateFileName(self.imgsXmd, self.classificationRound)
        self.imgsFn = updateFileName(self.imgsFn, self.classificationRound)
        self.info('Classification Round: %d' % self.classificationRound)
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
        if (self.lastCheck > mTime and self.lastCreationTime) and not self.lastRound:
            newParticlesBool = False
        else:
            newParticlesBool = True

        self.lastCheck = now
        return newParticlesBool

    def _createModelFile(self):
        with open(self._getExtraPath('classes_contrast_classes.star'), 'r') as file:
            # Read the lines of the file
            lines = file.readlines()
        # Open the file for writing
        with open(self._getExtraPath('classes_contrast_classes.star'), "w") as file:
            # Iterate through the lines
            for line in lines:
                # Replace "data_" with "data_particles" if found
                modified_line = line.replace("data_", "data_particles")
                # Write the modified line to the file
                file.write(modified_line)

        with open(self._getExtraPath('classes_images.star'), 'r') as file:
            lines = file.readlines()

            # Find the index of the last non-empty line
        last_non_empty_index = len(lines) - 1
        while last_non_empty_index >= 0 and lines[last_non_empty_index].strip() == "":
            last_non_empty_index -= 1

        # Modify the lines
        modified_lines = []
        for line in lines[:last_non_empty_index + 1]:
            # Replace "data_" with "data_particles" if found
            modified_line = line.replace("data_Particles", "data_particles")
            modified_lines.append(modified_line)

        # Write the modified lines back to the file
        with open(self._getExtraPath('classes_images.star'), "w") as file:
            file.writelines(modified_lines)

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

    def _updateParticle(self, item, row):
        if row is None:
            setattr(item, "_appendItem", False)
        else:
            if item.getObjId() == row.get(XMIPPCOLUMNS.itemId.value):
                item.setClassId(row.get(XMIPPCOLUMNS.ref.value))
                item.setTransform(rowToAlignment_emtable(row, ALIGN_2D))
            else:
                self.error('The particles ids are not synchronized')
                setattr(item, "_appendItem", False)

    def _updateClass(self, item):
        classId = item.getObjId()
        if classId in self._classesInfo:
            index, fn, row = self._classesInfo[classId]
            item.setAlignment2D()
            rep = item.getRepresentative()
            rep.setLocation(index, fn)
            rep.setSamplingRate(self.inputParticles.get().getSamplingRate())

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

    def _loadOutputAverageSet(self, outputName):
        outputRefs = getattr(self, outputName, None)

        if outputRefs is None:
            outputRefs = self._createSetOfAverages()
            outputRefs.copyInfo(self.inputParticles.get())
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
            - First or update classification with a smaller batch: last round of particles and not classification launch"""
        return (
                    len(newParticlesSet) >= self.classificationBatch.get() and self._isPcaDone() and not self._isClassificationDone()
                    and not self.classificationLaunch) \
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
            pass

    def _isClassificationDone(self):
        done = False
        if os.path.exists(self._getExtraPath(CLASSIFICATION_FILE)):
            done = True

        if self.mode == self.UPDATE_CLASSES:
            done = True

        return done

    def _setClassificationDone(self):
        with open(self._getExtraPath(CLASSIFICATION_FILE), "w") as file:
            pass

    def _writeLastDone(self, creationTime):
        """ Write to a text file the last item done. """
        with open(self._getExtraPath(LAST_DONE_FILE), 'w') as f:
            f.write('%s\n' % creationTime)

    def _getLastDone(self):
        """ Todo: use this function for the continue """
        # Open the file in read mode and read the number
        with open(self._getExtraPath(LAST_DONE_FILE), "r") as file:
            content = file.read()
        return str(content)

    # --------------------------- INFO functions --------------------------------
    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are not errors. If some errors are found, a list with
        the error messages will be returned.
        """

        errors = []
        if self.inputParticles.get().getDimensions()[0] > 256:
            errors.append("You should resize the particles."
                          " Sizes smaller than 128 pixels are recommended.")
        er = self.validateDLtoolkit()
        if er:
            errors.append(er)
        return errors

    def _warnings(self):
        validateMsgs = []
        if self.inputParticles.get().getDimensions()[0] > 128:
            validateMsgs.append("Particle sizes equal to or less"
                                " than 128 pixels are recommended.")
        elif self.inputParticles.get().getDimensions()[0] > 256:
            validateMsgs.append("Particle sizes equal to or less"
                                " than 128 pixels are recommended.")
        return validateMsgs


# --------------------------- Static functions --------------------------------
def updateFileName(filepath, round):
    filename = os.path.basename(filepath)
    new_filename = f"{filename[:filename.find('_')]}_{round}{filename[filename.rfind('.'):]}"
    return os.path.join(os.path.dirname(filepath), new_filename)

def rowToAlignment_emtable(alignmentRow, alignType):
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
                shifts[0] = -shifts[0]  # -x
        else:
            angles[2] = - alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.)
            psi = alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.)
            rot = alignmentRow.get(XMIPPCOLUMNS.angleRot.value, default=0.)
            if not np.isclose(rot, 0., atol=1e-6) and not np.isclose(psi, 0., atol=1e-6):
                print("HORROR rot and psi are different from zero in 2D case")
            angles[0] = alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.) \
                        + alignmentRow.get(XMIPPCOLUMNS.angleRot.value, default=0.)  # psi + rot

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
