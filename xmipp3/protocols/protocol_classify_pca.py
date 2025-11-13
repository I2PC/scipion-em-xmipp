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
import os
import emtable
import enum
import numpy as np

from pyworkflow import VERSION_3_0
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, EnumParam, IntParam, GPU_LIST)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.constants import BETA

from pwem.protocols import ProtClassify2D
from pwem.objects import SetOfClasses2D, Transform
from pwem.constants import ALIGN_NONE, ALIGN_2D, ALIGN_PROJ, ALIGN_3D

from xmipp3 import XmippProtocol
from xmipp3.convert import (writeSetOfParticles, writeSetOfClasses2D, xmippToLocation,
                            readSetOfParticles, matrixFromGeometry)


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

CONTRAST_AVERAGES_FILE = 'classes_contrast_classes.star'
AVERAGES_IMAGES_FILE = 'classes_images.star'
        
class XmippProtClassifyPca(ProtClassify2D, XmippProtocol):
    """ Classifies a set of images using Principal Component Analysis (PCA). This 2D classification groups (the number of groups can be set) are based on their similarities, assisting in the identification of different conformational states or particle populations. """
    
    _label= '2D classification pca'

    _lastUpdateVersion = VERSION_3_0
    _conda_env = 'xmipp_pyTorch'
    _devStatus = BETA
    
    # Mode
    CREATE_CLASSES = 0
    UPDATE_CLASSES = 1
    
    def __init__(self, **args):
        ProtClassify2D.__init__(self, **args)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form = self._defineCommonParams(form)
        form.addParallelSection(threads=1, mpi=8)

    def _defineCommonParams(self, form):
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
        form.addParam('coef', FloatParam, label="% variance", default=0.75, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of variance to determine the number of PCA components (between 0-1).'
                           ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('training', IntParam, default=40000,
                      label="particles for training",
                      help='Number of particles for PCA training')

        return form

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):

        """ Mainly prepare the command line for call classification program"""
        self.imgsOrigXmd = self._getExtraPath('images_original.xmd')
        self.imgsXmd = self._getTmpPath('images.xmd')
        self.imgsFn = self._getTmpPath('images.mrc')
        self.refXmd = self._getTmpPath('references.xmd')
        self.ref = self._getTmpPath('references.mrcs')
        self.sampling = self.inputParticles.get().getSamplingRate()
        self.acquisition = self.inputParticles.get().getAcquisition()
        mask = self.mask.get()
        sigma = self.sigma.get()
        if sigma == -1:
            sigma = self.inputParticles.get().getDimensions()[0]/3
        particlesTrain = min(self.training.get(), self.inputParticles.get().getSize())
        resolution = self.resolution.get()
        if resolution < 2 * self.sampling:
            resolution = (2 * self.sampling) + 0.5

    
        self._insertFunctionStep('convertInputStep', 
                                self.inputParticles.get(), self.imgsOrigXmd, self.imgsFn) #convert
        
        self._insertFunctionStep("pcaTraining", self.imgsFn, resolution, particlesTrain)
        
        self._insertFunctionStep("classification", self.imgsFn, self.numberOfClasses.get(), self.imgsOrigXmd, mask, sigma )
    
        self._insertFunctionStep('createOutputStep')


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


    #--------------------------- STEPS functions -------------------------------
    def convertInputStep(self, input, outputOrig, outputMRC):
        writeSetOfParticles(input, outputOrig)
        
        if self.mode == self.UPDATE_CLASSES: 
            
            if isinstance(self.initialClasses.get(), SetOfClasses2D):
                writeSetOfClasses2D(self.initialClasses.get(),
                                    self.refXmd, writeParticles=False)
            else:
                writeSetOfParticles(self.initialClasses.get(),
                                    self.refXmd)
                       
            args = ' -i  %s -o %s  '%(self.refXmd, self.ref)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1)
            
        
        if self.correctCtf: 
            args = ' -i  %s -o %s --sampling_rate %s '%(outputOrig, outputMRC, self.sampling)
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get())
            
        else:      
            args = ' -i  %s -o %s  '%(outputOrig, outputMRC)
            self.runJob("xmipp_image_convert", args, numberOfMpi=1) 
        
        
    def pcaTraining(self, inputIm, resolutionTrain, numTrain):
        gpuId = self.setGPU(oneGPU=True)
        args = ' -i %s  -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca  --batchPCA -g %s'% \
                (inputIm, self.sampling, resolutionTrain, self.coef.get(), numTrain, self._getExtraPath(), gpuId)

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca_train", args, numberOfMpi=1, env=env)
        
        
    def classification(self, inputIm, numClass, stfile, mask, sigma):
        gpuId = self.setGPU(oneGPU=True)
        args = ' -i %s -c %s -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt -o %s/classes -stExp %s -g %s' % \
                (inputIm, numClass, self._getExtraPath(), self._getExtraPath(),  self._getExtraPath(),
                 stfile, gpuId)
        if mask:
            args += ' --mask --sigma %s '%(sigma) 
            
        if self.mode == self.UPDATE_CLASSES:
            args += ' -r %s '%(self.ref)

        env = self.getCondaEnv()
        env['LD_LIBRARY_PATH'] = ''
        self.runJob("xmipp_classify_pca", args, numberOfMpi=1, env=env)
        
        
    def createOutputStep(self):
        """ Store the SetOfClasses2D object
        resulting from the protocol execution.
        """
    
        inputParticles = self.inputParticles#.get()

        classes2DSet = self._createSetOfClasses2D(inputParticles)
        self._fillClassesFromLevel(classes2DSet)

        self._defineOutputs(outputClasses=classes2DSet)
        self._defineSourceRelation(self.inputParticles, classes2DSet)
        
        self.createOutputAverages(classes2DSet)
        
    def createOutputAverages(self, outputClasses):
        classes = self._getExtraPath('classes.mrcs')
        outRefs = self._createSetOfAverages()
        outRefs.copyInfo(self.inputParticles.get())
        outRefs.setSamplingRate(self.sampling)
        readSetOfParticles(classes, outRefs)

        outRefs.setAlignment(ALIGN_2D)
        self._defineOutputs(outputAverages=outRefs)
        self._defineSourceRelation(self.inputParticles, outRefs)    
    
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
            summary.append('Two output sets are obtained:')
            summary.append('- The first one corresponds to the classes and should be used to select '
                           ' the particles corresponding to each class. In this case, the classes are '
                           ' displayed applying a contrast variation.')
            summary.append('- The second one corresponds solely to the representative classes (outputAverages)')
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
            modifiedLine = line.replace("data_Particles", "data_particles")
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

    def _fillClassesFromLevel(self, clsSet, update=False):
        """ Create the SetOfClasses2D from a given iteration. """
        self._createModelFile()
        self._loadClassesInfo(self._getExtraPath(CONTRAST_AVERAGES_FILE))
        # self._loadClassesInfo(self._getExtraPath('classes_classes.star'))
        mdIter = emtable.Table.iterRows('particles@' + self._getExtraPath(AVERAGES_IMAGES_FILE))

        params = {}
        if update:
            self.info(r'Last particle processed id is %s' % self.lastParticleProcessedId)
            params = {"where": "id > %s" % self.lastParticleProcessedId}

        # the particle with orientation parameters (all_parameters)
        with self._lock:
            clsSet.classifyItems(updateItemCallback=self._updateParticle,
                                 updateClassCallback=self._updateClass,
                                 itemDataIterator=mdIter,  # relion style
                                 iterParams=params)

    def _validate(self):
        """ Check if the installation of this protocol is correct.
		Can't rely on package function since this is a "multi package" package
		Returning an empty list means that the installation is correct
		and there are not errors. If some errors are found, a list with
		the error messages will be returned.
		"""
        error = self.validateDLtoolkit()
        return error
    
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
                angles[2] = - angles[2]    # - psi, COSS: this is mirroring X
                shifts[0] = - shifts[0]     # -x
        else:
            psi = alignmentRow.get(XMIPPCOLUMNS.anglePsi.value, default=0.)
            rot = alignmentRow.get(XMIPPCOLUMNS.angleRot.value, default=0.)
            if not np.isclose(rot, 0., atol=1e-6 ) and not np.isclose(psi, 0., atol=1e-6):
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
