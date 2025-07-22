# **************************************************************************
# *
# * Authors:     Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *              Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
# *              (produce residuals improved with projection subtraction)
# *              Daniel Marchan Torres (da.marchan@cnb.csic.es)
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
# **************************************************************************
from math import floor
import os
import json
import numpy as np

from pwem.protocols import ProtAnalysis3D
from pwem.objects import (SetOfClasses2D, Image, SetOfAverages, SetOfParticles,
                          SetOfVolumes, SetOfClasses3D, Volume, EMSet, Class3D)
import pwem.emlib.metadata as md
from pwem.emlib.image import ImageHandler
from pwem import emlib

from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow import UPDATED, PROD
from pyworkflow import VERSION_3_0
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam, BooleanParam, EnumParam)
from pyworkflow.utils.path import cleanPath, cleanPattern

from xmipp3.base import ProjMatcher
from xmipp3.convert import setXmippAttributes, xmippToLocation, getXmippAttribute
from ..convert import writeSetOfClasses2D, writeSetOfParticles

NEW_SAMPLING_RATE = 3.0
OUTPUTS_FN = "outputs.txt"

        
class XmippProtCompareReprojections(ProtAnalysis3D, ProjMatcher):
    """Compares a set of classes or averages with the corresponding projections of a reference volume.
    The set of images must have a 3D angular assignment and the protocol computes the residues
    (the difference between the experimental images and the reprojections). The zscore of the mean
    and variance of the residues are computed. Large values of these scores may indicate outliers.
    The protocol also analyze the covariance matrix of the residual and computes the logarithm of
    its determinant [Cherian2013]. The extremes of this score (called zScoreResCov), that is
    values particularly low or high, may indicate outliers."""

    _label = 'compare reprojections'
    _lastUpdateVersion = VERSION_3_0
    _devStatus = PROD
    _possibleOutputs = {}

    PARTICLES = 0
    VOLUME = 1
    BOTH = 2
    

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self._classesInfo = dict()

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputSet2D', PointerParam, label="Input images", important=True,
                      pointerClass='SetOfClasses2D, SetOfClasses3D, SetOfAverages, SetOfParticles')
        form.addParam('inputSet3D', PointerParam, label="Volume to compare images to", important=True,
                      pointerClass='Volume, SetOfVolumes, SetOfClasses3D',
                      help='Volume to be used for class comparison')
        form.addParam('useAssignment', BooleanParam, default=True, label='Use input angular assignment (if available)')
        form.addParam('optimizeGray', BooleanParam, default=False, label='Optimize gray scale')
        form.addParam('ignoreCTF', BooleanParam, default=True, label='Ignore CTF',
                      help='By ignoring the CTF you will create projections more similar to what a person expects, '
                           'while by using the CTF you will create projections more similar to what the microscope sees')
        form.addParam('doDownSample', BooleanParam, default=True, label='Downsample',
                      help='If accepted the input volumes and the input 2d classes will be downsample to 3 A/px. '
                           'This will help to reduce the computation time.')
        form.addParam('doEvaluateResiduals', BooleanParam, default=False, expertLevel=LEVEL_ADVANCED,
                      label='Evaluate residuals',
                      help='If this option is chosen, then the residual covariance matrix is calculated and '
                           'characterized. But this option takes time and disk space')
        form.addParam('symmetryGroup', StringParam, default="c1", label='Symmetry group',
                      help='See https://i2pc.github.io/docs/Utils/Conventions/index.html#symmetry for a description of the symmetry'
                           ' groups format. If no symmetry is present, give c1')
        form.addParam('angularSampling', FloatParam, default=5, expertLevel=LEVEL_ADVANCED,
                      label='Angular sampling rate',
                      help='In degrees.'
                      ' This sampling defines how fine the projection gallery from the volume is explored.')
        form.addParam('resol', FloatParam, label="Filter at resolution: ", default=3, expertLevel=LEVEL_ADVANCED,
                      help='Resolution (A) at which subtraction will be performed, filtering the volume projections.'
                           'Value 0 implies no filtering.')
        form.addParam('sigma', FloatParam, label="Decay of the filter (sigma): ", default=3, condition='resol',
                      help='Decay of the filter (sigma parameter) to smooth the mask transition',
                      expertLevel=LEVEL_ADVANCED)
        form.addSection(label='Output')
        form.addParam('doRanking', BooleanParam, default=True, label='Rank the set of Volumes/3D Classes',
                      help='If accepted please check that you have an input set with several 3D objects '
                           '(Classes 3D or Volumes). It will rank all the Volume 3D Class/Volume.')
        form.addParam('doExtraction', BooleanParam, default=True, label='Extract best Volume/3D Class',
                      condition='doRanking',
                      help='If accepted please check that you have an input set with several 3D objects '
                           '(Classes 3D or Volumes). It will extract the Volume and/or the Particles from '
                           'the best rated 3D Class/Volume.')
        form.addParam('extractOption', EnumParam,
                      choices=['Particles', 'Volume', 'Both'],
                      default=self.BOTH,
                      condition='doExtraction and doRanking',
                      label="Extraction option", display=EnumParam.DISPLAY_COMBO,
                      help='Select an option to extract from the 3D Classes: \n '
                           '_Particles_: Extract the set of particles from the selected class. \n '
                           '_Volume_: Extract the volume from the selected class. \n'
                           '_Both_: Extract the volume and particles from the selected class.')

        form.addParallelSection(threads=0, mpi=8)
    
    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        """ Convert input images if necessary """
        self.initialStep()
        self._insertFunctionStep(self.convertStep)

        if self.doDownSample:
            self._insertFunctionStep(self.downSampleStep, self.imgsOrigFn)

        if self._useProjMatching():
            self._insertFunctionStep(self.insertProjMatchStep, self.fnVolDict, self.angularSampling.get(),
                                     self.symmetryGroup.get(), self.anglesDict, self.galleryDict)
        else:
            tmpDict = self.anglesDict
            self.anglesDict = {volId: self.imgsFn for volId in tmpDict}

        self._insertFunctionStep(self.produceResiduals, self.anglesDict,
                                 NEW_SAMPLING_RATE if self.doDownSample else self.inputSet3D.get().getSamplingRate())

        if self.doEvaluateResiduals.get():
            self._insertFunctionStep(self.evaluateResiduals)

        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions ---------------------------------------------------
    def initialStep(self):
        self.samplingRateVol = self.inputSet3D.get().getSamplingRate()
        self.samplingRateAverages = self.inputSet2D.get().getSamplingRate()
        self.imgsOrigFn = self._getExtraPath('residuals.xmd')
        self.fnVolDict = {}
        self.anglesDict = {}
        self.galleryDict = {}
        self.outputNamesDict = {}
        volSet = self.inputSet3D.get()
        self.xDim = self._getWorkingDimensions(self.samplingRateAverages, self.samplingRateVol, self.doDownSample.get())

        if self.doDownSample and self.samplingRateAverages < NEW_SAMPLING_RATE:
            self.imgsFn = self._getExtraPath('residuals_downsample.xmd')
        else:
            self.imgsFn = self.imgsOrigFn

        if isinstance(volSet, EMSet):
            for vol in volSet.iterItems(orderBy='id', direction='ASC'):
                vid = vol.getObjId()
                self.fnVolDict[vid] = self._getTmpPath("volume_%d.vol" % vid)
                self.anglesDict[vid] = self._getExtraPath('angles_%d.xmd' % vid)
                self.galleryDict[vid] = self._getExtraPath('gallery_%d.stk' % vid)
                self.outputNamesDict[vid] = "reprojections_vol%d" % vid
        else:
            vid = volSet.getObjId()
            self.fnVolDict[vid] = self._getTmpPath("volume_%d.vol" % vid)
            self.anglesDict[vid] = self._getExtraPath('angles_%d.xmd' % vid)
            self.galleryDict[vid] = self._getExtraPath('gallery_%d.stk' % vid)
            self.outputNamesDict[vid] = "reprojections_vol%d" % vid

        with open(self._getExtraPath(OUTPUTS_FN), 'w'):
            pass  # Creates output file for visualizations warnings


    def convertStep(self):
        # Convert input images (SetOfClasses2D, SetOfAverages and SetOfParticles)
        imgSet = self.inputSet2D.get()
        imgsFn = self.imgsOrigFn
        if isinstance(imgSet, SetOfClasses2D):
            writeSetOfClasses2D(imgSet, imgsFn, writeParticles=True)
        else:
            writeSetOfParticles(imgSet, imgsFn)
        # Convert input volumes (SetOfClasses3D, SetOfVolumes and Volume)
        volSet = self.inputSet3D.get()
        for volId, fnVol in self.fnVolDict.items():
            if isinstance(volSet, Volume):
                vol = volSet
            else:
                volCl = volSet.getItem("id", volId)
                if isinstance(volCl, Class3D):
                    vol = volCl.getRepresentative().clone()
                else:
                    vol = volCl
            img = ImageHandler()
            fnVol = self.fnVolDict[volId]
            self.info("Registering volume %s" % fnVol)
            img.convert(vol, fnVol)

            # In case input volume does not match 2D references size (If downsample is activated this is not needed)
            xdimVol = vol.getDim()[0]
            xdimImg = self._getDimensionsImages()
            if xdimVol != xdimImg and not self.doDownSample:
                self.xDim = xdimImg
                self.runJob("xmipp_image_resize", "-i %s --dim %d"
                            % (fnVol, self.xDim), numberOfMpi=1)

    def downSampleStep(self, imgsOrigFn):
        # Calculate new sampling rate
        if NEW_SAMPLING_RATE < self.samplingRateVol or NEW_SAMPLING_RATE < self.samplingRateAverages:
            newSamplingRate = max(self.samplingRateVol, self.samplingRateAverages)
            self.info("The target sampling rate is smaller than the inputs sampling rate, new target sampling rate "
                      "is set to the biggest one of both inputs %f" % newSamplingRate)
        else:
            newSamplingRate = NEW_SAMPLING_RATE

        # Downsample the average 2D classes
        if self.samplingRateAverages < newSamplingRate:
            imgsFn = self._getExtraPath('residuals_downsample.mrcs')
            self.info('Downsampling the input images...')
            self.runJob("xmipp_image_resize", "-i %s -o %s --dim %d"
                        % (imgsOrigFn, imgsFn, self.xDim), numberOfMpi=1)
        else:
            self.info('The target sampling rate %f <= %f the current one, skipping downsample for the 2D Images.'
                      % (newSamplingRate, self.samplingRateAverages))

        # Downsample Volume/s
        if self.samplingRateVol < newSamplingRate:
            self.info('Downsampling the input volumes...')
            for fnVol in self.fnVolDict.values():
                self.runJob("xmipp_image_resize", "-i %s --dim %d"  # Should we store somewhere the resized Vol?
                            % (fnVol, self.xDim), numberOfMpi=self.numberOfMpi)
        else:
            self.info('The target sampling rate %f <= %f the current one, skipping downsample for the 3D Volumes.'
                      % (newSamplingRate, self.samplingRateVol))

    def insertProjMatchStep(self, volumesDict, angularSampling, symmetryGroup, fnAnglesDict, galleryDict):
        xDim = self.xDim
        images = self.imgsFn
        for volId, fnVol in volumesDict.items():
            self.info('Generating a gallery of projections for volume %s' % fnVol)
            fnAngles = fnAnglesDict[volId]
            fnGallery = galleryDict[volId]
            self.projMatchStepAdapted(fnVol, angularSampling, symmetryGroup, images, fnAngles, fnGallery, xDim, volId)

    def projMatchStepAdapted(self, volume, angularSampling, symmetryGroup, images, fnAngles, fnGallery, xDim, volId):
        # Generate gallery of projections
        if volume.endswith('.mrc'):
            volume += ":mrc"

        self.runJob("xmipp_angular_project_library",
                    "-i %s -o %s --sampling_rate %f --sym %s --method fourier 1 0.25 bspline "
                    "--compute_neighbors --angular_distance -1 --experimental_images %s"
                    % (volume, fnGallery, angularSampling, symmetryGroup, images))

        # Assign angles
        self.runJob("xmipp_angular_projection_matching",
                    "-i %s -o %s --ref %s --Ri 0 --Ro %s --max_shift 1000 "
                    "--search5d_shift %s --search5d_step  %s --append"
                    % (images, fnAngles, fnGallery, str(xDim / 2),
                       str(int(xDim / 10)), str(int(xDim / 25))))

        cleanPath(self._getExtraPath('gallery_%d_sampling.xmd' % volId))
        cleanPath(self._getExtraPath('gallery_%d_angles.doc' % volId))
        cleanPath(self._getExtraPath('gallery_%d.doc' % volId))

        # Write angles in the original file and sort
        MD = emlib.MetaData(fnAngles)
        for id in MD:
            galleryReference = MD.getValue(emlib.MDL_REF, id)
            MD.setValue(emlib.MDL_IMAGE_REF, "%05d@%s" % (galleryReference + 1, fnGallery), id)
        MD.write(fnAngles)

    def produceResiduals(self, anglesDict, tS):
        self.fnResidualsDict = {}
        for volId, fnAngles in anglesDict.items():
            anglesOutFn = self._getExtraPath("anglesCont_%d.stk" % volId)
            projectionsOutFn = self._getExtraPath("projections_%d.stk" % volId)
            fnVol = self.fnVolDict[volId]
            xDim = self.xDim
            args = "-i %s -o %s --ref %s --optimizeAngles --optimizeShift --max_shift %d --oprojections %s --sampling %f" \
                   % (fnAngles, anglesOutFn, fnVol, floor(xDim*0.05), projectionsOutFn, tS)

            fnResiduals = self._getExtraPath("residuals_%d.mrcs" % volId)
            self.fnResidualsDict[volId] = fnResiduals

            if self.doEvaluateResiduals:
                args += " --oresiduals %s" % fnResiduals

            if self.ignoreCTF:
                args += " --ignoreCTF "
            if self.optimizeGray:
                args += "--optimizeGray --max_gray_scale 0.95 "

            self.runJob("xmipp_angular_continuous_assign2", args)

    def evaluateResiduals(self):
        for volId, fnResiduals in self.fnResidualsDict.items():
            # Evaluate each image
            fnAutoCorrelations = self._getExtraPath("autocorrelations_%d.xmd" % volId)
            stkAutoCorrelations = self._getExtraPath("autocorrelations_%d.mrcs" % volId)
            stkResiduals = fnResiduals
            anglesOutFn = self._getExtraPath("anglesCont_%d.xmd" % volId)
            self.runJob("xmipp_image_residuals", " -i %s -o %s --save_metadata_stack %s"
                        % (stkResiduals, stkAutoCorrelations, fnAutoCorrelations), numberOfMpi=1)
            self.runJob("xmipp_metadata_utilities", '-i %s --operate rename_column "image imageResidual"'
                        % fnAutoCorrelations, numberOfMpi=1)
            self.runJob("xmipp_metadata_utilities", '-i %s --set join %s imageResidual'
                        % (anglesOutFn, fnAutoCorrelations), numberOfMpi=1)
            cleanPath(fnAutoCorrelations)

    def createOutputStep(self):
        fnImgs = self._getExtraPath('images.stk')
        if os.path.exists(fnImgs):
            cleanPath(fnImgs)

        imgSet = self.inputSet2D.get()
        for volId, anglesFn in self.anglesDict.items():
            imgFn = self._getExtraPath("anglesCont_%d.xmd" % volId)
            self.newAssignmentPerformed = os.path.exists(anglesFn)
            # Special case for 2D classes
            if isinstance(imgSet, SetOfClasses2D):
                outputSet = self._createSetOfClasses2D(imgSet.getImages(), suffix="_vol%d" % volId)
                outputSet.copyInfo(imgSet)
                outputSet.appendFromClasses(imgSet, updateClassCallback=lambda clazz: self._updateClass(clazz, imgFn))
                self._classesInfo = dict()  # For every output you need to reset this value so the _updateClass works
            # Particles or Averages
            else:
                if isinstance(imgSet, SetOfAverages):
                    outputSet = self._createSetOfAverages(suffix="_vol%d" % volId)
                else:
                    outputSet = self._createSetOfParticles(suffix="_vol%d" % volId)
                    if not self.newAssignmentPerformed:
                        outputSet.setAlignmentProj()

                outputSet.copyInfo(imgSet)
                outputSet.setDim((self.xDim, self.xDim, 1))
                outputSet.setObjLabel("")  # To avoid renaming based on the label
                self.iterMd = md.iterRows(imgFn, md.MDL_ITEM_ID)
                self.lastRow = next(self.iterMd)
                outputSet.copyItems(imgSet, updateItemCallback=self._processRow)

            name = self.outputNamesDict[volId]
            self._possibleOutputs[name] = outputSet
            self._defineOutputs(**{name: outputSet})
            self._defineSourceRelation(self.inputSet2D, outputSet)

        if self.doRanking:
            bestVolId = self.computeRankingVolumes(self._possibleOutputs)
            if self.doExtraction:
                volCl = self.inputSet3D.get().getItem("id", bestVolId)  # It may be a Volume or a 3D Class
                outputParticles, outputVol = self._extractElementsFrom3D(volCl)
                if outputParticles:
                    particlesName = "particles_bestVol"
                    self._defineOutputs(**{particlesName: outputParticles})
                    self._defineSourceRelation(self.inputSet3D, outputParticles)

                if outputVol:
                    volName = "bestVolume"
                    self._defineOutputs(**{volName: outputVol})
                    self._store(outputVol)
                    self._defineSourceRelation(self.inputSet3D, outputVol)

        self.writeOutputDict()  # For visualization purpose

    def _updateClass(self, clazz, mdFile):
        """ Callback to update the class"""

        classId = clazz.getObjId()

        if classId not in self._classesInfo:
            self._classesInfo[classId] = clazz
            # Get the row
            row = self._getMdRow(mdFile, classId)
            self._createItemMatrix(clazz, row)

    def _getMdRow(self, mdFile, id):
        """ To get a row. Maybe there is way to request a specific row."""
        for row in md.iterRows(mdFile):
            if row.getValue(md.MDL_ITEM_ID) == id:
                return row

        raise Exception("Missing row %s at %s" % (id, mdFile))

    def _processRow(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_ITEM_ID):
            count += 1
            if count:
                self._createItemMatrix(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None
        particle._appendItem = count > 0

    def _createItemMatrix(self, particle, row):
        setXmippAttributes(particle, row,
                           emlib.MDL_COST, emlib.MDL_CONTINUOUS_GRAY_A,
                           emlib.MDL_CONTINUOUS_GRAY_B, emlib.MDL_CONTINUOUS_X,
                           emlib.MDL_CONTINUOUS_Y,
                           emlib.MDL_CORRELATION_IDX, emlib.MDL_CORRELATION_MASK,
                           emlib.MDL_CORRELATION_WEIGHT, emlib.MDL_IMED)
        if self.doEvaluateResiduals:
            setXmippAttributes(particle, row,
                               emlib.MDL_ZSCORE_RESVAR, emlib.MDL_ZSCORE_RESMEAN,
                               emlib.MDL_ZSCORE_RESCOV)
        def __setXmippImage(label):
            attr = '_xmipp_' + emlib.label2Str(label)
            if not hasattr(particle, attr):
                img = Image()
                setattr(particle, attr, img)
                img.setSamplingRate(particle.getSamplingRate())
            else:
                img = getattr(particle, attr)
            img.setLocation(xmippToLocation(row.getValue(label)))
        
        __setXmippImage(emlib.MDL_IMAGE)
        __setXmippImage(emlib.MDL_IMAGE_REF)
        if self.doEvaluateResiduals:
            __setXmippImage(emlib.MDL_IMAGE_RESIDUAL)
            __setXmippImage(emlib.MDL_IMAGE_COVARIANCE)

    def _useProjMatching(self):
        """ Determine if it is necessary to perform projection matching step (because there is not input alignment)"""
        imgSet = self.inputSet2D.get()
        if not self.useAssignment or isinstance(imgSet, SetOfClasses2D) \
                or (isinstance(imgSet, SetOfAverages) and not imgSet.hasAlignmentProj()) or \
                (isinstance(imgSet, SetOfParticles) and not imgSet.hasAlignmentProj()):
            return True
        else:
            return False

    def _computeResiduals(self, fnVol):
        if fnVol.endswith('.mrc'):
            fnVol += ':mrc'
        program = "xmipp_subtract_projection"
        args = '-i %s --ref %s -o %s --save %s --max_resolution %f --sigma %d --oroot %s' % \
               (self.imgsFn, fnVol, self._getExtraPath("residuals.xmd"), self._getExtraPath(''), self.resol.get(),
                self.sigma.get(), self._getExtraPath("residual_part"))
        self.runJob(program, args, numberOfMpi=1)
        mrcsresiduals = self._getExtraPath("residuals.xmd")
        args2 = " -i %s -o %s" % (mrcsresiduals, self.fnResiduals)
        self.runJob("xmipp_image_convert", args2, numberOfMpi=1)
        fnNewParticles = self._getExtraPath("images.stk")
        if os.path.exists(fnNewParticles):
            cleanPath(fnNewParticles)
        if os.path.exists(mrcsresiduals):
            cleanPath(mrcsresiduals)
        cleanPattern(self._getExtraPath("residual_part*.stk"))

    def computeRankingVolumes(self, outputSetDict):
        self.info('Ranking the best volumes...')
        resultsDict = {}
        meanCostDict = {}
        for outname, outputSet in outputSetDict.items():
            mdLabel = emlib.MDL_COST
            xmippCostValues = {avg.getObjId(): getXmippAttribute(avg, mdLabel).get()
                               for avg in outputSet}
            resultsDict[outname] = xmippCostValues
            meanCostDict[outname] = np.mean(list(xmippCostValues.values()))

        # Find the key-value pair with the maximum value
        bestVolume, bestScore = max(meanCostDict.items(), key=lambda item: item[1])
        bestVolume = bestVolume.split('_')[-1]
        volumeId = int(bestVolume[3:])
        self.info(meanCostDict)
        msg = "The volume with the best score has id %d and score %f" % (volumeId, bestScore)
        self.info(msg)
        self._storeSummaryInfo(msg)

        return volumeId

    def _extractElementsFrom3D(self, volCl):
        """ Extract the elements (particles and/or volume) from the 3D class and create the output """
        outputParticles, outputVol = self._getOutputSet()
        if isinstance(volCl, Class3D):
            self.info('The extraction 3D class have id %d with size %d' % (volCl.getObjId(), volCl.getSize()))
            vol = volCl.getRepresentative().clone()
            if outputParticles is not None:
                # Go through all items and append them
                for image in volCl:
                    newImage = image.clone()
                    outputParticles.append(newImage)
        else:
            self.info('The extraction 3D Volume have id %d' % volCl.getObjId())
            vol = volCl.clone()

        if outputVol is not None:
            # Get the corresponding volume from the 3D class
            outputVol.copyInfo(volCl)
            outputVol.setLocation(vol.getLocation())
            if vol.hasOrigin():
                outputVol.setOrigin(vol.getOrigin())

        return outputParticles, outputVol

    def _getOutputSet(self):
        """ Creates the output sets so they can be filled """
        outputParticles = None
        outputVol = None

        if self.extractOption.get() == self.PARTICLES:
            self.info("Creating set of particles")
            outputParticles = createSetOfParticles(self.inputSet3D.get(), self._getPath())

        elif self.extractOption.get() == self.VOLUME:
            self.info("Creating volume")
            outputVol = createRepresentativeVolume(self.inputSet3D.get())
        else:  # Both
            self.info("Creating both the volume and the set of particles")
            outputParticles = createSetOfParticles(self.inputSet3D.get(), self._getPath())
            outputVol = createRepresentativeVolume(self.inputSet3D.get())

        return outputParticles, outputVol


    def getOutputNamesDict(self):
        return self.outputNamesDict

    def writeOutputDict(self):
        """Write a dictionary to a text file."""
        dictionary = self.getOutputNamesDict()
        filePath = self._getExtraPath(OUTPUTS_FN)
        with open(filePath, 'w') as file:
            json.dump(dictionary, file)

    def readOutputDict(self):
        """Read a dictionary from a text file."""
        filePath = self._getExtraPath(OUTPUTS_FN)
        with open(filePath, 'r') as file:
            dictionary = json.load(file)
        return dictionary

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        summary.append("Images evaluated: %i" % self.inputSet2D.get().getSize())
        summary.append("Volume: %s" % self.inputSet3D.getNameId())

        return summary

    def _storeSummaryInfo(self, rankingMsg):
        self.summaryVar.set(rankingMsg)

    def _validate(self):
        errors = []
        if self.doRanking:
            if not isinstance(self.inputSet3D.get(), EMSet):
                errors.append("The input 3D must be a Set of Volumes or 3D classes to run the ranking option.")
            if self.doExtraction:
                if (isinstance(self.inputSet3D.get(), SetOfVolumes) and
                        (self.extractOption.get() == self.PARTICLES or self.extractOption.get() == self.BOTH)):
                    errors.append("The input 3D must be a Set of Classes 3D in order to extract its particles. "
                                  "Please change to extract Volume option.")
        return errors

    def _methods(self):
        methods = []
        if hasattr(self, 'outputParticles'):
            methods.append("We evaluated %i input images %s regarding to volume %s." %
                           (self.inputSet2D.get().getSize(), self.getObjectTag('inputSet2D'),
                            self.getObjectTag('inputSet3D')))
            methods.append("The residuals were evaluated according to their mean, variance and covariance structure "
                           "[Cherian2013].")
        return methods
    
    # --------------------------- UTILS functions --------------------------------------------
    def _getDimensionsImages(self):
        imgSet = self.inputSet2D.get()
        if isinstance(imgSet, SetOfClasses2D):
            xDim = imgSet.getImages().getDim()[0]
        else:
            xDim = imgSet.getDim()[0]
        return xDim

    def _getDimensionsVol(self):
        volSet = self.inputSet3D.get()
        if isinstance(volSet, EMSet):
            vol = volSet.getFirstItem()
        else:
            vol = volSet
        xDimVol = vol.getDim()[0]
        return xDimVol

    def _getWorkingDimensions(self, srImages, srVol, doDownSample):
        if doDownSample:
            if srImages < srVol:
                if NEW_SAMPLING_RATE < srVol:
                    newDim = self._getDimensionsVol()
                else:
                    factor = srVol / NEW_SAMPLING_RATE
                    newDim = self._getDimensionsVol() * factor
            else:
                if NEW_SAMPLING_RATE < srImages:
                    newDim = self._getDimensionsImages()
                else:
                    factor = srImages / NEW_SAMPLING_RATE
                    newDim = self._getDimensionsImages() * factor
        else:
            if srImages < srVol:
                newDim = self._getDimensionsVol()
            else:
                newDim = self._getDimensionsImages()

        return newDim

#  ---------------------------- HELPERS --------------------------------------
def createRepresentativeVolume(classesSet):
    """ Creates a Volume from the corresponding set from the representative of a set of classes """
    volInput = classesSet.getFirstItem()
    vol = Volume()  # Create an instance of the volume
    vol.copyInfo(volInput)
    return vol


def createSetOfParticles(classesSet, path):
    """ Creates the corresponding set of particles from the input set of classes """
    images = classesSet.getImages()
    particles = SetOfParticles.create(outputPath=path)
    particles.copyInfo(images)
    return particles
