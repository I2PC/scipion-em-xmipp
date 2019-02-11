# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************

from glob import glob
import math
import numpy as np
from os.path import join, exists

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import PointerParam, StringParam, FloatParam, \
    BooleanParam, IntParam
from pyworkflow.utils.path import cleanPath, makePath, copyFile, moveFile
from pyworkflow.em.protocol import ProtClassify3D
from pyworkflow.em.metadata.utils import getFirstRow, getSize
from pyworkflow.em.convert import ImageHandler
import pyworkflow.em.metadata as md
import pyworkflow.em as em

import xmippLib
from xmippLib import MetaData, MD_APPEND, MDL_CLASS_COUNT
from xmipp3.convert import createItemMatrix, setXmippAttributes, \
    writeSetOfParticles, readSetOfParticles


class XmippProtReconstructHeterogeneous(ProtClassify3D):
    """3D Reconstruction with heterogeneous datasets"""
    _label = 'significant heterogeneity'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputParticles', PointerParam, label="Full-size Images",
                      important=True,
                      pointerClass='SetOfParticles', allowsNull=False,
                      help='Select a set of images at full resolution')
        form.addParam('inputVolumes', PointerParam, label="Initial volumes",
                      important=True,
                      pointerClass='SetOfVolumes',
                      help='Select a set of volumes')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='See http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/Symmetry for a description of the symmetry groups format. '
                           'If no symmetry is present, give c1. You can give several symmetries, e.g., c5 c6, meaning that the first '
                           'volume is c5 and the second c6')
        form.addParam('particleRadius', IntParam, default=-1,
                      label='Radius of particle (px)',
                      help='This is the radius (in pixels) of the spherical mask covering the particle in the input images')
        form.addParam('targetResolution', FloatParam, default=8,
                      label='Target resolution',
                      help='Target resolution to solve for the heterogeneity')
        form.addParam('computeDiff', BooleanParam, default=False,
                      label="Compute the difference volumes")
        form.addParam('useGpu', BooleanParam, default=False, label="Use GPU")

        form.addSection(label='Angular assignment')
        form.addParam('numberOfIterations', IntParam, default=3,
                      label='Number of iterations')
        form.addParam('nextMask', PointerParam, label="Mask",
                      pointerClass='VolumeMask', allowsNull=True,
                      help='The mask values must be between 0 (remove these pixels) and 1 (let them pass). Smooth masks are recommended.')
        form.addParam('angularMaxShift', FloatParam, label="Max. shift (%)",
                      default=10,
                      help='Maximum shift as a percentage of the image size')
        line = form.addLine('Tilt angle:',
                            help='0 degrees represent top views, 90 degrees represent side views',
                            expertLevel=LEVEL_ADVANCED)
        line.addParam('angularMinTilt', FloatParam, label="Min.", default=0,
                      expertLevel=LEVEL_ADVANCED)
        line.addParam('angularMaxTilt', FloatParam, label="Max.", default=90,
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('numberOfReplicates', IntParam,
                      label="Max. Number of Replicates", default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help="Significant alignment is allowed to replicate each image up to this number of times")
        form.addParam("numberVotes", IntParam, label="Number of votes", default=3,
                      expertLevel=LEVEL_ADVANCED,
                      help="Number of votes for classification (maximum 5)")
        form.addParam('stochastic', BooleanParam, label="Stochastic",
                      default=False,
                      help="Stochastic optimization")
        form.addParam("stochasticAlpha", FloatParam, label="Relaxation factor",
                      default=0.1, condition="stochastic",
                      expertLevel=LEVEL_ADVANCED,
                      help="Relaxation factor (between 0 and 1). Set it closer to 0 if the random subset size is small")
        form.addParam("stochasticN", IntParam, label="Subset size", default=200,
                      condition="stochastic", expertLevel=LEVEL_ADVANCED,
                      help="Number of images in the random subset")

        form.addParallelSection(threads=1, mpi=8)

    def getNumberOfReconstructedVolumes(self):
        return len(self.inputVolumes.get())

    def parseSymList(self):
        self.symList = self.symmetryGroup.get().strip().split()
        if len(self.symList) < self.getNumberOfReconstructedVolumes():
            self.symList += self.symList * (
                    self.getNumberOfReconstructedVolumes() - len(self.symList))

    # --------------------------- STEPS functions ---------------------------------------------------
    def _insertAllSteps(self):
        self.imgsFn = self._getExtraPath('images.xmd')
        self._insertFunctionStep('convertInputStep',
                                 self.inputParticles.getObjId())

        self.TsOrig = self.inputParticles.get().getSamplingRate()
        firstIteration = 1
        for self.iteration in range(self.numberOfIterations.get()):
            self.insertIteration(firstIteration + self.iteration)
        self._insertFunctionStep("createOutput")

    def insertIteration(self, iteration):
        self._insertFunctionStep('globalAssignment', iteration)
        self._insertFunctionStep('classifyParticles', iteration)
        self._insertFunctionStep('reconstruct', iteration)
        self._insertFunctionStep('postProcessing', iteration)
        self._insertFunctionStep('cleanDirectory', iteration)
        if iteration > 1:
            self._insertFunctionStep('evaluateConvergence', iteration)

    def readInfoField(self, fnDir, block, label):
        mdInfo = xmippLib.MetaData("%s@%s" % (block, join(fnDir, "info.xmd")))
        return mdInfo.getValue(label, mdInfo.firstObject())

    def writeInfoField(self, fnDir, block, label, value):
        mdInfo = xmippLib.MetaData()
        objId = mdInfo.addObject()
        mdInfo.setValue(label, value, objId)
        mdInfo.write("%s@%s" % (block, join(fnDir, "info.xmd")),
                     xmippLib.MD_APPEND)

    def convertInputStep(self, inputParticlesId):
        writeSetOfParticles(self.inputParticles.get(), self.imgsFn)
        self.runJob('xmipp_metadata_utilities',
                    '-i %s --fill image1 constant noImage' % self.imgsFn,
                    numberOfMpi=1)
        self.runJob('xmipp_metadata_utilities',
                    '-i %s --operate modify_values "image1=image"' % self.imgsFn,
                    numberOfMpi=1)
        self.runJob('xmipp_metadata_utilities',
                    '-i %s --fill particleId constant 1' % self.imgsFn,
                    numberOfMpi=1)
        self.runJob('xmipp_metadata_utilities',
                    '-i %s --operate modify_values "particleId=itemId"' % self.imgsFn,
                    numberOfMpi=1)
        imgsFnId = self._getExtraPath('imagesId.xmd')
        self.runJob('xmipp_metadata_utilities',
                    '-i %s --operate keep_column particleId -o %s' % (
                    self.imgsFn, imgsFnId), numberOfMpi=1)

        TsCurrent = max(self.TsOrig, self.targetResolution.get() / 3)
        Xdim = self.inputParticles.get().getDimensions()[0]
        newXdim = long(round(Xdim * self.TsOrig / TsCurrent))
        if newXdim < 40:
            newXdim = long(40)
            TsCurrent = Xdim * (self.TsOrig / newXdim)
        fnDir = self._getExtraPath()
        self.writeInfoField(fnDir, "sampling", xmippLib.MDL_SAMPLINGRATE,
                            TsCurrent)
        self.writeInfoField(fnDir, "size", xmippLib.MDL_XSIZE, newXdim)

        # Prepare images
        print "Preparing images to sampling rate=", TsCurrent
        fnNewParticles = join(fnDir, "imagesResized.stk")
        fnNewMetadata = join(fnDir, "imagesResized.xmd")
        if newXdim != Xdim:
            self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
            self.imgsFn, fnNewParticles, newXdim),
                        numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
        else:
            self.runJob("xmipp_image_convert",
                        "-i %s -o %s --save_metadata_stack %s" % (
                        self.imgsFn, fnNewParticles, fnNewMetadata),
                        numberOfMpi=1)

        R = self.particleRadius.get()
        if R <= 0:
            R = self.inputParticles.get().getDimensions()[0] / 2
        R = min(round(R * self.TsOrig / TsCurrent * (
                1 + self.angularMaxShift.get() * 0.01)), newXdim / 2)
        self.runJob("xmipp_transform_mask",
                    "-i %s --mask circular -%d" % (fnNewParticles, R),
                    numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())

        args = "-i %s --sampling %f --fourier low_pass %f" % (
        fnNewMetadata, TsCurrent, self.targetResolution)
        self.runJob("xmipp_transform_filter", args,
                    numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())

        # Prepare mask
        img = ImageHandler()
        if self.nextMask.hasValue():
            fnMask = join(fnDir, "mask.vol")
            maskObject = self.nextMask.get()
            img.convert(maskObject, fnMask)
            self.runJob('xmipp_image_resize', "-i %s --factor %f" % (
            fnMask, maskObject.getSamplingRate() / TsCurrent), numberOfMpi=1)
            self.runJob('xmipp_transform_window',
                        "-i %s --size %d" % (fnMask, newXdim), numberOfMpi=1)

        # Prepare volumes
        fnDir = self._getExtraPath('Iter000')
        makePath(fnDir)
        listVolumesToProcess = []
        i = 1
        for vol in self.inputVolumes.get():
            fnVol = join(fnDir, "volume%02d.mrc" % i)
            img.convert(vol, fnVol)
            TsVol = vol.getSamplingRate()
            if TsVol != TsCurrent:
                self.runJob('xmipp_image_resize',
                            "-i %s --factor %f" % (fnVol, TsVol / TsCurrent),
                            numberOfMpi=1)
            self.runJob('xmipp_transform_window',
                        "-i %s --size %d" % (fnVol, newXdim), numberOfMpi=1)
            args = "-i %s --sampling %f --fourier low_pass %f" % (
            fnVol, TsCurrent, self.targetResolution)
            self.runJob("xmipp_transform_filter", args, numberOfMpi=1)
            listVolumesToProcess.append(True)
            i += 1
        self._saveVolumesToProcess(listVolumesToProcess)

    def prepareReferences(self, fnDirPrevious, fnDir, TsCurrent, Xdim):
        fnMask = ''
        listVolumesToProcess = self._readVolumesToProcess()
        if self.nextMask.hasValue():
            fnMask = self._getExtraPath("mask.vol")
        for i in range(0, self.getNumberOfReconstructedVolumes()):
            if (listVolumesToProcess[i] == False):
                continue
            fnPreviousVol = join(fnDirPrevious, "volume%02d.mrc" % (i + 1))
            fnReferenceVol = join(fnDir, "volumeRef%02d.mrc" % (i + 1))
            copyFile(fnPreviousVol, fnReferenceVol)
            self.runJob('xmipp_transform_filter',
                        '-i %s --fourier low_pass %f --sampling %f' % \
                        (
                        fnReferenceVol, self.targetResolution.get(), TsCurrent),
                        numberOfMpi=1)
            # AJ duda: se filtra dos veces??
            R = self.particleRadius.get()
            if R <= 0:
                R = self.inputParticles.get().getDimensions()[
                        0] / 2 * self.TsOrig
            self.runJob('xmipp_transform_mask', '-i %s --mask circular -%d' % \
                        (fnReferenceVol, round(R * self.TsOrig / TsCurrent)),
                        numberOfMpi=1)
            self.runJob('xmipp_transform_threshold',
                        '-i %s --select below 0 --substitute value 0' % fnReferenceVol,
                        numberOfMpi=1)
            if fnMask != '':
                self.runJob('xmipp_image_operate',
                            '-i %s --mult %s' % (fnReferenceVol, fnMask),
                            numberOfMpi=1)

    def calculateAngStep(self, newXdim, TsCurrent, ResolutionAlignment):
        k = newXdim * TsCurrent / ResolutionAlignment  # Freq. index
        return math.atan2(1, k) * 180.0 / math.pi  # Corresponding angular step

    def globalAssignment(self, iteration):
        fnDirPrevious = self._getExtraPath("Iter%03d" % (iteration - 1))
        fnDirCurrent = self._getExtraPath("Iter%03d" % iteration)
        makePath(fnDirCurrent)

        TsCurrent = self.readInfoField(self._getExtraPath(), "sampling",
                                       xmippLib.MDL_SAMPLINGRATE)
        newXdim = self.readInfoField(self._getExtraPath(), "size",
                                     xmippLib.MDL_XSIZE)
        self.prepareReferences(fnDirPrevious, fnDirCurrent, TsCurrent, newXdim)

        # Calculate angular step at this resolution
        angleStep = self.calculateAngStep(newXdim, TsCurrent,
                                          self.targetResolution.get())
        angleStep = max(angleStep, 5.0)
        self.writeInfoField(fnDirCurrent, "angleStep", xmippLib.MDL_ANGLE_DIFF,
                            float(angleStep))

        # Generate projections
        fnImgs = self._getExtraPath("imagesResized.xmd")
        fnImgsToUse = fnImgs
        if self.stochastic and iteration < self.numberOfIterations.get():
            fnImgsToUse = join(fnDirCurrent,
                               "imagesResized%02d.xmd" % iteration)
            self.runJob("xmipp_metadata_utilities",
                        "-i %s -o %s --operate random_subset %d" % (
                        fnImgs, fnImgsToUse, self.stochasticN),
                        numberOfMpi=1)
        self.parseSymList()
        listVolumesToProcess = self._readVolumesToProcess()
        print("listVolumesToProcess", listVolumesToProcess)
        for i in range(1, self.getNumberOfReconstructedVolumes() + 1):
            if (listVolumesToProcess[i - 1] == False):
                continue
            fnAngles = join(fnDirCurrent, "angles%02d.xmd" % i)
            fnLocalStk = join(fnDirCurrent, "anglesCont%02d.stk" % i)
            if not exists(fnLocalStk):
                # Create defocus groups
                row = getFirstRow(fnImgsToUse)
                if row.containsLabel(xmippLib.MDL_CTF_MODEL) or row.containsLabel(
                        xmippLib.MDL_CTF_DEFOCUSU):
                    self.runJob("xmipp_ctf_group",
                                "--ctfdat %s -o %s/ctf:stk --pad 1.0 --sampling_rate %f --phase_flipped  --error 0.1 --resol %f" % \
                                (fnImgsToUse, fnDirCurrent, TsCurrent,
                                 self.targetResolution.get()), numberOfMpi=1)
                    moveFile("%s/ctf_images.sel" % fnDirCurrent,
                             "%s/ctf_groups.xmd" % fnDirCurrent)
                    cleanPath("%s/ctf_split.doc" % fnDirCurrent)
                    mdInfo = xmippLib.MetaData(
                        "numberGroups@%s" % join(fnDirCurrent, "ctfInfo.xmd"))
                    fnCTFs = "%s/ctf_ctf.stk" % fnDirCurrent
                    numberGroups = mdInfo.getValue(xmippLib.MDL_COUNT,
                                                   mdInfo.firstObject())
                    ctfPresent = True
                else:
                    numberGroups = 1
                    ctfPresent = False
                    fnCTFs = ""

                # Generate projections
                fnReferenceVol = join(fnDirCurrent, "volumeRef%02d.mrc" % i)
                fnGallery = join(fnDirCurrent, "gallery%02d.stk" % i)
                fnGalleryMd = join(fnDirCurrent, "gallery%02d.xmd" % i)
                args = "-i %s -o %s --sampling_rate %f --perturb %f --sym %s --min_tilt_angle %f --max_tilt_angle %f" % \
                       (fnReferenceVol, fnGallery, angleStep,
                        math.sin(angleStep * math.pi / 180.0) / 4,
                        self.symList[i - 1],
                        self.angularMinTilt.get(), self.angularMaxTilt.get())
                args += " --compute_neighbors --angular_distance -1 --experimental_images %s" % fnImgsToUse
                self.runJob("xmipp_angular_project_library", args,
                            numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
                cleanPath(join(fnDirCurrent, "gallery%02d_sampling.xmd" % i))
                moveFile(join(fnDirCurrent, "gallery%02d.doc" % i), fnGalleryMd)

                noAsignedGroups = 0
                for j in range(1, numberGroups + 1):
                    fnAnglesGroup = join(fnDirCurrent,
                                         "angles_group%02d_%03d.xmd" % (i, j))
                    if not exists(fnAnglesGroup):
                        if ctfPresent:
                            fnGroup = "ctfGroup%06d@%s/ctf_groups.xmd" % (
                            j, fnDirCurrent)
                            fnCTF = "%d@%s/ctf_ctf.stk" % (j, fnDirCurrent)
                            fnGalleryGroup = fnGallery = join(fnDirCurrent,
                                                              "gallery%02d_%06d.stk" % (
                                                              i, j))
                            fnGalleryGroupMd = fnGallery = join(fnDirCurrent,
                                                                "gallery%02d_%06d.xmd" % (
                                                                i, j))
                            self.runJob("xmipp_transform_filter",
                                        "-i %s -o %s --fourier binary_file %s --save_metadata_stack %s --keep_input_columns" % \
                                        (fnGalleryMd, fnGalleryGroup, fnCTF,
                                         fnGalleryGroupMd),
                                        numberOfMpi=min(
                                            self.numberOfMpi.get() * self.numberOfThreads.get(),
                                            24))
                        else:
                            fnGroup = fnImgsToUse
                            fnGalleryGroup = fnGallery
                            fnGalleryGroupMd = fnGalleryMd
                        if getSize(fnGroup) == 0:  # If the group is empty
                            continue
                        maxShift = round(
                            self.angularMaxShift.get() * newXdim / 100)

                        fnAnglesSignificant = join(fnDirCurrent,
                                                   "angles_iter001_00.xmd")
                        if not self.useGpu:
                            args = '-i %s --initgallery %s --maxShift %d --odir %s --dontReconstruct --useForValidation %d --dontApplyFisher' % \
                                   (fnGroup, fnGalleryGroupMd, maxShift,
                                    fnDirCurrent,
                                    self.numberOfReplicates.get() - 1)
                            self.runJob('xmipp_reconstruct_significant', args,
                                        numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
                        else:  # AJ use gpu
                            args = '-i_exp %s -i_ref %s --maxShift %d -o %s --keep_best %d' % \
                                   (fnGroup, fnGalleryGroupMd, maxShift,
                                    fnAnglesSignificant,
                                    self.numberOfReplicates.get())
                            self.runJob('xmipp_cuda_correlation', args,
                                        numberOfMpi=1)

                        if (exists(fnAnglesSignificant) and getSize(
                                fnAnglesSignificant) > 0):
                            print("getSize(fnAnglesSignificant)",
                                  getSize(fnAnglesSignificant))
                            moveFile(fnAnglesSignificant, fnAnglesGroup)
                            cleanPath(
                                join(fnDirCurrent, "images_iter001_00.xmd"))
                            cleanPath(join(fnDirCurrent,
                                           "images_significant_iter001_00.xmd"))
                        else:  # AJ que pasa cuando no se asignan imagenes a ese volumen
                            noAsignedGroups += 1
                            print("noAsignedGroups", noAsignedGroups)
                            if noAsignedGroups == numberGroups:
                                listVolumesToProcess[i - 1] = False
                                self._saveVolumesToProcess(listVolumesToProcess)
                                # raise Exception("There is no angular assignment for this volume")
                                print(
                                "Exception: There is no angular assignment for this volume")
                            continue
                    if exists(fnAnglesGroup):
                        if not exists(fnAngles) and exists(fnAnglesGroup):
                            copyFile(fnAnglesGroup, fnAngles)
                        else:
                            if exists(fnAngles) and exists(fnAnglesGroup):
                                self.runJob("xmipp_metadata_utilities",
                                            "-i %s --set union_all %s" % (
                                            fnAngles, fnAnglesGroup),
                                            numberOfMpi=1)

                if (listVolumesToProcess[i - 1] == True):
                    self.runJob("xmipp_metadata_utilities",
                                "-i %s --set join %s image" % (
                                fnAngles, fnImgsToUse), numberOfMpi=1)
                    self.runJob("rm -f", fnDirCurrent + "/gallery*",
                                numberOfMpi=1)

                    """                
                    fnReferenceVol=join(fnDirCurrent,"volumeRef%02d.mrc"%i)
                    fnGallery=join(fnDirCurrent,"gallery%02d.stk"%i)
                    fnGalleryXmd=join(fnDirCurrent,"gallery%02d.doc"%i)
                    args="-i %s -o %s --sampling_rate %f --perturb %f --sym %s --min_tilt_angle %f --max_tilt_angle %f"%\
                         (fnReferenceVol,fnGallery,angleStep,math.sin(angleStep*math.pi/180.0)/4,self.symList[i-1],
                          self.angularMinTilt.get(),self.angularMaxTilt.get())
                    args+=" --compute_neighbors --angular_distance -1 --experimental_images %s"%fnImgs
                    self.runJob("xmipp_angular_project_library",args,numberOfMpi=self.numberOfMpi.get()*self.numberOfThreads.get())
                    cleanPath(join(fnDirCurrent,"gallery%02d_sampling.xmd"%i))
                    cleanPath(join(fnDirCurrent,"gallery%02d_angles.doc"%i))

                    maxShift=round(self.angularMaxShift.get()*newXdim/100)
                    args='-i %s --initgallery %s --maxShift %d --odir %s --dontReconstruct --useForValidation %d --dontApplyFisher'%\
                         (fnImgsToUse,fnGalleryXmd,maxShift,fnDirCurrent,self.numberOfReplicates.get()-1)
                    self.runJob('xmipp_reconstruct_significant',args,numberOfMpi=self.numberOfMpi.get()*self.numberOfThreads.get())
                    fnAnglesSignificant = join(fnDirCurrent,"angles_iter001_00.xmd")
                    if exists(fnAnglesSignificant): 
                        moveFile(fnAnglesSignificant,fnAngles)
                        self.runJob("xmipp_metadata_utilities",'-i %s --operate sort itemId'%fnAngles,numberOfMpi=1)
                        cleanPath(join(fnDirCurrent,"images_iter001_00.xmd"))
                        cleanPath(join(fnDirCurrent,"images_significant_iter001_00.xmd"))
                    else:
                        raise Exception("There is no angular assignment for this volume")
                    cleanPath(fnGallery)
                    cleanPath(fnGalleryXmd)
                    """

                    args = "-i %s -o %s --sampling %f --Rmax %d --padding 2 --ref %s --max_resolution %f" % \
                           (fnAngles, fnLocalStk, TsCurrent, newXdim / 2,
                            fnReferenceVol, 2 * TsCurrent)
                    args += " --optimizeShift --max_shift %f" % maxShift
                    args += " --optimizeAngles --max_angular_change %f" % (
                            2 * angleStep)
                    if self.numberOfMpi.get() * self.numberOfThreads.get() > 1:
                        args += " --Nsimultaneous 12"
                    if self.inputParticles.get().isPhaseFlipped():
                        args += " --phaseFlipped"
                    self.runJob("xmipp_angular_continuous_assign2", args,
                                numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
                    fnAnglesCont = join(fnDirCurrent, "anglesCont%02d.xmd" % i)
                    self.runJob('xmipp_metadata_utilities',
                                '-i %s --fill weight constant 1' % fnAnglesCont,
                                numberOfMpi=1)
                    self.runJob('xmipp_metadata_utilities',
                                '-i %s --operate modify_values "weight=weightContinuous2"' % fnAnglesCont,
                                numberOfMpi=1)

    def classifyParticles(self, iteration):
        fnDirCurrent = self._getExtraPath("Iter%03d" % iteration)

        # Gather all angles
        fnAnglesAll = join(fnDirCurrent, "anglesAll.xmd")
        mdVolumes = MetaData()
        listVolumesToProcess = self._readVolumesToProcess()
        print("listVolumesToProcess", listVolumesToProcess)
        for i in range(1, self.getNumberOfReconstructedVolumes() + 1):
            if (listVolumesToProcess[i - 1] == False):
                continue
            fnReferenceVol = join(fnDirCurrent, "volumeRef%02d.mrc" % i)
            fnOut = "angles_%02d@%s" % (i, fnAnglesAll)
            fnAngles = join(fnDirCurrent, "anglesCont%02d.xmd" % i)
            if exists(fnAngles):
                md = MetaData(fnAngles)
            else:
                md = MetaData()
            objId = mdVolumes.addObject()
            mdVolumes.setValue(xmippLib.MDL_IMAGE, fnReferenceVol, objId)
            md.write(fnOut, xmippLib.MD_APPEND)
        fnVols = join(fnDirCurrent, "referenceVolumes.xmd")
        mdVolumes.write(fnVols)

        # Classify the images
        # AJ busca el maximo de correlacion entre todos los volumenes a los que se ha asigando cada particula??
        fnImgsId = self._getExtraPath("imagesId.xmd")
        fnOut = join(fnDirCurrent, "classes.xmd")
        print("A correr",
              "xmipp_classify_significant --id %s --angles %s --ref %s -o %s" % (
              fnImgsId, fnAnglesAll, fnVols, fnOut))

        self.runJob("xmipp_classify_significant",
                    "--id %s --angles %s --ref %s -o %s --votes %d" % (
                    fnImgsId, fnAnglesAll, fnVols, fnOut, self.numberVotes), numberOfMpi=1)
        # cleanPath(fnVols)
        # cleanPath(fnAnglesAll)

        #AJ testing
        #copyFile("./correlations.txt", join(fnDirCurrent, "correlations.txt"))


    def reconstruct(self, iteration):
        fnDirCurrent = self._getExtraPath("Iter%03d" % iteration)
        fnDirPrevious = self._getExtraPath("Iter%03d" % (iteration - 1))
        fnOut = join(fnDirCurrent, "classes.xmd")
        fnRootVol = join(fnDirCurrent, "class_")
        TsCurrent = self.readInfoField(self._getExtraPath(), "sampling",
                                       xmippLib.MDL_SAMPLINGRATE)

        self.parseSymList()
        listVolumesToProcess = self._readVolumesToProcess()
        print("listVolumesToProcess", listVolumesToProcess)
        for i in range(1, self.getNumberOfReconstructedVolumes() + 1):
            if (listVolumesToProcess[i - 1] == False):
                continue
            self.runJob("xmipp_metadata_split",
                        "-i class%06d_images@%s --oroot %s_%06d_" % (
                        i, fnOut, fnRootVol, i), numberOfMpi=1)
            fnOutVolPrevious = join(fnDirPrevious, "volume%02d.mrc" % i)
            fnOutContPrevious = join(fnDirPrevious, "anglesCont%02d.stk" % i)
            for half in range(1, 3):
                fnAnglesToUse = "%s_%06d_%06d.xmd" % (fnRootVol, i, half)
                if not exists(fnAnglesToUse):
                    raise Exception(
                        "One of the halves cannot be reconstructed. Probably, class %d is too small" % i)

                row = getFirstRow(fnAnglesToUse)
                hasCTF = row.containsLabel(
                    xmippLib.MDL_CTF_DEFOCUSU) or row.containsLabel(
                    xmippLib.MDL_CTF_MODEL)
                fnCorrectedImagesRoot = join(fnDirCurrent,
                                             "images_corrected%02d" % i)
                deleteStack = False
                if hasCTF:
                    args = "-i %s -o %s.stk --save_metadata_stack %s.xmd --keep_input_columns" % (
                    fnAnglesToUse, fnCorrectedImagesRoot, fnCorrectedImagesRoot)
                    args += " --sampling_rate %f --correct_envelope" % TsCurrent
                    if self.inputParticles.get().isPhaseFlipped():
                        args += " --phase_flipped"
                    self.runJob("xmipp_ctf_correct_wiener2d", args,
                                numberOfMpi=min(
                                    self.numberOfMpi.get() * self.numberOfThreads.get(),
                                    24))
                    cleanPath(fnAnglesToUse)
                    fnAnglesToUse = fnCorrectedImagesRoot + ".xmd"
                    deleteStack = True
                    deletePattern = fnCorrectedImagesRoot + ".*"

                fnOutVol = "%s_%06d_half%d.vol" % (fnRootVol, i, half)
                if not self.useGpu:
                    args = "-i %s -o %s --sym %s --weight --thr %d" % (
                    fnAnglesToUse, fnOutVol, self.symList[i - 1],
                    self.numberOfThreads.get())
                    self.runJob("xmipp_reconstruct_fourier", args,
                                numberOfMpi=self.numberOfMpi.get())
                else:
                    args = "-i %s -o %s --sym %s --weight" % (
                    fnAnglesToUse, fnOutVol, self.symList[i - 1])
                    self.runJob('xmipp_cuda_reconstruct_fourier', args,
                                numberOfMpi=1)

                cleanPath(fnAnglesToUse)
                if deleteStack:
                    cleanPath(deletePattern)
                if self.stochastic and iteration < self.numberOfIterations.get():
                    fnAux = join(fnDirCurrent, "aux.vol")
                    self.runJob("xmipp_image_operate", "-i %s --mult %f" % (
                    fnOutVol, self.stochasticAlpha), numberOfMpi=1)
                    self.runJob("xmipp_image_operate",
                                "-i %s --mult %f -o %s" % (fnOutVolPrevious,
                                                           1 - self.stochasticAlpha.get(),
                                                           fnAux),
                                numberOfMpi=1)
                    self.runJob("xmipp_image_operate",
                                "-i %s --plus %s" % (fnOutVol, fnAux),
                                numberOfMpi=1)
                    cleanPath(fnAux)

            # AJ *** cleanPath(join(fnDirCurrent,"volumeRef%02d.mrc"%i))
            cleanPath(fnOutVolPrevious)
            cleanPath(fnOutContPrevious)

    def postProcessing(self, iteration):

        fnDirCurrent = self._getExtraPath("Iter%03d" % iteration)
        TsCurrent = self.readInfoField(self._getExtraPath(), "sampling",
                                       xmippLib.MDL_SAMPLINGRATE)
        fnRootVol = join(fnDirCurrent, "class_")

        fnMask = ''
        if self.nextMask.hasValue():
            fnMask = self._getExtraPath("mask.vol")
        else:
            R = self.particleRadius.get()
            if R <= 0:
                R = self.inputParticles.get().getDimensions()[
                        0] / 2 * self.TsOrig
            fnMask = self._getExtraPath("mask.mrc")
            self.runJob('xmipp_transform_mask',
                        '-i %s_000001_half1.vol --mask circular -%d --create_mask %s' % \
                        (fnRootVol, round(R * self.TsOrig / TsCurrent), fnMask),
                        numberOfMpi=1)
        fnCentered = join(fnDirCurrent, "volumeCentered.mrc")

        listVolumesToProcess = self._readVolumesToProcess()
        #print("listVolumesToProcess", listVolumesToProcess)
        for i in range(1, self.getNumberOfReconstructedVolumes() + 1):
            # Align the two volumes
            if (listVolumesToProcess[i - 1] == False):
                continue
            fnVol1 = "%s_%06d_half1.vol" % (fnRootVol, i)
            fnVol2 = "%s_%06d_half2.vol" % (fnRootVol, i)
            fnVolAvg = join(fnDirCurrent, "volume%02d.mrc" % i)
            self.runJob('xmipp_image_operate',
                        '-i %s --plus %s -o %s' % (fnVol1, fnVol2, fnVolAvg),
                        numberOfMpi=1)
            self.runJob('xmipp_image_operate', '-i %s --mult 0.5' % fnVolAvg,
                        numberOfMpi=1)
            self.runJob('xmipp_volume_align',
                        '--i1 %s --i2 %s --local --apply' % (fnVolAvg, fnVol1),
                        numberOfMpi=1)
            self.runJob('xmipp_volume_align',
                        '--i1 %s --i2 %s --local --apply' % (fnVolAvg, fnVol2),
                        numberOfMpi=1)

            # Remove untrusted background voxels
            fnRootRestored = join(fnDirCurrent, "volumeRestored")
            args = '--i1 %s --i2 %s --oroot %s --denoising 1 --mask binary_file %s' % (
            fnVol1, fnVol2, fnRootRestored, fnMask)
            self.runJob('xmipp_volume_halves_restoration', args, numberOfMpi=1)
            moveFile("%s_restored1.vol" % fnRootRestored, fnVol1)
            moveFile("%s_restored2.vol" % fnRootRestored, fnVol2)

            # Filter bank denoising
            args = '--i1 %s --i2 %s --oroot %s --filterBank 0.01 --mask binary_file %s' % (
            fnVol1, fnVol2, fnRootRestored, fnMask)
            self.runJob('xmipp_volume_halves_restoration', args, numberOfMpi=1)
            moveFile("%s_restored1.vol" % fnRootRestored, fnVol1)
            moveFile("%s_restored2.vol" % fnRootRestored, fnVol2)
            cleanPath("%s_filterBank.vol" % fnRootRestored)

            # Laplacian Denoising
            args = "-i %s --retinex 0.95 " + fnMask
            self.runJob('xmipp_transform_filter', args % fnVol1, numberOfMpi=1)
            self.runJob('xmipp_transform_filter', args % fnVol2, numberOfMpi=1)

            # Blind deconvolution
            args = '--i1 %s --i2 %s --oroot %s --deconvolution 1 --mask binary_file %s' % (
            fnVol1, fnVol2, fnRootRestored, fnMask)
            self.runJob('xmipp_volume_halves_restoration', args, numberOfMpi=1)
            moveFile("%s_restored1.vol" % fnRootRestored, fnVol1)
            moveFile("%s_restored2.vol" % fnRootRestored, fnVol2)
            self.runJob("xmipp_image_convert",
                        "-i %s_convolved.vol -o %s -t vol" % (
                        fnRootRestored, fnVolAvg), numberOfMpi=1)
            cleanPath("%s_convolved.vol" % fnRootRestored)
            cleanPath("%s_deconvolved.vol" % fnRootRestored)

            # Difference evaluation and production of a consensus average
            args = '--i1 %s --i2 %s --oroot %s --difference 2 2 --mask binary_file %s' % (
            fnVol1, fnVol2, fnRootRestored, fnMask)
            self.runJob('xmipp_volume_halves_restoration', args, numberOfMpi=1)
            self.runJob("xmipp_image_convert",
                        "-i %s_avgDiff.vol -o %s -t vol" % (
                        fnRootRestored, fnVolAvg), numberOfMpi=1)
            cleanPath("%s_avgDiff.vol" % fnRootRestored)
            moveFile("%s_restored1.vol" % fnRootRestored, fnVol1)
            moveFile("%s_restored2.vol" % fnRootRestored, fnVol2)

            # FSC
            self.runJob("xmipp_image_operate",
                        "-i %s --mult %s" % (fnVol1, fnMask), numberOfMpi=1)
            self.runJob("xmipp_image_operate",
                        "-i %s --mult %s" % (fnVol2, fnMask), numberOfMpi=1)
            self.runJob('xmipp_transform_threshold',
                        '-i %s --select below 0 --substitute value 0 ' % fnVol1,
                        numberOfMpi=1)
            self.runJob('xmipp_transform_threshold',
                        '-i %s --select below 0 --substitute value 0 ' % fnVol2,
                        numberOfMpi=1)

            fnFsc = join(fnDirCurrent, "fsc%02d.xmd" % i)
            self.runJob('xmipp_resolution_fsc',
                        '--ref %s -i %s -o %s --sampling_rate %f' \
                        % (fnVol1, fnVol2, fnFsc, TsCurrent), numberOfMpi=1)
            self.runJob('xmipp_transform_filter',
                        '-i %s --fourier fsc %s --sampling %f' % (
                        fnVolAvg, fnFsc, TsCurrent), numberOfMpi=1)
            self.runJob('xmipp_transform_filter',
                        '-i %s --fourier low_pass 0.25' % fnVolAvg,
                        numberOfMpi=1)
            # cleanPath(fnVol1)
            # cleanPath(fnVol2)

            self.runJob('xmipp_image_header',
                        '-i %s --sampling_rate %f' % (fnVolAvg, TsCurrent),
                        numberOfMpi=1)

            if i == 1:
                copyFile(fnVolAvg, fnCentered)
            else:
                self.runJob("xmipp_image_operate",
                            "-i %s --plus %s" % (fnCentered, fnVolAvg),
                            numberOfMpi=1)

        # Align all volumes with respect to center
        print("listVolumesToProcess", listVolumesToProcess)
        for i in range(1, self.getNumberOfReconstructedVolumes() + 1):
            if (listVolumesToProcess[i - 1] == False):
                continue
            fnVoli = join(fnDirCurrent, "volume%02d.mrc" % i)
            self.runJob('xmipp_volume_align',
                        '--i1 %s --i2 %s --local --apply' % (
                        fnCentered, fnVoli), numberOfMpi=1)
        cleanPath(fnCentered)

        # Align all volumes with respect to the first one taking care of the mirror
        fnVol1 = join(fnDirCurrent, "volume%02d.mrc" % 1)
        I1 = xmippLib.Image(fnVol1)
        for i in range(2, self.getNumberOfReconstructedVolumes() + 1):
            if (listVolumesToProcess[i - 1] == False):
                continue
            fnVoli = join(fnDirCurrent, "volume%02d.mrc" % i)
            fnVoliAux1 = join(fnDirCurrent, "volume%02d_aux1.mrc" % i)
            copyFile(fnVoli, fnVoliAux1)
            self.runJob("xmipp_volume_align",
                        "--i1 %s --i2 %s --frm --apply" % (fnVol1, fnVoliAux1),
                        numberOfMpi=1)
            Iaux = xmippLib.Image(fnVoliAux1)
            corr1 = I1.correlation(Iaux)

            fnVoliAux2 = join(fnDirCurrent, "volume%02d_aux2.mrc" % i)
            self.runJob("xmipp_transform_mirror",
                        "-i %s -o %s --flipX" % (fnVoli, fnVoliAux2),
                        numberOfMpi=1)
            self.runJob("xmipp_volume_align",
                        "--i1 %s --i2 %s --frm --apply" % (fnVol1, fnVoliAux2),
                        numberOfMpi=1)
            Iaux = xmippLib.Image(fnVoliAux2)
            corr2 = I1.correlation(Iaux)

            if corr1 > corr2:
                moveFile(fnVoliAux1, fnVoli)
            else:
                moveFile(fnVoliAux2, fnVoli)
            cleanPath(fnVoliAux1)
            cleanPath(fnVoliAux2)

        # Rewrite the first block of the classes.xmd with the representative
        fnClasses = "classes@" + join(fnDirCurrent, "classes.xmd")
        classesmd = md.MetaData(fnClasses)
        for objId in classesmd:
            ref3d = classesmd.getValue(md.MDL_REF3D, objId)
            classesmd.setValue(md.MDL_IMAGE,
                               join(fnDirCurrent, "volume%02d.mrc" % ref3d),
                               objId)
        classesmd.write(fnClasses, md.MD_APPEND)

        # Write the images.xmd
        mdAll = md.MetaData()
        for i in range(1, self.getNumberOfReconstructedVolumes() + 1):
            mdi = md.MetaData(
                "class%06d_images@" % i + join(fnDirCurrent, "classes.xmd"))
            mdAll.unionAll(mdi)
        mdAll.write(join(fnDirCurrent, "images.xmd"))

    def cleanDirectory(self, iteration):
        pass
        # fnDirCurrent=self._getExtraPath("Iter%03d"%iteration)
        # cleanPattern(join(fnDirCurrent,"anglesCont*"))
        # cleanPattern(join(fnDirCurrent,"images_corrected*"))
        # cleanPattern(join(fnDirCurrent,"angles_group*"))
        # cleanPattern(join(fnDirCurrent,"ctf*"))

    def evaluateConvergence(self, iteration):
        if not self.stochastic:
            fnDirCurrent = self._getExtraPath("Iter%03d" % iteration)
            fnDirPrevious = self._getExtraPath("Iter%03d" % (iteration - 1))
            fnIntersection = join(fnDirCurrent, "intersection.xmd")
            fnUnion = join(fnDirCurrent, "union.xmd")
            fnAngleDistance = join(fnDirCurrent, "angle_distance")
            N = self.getNumberOfReconstructedVolumes()
            coocurrence = np.zeros((N, N))
            sizeClasses = np.zeros((N, N))
            self.parseSymList()
            listVolumesToProcess = self._readVolumesToProcess()
            print("listVolumesToProcess", listVolumesToProcess)
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    if (listVolumesToProcess[i - 1] == False or
                            listVolumesToProcess[j - 1] == False):
                        continue
                    fnCurrent = "class%06d_images@%s" % (
                    i, join(fnDirCurrent, "classes.xmd"))
                    fnPrevious = "class%06d_images@%s" % (
                    j, join(fnDirPrevious, "classes.xmd"))
                    self.runJob("xmipp_metadata_utilities",
                                "-i %s --set intersection %s itemId -o %s" % (
                                fnCurrent, fnPrevious, fnIntersection),
                                numberOfMpi=1)
                    self.runJob("xmipp_metadata_utilities",
                                "-i %s --set union %s itemId -o %s" % (
                                fnCurrent, fnPrevious, fnUnion), numberOfMpi=1)

                    sizeIntersection = float(md.getSize(fnIntersection))
                    sizeUnion = float(md.getSize(fnUnion))
                    sizeClasses[i - 1, j - 1] = sizeIntersection
                    coocurrence[i - 1, j - 1] = sizeIntersection / sizeUnion

                    if i == j:
                        self.runJob("xmipp_angular_distance",
                                    "--ang1 %s --ang2 %s --oroot %s --sym %s --check_mirrors --compute_weights 1 particleId 0.5" % \
                                    (fnPrevious, fnCurrent, fnAngleDistance,
                                     self.symList[i - 1]), numberOfMpi=1)
                        distances = md.MetaData(
                            fnAngleDistance + "_weights.xmd")
                        angDistance = distances.getColumnValues(
                            md.MDL_ANGLE_DIFF)
                        avgAngDistance = reduce(lambda x, y: x + y,
                                                angDistance) / len(angDistance)
                        shiftDistance = distances.getColumnValues(
                            md.MDL_SHIFT_DIFF)
                        avgShiftDistance = reduce(lambda x, y: x + y,
                                                  shiftDistance) / len(
                            shiftDistance)
                        print(
                                "Class %d: average angular diff=%f      average shift diff=%f" % (
                        i, avgAngDistance, avgShiftDistance))

            # print("Size of the intersections")
            # print(sizeClasses)
            # print(' ')
            # print('Stability of the classes (coocurrence)')
            # print(coocurrence)

            cleanPath(fnIntersection)
            cleanPath(fnUnion)
            cleanPath(fnAngleDistance + "_weights.xmd")

    def createOutput(self):
        # get last iteration
        fnIterDir = glob(self._getExtraPath("Iter*"))
        lastIter = len(fnIterDir) - 1
        self.fnLastDir = self._getExtraPath("Iter%03d" % lastIter)

        fnLastImages = join(self.fnLastDir, "images.xmd")
        if not exists(fnLastImages):
            raise Exception("The file %s does not exist" % fnLastImages)
        partSet = self.inputParticles.get()
        self.Ts = self.readInfoField(self._getExtraPath(), "sampling",
                                     xmippLib.MDL_SAMPLINGRATE)
        self.scaleFactor = self.Ts / partSet.getSamplingRate()

        classes3D = self._createSetOfClasses3D(partSet)
        # Let use an special iterator to skip missing particles
        # discarded by classification (in case of cl2d)
        setMdIter = md.SetMdIterator(fnLastImages,
                                     sortByLabel=md.MDL_PARTICLE_ID,
                                     updateItemCallback=self._updateParticle)

        classes3D.classifyItems(updateItemCallback=setMdIter.updateItem,
                                updateClassCallback=self._updateClass)
        self._defineOutputs(outputClasses=classes3D)
        self._defineSourceRelation(self.inputParticles, classes3D)

        # # create a SetOfVolumes and define its relations
        # volumes = self._createSetOfVolumes()
        # volumes.setSamplingRate(self.inputParticles.get().getSamplingRate())
        #
        # for class3D in classes3D:
        #     vol = class3D.getRepresentative()
        #     vol.setObjId(class3D.getObjId())
        #     volumes.append(vol)
        #
        # self._defineOutputs(outputVolumes=volumes)
        # self._defineSourceRelation(self.inputParticles, volumes)

        #create a set of particles with the no-assigned particles
        fnSubtracted = join(self.fnLastDir, "imagesNotAssig.xmd")
        self.runJob("xmipp_metadata_utilities",
                    "-i %s --set subtraction %s itemId -o %s" % (self.imgsFn, fnLastImages, fnSubtracted),
                    numberOfMpi=1)
        if getSize(fnSubtracted)>0:
            particlesNotAssig = self._createSetOfParticles("_NotAssigned")
            readSetOfParticles(fnSubtracted, particlesNotAssig)
            particlesNotAssig.copyInfo(partSet)
            self._defineOutputs(particlesNotAssigned=particlesNotAssig)
            self._defineSourceRelation(self.inputParticles, particlesNotAssig)


    def _updateParticle(self, particle, row):
        particle.setClassId(row.getValue(md.MDL_REF3D))
        if row.containsLabel(xmippLib.MDL_CONTINUOUS_X):
            row.setValue(xmippLib.MDL_SHIFT_X, row.getValue(xmippLib.MDL_CONTINUOUS_X) * self.scaleFactor)
            row.setValue(xmippLib.MDL_SHIFT_Y, row.getValue(xmippLib.MDL_CONTINUOUS_Y) * self.scaleFactor)
            row.setValue(xmippLib.MDL_FLIP, row.getValue(xmippLib.MDL_CONTINUOUS_FLIP))
        else:
            row.setValue(xmippLib.MDL_SHIFT_X, row.getValue(xmippLib.MDL_SHIFT_X) * self.scaleFactor)
            row.setValue(xmippLib.MDL_SHIFT_Y, row.getValue(xmippLib.MDL_SHIFT_Y) * self.scaleFactor)
        setXmippAttributes(particle, row, xmippLib.MDL_SHIFT_X,
                           xmippLib.MDL_SHIFT_Y, xmippLib.MDL_ANGLE_ROT,
                           xmippLib.MDL_ANGLE_TILT, xmippLib.MDL_ANGLE_PSI,
                           xmippLib.MDL_MAXCC, xmippLib.MDL_WEIGHT)
        createItemMatrix(particle, row, align=em.ALIGN_PROJ)

    def _updateClass(self, item):
        classId = item.getObjId()
        item.setAlignmentProj()
        #item.setSamplingRate(self.Ts)
        #item.getRepresentative().setFileName(
        #    join(self.fnLastDir, "volume%02d.mrc" % classId))
        if self.scaleFactor!=1.0:
            newXdim=self.inputParticles.get().getDimensions()[0]
            self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (
                join(self.fnLastDir, "volume%02d.mrc" % classId),
                join(self.fnLastDir, "volume%02d.mrc" % classId), newXdim),
                        numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
        item.setSamplingRate(self.inputParticles.get().getSamplingRate())
        item.getRepresentative().setFileName(
            join(self.fnLastDir, "volume%02d.mrc" % classId))

    def _saveVolumesToProcess(self, volumesToProcess):
        fn = open(self._getExtraPath('volumesToProcess.txt'), 'w')
        for flag in volumesToProcess:
            if flag:
                fn.write("1 ")
            else:
                fn.write("0 ")
        fn.close()

    def _readVolumesToProcess(self):
        volumesToProcess = []
        if exists(self._getExtraPath('volumesToProcess.txt')):
            fn = open(self._getExtraPath('volumesToProcess.txt'), 'r')
            text = fn.read()
            listText = text.split(' ')
            for value in listText:
                if value == '1':
                    volumesToProcess.append(True)
                elif value == '0':
                    volumesToProcess.append(False)
            fn.close()
        else:
            volumesToProcess = 0
        return volumesToProcess