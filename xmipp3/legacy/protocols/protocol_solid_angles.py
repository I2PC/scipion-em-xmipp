# **************************************************************************
# *
# * Authors:     C.O.S. Sorzano (coss@cnb.csic.es)
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

from os.path import join, exists
import math
import os

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_1_1
from pyworkflow.utils import makePath, cleanPattern, moveFile, cleanPath
from pwem.emlib.image import ImageHandler
from pwem.constants import ALIGN_PROJ
from pwem.objects import Image, Volume
from pwem.protocols import ProtAnalysis3D
import pwem.emlib.metadata as md
from pyworkflow import BETA, UPDATED, NEW, PROD

from pwem import emlib
from xmipp3.base import findRow, readInfoField, writeInfoField, isXmippCudaPresent
from xmipp3.convert import (rowToAlignment, setXmippAttributes, xmippToLocation,
                            createItemMatrix, writeSetOfParticles)
from xmipp3.constants import SYM_URL, CUDA_ALIGN_SIGNIFICANT


class XmippProtSolidAngles(ProtAnalysis3D):
    """    
    Construct image groups based on the angular assignment. All images assigned within a solid angle
    are assigned to a class. Classes are not exclusive and an image may be assigned to multiple classes
    """

    _label = 'solid angles'
    _lastUpdateVersion = VERSION_1_1
    _devStatus = BETA

    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):

        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')

        form.addParam('inputVolume', params.PointerParam, pointerClass='Volume',
                      label="Input volume",
                      help='Select the input volume.')

        form.addParam('inputParticles', params.PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",
                      help='Select the input experimental images with an '
                           'angular assignment.')

        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry group",
                      help='See %s page for a description of the symmetries '
                           'accepted by Xmipp' % SYM_URL)

        form.addParam('angularSampling', params.FloatParam, default=5,
                      label='Angular sampling',
                      expertLevel=params.LEVEL_ADVANCED, help="In degrees")

        form.addParam('angularDistance', params.FloatParam, default=10,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Angular distance',
                      help="In degrees. An image belongs to a group if its "
                           "distance is smaller than this value")

        form.addParam('maxShift', params.FloatParam, default=15,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Maximum shift',
                      help="In pixels")

        form.addSection("Directional Classes")

        form.addParam('directionalClasses', params.IntParam, default=1,
                      label='Number of directional classes',
                      help="By default only one class will be computed for "
                           "each projection direction. More classes could be"
                           "computed and this is needed for protocols "
                           "split-volume. ")

        form.addParam('homogeneize', params.IntParam, default=-1,
                      label='Homogeneize groups',
                      condition="directionalClasses==1",
                      help="Set to -1 for no homogeneization. Set to 0 for homogeneizing "
                           "to the minimum of class size. Set to any other number to "
                           "homogeneize to that particular number")

        form.addParam('targetResolution', params.FloatParam, default=10,
                      condition="directionalClasses > 1",
                      label='Target resolution (A)')

        form.addParam('refineAngles', params.BooleanParam, default=True,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Refine angles',
                      help="Refine the angles of the classes using a "
                           "continuous angular assignment")

        form.addParam('cl2dIterations', params.IntParam, default=5,
                      expertLevel=params.LEVEL_ADVANCED,
                      condition="directionalClasses > 1",
                      label='Number of CL2D iterations')

        form.addSection("Split volume")
        form.addParam('splitVolume', params.BooleanParam, label="Split volume",
                      condition="directionalClasses > 1", default=False,
                      help='If desired, the protocol can use the directional classes calculated in this protocol to divide the input volume '
                           'into 2 distinct 3D classes as measured by PCA. If the PCA component is just noise, it means that the algorithm '
                           'does not find a difference between the 2D classes')
        form.addParam('mask', params.PointerParam, label="Mask",
                      pointerClass='VolumeMask', allowsNull=True,
                      condition="splitVolume",
                      help='The mask values must be binary: 0 (remove these voxels) and 1 (let them pass).')
        form.addParam('Nrec', params.IntParam,
                      label="Number of reconstructions", default=5000,
                      condition="splitVolume",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Number of random reconstructions to perform");
        form.addParam('Nsamples', params.IntParam,
                      label="Number of images/reconstruction", default=15,
                      condition="splitVolume",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Number of images per reconstruction. Consider that reconstructions with symmetry c1 will be perfomed");
        form.addParam('alpha', params.FloatParam, label="Confidence level",
                      default=0.05, condition="splitVolume",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="This parameter is alpha. Two volumes, one at alpha/2 and another one at 1-alpha/2, will be generated");

        form.addParallelSection(threads=0, mpi=8)

    # --------------------------- INSERT steps functions ------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep',
                                 self.inputParticles.get().getObjId(),
                                 self.inputVolume.get().getObjId())

        self._insertFunctionStep('constructGroupsStep',
                                 self.inputParticles.get().getObjId(),
                                 self.angularSampling.get(),
                                 self.angularDistance.get(),
                                 self.symmetryGroup.get())

        self._insertFunctionStep('classifyGroupsStep')
        if self.directionalClasses.get() == 1 and self.homogeneize.get() >= 0:
            self._insertFunctionStep('homogeneizeStep')

        if self.refineAngles:
            self._insertFunctionStep('refineAnglesStep')

        if self.splitVolume and self.directionalClasses.get() > 1:
            self._insertFunctionStep("splitVolumeStep")

        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions -------------------------------
    def convertInputStep(self, particlesId, volId):
        """ Write the input images as a Xmipp metadata file.
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        inputParticles = self.inputParticles.get()
        inputVolume = self.inputVolume.get()

        writeSetOfParticles(inputParticles, self._getExpParticlesFn())

        img = ImageHandler()
        img.convert(inputVolume, self._getInputVolFn())

        Xdim = inputParticles.getXDim()
        Ts = inputParticles.getSamplingRate()
        if self._useSeveralClasses():
            # Scale particles
            newTs = self.targetResolution.get() * 0.4
            newTs = max(Ts, newTs)
            newXdim = int(Xdim * Ts / newTs)
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (self._getExpParticlesFn(),
                         self._getTmpPath('scaled_particles.stk'),
                         self._getTmpPath('scaled_particles.xmd'),
                         newXdim))
            # Scale volume
            Xdim = inputVolume.getXDim()
            if Xdim != newXdim:
                self.runJob("xmipp_image_resize", "-i %s --dim %d"
                            % (self._getInputVolFn(), newXdim), numberOfMpi=1)
            
            Xdim=newXdim
            Ts=newTs
        writeInfoField(self._getExtraPath(),"sampling",emlib.MDL_SAMPLINGRATE,Ts)
        writeInfoField(self._getExtraPath(),"size",emlib.MDL_XSIZE,int(Xdim))

    def constructGroupsStep(self, particlesId, angularSampling,
                            angularDistance, symmetryGroup):
        args = '-i %s ' % self._getInputVolFn()
        args += '-o %s ' % self._getExtraPath("gallery.stk")
        args += '--sampling_rate %f ' % self.angularSampling
        args += '--sym %s ' % self.symmetryGroup
        args += '--method fourier 1 0.25 bspline --compute_neighbors '
        args += '--angular_distance %f ' % self.angularDistance
        args += '--experimental_images %s ' % self._getInputParticlesFn()
        args += '--max_tilt_angle 90 '

        # Create a gallery of projections of the input volume
        # with the given angular sampling
        self.runJob("xmipp_angular_project_library", args)

        args = '--i1 %s ' % self._getInputParticlesFn()
        args += '--i2 %s ' % self._getExtraPath("gallery.doc")
        args += '-o %s ' % self._getExtraPath("neighbours.xmd")
        args += '--dist %f ' % self.angularDistance
        args += '--sym %s ' % self.symmetryGroup
        args += '--check_mirrors '

        # Compute several groups of the experimental images into
        # different angular neighbourhoods
        self.runJob("xmipp_angular_neighbourhood", args, numberOfMpi=1)

    def classifyOneGroup(self, projNumber, projMdBlock, projRef,
                         mdClasses, mdImages):
        """ Classify one of the neighbourhood groups if not empty.
         Class information will be stored in output metadata: mdOut
        """
        blockSize = md.getSize(projMdBlock)
        Nclasses = self.directionalClasses.get()
        Nlevels = int(math.ceil(math.log(Nclasses) / math.log(2)))

        # Skip projection directions with not enough images to
        # create a given number of classes
        if blockSize / Nclasses < 10:
            return

        fnDir = self._getExtraPath("direction_%s" % projNumber)
        makePath(fnDir)

        # Run CL2D classification for the images assigned to one direction
        args = "-i %s " % projMdBlock
        args += "--odir %s " % fnDir
        args += "--ref0 %s --iter %d --nref %d " % (
        projRef, self.cl2dIterations, Nclasses)
        args += "--distance correlation --classicalMultiref "
        args += "--maxShift %f " % self.maxShift
        try:
                self.runJob("xmipp_classify_CL2D", args)
        except:
            return

        # After CL2D the stk and xmd files should be produced
        classesXmd = join(fnDir, "level_%02d/class_classes.xmd" % Nlevels)
        classesStk = join(fnDir, "level_%02d/class_classes.stk" % Nlevels)

        # Let's check that the output was produced
        if not exists(classesStk):
            return

        # Run align of the class average and the projection representative
        fnAlignRoot = join(fnDir, "classes")
        args = "-i %s " % classesStk
        args += "--ref %s " % projRef
        args += " --oroot %s --iter 1" % fnAlignRoot
        self.runJob("xmipp_image_align", args, numberOfMpi=1)

        # Apply alignment
        args = "-i %s_alignment.xmd --apply_transform" % fnAlignRoot
        self.runJob("xmipp_transform_geometry", args, numberOfMpi=1)

        for classNo in range(1, Nclasses + 1):
            localImagesMd = emlib.MetaData("class%06d_images@%s"
                                              % (classNo, classesXmd))

            # New class detected
            self.classCount += 1
            # Check which images have not been assigned yet to any class
            # and assign them to this new class
            for objId in localImagesMd:
                imgId = localImagesMd.getValue(emlib.MDL_ITEM_ID, objId)
                # Add images not classify yet and store their class number
                if imgId not in self.classImages:
                    self.classImages.add(imgId)
                    newObjId = mdImages.addObject()
                    mdImages.setValue(emlib.MDL_ITEM_ID, imgId, newObjId)
                    mdImages.setValue(emlib.MDL_REF2, self.classCount,
                                      newObjId)

            newClassId = mdClasses.addObject()
            mdClasses.setValue(emlib.MDL_REF, projNumber, newClassId)
            mdClasses.setValue(emlib.MDL_REF2, self.classCount, newClassId)
            mdClasses.setValue(emlib.MDL_IMAGE, "%d@%s" %
                               (classNo, classesStk), newClassId)
            mdClasses.setValue(emlib.MDL_IMAGE1, projRef, newClassId)
            mdClasses.setValue(emlib.MDL_CLASS_COUNT, localImagesMd.size(),
                               newClassId)

    def classifyGroupsStep(self):
        # Create two metadatas, one for classes and another one for images
        mdClasses = emlib.MetaData()
        mdImages = emlib.MetaData()

        fnNeighbours = self._getExtraPath("neighbours.xmd")
        fnGallery = self._getExtraPath("gallery.stk")

        self.classCount = 0
        self.classImages = set()

        for block in emlib.getBlocksInMetaDataFile(fnNeighbours):
            # Figure out the projection number from the block name
            projNumber = int(block.split("_")[1])

            self.classifyOneGroup(projNumber,
                                  projMdBlock="%s@%s" % (block, fnNeighbours),
                                  projRef="%06d@%s" % (projNumber, fnGallery),
                                  mdClasses=mdClasses,
                                  mdImages=mdImages)

        galleryMd = emlib.MetaData(self._getExtraPath("gallery.doc"))
        # Increment the reference number to starts from 1
        galleryMd.operate("ref=ref+1")
        mdJoined = emlib.MetaData()
        # Add extra information from the gallery metadata
        mdJoined.join1(mdClasses, galleryMd, emlib.MDL_REF)
        # Remove unnecessary columns
        md.keepColumns(mdJoined, "ref", "ref2", "image", "image1",
                       "classCount", "angleRot", "angleTilt")

        # Write both classes and images
        fnDirectional = self._getDirectionalClassesFn()
        self.info("Writting classes info to: %s" % fnDirectional)
        mdJoined.write(fnDirectional)

        fnDirectionalImages = self._getDirectionalImagesFn()
        self.info("Writing images info to: %s" % fnDirectionalImages)
        mdImages.write(fnDirectionalImages)

    def homogeneizeStep(self):
        minClass = self.homogeneize.get()
        fnNeighbours = self._getExtraPath("neighbours.xmd")

        # Look for the block with the minimum number of images
        if minClass == 0:
            minClass = 1e38
            for block in emlib.getBlocksInMetaDataFile(fnNeighbours):
                projNumber = int(block.split("_")[1])
                fnDir = self._getExtraPath("direction_%d" % projNumber,
                                           "level_00", "class_classes.xmd")
                if exists(fnDir):
                    blockSize = md.getSize("class000001_images@" + fnDir)
                    if blockSize < minClass:
                        minClass = blockSize

        # Construct the homogeneized metadata
        mdAll = emlib.MetaData()
        mdSubset = emlib.MetaData()
        mdRandom = emlib.MetaData()
        for block in emlib.getBlocksInMetaDataFile(fnNeighbours):
            projNumber = int(block.split("_")[1])
            fnDir = self._getExtraPath("direction_%d" % projNumber, "level_00",
                                       "class_classes.xmd")
            if exists(fnDir):
                mdDirection = emlib.MetaData("class000001_images@" + fnDir)
                mdRandom.randomize(mdDirection)
                mdSubset.selectPart(mdRandom, 0,
                                    min(mdRandom.size(), minClass))
                mdAll.unionAll(mdSubset)
        mdAll.removeDuplicates(md.MDL_ITEM_ID)
        mdAll.sort(md.MDL_ITEM_ID)
        mdAll.fillConstant(md.MDL_PARTICLE_ID, 1)
        fnHomogeneous = self._getExtraPath("images_homogeneous.xmd")
        mdAll.write(fnHomogeneous)
        self.runJob("xmipp_metadata_utilities",
                    '-i %s --operate modify_values "particleId=itemId"' % fnHomogeneous,
                    numberOfMpi=1)

    def refineAnglesStep(self):
        fnTmpDir = self._getTmpPath()
        fnDirectional = self._getDirectionalClassesFn()
        inputParticles = self.inputParticles.get()
        newTs = readInfoField(self._getExtraPath(),"sampling",emlib.MDL_SAMPLINGRATE)
        newXdim = readInfoField(self._getExtraPath(),"size",emlib.MDL_XSIZE)

        # Generate projections
        fnGallery = join(fnTmpDir, "gallery.stk")
        fnGalleryMd = join(fnTmpDir, "gallery.doc")
        fnVol = self._getInputVolFn()
        args = "-i %s -o %s --sampling_rate %f --sym %s" % \
               (fnVol, fnGallery, 5.0, self.symmetryGroup)
        args += " --compute_neighbors --angular_distance -1 --experimental_images %s" % fnDirectional
        self.runJob("xmipp_angular_project_library", args,
                    numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())

        # Global angular assignment
        maxShift = 0.15 * newXdim
        fnAngles = join(fnTmpDir, "angles_iter001_00.xmd")
        if not self.useGpu.get():
            args = '-i %s --initgallery %s --maxShift %d --odir %s --dontReconstruct --useForValidation 0' % \
                   (fnDirectional, fnGalleryMd, maxShift, fnTmpDir)
            self.runJob('xmipp_reconstruct_significant', args,
                        numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
        else:
            count=0
            GpuListCuda=''
            if self.useQueueForSteps() or self.useQueue():
                GpuList = os.environ["CUDA_VISIBLE_DEVICES"]
                GpuList = GpuList.split(",")
                for elem in GpuList:
                    GpuListCuda = GpuListCuda+str(count)+' '
                    count+=1
            else:
                GpuList = ' '.join([str(elem) for elem in self.getGpuList()])
                GpuListAux = ''
                for elem in self.getGpuList():
                    GpuListCuda = GpuListCuda+str(count)+' '
                    GpuListAux = GpuListAux+str(elem)+','
                    count+=1
                os.environ["CUDA_VISIBLE_DEVICES"] = GpuListAux

            args = '-i %s -r %s -o %s --dev %s ' % (fnDirectional, fnGalleryMd, fnAngles, GpuListCuda)
            self.runJob(CUDA_ALIGN_SIGNIFICANT, args, numberOfMpi=1)

        self.runJob("xmipp_metadata_utilities",
                    "-i %s --operate drop_column ref" % fnAngles, numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities",
                    "-i %s --set join %s ref2" % (fnAngles, fnDirectional),
                    numberOfMpi=1)

        # Local angular assignment
        fnAnglesLocalStk = self._getPath("directional_local_classes.stk")
        args = "-i %s -o %s --sampling %f --Rmax %d --padding %d --ref %s --max_resolution %f --applyTo image1 " % \
               (fnAngles, fnAnglesLocalStk, newTs, newXdim / 2, 2, fnVol,
                self.targetResolution)
        args += " --optimizeShift --max_shift %f" % maxShift
        args += " --optimizeAngles --max_angular_change %f" % self.angularDistance
        self.runJob("xmipp_angular_continuous_assign2", args,
                    numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
        moveFile(self._getPath("directional_local_classes.xmd"),
                 self._getDirectionalClassesFn())

        cleanPattern(self._getExtraPath("direction_*"))

    def splitVolumeStep(self):
        newTs = readInfoField(self._getExtraPath(),"sampling",emlib.MDL_SAMPLINGRATE)
        newXdim = readInfoField(self._getExtraPath(),"size",emlib.MDL_XSIZE)
        fnMask = ""
        if self.mask.hasValue():
            fnMask = self._getExtraPath("mask.vol")
            img = ImageHandler()
            img.convert(self.mask.get(), fnMask)
            self.runJob('xmipp_image_resize',
                        "-i %s --dim %d" % (fnMask, newXdim), numberOfMpi=1)
            self.runJob('xmipp_transform_threshold',
                        "-i %s --select below 0.5 --substitute binarize" % fnMask,
                        numberOfMpi=1)

        args = "-i %s --oroot %s --Nrec %d --Nsamples %d --sym %s --alpha %f" % \
               (self._getDirectionalClassesFn(), self._getExtraPath("split"),
                self.Nrec.get(), self.Nsamples.get(),
                self.symmetryGroup.get(), self.alpha.get())
        if fnMask != "":
            args += " --mask binary_file %s" % fnMask
        self.runJob("xmipp_classify_first_split", args, numberOfMpi=1)

    def createOutputStep(self):
        inputParticles = self.inputParticles.get()
        if not self._useSeveralClasses():
            newTs = inputParticles.getSamplingRate()
        else:
            newTs = readInfoField(self._getExtraPath(),"sampling",emlib.MDL_SAMPLINGRATE)

        self.mdClasses = emlib.MetaData(self._getDirectionalClassesFn())
        self.mdImages = emlib.MetaData(self._getDirectionalImagesFn())

        classes2D = self._createSetOfClasses2D(inputParticles)
        classes2D.getImages().setSamplingRate(newTs)

        self.averageSet = self._createSetOfAverages()
        self.averageSet.copyInfo(inputParticles)
        self.averageSet.setAlignmentProj()
        self.averageSet.setSamplingRate(newTs)

        # Let's use a SetMdIterator because it should be less particles
        # in the metadata produced than in the input set
        iterator = md.SetMdIterator(self.mdImages, sortByLabel=md.MDL_ITEM_ID,
                                    updateItemCallback=self._updateParticle,
                                    skipDisabled=True)

        fnHomogeneous = self._getExtraPath("images_homogeneous.xmd")
        if exists(fnHomogeneous):
            homogeneousSet = self._createSetOfParticles()
            homogeneousSet.copyInfo(inputParticles)
            homogeneousSet.setSamplingRate(newTs)
            homogeneousSet.setAlignmentProj()
            self.iterMd = md.iterRows(fnHomogeneous, md.MDL_PARTICLE_ID)
            self.lastRow = next(self.iterMd)
            homogeneousSet.copyItems(inputParticles,
                                     updateItemCallback=self._updateHomogeneousItem)
            self._defineOutputs(outputHomogeneous=homogeneousSet)
            self._defineSourceRelation(self.inputParticles, homogeneousSet)

        classes2D.classifyItems(updateItemCallback=iterator.updateItem,
                                updateClassCallback=self._updateClass)

        self._defineOutputs(outputClasses=classes2D)
        self._defineOutputs(outputAverages=self.averageSet)
        self._defineSourceRelation(self.inputParticles, classes2D)
        self._defineSourceRelation(self.inputParticles, self.averageSet)

        if self.splitVolume and self.directionalClasses.get() > 1:
            volumesSet = self._createSetOfVolumes()
            volumesSet.setSamplingRate(newTs)
            for i in range(2):
                fnVol = self._getExtraPath("split_v%d.vol" % (i + 1))
                fnMrc = self._getExtraPath("split_v%d.mrc" % (i + 1))
                self.runJob("xmipp_image_convert", "-i %s -o %s -t vol" % (fnVol, fnMrc), numberOfMpi=1)
                self.runJob("xmipp_image_header", "-i %s --sampling_rate %f" % (fnMrc, newTs), numberOfMpi=1)
                cleanPath(fnVol)

                vol = Volume()
                vol.setLocation(1,fnMrc)
                volumesSet.append(vol)

            self._defineOutputs(outputVolumes=volumesSet)
            self._defineSourceRelation(inputParticles, volumesSet)

    def _updateHomogeneousItem(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(
                md.MDL_PARTICLE_ID):
            count += 1
            if count:
                createItemMatrix(particle, self.lastRow, align=ALIGN_PROJ)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None

        particle._appendItem = count > 0

    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(emlib.MDL_REF2))

    def _updateClass(self, item):
        classId = item.getObjId()
        classRow = findRow(self.mdClasses, emlib.MDL_REF2, classId)

        representative = item.getRepresentative()
        representative.setTransform(rowToAlignment(classRow, ALIGN_PROJ))
        representative.setLocation(
            xmippToLocation(classRow.getValue(emlib.MDL_IMAGE)))
        setXmippAttributes(representative, classRow, emlib.MDL_ANGLE_ROT)
        setXmippAttributes(representative, classRow, emlib.MDL_ANGLE_TILT)
        setXmippAttributes(representative, classRow, emlib.MDL_CLASS_COUNT)

        self.averageSet.append(representative)

        reprojection = Image()
        reprojection.setLocation(
            xmippToLocation(classRow.getValue(emlib.MDL_IMAGE1)))
        item.reprojection = reprojection

    # --------------------------- INFO functions -------------------------------

    def _validate(self):
        validateMsgs = []
        # if there are Volume references, it cannot be empty.
        if self.inputVolume.get() and not self.inputVolume.hasValue():
            validateMsgs.append('Please provide an input reference volume.')
        if self.inputParticles.get() and not self.inputParticles.hasValue():
            validateMsgs.append('Please provide input particles.')
        if self.useGpu and not isXmippCudaPresent():
            validateMsgs.append("You have asked to use GPU, but I cannot find the Xmipp GPU programs")
        return validateMsgs

    def _summary(self):
        summary = []
        return summary

    # ----------------------- UTILITY FUNCTIONS ---------------------------------

    def _useSeveralClasses(self):
        return self.directionalClasses > 1

    def _getExpParticlesFn(self):
        return self._getPath('input_particles.xmd')

    def _getInputParticlesFn(self):
        if self._useSeveralClasses():
            return self._getTmpPath('scaled_particles.xmd')
        else:
            return self._getExpParticlesFn()

    def _getInputVolFn(self):
        return self._getTmpPath('volume.vol')

    def _getDirectionalClassesFn(self):
        return self._getPath("directional_classes.xmd")

    def _getDirectionalImagesFn(self):
        return self._getPath("directional_images.xmd")
