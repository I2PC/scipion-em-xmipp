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

import pyworkflow.protocol.params as params
from pyworkflow import VERSION_2_0
from pyworkflow.utils.path import makePath, cleanPattern, moveFile
from pyworkflow.em.convert import ImageHandler
from pyworkflow.em.constants import ALIGN_PROJ
from pyworkflow.em.data import Image, Volume
from pyworkflow.em.protocol import ProtAnalysis3D
from xmipp3.convert import createItemMatrix, writeSetOfParticles, \
    rowToAlignment, setXmippAttributes, xmippToLocation
import pyworkflow.em.metadata as md
import pyworkflow.em as em

import xmippLib
from xmipp3.base import findRow
from xmipp3.constants import SYM_URL
import numpy as np


class XmippProtSplitVolumeHierarchical(ProtAnalysis3D):
    """
    Construct image groups based on the angular assignment. All images assigned within a solid angle
    are assigned to a class. Classes are not exclusive and an image may be assigned to multiple classes
    """

    _label = 'split volume hierarchical'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
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

        form.addParam('cl2dIterations', params.IntParam, default=5,
                      expertLevel=params.LEVEL_ADVANCED,
                      condition="directionalClasses > 1",
                      label='Number of CL2D iterations')

        form.addParam('maxCLimgs', params.IntParam, default=5000,
                      condition="directionalClasses > 1",
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Max. Number of images per cone',
                      help='If there are more than this number of images in a cone, '
                           'then a random subset of this size is taken. Set to -1'
                           'to disable this option.')

        form.addSection("Split volume")
        form.addParam('splitVolume', params.BooleanParam, label="Split volume",
                      condition="directionalClasses > 1", default=False,
                      help='If desired, the protocol can use the directional classes calculated in this protocol to divide the input volume '
                           'into 2 distinct 3D classes as measured by PCA. If the PCA component is just noise, it means that the algorithm '
                           'does not find a difference between the 2D classes')
        form.addParam('Niter', params.IntParam,
                      label="Number of iterations", default=5000,
                      condition="splitVolume",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Number of iterations to perform the volume splitting.")
        form.addParam('Nrec', params.IntParam,
                      label="Number of reconstructions", default=5,
                      condition="splitVolume",
                      expertLevel=params.LEVEL_ADVANCED,
                      help="Number of reconstructions to perform the hierarchical clustering.")

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

        self._insertFunctionStep('refineAnglesStep')

        if self.splitVolume and self.directionalClasses.get() > 1:
            self._insertFunctionStep("splitVolumeStep")

        self._insertFunctionStep('cleaningStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions -------------------------------

    def readInfoField(self, fnDir, block, label):
        mdInfo = xmippLib.MetaData("%s@%s" % (block, join(fnDir, "iterInfo.xmd")))
        return mdInfo.getValue(label, mdInfo.firstObject())

    def writeInfoField(self, fnDir, block, label, value):
        mdInfo = xmippLib.MetaData()
        objId = mdInfo.addObject()
        mdInfo.setValue(label, value, objId)
        mdInfo.write("%s@%s" % (block, join(fnDir, "iterInfo.xmd")),
                     xmippLib.MD_APPEND)

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

        if self._useSeveralClasses():
            # Scale particles
            Xdim = inputParticles.getXDim()
            Ts = inputParticles.getSamplingRate()
            newTs = self.targetResolution.get() * 0.4
            newTs = max(Ts, newTs)
            newXdim = long(Xdim * Ts / newTs)
            self.writeInfoField(self._getExtraPath(), "sampling",
                                xmippLib.MDL_SAMPLINGRATE, newTs)
            self.writeInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE,
                                newXdim)
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
        fnToUse = projMdBlock
        if self.maxCLimgs>0 and blockSize>self.maxCLimgs:
            fnToUSe = self._getTmpPath("coneImages.xmd")
            self.runJob("xmipp_metadata_utilities","-i %s -o %s --operate random_subset %d"\
                        %(projMdBlock,fnToUSe,self.maxCLimgs))
        Nclasses = self.directionalClasses.get()
        Nlevels = int(math.ceil(math.log(Nclasses) / math.log(2)))

        # Skip projection directions with not enough images to
        # create a given number of classes
        if blockSize / Nclasses < 10:
            return

        fnDir = self._getExtraPath("direction_%s" % projNumber)
        if not exists(join(fnDir,"level_00")):
            makePath(fnDir)

            # Run CL2D classification for the images assigned to one direction
            args = "-i %s " % fnToUse
            args += "--odir %s " % fnDir
            args += "--ref0 %s --iter %d --nref %d " % (
            projRef, self.cl2dIterations, Nclasses)
            args += "--distance correlation --classicalMultiref "
            args += "--maxShift %f " % self.maxShift
            try:
                self.runJob("xmipp_classify_CL2D", args, numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
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
            localImagesMd = xmippLib.MetaData("class%06d_images@%s"
                                           % (classNo, classesXmd))

            # New class detected
            self.classCount += 1
            # Check which images have not been assigned yet to any class
            # and assign them to this new class
            for objId in localImagesMd:
                imgId = localImagesMd.getValue(xmippLib.MDL_ITEM_ID, objId)
                # Add images not classify yet and store their class number
                if imgId not in self.classImages:
                    self.classImages.add(imgId)
                    newObjId = mdImages.addObject()
                    mdImages.setValue(xmippLib.MDL_ITEM_ID, imgId, newObjId)
                    mdImages.setValue(xmippLib.MDL_REF2, self.classCount, newObjId)

            newClassId = mdClasses.addObject()
            mdClasses.setValue(xmippLib.MDL_REF, projNumber, newClassId)
            mdClasses.setValue(xmippLib.MDL_REF2, self.classCount, newClassId)
            mdClasses.setValue(xmippLib.MDL_IMAGE, "%d@%s" %
                               (classNo, classesStk), newClassId)
            mdClasses.setValue(xmippLib.MDL_IMAGE1, projRef, newClassId)
            mdClasses.setValue(xmippLib.MDL_CLASS_COUNT, localImagesMd.size(),
                               newClassId)

    def classifyGroupsStep(self):
        # Create two metadatas, one for classes and another one for images
        mdClasses = xmippLib.MetaData()
        mdImages = xmippLib.MetaData()

        fnNeighbours = self._getExtraPath("neighbours.xmd")
        fnGallery = self._getExtraPath("gallery.stk")

        self.classCount = 0
        self.classImages = set()

        for block in xmippLib.getBlocksInMetaDataFile(fnNeighbours):
            # Figure out the projection number from the block name
            projNumber = int(block.split("_")[1])

            self.classifyOneGroup(projNumber,
                                  projMdBlock="%s@%s" % (block, fnNeighbours),
                                  projRef="%06d@%s" % (projNumber, fnGallery),
                                  mdClasses=mdClasses,
                                  mdImages=mdImages)

        galleryMd = xmippLib.MetaData(self._getExtraPath("gallery.doc"))
        # Increment the reference number to starts from 1
        galleryMd.operate("ref=ref+1")
        mdJoined = xmippLib.MetaData()
        # Add extra information from the gallery metadata
        mdJoined.join1(mdClasses, galleryMd, xmippLib.MDL_REF)
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
            for block in xmippLib.getBlocksInMetaDataFile(fnNeighbours):
                projNumber = int(block.split("_")[1])
                fnDir = self._getExtraPath("direction_%d" % projNumber,
                                           "level_00", "class_classes.xmd")
                if exists(fnDir):
                    blockSize = md.getSize("class000001_images@" + fnDir)
                    if blockSize < minClass:
                        minClass = blockSize

        # Construct the homogeneized metadata
        mdAll = xmippLib.MetaData()
        mdSubset = xmippLib.MetaData()
        mdRandom = xmippLib.MetaData()
        for block in xmippLib.getBlocksInMetaDataFile(fnNeighbours):
            projNumber = int(block.split("_")[1])
            fnDir = self._getExtraPath("direction_%d" % projNumber, "level_00",
                                       "class_classes.xmd")
            if exists(fnDir):
                mdDirection = xmippLib.MetaData("class000001_images@" + fnDir)
                mdRandom.randomize(mdDirection)
                mdSubset.selectPart(mdRandom, 0L,
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
        newTs = self.readInfoField(self._getExtraPath(), "sampling",
                                   xmippLib.MDL_SAMPLINGRATE)
        newXdim = self.readInfoField(self._getExtraPath(), "size",
                                     xmippLib.MDL_XSIZE)

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
        args = '-i %s --initgallery %s --maxShift %d --odir %s --dontReconstruct --useForValidation 0' % \
               (fnDirectional, fnGalleryMd, maxShift, fnTmpDir)
        self.runJob('xmipp_reconstruct_significant', args,
                    numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
        fnAngles = join(fnTmpDir, "angles_iter001_00.xmd")
        self.runJob("xmipp_metadata_utilities",
                    "-i %s --operate drop_column ref" % fnAngles, numberOfMpi=1)
        self.runJob("xmipp_metadata_utilities",
                    "-i %s --set join %s ref2" % (fnAngles, fnDirectional),
                    numberOfMpi=1)

        # Local angular assignment
        fnAnglesLocalStk = self._getPath("directional_local_classes.stk")
        args = "-i %s -o %s --sampling %f --Rmax %d --padding %d --ref %s --max_resolution %f --applyTo image1 --Nsimultaneous %d" % \
               (fnAngles, fnAnglesLocalStk, newTs, newXdim / 2, 2, fnVol,
                self.targetResolution, 8)
        args += " --optimizeShift --max_shift %f" % maxShift
        args += " --optimizeAngles --max_angular_change %f" % self.angularDistance
        self.runJob("xmipp_angular_continuous_assign2", args,
                    numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
        moveFile(self._getPath("directional_local_classes.xmd"),
                 self._getDirectionalClassesFn())

        cleanPattern(self._getExtraPath("direction_*"))

    def splitVolumeStep(self):
        mdDirectional = md.MetaData(self._getDirectionalClassesFn())
        ref2vals = mdDirectional.getColumnValues(xmippLib.MDL_REF2)
        ref2Max = max(ref2vals)

        matrixCoOc=np.zeros((ref2Max, ref2Max))

        mpiCommand = self.getHostConfig().mpiCommand.get()
        mpiCommand2 = mpiCommand % {'JOB_NODES': self.numberOfMpi.get() * self.numberOfThreads.get(),
                                   'COMMAND': ''}
        # print(mpiCommand2)
        # print(self.numberOfMpi.get())
        # print(self.numberOfThreads.get())

        fails = 0
        for i in range(self.Nrec):
            fnRoot = self._getExtraPath("split%06d"%i)
            args = "-i %s --oroot %s --Niter %d --sym %s --mpiCommand '%s'" % \
                   (self._getDirectionalClassesFn(), fnRoot,
                    self.Niter.get(), self.symmetryGroup.get(), mpiCommand2)
            try:
                self.runJob("xmipp_classify_first_split3", args, numberOfMpi=1)
            except:
                fails+=1
                continue

            ########### build co-ocurrence matrix ##############
            outMd = md.MetaData(fnRoot+'_avg1.xmd')
            listObjRef1=[]
            for id1 in outMd:
                refObj = outMd.getValue(xmippLib.MDL_REF2, id1)
                listObjRef1.append(refObj)
            for val1 in listObjRef1:
                for val2 in listObjRef1:
                    #print("MD1: val1= ", val1, " val2= ", val2)
                    matrixCoOc[val1-1][val2-1]+=1
            #print("matrixCoOc ", matrixCoOc)

            outMd = md.MetaData(fnRoot + '_avg2.xmd')
            listObjRef2 = []
            for id1 in outMd:
                refObj = outMd.getValue(xmippLib.MDL_REF2, id1)
                listObjRef2.append(refObj)
            for val1 in listObjRef2:
                for val2 in listObjRef2:
                    #print("MD1: val1= ", val1, " val2= ", val2)
                    matrixCoOc[val1 - 1][val2 - 1] += 1
            #print("matrixCoOc ", matrixCoOc)

        #Changing from co-ocurrence matrix to distance matrix (avoiding divide by zero)
        if fails==self.Nrec: #AJ revisar esto
            raise Exception('xmipp_classify_first_split3 has failed')
        for i in range(ref2Max):
            for j in range(ref2Max):
                if matrixCoOc[i][j]==0:
                    matrixCoOc[i][j]+=0.001
                matrixCoOc[i][j] = 1.0/matrixCoOc[i][j]
        np.savetxt(self._getExtraPath('coocurrenceMatrix.txt'), matrixCoOc, fmt='%.4e')

        ##### hierarchical clustering algorithm #########
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=2, linkage="complete", affinity="euclidean")
        model.fit(matrixCoOc)

        listLabels = model.labels_
        #print(model.labels_)

        ####### build the final volumes with the selected images by clustering #########
        defMd1 = md.MetaData()
        defMd2 = md.MetaData()
        origMd = md.MetaData(self._getDirectionalClassesFn())
        for row in md.iterRows(origMd):
            refIdx = origMd.getValue(xmippLib.MDL_REF2, row.getObjId())
            if(listLabels[refIdx-1]==0):
                row.addToMd(defMd1)
            else:
                row.addToMd(defMd2)

        defMd1.write(self._getExtraPath("split1.xmd"))
        defMd2.write(self._getExtraPath("split2.xmd"))

        args = "-i %s -o %s --max_resolution 0.25 --sym %s -v 0" % \
               (self._getExtraPath("split1.xmd"),
                self._getExtraPath("split1.vol"), self.symmetryGroup.get())
        self.runJob("xmipp_reconstruct_fourier", args, numberOfMpi=1)

        args = "-i %s -o %s --max_resolution 0.25 --sym %s -v 0" % \
               (self._getExtraPath("split2.xmd"),
                self._getExtraPath("split2.vol"), self.symmetryGroup.get())
        self.runJob("xmipp_reconstruct_fourier", args, numberOfMpi=1)

    def cleaningStep(self):
        cleanPattern(self._getExtraPath("gallery*"))
        cleanPattern(self._getExtraPath("mask.vol"))
        cleanPattern(self._getExtraPath("neighbours.xmd"))
        cleanPattern(self._getExtraPath("split00*"))

    def createOutputStep(self):

        inputParticles = self.inputParticles.get()
        # if not self._useSeveralClasses():
        #     newTs = inputParticles.getSamplingRate()
        # else:
        #     newTs = self.readInfoField(self._getExtraPath(), "sampling",
        #                                xmipp.MDL_SAMPLINGRATE)

        self.mdClasses = xmippLib.MetaData(self._getDirectionalClassesFn())
        self.mdImages = xmippLib.MetaData(self._getDirectionalImagesFn())

        origTs = inputParticles.getSamplingRate()
        lastTs = self.readInfoField(self._getExtraPath(), "sampling",
                                       xmippLib.MDL_SAMPLINGRATE)

        if origTs!=lastTs:
            newXdim=inputParticles.getXDim()
            self.runJob("xmipp_image_resize", "-i %s -o %s --save_metadata_stack %s --fourier %d"
                        % (self._getDirectionalClassesFn(),
                           self._getPath("aux_directional_local_classes.stk"),
                           self._getPath("aux_directional_classes.xmd"),
                           newXdim), numberOfMpi=1)
            from shutil import copy
            copy(self._getPath("aux_directional_local_classes.stk"),
                 self._getPath("directional_local_classes.stk"))
            copy(self._getPath("aux_directional_classes.xmd"),
                 self._getPath("directional_classes.xmd"))
            cleanPattern(self._getPath("aux_directional*"))


        classes2D = self._createSetOfClasses2D(inputParticles)
        #classes2D.getImages().setSamplingRate(newTs)
        classes2D.getImages().setSamplingRate(origTs)

        self.averageSet = self._createSetOfAverages()
        self.averageSet.copyInfo(inputParticles)
        self.averageSet.setAlignmentProj()
        #self.averageSet.setSamplingRate(newTs)
        self.averageSet.setSamplingRate(origTs)

        # Let's use a SetMdIterator because it should be less particles
        # in the metadata produced than in the input set
        iterator = md.SetMdIterator(self.mdImages, sortByLabel=md.MDL_ITEM_ID,
                                    updateItemCallback=self._updateParticle,
                                    skipDisabled=True)

        fnHomogeneous = self._getExtraPath("images_homogeneous.xmd")
        if exists(fnHomogeneous):
            if origTs != lastTs:
                newXdim = inputParticles.getXDim()
                self.runJob("xmipp_image_resize", "-i %s --dim %d"
                            % (fnHomogeneous, newXdim), numberOfMpi=1)
            homogeneousSet = self._createSetOfParticles()
            homogeneousSet.copyInfo(inputParticles)
            #homogeneousSet.getImages().setSamplingRate(newTs)
            homogeneousSet.getImages().setSamplingRate(origTs)
            homogeneousSet.setAlignmentProj()
            self.iterMd = md.iterRows(fnHomogeneous, md.MDL_PARTICLE_ID)
            self.lastRow = next(self.iterMd)
            homogeneousSet.copyItems(inputParticles,
                                     updateItemCallback=self._updateHomogeneousItem)
            self._defineOutputs(outputHomogeneous=homogeneousSet)
            self._defineSourceRelation(self.inputParticles, homogeneousSet)

        #AJ testing
        #AJ por que desaparece una clase que tiene imagenes asignadas
        listRefId=[]
        for row in md.iterRows(self.mdClasses, xmippLib.MDL_REF2):

            refId = row.getValue(xmippLib.MDL_REF2, row.getObjId())
            if len(listRefId)>0 and refId != listRefId[-1]+1:
                whereEnd = listRefId[-1]+1
                for i in range(refId-whereEnd):
                    rowNew = row
                    rowNew.setValue(xmippLib.MDL_REF2, listRefId[-1]+i+1)
                    rowNew.setValue(xmippLib.MDL_IMAGE, 'None')
                    rowNew.setValue(xmippLib.MDL_IMAGE1, 'None')
                    rowNew.addToMd(self.mdClasses)
                    listRefId.append(listRefId[-1]+i+1)

                listRefId.append(refId)
            else:
                listRefId.append(refId)

        self.mdClasses.write(self._getDirectionalClassesFn())
        self.mdClasses = xmippLib.MetaData(self._getDirectionalClassesFn())
        #END AJ


        classes2D.classifyItems(updateItemCallback=iterator.updateItem,
                                updateClassCallback=self._updateClass)

        self._defineOutputs(outputClasses=classes2D)
        self._defineOutputs(outputAverages=self.averageSet)
        self._defineSourceRelation(self.inputParticles, classes2D)
        self._defineSourceRelation(self.inputParticles, self.averageSet)

        if self.splitVolume and self.directionalClasses.get() > 1:
            volumesSet = self._createSetOfVolumes()
            #volumesSet.setSamplingRate(newTs)
            volumesSet.setSamplingRate(origTs)
            for i in range(2):
                vol = Volume()
                if origTs != lastTs:
                    newXdim = inputParticles.getXDim()
                    self.runJob("xmipp_image_resize", "-i %s --dim %d"
                                % (self._getExtraPath("split%d.vol" % (i + 1)),
                                   newXdim), numberOfMpi=1)
                vol.setLocation(1, self._getExtraPath("split%d.vol" % (i + 1)))
                volumesSet.append(vol)

            self._defineOutputs(outputVolumes=volumesSet)
            self._defineSourceRelation(inputParticles, volumesSet)

    def _updateHomogeneousItem(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(
                md.MDL_PARTICLE_ID):
            count += 1
            if count:
                createItemMatrix(particle, self.lastRow, align=em.ALIGN_PROJ)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None

        particle._appendItem = count > 0


    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(xmippLib.MDL_REF2))

    def _updateClass(self, item):
        classId = item.getObjId()
        classRow = findRow(self.mdClasses, xmippLib.MDL_REF2, classId)

        if classRow is not None:
            representative = item.getRepresentative()
            representative.setTransform(rowToAlignment(classRow, ALIGN_PROJ))
            representative.setLocation(
                xmippToLocation(classRow.getValue(xmippLib.MDL_IMAGE)))
            setXmippAttributes(representative, classRow, xmippLib.MDL_ANGLE_ROT)
            setXmippAttributes(representative, classRow, xmippLib.MDL_ANGLE_TILT)
            setXmippAttributes(representative, classRow, xmippLib.MDL_CLASS_COUNT)

            self.averageSet.append(representative)

            reprojection = Image()
            reprojection.setLocation(
                xmippToLocation(classRow.getValue(xmippLib.MDL_IMAGE1)))
            item.reprojection = reprojection


    # --------------------------- INFO functions -------------------------------

    def _validate(self):
        validateMsgs = []
        # if there are Volume references, it cannot be empty.
        if self.inputVolume.get() and not self.inputVolume.hasValue():
            validateMsgs.append('Please provide an input reference volume.')
        if self.inputParticles.get() and not self.inputParticles.hasValue():
            validateMsgs.append('Please provide input particles.')
        if self.angularSampling.get()>40:
            validateMsgs.append("The angular sampling must be <= 40")
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
