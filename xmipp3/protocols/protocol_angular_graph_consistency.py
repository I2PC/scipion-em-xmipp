# **************************************************************************
# *
# * Authors:         Jeison Mendez (jmendez@utp.edu.co) (2020)
# *
# * Universidad Nacional Autónoma de México, UNAM
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

from glob import glob
import math
from os.path import join, isfile, exists
from shutil import copyfile, copy
import os

from pyworkflow.protocol.params import (PointerParam, FloatParam,
                                        STEPS_PARALLEL,
                                        StringParam, BooleanParam, IntParam,
                                        LEVEL_ADVANCED, USE_GPU, GPU_LIST)

from pyworkflow.protocol import String, Float

from pyworkflow.protocol import params

from pyworkflow.utils.path import (moveFile, makePath, cleanPattern, copyFile, cleanPath)
from pyworkflow.gui.plotter import Plotter

from pwem.objects import Volume, SetOfVolumes
from pwem import emlib
from pwem.emlib.image import ImageHandler
from pwem.emlib.metadata import getFirstRow, getSize, iterRows
import pwem.emlib.metadata as md
from pwem.protocols import ProtAnalysis3D
from pwem.constants import ALIGN_PROJ

from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes, \
    getImageLocation, setXmippAttributes, createItemMatrix, readSetOfParticles, readSetOfImages
from _ast import Try


class XmippProtAngularGraphConsistency(ProtAnalysis3D):
    """
    Performs soft alignment validation of a set of particles previously aligned
    confronting them using Graph filtered correlations representation. This
    protocol produces an histogram with two groups of particles.
    """
    _label = 'angular graph consistency'
    
    def __init__(self, *args, **kwargs):
        ProtAnalysis3D.__init__(self, *args, **kwargs)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        form.addSection(label='Input')
        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input volume", important=True,
                      help='Select the input volume(s).')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignment',
                      label="Input particles", important=True,
                      help='Select the input projection images.')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group",
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp')
        form.addParam('angularSampling', FloatParam, default=3,
                      expertLevel=LEVEL_ADVANCED,
                      label="Angular Sampling (degrees)",
                      help='Angular distance (in degrees) between neighboring projection points ')
        form.addParam('ccLevel', FloatParam, default=0.95,
                      expertLevel=LEVEL_ADVANCED,
                      label="correlation level for validation",
                      help='threshold correlation to be used in validation. Keep this value in the range [0.90 -- 0.97]')
        form.addParam('minTilt', FloatParam, default=0,
                      expertLevel=LEVEL_ADVANCED,
                      label="Minimum allowed tilt angle",
                      help='Tilts below this value will not be considered for the alignment')
        form.addParam('maxTilt', FloatParam, default=180,
                      expertLevel=LEVEL_ADVANCED,
                      label="Maximum allowed tilt angle without mirror check",
                      help='Tilts above this value will not be considered for the alignment without mirror check')
        form.addParam('maximumTargetResolution', FloatParam, default=8,
                      label='Target resolution (A)',
                      help='Low pass filter the particles to this resolution. This usually helps a lot obtaining good alignment. You should have a good'
                           ' reason to modify this value outside the range  [8-10] A')

        form.addParallelSection(threads=1, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self.imgsFn = self._getExtraPath('images.xmd')
        self._insertFunctionStep('convertInputStep',
                                             self.inputParticles.get().getObjId())
        self.TsOrig = self.inputParticles.get().getSamplingRate()
        self._insertFunctionStep('doIteration000')
        self._maximumTargetResolution = self.maximumTargetResolution.get()
        self._insertFunctionStep('globalAssignment')  
        self._insertFunctionStep('cleanDirectory') 
        self._insertFunctionStep("createOutput")        

    def convertInputStep(self, inputParticlesId):
        writeSetOfParticles(self.inputParticles.get(), self.imgsFn)
        self.runJob('xmipp_metadata_utilities', '-i %s --fill image1 constant noImage' % self.imgsFn, numberOfMpi=1)
        self.runJob('xmipp_metadata_utilities', '-i %s --operate modify_values "image1=image"' % self.imgsFn, numberOfMpi=1)
        self.runJob('xmipp_metadata_utilities', '-i %s --fill particleId constant 1' % self.imgsFn, numberOfMpi=1)
        self.runJob('xmipp_metadata_utilities', '-i %s --operate modify_values "particleId=itemId"' % self.imgsFn, numberOfMpi=1)
        imgsFnId = self._getExtraPath('imagesId.xmd')
        self.runJob('xmipp_metadata_utilities', '-i %s --operate keep_column particleId -o %s' % (self.imgsFn, imgsFnId), numberOfMpi=1)

    def createOutput(self):
        fnLastDir = self._getExtraPath("Iter1") 
        fnDirGlobal = fnLastDir
        Ts = self.readInfoField(fnDirGlobal, "sampling", emlib.MDL_SAMPLINGRATE)

        fnAnglesDisc=join(fnDirGlobal,"anglesDisc.xmd")
        if exists(fnAnglesDisc):
            fnAngles = self._getPath("angles.xmd")
            self.runJob('xmipp_metadata_utilities','-i %s -o %s --operate modify_values "image=image1"'%(fnAnglesDisc,fnAngles), numberOfMpi=1)
            self.runJob('xmipp_metadata_utilities','-i %s --operate sort particleId'%fnAngles,numberOfMpi=1)
            self.runJob('xmipp_metadata_utilities','-i %s --operate drop_column image1'%fnAngles,numberOfMpi=1)
            self.runJob('xmipp_metadata_utilities','-i %s --operate modify_values "itemId=particleId"'%fnAngles,numberOfMpi=1)
            imgSet=self.inputParticles.get()
            self.scaleFactor = Ts/imgSet.getSamplingRate()
            outImgSet= self._createSetOfParticles()
            outImgSet.copyInfo(imgSet)
            outImgSet.setAlignmentProj()
            outImgSet.setIsPhaseFlipped(imgSet.isPhaseFlipped())
            self.iterMd = md.iterRows(fnAngles, md.MDL_PARTICLE_ID)
            self.lastRow = next(self.iterMd)
            outImgSet.copyItems(imgSet,
                                updateItemCallback=self._updateItem)
            self._defineOutputs(outputParticles=outImgSet)
            self._defineSourceRelation(self.inputParticles, outImgSet)
            
            mdParticles = emlib.MetaData(fnAngles)
            ccGraphPrevList = mdParticles.getColumnValues(emlib.MDL_GRAPH_CC_PREVIOUS)
            ccAsigDirRefList = mdParticles.getColumnValues(emlib.MDL_ASSIGNED_DIR_REF_CC)
            angDistPrevList = mdParticles.getColumnValues(emlib.MDL_GRAPH_DISTANCE2MAX_PREVIOUS)
            
            # threshold            
            th_ccGraph = self.otsu(ccGraphPrevList)
            th_ccAsigDir = self.otsu(ccAsigDirRefList)
            th_angDist = self.otsu(angDistPrevList)
            if (th_ccGraph < 0.50) and (th_ccAsigDir < 0.50):
                print("applying default thresholds")
                th_ccGraph = 0.95
                th_ccAsigDir = 0.95
                th_angDist = 3.0 * self.angularSampling.get() 
                print(th_ccGraph, th_ccAsigDir, th_angDist)
            
            print("\033[1;33Thresholds:\033[0;33 \n")
            print('correlation with projection in Graph max direction:',th_ccGraph,
                  '\ncorrelation with assigned projection:',th_ccAsigDir,
                  '\nangular distance to maxGraph:',th_angDist)
            
            # a second output set
            fnOutParticles = self._getPath('anglesAux.xmd')
            copy(fnAngles, fnOutParticles)
            mdParticles = emlib.MetaData(fnOutParticles)
            n_false = 0;
            
            nParticles = 0
            for row in iterRows(mdParticles):
                nParticles += 1
                objId = row.getObjId()
                assigDirRefCC = row.getValue(emlib.MDL_ASSIGNED_DIR_REF_CC)
                graphCCPrev = row.getValue(emlib.MDL_GRAPH_CC_PREVIOUS)
                graphDistMaxGraphPrev = row.getValue(emlib.MDL_GRAPH_DISTANCE2MAX_PREVIOUS)

                if (assigDirRefCC < th_ccAsigDir) and ( graphCCPrev < th_ccGraph ) and (graphDistMaxGraphPrev > th_angDist):
                    n_false += 1
                    mdParticles.setValue(emlib.MDL_ENABLED, -1, objId)
            mdParticles.write(fnOutParticles) 
            print('to be disabled:',n_false)          
            self.subsets = []
            i=0
            self.subsets.append(self._createSetOfParticles(str(i)))
            self.subsets[i].copyInfo(self.inputParticles.get())
            readSetOfParticles(fnOutParticles, self.subsets[i])
            result = {'outputParticlesAux' : self.subsets[i]}
            self._defineOutputs(**result)
            self._store(self.subsets[i])

            # extra output info
            p_gsp = (1 - n_false/nParticles)* 100
            self.percentage = Float(p_gsp)
            self._store()
            
    def _updateItem(self, particle, row):
        count = 0
        while self.lastRow and particle.getObjId() == self.lastRow.getValue(md.MDL_PARTICLE_ID):
            count += 1
            if count:
                self._createItemMatrix(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None

        particle._appendItem = count > 0
        
    def _createItemMatrix(self, particle, row):
        row.setValue(emlib.MDL_SHIFT_X, row.getValue(emlib.MDL_SHIFT_X)*self.scaleFactor)
        row.setValue(emlib.MDL_SHIFT_Y, row.getValue(emlib.MDL_SHIFT_Y)*self.scaleFactor)
        
        setXmippAttributes(particle, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT,
                           md.MDL_ANGLE_PSI, md.MDL_SHIFT_X, md.MDL_SHIFT_Y,
                           md.MDL_MAXCC, md.MDL_WEIGHT,
                           md.MDL_MAXCC_PREVIOUS, md.MDL_GRAPH_CC_PREVIOUS,
                           md.MDL_ASSIGNED_DIR_REF_CC, md.MDL_GRAPH_DISTANCE2MAX_PREVIOUS,
                           md.MDL_GRAPH_CC, md.MDL_GRAPH_DISTANCE2MAX)
        
        createItemMatrix(particle, row, align=ALIGN_PROJ)

    def doIteration000(self):
        fnDirCurrent = self._getExtraPath('Iter0')
        makePath(fnDirCurrent)

        # Get volume sampling rate
        if self.inputVolume.get() is None:
            TsCurrent = self.inputParticles.get().getSamplingRate()
        else:
            TsCurrent = self.inputVolume.get().getSamplingRate()
        self.writeInfoField(fnDirCurrent, "sampling", emlib.MDL_SAMPLINGRATE, TsCurrent)

        # Copy reference volume and window if necessary
        XDim = self.inputParticles.get().getDimensions()[0]
        newXDim = int(round(XDim * self.TsOrig / TsCurrent))
        self.writeInfoField(fnDirCurrent, "size", emlib.MDL_XSIZE, newXDim)
        
        img = ImageHandler()

        fnVol1 = join(fnDirCurrent, "volume.vol")  

        vol = self.inputVolume.get()
        img.convert(vol, fnVol1)
        volXdim = vol.getDim()[0]
        if newXDim != volXdim:
            self.runJob('xmipp_transform_window', "-i %s --size %d" % (fnVol1, newXDim), numberOfMpi=1)

    def globalAssignment(self):  
        fnDirIter0 = self._getExtraPath("Iter0") 
        fnGlobal = self._getExtraPath("Iter1")
        makePath(fnGlobal)
        previousResolution = self._maximumTargetResolution
        targetResolution = self._maximumTargetResolution
        TsCurrent = max(self.TsOrig,targetResolution/3)
        getShiftsFrom = ''
        self.prepareImages(fnDirIter0, fnGlobal, TsCurrent, getShiftsFrom)
        TsCurrent = self.readInfoField(fnGlobal, "sampling", emlib.MDL_SAMPLINGRATE)  # Prepare images may have changed it
        self.prepareReferences(fnDirIter0, fnGlobal, TsCurrent, targetResolution)
        
        # Calculate angular step at this resolution
        ResolutionAlignment = previousResolution
        self.nextLowPass = True
        self.nextResolutionOffset = 2
        if self.nextLowPass:
            ResolutionAlignment += self.nextResolutionOffset
        newXDim = self.readInfoField(fnGlobal, "size", emlib.MDL_XSIZE)
        angleStep = self.calculateAngStep(newXDim, TsCurrent, ResolutionAlignment)
        angleStep = max(angleStep, self.angularSampling.get())
        self.writeInfoField(fnGlobal, "angleStep", emlib.MDL_ANGLE_DIFF, float(angleStep))

        # Global alignment
        self.numberOfPerturbations = 1
        perturbationList = [chr(x) for x in range(ord('a'), ord('a') + self.numberOfPerturbations)]

        fnDirAssignment = join(fnGlobal, "assignment")
        fnImgs = join(fnGlobal, "images.xmd")
        makePath(fnDirAssignment)

        # Create defocus groups
        row = getFirstRow(fnImgs)
        if row.containsLabel(emlib.MDL_CTF_MODEL) or row.containsLabel(emlib.MDL_CTF_DEFOCUSU):
            self.runJob("xmipp_ctf_group", "--ctfdat %s -o %s/ctf:stk --pad 1.0 --sampling_rate %f --phase_flipped  --error 0.1 --resol %f" % \
                        (fnImgs, fnDirAssignment, TsCurrent, targetResolution), numberOfMpi=1)
            moveFile("%s/ctf_images.sel" % fnDirAssignment, "%s/ctf_groups.xmd" % fnDirAssignment)
            cleanPath("%s/ctf_split.doc" % fnDirAssignment)
            mdInfo = emlib.MetaData("numberGroups@%s" % join(fnDirAssignment, "ctfInfo.xmd"))
            numberGroups = mdInfo.getValue(emlib.MDL_COUNT, mdInfo.firstObject())
            ctfPresent = True
        else:
            numberGroups = 1
            ctfPresent = False
            
        # Generate projections
        fnReferenceVol = join(fnGlobal, "volumeRef.vol") 
        for subset in perturbationList:
            fnGallery = join(fnDirAssignment, "gallery%s.stk" % subset)
            fnGalleryMd = join(fnDirAssignment, "gallery%s.xmd" % subset)
            args = "-i %s -o %s --sampling_rate %f --sym %s --min_tilt_angle %f --max_tilt_angle %f --perturb %f " % \
                   (fnReferenceVol, fnGallery, angleStep, self.symmetryGroup, self.minTilt.get(), self.maxTilt.get(), math.sin(angleStep * math.pi / 180.0) / 4)
            args += " --compute_neighbors --angular_distance -1 --experimental_images %s" % self._getExtraPath("images.xmd")
            self.runJob("xmipp_angular_project_library", args, numberOfMpi = self.numberOfMpi.get() * self.numberOfThreads.get())
            cleanPath(join(fnDirAssignment, "gallery_angles%s.doc" % subset))
            moveFile(join(fnDirAssignment, "gallery%s.doc" % subset), fnGalleryMd)
            fnAngles = join(fnGlobal, "anglesDisc%s.xmd" % subset)
            for j in range(1, numberGroups + 1):
                fnAnglesGroup = join(fnDirAssignment, "angles_group%03d%s.xmd" % (j, subset))
                if not exists(fnAnglesGroup):
                    if ctfPresent:
                        fnGroup = "ctfGroup%06d@%s/ctf_groups.xmd" % (j, fnDirAssignment)
                        fnCTF = "%d@%s/ctf_ctf.stk" % (j, fnDirAssignment)
                        fnGalleryGroup = join(fnDirAssignment, "gallery%s_%06d.stk" % (subset, j))
                        fnGalleryGroupMd = join(fnDirAssignment, "gallery%s_%06d.xmd" % (subset, j))
                        self.runJob("xmipp_transform_filter", "-i %s -o %s --fourier binary_file %s --save_metadata_stack %s --keep_input_columns" % \
                                    (fnGalleryMd, fnGalleryGroup, fnCTF, fnGalleryGroupMd),
                                    numberOfMpi=min(self.numberOfMpi.get(), 24))
                    else:
                        fnGroup = fnImgs
                        fnGalleryGroupMd = fnGalleryMd
                    if getSize(fnGroup) == 0:  # If the group is empty
                        continue
                    self.angularMaxShift = 10
                    maxShift = round(self.angularMaxShift * newXDim / 100)
                    R = self.inputParticles.get().getDimensions()[0] / 2
                    R = R * self.TsOrig / TsCurrent    
                    args = '-i %s -o %s -ref %s -sampling %f -odir %s --Nsimultaneous %d -angleStep %f --maxShift %f --sym %s --useForValidation --refVol %s' % \
                        (fnGroup, fnAnglesGroup, fnGalleryGroupMd, TsCurrent, fnDirAssignment, self.numberOfMpi.get() * self.numberOfThreads.get(), angleStep, maxShift, self.symmetryGroup, fnReferenceVol)
                    self.runJob('xmipp_angular_assignment_mag', args, numberOfMpi=self.numberOfMpi.get() * self.numberOfThreads.get())
                    if exists(fnAnglesGroup):
                        if not exists(fnAngles) and exists(fnAnglesGroup):
                            copyFile(fnAnglesGroup, fnAngles)
                        else:
                            if exists(fnAngles) and exists(fnAnglesGroup):
                                self.runJob("xmipp_metadata_utilities", "-i %s --set union_all %s" % (fnAngles, fnAnglesGroup), numberOfMpi=1)
            if exists(fnAngles) and exists(fnImgs):
                self.runJob("xmipp_metadata_utilities", "-i %s --set join %s image" % (fnAngles, fnImgs), numberOfMpi=1)
            self.saveSpace = True
            if self.saveSpace and ctfPresent:
                self.runJob("rm -f", fnDirAssignment + "/gallery*", numberOfMpi=1)  
                
        # Evaluate the stability of the alignment
        fnOut = join(fnGlobal, "anglesDisc")
        for set1 in perturbationList:
            fnOut1 = join(fnGlobal, "anglesDisc%s" % set1)
            fnAngles1 = fnOut1 + ".xmd"
            counter2 = 0
            for set2 in perturbationList:
                if set1 == set2:
                    continue
                fnAngles2 = join(fnGlobal, "anglesDisc%s.xmd" % set2)
                fnOut12 = join(fnGlobal, "anglesDisc%s%s" % (set1, set2))
                self.runJob("xmipp_angular_distance", "--ang1 %s --ang2 %s --oroot %s --sym %s --compute_weights 1 particleId 0.5 --check_mirrors --set 0" % (fnAngles2, fnAngles1, fnOut12, self.symmetryGroup), numberOfMpi=1)
                self.runJob("xmipp_metadata_utilities", '-i %s --operate keep_column "angleDiff0 shiftDiff0 weightJumper0"' % (fnOut12 + "_weights.xmd"), numberOfMpi=1)
                if counter2 == 0:
                    mdAllWeights = emlib.MetaData(fnOut12 + "_weights.xmd")
                    counter2 = 1
                else:
                    mdWeight = emlib.MetaData(fnOut12 + "_weights.xmd")
                    if mdWeight.size() == mdAllWeights.size():
                        counter2 += 1
                        for id1, id2 in izip(mdWeight, mdAllWeights):
                            angleDiff0 = mdWeight.getValue(emlib.MDL_ANGLE_DIFF0, id1)
                            shiftDiff0 = mdWeight.getValue(emlib.MDL_SHIFT_DIFF0, id1)
                            weightJumper0 = mdWeight.getValue(emlib.MDL_WEIGHT_JUMPER0, id1)

                            angleDiff0All = mdAllWeights.getValue(emlib.MDL_ANGLE_DIFF0, id2)
                            shiftDiff0All = mdAllWeights.getValue(emlib.MDL_SHIFT_DIFF0, id2)
                            weightJumper0All = mdAllWeights.getValue(emlib.MDL_WEIGHT_JUMPER0, id2)

                            mdAllWeights.setValue(emlib.MDL_ANGLE_DIFF0, angleDiff0 + angleDiff0All, id2)
                            mdAllWeights.setValue(emlib.MDL_SHIFT_DIFF0, shiftDiff0 + shiftDiff0All, id2)
                            mdAllWeights.setValue(emlib.MDL_WEIGHT_JUMPER0, weightJumper0 + weightJumper0All, id2)
            if counter2 > 1:
                iCounter2 = 1.0 / counter2
                for id in mdAllWeights:
                    angleDiff0All = mdAllWeights.getValue(emlib.MDL_ANGLE_DIFF0, id)
                    shiftDiff0All = mdAllWeights.getValue(emlib.MDL_SHIFT_DIFF0, id)
                    weightJumper0All = mdAllWeights.getValue(emlib.MDL_WEIGHT_JUMPER0, id)

                    mdAllWeights.setValue(emlib.MDL_ANGLE_DIFF0, angleDiff0All * iCounter2, id)
                    mdAllWeights.setValue(emlib.MDL_SHIFT_DIFF0, shiftDiff0All * iCounter2, id)
                    mdAllWeights.setValue(emlib.MDL_WEIGHT_JUMPER0, weightJumper0All * iCounter2, id)
            if counter2 > 0:
                mdAllWeights.write(fnOut1 + "_weights.xmd")
                self.runJob("xmipp_metadata_utilities", '-i %s --set merge %s' % (fnAngles1, fnOut1 + "_weights.xmd"), numberOfMpi=1)
            if not exists(fnOut + ".xmd") and exists(fnAngles1):
                copyFile(fnAngles1, fnOut + ".xmd")
            else:
                if exists(fnAngles1) and exists(fnOut + ".xmd"):
                    self.runJob("xmipp_metadata_utilities", '-i %s --set union_all %s' % (fnOut + ".xmd", fnAngles1), numberOfMpi=1)
        cleanPath(join(fnGlobal, "anglesDisc*_weights.xmd"))    

    def calculateAngStep(self, newXDim, TsCurrent, ResolutionAlignment):
        k = newXDim * TsCurrent / ResolutionAlignment  # Freq. index
        return math.atan2(1, k) * 180.0 / math.pi  # Corresponding angular step

    def checkInfoField(self, fnDir, block):
        fnInfo = join(fnDir, "iterInfo.xmd")
        if not exists(fnInfo):
            return False
        blocks = emlib.getBlocksInMetaDataFile(fnInfo)
        return block in blocks
            
    def writeInfoField(self, fnDir, block, label, value):
        mdInfo = emlib.MetaData()
        objId = mdInfo.addObject()
        mdInfo.setValue(label, value, objId)
        mdInfo.write("%s@%s" % (block, join(fnDir, "iterInfo.xmd")), emlib.MD_APPEND)
        
    def readInfoField(self, fnDir, block, label):
        mdInfo = emlib.MetaData("%s@%s" % (block, join(fnDir, "iterInfo.xmd")))
        return mdInfo.getValue(label, mdInfo.firstObject())    
    
    def prepareImages(self, fnDirPrevious, fnDir, TsCurrent, getShiftsFrom=''):
        if self.checkInfoField(fnDir, "count"):
            state = self.readInfoField(fnDir, "count", emlib.MDL_COUNT)
            if state >= 1:
                return        
        print("Preparing images to sampling rate=", TsCurrent)
        XDim = self.inputParticles.get().getDimensions()[0]
        newXDim = int(round(XDim * self.TsOrig / TsCurrent))
        if newXDim < 40:
            newXDim = int(40)
            TsCurrent = XDim * (self.TsOrig / newXDim)
        elif newXDim % 2 == 1:
            newXDim += 1
            TsCurrent = XDim * (self.TsOrig / newXDim)
        self.writeInfoField(fnDir, "sampling", emlib.MDL_SAMPLINGRATE, TsCurrent)
        self.writeInfoField(fnDir, "size", emlib.MDL_XSIZE, newXDim)
        
        # Prepare particles
        fnNewParticles = join(fnDir, "images.stk")
        if newXDim != XDim:
            self.runJob("xmipp_image_resize", "-i %s -o %s --fourier %d" % (self.imgsFn, fnNewParticles, newXDim), numberOfMpi=min(self.numberOfMpi.get(), 24))
        else:
            self.runJob("xmipp_image_convert", "-i %s -o %s --save_metadata_stack %s" % (self.imgsFn, fnNewParticles, join(fnDir, "images.xmd")),
                        numberOfMpi=1)
        R = self.inputParticles.get().getDimensions()[0] / 2
        self.angularMaxShift = 10
        R = min(round(R * self.TsOrig / TsCurrent * (1 + self.angularMaxShift * 0.01)), newXDim / 2)
        self.runJob("xmipp_transform_mask", "-i %s --mask circular -%d" % (fnNewParticles, R), numberOfMpi=min(self.numberOfMpi.get(), 24))
        fnSource = join(fnDir, "images.xmd")

        if not self.inputParticles.get().isPhaseFlipped():
            self.runJob("xmipp_ctf_correct_phase", "-i %s --sampling_rate %f" % (fnSource, TsCurrent),
                        numberOfMpi=min(self.numberOfMpi.get(), 24))

        self.writeInfoField(fnDir, "count", emlib.MDL_COUNT, int(1))   
        
    def prepareReferences(self, fnDirPrevious, fnDir, TsCurrent, targetResolution):
        if self.checkInfoField(fnDir, "count"):
            state = self.readInfoField(fnDir, "count", emlib.MDL_COUNT)
            if state >= 2:
                return

        print("Preparing references to sampling rate=", TsCurrent)
        fnMask = ''
        newXDim = self.readInfoField(fnDir, "size", emlib.MDL_XSIZE)
        oldXdim = self.readInfoField(fnDirPrevious, "size", emlib.MDL_XSIZE)
        fnPreviousVol = join(fnDirPrevious, "volume.vol") 
        fnReferenceVol = join(fnDir, "volumeRef.vol") 
        if oldXdim != newXDim:
            self.runJob("xmipp_image_resize", "-i %s -o %s --dim %d" % (fnPreviousVol, fnReferenceVol, newXDim), numberOfMpi=1)
        else:
            copyFile(fnPreviousVol, fnReferenceVol)
        self.nextLowPass = True
        if self.nextLowPass:
            self.nextResolutionOffset = 2  # Resolution offset (A)
            self.runJob('xmipp_transform_filter', '-i %s --fourier low_pass %f --sampling %f' % \
                        (fnReferenceVol, targetResolution + self.nextResolutionOffset, TsCurrent), numberOfMpi=1)
        self.nextSpherical = True
        if self.nextSpherical:
            R = self.inputParticles.get().getDimensions()[0] / 2
            self.runJob('xmipp_transform_mask', '-i %s --mask circular -%d' % \
                        (fnReferenceVol, round(R * self.TsOrig / TsCurrent)), numberOfMpi=1)
        self.nextPositivity = True
        if self.nextPositivity:
            self.runJob('xmipp_transform_threshold', '-i %s --select below 0 --substitute value 0' % fnReferenceVol, numberOfMpi=1)

        if fnMask != '':
            cleanPath(fnMask)
        self.writeInfoField(fnDir, "count", emlib.MDL_COUNT, int(2)) 

    def cleanDirectory(self):
        fnDirCurrent = self._getExtraPath("Iter1") 
        if self.saveSpace:
            fnGlobal = fnDirCurrent
            if exists(fnGlobal):
                cleanPath(join(fnGlobal, "images.stk"))
            if exists(fnGlobal):
                cleanPath(join(fnGlobal, "images.xmd"))
                cleanPath(join(fnGlobal, "volumeRef.vol"))

    # --------------------------- INFO functions --------------------------------------------
    def _validate(self):
        validateMsgs = []
        # if there are Volume references, it cannot be empty.
        if self.inputVolume.get() and not self.inputVolume.hasValue():
            validateMsgs.append('Please provide an input reference volume.')
        if self.inputParticles.get() and not self.inputParticles.hasValue():
            validateMsgs.append('Please provide input particles.')
        return validateMsgs
    
    def _summary(self):
        summary = [
            "Input particles:  %s" % self.inputParticles.get().getNameId()]
        summary.append("-----------------")
        
        if (not hasattr(self, 'outputParticles')):
            summary.append("Output not ready yet.")
        else:
            if self.percentage == 100:
                text = 'After validation, all of the images are likely to be within the realiable assignment zone'
            else:
                text = 'After validation, a %.2f' % self.percentage
                text += r'% of the images are likely to be within the reliable assignment zone'
            summary.append(text)
        return summary    

    def _methods(self):
        messages = []
        if (hasattr(self, 'outputParticles')):
            messages.append('correlation values of previous alignment process'
                            ' are modified according to the spherical distance from'
                            ' the assigned direction to a soft and high-valued correlation zone'
                            ' of neighboring projection directions')
        return messages
    
    def otsu(self, ccList):
        # from golden-highres
        import numpy as np

        ccNumber = len(ccList)
        meanWeight = 1.0 / ccNumber
        his, bins = np.histogram(ccList, int((max(ccList)-min(ccList))/0.01))
        finalThres = -1
        finalVal = -1
        ccArange = np.arange(min(ccList),max(ccList),0.01)
        if len(ccArange)==(len(his)+1):
            ccArange=ccArange[:-1]
        for t, j in enumerate(bins[1:-1]):
            idx = t+1
            pcb = np.sum(his[:idx])
            pcf = np.sum(his[idx:])
            Wb = pcb * meanWeight
            Wf = pcf * meanWeight

            mub = np.sum(ccArange[:idx] * his[:idx]) / float(pcb)
            muf = np.sum(ccArange[idx:] * his[idx:]) / float(pcf)
            value = Wb * Wf * (mub - muf) ** 2

            if value > finalVal:
                finalThres = bins[idx]
                finalVal = value

        return finalThres