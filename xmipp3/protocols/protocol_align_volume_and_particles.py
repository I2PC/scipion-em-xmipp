# **************************************************************************
# *
# * Authors:     ajimenez@cnb.csic.es
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

import numpy as np
import pyworkflow.protocol.params as params
import pyworkflow.em as em
from ..convert import writeSetOfParticles, readSetOfParticles
from pyworkflow.em.convert import ImageHandler
from xmipp3.utils import writeInfoField, readInfoField
from pyworkflow.utils import moveFile
import pyworkflow.em.metadata as md
from pyworkflow.em.data import Transform
from pyworkflow.em import ALIGN_PROJ
from xmipp3.convert import rowToAlignment, alignmentToRow, setXmippAttributes
from xmipp3.constants import SYM_URL
import xmippLib
from shutil import copy

ALIGN_MASK_CIRCULAR = 0
ALIGN_MASK_BINARY_FILE = 1

ALIGN_ALGORITHM_EXHAUSTIVE = 0
ALIGN_ALGORITHM_LOCAL = 1
ALIGN_ALGORITHM_EXHAUSTIVE_LOCAL = 2
ALIGN_ALGORITHM_FAST_FOURIER = 3


class XmippProtAlignVolumeParticles(em.ProtAlignVolume):
    """ 
    Aligns a set of volumes using cross correlation 
    or a Fast Fourier method. 
    
    *Note:* Fast Fourier requires compilation of Xmipp with --cltomo flag
     """
    _label = 'align volume and particles'
    nVols = 0
    
    def __init__(self, **args):
        em.ProtAlignVolume.__init__(self, **args)
        self.stepsExecutionMode = em.STEPS_PARALLEL
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Volume parameters')
        form.addParam('inputReference', params.PointerParam, pointerClass='Volume', 
                      label="Reference volume", important=True, 
                      help='Reference volume to be used for the alignment.')    
        form.addParam('inputVolumes', params.PointerParam, pointerClass='Volume',  
                      label="Input volume", important=True, 
                      help='Select one volume to be aligned against the reference volume.')
        form.addParam('refParticles', params.PointerParam, pointerClass='SetOfParticles',  
                      label="Reference particles", important=True, 
                      help='Reference set of particles to be used for the alignment.')
        form.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles',  
                      label="Input particles", important=True, 
                      help='Select one set of particles to be aligned against the reference set of particles using the transformation calculated with the reference and input volumes.')
        form.addParam('targetResolution', params.FloatParam, label="Target resolution", default=3.0,
            help="In Angstroms, the images and the volumes are rescaled so that this resolution is at "
                 "2/3 of the Fourier spectrum.")
        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry group",
                      help='See %s page for a description of the symmetries '
                           'accepted by Xmipp' % SYM_URL)
        
        group1 = form.addGroup('Mask')
        group1.addParam('applyMask', params.BooleanParam, default=False, 
                      label='Apply mask?',
                      help='Apply a 3D Binary mask to the volumes')
        group1.addParam('maskType', params.EnumParam, choices=['circular','binary file'], default=ALIGN_MASK_CIRCULAR, 
                      label='Mask type', display=params.EnumParam.DISPLAY_COMBO, condition='applyMask',
                      help='Select the type of mask you want to apply')
        group1.addParam('maskRadius', params.IntParam, default=-1, condition='applyMask and maskType==%d' % ALIGN_MASK_CIRCULAR,
                      label='Mask radius', 
                      help='Insert the radius for the mask')
        group1.addParam('maskFile', params.PointerParam, condition='applyMask and maskType==%d' % ALIGN_MASK_BINARY_FILE,
                      pointerClass='VolumeMask', label='Mask file', 
                      help='Select the volume mask object')
        
        form.addSection(label='Search strategy')
        form.addParam('alignmentAlgorithm', params.EnumParam, default=ALIGN_ALGORITHM_EXHAUSTIVE, 
                      choices=['exhaustive',
                               'local', 
                               'exhaustive + local', 
                               'fast fourier'], 
                      label='Alignment algorithm', display=params.EnumParam.DISPLAY_COMBO,
                      help='Exhaustive searches all possible combinations within a search space.'
                            'Local searches around a given position.'
                            'Be aware that the Fast Fourier algorithm requires a special compilation'
                            'of Xmipp (--cltomo flag). It performs the same job as the  '
                            'exhaustive method but much faster.')
        
        anglesCond = 'alignmentAlgorithm!=%d' % ALIGN_ALGORITHM_LOCAL
        
        group = form.addGroup('Angles range', condition=anglesCond, expertLevel=params.LEVEL_ADVANCED)
        
        line = group.addLine('Rotational angle (deg)')
        line.addParam('minRotationalAngle', params.FloatParam, default=0, label='Min')
        line.addParam('maxRotationalAngle', params.FloatParam, default=360, label='Max')
        line.addParam('stepRotationalAngle', params.FloatParam, default=5, label='Step')
        
        line = group.addLine('Tilt angle (deg)', expertLevel=params.LEVEL_ADVANCED)        
        line.addParam('minTiltAngle', params.FloatParam, default=0, label='Min')        
        line.addParam('maxTiltAngle', params.FloatParam, default=180, label='Max')
        line.addParam('stepTiltAngle', params.FloatParam, default=5, label='Step')
        
        line = group.addLine('Inplane angle (deg)', expertLevel=params.LEVEL_ADVANCED)        
        line.addParam('minInplaneAngle', params.FloatParam, default=0, label='Min')        
        line.addParam('maxInplaneAngle', params.FloatParam, default=360, label='Max')
        line.addParam('stepInplaneAngle', params.FloatParam, default=5, label='Step')       
        
        group = form.addGroup('Shifts range', condition=anglesCond, expertLevel=params.LEVEL_ADVANCED)
        line = group.addLine('Shift X (px)')        
        line.addParam('minimumShiftX', params.FloatParam, default=0, label='Min')        
        line.addParam('maximumShiftX', params.FloatParam, default=0, label='Max')
        line.addParam('stepShiftX', params.FloatParam, default=1, label='Step') 
        
        line = group.addLine('Shift Y (px)', expertLevel=params.LEVEL_ADVANCED)        
        line.addParam('minimumShiftY', params.FloatParam, default=0, label='Min')        
        line.addParam('maximumShiftY', params.FloatParam, default=0, label='Max')
        line.addParam('stepShiftY', params.FloatParam, default=1, label='Step')        
        
        line = group.addLine('Shift Z (px)', expertLevel=params.LEVEL_ADVANCED)        
        line.addParam('minimumShiftZ', params.FloatParam, default=0, label='Min')        
        line.addParam('maximumShiftZ', params.FloatParam, default=0, label='Max')
        line.addParam('stepShiftZ', params.FloatParam, default=1, label='Step')         
        
        line = form.addLine('Scale ', expertLevel=params.LEVEL_ADVANCED, condition=anglesCond)        
        line.addParam('minimumScale', params.FloatParam, default=1, label='Min')        
        line.addParam('maximumScale', params.FloatParam, default=1, label='Max')
        line.addParam('stepScale', params.FloatParam, default=0.005, label='Step')          
                        
        group = form.addGroup('Initial values', 
                              condition='alignmentAlgorithm==%d' % ALIGN_ALGORITHM_LOCAL, 
                              expertLevel=params.LEVEL_ADVANCED)
        line = group.addLine('Initial angles')        
        line.addParam('initialRotAngle', params.FloatParam, default=0, label='Rot')        
        line.addParam('initialTiltAngle', params.FloatParam, default=0, label='Tilt')
        line.addParam('initialInplaneAngle', params.FloatParam, default=0, label='Psi') 

        line = group.addLine('Initial shifts ', expertLevel=params.LEVEL_ADVANCED)        
        line.addParam('initialShiftX', params.FloatParam, default=0, label='X')        
        line.addParam('initialShiftY', params.FloatParam, default=0, label='Y')
        line.addParam('initialShiftZ', params.FloatParam, default=0, label='Z')    
        
        group.addParam('optimizeScale', params.BooleanParam, default=False, expertLevel=params.LEVEL_ADVANCED,
                      label='Optimize scale',
                      help='Choose YES if you want to optimize the scale of input volume/s based on the reference')
        group.addParam('initialScale', params.FloatParam, default=1, expertLevel=params.LEVEL_ADVANCED, condition='optimizeScale',
                      label='Initial scale')  
        
        form.addParallelSection(threads=8, mpi=1)
        
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):

        #Some definitions of filenames
        self.fnRefVol = self._getExtraPath("refVolume.vol")
        self.fnInputVol = self._getExtraPath("inputVolume.vol")
        self.imgsRefFn = self._getExtraPath("referenceParticles.xmd")
        self.imgsInputFn = self._getExtraPath("inputParticles.xmd")

        # Iterate through all input volumes and align them 
        # againt the reference volume
        maskArgs = self._getMaskArgs()
        alignArgs = self._getAlignArgs()
        alignSteps = []

        stepId0 = self._insertFunctionStep('convertStep', prerequisites=[])
        alignSteps.append(stepId0) 
        stepId1 = self._insertFunctionStep('alignVolumeStep', maskArgs, 
                                      alignArgs, prerequisites=alignSteps)
        alignSteps.append(stepId1)     
        stepId2 = self._insertFunctionStep('alignParticlesStep', prerequisites=alignSteps)
        alignSteps.append(stepId2)
        stepId3 = self._insertFunctionStep('compareAlignStep', self.symmetryGroup, prerequisites=alignSteps)
        alignSteps.append(stepId3)

        self._insertFunctionStep('createOutputStep', prerequisites=alignSteps)
        
    #--------------------------- STEPS functions --------------------------------------------


    def convertStep(self):
        
        #Resizing inputReference volume
        ih = ImageHandler()
        ih.convert(self.inputReference.get(), self.fnRefVol)
        Xdim = self.inputReference.get().getDim()[0]
        Ts = self.inputReference.get().getSamplingRate()
        newTs = self.targetResolution.get() * 1.0/3.0
        newTs = max(Ts, newTs)
        self.newXdim = long(Xdim * Ts / newTs)
        writeInfoField(self._getExtraPath(), "sampling", xmippLib.MDL_SAMPLINGRATE, newTs)
        writeInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE, self.newXdim)
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize","-i %s --dim %d"%(self.fnRefVol, self.newXdim), numberOfMpi=1)

        #Resizing inputVolumes
        ih = ImageHandler()
        ih.convert(self.inputVolumes.get(), self.fnInputVol)
        Xdim = self.inputVolumes.get().getDim()[0]
        if Xdim != self.newXdim:
            self.runJob("xmipp_image_resize","-i %s --dim %d"%(self.fnInputVol, self.newXdim), numberOfMpi=1)

        #Resizing refParts
        refParts = self.refParticles.get()
        writeSetOfParticles(refParts, self.imgsRefFn)
        Xdim = refParts.getXDim()
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (self.imgsRefFn,
                         self._getExtraPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         self.newXdim))
            moveFile(self._getExtraPath('scaled_particles.xmd'), self.imgsRefFn)

        #Resizing inputParts
        inputParts = self.inputParticles.get()
        writeSetOfParticles(inputParts, self.imgsInputFn)
        Xdim = inputParts.getXDim()
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (self.imgsRefFn,
                         self._getExtraPath('scaled_particles.stk'),
                         self._getExtraPath('scaled_particles.xmd'),
                         self.newXdim))
            moveFile(self._getExtraPath('scaled_particles.xmd'), self.imgsInputFn)


    def alignVolumeStep(self, maskArgs, alignArgs):  

        fhInputTranMat = self._getExtraPath('transformation-matrix.txt')
        outVolFn = self._getExtraPath("inputVolumeAligned.vol")
      
        args = "--i1 %s --i2 %s --apply %s" % (self.fnRefVol, self.fnInputVol, outVolFn)
        args += maskArgs
        args += alignArgs
        args += " --copyGeo %s" % fhInputTranMat        
        self.runJob("xmipp_volume_align", args)

        if self.alignmentAlgorithm == ALIGN_ALGORITHM_EXHAUSTIVE_LOCAL:
            args = "--i1 %s --i2 %s --apply --local" % (self.fnRefVol, outVolFn)
            args += " --copyGeo %s" % fhInputTranMat
            self.runJob("xmipp_volume_align", args)
	

    def alignParticlesStep(self):

        fhInputTranMat = self._getExtraPath('transformation-matrix.txt')
        outParticlesFn = self._getExtraPath('outputParticles.xmd')
        transMatFromFile = np.loadtxt(fhInputTranMat)
        transformationMat = np.reshape(transMatFromFile,(4,4))
        transform = em.Transform()
        transform.setMatrix(transformationMat)

        resultMat = Transform()
        outputParts = md.MetaData()
        mdToAlign = md.MetaData(self.imgsInputFn)
        for row in md.iterRows(mdToAlign):
            inMat = rowToAlignment(row, ALIGN_PROJ)
            partTransformMat = inMat.getMatrix()
            partTransformMatrix = np.matrix(partTransformMat)
            #print(partTransformMatrix, transformationMat)
            newTransformMatrix = np.matmul(transformationMat,partTransformMatrix)
            resultMat.setMatrix(newTransformMatrix)
            #print(newTransformMatrix, resultMat)
            rowOut = md.Row()
            rowOut.copyFromRow(row)
            alignmentToRow(resultMat, rowOut, ALIGN_PROJ)
            rowOut.addToMd(outputParts)
        outputParts.write(outParticlesFn)


    def compareAlignStep(self, symmetryGroup):

        copy(self._getExtraPath('outputParticles.xmd'),
             self._getExtraPath("angles1.xmd"))
        copy(self.imgsRefFn, self._getExtraPath("angles2.xmd"))

        self.runJob("xmipp_metadata_utilities","-i %s -o %s --operate keep_column itemId"%\
                    (self._getExtraPath("angles1.xmd"),self._getTmpPath("ids1.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s -o %s --operate keep_column itemId"%\
                    (self._getExtraPath("angles2.xmd"),self._getTmpPath("ids2.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set intersection %s itemId -o %s"%\
                    (self._getTmpPath("ids1.xmd"),self._getTmpPath("ids2.xmd"),self._getTmpPath("ids.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set intersection %s itemId -o %s"%\
                    (self._getExtraPath("angles1.xmd"),self._getTmpPath("ids.xmd"),self._getExtraPath("angles1_common.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set intersection %s itemId -o %s"%\
                    (self._getExtraPath("angles2.xmd"),self._getTmpPath("ids.xmd"),self._getExtraPath("angles2_common.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --operate sort itemId"%self._getExtraPath("angles1_common.xmd"))
        self.runJob("xmipp_metadata_utilities","-i %s --operate sort itemId"%self._getExtraPath("angles2_common.xmd"))

        self.runJob("xmipp_angular_distance","--ang1 %s --ang2 %s --sym %s --check_mirrors --oroot %s"%\
                    (self._getExtraPath("angles1_common.xmd"),self._getExtraPath("angles2_common.xmd"),
                     self.symmetryGroup,self._getTmpPath("angular_distance")))
        self.runJob("xmipp_metadata_utilities",'-i %s -o %s --operate keep_column "angleDiff shiftDiff"'%\
                    (self._getTmpPath("angular_distance.xmd"),self._getTmpPath("diffs.xmd")))
        self.runJob("xmipp_metadata_utilities","-i %s --set merge %s"%\
                    (self._getExtraPath("angles1_common.xmd"),self._getTmpPath("diffs.xmd")))
    

    def createOutputStep(self):   

        outVolFn = self._getExtraPath("inputVolumeAligned.vol")
        Xdim = self.inputReference.get().getDim()[0]
        newXdim = readInfoField(self._getExtraPath(), "size", xmippLib.MDL_XSIZE)
        if Xdim != newXdim:
            self.runJob("xmipp_image_resize", "-i %s --dim %d"
                        % (outVolFn, Xdim), numberOfMpi=1)
  
        outVol = em.Volume()
        outVol.setLocation(outVolFn)
        #set transformation matrix             
        fhInputTranMat = self._getExtraPath('transformation-matrix.txt')
        transMatFromFile = np.loadtxt(fhInputTranMat)
        transformationMat = np.reshape(transMatFromFile,(4,4))
        transform = em.Transform()
        transform.setMatrix(transformationMat)
        outVol.setTransform(transform)
        outVol.setSamplingRate(self.inputVolumes.get().getSamplingRate())

        outputArgs = {'outputVolume': outVol}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputVolumes, outVol)

        #particles....
        # outParticlesFn = self._getExtraPath('angles1_common.xmd')
        outParticlesFn = self._getExtraPath('outputParticles.xmd')
        Xdim = self.inputParticles.get().getXDim()
        if self.newXdim != Xdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --save_metadata_stack %s --fourier %d" %
                        (outParticlesFn,
                         self._getExtraPath('scaled_output_particles.stk'),
                         self._getExtraPath('scaled_output_particles.xmd'),
                         self.newXdim))
            moveFile(self._getExtraPath('scaled_output_particles.xmd'), outParticlesFn)
        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(self.inputParticles.get())
        outputParticles.setAlignmentProj()
        readSetOfParticles(outParticlesFn, outputParticles)

        # self.iterMd = md.iterRows(self._getExtraPath("angles1_common.xmd"),
        #                           md.MDL_ITEM_ID)
        # self.lastRow = next(self.iterMd)
        # outputParticles.copyItems(self.inputParticles.get(),
        #                     updateItemCallback=self._updateItem)
            
        outputArgs = {'outputParticles': outputParticles}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputParticles, outputParticles)


    def _updateItem(self, particle, row):
        count = 0

        while self.lastRow and particle.getObjId() == self.lastRow.getValue(
                md.MDL_ITEM_ID):
            count += 1
            if count:
                self._createItemMatrix(particle, self.lastRow)
            try:
                self.lastRow = next(self.iterMd)
            except StopIteration:
                self.lastRow = None

        particle._appendItem = count > 0

    def _createItemMatrix(self, particle, row):
        setXmippAttributes(particle, row, xmippLib.MDL_SHIFT_DIFF,
                           xmippLib.MDL_ANGLE_DIFF)

    #--------------------------- INFO functions --------------------------------------------
    
    def _validate(self):
        errors = []
        return errors
    
    def _summary(self):
        summary = []
        nVols = self._getNumberOfInputs()
            
        if nVols > 0:
            summary.append("Volumes to align: *%d* " % nVols)
        else:
            summary.append("No volumes selected.")
        summary.append("Alignment method: %s" % self.getEnumText('alignmentAlgorithm'))
                
        return summary
    
    def _methods(self):
        nVols = self._getNumberOfInputs()
        
        if nVols > 0:
            methods = 'We aligned %d volumes against a reference volume using ' % nVols
            #TODO: Check a more descriptive way to add the reference and 
            # all aligned volumes to the methods (such as obj.getNameId())
            # also to show the number of volumes from each set in the input.
            # This approach implies to consistently include also the outputs
            # ids to be tracked in all the workflow's methods.
            if self.alignmentAlgorithm == ALIGN_ALGORITHM_FAST_FOURIER:
                methods += ' the Fast Fourier alignment described in [Chen2013].' 
                
            elif self.alignmentAlgorithm == ALIGN_ALGORITHM_LOCAL:
                methods += ' a local search of the alignment parameters.'
            elif self.alignmentAlgorithm == ALIGN_ALGORITHM_EXHAUSTIVE:
                methods += ' an exhaustive search.'
            elif self.alignmentAlgorithm == ALIGN_ALGORITHM_EXHAUSTIVE_LOCAL:
                methods += ' an exhaustive search followed by a local search.'
        else:
            methods = 'No methods available yet.'
            
        return [methods]
        
    def _citations(self):
        if self.alignmentAlgorithm == ALIGN_ALGORITHM_FAST_FOURIER:
            return ['Chen2013']
        
    #--------------------------- UTILS functions --------------------------------------------
    def _iterInputVolumes(self):
        """ Iterate over all the input volumes. """
        for pointer in self.inputVolumes:
            item = pointer.get()
            if item is None:
                break
            itemId = item.getObjId()
            if isinstance(item, em.Volume):
                item.outputName = self._getExtraPath('output_vol%06d.vol' % itemId)
                # If item is a Volume and label is empty
                if not item.getObjLabel():
                    # Volume part of a set
                    if item.getObjParentId() is None:
                        item.setObjLabel("%s.%s" % (pointer.getObjValue(), pointer.getExtended()))
                    else:
                        item.setObjLabel('%s.%s' % (self.getMapper().getParent(item).getRunName(), item.getClassName()))
                yield item
            elif isinstance(item, em.SetOfVolumes):
                for vol in item:
                    vol.outputName = self._getExtraPath('output_vol%06d_%03d.vol' % (itemId, vol.getObjId()))
                    # If set item label is empty
                    if not vol.getObjLabel():
                        # if set label is not empty use it
                        if item.getObjLabel():
                            vol.setObjLabel("%s - %s%s" % (item.getObjLabel(), vol.getClassName(), vol.getObjId()))
                        else:
                            vol.setObjLabel("%s - %s%s" % (self.getMapper().getParent(item).getRunName(), vol.getClassName(), vol.getObjId()))
                    yield vol
                    
    def _getNumberOfInputs(self):
        """ Return the total number of input volumes. """
        nVols = 0
        for _ in self._iterInputVolumes():
            nVols += 1    
            
        return nVols    
                    
    def _getMaskArgs(self):
        maskArgs = ''
        if self.applyMask:
            if self.maskType == ALIGN_MASK_CIRCULAR:
                maskArgs+=" --mask circular -%d" % self.maskRadius
            else:
                maskArgs+=" --mask binary_file %s" % self.volMask
        return maskArgs
    
    def _getAlignArgs(self):
        alignArgs = ''
        
        if self.alignmentAlgorithm == ALIGN_ALGORITHM_FAST_FOURIER:
            alignArgs += " --frm"
            
        elif self.alignmentAlgorithm == ALIGN_ALGORITHM_LOCAL:
            alignArgs += " --local --rot %f %f 1 --tilt %f %f 1 --psi %f %f 1 -x %f %f 1 -y %f %f 1 -z %f %f 1" %\
               (self.initialRotAngle, self.initialRotAngle,
                self.initialTiltAngle, self.initialTiltAngle,
                self.initialInplaneAngle, self.initialInplaneAngle,
                self.initialShiftX, self.initialShiftX,
                self.initialShiftY, self.initialShiftY,
                self.initialShiftZ,self.initialShiftZ)   
            if self.optimizeScale:
                alignArgs += " --scale %f %f 0.005" %(self.initialScale, self.initialScale)
            else:
                alignArgs += " --dontScale"
        else: # Exhaustive or Exhaustive+Local
            alignArgs += " --rot %f %f %f --tilt %f %f %f --psi %f %f %f -x %f %f %f -y %f %f %f -z %f %f %f --scale %f %f %f" %\
               (self.minRotationalAngle, self.maxRotationalAngle, self.stepRotationalAngle,
                self.minTiltAngle, self.maxTiltAngle, self.stepTiltAngle,
                self.minInplaneAngle, self.maxInplaneAngle, self.stepInplaneAngle,
                self.minimumShiftX, self.maximumShiftX, self.stepShiftX,
                self.minimumShiftY, self.maximumShiftY, self.stepShiftY,
                self.minimumShiftZ, self.maximumShiftZ, self.stepShiftZ,
                self.minimumScale, self.maximumScale, self.stepScale)
               
        return alignArgs
          
