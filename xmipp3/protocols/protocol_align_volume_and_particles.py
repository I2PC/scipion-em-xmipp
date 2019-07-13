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
from pyworkflow import VERSION_2_0
from pyworkflow.em.convert import ImageHandler
import pyworkflow.em.metadata as md
from pyworkflow.em.data import Transform
from pyworkflow.em import ALIGN_PROJ
from xmipp3.convert import rowToAlignment, alignmentToRow, writeSetOfParticles, readSetOfParticles
from xmipp3.constants import SYM_URL

ALIGN_MASK_CIRCULAR = 0
ALIGN_MASK_BINARY_FILE = 1


class XmippProtAlignVolumeParticles(em.ProtAlignVolume):
    """ 
    Aligns a volume (inputVolume) using a Fast Fourier method
    with respect to a reference one (inputReference).
     The obtained alignment parameters are used to align the set of particles
     (inputParticles) that generated the input volume.
     """
    _label = 'align volume and particles'
    _lastUpdateVersion = VERSION_2_0
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
        form.addParam('inputVolume', params.PointerParam, pointerClass='Volume',
                      label="Input volume", important=True, 
                      help='Select one volume to be aligned against the reference volume.')
        form.addParam('inputParticles', params.PointerParam, pointerClass='SetOfParticles',
                      label="Input particles", important=True, 
                      help='Select one set of particles to be aligned against '
                           'the reference set of particles using the transformation '
                           'calculated with the reference and input volumes.')
        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry group",
                      help='See %s page for a description of the symmetries '
                           'accepted by Xmipp' % SYM_URL)
        
        group1 = form.addGroup('Mask')
        group1.addParam('applyMask', params.BooleanParam, default=False, 
                      label='Apply mask?',
                      help='Apply a 3D Binary mask to the volumes')
        group1.addParam('maskType', params.EnumParam,
                        choices=['circular','binary file'],
                        default=ALIGN_MASK_CIRCULAR,
                        label='Mask type', display=params.EnumParam.DISPLAY_COMBO,
                        condition='applyMask',
                      help='Select the type of mask you want to apply')
        group1.addParam('maskRadius', params.IntParam, default=-1,
                        condition='applyMask and maskType==%d' % ALIGN_MASK_CIRCULAR,
                        label='Mask radius',
                        help='Insert the radius for the mask')
        group1.addParam('maskFile', params.PointerParam,
                        condition='applyMask and maskType==%d' % ALIGN_MASK_BINARY_FILE,
                        pointerClass='VolumeMask', label='Mask file',
                        help='Select the volume mask object')

        form.addParallelSection(threads=8, mpi=1)
        
    #--------------------------- INSERT steps functions --------------------------------------------    
    def _insertAllSteps(self):

        #Some definitions of filenames
        self.fnRefVol = self._getExtraPath("refVolume.vol")
        self.fnInputVol = self._getExtraPath("inputVolume.vol")
        self.imgsInputFn = self._getExtraPath("inputParticles.xmd")

        maskArgs = self._getMaskArgs()
        alignSteps = []

        stepId0 = self._insertFunctionStep('convertStep', prerequisites=[])
        alignSteps.append(stepId0) 
        stepId1 = self._insertFunctionStep('alignVolumeStep', maskArgs,
                                           prerequisites=alignSteps)
        alignSteps.append(stepId1)     
        stepId2 = self._insertFunctionStep('alignParticlesStep',
                                           prerequisites=alignSteps)
        alignSteps.append(stepId2)

        self._insertFunctionStep('createOutputStep', prerequisites=alignSteps)
        
    #--------------------------- STEPS functions --------------------------------------------


    def convertStep(self):

        inputParts = self.inputParticles.get()
        writeSetOfParticles(inputParts, self.imgsInputFn)

        #Resizing inputs
        ih = ImageHandler()
        ih.convert(self.inputReference.get(), self.fnRefVol)
        XdimRef = self.inputReference.get().getDim()[0]
        ih.convert(self.inputVolume.get(), self.fnInputVol)
        XdimInput = self.inputVolume.get().getDim()[0]

        if XdimRef!=XdimInput:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" %
                        (self.fnRefVol, XdimInput), numberOfMpi=1)


    def alignVolumeStep(self, maskArgs):

        fhInputTranMat = self._getExtraPath('transformation-matrix.txt')
        outVolFn = self._getExtraPath("inputVolumeAligned.vol")
      
        args = "--i1 %s --i2 %s --apply %s" % \
               (self.fnRefVol, self.fnInputVol, outVolFn)
        args += maskArgs
        args += " --frm "
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
            newTransformMatrix = np.matmul(transformationMat, partTransformMatrix)
            resultMat.setMatrix(newTransformMatrix)
            rowOut = md.Row()
            rowOut.copyFromRow(row)
            alignmentToRow(resultMat, rowOut, ALIGN_PROJ)
            rowOut.addToMd(outputParts)
        outputParts.write(outParticlesFn)


    def createOutputStep(self):   

        outVolFn = self._getExtraPath("inputVolumeAligned.vol")
        outVol = em.Volume()
        outVol.setLocation(outVolFn)
        #set transformation matrix             
        fhInputTranMat = self._getExtraPath('transformation-matrix.txt')
        transMatFromFile = np.loadtxt(fhInputTranMat)
        transformationMat = np.reshape(transMatFromFile,(4,4))
        transform = em.Transform()
        transform.setMatrix(transformationMat)
        outVol.setTransform(transform)
        outVol.setSamplingRate(self.inputVolume.get().getSamplingRate())

        outputArgs = {'outputVolume': outVol}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputVolume, outVol)

        #particles....
        outParticlesFn = self._getExtraPath('outputParticles.xmd')
        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(self.inputParticles.get())
        outputParticles.setAlignmentProj()
        readSetOfParticles(outParticlesFn, outputParticles)
        outputArgs = {'outputParticles': outputParticles}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputParticles, outputParticles)


    #--------------------------- INFO functions --------------------------------------------
    
    def _validate(self):
        errors = []
        return errors
    
    def _summary(self):
        summary = []
        summary.append("Alignment method: %s" % self.getEnumText('alignmentAlgorithm'))
        return summary
    
    def _methods(self):
        methods = 'We aligned a volume against a reference volume using '
        methods += ' the Fast Fourier alignment described in [Chen2013].'
        return [methods]
        
    def _citations(self):
        return ['Chen2013']
        
    #--------------------------- UTILS functions -------------------------------
    def _getMaskArgs(self):
        maskArgs = ''
        if self.applyMask:
            if self.maskType == ALIGN_MASK_CIRCULAR:
                maskArgs+=" --mask circular -%d" % self.maskRadius
            else:
                maskArgs+=" --mask binary_file %s" % self.volMask
        return maskArgs

          
