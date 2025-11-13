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
import enum

import numpy as np
import pyworkflow.protocol.params as params
from pwem.convert.headers import setMRCSamplingRate
from pyworkflow.protocol.constants import LEVEL_ADVANCED

from pwem.protocols import ProtAlignVolume
from pwem.emlib.image import ImageHandler
from pwem.objects import Transform, Volume, SetOfParticles
from pyworkflow.utils import weakImport

from pyworkflow.utils.path import cleanPath

from xmipp3.constants import SYM_URL
from pyworkflow import BETA, UPDATED, NEW, PROD

ALIGN_MASK_CIRCULAR = 0
ALIGN_MASK_BINARY_FILE = 1

ALIGN_GLOBAL = 0
ALIGN_LOCAL = 1

pointerClasses = [SetOfParticles]
with weakImport("tomo"):
    from tomo.objects import SetOfSubTomograms
    pointerClasses.append(SetOfSubTomograms)

class AlignVolPartOutputs(enum.Enum):
    Volume = Volume
    Particles = SetOfParticles

class XmippProtAlignVolumeParticles(ProtAlignVolume):
    """ 
    Aligns a volume (inputVolume) using a Fast Fourier method
    with respect to a reference one (inputReference).
     The obtained alignment parameters are used to align the set of particles or subtomograms
     (inputParticles) that generated the input volume.
     """
    _label = 'align volume and particles'
    _possibleOutputs = AlignVolPartOutputs
    _devStatus = UPDATED
    nVols = 0

    
    def __init__(self, **args):
        ProtAlignVolume.__init__(self, **args)

        # These 2 must match the output enum above.
        self.Volume = None
        self.Particles = None

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Volume parameters')
        form.addParam('inputReference', params.PointerParam, pointerClass='Volume', 
                      label="Reference volume", important=True, 
                      help='Reference volume to be used for the alignment.')    
        form.addParam('inputVolume', params.PointerParam, pointerClass='Volume',
                      label="Input volume", important=True, 
                      help='Select one volume to be aligned against the reference volume.')
        form.addParam('inputParticles', params.PointerParam, pointerClass=pointerClasses,
                      label="Input particles", important=True, 
                      help='Select one set of particles to be aligned against '
                           'the reference set of particles using the transformation '
                           'calculated with the reference and input volumes.')
        form.addParam('alignmentMode', params.EnumParam, default=ALIGN_GLOBAL, choices=["Global","Local"],
                      label="Alignment mode")
        form.addParam('considerMirrors', params.BooleanParam, default=False,
                      label='Consider mirrors')
        form.addParam('symmetryGroup', params.StringParam, default='c1',
                      label="Symmetry group",
                      help='See %s page for a description of the symmetries '
                           'accepted by Xmipp' % SYM_URL)
        form.addParam('wrap', params.BooleanParam, default=False,
                      label='Wrap', expertLevel=LEVEL_ADVANCED,
                      help='Wrap the input volume when aligning to the reference')
        
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

        maskArgs = self._getMaskArgs()

        self._insertFunctionStep(self.convertStep)
        self._insertFunctionStep(self.alignVolumeStep, maskArgs)
        self._insertFunctionStep(self.createOutputStep)
        
    #--------------------------- STEPS functions --------------------------------------------


    def convertStep(self):

        # Resizing inputs
        ih = ImageHandler()
        ih.convert(self.inputReference.get(), self.fnRefVol)
        XdimRef = self.inputReference.get().getDim()[0]
        ih.convert(self.inputVolume.get(), self.fnInputVol)
        XdimInput = self.inputVolume.get().getDim()[0]

        if XdimRef!=XdimInput:
            self.runJob("xmipp_image_resize", "-i %s --dim %d" %
                        (self.fnRefVol, XdimInput), numberOfMpi=1)


    def alignVolumeStep(self, maskArgs):

        fhInputTranMat = self.getTransformationFile()
        outVolFn = self.getOutputAlignedVolumePath()
      
        args = "--i1 %s --i2 %s --apply %s" % \
               (self.fnRefVol, self.fnInputVol, outVolFn)
        args += maskArgs
        if self.alignmentMode.get()==ALIGN_GLOBAL:
            args += " --frm"
        else:
            args += " --local"
        if self.considerMirrors:
            args += " --consider_mirror"
        args += " --copyGeo %s" % fhInputTranMat
        if not self.wrap:
            args += ' --dontWrap'
        self.runJob("xmipp_volume_align", args)
        cleanPath(self.fnRefVol)
        cleanPath(self.fnInputVol)

    def getAlignmentMatrix(self):
        if not (hasattr(self, '_lhsAlignmentMatrix') and hasattr(self, '_rhsAlignmentMatrix')):
            fhInputTranMat = self.getTransformationFile()
            transMatFromFile = np.loadtxt(fhInputTranMat)
            self._lhsAlignmentMatrix = np.reshape(transMatFromFile, (4, 4))
            self._rhsAlignmentMatrix = np.eye(4)
            
            if np.linalg.det(self._lhsAlignmentMatrix[:3,:3]) < 0:
                self._rhsAlignmentMatrix[2,2] = -1

        return self._lhsAlignmentMatrix, self._rhsAlignmentMatrix

    def getTransformationFile(self):
        return self._getExtraPath('transformation-matrix.txt')


    def createOutputStep(self):   

        # VOLUME aligned to the reference
        outVolFn = self.getOutputAlignedVolumePath()
        Ts = self.inputVolume.get().getSamplingRate()
        outVol = Volume()
        outVol.setLocation(outVolFn)
        # Set the mrc header for sampling rate.
        setMRCSamplingRate(outVolFn, Ts)

        # Set transformation matrix
        fhInputTranMat = self.getTransformationFile()
        transMatFromFile = np.loadtxt(fhInputTranMat)
        transformationMat = np.reshape(transMatFromFile,(4,4))
        transform = Transform()
        transform.setMatrix(transformationMat)
        outVol.setTransform(transform)
        outVol.setSamplingRate(Ts)

        outputArgs = {AlignVolPartOutputs.Volume.name: outVol}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputVolume, outVol)

        # PARTICLES ....
        inputParts = self.inputParticles.get()
        outputParticles = inputParts.create(self._getExtraPath())
        outputParticles.copyInfo(self.inputParticles.get())
        outputParticles.setAlignmentProj()

        # Clone set
        #readSetOfParticles(outParticlesFn, outputParticles)
        outputParticles.copyItems(inputParts,updateItemCallback=self._updateParticleTransform)
        outputArgs = {AlignVolPartOutputs.Particles.name: outputParticles}
        self._defineOutputs(**outputArgs)
        self._defineSourceRelation(self.inputParticles, outputParticles)

    def _updateParticleTransform(self, particle, row):
        lhs, rhs = self.getAlignmentMatrix()
        alignment = np.array(particle.getTransform().getMatrix())
        alignment2 = lhs @ alignment @ rhs

        particle.getTransform().setMatrix(alignment2)

    def getOutputAlignedVolumePath(self):
        outVolFn = self._getExtraPath("inputVolumeAligned.mrc")
        return outVolFn

    #--------------------------- INFO functions --------------------------------------------
    
    def _validate(self):
        errors = []
        if self.inputParticles.get().hasAlignment() is False:
            errors.append("Input particles need to be aligned (they should have transformation matrix)")
        return errors
    
    def _summary(self):
        summary = []
        summary.append("Alignment method: %s" % self.getEnumText('alignmentMode'))
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
                maskArgs+=" --mask binary_file %s" % self.maskFile.get().getFileName()
        return maskArgs

          
