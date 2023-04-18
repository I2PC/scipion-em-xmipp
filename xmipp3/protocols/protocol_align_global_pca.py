# **************************************************************************
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
# **************************************************************************
"""
Protocol to perform global alignment
"""

from os.path import join, exists, split
import os

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.protocol.params import (PointerParam, StringParam, FloatParam,
                                        BooleanParam, IntParam, EnumParam,
                                        NumericListParam, USE_GPU, GPU_LIST)
from pyworkflow.utils.path import (cleanPath, makePath, copyFile, moveFile,
                                   createLink, cleanPattern)
from pwem.protocols import ProtRefine3D
from pwem.objects import SetOfVolumes, Volume
from pwem.emlib.metadata import getFirstRow, getSize
from pyworkflow.utils.utils import getFloatListFromValues
from pwem.emlib.image import ImageHandler
import pwem.emlib.metadata as md
import xmipp3

from pwem import emlib
from xmipp3.convert import  writeSetOfParticles, readSetOfParticles, xmippToLocation

def updateEnviron(gpuNum):
    """ Create the needed environment for TensorFlow programs. """
    print("updating environ to select gpu %s" % (gpuNum))
    if gpuNum == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuNum)


class XmippProtAlignGlobalPca(ProtRefine3D, xmipp3.XmippProtocol):
    """This is a 3D global refinement protocol"""
    _label = 'alignPca'
    _lastUpdateVersion = VERSION_2_0
    _conda_env = 'flexutils-tensorflow'
    # _conda_env = 'xmipp_DLTK_v0.3'
    

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")


        form.addSection(label='Input')

        form.addParam('inputParticles', PointerParam, label="Experimental Images", important=True,
                      pointerClass='SetOfParticles', allowsNull=True,
                      help='Select a set of images at full resolution')
        # form.addParam('inputReferences', PointerParam, label="References Images", important=True,
        #               pointerClass='SetOfParticles', allowsNull=True,
        #               help='Select a set of references at full resolution')
        form.addParam('inputVolume', PointerParam, label="Initial volumes", important=True,
                      pointerClass='Volume', allowsNull=True,
                      help='Select a initial volume . ')
        form.addParam('corectCtf', BooleanParam, default=True,
                      label='Correct CTF?',
                      help='If you set to *Yes*, the CTF of the experimental particles will be corrected')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        form.addParam('angleGallery',FloatParam, label="angle for references", default=5, expertLevel=LEVEL_ADVANCED,
                      help='Distance in degrees between sampling points for generate gallery of references images')
                  
        form.addSection(label='Pca training')
        
        form.addParam('resolution',FloatParam, label="max resolution", default=10,
                      help='Maximum resolution to be consider for alignment')
        form.addParam('coef' ,FloatParam, label="% variance", default=0.7, expertLevel=LEVEL_ADVANCED,
                      help='Percentage of variance to determine the number of coefficients to be considers (between 0-1).'
                      ' The higher the percentage, the higher the accuracy, but the calculation time increases.')
        form.addParam('numPart' ,IntParam, label="number of particles for PCA", default=20000, 
                      expertLevel=LEVEL_ADVANCED,
                      help='number of particles to be consider for PCA training')
        
        
        form.addSection(label='Global alignment')
        form.addParam('applyShift', BooleanParam, default=False,
                      label='Consider previous alignment?',
                      help='If you set to *Yes*, the particles will be centered acording to previous alignment')
        form.addParam('numberOfIterations', IntParam, default=1, label='Number of iterations')
        form.addParam('angle' ,IntParam, label="angular sampling", default=8, 
                      help='Angular sampling for particles alignment')
        form.addParam('shift' ,IntParam, label="shift sampling", default=4, 
                      help='Sampling rate in the alignment')
        form.addParam('MaxShift', IntParam, label="Max. shift", default=20, expertLevel=LEVEL_ADVANCED,
                      help='Maximum shift for translational search')
        # form.addParam('MaxShift', IntParam, label="Max. shift (%)", default=10, expertLevel=LEVEL_ADVANCED,
        #               help='Maximum shift as a percentage of the image size')
        


        form.addParallelSection(threads=1, mpi=16)

    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        
        updateEnviron(self.gpuList.get())
        
        self.imgsFnXmd = self._getExtraPath('images_original.xmd')
        self.imgsFn = self._getExtraPath('images.mrcs')
        self.refsFn = self._getExtraPath('references.mrcs')
        self.refsFnXmd = self._getExtraPath('references.xmd')
        self.sampling = self.inputParticles.get().getSamplingRate()
        size =self.inputParticles.get().getDimensions()[0]
        # self.MaxShift = int(size * self.MaxShift.get() / 100)
        self.MaxShift = self.MaxShift.get()
        iterations = self.numberOfIterations.get()


        self._insertFunctionStep('convertInputStep', self.inputParticles.get(), self.imgsFnXmd)
        self._insertFunctionStep("createGallery")
        self._insertFunctionStep("pcaTraining")
        self._insertFunctionStep("globalAlign", self.angle.get(), self.shift.get(), self.MaxShift)       
        self._insertFunctionStep("createOutput")
    

    #--------------------------- STEPS functions ---------------------------------------------------        
    def convertInputStep(self, inputFn, outputFn):
        writeSetOfParticles(inputFn, outputFn)  

        if self.corectCtf:  
            args = ' -i  %s -o %s --sampling_rate %s '%(self.imgsFnXmd, self.imgsFn, self.sampling)
            self.runJob("xmipp_ctf_correct_wiener2d", args, numberOfMpi=self.numberOfMpi.get()) 
        else:
            args = ' -i  %s -o %s '%(self.imgsFnXmd, self.imgsFn)
            self.runJob("xmipp_image_convert",args, numberOfMpi=1)           
    
    def createGallery(self):
        refVol = self.inputVolume.get().getFileName()
        args = ' -i  %s --sym %s --sampling_rate %s  -o %s '% \
                (refVol, self.symmetryGroup.get(), self.angleGallery.get(), self.refsFn)
        self.runJob("xmipp_angular_project_library", args)
        moveFile( self._getExtraPath('references.doc'), self.refsFnXmd)
    
    
    def pcaTraining(self):
        args = ' -i %s -n 1 -s %s -hr %s -lr 530 -p %s -t %s -o %s/train_pca'% \
                (self.imgsFn, self.sampling, self.resolution.get(), self.coef.get(), self.numPart.get(), self._getExtraPath())
        program = self.getProgram("train_pca.py")
        self.runJob(program, args, numberOfMpi=1)
        
    def globalAlign(self, angle, shift, MaxShift):
        args = ' -i %s -r %s -a %s -sh %s -msh %s -b %s/train_pca_bands.pt -v %s/train_pca_vecs.pt  -o %s/newStar_exp.xmd -stExp %s  -stRef %s  -s %s '% \
                (self.imgsFn, self.refsFn, angle, shift, MaxShift,\
                 self._getExtraPath(), self._getExtraPath(), self._getExtraPath(), self.imgsFnXmd, self.refsFnXmd, self.sampling)
        if self.applyShift:
            args += ' --apply_shifts ' 
        program = self.getProgram("align_images.py")        
        self.runJob(program, args, numberOfMpi=1)


    def createOutput(self):
        
        imgSet = self._getExtraPath('newStar_exp.xmd')
        partSet = self._createSetOfParticles()             
        EXTRA_LABELS = [
            #emlib.MDL_COST
        ]
         # Fill
        readSetOfParticles(
            imgSet,
            partSet,
            extraLabels=EXTRA_LABELS
        )
        partSet.setSamplingRate(self.inputParticles.get().getSamplingRate())       
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(self.inputParticles, partSet)
        
        # imgSet = self.inputParticles.get()
        # partSet = imgSet.create(self._getExtraPath())
        # partSet.copyInfo(imgSet)
        # # partSet = self._createSetOfParticles()
        # imgFn = self._getExtraPath('newStar_exp.xmd')
        # # partSet.copyInfo(imgSet)
        # # partSet.setIsPhaseFlipped(True)
        # partSet.copyItems(imgSet,
        #                     updateItemCallback=self._updateLocation,
        #                     itemDataIterator=md.iterRows(imgFn))
        #
        # self._defineOutputs(outputParticles=partSet)
        # self._defineSourceRelation(imgSet, partSet)
    
    #--------------------------- INFO functions --------------------------------------------
    def _validate(self):
        errors = []
    
        return errors
    
    def _warnings(self):
        warnings = []
        return warnings
    
    def _summary(self):
        summary = []
        summary.append("Symmetry: %s" % self.symmetryGroup.get())   
        return summary
    

    #--------------------------- UTILS functions --------------------------------------------
    def getTensorflowActivation(self):
        return "conda activate flexutils-tensorflow"
    
    def getProgram(self, program):
        cmd = '%s %s && ' % (xmipp3.Plugin.getCondaActivationCmd(), self.getTensorflowActivation())
        return cmd + ' %(program)s ' % locals()
    
    def _updateLocation(self, item, row):
        index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
        item.setLocation(index, filename)
    