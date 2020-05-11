# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Ruben Sanchez Garcia (rsanchez@cnb.csic.es)
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

import os
from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, FloatParam, EnumParam, LEVEL_ADVANCED,
                                        StringParam, GPU_LIST, BooleanParam)
from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume
import xmipp3

INPUT_VOL_BASENAME="inputVol.mrc"
INPUT_MASK_BASENAME="inputMask.mrc"
POSTPROCESS_VOL_BASENAME= "deepPostProcess.mrc"

class XmippProtDeepVolPostProc(ProtAnalysis3D, xmipp3.XmippProtocol):
    """    
    Given a map the protocol assigns local resolutions to each voxel of the map.
    """
    _label = 'deep volPostProcessing'
    _lastUpdateVersion = VERSION_2_0

    NORMALIZATION_AUTO=0
    NORMALIZATION_STATS=1
    NORMALIZATION_MASK=2
    NORMALIZATION_OPTIONS=["Automatic normalization", "Normalization from statistics", "Normalization from binary mask" ]
    
    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    
    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        
        form.addHidden(GPU_LIST, StringParam, default='0',
                       label="Choose GPU ID",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on. Select "
                            "the GPU ID in which the protocol will run (select only 1 GPU)")

        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Volume", 
                      help='Select a volume to apply deep postprocessing. Unmasked input required')


        form.addParam('normalization', EnumParam,
                      choices=self.NORMALIZATION_OPTIONS,
                      default=self.NORMALIZATION_AUTO,
                      label='Input normalization',
                      help='Input normalization is critical for the algorithm to work.\nIf you select *%s* input will be'
                           'automatically normalized (generally works but may fail).\nIf you select *%s* input will be'
                           'normalized according the statistics of the noise of the volume and thus, you will need to provide'
                           'the mean and standard deviation of the noise. Additionally, a binary mask (1 protein, 0 not protein) '
                           'for the protein can be used for normalization if you select *%s* . The mask should be as tight '
                           'as possible.\nBad results may be obtained if normalization does not work'%tuple(self.NORMALIZATION_OPTIONS))

        form.addParam('inputMask', PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      condition=" normalization==%s"%self.NORMALIZATION_MASK,
                      label="binary mask",
                      help='The mask determines which voxels are protein (1) and which are not (0)'
                      ' and which are not')

        form.addParam('noiseMean', FloatParam,
                      allowsNull=True,
                      condition=" normalization==%s"%self.NORMALIZATION_STATS,
                      label="noise mean",
                      help='The mean of the noise used to normalize the input')

        form.addParam('noiseStd', FloatParam,
                      allowsNull=True,
                      condition=" normalization==%s"%self.NORMALIZATION_STATS,
                      label="noise standard deviation",
                      help='The standard deviation of the noise used to normalize the input')


        form.addParam('performCleaningStep', BooleanParam,
                      default=False, expertLevel=LEVEL_ADVANCED,
                      label='Remove small CC after processing',
                      help='If you set to *Yes*, a post-processing step will be launched to remove small connected components'
                           'that are likely noise. This step may remove protein in some unlikely situations, but generally, it'
                           'slighly improves results')

        form.addParam('sizeFraction_CC', FloatParam, default=0.1,
                      allowsNull=False,  expertLevel=LEVEL_ADVANCED,
                      condition=" performCleaningStep",
                      label="Relative size (0. to 1.) CC to remove",
                      help='The relative size of a small connected component to be removed, as the fraction of total voxels>0 ')


    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('deepVolPostProStep')
        self._insertFunctionStep('createOutputStep')

    def _inputVol2Mrc(self, inputFname, outputFname):
        if inputFname.endswith(".mrc"):
          os.symlink(os.path.abspath(inputFname), outputFname)
        else:
          self.runJob('xmipp_image_transform', " -i %s -o %s:mrc -t vol" % (inputFname, outputFname))

    def convertInputStep(self):
        """ Read the input volume.
        """      

        self._inputVol2Mrc(self.inputVolume.get().getFileName(), self._getTmpPath(INPUT_VOL_BASENAME))
        if  self.inputMask.get() is not None:
          self._inputVol2Mrc(self.inputMask.get().getFileName(), self._getTmpPath( INPUT_MASK_BASENAME))


    def deepVolPostProStep(self):
        inputFname= os.path.abspath(self._getTmpPath(INPUT_VOL_BASENAME))
        outputFname= os.path.abspath(self._getExtraPath(POSTPROCESS_VOL_BASENAME))
        if os.path.isfile(outputFname):
          return
        params=" -i %s"%inputFname
        params+=" -o %s"%outputFname
        params+= " --sampling_rate %f"%self.inputVolume.get().getSamplingRate()

        if self.normalization==self.NORMALIZATION_MASK:
          params+= " --binaryMask %s"%(os.path.abspath(self._getTmpPath(INPUT_MASK_BASENAME)))
        elif self.normalization==self.NORMALIZATION_STATS:
          params+= " --noise_stats_mean %f --noise_stats_std %f"%(self.noiseMean, self.noiseStd)

        if self.performCleaningStep:
          params+= " --cleaningStrengh %f"%self.sizeFraction_CC.get()

        self.runJob("xmipp_deep_volume_postprocessing", params, numberOfMpi=1)

                                                        
    def createOutputStep(self):

        volume=Volume()
        volume.setFileName(self._getExtraPath(POSTPROCESS_VOL_BASENAME))

        volume.setSamplingRate(self.inputVolume.get().getSamplingRate())
        self._defineOutputs(postProcessed_Volume=volume)
        self._defineTransformRelation(self.inputVolume, volume)

                
    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        if hasattr(self, 'postProcessed_Volume'):
            messages.append(
                'Information about the method/article in ' + "???")
        return messages
    
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are not errors. If some errors are found, a list with
        the error messages will be returned.
        """
        # error=self.validateDLtoolkit(model="deepRes") #TODO: Change to deepVolPostPro
        error=[]
        return error
    
    def _citations(self):
        return ['XXXX']

