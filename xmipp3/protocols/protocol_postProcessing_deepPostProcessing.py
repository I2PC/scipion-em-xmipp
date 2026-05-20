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
from pyworkflow import VERSION_3_0
from pyworkflow.protocol.params import (PointerParam, FloatParam, EnumParam, LEVEL_ADVANCED,
                                        StringParam, GPU_LIST, BooleanParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pwem.objects import Volume
from xmipp3.base import XmippProtocol
from pyworkflow.utils import createLink

INPUT_VOL_BASENAME="inputVol.mrc"
INPUT_HALF1_BASENAME="inputHalf1.mrc"
INPUT_HALF2_BASENAME="inputHalf2.mrc"

INPUT_MASK_BASENAME="inputMask.mrc"
POSTPROCESS_VOL_BASENAME= "deepPostProcess.mrc"

class XmippProtDeepVolPostProc(ProtAnalysis3D, XmippProtocol):
    """    
    Given a map the protocol performs automatic deep post-processing to enhance
    visualization. Usage guide at https://github.com/rsanchezgarc/deepEMhancer

    AI Generated

    ## Overview

    The deepEMhancer protocol performs automatic deep-learning-based
    post-processing of cryo-EM maps.

    Its purpose is to improve the visual interpretability of a map by enhancing
    structural features, sharpening density, and suppressing noise-like regions.
    The protocol uses a trained neural network model from deepEMhancer and
    produces a post-processed volume that can be inspected, interpreted, or used
    for visualization.

    deepEMhancer is especially useful when the user wants to obtain a visually
    clearer map from an unsharpened, unmasked input volume or from a pair of half
    maps. However, the output should be interpreted carefully. The protocol
    enhances the map using a learned model; it does not replace standard
    validation, FSC analysis, local resolution estimation, or careful inspection of
    the original data.

    ## Inputs and General Workflow

    The protocol can work with either:

    - a single input volume;
    - two half maps.

    The input data are converted or linked as MRC files. If half maps are used,
    they may be read from the half-map information attached to an imported volume,
    or they may be provided explicitly as two separate volumes.

    The protocol then runs the deepEMhancer post-processing program using the
    selected normalization strategy, neural-network model, GPU, and batch size. The
    resulting post-processed map is registered in Scipion as an output volume.

    The output volume preserves the sampling rate and origin information from the
    input volume or half maps.

    ## Input Volume

    When **Would you like to use half maps?** is set to **No**, the protocol uses a
    single **Input Volume**.

    This input should be an unmasked and non-sharpened map. This is important
    because deepEMhancer expects input maps that have not already been strongly
    post-processed. If a map has already been sharpened, masked, or aggressively
    filtered, the neural-network output may be less reliable or may overemphasize
    features introduced by previous processing.

    The input volume should correspond to the structure that the user wants to
    enhance for visualization and interpretation.

    ## Use of Half Maps

    When **Would you like to use half maps?** is set to **Yes**, the protocol uses
    two half maps instead of a single full map.

    Half maps are independently reconstructed maps from two halves of the particle
    data. They contain useful information about signal and noise consistency and
    are often the preferred input for post-processing and validation workflows.

    The protocol supports two ways of providing half maps:

    - using half maps already attached to an imported volume;
    - providing half map 1 and half map 2 explicitly as separate volumes.

    Using half maps can give the post-processing method additional information
    about reproducible signal and noise behavior.

    ## Half Maps Attached to the Volume

    If **Are the half maps included in the volume?** is set to **Yes**, the
    protocol obtains the half-map file names from the selected input volume.

    This is the usual option when the volume was imported or generated in Scipion
    with associated half maps.

    If this option is set to **No**, the user must provide **Volume Half 1** and
    **Volume Half 2** manually.

    The two half maps should correspond to the same reconstruction, have the same
    box size, sampling rate, origin, and orientation, and represent independent
    halves of the same dataset.

    ## Input Normalization

    The **Input normalization** parameter is one of the most important settings of
    the protocol.

    Normalization is critical because the neural network expects map intensities
    to be on a scale compatible with the data used during training. Poor
    normalization can lead to poor enhancement, excessive sharpening, loss of
    density, or artificial-looking results.

    The protocol provides three normalization modes:

    - automatic normalization;
    - normalization from noise statistics;
    - normalization from a binary mask.

    If the result is not satisfactory, trying a different normalization strategy is
    often one of the first things to do.

    ## Automatic Normalization

    With **Automatic normalization**, the protocol estimates the required
    normalization automatically.

    This is the simplest option and is often a good starting point. It avoids the
    need to provide a mask or explicit noise statistics.

    However, automatic normalization may fail in some cases, especially if the map
    has unusual intensity distribution, strong artifacts, large empty regions,
    strong masking effects, or non-standard preprocessing.

    If the output looks too aggressive, too weak, or biologically implausible, the
    user should consider trying one of the other normalization modes.

    ## Normalization from Statistics

    With **Normalization from statistics**, the user provides the mean and standard
    deviation of the noise.

    The parameters are:

    - **noise mean**;
    - **noise standard deviation**.

    This mode gives the user more control over the input intensity scaling. It can
    be useful when the noise statistics are known or can be estimated reliably from
    background regions.

    Incorrect noise statistics can strongly affect the result. A wrong standard
    deviation may cause the model to under-enhance or over-enhance the map.

    ## Normalization from Binary Mask

    With **Normalization from binary mask**, the user provides a binary mask
    indicating which voxels correspond to protein and which correspond to
    background.

    The mask should contain:

    - value 1 for protein or molecular density;
    - value 0 for non-protein or background.

    The mask should be as tight as possible while still including the relevant
    protein density. A mask that is too loose may include too much background in
    the normalization. A mask that is too tight may exclude real density and affect
    the enhancement.

    When this normalization mode is selected, the protocol uses a model checkpoint
    specifically intended for masked normalization.

    ## Model Power

    The **Model power** parameter selects which deepEMhancer model target is used.

    The available options are:

    **Tight target** produces a more sharpened result. It may enhance structural
    features strongly, but in some cases it may also remove or suppress weak
    regions of the protein.

    **Wide target** is less aggressive. It usually preserves more regions of the
    protein, although the output may appear less sharply enhanced.

    **HighRes** is recommended for high-resolution volumes.

    The choice depends on the quality of the map and the purpose of the
    post-processing. For visualization, the tight model may be attractive, but the
    wide model can be safer when weak or flexible regions should be preserved.

    ## Cleaning Small Connected Components

    The option **Remove small CC after processing** enables an additional cleaning
    step that removes small connected components after post-processing.

    These small components are often noise-like isolated regions. Removing them can
    make the output map cleaner.

    The cleaning strength is controlled by **Relative size CC to remove**, which
    defines the relative size of connected components to remove as a fraction of
    the total number of positive voxels.

    This option can slightly improve visual results, but it should be used with
    care. In unusual cases, small real protein regions could be removed, especially
    for fragmented, flexible, or low-occupancy density.

    ## Batch Size

    The **Batch size** parameter controls how many cubes of the volume are processed
    simultaneously by the neural network.

    A larger batch size may improve GPU utilization and speed. A smaller batch size
    uses less GPU memory.

    If a CUDA out-of-memory error occurs, reduce the batch size. If GPU memory is
    underused and processing is slow, increasing the batch size may improve
    performance.

    This parameter affects computational performance, not the biological meaning of
    the output.

    ## GPU Execution

    deepEMhancer uses GPU execution.

    The protocol allows selecting the GPU ID through the hidden GPU parameter. In a
    queue environment, the protocol can use the GPU resources assigned by the
    queue. Otherwise, it uses the selected GPU list.

    The protocol also enables TensorFlow GPU memory growth to reduce the chance
    that the process reserves all GPU memory at once.

    If the required deep-learning toolkit or trained model is not available, the
    protocol validation reports an installation error.

    ## Output Volume

    The main output is **Volume**, the deepEMhancer post-processed map.

    The output volume is written as an MRC file and registered in Scipion with the
    same sampling rate and origin as the input map or half maps.

    This volume is intended primarily for enhanced visualization and
    interpretation. It may help reveal secondary-structure elements, connectivity,
    and local features more clearly than the raw or unsharpened input map.

    The output should be compared with the original map and, when available, with
    the half maps, FSC curves, and local-resolution estimates.

    ## Interpretation and Cautions

    deepEMhancer produces an enhanced map using a learned model. This can be very
    useful, but it also means that the output should not be interpreted as a purely
    experimental density map in the same way as the original reconstruction.

    Enhanced features should be checked against the input map, half maps, and
    independent validation evidence. This is especially important for weak density,
    flexible regions, ligands, peripheral domains, or regions near the noise level.

    The protocol is a post-processing and visualization tool. It does not replace
    map validation.

    ## Practical Recommendations

    Use unmasked, non-sharpened maps as input when using a single volume.

    Use half maps when they are available, because they provide information about
    reproducible signal.

    Start with automatic normalization. If the result is unsatisfactory, try
    normalization from a binary mask or from noise statistics.

    Use the tight model when stronger sharpening is desired and the density is
    robust. Use the wide model when preserving weak or extended regions is more
    important. Use the high-resolution model for high-resolution maps.

    Inspect the output together with the original input map. Do not rely only on
    the enhanced map for biological conclusions.

    Reduce the batch size if GPU memory errors occur.

    Use the cleaning option cautiously, especially for maps with small real
    features, flexible regions, or fragmented density.

    ## Final Perspective

    deepEMhancer is a deep-learning-based post-processing protocol for improving
    the visual quality of cryo-EM maps.

    For biological users, its main value is that it can produce clearer and more
    interpretable density maps, especially for visualization, figure preparation,
    and model-building guidance.

    The enhanced map should be treated as an aid to interpretation, not as a
    replacement for the original reconstruction or for standard validation
    procedures. Used carefully, deepEMhancer can be a powerful tool for revealing
    structural features while keeping the user aware of the need for independent
    validation.
    """
    _label = 'deepEMhancer'
    _conda_env = 'xmipp_deepEMhancer'
    _lastUpdateVersion = VERSION_3_0

    NORMALIZATION_AUTO=0
    NORMALIZATION_STATS=1
    NORMALIZATION_MASK=2
    NORMALIZATION_OPTIONS=["Automatic normalization", "Normalization from statistics", "Normalization from binary mask"]

    TIGHT_MODEL=0
    WIDE_MODEL=1
    HI_RES=2
    MODEL_TARGET_OPTIONS=["tight target", "wide target", "highRes"]

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


        form.addParam('useHalfMapsInsteadVol', BooleanParam, default=False,
                      label="Would you like to use half maps?",
                      help='DeepEMhancer uses either half maps or non-sharpened non-masked input volumes. Please, select the type of input map(s) you will provide')

        form.addParam('halfMapsAttached', BooleanParam, default=True,
                      condition='useHalfMapsInsteadVol',
                      label="Are the half maps included in the volume?",
                      help='When you import a map, you can associate half maps to it. Select *yes* if the half maps are associated'
                           'to the input volume. If half maps are not associated, select *No* and'
                           'you will be able to provide then as regular maps')


        form.addParam('inputHalf1', PointerParam, pointerClass='Volume',
                      label="Volume Half 1", important=True,
                      condition='useHalfMapsInsteadVol and not halfMapsAttached',
                      help='Select half map 1 to apply deep postprocessing. ')

        form.addParam('inputHalf2', PointerParam, pointerClass='Volume',
                      label="Volume Half 2", important=True,
                      condition='useHalfMapsInsteadVol and not halfMapsAttached',
                      help='Select half map 2 to apply deep postprocessing. ')
        
        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input Volume", important=True,
                      condition='not useHalfMapsInsteadVol or halfMapsAttached',
                      help='Select a volume to apply deep postprocessing. Unmasked, non-sharpened input required')


        form.addParam('normalization', EnumParam,
                      choices=self.NORMALIZATION_OPTIONS,
                      default=self.NORMALIZATION_AUTO,
                      label='Input normalization',
                      help='Input normalization is critical for the algorithm to work.\nIf you select *%s* input will be'
                           'automatically normalized (generally works but may fail).\nIf you select *%s* input will be'
                           'normalized according the statistics of the noise of the volume and thus, you will need to provide'
                           'the mean and standard deviation of the noise. Additionally, a binary mask (1 protein, 0 not protein) '
                           'for the protein can be used for normalization if you select *%s* . The mask should be as tight '
                           'as possible.\nnBad results may be obtained if normalization does not work, so you may want to try '
                           'different options if not good enough results are observerd'%tuple(self.NORMALIZATION_OPTIONS))

        form.addParam('inputMask', PointerParam, pointerClass='VolumeMask',
                      allowsNull=True,
                      condition=" normalization==%s"%self.NORMALIZATION_MASK,
                      label="binary mask",
                      help='The mask determines which voxels are protein (1) and which are not (0)')

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


        form.addParam('modelType', EnumParam,
                      condition=" normalization in [%s, %s]"%(self.NORMALIZATION_STATS,self.NORMALIZATION_AUTO),
                      choices=self.MODEL_TARGET_OPTIONS,
                      default=self.TIGHT_MODEL,
                      label='Model power',
                      help='Select the deep learning model to use.\nIf you select *%s* the postprocessing will be more sharpen,'
                           ' but some regions of the protein could be masked out.\nIf you select *%s* input will be less sharpen'
                           ' but most of the regions of the protein will be preserved\nOption *%s*,  is recommended for high'
                           ' resolution volumes'%tuple(self.MODEL_TARGET_OPTIONS))


        form.addParam('performCleaningStep', BooleanParam,
                      default=False, expertLevel=LEVEL_ADVANCED,
                      label='Remove small CC after processing',
                      help='If you set to *Yes*, a post-processing step will be launched to remove small connected components'
                           'that are likely noise. This step may remove protein in some unlikely situations, but generally, it'
                           'slighly improves results')

        form.addParam('sizeFraction_CC', FloatParam, default=0.05,
                      allowsNull=False,  expertLevel=LEVEL_ADVANCED,
                      condition=" performCleaningStep",
                      label="Relative size (0. to 1.) CC to remove",
                      help='The relative size of a small connected component to be removed, as the fraction of total voxels>0 ')

        form.addParam('batch_size', IntParam, default=8,
                      allowsNull=False,  expertLevel=LEVEL_ADVANCED,
                      label="Batch size",
                      help='Number of cubes to process simultaneously. Make it lower if CUDA Out Of Memory error happens and increase it if low GPU performance observed')

    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
            # Convert input into xmipp Metadata format

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('deepVolPostProStep')
        self._insertFunctionStep('createOutputStep')

    def _inputVol2Mrc(self, inputFname, outputFname):
        if inputFname.endswith(".mrc") or inputFname.endswith(".map"):
          if not os.path.exists(outputFname):
            createLink(inputFname, outputFname)
        else:
          self.runJob('xmipp_image_convert', " -i %s -o %s:mrc -t vol" % (inputFname, outputFname))

    def convertInputStep(self):
        """ Read the input volume.
        """

        if self.useHalfMapsInsteadVol.get():
          if self.halfMapsAttached.get():
            half1Fname, half2Fname = self.inputVolume.get().getHalfMaps().split(',')
          else:
            half1Fname, half2Fname =self.inputHalf1.get().getFileName(), self.inputHalf2.get().getFileName()

          self._inputVol2Mrc(half1Fname, self._getTmpPath(INPUT_HALF1_BASENAME))
          self._inputVol2Mrc(half2Fname, self._getTmpPath(INPUT_HALF2_BASENAME))

        else:
          self._inputVol2Mrc(self.inputVolume.get().getFileName(), self._getTmpPath(INPUT_VOL_BASENAME))

        if  self.inputMask.get() is not None:
          self._inputVol2Mrc(self.inputMask.get().getFileName(), self._getTmpPath(INPUT_MASK_BASENAME))


    def deepVolPostProStep(self):
        outputFname= self._getExtraPath(POSTPROCESS_VOL_BASENAME)
        if os.path.isfile(outputFname):
          return

        if self.useHalfMapsInsteadVol.get():
          half1= self._getTmpPath(INPUT_HALF1_BASENAME)
          half2= self._getTmpPath(INPUT_HALF2_BASENAME)
          params=" -i %s -i2 %s"%(half1, half2)
        else:
          inputFname = self._getTmpPath(INPUT_VOL_BASENAME)
          params=" -i %s "%inputFname

        params+=" -o %s "%outputFname
        params+= " --sampling_rate %f "%(self.inputVolume.get().getSamplingRate() if self.inputVolume.get() is not None
                                                                          else self.inputHalf1.get().getSamplingRate())
        params+= " -b %s " %(self.batch_size)

        if self.useQueueForSteps() or self.useQueue():
          params += ' -g all '
        else:
          params += ' -g %s' % (",".join([str(elem) for elem in self.getGpuList()]))

        if self.normalization==self.NORMALIZATION_MASK:
          params+= " --binaryMask %s "%(self._getTmpPath(INPUT_MASK_BASENAME))
        elif self.normalization==self.NORMALIZATION_STATS:
          params+= " --noise_stats_mean %f --noise_stats_std %f "%(self.noiseMean, self.noiseStd)


        if self.performCleaningStep:
          params+= " --cleaningStrengh %f" %self.sizeFraction_CC.get()
        else:
          params+= " --cleaningStrengh -1 "

        if  self.normalization in [self.NORMALIZATION_AUTO, self.NORMALIZATION_STATS]:
          if self.modelType == self.TIGHT_MODEL:
            params+= " --checkpoint %s "%self.getModel("deepEMhancer_v016", "production_checkpoints/deepEMhancer_tightTarget.hd5")
          elif self.modelType == self.HI_RES:
            params+= " --checkpoint  %s "%self.getModel("deepEMhancer_v016", "production_checkpoints/deepEMhancer_highRes.hd5")
          else:
            params+= " --checkpoint  %s "%self.getModel("deepEMhancer_v016", "production_checkpoints/deepEMhancer_wideTarget.hd5")
        else: #self.NORMALIZATION_MASK
          params+= " --checkpoint  %s "%self.getModel("deepEMhancer_v016", "production_checkpoints/deepEMhancer_masked.hd5")

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self.runJob("xmipp_deep_volume_postprocessing", params, numberOfMpi=1)

                                                        
    def createOutputStep(self):

        volume=Volume()
        volume.setFileName(self._getExtraPath(POSTPROCESS_VOL_BASENAME))

        if self.useHalfMapsInsteadVol.get():
            if self.halfMapsAttached.get():
                inVol = self.inputVolume.get()
            else:
                inVol = self.inputHalf1.get()

            volume.setSamplingRate(inVol.getSamplingRate())
            volume.setOrigin(inVol.getOrigin(force=True))

            self._defineOutputs(Volume=volume)
            self._defineTransformRelation(inVol, volume)
            if not self.halfMapsAttached.get():
              self._defineTransformRelation(self.inputHalf2, volume)
        else:
            inVol = self.inputVolume.get()
            volume.setSamplingRate(inVol.getSamplingRate())
            volume.setOrigin(inVol.getOrigin(force=True))

            self._defineOutputs(Volume=volume)
            self._defineTransformRelation(self.inputVolume, volume)


                
    # --------------------------- INFO functions ------------------------------

    def _methods(self):
        messages = []
        messages.append(
                "Information about the method in " + "Sanchez-Garcia et al., 2020 ( https://doi.org/10.1101/2020.06.12.148296 )")
        return messages
    
    def _summary(self):
        summary = []
        if self.useHalfMapsInsteadVol.get():
          summary.append("Input: half maps")
        else:
          summary.append("Input: raw data map")

        if self.normalization == self.NORMALIZATION_AUTO:
          summary.append("Normalization: auto")
        elif self.normalization == self.NORMALIZATION_STATS:
          summary.append("Normalization: manual statistics")
        elif self.normalization == self.NORMALIZATION_MASK:
          summary.append("Normalization: from mask")

        return summary

    def _validate(self):
        """ Check if the installation of this protocol is correct.
        Can't rely on package function since this is a "multi package" package
        Returning an empty list means that the installation is correct
        and there are not errors. If some errors are found, a list with
        the error messages will be returned.
        """
        error=self.validateDLtoolkit(model="deepEMhancer_v016")

        return error
    
    def _citations(self):
        return ['Sanchez-Garcia, 2020, https://doi.org/10.1101/2020.06.12.148296']

