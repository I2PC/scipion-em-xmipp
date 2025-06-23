# **************************************************************************
# *
# * Authors:     Jorge Garcia Condado (jorgegarciacondado@gmail.com)
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

from pwem.protocols import EMProtocol
from pwem.objects import Volume

from pyworkflow.protocol.params import PointerParam, FloatParam
from pyworkflow.object import Float
from pyworkflow.utils.path import cleanPath

from xmipp3.base import XmippProtocol
from xmipp3.convert import getImageLocation
from pyworkflow import BETA, UPDATED, NEW, PROD

class XmippProtDeepHand(EMProtocol, XmippProtocol):
    """Protocol to returns handedness of structure from trained deep learning model.

    (AI Generated)

    Purpose and Scope. This protocol determines the handedness of a cryo-EM map using a pre-trained deep learning model.
    Handedness prediction in single-particle cryo-EM is a non-trivial task, since the reconstruction process does not
    inherently resolve the absolute hand of the structure. This protocol automates the detection of left-handed
    structures and, if necessary, flips the volume to generate the correct right-handed version. It is especially
    useful for post-processing before deposition or interpretation of newly resolved structures.

    Inputs. The user provides a 3D volume to be analyzed. This input volume must be properly scaled in physical units
    (sampling rate defined). The user also sets a threshold to define a mask based on the density values in the volume.
    Two additional thresholds are required: the alpha threshold, which controls the detection of alpha-helical
    structures within the volume, and the hand threshold, which defines the decision boundary beyond which the volume
    is considered to be left-handed and should be flipped.

    Protocol Behavior. The protocol proceeds in several steps. First, it preprocesses the input volume by rescaling it
    to 1 Å per voxel and applying a mask to isolate significant density regions based on the user-defined threshold. It
    also applies a low-pass filter to 5 Å to enhance the signal of secondary structure elements relevant to hand
    prediction.

    Next, it runs a deep learning model trained on cryo-EM data to detect alpha helices and estimate the handedness of
    the volume. This is done using two neural networks: one to identify secondary structure elements and another to
    classify the overall hand of the structure.

    Based on the predicted hand value, which ranges from 0 to 1, the protocol decides whether to flip the volume.
    Values close to 0 indicate a right-handed structure, while values near 1 indicate a left-handed one. If the hand
    value exceeds the user-defined threshold, the structure is mirrored along the X-axis using xmipp_transform_mirror.

    Outputs. The protocol produces two main outputs. First, it returns a numerical value that quantifies the predicted
    handedness. Second, it provides a 3D volume: either the original input volume (if it was determined to be
    right-handed), or a flipped version (if it was predicted to be left-handed). The resulting volume can be directly
    used for downstream interpretation or deposition.

    User Workflow. The user selects the input volume and specifies the mask threshold. The alpha and hand thresholds
    are pre-filled with suggested values (0.7 and 0.6, respectively) but can be adjusted as needed. The protocol is
    then launched. After execution, the user can inspect the predicted hand value and see whether the volume was
    flipped. The output volume is accessible through the standard Scipion viewer.

    Interpretation and Best Practices. Predicted hand values close to 0 strongly suggest a right-handed structure,
    which is generally the correct biological hand. Values above 0.5 indicate increasing likelihood of a left-handed
    structure. The threshold for flipping is set to 0.6 by default but can be adjusted based on confidence or empirical
    evaluation. Users are advised to visually inspect the output and, if possible, validate it with independent
    biochemical or structural data.

    Note that preprocessing assumes the presence of alpha helices and moderate resolution (~5 Å), as the deep learning
    model is trained on structures containing discernible secondary structure. For low-resolution volumes or
    non-protein densities, predictions may be unreliable.
    """

    _label ="deep hand"
    _conda_env = "xmipp_pyTorch"
    _devStatus = UPDATED

    def __init__(self, *args, **kwargs):
        EMProtocol.__init__(self, *args, **kwargs)
        XmippProtocol.__init__(self)
        self.vResizedVolFile = 'resizedVol.mrc'
        self.vMaskFile = 'mask.mrc'
        self.vFilteredVolFile = 'filteredVol.mrc'

    def _defineParams(self, form):
        form.addSection('Input')
        form.addParam('inputVolume', PointerParam, pointerClass="Volume",
                      label='Input Volume', allowsNull=False,
                      important=True, help="Volume to process")
        form.addParam('threshold', FloatParam, label='Mask Threshold',
                      allowsNull=False, important=True,  help="Threshold for mask creation")
        form.addParam('thresholdAlpha', FloatParam, label='Alpha Threshold',
                      default=0.7, help="Threshold for alpha helix determination")
        form.addParam('thresholdHand', FloatParam, label='Hand Threshold',
                      default=0.6, help="Hand threshold to flip volume")

# --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('preprocessStep')
        self._insertFunctionStep('predictStep')
        self._insertFunctionStep('flipStep')
        self._insertFunctionStep('createOutputStep')

    def preprocessStep(self):
        # Get volume information
        volume = self.inputVolume.get()
        fn_vol = getImageLocation(volume)
        T_s = volume.getSamplingRate()

        # Paths to new files created
        self.resizedVolFile = self._getPath(self.vResizedVolFile)
        self.maskFile = self._getPath(self.vMaskFile)
        self.filteredVolFile = self._getPath(self.vFilteredVolFile)

        # Resize to 1A/px
        self.runJob("xmipp_image_resize", "-i %s -o %s --factor %f" %
                    (fn_vol, self.resizedVolFile, T_s))

        # Threshold to obtain mask
        self.runJob("xmipp_transform_threshold",
                    "-i %s -o %s --select below %f --substitute binarize"
                    % (self.resizedVolFile, self.maskFile, self.threshold.get()))

        # Filter to 5A
        self.runJob("xmipp_transform_filter", "-i %s -o %s "\
                    "--fourier low_pass %f --sampling 1"
                    % (self.resizedVolFile, self.filteredVolFile, 5.0))

    def predictStep(self):

        # Get saved models DTlK
        alpha_model= self.getModel('deepHand', '5A_SSE_experimental.pth')
        hand_model = self.getModel('deepHand', '5A_TL_hand_alpha.pth')

        # Predict hand
        args = "--alphaModel %s --handModel %s -o %s " \
               "--alphaThr %f --pathVf %s --pathVmask %s" % (
                alpha_model, hand_model, self._getExtraPath(),
                self.thresholdAlpha.get(), self._getPath(self.vFilteredVolFile),
                self._getPath(self.vMaskFile))

        env = self.removeCudaFromEnvPath(self.getCondaEnv())
        self.runJob("xmipp_deep_hand", args, env=env)

        # Store hand value
        hand_file = self._getExtraPath('hand.txt')
        f = open(hand_file)
        self.hand = Float(float(f.read()))
        f.close()

    def removeCudaFromEnvPath(self, env):
        paths = env['LD_LIBRARY_PATH'].split(':')
        finalPath = ''
        for p in paths:
            if p.find('cuda') != -1:
                pass
            else:
                finalPath+= p + ':'
        env['LD_LIBRARY_PATH'] = finalPath
        return env

    def flipStep(self):
        if self.hand.get() > self.thresholdHand.get():
            volume = self.inputVolume.get()
            fn_vol = getImageLocation(volume)
            pathFlipVol = self._getPath('flipVol.mrc')
            self.runJob("xmipp_transform_mirror", "-i %s -o %s --flipX" \
                            % (fn_vol, pathFlipVol))

    def createOutputStep(self):
        self._defineOutputs(outputHand=self.hand)

        if self.hand.get() > self.thresholdHand.get():
            vol = Volume()
            volFile = self._getPath('flipVol.mrc')
            vol.setFileName(volFile)
            Ts = self.inputVolume.get().getSamplingRate()
            vol.setSamplingRate(Ts)
            self.runJob("xmipp_image_header","-i %s --sampling_rate %f"%(volFile,Ts))
            self._defineOutputs(outputVol=vol)
        else:
            self._defineOutputs(outputVol=self.inputVolume.get())
        self._defineSourceRelation(self.inputVolume, self.outputVol)

        cleanPath(self._getPath(self.vResizedVolFile))
        cleanPath(self._getPath(self.vMaskFile))
        cleanPath(self._getPath(self.vFilteredVolFile))

# --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []

        if hasattr(self, 'outputHand'):
            summary.append('Hand value is: %f' %self.outputHand.get())
            summary.append('Hand values close to 1 mean the structure is predicted to be left handed')
            summary.append('Hand values close to 0 mean the structure is predicted to be right handed')
            if self.outputHand.get() > self.thresholdHand.get():
                summary.append('Volume was flipped as it was deemed to be left handed')
            else:
                summary.append('Volume was not flipped as it was deemed to be right handed')
        else:
            summary.append("Output volume and handedness not ready yet.")
        return summary

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        errors = []

        if self.thresholdAlpha.get() > 1.0 or self.thresholdAlpha.get() < 0.0:
           errors.append("Alpha threshold must be between 0.0 and 1.0")
        if self.thresholdHand.get() > 1.0 or self.thresholdHand.get() < 0.0:
           errors.append("Hand threshold must be between 0.0 and 1.0")

        return errors
