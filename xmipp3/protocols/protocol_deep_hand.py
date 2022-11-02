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
from pwem.emlib.image import ImageHandler
from pwem.objects import Volume

from pyworkflow.protocol.params import PointerParam, FloatParam, IntParam
from pyworkflow.object import Float
from pyworkflow.utils.path import cleanPath

from xmipp3.base import XmippProtocol
from xmipp3.convert import getImageLocation

class XmippProtDeepHand(EMProtocol, XmippProtocol):
    """Protocol to returns handedness of structure from trained deep learning model
    """

    _label ="deep hand"
    _conda_env = "xmipp_deepHand"

    def __init__(self, *args, **kwargs):
        EMProtocol.__init__(self, *args, **kwargs)
        XmippProtocol.__init__(self)

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
        self.resizedVolFile = self._getPath('resizedVol.mrc')
        self.maskFile = self._getPath('mask.mrc')
        self.filteredVolFile = self._getPath('filteredVol.mrc')

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
                self.thresholdAlpha.get(), self._getPath('filteredVol.mrc'),
                self._getPath('mask.mrc'))
        self.runJob("xmipp_deep_hand", args, env=self.getCondaEnv())

        # Store hand value
        hand_file = self._getExtraPath('hand.txt')
        f = open(hand_file)
        self.hand = Float(float(f.read()))
        f.close()

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

        cleanPath(self._getPath('resizedVol.mrc'))
        cleanPath(self._getPath('mask.mrc'))
        cleanPath(self._getPath('filteredVol.mrc'))

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
