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

from pyworkflow.protocol.params import PointerParam, FloatParam
from pyworkflow.object import Float

from xmipp3.base import XmippProtocol
from xmipp3.convert import getImageLocation

class XmippProtDeepHand(EMProtocol, XmippProtocol):

    _label ="deep hand"
    _cond_env = "xmipp_deepHand"

    def __init__(self, *args, **kwargs):
        EMProtocol.__init__(self, *args, **kwargs)
        XmippProtocol.__init__(self, *args, **kwargs)

    def _defineParams(self, form):

        form.addSection('Input')
        form.addParam('inputVolume', PointerParam, pointerClass="Volume",
                      label='Input Volume', allowsNull=False,
                      important=True, help="Volume to process")
        form.addParam('threshold', FloatParam, label='Threshold Mask',
                      allowsNull=False, important=True,  help="Threshold for mask creation")
        form.addParam('thresholdAlpha', FloatParam, label='Alpha Threshold',
                      default=0.7, help="Threshold for alpha helix determination")

# --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('preprocessStep')
        self._insertFunctionStep('predictStep')
        self._insertFunctionStep('createOutputStep')

    def preprocessStep(self):

        # Get volume information
        volume = self.inputVolume.get()
        fnVol = getImageLocation(volume)
        Ts = volume.getSamplingRate()

        # Paths to new files created
        self.resizedVolFile = self._getPath('resizedVol.mrc')
        self.maskFile = self._getPath('mask.mrc')
        self.filteredVolFile = self._getPath('filteredVol.mrc')

        # Resize to 1A/px
        self.runJob("xmipp_image_resize", "-i %s -o %s --factor %f" %
                    (fnVol, self.resizedVolFile, Ts))

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
        args = "%s %s %s %f %s %s" % (
                alpha_model, hand_model, self._getExtraPath(),
                self.thresholdAlpha.get(), self.filteredVolFile, self.maskFile)
        self.runJob("xmipp_deep_hand", args, env=self.getCondaEnv())

    def createOutputStep(self):
        handFile = self._getExtraPath('hand.txt')
        f = open(handFile)
        self.hand = Float(float(f.read()))
        f.close()
        self._defineOutputs(outputHand=self.hand)

# --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        errors = []
        return errors
