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

import torch
import time
import numpy as np
import os

from torch import nn
from torch import optim

from xmipp3.base import XmippProtocol


class XmippProtDeepHand(XmippProtocol, EMProtocol):

    _label ="deep hand"

    def __init__(self, *args, **kwargs):
        XmippProtocol.__init__(self, *args, **kwargs)
        EMProtocol.__init__(self, *args, **kwargs)

    def _defineParams(self, form):

        form.addSection('Input')
        form.addParam('inputVolume', PointerParam, pointerClass="Volume",
                      label='Input Volume', allowsNull=False,
                      important=True, help="Volume to process")
        form.addParam('inputMask', PointerParam, pointerClass="Volume",
                      label='Input Mask', allowsNull=False,
                      important=True, help="Non background voxel mask")
        form.addParam('thresholdAlpha', FloatParam, label='Alpha Threshold',
                      default=0.7, help="Threshold for alpha helix determination")

# --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('predictStep')
        self._insertFunctionStep('createOutputStep')

    def predictStep(self):

        # Create pipeline object from saved models DTlK
        alpha_model= self.getModel('deepHand', '5A_SSE_experimental.pth')
        hand_model = self.getModel('deepHand', '5A_TL_hand_alpha.pth')
        pipeline = HaPi(alpha_model, hand_model)

        # Obtain Volume and Mask
        ih = ImageHandler()
        Vf = ih.read(self.inputVolume.get()).getData()
        Vmask = ih.read(self.inputMask.get()).getData()

        # Predict hand
        hand = pipeline.predict(Vf, Vmask, self.thresholdAlpha.get(), 2048)
        self.hand = Float(hand)

    def createOutputStep(self):
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

# --------------------------- MODEL -----------------------------------------


class EM3DNet(nn.Module):
    """ 3D CNN to estiamte labels from a set of boxes."""

    def __init__(self):
        super().__init__()

        # Convlutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4,
                               kernel_size=5, padding=1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8,
                               kernel_size=5, padding=1)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16,
                               kernel_size=3, padding=0)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=0)
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size=2, padding=0)

        # Linear layers
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Assumes boxes have dimension 11x11x11
        # Pass the input tensor through the CNN operations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        # Flatten the tensor into a vector
        x = x.view(-1, 512)
        # Pass the tensor through the FC layes
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

class AlphaVolNet(EM3DNet):
    """Model that detects alpha helices within a volume."""

    def __init__(self, trained_model):

        super().__init__()

        # Box dimensions set at 11x11x11 from design of CNN
        self.box_dim = 11

        # Load devices availbale
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

        # Load model
        self.model_file = trained_model
        state_dict = torch.load(self.model_file, map_location=self.device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(self, x):
        """Predict a label for each box."""

        # Load data to available device
        x = x.to(self.device)

        # Pass through network
        pred = self.forward(x)

        return pred

    def normalize(self, box):
        """ Normalize boxes to between 0 and 1."""

        box[box < 0] = 0
        if np.min(box) != np.max(box):
            box = (box-np.min(box))/(np.max(box)-np.min(box))

        return box

    def predict_volume(self, Vf, Vmask, batch_size):
        """Predict regions within volume that contain alpha helices."""

        # Array to store probabilities
        alpha_probs = np.zeros(Vf.shape)

        # Get coordinates of mask
        coors = np.argwhere((Vmask))

        # Variables for boxes
        box_hw = int((self.box_dim-1)/2)
        n_batches = int(np.ceil(coors.shape[0]/batch_size))

        # Pack set of boxes into batches to pass through network
        for i in range(n_batches):

            # Last batch is smaller
            if i == n_batches-1:
                batch = coors.shape[0] % batch_size
            else:
                batch = batch_size

            # Array to store batches
            boxes = np.zeros([batch, self.box_dim, self.box_dim, self.box_dim])

            # Obtain for each batch the normalized box at each loaction
            for j in range(batch):
                x, y, z = coors[i*batch_size+j]
                box = Vf[x-box_hw:x+box_hw+1, y -
                         box_hw:y+box_hw+1, z-box_hw:z+box_hw+1]
                # If box not the correct shape ignore
                if box.shape == (self.box_dim, self.box_dim, self.box_dim):
                    boxes[j, :, :, :] = self.normalize(box)

            # Run through network
            boxes = torch.from_numpy(boxes.astype(np.float32))[
                :, None, :, :, :]
            predictions = self.predict(boxes)

            # Store in appropriate part of the volume
            for j, pred in enumerate(predictions.flatten()):
                x, y, z = coors[i*batch_size+j]
                alpha_probs[x, y, z] = float(pred)

        return alpha_probs

    def precision(self, Vf, Vmask, Vmask_alpha, thr, batch_size):
        """Given a threshold for probabilities evaluate precision."""

        # Obtain location of alphahelics
        alpha_probs = self.predict_volume(Vf, Vmask, batch_size)

        # Alpha mask by thresholding
        alpha_mask = alpha_probs > thr

        # Find number of true positivies and false positives
        TP = np.sum(np.logical_and(alpha_mask, Vmask_alpha))
        FP = np.sum(np.logical_and(alpha_mask, np.logical_not(Vmask_alpha)))

        # If no TP or FP return nan to avoid division by zero
        if TP+FP == 0:
            precision = np.nan
        else:
            precision = TP/(TP+FP)

        return TP, FP, precision


class HandNet(EM3DNet):
    """Model that predicts a volumes handedness from given alpha mask."""

    def __init__(self, trained_model):

        super().__init__()

        # Box dimensions set at 11x11x11 from design of CNN
        self.box_dim = 11

        # Load devices availbale
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

        # Load model
        self.model_file = trained_model
        state_dict = torch.load(self.model_file, map_location=self.device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(self, x):
        """Predict a label for each box."""

        # Load data to available device
        x = x.to(self.device)

        # Pass through network
        pred = self.forward(x)

        return pred

    def normalize(self, box):
        """Normalize boxes to between 0 and 1."""

        box[box < 0] = 0
        if np.min(box) != np.max(box):
            box = (box-np.min(box))/(np.max(box)-np.min(box))

        return box

    def predict_volume_consensus(self, Vf, Alpha_mask, batch_size):
        """Predict volume handedness by means of consensus voting."""

        # Hand predictions
        hand_predictions = np.zeros(np.sum(Alpha_mask))

        # Get coordinates of alphas
        coors = np.argwhere((Alpha_mask))

        # Variables for boxes
        box_hw = int((self.box_dim-1)/2)
        n_batches = int(np.ceil(coors.shape[0]/batch_size))

        # Pack set of boxes into batches to pass through network
        for i in range(n_batches):

            # Last batch is smaller
            if i == n_batches-1:
                batch = coors.shape[0] % batch_size
            else:
                batch = batch_size

            # Array to store batches
            boxes = np.zeros([batch, self.box_dim, self.box_dim, self.box_dim])

            # Obtain for each batch the normalized box at each loaction
            for j in range(batch):
                x, y, z = coors[i*batch_size+j]
                box = Vf[x-box_hw:x+box_hw+1, y -
                         box_hw:y+box_hw+1, z-box_hw:z+box_hw+1]
                boxes[j, :, :, :] = self.normalize(box)

            # Run through network
            boxes = torch.from_numpy(boxes.astype(np.float32))[
                :, None, :, :, :]
            predictions = self.predict(boxes)

            # Store predictions
            hand_predictions[i*batch_size:i*batch_size +
                             batch] = predictions.cpu().flatten().numpy()

        # Consensus voting is by taking average
        consensus_predictions = np.mean(hand_predictions)

        return consensus_predictions


class HaPi():
    """ Ha(ndedness) Pi(peline) is a model that predicts from a given
        electron density map volume its handedness.
    """

    def __init__(self, alpha_model, hand_model):

        # Load models
        self.model_alpha = AlphaVolNet(alpha_model)
        self.model_hand = HandNet(hand_model)

    def predict(self, Vf, Vmask, thr, batch_size):
        """ Predict handedness of an electron desnity map.

        A label of 0 means that it has correct handedness, conversely
        a label of 1 means that it has incorrect hand.

        Parameters:
        -----------
        Vf -- Numpy array of volume density values
        Vmask -- Nummpy array mask of non-background values
        thr -- Threshold for the probabilities of alpha estimation
        batch_size -- Number of boxes to pass to prediction model at time

        """

        # Obtain location of alphahelics
        alpha_probs = self.model_alpha.predict_volume(Vf, Vmask, batch_size)

        # Alpha mask by thresholding
        alpha_mask = alpha_probs > thr

        # Predict hand
        if alpha_mask.any():
            handness = self.model_hand.predict_volume_consensus(
                Vf, alpha_mask, batch_size)
        else:
            handness = None

        return handness

    def evaluate(self, Vf, Vmask, thr, batch_size, label):
        """ Evaluate wether prediction matches given label."""

        # Predict hand
        prediction = self.predict(Vf, Vmask, thr, batch_size)

        if prediction is None:
            result = np.nan
        else:
            # Assign hand
            hand = prediction > 0.5

            # Check wether it is correct
            result = hand == label

        return prediction, result
