# **************************************************************************
# *
# * Authors:     L. del Cano (ldelcano@cnb.csic.es)
# *              Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
This module implement the wrappers aroung Xmipp CL2D protocol
visualization program.
"""
from cProfile import label
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import IntParam, LabelParam, BooleanParam

from xmipp3.protocols.protocol_split_volume import XmippProtSplitvolume

import numpy as np
import matplotlib.pyplot as plt

class XmippViewerSplitVolume(ProtocolViewer):
    """ Viewer for Split Volumes protocol
    """
    _label = 'viewer split volume'
    _targets = [XmippProtSplitvolume]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label='Graph')
        form.addParam('displayDistanceImage', LabelParam, label='Angular distance matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the angular distance computed for each image pair')
        form.addParam('displayCorrelationImage', LabelParam, label='Correlation matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the correlation computed for each image pair')
        form.addParam('displayWeightImage', LabelParam, label='Weight matrix',
                        help='Shows an image where each pixel\'s colour corresponds to '
                        'the weight of each graph edge')
        form.addParam('displayLabelImage', LabelParam, label='Classification',
                        help='Shows an image where each column\'s colour corresponds to '
                        'the label assigned to each image')
        form.addParam('displayLabelHistogram', LabelParam, label='Class sizes',
                        help='Shows a bar plot with the sizes of each class')

    def _getVisualizeDict(self):
        return {
            'displayDistanceImage': self._displayDistanceImage,
            'displayCorrelationImage': self._displayCorrelationImage,
            'displayWeightImage': self._displayWeightImage,
            'displayLabelImage': self._displayLabelImage,
            'displayLabelHistogram': self._displayLabelHistogram,
        }
    
    # --------------------------- DEFINE display functions ----------------------
    def _displayDistanceImage(self, e):
        distances = self._readAngularDistances()
        return self._showImagePairImage(distances, 'Angular distances')

    def _displayCorrelationImage(self, e):
        correlations = self._readCorrelations()
        return self._showImagePairImage(correlations, 'Correlation')
    
    def _displayWeightImage(self, e):
        weights = self._readWeights()
        return self._showImagePairImage(weights, 'Weights')

    def _displayLabelImage(self, e):
        labels = self._readLabels()
        labels = np.array([labels]) # It needs to be bidimensional to work

        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(labels, origin='lower', aspect='auto', interpolation='none'))
        ax.set_xlabel('Image number')
        ax.set_title('Classification')
        return [fig]

    def _displayLabelHistogram(self, e):
        labels = self._readLabels()
        nClasses = int(labels.max())+1

        fig, ax = plt.subplots()
        ax.hist(labels, bins=nClasses)
        ax.set_xlabel('Class number')
        ax.set_ylabel('Class size')
        ax.set_title('Class sizes')
        return [fig]

    # --------------------------- UTILS functions -----------------------------
    def _readAngularDistances(self):
        return self.protocol._readAngularDistances()

    def _readCorrelations(self):
        return self.protocol._readCorrelations()

    def _readWeights(self):
        return self.protocol._readWeights()

    def _readLabels(self):
        return self.protocol._readLabels()

    def _showImagePairImage(self, img, title):
        fig, ax = plt.subplots()
        plt.colorbar(ax.imshow(img, origin='lower', aspect='auto', interpolation='none'))
        ax.set_xlabel('Image number')
        ax.set_ylabel('Image number')
        ax.set_title(title)
        return [fig]

