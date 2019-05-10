# **************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
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

from pyworkflow.em.viewers import ObjectView, EmPlotter
from pyworkflow.em.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, SORT_BY
from pyworkflow.protocol.params import IntParam, LabelParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

import xmippLib
from xmipp3.convert import getXmippAttribute
from xmipp3.protocols.protocol_carbon_screen import XmippProtCarbonScreen
from .plotter import XmippPlotter

class XmippDeepConsensusViewer(ProtocolViewer):
    """         
        Viewer for the 'Xmipp - deep carbon cleaner' protocols.\n
        Select those cooridantes with high (close to 1.0)
        'zScoreDeepLearning2' value and save them.
        The Histogram may help you to decide a threshold.
    """
    _label = 'viewer deep Carbon CLeaner'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtCarbonScreen]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('noteViz', LabelParam, label="\n")
        # form.addParam('visualizeParticles', LabelParam, important=True,
        #               label="Select particles/coordinates with high "
        #                     "'zScoreDeepLearning1' values",
        #               help="A viewer with all particles/coordinates "
        #                    "with a 'zScoreDeepLearning1' attached "
        #                    "will be launched. Select all those "
        #                    "particles/coordinates with high scores and "
        #                    "save them.\n"
        #                    "Particles can be sorted by any column.")
        form.addParam('visualizeHistogram', IntParam, default=100,
                      label="Visualize Deep Scores Histogram (Bin size)",
                      help="Plot an histogram of the 'zScoreDeepLearning2' "
                           "to visual setting of a good threshold.")
        form.addParam('visualizeCoordinates', IntParam,  default=100,
                      label="Visualize the good coordinates (threshold)",
                      help="Visualize the coordinates considered good according"
                           " to the threshold indicated in the box.\n"
                           "If you are agree with the result, save the result"
                           " with the '+Coordinates'")

    def _getVisualizeDict(self):
        return {'visualizeCoordinates': self._visualizeCoordinates,
                'visualizeHistogram': self._visualizeHistogram}

    def _visualizeCoordinates(self, e=None):
        views = []

        # labels = 'id enabled _index _filename _xmipp_zScoreDeepLearning1 '
        # labels += '_xmipp_zScore _xmipp_cumulativeSSNR '
        # labels += '_xmipp_scoreEmptiness'
        #
        # otherParam = {}
        # objId = 0
        # if (isinstance(self.protocol, XmippProtScreenDeepConsensus) and
        #     self.protocol.hasAttribute('outputCoordinates')):
        #     fnParts = self.protocol._getPath("particles.sqlite")
        #     objId = self.protocol.outputCoordinates.strId()
        #     otherParam = {'other': 'deepCons'}
        #
        # elif (isinstance(self.protocol, XmippProtScreenDeepLearning) and
        #       self.protocol.hasAttribute('outputParticles')):
        #     parts = self.protocol.outputParticles
        #     fnParts = parts.getFileName()
        #     objId = parts.strId()
        #
        # if objId:
        #     views.append(ObjectView(
        #         self._project, objId, fnParts,
        #         viewParams={ORDER: labels, VISIBLE: labels,
        #                     SORT_BY: '_xmipp_zScoreDeepLearning1 asc',
        #                     RENDER: '_filename',
        #                     MODE: MODE_MD}, **otherParam))
        # else:
        #     print(" > Not output found, yet.")

        return views

    def _visualizeHistogram(self, e=None):
        views = []
        numberOfBins = self.visualizeHistogram.get()

        # if self.protocol.hasAttribute('outputCoordinates'):
        #     outCoords = self.protocol.outputCoordinates
        #     if getXmippAttribute(outCoords.getFirstItem(),
        #                          xmippLib.MDL_ZSCORE_DEEPLEARNING2):
        #         plotter = EmPlotter()
        #         plotter.createSubPlot("Deep carbon score",
        #                               "Deep carbon score",
        #                               "Number of Coordinates")
        #         cScores = [getXmippAttribute(coord, xmippLib.MDL_ZSCORE_DEEPLEARNING2)
        #                    for coord in outCoords]
        #         plotter.plotHist(cScores, nbins=numberOfBins)
        #         views.append(plotter)
        #     else:
        #         print(" > 'outputCoordinates' don't have 'xmipp_zScoreDeepLearning2' label.")
        # else:
        #     print(" > Output not ready yet.")

        return views
