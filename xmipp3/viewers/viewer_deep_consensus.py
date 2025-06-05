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

from pwem.viewers import ObjectView
from pwem.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, SORT_BY
from pyworkflow.protocol.params import IntParam, LabelParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

from pwem import emlib
from xmipp3.protocols.protocol_screen_deepConsensus import XmippProtScreenDeepConsensus
from .plotter import XmippPlotter

class XmippDeepConsensusViewer(ProtocolViewer):
    """         
        Viewer for the 'Xmipp - deep consensus picker' and
        'Xmipp - screen deep learning' protocols.\n
        Select those particles/cooridantes with high (close to 1.0) 'zScoreDeepLearning1' value and save them.
        The Histogram may help you to decide a threshold.
    """
    _label = 'viewer deepConsensus'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtScreenDeepConsensus]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('noteViz', LabelParam, label="\n")
        form.addParam('visualizeParticles', LabelParam, important=True,
                      label="Select particles/coordinates with high "
                            "'zScoreDeepLearning1' values",
                      help="A viewer with all particles/coordinates "
                           "with a 'zScoreDeepLearning1' attached "
                           "will be launched. Select all those "
                           "particles/coordinates with high scores and "
                           "save them.\n"
                           "Particles can be sorted by any column.")
        form.addParam('visualizeHistogram', IntParam, default=100,
                      label="Visualize Deep Scores Histogram (Bin size)",
                      help="Plot a histogram of the 'zScoreDeepLearning1' "
                           "to visual setting of a threshold.")

    def _getVisualizeDict(self):
        return {'visualizeParticles': self._visualizeParticles,
                'visualizeHistogram': self._visualizeHistogram}

    def _visualizeParticles(self, e=None):
        views = []

        labels = 'id enabled _index _filename _xmipp_zScoreDeepLearning1 '
        labels += '_xmipp_zScore _xmipp_cumulativeSSNR '
        labels += '_xmipp_scoreEmptiness'

        otherParam = {}
        objId = 0
        if (isinstance(self.protocol, XmippProtScreenDeepConsensus) and
            self.protocol.hasAttribute('outputCoordinates')):
            fnParts = self.protocol._getPath("particles.sqlite")
            objId = self.protocol.outputCoordinates.strId()
            otherParam = {'other': 'deepCons'}


        if objId:
            views.append(ObjectView(
                self._project, objId, fnParts,
                viewParams={ORDER: labels, VISIBLE: labels,
                            SORT_BY: '_xmipp_zScoreDeepLearning1 asc',
                            RENDER: '_filename',
                            MODE: MODE_MD}, **otherParam))
        else:
            print(" > Not output found, yet.")

        return views

    def _visualizeHistogram(self, e=None):
        views = []
        numberOfBins = self.visualizeHistogram.get()

        fnXml = self.protocol._getPath('particles.xmd')
        if os.path.isfile(fnXml):
            md = emlib.MetaData(fnXml)
            if md.containsLabel(emlib.MDL_ZSCORE_DEEPLEARNING1):
                xplotter = XmippPlotter(windowTitle="Deep consensus score")
                xplotter.createSubPlot("Deep consensus score",
                                       "Deep consensus score",
                                       "Number of Particles")
                xplotter.plotMd(md, False,
                                mdLabelY=emlib.MDL_ZSCORE_DEEPLEARNING1,
                                nbins=numberOfBins)
                views.append(xplotter)
            else:
                print(" > '%s' don't have 'xmipp_zScoreDeepLearning1' label."
                      % fnXml)
        else:
            print(" > Metadata file is not found in '%s'" % fnXml)

        return views
