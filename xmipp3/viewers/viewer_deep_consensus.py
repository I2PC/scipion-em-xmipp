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

from pyworkflow.em.viewers import EmPlotter, ObjectView
from pyworkflow.em.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, SORT_BY
from pyworkflow.protocol.params import IntParam, LabelParam, StringParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

import xmippLib
from xmipp3.protocols.protocol_screen_deepConsensus import XmippProtScreenDeepConsensus
from xmipp3.protocols.protocol_screen_deeplearning import XmippProtScreenDeepLearning


class XmippDeepConsensusViewer(ProtocolViewer):
    """ This protocol computes the maximum resolution up to which two
     CTF estimations would be ``equivalent'', defining ``equivalent'' as having
      a wave aberration function shift smaller than 90 degrees
    """
    _label = 'viewer Deep Consensus'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtScreenDeepConsensus, XmippProtScreenDeepLearning]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('noteViz', LabelParam,
                      label='A score has been attached to all the input '
                            'coordinates in order to set a manual threshold.')
        form.addParam('visualizeParticles', LabelParam,
                      label="Select particles/coordinates", important=True,
                      help="A viewer with all the extracted particles "
                           "will be launched. Every particle have "
                           "the 'deepConsensus Score' attached.\n"
                           "Select all those particles and click on "
                           "'+Coordinates' to save the selection.\n"
                           "Particles can be sort by any column.")
        form.addParam('visualizeHistogram', IntParam, default=200,
                      label="Visualize Histogram (Bin size)",
                      help="Histogram of the 'deepConsensus Score' "
                           "to help in the thresholding.")
        form.addParam('noteViz2', LabelParam,
                      label='*Notice, the outputParticles have a reduced '
                            'size since they have been extracted only to '
                            'inspection proporses. Therefore, a full size of '
                            'setOfParticles must be extracted using the '
                            'selected coordinates from this viewer.')

    def _getVisualizeDict(self):
        return {'visualizeParticles': self._visualizeParticles,
                'visualizeHistogram': self._visualizeHistogram}

    def _visualizeParticles(self, e=None):
        views = []

        labels = 'id enabled _index _filename _xmipp_zScoreDeepLearning1 '
        labels += '_xmipp_zScore _xmipp_cumulativeSSNR '
        labels += '_xmipp_scoreEmptiness'

        otherParam = {}
        if (isinstance(self.protocol, XmippProtScreenDeepConsensus) and
            self.protocol.hasAttribute('outputCoordinates')):
            coordsId = self.protocol.outputCoordinates.strId()
            otherParam = {'other': '%s,deepCons' % coordsId}

        if hasattr(self.protocol, 'outputParticles'):
            parts = self.protocol.outputParticles
            fnParts = parts.getFileName()

            views.append(ObjectView(
                self._project, parts.strId(), fnParts,
                viewParams={ORDER: labels, VISIBLE: labels,
                            SORT_BY: '_xmipp_zScoreDeepLearning1 asc',
                            RENDER: '_filename',
                            MODE: MODE_MD}, **otherParam))

        return views

    def _visualizeHistogram(self, e=None):
        views = []
        numberOfBins = self.visualizeHistogram.get()

        fnXml = self.protocol._getPath('particles.xmd')
        if os.path.isfile(fnXml):
            md = xmippLib.MetaData(fnXml)
            if md.containsLabel(xmippLib.MDL_ZSCORE_DEEPLEARNING1):
                from plotter import XmippPlotter
                xplotter = XmippPlotter(windowTitle="Deep consensus score")
                xplotter.createSubPlot("Deep consensus score",
                                       "Deep consensus score",
                                       "Number of Particle")
                xplotter.plotMd(md, False,
                                mdLabelY=xmippLib.MDL_ZSCORE_DEEPLEARNING1,
                                nbins=numberOfBins)
                views.append(xplotter)
            else:
                print("'%s' don't have 'xmipp_Zscore_deepLearning1' label."
                      % fnXml)
        else:
            print("Metadata file is not found in '%s'" % fnXml)

        return views
