# **************************************************************************
# *
# * Authors:     Roberto Marabini (roberto@cnb.csic.es)
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

from pyworkflow.em.viewers import EmPlotter, ObjectView, MicrographsView
from pyworkflow.em.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, ZOOM
from pyworkflow.protocol.params import IntParam, LabelParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

from xmipp3.protocols.protocol_ctf_consensus import XmippProtCTFConsensus


class XmippCTFConsensusViewer(ProtocolViewer):
    """ This protocol computes the maximum resolution up to which two
     CTF estimations would be ``equivalent'', defining ``equivalent'' as having
      a wave aberration function shift smaller than 90 degrees
    """
    _label = 'viewer CTF Consensus'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtCTFConsensus]
    _memory = False
    resolutionThresholdOLD = -1
    # temporary metadata file with ctf that has some resolution greathan than X
    tmpMetadataFile = 'viewersTmp.sqlite'

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        group = form.addGroup('Valid CTFs')
        group.addParam('visualizePairs', LabelParam,
                       label="Visualize CTFs + max resolution",
                       help="Reference CTF plus a new column with "
                            "resolution up to which reference CTF and target "
                            "reference are similar.")
        group.addParam('visualizeMics', LabelParam,
                       label="Visualize passed micrographs",
                       help="Visualize those micrographs associated with "
                            "the considered valid CTFs.")
        group2 = form.addGroup('Discarded CTFs')
        group2.addParam('visualizePairsDiscarded', LabelParam,
                        label="Visualize discarded CTFs",
                        help="Reference CTF plus a new column with "
                             "resolution up to which reference CTF and target "
                             "reference are similar (for discarded CTFs)")
        group2.addParam('visualizeMicsDiscarded', LabelParam,
                        label="Visualize discarded micrographs",
                        help="Visualize those micrographs associated with "
                             "the discarded CTFs.")
        form.addParam('justSpace', LabelParam, label="")
        form.addParam('visualizeHistogram', IntParam, default=10,
                      label="Visualize Histogram (Bin size)",
                      help="Histogram of the resolution at which two methods"
                           " are equivalent")

    def _getVisualizeDict(self):
        return {
                 'visualizePairs': self._visualizePairs,
                 'visualizeMics': self._visualizeMics,
                 'visualizePairsDiscarded': self._visualizePairsDiscarded,
                 'visualizeMicsDiscarded': self._visualizeMicsDiscarded,
                 'visualizeHistogram': self._visualizeHistogram
                }

    def _visualizePairs(self, e=None):
        views = []

        # display metadata with selected variables
        labels = 'id enabled _psdFile _micObj_filename _resolution ' \
                 '_xmipp_consensus_resolution _xmipp_discrepancy_astigmatism' \
                 ' _defocusU _defocusV _defocusAngle'
        if hasattr(self.protocol, "outputCTF"):
            views.append(ObjectView(
                self._project, self.protocol.strId(),
                self.protocol.outputCTF.getFileName(),
                viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels}))
        else:
            self.infoMessage('%s do not have outputCTF, yet.'
                             % self.protocol.getObjLabel(),
                             title='Info message').show()
        return views

    def _visualizeMics(self, e=None):
        views = []

        if hasattr(self.protocol, "outputMicrographs"):
            views.append(MicrographsView(self.getProject(),
                                         self.protocol.outputMicrographs))
        else:
            self.infoMessage('%s do not have outputMicrographs, yet.'
                             % self.protocol.getObjLabel(),
                             title='Info message').show()
        return views

    def _visualizePairsDiscarded(self, e=None):
        views = []

        # display metadata with selected variables
        labels = 'id enabled _psdFile _micObj_filename _resolution ' \
                 '_xmipp_consensus_resolution _xmipp_discrepancy_astigmatism' \
                 ' _defocusU _defocusV _defocusAngle'

        if hasattr(self.protocol, "outputCTFDiscarded"):
            views.append(ObjectView(
                self._project, self.protocol.strId(),
                self.protocol.outputCTFDiscarded.getFileName(),
                viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels}))
        else:
            self.infoMessage('%s has not discarded CTFs'
                             % self.protocol.getObjLabel(),
                             title='Info message').show()
        return views

    def _visualizeMicsDiscarded(self, e=None):
        views = []

        if hasattr(self.protocol, "outputMicrographsDiscarded"):
            views.append(MicrographsView(self.getProject(),
                                         self.protocol.outputMicrographsDiscarded))
        else:
            self.infoMessage('%s do not have outputMicrographsDiscarded.'
                             % self.protocol.getObjLabel(),
                             title='Info message').show()
        return views

    def _visualizeHistogram(self, e=None):
        views = []
        numberOfBins = self.visualizeHistogram.get()
        if hasattr(self.protocol, "outputCTF"):
            numberOfBins = min(numberOfBins, self.protocol.outputCTF.getSize())
            plotter = EmPlotter()
            plotter.createSubPlot("Resolution Discrepancies histogram",
                                  "Resolution (A)", "# of Comparisons")
            resolution = [ctf.getResolution() for ctf in
                          self.protocol.outputCTF]
            plotter.plotHist(resolution, nbins=numberOfBins)
            views.append(plotter)

        if hasattr(self.protocol, "outputCTFDiscarded"):
            numberOfBins = min(numberOfBins,
                               self.protocol.outputCTFDiscarded.getSize())
            plotter = EmPlotter()
            plotter.createSubPlot(
                "Resolution Discrepancies histogram (discarded)",
                "Resolution (A)", "# of Comparisons")
            resolution = [ctf.getResolution() for ctf in
                          self.protocol.outputCTFDiscarded]
            plotter.plotHist(resolution, nbins=numberOfBins)
            views.append(plotter)
        return views