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

from pwem.viewers import EmPlotter, ObjectView, MicrographsView, CtfView
from pwem.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, ZOOM
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

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        group = form.addGroup('Valid CTFs')
        group.addParam('visualizeCtfAccepted', LabelParam,
                       label="Visualize CTFs + max resolution",
                       help="Reference CTF plus a new column with "
                            "resolution up to which reference CTF and target "
                            "reference are similar.")
        group.addParam('visualizeMicsAccepted', LabelParam,
                       label="Visualize passed micrographs",
                       help="Visualize those micrographs associated with "
                            "the considered valid CTFs.")
        group2 = form.addGroup('Discarded CTFs')
        group2.addParam('visualizeCtfDiscarded', LabelParam,
                        label="Visualize discarded CTFs",
                        help="Reference CTF plus a new column with "
                             "resolution up to which reference CTF and target "
                             "reference are similar (for discarded CTFs)")
        group2.addParam('visualizeMicsDiscarded', LabelParam,
                        label="Visualize discarded micrographs",
                        help="Visualize those micrographs associated with "
                             "the discarded CTFs.")
        if self.protocol.calculateConsensus.get():
            form.addParam('justSpace', LabelParam, label="")
            form.addParam('visualizeHistogram', IntParam, default=10,
                          label="Visualize Histogram (Bin size)",
                          help="Histogram of the resolution at which two methods"
                               " are equivalent")

    def _getVisualizeDict(self):
        return {
                 'visualizeCtfAccepted': self._visualizeCtfAccepted,
                 'visualizeMicsAccepted': self._visualizeMicsAccepted,
                 'visualizeCtfDiscarded': self._visualizeCtfDiscarded,
                 'visualizeMicsDiscarded': self._visualizeMicsDiscarded,
                 'visualizeHistogram': self._visualizeHistogram
                }

    def _visualizeCtfs(self, objName):
        views = []

        # display metadata with selected variables
        labels = ('id enabled _micObj._filename _psdFile _ctf2_psdFile '
                  '_xmipp_ctfmodel_quadrant '
                  '_defocusU _defocusV _ctf2_defocus_diff '
                  '_astigmatism _ctf2_astigmatism '
                  '_defocusRatio _ctf2_defocusRatio '
                  '_defocusAngle _ctf2_defocusAngle_diff '
                  '_resolution _ctf2_resolution _consensus_resolution '
                  '_phaseShift _ctf2_phaseShift_diff '
                  '_fitQuality _ctf2_fitQuality ')

        if self.protocol.useCritXmipp.get():
            labels += ' '.join(CtfView.EXTRA_LABELS)

        if self.protocol.hasAttribute(objName):
            set = getattr(self.protocol, objName)
            views.append(ObjectView(
                self._project, set.getObjId(),set.getFileName(),
                viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels}))
        else:
            self.infoMessage('%s does not have %s%s'
                             % (self.protocol.getObjLabel(), objName, 
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

    def _visualizeMics(self, objName):
        views = []

        if self.protocol.hasAttribute(objName):
            views.append(MicrographsView(self.getProject(),
                                         getattr(self.protocol, objName)))
        else:
            self.infoMessage('%s does not have %s%s' 
                             % (self.protocol.getObjLabel(), objName, 
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

    def _visualizeCtfAccepted(self, e=None):
        return self._visualizeCtfs("outputCTF")

    def _visualizeCtfDiscarded(self, e=None):
        return self._visualizeCtfs("outputCTFDiscarded")

    def _visualizeMicsAccepted(self, e=None):
        return self._visualizeMics("outputMicrographs")

    def _visualizeMicsDiscarded(self, e=None):
        return self._visualizeMics("outputMicrographsDiscarded")

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

# TODO: Add histograms using viewer_eliminate_empty_images.plotMultiHistogram()
    
def getStringIfActive(prot):
    return ', yet.' if prot.isActive() else '.'
