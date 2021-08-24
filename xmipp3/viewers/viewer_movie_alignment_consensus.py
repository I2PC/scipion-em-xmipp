# **************************************************************************
# *
# * Authors:     Daniel Marchan (da.marchan@cnb.csic.es)
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

from pwem.viewers import EmPlotter, ObjectView, MicrographsView, ImageView
from pwem.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER, ZOOM
from pyworkflow.protocol.params import IntParam, LabelParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from xmipp3.protocols.protocol_movie_alignment_consensus import XmippProtConsensusMovieAlignment
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os




class XmippMovieAlignmentConsensusViewer(ProtocolViewer):
    """ This protocol computes the maximum resolution up to which two
     CTF estimations would be ``equivalent'', defining ``equivalent'' as having
      a wave aberration function shift smaller than 90 degrees
    """
    _label = 'viewer Movie Alignment Consensus'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtConsensusMovieAlignment]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        group = form.addGroup('Valid Micrographs')
        group.addParam('visualizeMicsAccepted', LabelParam,
                       label="Visualize Micrographs",
                       help="Reference Mics plus three new columns with "
                            "the global alignment correlation, max error and RMSE of the reference alignment"
                            "and target alignment.")
        group.addParam('visualizeMoviesAccepted', LabelParam,
                       label="Visualize passed movies",
                       help="Visualize those movies associated with "
                            "the considered valid alignment.")
        group2 = form.addGroup('Discarded Micrographs')
        group2.addParam('visualizeMicsDiscarded', LabelParam,
                        label="Visualize discarded Micrographs",
                        help="Reference Mics plus three new columns with "
                             "the global alignment correlation, max error and RMSE of the reference alignment"
                             "and target alignment.(for discarded Mics)")
        group2.addParam('visualizeMoviesDiscarded', LabelParam,
                        label="Visualize discarded movies",
                        help="Visualize those movies associated with "
                             "the discarded alignment.")

        form.addParam('justSpace', LabelParam, condition=self.protocol.trajectoryPlot.get(), label="")
        form.addParam('IDMic', IntParam, condition=self.protocol.trajectoryPlot.get(),
                      label="ID MIC to visualize",
                      help="Graph of the two global alignment trajectories")
        form.addParam('visualizeTrajectories', LabelParam, condition=self.protocol.trajectoryPlot.get(),
                      label="Visualize Global Alignment Trajectories",
                      help="Graph of the two global alignment trajectories")



    def _getVisualizeDict(self):
        return {
            'visualizeMicsAccepted': self._visualizeMicsAccepted,
            'visualizeMoviesAccepted': self._visualizeMoviesAccepted,
            'visualizeMicsDiscarded': self._visualizeMicsDiscarded,
            'visualizeMoviesDiscarded': self._visualizeMoviesDiscarded,
            'visualizeTrajectories': self._visualizeTrajectories
        }

    def _visualizeMics(self, objName):
        views = []
        # display metadata with selected variables
        labels = ('id enabled _filename psdCorr._filename '
                  'plotCart._filename '
                  '_alignment_corr '
                  '_alignment_rmse_error '
                  '_alignment_max_error ')
        render = ('psdCorr._filename '
                  'plotCart._filename')

        if self.protocol.hasAttribute(objName):
            set = getattr(self.protocol, objName)
            views.append(ObjectView(
                self._project, set.getObjId(), set.getFileName(),
                viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels, RENDER: render}))
        else:
            self.infoMessage('%s does not have %s%s'
                             % (self.protocol.getObjLabel(), objName,
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

    def _visualizeMovies(self, objName):
        views = []
        labels = ('id enabled _filename samplingRate '
                  '_alignment._xshifts '
                  '_alignment._yshifts ')
        render = ()
        if self.protocol.hasAttribute(objName):
            set = getattr(self.protocol, objName)
            views.append(ObjectView(self._project, set.getObjId(), set.getFileName(),
                                    viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels, RENDER: render}))
        else:
            self.infoMessage('%s does not have %s%s'
                             % (self.protocol.getObjLabel(), objName,
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

    def _visualizeMicsAccepted(self, e=None):
        return self._visualizeMics("outputMicrographs")

    def _visualizeMicsDiscarded(self, e=None):
        return self._visualizeMics("outputMicrographsDiscarded")

    def _visualizeMoviesAccepted(self, e=None):
        return self._visualizeMovies("outputMovies")

    def _visualizeMoviesDiscarded(self, e=None):
        return self._visualizeMovies("outputMoviesDiscarded")

    def _visualizeTrajectories(self, e=None):
        views = []
        id = self.IDMic.get()

        fn = self.protocol._getTrajectoriesPlot(id)
        if os.path.exists(fn):
            # Y SI HACEMOS UN PLT.READ"
            img = mpimg.imread(fn)
            fig = plt.imshow(img)
            fig.set_cmap('hot')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()
        else:
            print('ID does not exists')
            #numberOfBins = min(numberOfBins, self.protocol.outputCTF.getSize())
            #plotter = EmPlotter()
            #plotter.createSubPlot("Resolution Discrepancies histogram",
             #                     "Resolution (A)", "# of Comparisons")
            #resolution = [ctf.getResolution() for ctf in
             #             self.protocol.outputCTF]
            #plotter.plotHist(resolution, nbins=numberOfBins)
            #views.append(plotter)

        #if hasattr(self.protocol, "outputCTFDiscarded"):
         #   numberOfBins = min(numberOfBins,
          #                     self.protocol.outputCTFDiscarded.getSize())
          #  plotter = EmPlotter()
           # plotter.createSubPlot(
            #    "Resolution Discrepancies histogram (discarded)",
              #   "Resolution (A)", "# of Comparisons")
            #resolution = [ctf.getResolution() for ctf in
             #             self.protocol.outputCTFDiscarded]
            #plotter.plotHist(resolution, nbins=numberOfBins)
            #views.append(plotter)
        return views


# TODO: Add histograms using viewer_eliminate_empty_images.plotMultiHistogram()

def getStringIfActive(prot):
    return ', yet.' if prot.isActive() else '.'
