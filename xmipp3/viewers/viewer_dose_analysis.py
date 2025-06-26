# **************************************************************************
# *
# * Authors:     Daniel Marchán Torres (da.marchan@cnb.csic.es)
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

from pyworkflow.viewer import Viewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import LabelParam
from pwem.viewers import showj, EmProtocolViewer, ObjectView
from pwem.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE, RENDER
from xmipp3.protocols.protocol_movie_dose_analysis import XmippProtMovieDoseAnalysis
import matplotlib.pyplot as plt
import os



class XmippMovieDoseAnalysisViewer(EmProtocolViewer):
    """ This viewer is intended to visualize the selection made by
        the Xmipp - Movie poisson count protocol.
    """
    _label = 'viewer Movie Dose Analysis'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtMovieDoseAnalysis]


    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('visualizeMovies', LabelParam,
                      label="Visualize accepted movies",
                      help="Visualize movies with the respective dose scores.")
        form.addParam('visualizeDiscardedMovies', LabelParam,
                      label="Visualize discarded movies",
                      help="Visualize discarded movies with the respective dose scores.")
        form.addParam('visualizeDoseVsTime', LabelParam,
                      label="Visualize Dose vs Time",
                      help="Visualize plot dose vs time.")
        form.addParam('visualizeDoseDiffVsTime', LabelParam,
                      label="Visualize Dose difference vs Time",
                      help="Visualize plot dose difference against median dose vs time.")

    def _getVisualizeDict(self):
        return {
                 'visualizeMovies': self._visualizeMoviesF,
                 'visualizeDiscardedMovies' : self._visualizeDiscardedMoviesF,
                 'visualizeDoseVsTime': self._visualizeDoseTime,
                 'visualizeDoseDiffVsTime': self._visualizeDoseDiffTime
                }

    def _visualizeMoviesF(self, e=None):
        return self._visualizeMovies("outputMovies")

    def _visualizeDiscardedMoviesF(self, e=None):
        return self._visualizeMovies("outputMoviesDiscarded")

    def _visualizeDoseTime(self, e=None):
        if os.path.exists(self.protocol.getDosePlot()):
            image = plt.imread(self.protocol.getDosePlot())
            # Get the image dimensions (height, width)
            height, width, _ = image.shape
            # Convert pixels to inches for the figure size (assuming 100 DPI)
            dpi = 100
            figsize = (width / dpi, height / dpi)
            # Create the figure with the calculated size
            plt.figure(figsize=figsize)
            # Display the image without axes
            fig = plt.imshow(image)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            # Show the image
            plt.show()

    def _visualizeDoseDiffTime(self, e=None):
        if os.path.exists(self.protocol.getDoseDiffPlot()):
            image = plt.imread(self.protocol.getDoseDiffPlot())
            # Get the image dimensions (height, width)
            height, width, _ = image.shape
            # Convert pixels to inches for the figure size (assuming 100 DPI)
            dpi = 100
            figsize = (width / dpi, height / dpi)
            # Create the figure with the calculated size
            plt.figure(figsize=figsize)
            # Display the image without axes
            fig = plt.imshow(image)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            # Show the image
            plt.show()

    def _visualizeMovies(self, objName):
        views = []

        labels = 'id _filename _samplingRate _acquisition._dosePerFrame ' \
                 '_acquisition._doseInitial _MEAN_DOSE_PER_ANGSTROM2 _STD_DOSE_PER_ANGSTROM2 ' \
                 '_DIFF_TO_DOSE_PER_ANGSTROM2 '

        if self.protocol.hasAttribute(objName):
            setMovies = getattr(self.protocol, objName)
            views.append(ObjectView(
                self._project, setMovies.getObjId(), setMovies.getFileName(),
                viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels})) # FIXME: DO NOT RENDER THE FILENAME
        else:
            self.infoMessage('%s does not have %s%s'
                             % (self.protocol.getObjLabel(), objName,
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

def getStringIfActive(prot):
    return ', yet.' if prot.isActive() else '.'