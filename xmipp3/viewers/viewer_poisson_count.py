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
from pwem.viewers.showj import MODE, MODE_MD, ORDER, VISIBLE
from xmipp3.protocols.protocol_movie_poisson_count import XmippProtMoviePoissonCount
import matplotlib.pyplot as plt



class XmippMoviePoissonCountViewer(EmProtocolViewer):
    """ This viewer is intended to visualize the selection made by
        the Xmipp - Movie poisson count protocol.
    """
    _label = 'viewer Movie Poisson Count'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtMoviePoissonCount]


    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('visualizeMovies', LabelParam,
                      label="Visualize movies",
                      help="Visualize movies with the respective dose scores.")
        form.addParam('visualizeDoseVsTime', LabelParam,
                      label="Visualize Dose vs Time",
                      help="Visualize plot dose vs time.")
        form.addParam('visualizeDoseDiffVsTime', LabelParam,
                      label="Visualize Dose difference vs Time",
                      help="Visualize plot dose difference against median dose vs time.")

    def _getVisualizeDict(self):
        return {
                 'visualizeMovies': self._visualizeMoviesF,
                 'visualizeDoseVsTime': self._visualizeDoseTime,
                 'visualizeDoseDiffVsTime': self._visualizeDoseDiffTime
                }

    def _visualizeMoviesF(self, e=None):
        return self._visualizeMovies("outputMovies")

    def _visualizeDoseTime(self, e=None):
        views = []
        if self.protocol.hasAttribute("outputMovies"):
            image = plt.imread(self.protocol.getDosePlot())
            plt.figure()
            fig = plt.imshow(image)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()

        #return views

    def _visualizeDoseDiffTime(self, e=None):
        views = []
        if self.protocol.hasAttribute("outputMovies"):
            image = plt.imread(self.protocol.getDoseDiffPlot())
            plt.figure()
            fig = plt.imshow(image)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.show()

        #return views

    def _visualizeMovies(self, objName):
        views = []

        labels = 'id _filename _samplingRate _acquisition._dosePerFrame' \
                 ' _acquisition._doseInitial _MEAN_DOSE_PER_FRAME _STD_DOSE_PER_FRAME'

        if self.protocol.hasAttribute(objName):
            set = getattr(self.protocol, objName)
            views.append(ObjectView(
                self._project, set.getObjId(), set.getFileName(),
                viewParams={MODE: MODE_MD, ORDER: labels, VISIBLE: labels}))
        else:
            self.infoMessage('%s does not have %s%s'
                             % (self.protocol.getObjLabel(), objName,
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

def getStringIfActive(prot):
    return ', yet.' if prot.isActive() else '.'