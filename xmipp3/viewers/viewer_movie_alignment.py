# **************************************************************************
# *
# * Authors:     J.M. de la Rosa Trevin (jmdelarosa@cnb.csic.es)
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

from pyworkflow.viewer import Viewer, DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
import pwem.viewers.showj as showj
from pyworkflow.protocol.params import LabelParam

from xmipp3.protocols.protocol_movie_opticalflow import (XmippProtOFAlignment,
                                                 OBJCMD_MOVIE_ALIGNCARTESIAN)
from xmipp3.protocols.protocol_movie_correlation import XmippProtMovieCorr
from xmipp3.protocols.protocol_movie_max_shift import XmippProtMovieMaxShift
from .viewer_ctf_consensus import getStringIfActive

class XmippMovieAlignViewer(Viewer):
    _targets = [XmippProtOFAlignment, XmippProtMovieCorr]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    _label = 'viewer optical/correlation alignment'

    def _visualize(self, obj, **kwargs):
        views = []

        if obj.hasAttribute('outputMicrographs'):
            views.append(self.objectView(obj.outputMicrographs,
                                         viewParams=getViewParams()))
        elif obj.hasAttribute('outputMovies'):
            views.append(self.objectView(obj.outputMovies,
                                         viewParams=getViewParams()))
        else:
            views.append(self.infoMessage("Output (micrographs or movies) has "
                                          "not been produced yet."))

        return views


class XmippMovieMaxShiftViewer(ProtocolViewer):
    """ This protocol computes the maximum resolution up to which two
     CTF estimations would be ``equivalent'', defining ``equivalent'' as having
      a wave aberration function shift smaller than 90 degrees
    """
    _label = 'viewer Movie Max Shift'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtMovieMaxShift]
    _memory = False
    resolutionThresholdOLD = -1
    # temporary metadata file with ctf that has some resolution greathan than X
    tmpMetadataFile = 'viewersTmp.sqlite'

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('visualizeMics', LabelParam,
                       label="Visualize passed micrographs",
                       help="Visualize those micrographs considered valid.")
        form.addParam('visualizeMicsDiscarded', LabelParam,
                        label="Visualize discarded micrographs",
                        help="Visualize discarded micrographs.")

    def _getVisualizeDict(self):
        return {
                 'visualizeMics': self._visualizeMics,
                 'visualizeMicsDiscarded': self._visualizeMicsDiscarded
                }

    def _visualizeAny(self, outNameCondition):
        views = []

        for outName, outObj in self.protocol.iterOutputAttributes():
            if outNameCondition(outName):
                views.append(self.objectView(outObj, viewParams=getViewParams()))
                break
        if not views:
            outputType = 'discarded' if outNameCondition('Discarded') else 'accepted'
            self.infoMessage('%s does not have %s outputs%s'
                             % (self.protocol.getObjLabel(), outputType,
                                getStringIfActive(self.protocol)),
                             title='Info message').show()
        return views

    def _visualizeMics(self, e=None):
        return self._visualizeAny(lambda x: not x.endswith('Discarded'))

    def _visualizeMicsDiscarded(self, e=None):
        return self._visualizeAny(lambda x: x.endswith('Discarded'))

def getViewParams():
    plotLabels = ('psdCorr._filename plotPolar._filename '
                  'plotCart._filename plotGlobal._filename')
    labels = plotLabels + ' _filename '
    viewParams = {showj.MODE: showj.MODE_MD,
                  showj.ORDER: 'id ' + labels,
                  showj.VISIBLE: 'id ' + labels,
                  showj.RENDER: plotLabels,
                  showj.ZOOM: 20,
                  showj.OBJCMDS: "'%s'" % OBJCMD_MOVIE_ALIGNCARTESIAN
                  }

    return viewParams
