# **************************************************************************
# *
# * Authors:     Carlos Oscar Sanchez Sorzano
# *              Estrella Fernandez Gimenez
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

from os.path import exists
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from xmipp3.protocols.protocol_subtomo_map_back import XmippProtSubtomoMapBack

class XmippSubtomoMapBackViewer(ProtocolViewer):
    """ Subtomograms mapped back into the original tomogram, with different representation options
    """
    _label = 'viewer subtomogram map back'
    _targets = [XmippProtSubtomoMapBack]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displaySubtomograms', label='Display subtomograms:',
                      help="""The subtomogram will appear mapped back into the original tomogram location 
                      with the previously selected painting mode""")

    def _getVisualizeDict(self):
        return {'displaySubtomograms': self._viewSubtomograms}

    def _viewSubtomograms(self, paramName=None):
        views=[]
        # fnTomo = self.protocol._getExtraPath("paintedTomogram.mrc")
        # fnSubtomos = self.protocol._getExtraPath("paintedSubtomograms.mrc")
        # if exists(fnTomo):
        #     try:
        #         views.append()
        #     except:
        #         pass

        return views

