# **************************************************************************
# *
# * Authors:     Estrella Fernandez Gimenez
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

from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import IntParam, LabelParam
from pwem.viewers import ObjectView, showj
from xmipp3.protocols.protocol_subtract_projection import XmippProtSubtractProjection


class XmippSubtractProjectionViewer(ProtocolViewer):
    """ Visualization of the output of subtract projection protocol """
    _label = 'viewer subtract projection'
    _targets = [XmippProtSubtractProjection]
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
        form.addParam('displayR2', LabelParam, default=False, label='Display subtracted particles')

    def _getVisualizeDict(self):
        return {'displayR2': self._viewR2}

    def _viewR2(self, paramName=None):
        views = []
        if hasattr(self.protocol, "outputParticles"):
            obj = self.protocol.outputParticles
            fn = obj.getFileName()
            labels = 'id enabled _index _filename _xmipp_R2subtraction'
            views.append(ObjectView(self._project, obj.strId(), fn, viewParams={showj.ORDER: labels,
                                                                                showj.VISIBLE: labels,
                                                                                showj.RENDER: '_filename',
                                                                                showj.MODE: showj.MODE_MD}))
        return views