# **************************************************************************
# *
# * Authors:  Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es), May 2013
# *           Slavica Jonic                (jonic@impmc.upmc.fr)
# * Ported to Scipion:
# *           J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es), Nov 2014
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
from os.path import exists, join

from pyworkflow.protocol.params import (EnumParam, NumericRangeParam,
                                        LabelParam, IntParam, FloatParam)
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO
from pwem.viewers import (ObjectView, showj, EmProtocolViewer, ChimeraAngDist)

from pwem.emlib import (MDL_SAMPLINGRATE, MDL_ANGLE_ROT, MDL_ANGLE_TILT,
                        MDL_RESOLUTION_FREQ, MDL_RESOLUTION_FRC, MetaData)
from xmipp3.protocols.protocol_deep_align import XmippProtDeepAlign

ANGDIST_2DPLOT = 0
ANGDIST_HEATMAP = 1


class XmippDeepAlignViewer(EmProtocolViewer):
    """ Visualize the output of protocol reconstruct highres """
    _label = 'viewer deep align'
    _targets = [XmippProtDeepAlign]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        group = form.addGroup('Particles')
        group.addParam('showOutputParticles', LabelParam, default=False, label='Display output particles')
        group.addParam('showAngDist', EnumParam, choices=['2D plot', 'heatmap'],
                       display=EnumParam.DISPLAY_HLIST, default=ANGDIST_2DPLOT,
                       label='Display angular distribution',
                       help='*2D plot*: display angular distribution as interative 2D in matplotlib.')
        group.addParam('spheresScale', IntParam, default=-1,
                       expertLevel=LEVEL_ADVANCED,
                       label='Spheres size')

    def _getVisualizeDict(self):
        return {
            'showOutputParticles': self._showOutputParticles,
            'showAngDist': self._showAngularDistribution
        }

    def _showOutputParticles(self, paramName=None):
        views = []
        if hasattr(self.protocol, "outputParticles"):
            obj = self.protocol.outputParticles
            fn = obj.getFileName()
            labels = 'id enabled _filename '
            labels += '_ctfModel._defocusU _ctfModel._defocusV _xmipp_shiftX _xmipp_shiftY _xmipp_angleTilt '
            labels += ' _xmipp_angleRot  _xmipp_anglePsi _xmipp_maxCC _xmipp_weight '
            views.append(ObjectView(self._project, obj.strId(), fn,
                                    viewParams={showj.ORDER: labels,
                                                showj.VISIBLE: labels,
                                                showj.MODE: showj.MODE_MD,
                                                showj.RENDER: '_filename'}))
        return views

    # ===============================================================================
    # showAngularDistribution
    # ===============================================================================
    def _showAngularDistribution(self, paramName=None):
        views = []
        angDist = self._createAngDist2D(heatmap=self.showAngDist == ANGDIST_HEATMAP)
        if angDist is not None:
            views.append(angDist)
        return views

    def _iterAngles(self, fnAngles):
        md = MetaData(fnAngles)
        for objId in md:
            rot = md.getValue(MDL_ANGLE_ROT, objId)
            tilt = md.getValue(MDL_ANGLE_TILT, objId)
            yield rot, tilt

    def _createAngDist2D(self, heatmap):
        fnDir = self.protocol._getExtraPath()
        fnAngles = join(fnDir, "outConesParticles.xmd")
        view = None
        if not exists(fnAngles):
            fnAngles = join(fnDir,"outputParticles.xmd")
        if exists(fnAngles):
            fnAnglesSqLite = join(fnDir, "outConesParticles.sqlite")
            from pwem.viewers import EmPlotter
            if not exists(fnAnglesSqLite):
                from pwem.emlib.metadata import getSize
                self.createAngDistributionSqlite(fnAnglesSqLite, getSize(fnAngles),
                                                 itemDataIterator=self._iterAngles(fnAngles))
            view = EmPlotter(x=1, y=1, windowTitle="Angular distribution")
            if heatmap:
                axis = view.plotAngularDistributionFromMd(fnAnglesSqLite, '', histogram=True)
                view.getFigure().colorbar(axis)
            else:
                view.plotAngularDistributionFromMd(fnAnglesSqLite, '')
        return view
