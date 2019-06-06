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

from glob import glob
from os.path import exists, join

from pyworkflow.protocol.params import LabelParam
from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer

from xmippLib import (MDL_SAMPLINGRATE, MDL_ANGLE_ROT, MDL_ANGLE_TILT,
                   MDL_RESOLUTION_FREQ, MDL_RESOLUTION_FRC, MetaData)
from xmipp3.convert import getImageLocation
from xmipp3.protocols.protocol_deep_cones3D import XmippProtDeepCones3D
from xmipp3.protocols.protocol_deep_alignment3D import XmippProtDeepAlignment3D
from xmipp3.protocols.protocol_deep_cones3D_highresGT import XmippProtDeepCones3DGT
from xmipp3.protocols.protocol_deep_cones3D_highresGT_prueba import XmippProtDeepCones3DGT_2
from .plotter import XmippPlotter
from xmipp3.protocols.protocol_deep_cones3D_tests import XmippProtDeepCones3DTst

# ITER_LAST = 0
# ITER_SELECTION = 1
#
# ANGDIST_2DPLOT = 0
# ANGDIST_CHIMERA = 1
#
# VOLUME_SLICES = 0
# VOLUME_CHIMERA = 1

class XmippDeepAsignmentViewer(ProtocolViewer):
    """ Visualize the output of protocol reconstruct highres """
    _label = 'viewer deep asignment'
    _targets = [XmippProtDeepCones3D, XmippProtDeepAlignment3D,
                XmippProtDeepCones3DGT, XmippProtDeepCones3DTst,
                XmippProtDeepCones3DGT_2]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('showAngDist', LabelParam, default=False,
                       label='Display angular distribution',
                       help='Display angular distribution as interative 2D in matplotlib.\n')

    def _getVisualizeDict(self):
        return {
                'showAngDist': self._showAngularDistribution
                }
    
    # def _validate(self):
    #     if self.lastIter is None:
    #         return ['There are not iterations completed.']
    #
    # def _load(self):
    #     """ Load selected iterations and classes 3D for visualization mode. """
    #     self.firstIter = 1
    #     self.lastIter = self.protocol.getLastFinishedIter()
    #
    #     if self.viewIter.get() == ITER_LAST:
    #         self._iterations = [self.lastIter]
    #     else:
    #         self._iterations = self._getListFromRangeString(self.iterSelection.get())
    #
    #     from matplotlib.ticker import FuncFormatter
    #     self._plotFormatter = FuncFormatter(self._formatFreq)
    # #
    # def _showFSC(self, paramName=None):
    #     xplotter = XmippPlotter(windowTitle="FSC")
    #     a = xplotter.createSubPlot("FSC", "Frequency (1/A)", "FSC")
    #     legends = []
    #     for it in self._iterations:
    #         fnDir = self.protocol._getExtraPath("Iter%03d"%it)
    #         fnFSC = join(fnDir,"fsc.xmd")
    #         if exists(fnFSC):
    #             legends.append('Iter %d' % it)
    #             self._plotFSC(a, fnFSC)
    #             xplotter.showLegend(legends)
    #     a.plot([self.minInv, self.maxInv],[self.resolutionThreshold.get(), self.resolutionThreshold.get()], color='black', linestyle='--')
    #     a.grid(True)
    #     views = []
    #     views.append(xplotter)
    #     return views
    #
    # def _plotFSC(self, a, fnFSC):
    #     md = MetaData(fnFSC)
    #     resolution_inv = [md.getValue(MDL_RESOLUTION_FREQ, f) for f in md]
    #     frc = [md.getValue(MDL_RESOLUTION_FRC, f) for f in md]
    #     self.maxFrc = max(frc)
    #     self.minInv = min(resolution_inv)
    #     self.maxInv = max(resolution_inv)
    #     a.plot(resolution_inv, frc)
    #     a.xaxis.set_major_formatter(self._plotFormatter)
    #     a.set_ylim([-0.1, 1.1])

    # def _formatFreq(self, value, pos):
    #     """ Format function for Matplotlib formatter. """
    #     inv = 999
    #     if value:
    #         inv = 1/value
    #     return "1/%0.2f" % inv
    #
    # def _showVolume(self, paramName=None):
    #     choice = self.displayVolume.get()
    #     views=[]
    #     for it in self._iterations:
    #         fnDir = self.protocol._getExtraPath("Iter%03d"%it)
    #         if choice == 0:
    #             if self.protocol.alignmentMethod==self.protocol.GLOBAL_ALIGNMENT:
    #                 fnDir = join(fnDir,'globalAssignment')
    #             else:
    #                 fnDir = join(fnDir,'localAssignment')
    #             fnVolume = join(fnDir,"volumeRef01.vol")
    #         elif choice == 1:
    #             fnVolume = join(fnDir,"volumeAvg.mrc")
    #         elif choice == 2:
    #             fnVolume = join(fnDir,"volumeAvgFiltered.mrc")
    #         if exists(fnVolume):
    #             samplingRate=self.protocol.readInfoField(fnDir,"sampling",MDL_SAMPLINGRATE)
    #             views.append(ObjectView(self._project, None, fnVolume, viewParams={showj.RENDER: 'image', showj.SAMPLINGRATE: samplingRate}))
    #     return views

    # def _showOutputParticles(self, paramName=None):
    #     views = []
    #     if hasattr(self.protocol, "outputParticles"):
    #         obj = self.protocol.outputParticles
    #         fn = obj.getFileName()
    #         labels = 'id enabled _filename _xmipp_zScore _xmipp_cumulativeSSNR '
    #         labels += '_ctfModel._defocusU _ctfModel._defocusV _xmipp_shiftX _xmipp_shiftY _xmipp_tilt _xmipp_scale _xmipp_maxCC _xmipp_weight'
    #         labels += " _xmipp_cost _xmipp_weightContinuous2 _xmipp_angleDiff0 _xmipp_weightJumper0 _xmipp_angleDiff _xmipp_weightJumper _xmipp_angleDiff2 _xmipp_weightJumper2"
    #         labels += "_xmipp_weightSSNR _xmipp_maxCCPerc _xmipp_costPerc _xmipp_continuousScaleX _xmipp_continuousScaleY _xmipp_continuousX _xmipp_continuousY _xmipp_continuousA _xmipp_continuousB"
    #         views.append(ObjectView(self._project, obj.strId(), fn,
    #                                       viewParams={showj.ORDER: labels,
    #                                                   showj.VISIBLE: labels,
    #                                                   showj.MODE: showj.MODE_MD,
    #                                                   showj.RENDER:'_filename'}))
    #     return views

    # def _showInternalParticles(self, paramName=None):
    #     views = []
    #     for it in self._iterations:
    #         fnDir = self.protocol._getExtraPath("Iter%03d"%it)
    #         fnAngles = join(fnDir,"angles.xmd")
    #         if exists(fnAngles):
    #             views.append(DataView(fnAngles, viewParams={showj.MODE: showj.MODE_MD}))
    #     return views
    #
    # def _plotHistogramAngularMovement(self, paramName=None):
    #     views = []
    #     for it in self._iterations:
    #         fnDir = self.protocol._getExtraPath("Iter%03d"%it)
    #         fnAngles = join(fnDir,"angles.xmd")
    #         if self.protocol.weightJumper and it>1:
    #             import xmippLib
    #             xplotter = XmippPlotter(windowTitle="Jumper weight")
    #             a = xplotter.createSubPlot("Jumper weight", "Weight", "Count")
    #             xplotter.plotMdFile(fnAngles,xmippLib.MDL_WEIGHT_JUMPER,xmippLib.MDL_WEIGHT_JUMPER,nbins=100)
    #             views.append(xplotter)
    #     return views
    
#===============================================================================
# showAngularDistribution
#===============================================================================
    def _showAngularDistribution(self, paramName=None):
        views = []
        angDist = self._createAngDist2D()
        if angDist is not None:
            views.append(angDist)
        return views
    
    def _iterAngles(self, fnAngles):
        md=MetaData(fnAngles)
        for objId in md:
            rot = md.getValue(MDL_ANGLE_ROT, objId)
            tilt = md.getValue(MDL_ANGLE_TILT, objId)
            yield rot, tilt


    def _createAngDist2D(self):
        fnDir = self.protocol._getExtraPath()
        fnAngles = join(fnDir,"outConesParticles.xmd")
        view=None
        if not exists(fnAngles):
            fnAngles = join(fnDir,"outputParticles.xmd")
        if exists(fnAngles):
            fnAnglesSqLite = join(fnDir, "outConesParticles.sqlite")
            if not exists(fnAnglesSqLite):
                from pyworkflow.em.metadata.utils import getSize
                self.createAngDistributionSqlite(fnAnglesSqLite, getSize(fnAngles), itemDataIterator=self._iterAngles(fnAngles))
            from pyworkflow.em.viewers.plotter import EmPlotter
            view = EmPlotter(x=1, y=1, windowTitle="Angular distribution")
            view.plotAngularDistributionFromMd(fnAnglesSqLite, '')
        return view


    