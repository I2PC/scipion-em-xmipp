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
import pyworkflow.utils as pwutils
from pwem.viewers import (ObjectView, showj, EmProtocolViewer, ChimeraAngDist)

from pwem.emlib import (MDL_SAMPLINGRATE, MDL_ANGLE_ROT, MDL_ANGLE_TILT,
                   MDL_RESOLUTION_FREQ, MDL_RESOLUTION_FRC, MetaData)
from xmipp3.protocols.protocol_reconstruct_highres import XmippProtReconstructHighRes
from .plotter import XmippPlotter

ITER_LAST = 0
ITER_SELECTION = 1

ANGDIST_2DPLOT = 0
ANGDIST_CHIMERA = 1
ANGDIST_HEATMAP = 2

VOLUME_SLICES = 0
VOLUME_CHIMERA = 1

class XmippReconstructHighResViewer(EmProtocolViewer):
    """ Visualize the output of protocol reconstruct highres """
    _label = 'viewer reconstruct highres'
    _targets = [XmippProtReconstructHighRes]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('viewIter', EnumParam, choices=['last', 'selection'], default=ITER_LAST,
                      display=EnumParam.DISPLAY_HLIST,
                      label="Iteration to visualize", 
                      help="""
*last*: only the last iteration will be visualized.
*selection*: you may specify a range of iterations.
Examples:
"1,5-8,10" -> [1,5,6,7,8,10]
"2,6,9-11" -> [2,6,9,10,11]
"2 5, 6-8" -> [2,5,6,7,8]                      
                           """)
        form.addParam('iterSelection', NumericRangeParam, 
                      condition='viewIter==%d' % ITER_SELECTION, 
                      label="Iteration list",
                      help="Write the iteration list to visualize.")
  
        group = form.addGroup('Particles')
        group.addParam('showOutputParticles', LabelParam, default=False, label='Display output particles')
        group.addParam('showAngDist', EnumParam, choices=['2D plot', 'chimera', 'heatmap'],
                       display=EnumParam.DISPLAY_HLIST, default=ANGDIST_HEATMAP,
                       label='Display angular distribution',
                       help='*2D plot*: display angular distribution as interative 2D in matplotlib.\n'
                            '*chimera*: display angular distribution using Chimera with red spheres.')
        group.addParam('spheresScale', IntParam, default=-1,
                       expertLevel=LEVEL_ADVANCED,
                       label='Spheres size')
        group.addParam('plotHistogramAngularMovement', LabelParam, default=False,
                      label='Plot histogram with angular changes',
                      help="""Plot histogram with angular changes from one iteration to next. 
                              Available from iteration 2""")

        group = form.addGroup('Volumes')
        group.addParam('displayVolume', EnumParam, choices=['Reference', 'Reconstructed', 'Filtered'],
                       default=1, display=EnumParam.DISPLAY_COMBO,
                       label='Display volume',
                       help='Displays selected volume')
        group.addParam('showResolutionPlots', LabelParam, default=True,
                      label='Display resolution plots (FSC)')
        group.addParam('resolutionThreshold', FloatParam, default=0.143,
                      expertLevel=LEVEL_ADVANCED,
                      label='Threshold in resolution plots')

    def _getVisualizeDict(self):
        self._load()
        return {
                'displayVolume' : self._showVolume,
                'showOutputParticles' : self._showOutputParticles,
                'plotHistogramAngularMovement' : self._plotHistogramAngularMovement,
                'showAngDist': self._showAngularDistribution,
                'showResolutionPlots': self._showFSC
                }
    
    def _validate(self):
        if self.lastIter is None:
            return ['There are not iterations completed.'] 

    def _load(self):
        """ Load selected iterations and classes 3D for visualization mode. """
        self.firstIter = 1
        self.lastIter = self.protocol.getLastFinishedIter()
        
        if self.viewIter.get() == ITER_LAST:
            self._iterations = [self.lastIter]
        else:
            self._iterations = pwutils.getListFromRangeString(self.iterSelection.get())
            
        from matplotlib.ticker import FuncFormatter
        self._plotFormatter = FuncFormatter(self._formatFreq) 
    
    def _showFSC(self, paramName=None):
        xplotter = XmippPlotter(windowTitle="FSC")
        a = xplotter.createSubPlot("FSC", "Frequency (1/A)", "FSC")
        legends = []
        for it in self._iterations:
            fnDir = self.protocol._getExtraPath("Iter%03d"%it)
            fnFSC = join(fnDir,"fsc.xmd")
            if exists(fnFSC):
                legends.append('Iter %d' % it)
                self._plotFSC(a, fnFSC)
                xplotter.showLegend(legends)
        a.plot([self.minInv, self.maxInv],[self.resolutionThreshold.get(), self.resolutionThreshold.get()], color='black', linestyle='--')
        a.grid(True)
        views = []
        views.append(xplotter)
        return views

    def _plotFSC(self, a, fnFSC):
        md = MetaData(fnFSC)
        resolution_inv = [md.getValue(MDL_RESOLUTION_FREQ, f) for f in md]
        frc = [md.getValue(MDL_RESOLUTION_FRC, f) for f in md]
        self.maxFrc = max(frc)
        self.minInv = min(resolution_inv)
        self.maxInv = max(resolution_inv)
        a.plot(resolution_inv, frc)
        a.xaxis.set_major_formatter(self._plotFormatter)
        a.set_ylim([-0.1, 1.1])

    def _formatFreq(self, value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999
        if value:
            inv = 1/value
        return "1/%0.2f" % inv

    def _showVolume(self, paramName=None):
        choice = self.displayVolume.get()
        views=[]
        for it in self._iterations:
            fnDir = self.protocol._getExtraPath("Iter%03d"%it)
            if choice == 0:
                if self.protocol.alignmentMethod==self.protocol.GLOBAL_ALIGNMENT:
                    fnDir = join(fnDir,'globalAssignment')
                else:
                    fnDir = join(fnDir,'localAssignment')
                fnVolume = join(fnDir,"volumeRef01.vol")
            elif choice == 1:
                fnVolume = join(fnDir,"volumeAvg.mrc")
            elif choice == 2:
                fnVolume = join(fnDir,"volumeAvgFiltered.mrc")
            if exists(fnVolume):
                samplingRate=self.protocol.readInfoField(fnDir,"sampling",MDL_SAMPLINGRATE)
                views.append(ObjectView(self._project, None, fnVolume, viewParams={showj.RENDER: 'image', showj.SAMPLINGRATE: samplingRate}))
        return views

    def _showOutputParticles(self, paramName=None):
        views = []
        if hasattr(self.protocol, "outputParticles"):
            obj = self.protocol.outputParticles
            fn = obj.getFileName()
            labels = 'id enabled _filename _xmipp_zScore _xmipp_cumulativeSSNR '
            labels += '_ctfModel._defocusU _ctfModel._defocusV _xmipp_shiftX _xmipp_shiftY _xmipp_tilt _xmipp_scale _xmipp_maxCC _xmipp_weight'
            labels += " _xmipp_cost _xmipp_weightContinuous2 _xmipp_angleDiff0 _xmipp_weightJumper0 _xmipp_angleDiff _xmipp_weightJumper _xmipp_angleDiff2 _xmipp_weightJumper2"
            labels += "_xmipp_weightSSNR _xmipp_maxCCPerc _xmipp_costPerc _xmipp_continuousScaleX _xmipp_continuousScaleY _xmipp_continuousX _xmipp_continuousY _xmipp_continuousA _xmipp_continuousB"
            views.append(ObjectView(self._project, obj.strId(), fn,
                                          viewParams={showj.ORDER: labels, 
                                                      showj.VISIBLE: labels, 
                                                      showj.MODE: showj.MODE_MD,
                                                      showj.RENDER:'_filename'}))
        return views

    def _plotHistogramAngularMovement(self, paramName=None):
        views = []
        for it in self._iterations:
            fnDir = self.protocol._getExtraPath("Iter%03d"%it)
            fnAngles = join(fnDir,"angles.xmd")
            if it>1:
                from pwem import emlib
                xplotter = XmippPlotter(windowTitle="Angular difference")
                a = xplotter.createSubPlot("Angular difference", "Angle", "Count")
                xplotter.plotMdFile(fnAngles,emlib.MDL_ANGLE_DIFF,emlib.MDL_ANGLE_DIFF,nbins=100)
                views.append(xplotter)
        return views
    
#===============================================================================
# showAngularDistribution
#===============================================================================
    def _showAngularDistribution(self, paramName=None):
        views = []
        if self.showAngDist == ANGDIST_CHIMERA:
            for it in self._iterations:
                angDist = self._createAngDistChimera(it)
                if angDist is not None:
                    views.append(angDist)
                        
        elif self.showAngDist == ANGDIST_2DPLOT or self.showAngDist == ANGDIST_HEATMAP:
            for it in self._iterations:
                angDist = self._createAngDist2D(it, heatmap=self.showAngDist == ANGDIST_HEATMAP)
                if angDist is not None:
                    views.append(angDist)
        return views
    
    def _iterAngles(self, fnAngles):
        md=MetaData(fnAngles)
        for objId in md:
            rot = md.getValue(MDL_ANGLE_ROT, objId)
            tilt = md.getValue(MDL_ANGLE_TILT, objId)
            yield rot, tilt

    def _createAngDistChimera(self, it):
        fnDir = self.protocol._getExtraPath("Iter%03d"%it)
        fnAngles = join(fnDir,"angles.xmd")
        view=None
        if exists(fnAngles):
            fnAnglesSqLite = join(fnDir,"angles.sqlite")
            from pwem.emlib.metadata import getSize
            self.createAngDistributionSqlite(fnAnglesSqLite, getSize(fnAngles),
                                             itemDataIterator=self._iterAngles(fnAngles))
            vol = os.path.join(fnDir, "volumeAvg.mrc")
            extraFilesPath = self.protocol._getExtraPath()
            volOrigin = self.protocol.outputVolume.getShiftsFromOrigin()
            radius = self.spheresScale.get()
            view = ChimeraAngDist(vol, extraFilesPath,
                                  angularDistFile=fnAngles,
                                  spheresDistance=radius,
                                  voxelSize=self.protocol.outputVolume.getSamplingRate(),
                                  volOrigin=volOrigin,
                                  showProjection=True)

        return view
    
    def _createAngDist2D(self, it, heatmap):
        fnDir = self.protocol._getExtraPath("Iter%03d"%it)
        fnAngles = join(fnDir,"angles.xmd")
        view=None
        if exists(fnAngles):
            fnAnglesSqLite = join(fnDir,"angles.sqlite")
            from pwem.viewers import EmPlotter
            if not exists(fnAnglesSqLite):
                from pwem.emlib.metadata import getSize
                self.createAngDistributionSqlite(fnAnglesSqLite, getSize(fnAngles), itemDataIterator=self._iterAngles(fnAngles))
            view = EmPlotter(x=1, y=1, mainTitle="Iteration %d" % it, windowTitle="Angular distribution")
            if heatmap:
                axis = view.plotAngularDistributionFromMd(fnAnglesSqLite, '', histogram=True)
                view.getFigure().colorbar(axis)
            else:
                view.plotAngularDistributionFromMd(fnAnglesSqLite, '')
        return view
    