# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
"""
This module implement the wrappers around Xmipp NMA protocol
visualization program.
"""

from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from plotter import XmippPlotter
from pyworkflow.em import *
from protocol_nma import XmippProtNMA
import glob


CLASSES = 0
CLASS_CORES = 1
CLASS_STABLE_CORES = 2

        
class XmippNMAViewer(ProtocolViewer):
    """ Visualization of results from the NMA protocol.    
        Normally, NMA modes with high collectivity and low NMA score are preferred.
    """
    _label = 'viewer nma'
    _targets = [XmippProtNMA]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def setProtocol(self, protocol):
        ProtocolViewer.setProtocol(self, protocol)
        isEm = not isinstance(protocol.inputStructure.get(), PdbFile)
        self.isEm.set(isEm)
        
    def _defineParams(self, form):
        form.addSection(label='Visualization')
        #TODO: Just trying a trick to have hidden params
        form.addParam('isEm', BooleanParam, default=False, 
                      condition='False')
        form.addParam('displayPseudoAtom', BooleanParam, default=False, 
                      condition='isEm',
                      label="Display pseudoatoms representation?")
        form.addParam('displayPseudoAtomAproximation', BooleanParam, default=False,
                      condition='isEm',
                      label="Display pseudoatoms aproximation?")     
        form.addParam('displayModes', BooleanParam, default=False, 
                      label="Open list of modes?")
        form.addParam('displayMaxDistanceProfile', BooleanParam, default=False, 
                      label="Plot max distance profile?")     
        form.addParam('displayDistanceProfile', BooleanParam, default=False, 
                      label="Plot distance profile?")
        form.addParam('singleMode', IntParam, default=7,
              label='Open specific mode', condition='True')   
        
    def _getVisualizeDict(self):
        return {'displayPseudoAtom': self._viewParam,
                'displayPseudoAtomAproximation': self._viewParam,
                'displayModes': self._viewParam,
                'displayMaxDistanceProfile': self._viewParam,
                'displayDistanceProfile': self._viewParam,
                'singleMode': self._viewParam,
                } 
                        
    def _viewParam(self, paramName):
        views = []
        if paramName == 'displayPseudoAtom':
            views.append(CommandView('chimera ' + self.protocol._getPath("chimera.cmd")))
        elif paramName == 'displayPseudoAtomAproximation':
            views.append(ProjectDataView(self.protocol.inputStructure.get().getFirstItem().getFileName()))
            views.append(ProjectDataView(self.protocol._getExtraPath('pseudoatoms_approximation.vol')))
        elif paramName == 'displayModes':
            views.append(self.protocol._getPath('modes.xmd'))
        elif paramName == 'displayMaxDistanceProfile':
            fn = self.protocol._getExtraPath("maxAtomShifts.xmd")
            views.append(self._createShiftPlot(fn, "Maximum atom shifts", "maximum shift"))
        elif paramName == 'displayDistanceProfile':
            mode = self.singleMode.get()
            fn = self.protocol._getExtraPath("distanceProfiles","vec%d.xmd" % mode)
            views.append(self._createShiftPlot(fn, "Atom shifts for mode %d" % mode, "shift"))
        elif paramName == 'singleMode':
            if self.singleMode.hasValue():
                vmdFile = self.protocol._getExtraPath("animations", "animated_mode_%03d.vmd" % self.singleMode.get())
                views.append(CommandView("vmd -e %s" % vmdFile))
    
    def _createShiftPlot(self, mdFn, title, ylabel):
        plotter = XmippPlotter()
        plotter.createSubPlot(title, 'atom index', ylabel)
        plotter.plotMdFile(mdFn, None, xmipp.MDL_NMA_ATOMSHIFT)
        return plotter
    
