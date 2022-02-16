# **************************************************************************
# *
# * Authors:     L. del Cano (ldelcano@cnb.csic.es)
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
"""
This module implement the wrappers aroung Xmipp ML2D protocol
visualization program.
"""

import os

from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import LabelParam
from pyworkflow.protocol.params import EnumParam, StringParam
from pwem.viewers import ClassesView
from xmipp3.protocols.protocol_ml2d import XmippProtML2D


ITER_LAST = 0
ITER_SEL = 1
ITER_CHOICES = ['last', 'selection']


class XmippML2DViewer(ProtocolViewer):
    """ Wrapper to visualize different type of data objects
    with the Xmipp program xmipp_showj
    """
    _targets = [XmippProtML2D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    
    _label = 'viewer ml2d'
    _plotVars = ['doShowLL', 'doShowPmax', 'doShowSignalChange', 'doShowMirror']
    
    def _defineParams(self, form):
        form.addSection(label='Visualization')

        group = form.addGroup('Overall results')
        group.addParam('classesToShow', EnumParam, choices=ITER_CHOICES,
                      default=ITER_LAST, display=EnumParam.DISPLAY_HLIST,
                      label="Visualize 2D classes from iter",
                      help="Select from which iteration do you want to visualize classes")
        group.addParam('iterSelection', StringParam, condition='classesToShow==%d' % ITER_SEL,
                      label="Iter selection",
                      help="Select several iterations such as: 1,3,4 or 3-5 ")
        group.addParam('doShowPlots', LabelParam,
                       label="Show all plots per iteration?",
                      help='Visualize several plots.')
        
        group = form.addGroup('Iteration plots')
        group.addParam('doShowLL', LabelParam,
                       label="Show Log-Likehood over iterations?",
                       help='The Log-Likelihood value should increase.')
        group.addParam('doShowPmax', LabelParam,
                       label="Show maximum model probability?",
                       help='Show the maximum probability for a model, \n'
                            'this should tend to be a deltha function.')
        group.addParam('doShowSignalChange', LabelParam,
                       label="Show plot for signal change?",
                       help='Should approach to zero when convergence.')
        group.addParam('doShowMirror', LabelParam,
                       label="Show mirror fraction for last iteration?",
                       help='the mirror fraction of each class in last iteration.')
        
    
    def _getVisualizeDict(self):
        return {'classesToShow': self._viewIterRefs,
                'doShowPlots': self._viewAllPlots,
                'doShowLL': self._viewPlot,
                'doShowPmax': self._viewPlot,
                'doShowSignalChange': self._viewPlot,
                'doShowMirror': self._viewPlot}
        
    def _viewAllPlots(self, e=None):
        return createPlots(self.protocol, self._plotVars)
        
    def _viewPlot(self, paramName=None):
        return createPlots(self.protocol, [paramName])
            
    def _viewIterRefs(self, e=None):
        self.protocol._defineFileNames() # Load filename templates
        viewFinalClasses = False
        
        if self.classesToShow == ITER_LAST:
            if os.path.exists(self.protocol._getFileName("final_classes")):
                viewFinalClasses = True
            iterations = [self.protocol._lastIteration()]
        else:
            iterations = self._getListFromRangeString(self.iterSelection.get())
        
        views = []
        
        for it in iterations:
            if viewFinalClasses:
                fn = self.protocol._getFileName("final_classes")
            else:
                if it <= self.protocol._lastIteration():
                    fn = self.protocol._getIterClasses(it)
            
            views.append(ClassesView(self.getProject(),
                                    self.protocol.strId(), fn,
                                    self.protocol.inputParticles.get().strId()))
        return views


def createPlots(protML, selectedPlots):
    ''' Launch some plot for an ML2D protocol run '''
    from xmipp3.viewers.plotter import XmippPlotter
    from pwem import emlib
    
    protML._plot_count = 0
    lastIter = protML._lastIteration()
    if lastIter == 0:
        return
    refs = protML._getIterMdClasses(it=lastIter, block='classes')
#    if not exists(refs):
#        return 
#    blocks = getBlocksInMetaDataFile(refs)
#    lastBlock = blocks[-1]
    
    def doPlot(plotName):
        return plotName in selectedPlots

    # Remove 'mirror' from list if DoMirror is false
    if doPlot('doShowMirror') and not protML.doMirror:
        selectedPlots.remove('doShowMirror')
        
    n = len(selectedPlots)
    if n == 0:
        #showWarning("ML2D plots", "Nothing to plot", protML.master)
        print("No plots")
        return 
    elif n == 1:
        gridsize = [1, 1]
    elif n == 2:
        gridsize = [2, 1]
    else:
        gridsize = [2, 2]
        
    xplotter = XmippPlotter(x=gridsize[0], y=gridsize[1])
        
    # Create data to plot
    iters = range(0, lastIter+1, 1)
    ll = []
    pmax = []
    for iter in iters:
        logs = protML._getIterMdImages(it=iter, block='info')
        md = emlib.MetaData(logs)
        id = md.firstObject()
        ll.append(md.getValue(emlib.MDL_LL, id))
        pmax.append(md.getValue(emlib.MDL_PMAX, id))
            
    if doPlot('doShowLL'):
        a = xplotter.createSubPlot('Log-likelihood (should increase)', 'iterations', 'LL', yformat=True)
        a.plot(iters, ll)

    #Create plot of mirror for last iteration
    if doPlot('doShowMirror'):
        from numpy import arange
        from matplotlib.ticker import FormatStrFormatter
        md = emlib.MetaData(refs)
        mirrors = [md.getValue(emlib.MDL_MIRRORFRAC, id) for id in md]
        nrefs = len(mirrors)
        ind = arange(1, nrefs + 1)
        width = 0.85
        a = xplotter.createSubPlot('Mirror fractions on last iteration', 'classes', 'mirror fraction')
        a.set_xticks(ind + 0.45)
        a.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
        a.bar(ind, mirrors, width, color='b')
        a.set_ylim([0, 1.])
        #a.set_xlim([0.3, nrefs + 1])
        
    if doPlot('doShowPmax'):
        a = xplotter.createSubPlot('Probabilities distribution', 'iterations', 'Pmax/Psum') 
        a.plot(iters, pmax, color='green')
    
    if doPlot('doShowSignalChange'):
        md = emlib.MetaData()
        for iter in iters:
            fn = protML._getIterMdClasses(it=iter, block='classes')
            md2 = emlib.MetaData(fn)
            md2.fillConstant(emlib.MDL_ITER, str(iter))
            md.unionAll(md2)
        # 'iter(.*[1-9].*)@2D/ML2D/run_004/ml2d_iter_refs.xmd')
        # a = plt.subplot(gs[1, 1])
        # print("md: %s" % md)
        md2 = emlib.MetaData()
        md2.aggregate(md, emlib.AGGR_MAX, emlib.MDL_ITER, emlib.MDL_SIGNALCHANGE, emlib.MDL_MAX)
        signal_change = [md2.getValue(emlib.MDL_MAX, id) for id in md2]
        xplotter.createSubPlot('Maximum signal change', 'iterations', 'signal change')
        xplotter.plot(iters, signal_change, color='green')
    
    return [xplotter]