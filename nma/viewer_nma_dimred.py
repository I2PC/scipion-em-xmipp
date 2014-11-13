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
This module implement the wrappers aroung Xmipp CL2D protocol
visualization program.
"""

import os
from os.path import join, dirname, basename
import numpy as np
import Tkinter as tk
import ttk

import pyworkflow.gui as gui
from pyworkflow.gui.widgets import IconButton, HotButton
from pyworkflow.utils.properties import Icon, Color 
from pyworkflow.viewer import (ProtocolViewer, CommandView,
                               DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, BooleanParam
from protocol_nma_dimred import XmippProtDimredNMA
import xmipp

from plotter import XmippNmaPlotter


        
class XmippDimredNMAViewer(ProtocolViewer):
    """ Visualization of results from the NMA protocol
    """
    _label = 'viewer nma dimred'
    _targets = [XmippProtDimredNMA]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
        
    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayRawDeformation', StringParam, default='1',
                      label='Display raw deformation',
                      help='Type 1 to see the histogram of raw deformation number 1; \n'
                           'type 2 to see the histogram of raw deformation number 2, etc.\n'
                           'Type 1 2 to see the 2D plot of raw deformations number 1 vs 2.\n'
                           'Type 1 2 3 to see the 3D plot of raw deformations 1, 2 and 3; etc.'
                           )
        
        form.addParam('displayClustering', BooleanParam, default=False,
                      label='Open clustering tool?',
                      help='Open a GUI to visualize the images as points'
                           'and select some of them to create new clusters.')
        
        
    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation,
                'displayClustering': self._displayClustering,
                } 
                        
    def _viewRawDeformation(self, paramName):
        components = self.displayRawDeformation.get()
        return self._doViewRawDeformation(components)
        
    def _doViewRawDeformation(self, components):
#        components = map(int, self.displayRawDeformation.get().split())
        components = map(int, components.split())
        dim = len(components)
        views = []
        
        if dim > 0:
            modeList = [m - 1 for m in components]
            modeNameList = ['Mode %d' % m for m in components]
            missingList = []
                    
            if missingList:
                return [self.errorMessage("Invalid mode(s) *%s*\n." % (', '.join(missingList)), 
                              title="Invalid input")]
            
            # Actually plot
            plotter = XmippNmaPlotter(data=self.loadData()) 
            baseList = [basename(n) for n in modeNameList]
            
            if dim == 1:
                Point.XIND = modeList[0]
                plotter.plotArray1D("Histogram for %s" % baseList[0], 
                                    "Deformation value", "Number of images")
            else:
                Point.YIND = modeList[1]
                if dim == 2:
                    plotter.plotArray2D("%s vs %s" % tuple(baseList), *baseList)
                elif dim == 3:
                    Point.ZIND = modeList[2]
                    plotter.plotArray3D("%s %s %s" % tuple(baseList), *baseList)
            views.append(plotter)
            
        return views
    
    def _displayClustering(self, paramName):
        return [self.tkWindow(ClusteringWindow, 
                              dim=self.protocol.reducedDim.get(),
                              data=self.loadData()
                              )]
        
    def loadData(self):
        """ Iterate over the images and the output matrix txt file
        and create a Data object with theirs Points.
        """
        matrix = np.loadtxt(self.protocol.getOutputMatrixFile())
        particles = self.protocol.inputNMA.get().outputParticles
        
        data = Data()
        for i, particle in enumerate(particles):
            data.addPoint(Point(pointId=particle.getObjId(),
                                data=matrix[i, :],
                                weight=particle._xmipp_cost.get()))
            
        return data
    
    
class ClusteringWindow(gui.Window):
    def __init__(self, **kwargs):
        gui.Window.__init__(self, **kwargs)
        
        self.dim = kwargs.get('dim')
        self.data = kwargs.get('data')
        self.plotter = None
         
        content = tk.Frame(self.root)
        self._createContent(content)
        content.grid(row=0, column=0, sticky='news')
        content.columnconfigure(0, weight=1)
        #content.rowconfigure(1, weight=1)
        
    def _createContent(self, content):
        self._createFigureBox(content)
        self._createClusteringBox(content)
        
    def _addLabel(self, parent, text, r, c):
        label = tk.Label(parent, text=text, font=self.fontBold)
        label.grid(row=r, column=c, padx=5, pady=5, sticky='ne')
        return label
    
    def _createFigureBox(self, content):
        frame = tk.LabelFrame(content, text='Figure')
        frame.columnconfigure(2, weight=1)#, minsize=30)
        frame.columnconfigure(0, minsize=50)
        # Create the 'Axes' label
        self._addLabel(frame, 'Axes', 0, 0)
        
        # Create a listbox with x1, x2 ...
        listbox = tk.Listbox(frame, height=5, 
                             selectmode=tk.MULTIPLE, bg='white')        
        for x in range(1, self.dim+1):
            listbox.insert(tk.END, 'x%d' % x)        
        listbox.grid(row=0, column=1, padx=5, pady=5)
        self.listbox = listbox
        
        updateBtn = HotButton(frame, text='Update', command=self._onUpdateClick)
        updateBtn.grid(row=5, column=5, sticky='ne', padx=5, pady=5)
       
        frame.grid(row=0, column=0, sticky='new', padx=5, pady=(10, 5))

    def _createClusteringBox(self, content):
        frame = tk.LabelFrame(content, text='Clustering')
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(0, minsize=50)
        
        # Cluster line
        self._addLabel(frame, 'Cluster', 0, 0)
        self.clusterLabel = tk.Label(frame, text='0 / 100 points')
        self.clusterLabel.grid(row=0, column=1, sticky='nw', padx=5, pady=5)
        
        # Selection controls
        self._addLabel(frame, 'Selection', 1, 0)  
        selectionFrame = tk.Frame(frame)
        selectionFrame.grid(row=1, column=1, sticky='news')
        # --- Expression
        expressionRb = tk.Radiobutton(selectionFrame, text='Expression', value=1)
        expressionRb.grid(row=0, column=0, sticky='nw')
        expressionEntry = tk.Entry(selectionFrame)
        expressionEntry.grid(row=0, column=1)
        # --- Free hand
        freehandRb = tk.Radiobutton(selectionFrame, text='Free hand', value=2)
        freehandRb.grid(row=1, column=0, sticky='nw')
        
        
        # Create buttons frame        
        buttonsFrame = tk.Frame(frame, bg='green')
        buttonsFrame.grid(row=2, column=0, columnspan=5,
                          sticky='se', padx=5, pady=5)
        buttonsFrame.columnconfigure(0, weight=1)
        
        resetBtn = tk.Button(buttonsFrame, text='Reset')
        resetBtn.grid(row=0, column=0)
        createBtn = HotButton(buttonsFrame, text='Create Cluster')
        createBtn.grid(row=0, column=1)       
        
       
        frame.grid(row=1, column=0, sticky='new', padx=5, pady=(5, 10))
        
    def _onUpdateClick(self, e=None):
        components = self.listbox.curselection()
        dim = len(components)
            
        if dim > 0:
            modeList = components
            modeNameList = ['x%d' % (m+1) for m in components]
            missingList = []
                    
            if missingList:
                return [self.errorMessage("Invalid mode(s) *%s*\n." % (', '.join(missingList)), 
                              title="Invalid input")]
            
            if self.plotter is None or self.plotter.closed():
                self.plotter = XmippNmaPlotter(data=self.data)
                #self.plotter.useLastPlot = True
            else:
                self.plotter.clear()
            
            # Actually plot
            baseList = [basename(n) for n in modeNameList]
            
            Point.XIND = modeList[0]
            
            if dim == 1:
                self.plotter.plotArray1D("Histogram for %s" % baseList[0],  
                                    "Deformation value", "Number of images")
            else:
                Point.YIND = modeList[1]
                if dim == 2:
                    self.plotter.plotArray2D("%s vs %s" % tuple(baseList), *baseList)
                elif dim == 3:
                    Point.ZIND = modeList[2]
                    self.plotter.plotArray3D("%s %s %s" % tuple(baseList), *baseList)

            self.plotter.draw()


class Point():
    """ Return x, y 2d coordinates and some other properties
    such as weight and state.
    """
    # Indexes of data
    XIND = 0
    YIND = 1
    ZIND = 2
    # Selection states
    DISCARDED = -1
    NORMAL = 0
    SELECTED = 1
    
    def __init__(self, pointId, data, weight, state=0):
        self._id = pointId
        self._data = data
        self._weight = weight
        self._state = state
        
    def getId(self):
        return self._id
    
    def getX(self):
        return self._data[Point.XIND]
    
    def getY(self):
        return self._data[Point.YIND]
    
    def getZ(self):
        return self._data[Point.ZIND]
    
    def getWeight(self):
        return self._weight
    
    def getState(self):
        return self._state
    
    
class Data():
    """ Store data points. """
    def __init__(self):
        self._points = []
        
    def addPoint(self, point):
        self._points.append(point)
        
    def __iter__(self):
        for point in self._points:
            yield point
            
    def getXData(self):
        return [p.getX() for p in self]
    
    def getYData(self):
        return [p.getY() for p in self]
    
    def getZData(self):
        return [p.getZ() for p in self]
    
    def getWeights(self):
        return [p.getWeight() for p in self]
    
    