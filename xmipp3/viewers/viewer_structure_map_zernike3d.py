# **************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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
import numpy as np
from mpl_toolkits.mplot3d import proj3d

from pyworkflow.viewer import DESKTOP_TKINTER, WEB_DJANGO, ProtocolViewer
from pyworkflow.gui.plotter import plt
import pyworkflow.protocol.params as params

from xmipp3.protocols.protocol_structure_map_zernike3d import XmippProtStructureMapZernike3D


class XmippProtStructureMapZernike3DViewer(ProtocolViewer):
    """ Wrapper to visualize different type of data objects
    with the Xmipp program xmipp_showj
    """
    
    _label = 'viewer structure map sph'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtStructureMapZernike3D]


    def _defineParams(self, form):
        form.addSection(label='Show StructMap')
        form.addParam('numberOfDimensions', params.IntParam, default=2,
                      label="Number of dimensions",
                      help='In normal practice, it should be 1, 2 or, at most, 3.')
        form.addParam('map', params.EnumParam, choices=['Deformation', 'Correlation', 'Consensus'],
                      default=0, help='Choose the type of metric to display the coordinates.')
        form.addParam('twoSets', params.BooleanParam, default=False, label='Two set analysis',
                      help='Activate if the analysis of the deformation included the option of analysing '
                           'two sets of volumes independently.')
        form.addParam('doShowPlot', params.LabelParam,
                      label="Display the StructMap")
    
    def _getVisualizeDict(self):
        return {'doShowPlot': self._visualize}
    
        
    def _visualize(self, e=None):
        nDim = self.numberOfDimensions.get()
        if self.map.get()==0 and not self.twoSets:
            fnOutput = [self.protocol._defineResultsName(nDim)]
        elif self.map.get() == 1 and not self.twoSets:
            fnOutput = [self.protocol._defineResultsName2(nDim)]
        elif self.map.get()==0 and self.twoSets:
            fnOutput = [self.protocol._defineResultsName(nDim, 'Sub_1_'),
                        self.protocol._defineResultsName(nDim, 'Sub_2_')]
        elif self.map.get()==1 and self.twoSets:
            fnOutput = [self.protocol._defineResultsName2(nDim, 'Sub_1_'),
                        self.protocol._defineResultsName2(nDim, 'Sub_2_')]
        elif self.map.get() == 2 and nDim != 1:
            fnOutput = [self.protocol._defineResultsName3(nDim)]
        elif self.map.get() == 2 and nDim == 1:
            return [self.errorMessage('Consensus in only avalaible for two and three dimensions\n',
                                      title='Functionality not supported')]

        for file in fnOutput:
            if not os.path.exists(file):
                return [self.errorMessage('The necessary metadata was not produced\n'
                                          'Execute again the protocol\n',
                                          title='Missing result file')]

        coordinates = (np.loadtxt(file) for file in fnOutput)
        coordinates = np.vstack(coordinates)
        
        # Create labels
        count = 0
        labels = []
        _, _, _, idList = self.protocol._iterInputVolumes(self.protocol.inputVolumes, [], [], [], [])
        if self.protocol.secondSet:
            _, _, _, idList = self.protocol._iterInputVolumes(self.protocol.secondSet, [], [], [], idList)
        for idv in idList:
            # base=os.path.basename(voli)
            # fileName = os.path.splitext(base)[0]
            # count += 1
            labels.append("vol_%02d" % idv)
            #labels.append(fileName)
         
        val = 0
        if nDim == 1:
            AX = plt.subplot(111)
            plot = plt.plot(coordinates, np.zeros_like(coordinates) + val, 'o', c='g')
            plt.xlabel('Dimension 1', fontsize=11)
            AX.set_yticks([1])
            plt.title('StructMap')
             
            for label, x, y in zip(labels, coordinates, np.zeros_like(coordinates)):
                plt.annotate(label,
                             xy = (x, y), xytext = (x+val, y+val),
                             textcoords = 'data', ha = 'right', va = 'bottom',fontsize=9,
                             bbox = dict(boxstyle = 'round,pad=0.3', fc = 'yellow', alpha = 0.3))
            plt.grid(True)
            plt.show()
                         
        elif nDim == 2:
            plot = plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='o', c='g')
            plt.xlabel('Dimension 1', fontsize=11)
            plt.ylabel('Dimension 2', fontsize=11)
            plt.title('StructMap')
            for label, x, y in zip(labels, coordinates[:, 0], coordinates[:, 1]):
                plt.annotate(label,
                             xy = (x, y), xytext = (x+val, y+val),
                             textcoords = 'data', ha = 'right', va = 'bottom',fontsize=9,
                             bbox = dict(boxstyle = 'round,pad=0.3', fc = 'yellow', alpha = 0.3))
            plt.grid(True)
            plt.show()
                
        else: 
                         
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            
            ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], marker = 'o', c='g')
            ax.set_xlabel('Dimension 1', fontsize=11)
            ax.set_ylabel('Dimension 2', fontsize=11)
            ax.set_zlabel('Dimension 3', fontsize=11)
            ax.text2D(0.05, 0.95, "StructMap", transform=ax.transAxes)
                         
            x2, y2, _ = proj3d.proj_transform(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], ax.get_proj())
            Labels = []
            for i in range(len(coordinates[:, 0])):
                text = labels[i]
                label = ax.annotate(text,
                                    xycoords='data',
                                    xy = (x2[i], y2[i]), xytext = (x2[i]+val, y2[i]+val),
                                    textcoords = 'data', ha = 'right',
                                     va = 'bottom', fontsize=9,
                                    bbox = dict(boxstyle = 'round,pad=0.3',
                                                 fc = 'yellow', alpha = 0.3))
                                     
                Labels.append(label)
                
            def update_position(e):
                x2, y2, _ = proj3d.proj_transform(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], ax.get_proj())
                for i in range(len(coordinates[:, 0])):
                    label = Labels[i]
                    label.xytext = (x2[i],y2[i])
                    label.update_positions(fig.canvas.get_renderer())
                fig.canvas.draw()
            fig.canvas.mpl_connect('button_release_event', update_position)
            plt.show()
        
        return plot
        
    def _validate(self):
        errors = []
        
        numberOfDimensions=self.numberOfDimensions.get()
        if numberOfDimensions > 3 or numberOfDimensions < 1:
            errors.append("The number of dimensions should be 1, 2 or, at most, 3.")
        
        return errors