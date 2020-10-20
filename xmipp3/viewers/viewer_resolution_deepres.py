# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Erney Ramirez Aportela (eramirez@cnb.csic.es)
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
from pwem.objects import Volume
from pwem.wizards import ColorScaleWizardBase

from pwem.viewers import (LocalResolutionViewer, EmPlotter, ChimeraView,
                          DataView)
from pwem.constants import  AX_Z
from pyworkflow.protocol.params import (LabelParam, EnumParam,
                                        IntParam, LEVEL_ADVANCED)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pwem.emlib.metadata import MetaData, MDL_X, MDL_COUNT

from .plotter import XmippPlotter
from xmipp3.protocols.protocol_resolution_deepres import (XmippProtDeepRes,
                                                          OUTPUT_RESOLUTION_FILE,
                                                          FN_METADATA_HISTOGRAM,
                                                          OUTPUT_RESOLUTION_FILE_CHIMERA, RESIZE_VOL)


class XmippResDeepResViewer(LocalResolutionViewer):
    """
    Visualization tools for DeepRes results.
    
    DeepRes is a Xmipp package for computing the local resolution of 3D
    density maps studied in structural biology, primarily by cryo-electron
    microscopy (cryo-EM).
    """
    _label = 'viewer DeepRes'
    _targets = [XmippProtDeepRes]      
    _environments = [DESKTOP_TKINTER]

    def __init__(self, *args, **kwargs):
        ProtocolViewer.__init__(self, *args, **kwargs)


    def _defineParams(self, form):
        form.addSection(label='Visualization')
        
        form.addParam('doShowVolumeSlices', LabelParam,
                      label="Show resolution slices")
        
        form.addParam('doShowOriginalVolumeSlices', LabelParam,
                      label="Show original volume slices")

        form.addParam('doShowResHistogram', LabelParam,
                      label="Show resolution histogram")
        
        group = form.addGroup('Colored resolution Slices and Volumes')

        group.addParam('sliceAxis', EnumParam, default=AX_Z,
                       choices=['x', 'y', 'z'],
                       display=EnumParam.DISPLAY_HLIST,
                       label='Slice axis')

        group.addParam('doShowVolumeColorSlices', LabelParam,
              label="Show colored slices")
        
        group.addParam('doShowOneColorslice', LabelParam, 
                       expertLevel=LEVEL_ADVANCED, 
                      label='Show selected slice')
        group.addParam('sliceNumber', IntParam, default=-1,
                       expertLevel=LEVEL_ADVANCED, 
                       label='Show slice number')
        
        group.addParam('doShowChimera', LabelParam,
                      label="Show Resolution map in Chimera")

        ColorScaleWizardBase.defineColorScaleParams(group,
                                                    defaultHighest=self.protocol.max_res_init.get(),
                                                    defaultLowest=self.protocol.min_res_init.get())

    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        return {'doShowOriginalVolumeSlices': self._showOriginalVolumeSlices,
                'doShowVolumeSlices': self._showVolumeSlices,
                'doShowVolumeColorSlices': self._showVolumeColorSlices,
                'doShowOneColorslice': self._showOneColorslice,
                'doShowResHistogram': self._plotHistogram,
                'doShowChimera': self._showChimera,
                }
       
    def _showVolumeSlices(self, param=None):
        cm = DataView(self.protocol.resolution_Volume.getFileName())
        
        return [cm]
    
    def _showOriginalVolumeSlices(self, param=None):
        cm = DataView(self.protocol.inputVolume.get().getFileName())
        return [cm]
    
    def _showVolumeColorSlices(self, param=None):
        imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE)
        imgData, _, _, _ = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=2, y=2, mainTitle="Local Resolution Slices "
                                                     "along %s-axis."
                                                     %self._getAxis())
        #The slices to be shown are close to the center. Volume size is divided in 
        # 9 segments, the fouth central ones are selected i.e. 3,4,5,6
        for i in range(3, 7):
            sliceNumber = self.getSlice(i, imgData)
            a = xplotter.createSubPlot("Slice %s" % (sliceNumber+1), '', '')
            matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
            plot = xplotter.plotMatrix(a, matrix, self.lowest.get(), self.highest.get(),
                                       cmap=self._getColorName(),
                                       interpolation="nearest")
        xplotter.getColorBar(plot)
        return [xplotter]

    def _showOneColorslice(self, param=None):
        imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE)
        imgData, _, _, volDim = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=1, y=1, mainTitle="Local Resolution Slices "
                                                     "along %s-axis."
                                                     %self._getAxis())
        sliceNumber = self.sliceNumber.get()
        if sliceNumber < 0:
            sliceNumber = volDim[0]/2
        else:
            sliceNumber -= 1

        #sliceNumber has no sense to start in zero
        a = xplotter.createSubPlot("Slice %s" % (sliceNumber+1), '', '')
        matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
        plot = xplotter.plotMatrix(a, matrix, self.lowest.get(), self.highest.get(),
                                       cmap=self._getColorName(),
                                       interpolation="nearest")
        xplotter.getColorBar(plot)
        return [xplotter]
    
    def _plotHistogram(self, param=None):
        md = MetaData()
        md.read(self.protocol._getFileName(FN_METADATA_HISTOGRAM))
        x_axis = []
        y_axis = []

        for idx in md:
            x_axis_ = md.getValue(MDL_X, idx)
            y_axis_ = md.getValue(MDL_COUNT, idx)

            x_axis.append(x_axis_)
            y_axis.append(y_axis_)

        plotter = EmPlotter()
        plotter.createSubPlot("Resolutions Histogram",
                              "Resolution (A)", "# of Counts")
        barwidth = (x_axis[-1] - x_axis[0])/len(x_axis)

        plotter.plotDataBar(x_axis, y_axis, barwidth)

        return [plotter]

    def _getStepColors(self, minRes, maxRes, numberOfColors=13):
        inter = (maxRes - minRes) / (numberOfColors - 1)
        rangeList = []
        for step in range(0, numberOfColors):
            rangeList.append(round(minRes + step * inter, 2))
        return rangeList

    def _getAxis(self):
        return self.getEnumText('sliceAxis')

    def _showChimera(self, param=None):
        fnResVol = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA)

        fnOrigMap = self.protocol._getFileName(RESIZE_VOL)
        sampRate = 1.0 # Res volume and original volume are at different scales

        cmdFile = self.protocol._getExtraPath('chimera_resolution_map.py')
        self.createChimeraScript(cmdFile, fnResVol, fnOrigMap, sampRate,
                                 numColors=self.intervals.get(),
                                 lowResLimit=self.highest.get(),
                                 highResLimit=self.lowest.get())
        view = ChimeraView(cmdFile)
        return [view]