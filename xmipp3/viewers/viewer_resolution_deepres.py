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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from pyworkflow.utils import removeExt

from pwem.viewers import (LocalResolutionViewer, EmPlotter, ChimeraView,
                          DataView)
from pwem.constants import (COLOR_JET, COLOR_OTHER, COLOR_CHOICES, AX_Z)
from pyworkflow.protocol.params import (LabelParam, StringParam, EnumParam,
                                        IntParam, LEVEL_ADVANCED)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pwem.emlib.metadata import MetaData, MDL_X, MDL_COUNT
from pwem.emlib.image import ImageHandler

from .plotter import XmippPlotter
from xmipp3.protocols.protocol_resolution_deepres import (XmippProtDeepRes,
                                                          OUTPUT_RESOLUTION_FILE,
                                                          FN_METADATA_HISTOGRAM,
                                                          OUTPUT_RESOLUTION_FILE_CHIMERA,
                                                          RESIZE_VOL)


binaryCondition = ('(colorMap == %d) ' % (COLOR_OTHER))


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

    
    @staticmethod
    def getColorMapChoices():
        return plt.colormaps()
   
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
        group.addParam('colorMap', EnumParam, choices=COLOR_CHOICES,
                      default=COLOR_JET,
                      label='Color map',
                      help='Select the color map to apply to the resolution map. '
                            'http://matplotlib.org/1.3.0/examples/color/colormaps_reference.html.')
        
        group.addParam('otherColorMap', StringParam, default='jet',
                      condition = binaryCondition,
                      label='Customized Color map',
                      help='Name of a color map to apply to the resolution map.'
                      ' Valid names can be found at '
                      'http://matplotlib.org/1.3.0/examples/color/colormaps_reference.html')
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
        imgData, min_Res, max_Res = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=2, y=2, mainTitle="Local Resolution Slices "
                                                     "along %s-axis."
                                                     %self._getAxis())
        #The slices to be shown are close to the center. Volume size is divided in 
        # 9 segments, the fouth central ones are selected i.e. 3,4,5,6
        for i in range(3, 7):
            sliceNumber = self.getSlice(i, imgData)
            a = xplotter.createSubPlot("Slice %s" % (sliceNumber+1), '', '')
            matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
            plot = xplotter.plotMatrix(a, matrix, min_Res, max_Res,
                                       cmap=self.getColorMap(),
                                       interpolation="nearest")
        xplotter.getColorBar(plot)
        return [xplotter]

    def _showOneColorslice(self, param=None):
        imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE)
        imgData, min_Res, max_Res = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=1, y=1, mainTitle="Local Resolution Slices "
                                                     "along %s-axis."
                                                     %self._getAxis())
        sliceNumber = self.sliceNumber.get()
        if sliceNumber < 0:
            x ,_ ,_ ,_ = ImageHandler().getDimensions(imageFile)
            sliceNumber = x/2
        else:
            sliceNumber -= 1
        #sliceNumber has no sense to start in zero 
        a = xplotter.createSubPlot("Slice %s" % (sliceNumber+1), '', '')
        matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
        plot = xplotter.plotMatrix(a, matrix, min_Res, max_Res,
                                       cmap=self.getColorMap(),
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

    def _getAxis(self):
        return self.getEnumText('sliceAxis')

    def _showChimera(self, param=None):
        self.createChimeraScript()
        cmdFile = self.protocol._getPath('Chimera_resolution.cmd')
        view = ChimeraView(cmdFile)
        return [view]

    def numberOfColors(self, min_Res, max_Res, numberOfColors):
        inter = (max_Res - min_Res)/(numberOfColors-1)
        colors_labels = ()
        for step in range(0,numberOfColors):
            colors_labels += round(min_Res + step*inter,2),
        return colors_labels

    def createChimeraScript(self):
        fnRoot = "extra/"
        scriptFile = self.protocol._getPath('Chimera_resolution.cmd')
        fhCmd = open(scriptFile, 'w')
        imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA)
        #imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE)
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        imgData = imgData[imgData!=0]
        min_Res = round(np.amin(imgData)*100)/100
        max_Res = round(np.amax(imgData)*100)/100

        numberOfColors = 21
        colors_labels = self.numberOfColors(min_Res, max_Res, numberOfColors)
        colorList = self.colorMapToColorList(colors_labels, self.getColorMap())
        
        fnbase = removeExt(self.protocol.inputVolume.get().getFileName())
#         ext = getExt(self.protocol.inputVolume.get().getFileName())
#         fninput = abspath(fnbase + ext[0:4])
#         fhCmd.write("open %s\n" % fninput)
        fhCmd.write("open %s\n" % (fnRoot + RESIZE_VOL))
        
        fhCmd.write("open %s\n" % (fnRoot + OUTPUT_RESOLUTION_FILE_CHIMERA))
#        fhCmd.write("open %s\n" % (fnRoot + OUTPUT_RESOLUTION_FILE))        

#        smprt = self.protocol.inputVolume.get().getSamplingRate()        
        smprt = 1.0
        
        fhCmd.write("volume #0 voxelSize %s step 1\n" % (str(smprt)))
        fhCmd.write("volume #1 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("vol #1 hide\n")
        
        scolorStr = '%s,%s:' * numberOfColors
        scolorStr = scolorStr[:-1]

        line = ("scolor #0 volume #1 perPixel false cmap " 
                + scolorStr + "\n") % colorList
        fhCmd.write(line)

        scolorStr = '%s %s ' * numberOfColors
        str_colors = ()
        for idx, elem in enumerate(colorList):
            if (idx % 2 == 0):
                if ((idx % 8) == 0):
                    str_colors +=  str(elem),
                else:
                    str_colors += '" "',
            else:
                str_colors += elem,
        
        line = ("colorkey 0.01,0.05 0.02,0.95 " + scolorStr + "\n") % str_colors
        fhCmd.write(line)

        fhCmd.close()

    @staticmethod
    def colorMapToColorList(steps, colorMap):
        """ Returns a list of pairs resolution, hexColor to be used in chimera 
        scripts for coloring the volume and the colorKey """

        # Get the map used by DL2R
        colors = ()
        ratio = 255.0/(len(steps)-1)
        for index, step in enumerate(steps):
            colorPosition = int(round(index*ratio))
            rgb = colorMap(colorPosition)[:3]
            colors += step,
            rgbColor = mcolors.rgb2hex(rgb)
            colors += rgbColor,

        return colors
    
    def getColorMap(self):
        if (COLOR_CHOICES[self.colorMap.get()] == 'other'):
            cmap = cm.get_cmap(self.otherColorMap.get())
        else:
            cmap = cm.get_cmap(COLOR_CHOICES[self.colorMap.get()])
        if cmap is None:
            cmap = cm.jet
        return cmap

