# **************************************************************************
# *
# * Authors:     J.L. Vilas (jlvilas@cnb.csic.es)
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

from pyworkflow.gui.plotter import Plotter
from xmipp3.viewers.plotter import XmippPlotter
from pyworkflow.protocol.params import LabelParam, StringParam, EnumParam, IntParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pwem.viewers import ChimeraView, DataView, LocalResolutionViewer
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import hlines
from xmipp3.protocols.protocol_resolution_fso import \
        XmippProtFSO, OUTPUT_3DFSC, OUTPUT_SPHERE, OUTPUT_DIRECTIONAL_FILTER
from pwem.emlib.metadata import MetaData
from xmippLib import *
from pwem.emlib.image import ImageHandler
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import cm
import matplotlib.colors as mcolors
from pyworkflow.utils import getExt, removeExt
from os.path import abspath, exists
from collections import OrderedDict
import xmippLib

# Color maps
COLOR_JET = 0
COLOR_TERRAIN = 1
COLOR_GIST_EARTH = 2
COLOR_GIST_NCAR = 3
COLOR_GNU_PLOT = 4
COLOR_GNU_PLOT2 = 5
COLOR_OTHER = 6

COLOR_CHOICES = OrderedDict() #[-1]*(OP_RESET+1)

COLOR_CHOICES[COLOR_JET]  = 'jet'
COLOR_CHOICES[COLOR_TERRAIN] = 'terrain'
COLOR_CHOICES[COLOR_GIST_EARTH] = 'gist_earth'
COLOR_CHOICES[COLOR_GIST_NCAR] = 'gist_ncar'
COLOR_CHOICES[COLOR_GNU_PLOT] = 'gnuplot'
COLOR_CHOICES[COLOR_GNU_PLOT2] = 'gnuplot2'
COLOR_CHOICES[COLOR_OTHER] = 'other'

binaryCondition = ('(colorMap == %d) ' % (COLOR_OTHER))

#Axis code
AX_X = 0
AX_Y = 1
AX_Z = 2

class XmippProtFSOViewer(ProtocolViewer):
    """
    Visualization tools for the FSO, FSC, and 3DFSC.
    
    """
    _label = 'viewer FSO'
    _targets = [XmippProtFSO]      
    _environments = [DESKTOP_TKINTER]
    
    @staticmethod
    def getColorMapChoices():
        return plt.colormaps()
   
    def __init__(self, *args, **kwargs):
        ProtocolViewer.__init__(self, *args, **kwargs)


    def _defineParams(self, form):
        form.addSection(label='Visualization')
        
        form.addParam('doShowOriginalVolumeSlices', LabelParam,
              label="Original Half Maps slices")

        form.addParam('doShow3DFSC', LabelParam, label="3DFSC map")

        form.addParam('doShowDirectionalFilter', LabelParam, 
              label="Directionally filtered map")
        
        form.addParam('doShowFSC', LabelParam, label="Global FSC")

        form.addParam('doShowDirectionalResolution', LabelParam, 
              label="Show Directional Resolution on sphere")

        form.addParam('doCrossValidation', LabelParam, 
              label="Show Cross validation curve")

        groupDirFSC = form.addGroup('FSC Anisotropy')
        
        groupDirFSC.addParam('doShow3DFSCcolorSlices', LabelParam,
               label="Show 3DFSC Color slices")
        
        groupDirFSC.addParam('doShowChimera3DFSC', LabelParam,
               label="Show 3DFSC in Chimera")
        
        groupDirFSC.addParam('doShowDirectionalFSCCurve', LabelParam,
               label="Show directional FSC curve")

        groupDirFSC.addParam('fscNumber', IntParam, default=0,
               label='Show FSC direction')

        groupDirFSC.addParam('doShowAnisotropyCurve', LabelParam,
               label="Show FSO curve")

        group = form.addGroup('Choose a Color Map')

        group.addParam('colorMap', EnumParam, choices=list(COLOR_CHOICES.values()),
               default=COLOR_JET, label='Color map', 
               help='Select the color map to be applied'
                    'http://matplotlib.org/1.3.0/examples/color/colormaps_reference.html.')

        group.addParam('otherColorMap', StringParam, default='jet',
                      condition = binaryCondition, label='Customized Color map',
                      help='Name of a color map to apply to be applied. Valid names can be found at '
                            'http://matplotlib.org/1.3.0/examples/color/colormaps_reference.html')

        group.addParam('sliceAxis', EnumParam, default=AX_Z,
                       choices=['x', 'y', 'z'],
                       display=EnumParam.DISPLAY_HLIST,
                       label='Slice axis')
        group.addParam('doShowOneColorslice', LabelParam,  
                      label='Show selected slice')
        group.addParam('sliceNumber', IntParam, default=-1,
               label='Show slice number')

        
    def _getVisualizeDict(self):
        self.protocol._createFilenameTemplates()
        return {'doShowOriginalVolumeSlices': self._showOriginalVolumeSlices,
                'doShow3DFSC': self._show3DFSC,
		'doShowFSC': self._showFSCCurve,
		'doShowDirectionalResolution': self._showDirectionalResolution,
		'doCrossValidation': self._showCrossValidationCurve,
                'doShow3DFSCcolorSlices': self._show3DFSCcolorSlices,
		'doShowOneColorslice': self._showOneColorslice,
                'doShowChimera3DFSC': self._showChimera3DFSC,
                'doShowDirectionalFSCCurve': self._showDirectionalFSCCurve,
		'doShowAnisotropyCurve': self._showAnisotropyCurve,
		'doShowDirectionalFilter': self._showDirectionalFilter,
     }


    def _showOriginalVolumeSlices(self, param=None):
        cm = DataView(self.protocol.half1.get().getFileName())
        cm2 = DataView(self.protocol.half2.get().getFileName())
        return [cm, cm2]

    def _show3DFSC(self, param=None):
        cm = DataView(self.protocol._getExtraPath(OUTPUT_3DFSC))
        return [cm]

    def _showFSCCurve(self, paramName=None):
        fnmd = self.protocol._getExtraPath('fsc/'+'GlobalFSC.xmd')
        title = 'Global FSC'
        xTitle = 'Resolution (1/A)'
        yTitle = 'FSC (a.u.)'
        mdLabelX = xmippLib.MDL_RESOLUTION_FREQ
        mdLabelY = xmippLib.MDL_RESOLUTION_FRC
        self._plotCurveFSC(fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY)

    def _showCrossValidationCurve(self, paramName=None):
        fnmd = self.protocol._getExtraPath('fsc/'+'crossValidation.xmd')
        title = 'Cross Validation'
        xTitle = 'Angle (degrees)'
        yTitle = 'Score'
        mdLabelX = xmippLib.MDL_ANGLE_Y
        mdLabelY = xmippLib.MDL_SUM

        md = xmippLib.MetaData(fnmd)
        xplotter = XmippPlotter(figure=None)
        xplotter.plot_title_fontsize = 11
	
        a = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        xplotter.plotMdFile(md, mdLabelX, mdLabelY, 'g')
        a.grid(True)
	
        return plt.show(xplotter)

    def _show3DFSCcolorSlices(self, param=None):
        self._showColorSlices(OUTPUT_3DFSC, 1, '3DFSC Color Slices', 1, 1)

    def _showChimera3DFSC(self, param=None):
        fnmap = abspath(self.protocol._getFileName(OUTPUT_3DFSC)) #'extra/'+
        fnsphere = abspath(self.protocol._getFileName(OUTPUT_SPHERE)) #'extra/fsc/'
        self.createChimeraScript(fnmap, fnsphere)
        cmdFile = self.protocol._getExtraPath('chimeraVisualization.cmd')
        view = ChimeraView(cmdFile)
        return [view]

    def _showDirectionalFSCCurve(self, paramName=None):
        fnmd = self.protocol._getExtraPath('fsc/'+'fscDirection_%i.xmd'% self.fscNumber.get())
        title = 'Directional FSC'
        xTitle = 'Resolution (1/A)'
        yTitle = 'FSC (a.u.)'
        mdLabelX = xmippLib.MDL_RESOLUTION_FREQ
        mdLabelY = xmippLib.MDL_RESOLUTION_FRC
        self._plotCurveFSC(fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY)

    def _showAnisotropyCurve(self, paramName=None):
        fnmd = self.protocol._getExtraPath('fso.xmd')
        title = 'Anisotropy Curve'
        xTitle = 'Resolution (1/A)'
        yTitle = 'Anisotropy (a.u.)'
        mdLabelX = xmippLib.MDL_RESOLUTION_FREQ
        mdLabelY = xmippLib.MDL_RESOLUTION_FRC
        self._plotCurveAnisotropy(fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY)


    def _showDirectionalFilter(self, param=None):
        cm = DataView(self.protocol._getExtraPath(OUTPUT_DIRECTIONAL_FILTER))
        return [cm, cm2]

    def _showOneColorslice(self, param=None):
        img = ImageHandler().read(self.protocol._getExtraPath(OUTPUT_3DFSC))
        imgData = img.getData()
        #imgData2 = np.ma.masked_where(imgData < 0.001, imgData, copy=True)
        max_Res = np.nanmax(imgData)
        min_Res = np.nanmin(imgData)
	#imageFile = self.protocol._getExtraPath(OUTPUT_3DFSC)
        #imgData, min_Res, max_Res = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=1, y=1, mainTitle="3DFSC Slices "
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

        return [plt.show(xplotter)]

    def _formatFreq(self, value, pos):
        """ Format function for Matplotlib formatter. """
        inv = 999.
        if value:
            inv = 1 / value
        return "1/%0.2f" % inv

    def _plotCurveFSC(self, fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY):
        md = xmippLib.MetaData(fnmd)
        xplotter = XmippPlotter(figure=None)
        xplotter.plot_title_fontsize = 11
	
        a = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        xplotter.plotMdFile(md, mdLabelX, mdLabelY, 'g')

        a.xaxis.set_major_formatter(FuncFormatter(self._formatFreq))
        xx, yy = self._prepareDataForPlot( md, mdLabelX, mdLabelY)
        a.hlines(0.143, xx[0], xx[-1], colors = 'k', linestyles = 'dashed')
        a.grid(True)
	
        return plt.show(xplotter)

    def _plotCurveAnisotropy(self, fnmd, title, xTitle, yTitle, mdLabelX, mdLabelY):
        md = xmippLib.MetaData(fnmd)
        xplotter = XmippPlotter(figure=None)
        xplotter.plot_title_fontsize = 11
	
        a = xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        xplotter.plotMdFile(md, mdLabelX, mdLabelY, 'g')

        a.xaxis.set_major_formatter(FuncFormatter(self._formatFreq))
        xx, yy = self._prepareDataForPlot( md, mdLabelX, mdLabelY)
        a.hlines(0.9, xx[0], xx[-1], colors = 'k', linestyles = 'dashed')
        a.hlines(0.5, xx[0], xx[-1], colors = 'k', linestyles = 'dashed')
        a.hlines(0.1, xx[0], xx[-1], colors = 'k', linestyles = 'dashed')
        a.grid(True)
	
        return plt.show(xplotter)


    def _prepareDataForPlot(self, md, mdLabelX, mdLabelY):
        """ plot metadata columns mdLabelX and mdLabelY
            if nbins is in args then and histogram over y data is made
        """
        if mdLabelX:
            xx = []
        else:
            xx = range(1, md.size() + 1)
        yy = []
        for objId in md:
            if mdLabelX:
                xx.append(md.getValue(mdLabelX, objId))
            yy.append(md.getValue(mdLabelY, objId))
        return xx, yy


    def _showDirectionalResolution(self,zparam=None):
        fnmd = self.protocol._getExtraPath('fsc/Resolution_Distribution.xmd')
        titleName = 'Directional FSC distribution'
        self._showPolarPlotNew(fnmd, titleName, 'ResDist')


    def _showPolarPlotNew(self, fnmd, titleName, kind):
        pass

    def _showColorSlices(self, fileName, setrangelimits, titleFigure, lowlim, highlim):
        
        img = ImageHandler().read(self.protocol._getExtraPath(fileName))
        imgData = img.getData()
        #imgData2 = np.ma.masked_where(imgData < 0.001, imgData, copy=True)
        max_Res = np.nanmax(imgData)
        min_Res = np.nanmin(imgData)
        fig, im = self._plotVolumeSlices(titleFigure, imgData,
                                         min_Res, max_Res, self.getColorMap(), dataAxis=self._getAxis())
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.invert_yaxis()

        return plt.show(fig)
        
    def _createAngDist2D(self, path):
        view = XmippPlotter(x=1, y=1, mainTitle="Highest Resolution per Direction", windowTitle="Angular distribution")
        return view.plotAngularDistributionFromMd(path, 'directional resolution distribution',  min_w=0)
    
    def _getAxis(self):
        return self.getEnumText('sliceAxis')


    def _plotVolumeSlices(self, title, volumeData, vminData, vmaxData, cmap, **kwargs):
        """ Helper function to create plots of volumes slices. 
        Params:
            title: string that will be used as title for the figure.
            volumeData: numpy array representing a volume from where to take the slices.
            cmap: color map to represent the slices.
        """
        # Get some customization parameters, by providing with default values
        titleFontSize = kwargs.get('titleFontSize', 14)
        titleColor = kwargs.get('titleColor','#104E8B')
        sliceFontSize = kwargs.get('sliceFontSize', 10)
        sliceColor = kwargs.get('sliceColor', '#104E8B')
        size = kwargs.get('n', volumeData.shape[0])
        origSize = kwargs.get('orig_n', size)
        dataAxis = kwargs.get('dataAxis', 'z')
    
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        f.suptitle(title, fontsize=titleFontSize, color=titleColor, fontweight='bold')
    
        def getSlice(slice):
            if dataAxis == 'y':
                return volumeData[:,slice,:]
            elif dataAxis == 'x':
                return volumeData[:,:,slice]
            else:
                return volumeData[slice,:,:]
    
        def showSlice(ax, index):
            sliceTitle = 'Slice %s' % int(index*size/9)
            slice = int(index*origSize/9)
            ax.set_title(sliceTitle, fontsize=sliceFontSize, color=sliceColor)
            return ax.imshow(getSlice(slice), vmin=vminData, vmax=vmaxData,
                             cmap=self.getColorMap(), interpolation="nearest")
        
        im = showSlice(ax1, 3)
        showSlice(ax2, 4)
        showSlice(ax3, 5)
        showSlice(ax4, 6)
        
        return f, im 

    def _showChimera(self,  param=None):
        self.createChimeraScript(OUTPUT_DOA_FILE_CHIMERA, CHIMERA_CMD_DOA, CHIMERA_ELLIP)
        cmdFile = self.protocol._getPath('chimera_DoA.cmd')
        view = ChimeraView(cmdFile)
        return [view]
    

    def numberOfColors(self, min_Res, max_Res, numberOfColors):
        inter = (max_Res - min_Res)/(numberOfColors-1)
        colors_labels = ()
        for step in range(0,numberOfColors):
            colors_labels += round(min_Res + step*inter,2),
        return colors_labels


    def createChimeraScript(self, map1, map2):
        scriptFile = self.protocol._getExtraPath('chimeraVisualization.cmd')
        fhCmd = open(scriptFile, 'w')
        min_Val = 0.0
        max_Val = 1.0

        numberOfColors = 21
        colors_labels = self.numberOfColors(min_Val, max_Val, numberOfColors)
        colorList = self.colorMapToColorList(colors_labels, self.getColorMap())

        fhCmd.write("open %s\n" % map1)
        fhCmd.write("open %s\n" % map2)
        
        smprt = self.protocol.half1.get().getSamplingRate()
        fhCmd.write("volume #0 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #1 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #1 style mesh\n")
        fhCmd.write("vol #1 hide\n")
        
        scolorStr = '%s,%s:' * numberOfColors
        scolorStr = scolorStr[:-1]

        line = ("scolor #0 volume #0 perPixel false cmap " + scolorStr + "\n") % colorList
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
        """ Returns a list of pairs resolution, hexColor to be used in chimera scripts for coloring the volume and
        the colorKey """

        # Get the map used by monoRes
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
        if (COLOR_CHOICES[self.colorMap.get()] is 'other'): 
            cmap = cm.get_cmap(self.otherColorMap.get())
        else:
            cmap = cm.get_cmap(COLOR_CHOICES[self.colorMap.get()])
        if cmap is None:
            cmap = cm.jet
        return cmap
