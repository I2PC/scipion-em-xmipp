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


from pyworkflow.protocol.params import LabelParam, StringParam, EnumParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pwem.viewers import ChimeraView, DataView
from xmipp3.protocols.protocol_resolution_directional import XmippProtMonoDir
from pwem.emlib.metadata import MetaData
from pwem.emlib.image import ImageHandler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from pyworkflow.utils import getExt, removeExt
from os.path import abspath, exists
from collections import OrderedDict

from .plotter import XmippPlotter

from pwem.emlib import *

OUTPUT_RESOLUTION_FILE_CHIMERA = 'MG_Chimera_resolution.vol'

OUTPUT_VARIANCE_FILE_CHIMERA = 'MG_Chimera_resolution.vol'
CHIMERA_CMD_DOA = 'chimera_DoA.cmd'
CHIMERA_CMD_VARIANCE = 'chimera_Variance.cmd'
CHIMERA_CMD_SPH = 'chimera_Sph.cmd'
CHIMERA_ELLIP = 'ellipsoid.vol'

OUTPUT_RADIAL_AVERAGES = 'Radial_averages.xmd'
OUTPUT_RESOLUTION_MEAN = 'mean_volume.vol'
OUTPUT_RESOLUTION_LOWEST_FILE = 'lowestResolution.vol'
OUTPUT_RESOLUTION_HIGHEST_FILE = 'highestResolution.vol'
OUTPUT_VARIANCE_FILE = 'resolution_variance.vol'
OUTPUT_DOA_FILE = 'local_anisotropy.vol'
OUTPUT_DOA1_FILE = 'doaMetric.vol'
OUTPUT_DOA2_FILE = 'meanResdoa.vol'
OUTPUT_SPH_FILE = 'sphericity.vol'
OUTPUT_RADIAL_FILE = 'radial_resolution.vol'
OUTPUT_AZIMUHTAL_FILE = 'azimuthal_resolution.vol'
OUTPUT_THRESHOLDS_FILE = 'thresholds.xmd'
OUTPUT_ZSCOREMAP_FILE = 'zscoreMap.vol'

#TODO: prepare volumes for chimera
OUTPUT_VARIANCE_FILE_CHIMERA = 'varResolution_Chimera.vol'
OUTPUT_DOA_FILE_CHIMERA = 'local_anisotropy.vol'
OUTPUT_RESOLUTION_MEAN_CHIMERA = 'mean_volume_Chimera.vol'

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

class XmippMonoDirViewer(ProtocolViewer):
    """
    Visualization tools for MonoRes results.
    
    MonoDir is a Xmipp package for computing the local anisotropy of 3D
    density maps studied in structural biology, primarily by cryo-electron
    microscopy (cryo-EM).
    """
    _label = 'viewer MonoDir'
    _targets = [XmippProtMonoDir]      
    _environments = [DESKTOP_TKINTER]
    
    @staticmethod
    def getColorMapChoices():
        return plt.colormaps()
   
    def __init__(self, *args, **kwargs):
        ProtocolViewer.__init__(self, *args, **kwargs)


    def _defineParams(self, form):
        form.addSection(label='Visualization')
        
        form.addParam('doShowOriginalVolumeSlices', LabelParam,
              label="Show original volume slices")

        groupDoA = form.addGroup('Anisotropy information')
       
#        groupDoA.addParam('doShowDoAHistogram', LabelParam,
#              label="Show DoA histogram")
        
        groupDoA.addParam('doShowDoAColorPol', LabelParam,
               label="Show Half Interquartile Range")
        
        groupDoA.addParam('doShowDoAColorMean', LabelParam,
               label="Show ADR Average Directional Resolution")
        
#        groupDoA.addParam('doShowChimera', LabelParam,
#                       label="Show DoA map in Chimera")
        
        groupRadAzim = form.addGroup('Radial and azimuthal information')
        
        groupRadAzim.addParam('doShowRadialColorSlices', LabelParam,
               label="Show radial resolution colored slices")
        
        groupRadAzim.addParam('doShowAzimuthalColorSlices', LabelParam,
               label="Show azimuthal resolution colored slices")

        groupRadAzim.addParam('doShowDirectionsHistogram', LabelParam,
               label="Show directions histogram")
        
        groupRadAzim.addParam('doShowDirectionsSphere', LabelParam,
               label="Show directions sphere")
        
        groupRadAzim.addParam('doShowRadialAverages', LabelParam,
               label="Show radial averages")

        group = form.addGroup('Choose a Color Map')
        group.addParam('colorMap', EnumParam, choices=list(COLOR_CHOICES.values()),
                      default=COLOR_JET,
                      label='Color map',
                      help='Select the color map to be applied '
                            'https://matplotlib.org/1.3.0/examples/color/colormaps_reference.html.')
        
        group.addParam('otherColorMap', StringParam, default='jet',
                      condition = binaryCondition,
                      label='Customized Color map',
                      help='Name of a color map to apply to be applied. Valid names can be found at '
                            'https://matplotlib.org/1.3.0/examples/color/colormaps_reference.html')
        group.addParam('sliceAxis', EnumParam, default=AX_Z,
                       choices=['x', 'y', 'z'],
                       display=EnumParam.DISPLAY_HLIST,
                       label='Slice axis')

        
    def _getVisualizeDict(self):
        return {'doShowOriginalVolumeSlices': self._showOriginalVolumeSlices,
                'doShowDoAColorMean': self._showColorMeanResADR,
                'doShowDoAColorPol': self._showHalfInterQuartile,
                'doShowRadialColorSlices': self._showRadialColorSlices,
                'doShowAzimuthalColorSlices': self._showAzimuthalColorSlices,
                #'doShowRadialHistogram': self._plotHistogramRadial,
                #'doShowAzimuthalHistogram': self._plotHistogramAzimuthal,
                'doShowDirectionsHistogram': self._plotHistogramDirections,
                'doShowDirectionsSphere': self._show2DDistribution,
                'doShowRadialAverages': self._showRadialAverages
     }

    def _showDoASlices(self, param=None):
        cm = DataView(self.protocol._getExtraPath(OUTPUT_DOA_FILE))
        return [cm]  
 
    def _showHalfInterQuartile(self, param=None):
        self._showColorSlices(OUTPUT_DOA1_FILE, False, 'Half Interquartile Range', -1, -1)
        
    def _showColorMeanResADR(self, param=None):
        self._showColorSlices(OUTPUT_DOA2_FILE, False, 'Mean resolution (ADR)', -1, -1)
        
    def _showRadialColorSlices(self, param=None):
        self._showColorSlices(OUTPUT_RADIAL_FILE, False, 'Radial Resolution', -1, -1)

    def _showAzimuthalColorSlices(self, param=None):
        self._showColorSlices(OUTPUT_AZIMUHTAL_FILE, False, 'Tangential Resolution', -1, -1)

    def _show2DDistribution(self, param=None):
        self._createAngDist2D(self.protocol._getExtraPath('hist_prefdir.xmd'))
        
    def _showRadialAverages(self, paramName=None):
        xplotter = XmippPlotter(windowTitle="Resolution Radial Averages")
        a = xplotter.createSubPlot("Radial Averages", "Radius (pixel)", "Frequency (A)")
        legends = []
        
        list = ['RadialResolution', 'AzimuthalResolution', 'HighestResolution', 'LowestResolution', 'MonoRes']
        labellist = [MDL_VOLUME_SCORE1, MDL_VOLUME_SCORE2, MDL_VOLUME_SCORE3, MDL_VOLUME_SCORE4, MDL_AVG]
        
        for idx in range(5):
            fnDir = self.protocol._getExtraPath(OUTPUT_RADIAL_AVERAGES)
            lablmd = labellist[idx]
            if exists(fnDir):
                legends.append(list[idx])
                self._plotCurve(a, fnDir, lablmd)
                xplotter.showLegend(legends)
#         a.plot([self.minInv, self.maxInv],[self.resolutionThreshold.get(), self.resolutionThreshold.get()], color='black', linestyle='--')
        a.grid(True)
        views = []
        views.append(xplotter)
        return views
    

    def _plotCurve(self, a, fnDir, labelmd):
        print(labelmd)
        md = MetaData(fnDir)
        resolution = []
        radius = []
        for objId in md:
            resolution.append(md.getValue(labelmd, objId))
            radius.append(md.getValue(MDL_IDX, objId))
#         resolution_inv = [md.getValue(labelmd, objId) for objId in md]
#         frc = [md.getValue(MDL_IDX, objId) for objId in md]
        self.maxFrc = max(radius)
        self.minInv = min(resolution)
        self.maxInv = max(resolution)
        a.plot(radius, resolution, 'x')
#         a.xaxis.set_major_formatter(self._plotFormatter)
#         a.set_ylim([-0.1, 1.1])


    def _createPlot(self, title, xTitle, yTitle, md, mdLabelX, mdLabelY, color='g', figure=None):        
        xplotter = XmippPlotter(figure=figure)
        xplotter.plot_title_fontsize = 11
        xplotter.createSubPlot(title, xTitle, yTitle, 1, 1)
        xplotter.plotMdFile(md, mdLabelX, mdLabelY, color)
        return xplotter
        
    def _showOriginalVolumeSlices(self, param=None):
        if self.protocol.hasAttribute('halfVolumes'):
            cm = DataView(self.protocol.inputVolume.get().getFileName())
            cm2 = DataView(self.protocol.inputVolume2.get().getFileName())
            return [cm, cm2]
        else:
            cm = DataView(self.protocol.inputVolumes.get().getFileName())
            return [cm]
   
    def _showColorSlices(self, fileName, setrangelimits, titleFigure, lowlim, highlim):
        imageFile = self.protocol._getExtraPath(fileName)
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        imgData2 = np.ma.masked_where(imgData < 0.001, imgData, copy=True)
        if setrangelimits is True:
            fig, im = self._plotVolumeSlices(titleFigure, imgData2,
                                         lowlim, highlim, self.getColorMap(), dataAxis=self._getAxis())
        else:
            md =MetaData()
            md.read(self.protocol._getExtraPath(OUTPUT_THRESHOLDS_FILE))
            idx = 1
            val1 = md.getValue(MDL_RESOLUTION_FREQ, idx)
            val2 = md.getValue(MDL_RESOLUTION_FREQ2, idx)

            if val1>=val2:
                max_Res = val1 +0.01
            else:
                max_Res = val2

#             max_Res = np.nanmax(imgData2)
            min_Res = np.nanmin(imgData2)
            fig, im = self._plotVolumeSlices(titleFigure, imgData2,
                                         min_Res, max_Res, self.getColorMap(), dataAxis=self._getAxis())
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.invert_yaxis()

        return plt.show()

    def _plotHistogramDoA(self, param=None):
        self._plotHistogram('hist_DoA.xmd', 'DoA', 'DoA')

    def _plotHistogramDirections(self, param=None):
        self._plotHistogram('hist_prefdir.xmd', 'Highest Resolution per Direction', 'Direction')
        
    def _createAngDist2D(self, path):
        view = XmippPlotter(x=1, y=1, mainTitle="Highest Resolution per Direction", windowTitle="Angular distribution")
        
        view.plotAngularDistributionFromMd(path, 'directional resolution distribution')#,  min_w=0)
        return view.show()

    def _plotHistogram(self, fnhist, titlename, xname):
        md = MetaData()
        md.read(self.protocol._getPath('extra/'+fnhist))
        x_axis = []
        y_axis = []

        i = 0
        for idx in md:
            x_axis_ = md.getValue(MDL_X, idx)
            if i==0:
                x0 = x_axis_
            elif i==1:
                x1 = x_axis_
            y_axis_ = md.getValue(MDL_COUNT, idx)

            i+=1
            x_axis.append(x_axis_)
            y_axis.append(y_axis_)
        delta = x1-x0
        plt.figure()
        plt.bar(x_axis, y_axis, width = delta)
        plt.title(titlename+"Histogram")
        plt.xlabel(xname+"(a.u.)")
        plt.ylabel("Counts")
        
        return plt.show()
  
    
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
        self.createChimeraScriptDoA(OUTPUT_DOA_FILE_CHIMERA, CHIMERA_CMD_DOA, CHIMERA_ELLIP)
        cmdFile = self.protocol._getPath('chimera_DoA.cmd')
        view = ChimeraView(cmdFile)
        return [view]
    

    def numberOfColors(self, min_Res, max_Res, numberOfColors):
        inter = (max_Res - min_Res)/(numberOfColors-1)
        colors_labels = ()
        for step in range(0,numberOfColors):
            colors_labels += round(min_Res + step*inter,2),
        return colors_labels

    def createChimeraScriptDoA(self, infile, outfile, ellipfile):
        fnRoot = "extra/"
        scriptFile = self.protocol._getPath(outfile)
        fhCmd = open(scriptFile, 'w')
        imageFile = self.protocol._getExtraPath(infile)
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        min_Res = 0.0#round(np.amin(imgData)*100)/100
        max_Res = 1.0#round(np.amax(imgData)*100)/100

        numberOfColors = 21
        colors_labels = self.numberOfColors(min_Res, max_Res, numberOfColors)
        colorList = self.colorMapToColorList(colors_labels, self.getColorMap())
        
        fnbase = removeExt(self.protocol.inputVolumes.get().getFileName())
        ext = getExt(self.protocol.inputVolumes.get().getFileName())
        fninput = abspath(fnbase + ext[0:4])
        fhCmd.write("open %s\n" % fninput)
        fhCmd.write("open %s\n" % (fnRoot + infile))
        
        fhCmd.write("open %s\n" % (fnRoot + ellipfile))
        smprt = self.protocol.inputVolumes.get().getSamplingRate()
        fhCmd.write("volume #0 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #1 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #2 voxelSize %s\n" % (str(smprt)))
        fhCmd.write("volume #2 style mesh\n")
        fhCmd.write("vol #1 hide\n")
        
        scolorStr = '%s,%s:' * numberOfColors
        scolorStr = scolorStr[:-1]

        line = ("scolor #0 volume #1 perPixel false cmap " + scolorStr + "\n") % colorList
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

#     def plotAnisotropyResolution(self, path):        
#         from pwem.emlib import XmippPlotter as Xmplt
#  
#         print(path)
#         md = MetaData(path)
#         y = md.getColumnValues(MDL_COST)
#         x = md.getColumnValues(MDL_RESOLUTION_SSNR)
#          
#         plt.figure()
#         plt.plot(x, y,'x')
#         plt.title('resolutoin vs anisotropy')
#         plt.xlabel("resolution")
#         plt.ylabel("Anisotropy")
# #         
#         return plt.show()
    
    def plotAnisotropyResolution(self, path):        
        md = MetaData(path)
        y = md.getColumnValues(MDL_COST)
        x = md.getColumnValues(MDL_RESOLUTION_SSNR)
        
        xmax = np.amax(x)
        xmin = np.amin(x)
        
        ymax = np.amax(y)
        ymin = np.amin(y)
        resstep = 0.5
        anistep = 0.1
        
        from math import floor, ceil
        xbin = floor((xmax - xmin)/resstep)
        ybin = ceil((ymax - ymin)/0.05)
        
        print(xbin, ybin)
        
        from matplotlib.pyplot import contour, contourf
        plt.figure()
#         plt.plot(x, y,'x')
        plt.title('resolution vs anisotropy')
        plt.xlabel("Resolution")
        plt.ylabel("Anisotropy")
         
#         counts,ybins,xbins,image = plt.hist2d(x,y,bins=100)#)[xbin, ybin])
        img, xedges, yedges, imageAx = plt.hist2d(x, y, (50, 50), cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()
        
        xplotter = XmippPlotter(x=1, y=1, mainTitle="aaaaa "
                                                     "along %s-axis."
                                                     %self._getAxis())
        a = xplotter.createSubPlot("Slice ", '', '')
        matrix = img
        print(matrix)
        plot = xplotter.plotMatrix(a, matrix, 0, 200,
                                       cmap=self.getColorMap(),
                                       interpolation="nearest")
        xplotter.getColorBar(plot)
        print(matrix)
        
        return [xplotter]



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
        if (COLOR_CHOICES[self.colorMap.get()] == 'other'):
            cmap = cm.get_cmap(self.otherColorMap.get())
        else:
            cmap = cm.get_cmap(COLOR_CHOICES[self.colorMap.get()])
        if cmap is None:
            cmap = cm.jet
        return cmap
