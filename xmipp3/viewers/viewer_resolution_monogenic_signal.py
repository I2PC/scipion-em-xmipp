# -*- coding: utf-8 -*-
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
import os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from pwem.viewers.viewer_chimera import mapVolsWithColorkey
from pyworkflow.utils import getExt, removeExt, replaceExt
from os.path import abspath
import numpy as np

from pyworkflow.protocol.params import (LabelParam, StringParam, EnumParam,
                                        IntParam, LEVEL_ADVANCED)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER

from xmipp3.viewers.viewer_resolution_directional import (COLOR_OTHER,
                                                          COLOR_CHOICES,
                                                          COLOR_JET, AX_Z)

from pwem.convert import Ccp4Header
from pwem.emlib.image import ImageHandler
from pwem.viewers import (LocalResolutionViewer, EmPlotter, ChimeraView,
                          DataView, Chimera)
from pwem.emlib.metadata import MetaData, MDL_X, MDL_COUNT

from xmipp3.protocols.protocol_resolution_monogenic_signal import (
        XmippProtMonoRes, OUTPUT_RESOLUTION_FILE, FN_METADATA_HISTOGRAM,
        OUTPUT_RESOLUTION_FILE_CHIMERA, CHIMERA_RESOLUTION_VOL)
from .plotter import XmippPlotter
from pyworkflow.gui import plotter


binaryCondition = ('(colorMap == %d) ' % (COLOR_OTHER))


class XmippMonoResViewer(LocalResolutionViewer):
    """
    Visualization tools for MonoRes results.
    
    MonoRes is a Xmipp packagefor computing the local resolution of 3D
    density maps studied in structural biology, primarily by cryo-electron
    microscopy (cryo-EM).
    """
    _label = 'viewer MonoRes'
    _targets = [XmippProtMonoRes]      
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
        group.addParam('colorMap', EnumParam, choices=COLOR_CHOICES.values(),
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
        if self.protocol.hasAttribute('halfVolumes'):
            cm = DataView(self.protocol.inputVolume.get().getFileName())
            cm2 = DataView(self.protocol.inputVolume2.get().getFileName())
            return [cm, cm2]
        else:
            cm = DataView(self.protocol.inputVolumes.get().getFileName())
            return [cm]
    
    def _showVolumeColorSlices(self, param=None):
        imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE)
        if not os.path.exists(imageFile):
            imageFile = replaceExt(imageFile, 'vol')
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
        if not os.path.exists(imageFile):
            imageFile = replaceExt(imageFile, 'vol')
        imgData, min_Res, max_Res = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=1, y=1, mainTitle="Local Resolution Slices "
                                                     "along %s-axis."
                                                     %self._getAxis())
        sliceNumber = self.sliceNumber.get()
        if sliceNumber < 0:
            x ,_ ,_ ,_ = ImageHandler().getDimensions(imageFile)
            sliceNumber = int(x/2)
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

        _plotter = EmPlotter()
        _plotter.createSubPlot("Resolutions Histogram",
                              "Resolution (A)", "# of Counts")
        barwidth = (x_axis[-1] - x_axis[0])/len(x_axis)

        _plotter.plotDataBar(x_axis, y_axis, barwidth)

        return [_plotter]

    def _getAxis(self):
        return self.getEnumText('sliceAxis')

    def _showChimera(self, param=None):
        cmdFile = self.protocol._getPath('Chimera_resolution.py')
        self.createChimeraScript(cmdFile)
        view = ChimeraView(cmdFile)
        return [view]

    def _getStepColors(self, minRes, maxRes, numberOfColors=13):
        inter = (maxRes - minRes) / (numberOfColors - 1)
        rangeList = []
        for step in range(0, numberOfColors):
            rangeList.append(round(minRes + step * inter, 2))
        return rangeList

    def createChimeraScript(self, cmdFile):
        #  chimera python script
        imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA)
        if not os.path.exists(imageFile):
            imageFile = replaceExt(imageFile, 'vol')
        
        img = ImageHandler().read(imageFile)
        imgData = img.getData()
        min_Res = round(np.amin(imgData)*100)/100
        max_Res = round(np.amax(imgData)*100)/100

        numberOfColors = 21
        voldim = (img.getDimensions())[:-1]

        stepColors = self._getStepColors(min_Res, max_Res, numberOfColors)
        colorList = plotter.getHexColorList(stepColors, self.getColorMap())

        if self.protocol.halfVolumes.get():
            fnbase = removeExt(self.protocol.inputVolume.get().getFileName())
            inputVolume = self.protocol.inputVolume.get()
        else:
            fnbase = removeExt(self.protocol.inputVolumes.get().getFileName())
            inputVolume = self.protocol.inputVolumes.get()

        ext = getExt(inputVolume.getFileName())
        fninput = abspath(fnbase + ext[0:4])

        imageFile = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA)

        if not os.path.exists(imageFile):
            if self.protocol.halfVolumes.get():
                smprt = self.protocol.inputVolume.get().getSamplingRate()
            else:
                smprt = self.protocol.inputVolumes.get().getSamplingRate()

            imageFile = replaceExt(imageFile, 'vol')
            mapVolsWithColorkey(fninput,
                os.path.abspath(imageFile),
                stepColors,
                colorList,
                voldim,
                volOrigin=None,
                step=-1,
                sampling=smprt,
                scriptFileName=cmdFile,
                bgColorImage='black',
                showAxis=True)
        else:
            imageFileVolume = self.protocol._getFileName(OUTPUT_RESOLUTION_FILE_CHIMERA)
            header = Ccp4Header(imageFileVolume, readHeader=True)
            x, y, z = header.getSampling()
            smprt = x
            mapVolsWithColorkey(fninput,
                os.path.abspath(
                    self.protocol._getExtraPath(CHIMERA_RESOLUTION_VOL)),
                stepColors,
                colorList,
                voldim,
                volOrigin=None,
                step=1,
                sampling=smprt,
                scriptFileName=cmdFile,
                bgColorImage='black',
                showAxis=True)
    
    def getColorMap(self):
        if (COLOR_CHOICES[self.colorMap.get()] == 'other'):
            cmap = cm.get_cmap(self.otherColorMap.get())
        else:
            cmap = cm.get_cmap(COLOR_CHOICES[self.colorMap.get()])
        if cmap is None:
            cmap = cm.jet
        return cmap

