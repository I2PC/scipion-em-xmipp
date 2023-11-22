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
from pwem.wizards import ColorScaleWizardBase
from pyworkflow.utils import replaceExt
from os.path import exists

from pyworkflow.protocol.params import (LabelParam, EnumParam, PointerParam,
                                        IntParam, LEVEL_ADVANCED)
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER

from xmipp3.viewers.viewer_resolution_directional import AX_Z
from pwem.viewers import (LocalResolutionViewer, EmPlotter, ChimeraView,
                          DataView)
from pwem.emlib.metadata import MetaData, MDL_X, MDL_COUNT

from xmipp3.protocols.protocol_resolution_monogenic_signal import (
    XmippProtMonoRes, OUTPUT_RESOLUTION_FILE, FN_METADATA_HISTOGRAM,
    OUTPUT_RESOLUTION_FILE_CHIMERA)
from .plotter import XmippPlotter
from pyworkflow.gui import plotter


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
                       label="Show Resolution map in ChimeraX")

        ColorScaleWizardBase.defineColorScaleParams(group, defaultLowest=self.protocol.min_res_init,
                                                    defaultHighest=self.protocol.max_res_init)

        group.addParam('sharpenedMap', PointerParam, pointerClass='Volume',
                       label="(Optional) Color a sharpen map by local resolution in ChimeraX",
                       allowsNull=True,
                       help='Local resolution should be estimated with the raw maps instead'
                            ' of sharpen maps. Information about this in (Vilas et al '
                            'Current Opinion in Structural Biology 2021). This entry parameter '
                            'allows to color the local resolution in'
                            'a different map')

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
        if self.protocol.useHalfVolumes.get():
            if self.protocol.hasHalfVolumesFile.get():
                fn1, fn2 = self.protocol.associatedHalves.get().getHalfMaps().split(',')
            else:
                fn1 = self.protocol.halfMap1.get().getFileName()
                fn2 = self.protocol.halfMap2.get().getFileName()
            cm = DataView(fn1)
            cm2 = DataView(fn2)
            return [cm, cm2]
        else:
            cm = DataView(self.protocol.fullMap.get().getFileName())
            return [cm]

    def _showVolumeColorSlices(self, param=None):
        if (exists(self.protocol._getExtraPath("mgresolution.mrc"))):
            imageFile = self.protocol._getExtraPath("mgresolution.mrc")
        else:
            imageFile = self.protocol._getExtraPath(OUTPUT_RESOLUTION_FILE)
        if not os.path.exists(imageFile):
            imageFile = replaceExt(imageFile, 'mrc')
        imgData, min_Res, max_Res, voldim = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=2, y=2, mainTitle="Local Resolution Slices "
                                                    "along %s-axis."
                                                    % self._getAxis())
        # The slices to be shown are close to the center. Volume size is divided in
        # 9 segments, the fouth central ones are selected i.e. 3,4,5,6
        for i in range(3, 7):
            sliceNumber = self.getSlice(i, imgData)
            a = xplotter.createSubPlot("Slice %s" % (sliceNumber + 1), '', '')
            matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
            plot = xplotter.plotMatrix(a, matrix, self.lowest.get(), self.highest.get(),
                                       cmap=self.getColorMap(),
                                       interpolation="nearest")
        xplotter.getColorBar(plot)
        return [xplotter]

    def _showOneColorslice(self, param=None):
        if (exists(self.protocol._getExtraPath("mgresolution.mrc"))):
            imageFile = self.protocol._getExtraPath("mgresolution.mrc")
        else:
            imageFile = self.protocol._getExtraPath(OUTPUT_RESOLUTION_FILE)
        if not os.path.exists(imageFile):
            imageFile = replaceExt(imageFile, 'mrc')
        imgData, min_Res, max_Res, voldim = self.getImgData(imageFile)

        xplotter = XmippPlotter(x=1, y=1, mainTitle="Local Resolution Slices "
                                                    "along %s-axis."
                                                    % self._getAxis())
        sliceNumber = self.sliceNumber.get()
        if sliceNumber < 0:
            sliceNumber = int(voldim[0] / 2)
        else:
            sliceNumber -= 1
        # sliceNumber has no sense to start in zero
        a = xplotter.createSubPlot("Slice %s" % (sliceNumber + 1), '', '')
        matrix = self.getSliceImage(imgData, sliceNumber, self._getAxis())
        plot = xplotter.plotMatrix(a, matrix, self.lowest.get(), self.highest.get(),
                                   cmap=self.getColorMap(),
                                   interpolation="nearest")
        xplotter.getColorBar(plot)
        return [xplotter]

    def _plotHistogram(self, param=None):
        md = MetaData()
        md.read(self.protocol._getExtraPath(FN_METADATA_HISTOGRAM))
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
        if len(x_axis) == 1:
            barwidth = 0.1
            _plotter.plotDataBar(x_axis, y_axis, barwidth)
        else:
            barwidth = (x_axis[-1] - x_axis[0]) / len(x_axis)
            _plotter.plotDataBar(x_axis[:-2], y_axis[:-2], barwidth)

        return [_plotter]

    def _getAxis(self):
        return self.getEnumText('sliceAxis')

    def _showChimera(self, param=None):

        if (exists(self.protocol._getExtraPath("MG_Chimera_resolution.mrc"))):
            fnResVol = self.protocol._getExtraPath("MG_Chimera_resolution.mrc")
        else:
            fnResVol = self.protocol._getExtraPath(OUTPUT_RESOLUTION_FILE_CHIMERA)

        if self.sharpenedMap.get():
            fnOrigMap = self.sharpenedMap.get().getFileName()
            sampRate = self.sharpenedMap.get().getSamplingRate()
        else:
            if self.protocol.useHalfVolumes.get():
                if self.protocol.hasHalfVolumesFile.get():
                    vol = self.protocol.associatedHalves.get().getHalfMaps()
                    fnOrigMap, _unused = vol.split(',')
                    sampRate = self.protocol.associatedHalves.get().getSamplingRate()
                else:
                    vol = self.protocol.halfMap1.get()
                    fnOrigMap = vol.getFileName()
                    sampRate = vol.getSamplingRate()
            else:
                vol = self.protocol.fullMap.get()
                fnOrigMap = vol.getFileName()
                sampRate = vol.getSamplingRate()

        cmdFile = self.protocol._getExtraPath('chimera_resolution_map.py')
        self.createChimeraScript(cmdFile, fnResVol, fnOrigMap, sampRate,
                                 numColors=self.intervals.get(),
                                 lowResLimit=self.highest.get(),
                                 highResLimit=self.lowest.get())
        view = ChimeraView(cmdFile)
        return [view]

    def getColorMap(self):
        cmap = cm.get_cmap(self.colorMap.get())
        if cmap is None:
            cmap = cm.jet
        return cmap
