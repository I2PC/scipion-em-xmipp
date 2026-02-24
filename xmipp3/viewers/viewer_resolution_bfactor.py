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
from matplotlib.colors import ListedColormap
import numpy as np

from pyworkflow.protocol.params import LabelParam, FloatParam, IntParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER
from pwem.viewers import ChimeraView
from pwem import emlib

from xmipp3.protocols.protocol_resolution_bfactor import (XmippProtbfactorResolution,
                                                          FN_METADATA_BFACTOR_RESOLUTION)


class XmippBfactorResolutionViewer(ProtocolViewer):
    """
    The local resolution and local b-factor should present a correlation.
    This viewer provides the matching between them per residue.
    """
    _label = 'viewer resolution bfactor'
    _targets = [XmippProtbfactorResolution]
    _environments = [DESKTOP_TKINTER]

    def __init__(self, *args, **kwargs):
        ProtocolViewer.__init__(self, *args, **kwargs)

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('chain', IntParam, default=-1,
                      label="Chain",)
        groupColorScaleOptions = form.addGroup('Color scale options and representation')

        line = groupColorScaleOptions.addLine("Range normalized bfactor:",
                            help="Options to define the color scale limits and color set")

        line.addParam('highestBF', FloatParam, allowsNull =True,
                      label="Min",
                      help="Minumum value for the scale")

        line.addParam('lowestBF', FloatParam, allowsNull =True,
                      label="Max",
                      help="Maximum value of the scale.")

        line2 = groupColorScaleOptions.addLine("Range normalized local resolution:",
                                              help="Options to define the color scale limits and color set")

        line2.addParam('highestLR', FloatParam, allowsNull =True,
                      label="Highest",
                      help="Highest value for the scale")

        line2.addParam('lowestLR', FloatParam, allowsNull =True,
                      label="Lowest",
                      help="lowest value of the scale.")

        groupColorScaleOptions.addParam('doShowColorBands', LabelParam,
                      label="Show bfactor-local resolution comparison")
        form.addParam('doShowInChimera', LabelParam,
                      label="Show atomic structure in chimera")
                      

    def _getVisualizeDict(self):
        return {'doShowColorBands': self._showColorBands,
                'doShowInChimera': self._showInChimera,
                }

    def _showInChimera(self, param=None):
    
        md = emlib.MetaData()
        if self.chain.get()>0:
            fnXmd, ext = os.path.splitext(FN_METADATA_BFACTOR_RESOLUTION)
            fnXmd = fnXmd+str(self.chain.get())+ext
        else:
            fnXmd = FN_METADATA_BFACTOR_RESOLUTION
            print('full pdb')
        md.read(self.protocol._getExtraPath(fnXmd))
        lr = []
        bf = []
        r = []

        for idx in md:
            lr.append(md.getValue(emlib.MDL_RESOLUTION_LOCAL_RESIDUE, idx))
            bf.append(md.getValue(emlib.MDL_BFACTOR, idx))
            r.append(md.getValue(emlib.MDL_RESIDUE, idx))

        lr = np.array(lr)
        bf = np.array(bf)
        r = np.array(r)

        lowBF = self.lowestBF if self.lowestBF.get() else np.amax(bf)
        highBF = self.highestBF if self.highestBF.get() else np.amin(bf)
        lowLR = float(self.lowestLR) if self.lowestLR.get() else np.amax(lr)
        highLR = float(self.highestLR) if self.highestLR.get() else np.amin(lr)

        fnColoredModel = self.protocol.outputStructure.getFileName()
        scriptFile = self.protocol._getExtraPath('show_coloredAtomicModel.cxc')
        imageFile = os.path.abspath(fnColoredModel)
        
        #colorStr, scolorStr = self.defineColorMapandColorKey()

        OPEN_FILE = "open %s\n"
        fhCmd = open(scriptFile, 'w')

        views = []
        viridisCM = self.viridisColors(highLR, lowLR, useSeparator = True)
        viridisCMSeparator = self.viridisColors2(lowLR, highLR)

        fhCmd.write(OPEN_FILE % os.path.abspath(self.protocol.outputStructure.getFileName()))
        fhCmd.write('color byattribute a:bfactor #1 target cabs palette %s \n' % viridisCMSeparator[:-1])

        fhCmd.write("key %s\n" % viridisCM)
        fhCmd.write("key pos 0.779101,0.316779 size 0.0367816,0.410738")

        fhCmd.close()

        views.append(ChimeraView(scriptFile))

        return views

    def viridisColors(self, vmin, vmax, useSeparator=False):
        # 11 elements for color map
        values = np.linspace(vmin, vmax, 11)
        #viridis = cm.get_cmap("viridis")
        colorPlot = self.customColorMap(False)
        cmd = ''
        separator = ' '
        if useSeparator:
            separator = ':'
        for idx, v in enumerate(values):
            # Color RGBA (0-1)
            r, g, b, a = colorPlot((v - vmin) / (vmax - vmin))
            hexcolor = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
            if (idx==0 or idx==np.floor(len(values)/2) or idx==(len(values)-1)):
                cmd += f"{hexcolor}%s{v:.6g} " % separator
            else:
                cmd += f"{hexcolor}%s " % separator

        return cmd

    def customColorMap(self, reverse=True):
        newColors = ["#30123B", "#3B2F8F", "#3559C7", "#2E8BEF", "#1EBBD7", "#35E8B7", "#7CFC6F", "#B7F735", "#E7F21D", "#FDE725", "#FDE725"]

        if reverse:
            newColorMap = ListedColormap(newColors[::-1])
        else:
            newColorMap = ListedColormap(newColors)
        return newColorMap

    def viridisColors2(self, vmin, vmax, useSeparator=False):
        # 11 elements for color map
        values = np.linspace(vmin, vmax, 11)
        #viridis = cm.get_cmap("viridis")
        cmd = ''
        separator = ' '
        if useSeparator:
            separator = ':'
        colorPlot = self.customColorMap(True)
        for v in values:
            # Color RGBA (0-1)
            r, g, b, a = colorPlot((v - vmin) / (vmax - vmin))
            hexcolor = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))
            cmd += f"{v:.6g},{hexcolor}:" 
        return cmd


    def _showColorBands(self, param=None):
        """
        The local resolution and local b-factor are represented as color bands.
        """
        md = emlib.MetaData()
        if self.chain.get()>0:
            fnXmd, ext = os.path.splitext(FN_METADATA_BFACTOR_RESOLUTION)
            fnXmd = fnXmd+str(self.chain.get())+ext
        else:
            fnXmd = FN_METADATA_BFACTOR_RESOLUTION
        md.read(self.protocol._getExtraPath(fnXmd))
        lr = []
        bf = []
        r = []

        for idx in md:
            lr.append(md.getValue(emlib.MDL_RESOLUTION_LOCAL_RESIDUE, idx))
            bf.append(md.getValue(emlib.MDL_BFACTOR, idx))
            r.append(md.getValue(emlib.MDL_RESIDUE, idx))

        lr = np.array(lr)
        bf = np.array(bf)
        r = np.array(r)

        lowBF = self.lowestBF if self.lowestBF.get() else bf[0]
        highBF = self.highestBF if self.highestBF.get() else bf[-1]
        lowLR = self.lowestLR if self.lowestLR.get() else lr[0]
        highLR = self.highestLR if self.highestLR.get() else lr[-1]

        colorPlot = self.customColorMap(reverse=False)

        plt.figure()
        plt.subplot(211)
        plt.imshow(bf.reshape(1, len(bf)), vmin=lowBF, vmax=highBF, cmap=colorPlot, extent=[np.amin(r), np.amax(r), 0, 80])
        plt.xlabel('Residue')
        plt.title('B-factor')
        
        plt.colorbar()

        plt.subplot(212)
        #The magic numbers of 0 and 40 define the size of the vertical bands, they provide good visualization aspect
        plt.imshow(lr.reshape(1, len(lr)), vmin=lowLR, vmax=highLR, cmap=colorPlot, extent=[np.amin(r), np.amax(r), 0, 80])

        plt.xlabel('Residue')


        plt.colorbar()

        plt.show()
