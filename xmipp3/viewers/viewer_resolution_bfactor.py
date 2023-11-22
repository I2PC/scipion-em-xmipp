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
import numpy as np

from pyworkflow.protocol.params import LabelParam, FloatParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER

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

        groupColorScaleOptions = form.addGroup('Color scale options and representation')

        line = groupColorScaleOptions.addLine("Range normalized bfactor:",
                            help="Options to define the color scale limits and color set")

        line.addParam('highestBF', FloatParam, allowsNull =True,
                      label="Highest",
                      help="Highest value for the scale")

        line.addParam('lowestBF', FloatParam, allowsNull =True,
                      label="Lowest",
                      help="lowest value of the scale.")

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



    def _getVisualizeDict(self):
        return {'doShowColorBands': self._showColorBands,
                }

    def _showColorBands(self, param=None):
        """
        The local resolution and local b-factor are represented as color bands.
        """
        md = emlib.MetaData()
        md.read(self.protocol._getExtraPath(FN_METADATA_BFACTOR_RESOLUTION))
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

        plt.figure()
        plt.subplot(211)
        #The magic numbers of 0 and 40 define the size of the vertical bands, they provide good visualization aspect
        plt.imshow(lr.reshape(1, len(lr)), vmin=lowLR, vmax=highLR, cmap=plt.cm.viridis, extent=[np.amin(r), np.amax(r), 0, 80])

        plt.xlabel('Residue')
        plt.title('Normalized Local Resolution')
        plt.colorbar()

        plt.subplot(212)

        plt.imshow(bf.reshape(1, len(bf)), vmin=lowBF, vmax=highBF, cmap=plt.cm.viridis, extent=[np.amin(r), np.amax(r), 0, 80])
        plt.xlabel('Residue')
        plt.title('B-factor')
        plt.colorbar()

        plt.show()
