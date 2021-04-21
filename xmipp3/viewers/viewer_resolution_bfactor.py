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

from pyworkflow.protocol.params import LabelParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER

from pwem import emlib

from xmipp3.protocols.protocol_resolution_bfactor import (XmippProtbfactorResolution,
                                                          FN_METADATA_BFACTOR_RESOLUTION)


class XmippBfactorResolutionViewer(ProtocolViewer):
    """
    Visualization tools for bfactor local resolution matching results.
    
    The local resolution and b-factor should present a correlation.
    This viewer provides the matching between them per residue.
    """
    _label = 'viewer resolution bfactor'
    _targets = [XmippProtbfactorResolution]
    _environments = [DESKTOP_TKINTER]

    def __init__(self, *args, **kwargs):
        ProtocolViewer.__init__(self, *args, **kwargs)

    def _defineParams(self, form):
        form.addSection(label='Visualization')

        form.addParam('doShowColorBands', LabelParam,
                      label="Show bfactor-local resolution comparison")

    def _getVisualizeDict(self):
        return {'doShowColorBands': self._showColorBands,
                }

    def _showColorBands(self, param=None):
        md = emlib.MetaData()
        md.read(self.protocol._getExtraPath(FN_METADATA_BFACTOR_RESOLUTION))
        lr = []
        bf = []
        r = []

        for idx in md:
            lr_aux = md.getValue(emlib.MDL_RESOLUTION_LOCAL_RESIDUE, idx)
            bf_aux = md.getValue(emlib.MDL_BFACTOR, idx)
            r_aux = md.getValue(emlib.MDL_RESIDUE, idx)

            lr.append(lr_aux)
            bf.append(bf_aux)
            r.append(r_aux)
        lr = np.array(lr)
        bf = np.array(bf)
        r = np.array(r)
        plt.subplot(211)
        plt.imshow(lr.reshape(1, len(lr)), cmap=plt.cm.viridis, extent=[r[0], r[-1], 0, 40])
        plt.xlabel('Residue')
        plt.title('Normalized Local Resolution')
        plt.colorbar()

        plt.subplot(212)
        plt.imshow(bf.reshape(1, len(bf)), cmap=plt.cm.viridis, extent=[r[0], r[-1], 0, 40])
        plt.xlabel('Residue')
        plt.title('B-factor')
        plt.colorbar()

        plt.show()

