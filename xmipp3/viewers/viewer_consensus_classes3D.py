# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

"""
This module implement the wrappers around
visualization program.
"""
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)

from pwem.viewers.showj import *

from xmipp3.protocols.protocol_consensus_classes3D import XmippProtConsensusClasses3D


class XmippConsensusClasses3DViewer(ProtocolViewer):
    """ Visualization of results from the consensus classes 3D protocol
    """
    _label = 'viewer consensus classes 3D'
    _targets = [XmippProtConsensusClasses3D]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    def _defineParams(self, form):
        pass

    def _getVisualizeDict(self):
        pass
    
    # --------------------------- UTILS functions ------------------------------

