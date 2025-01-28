# **************************************************************************
# *
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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

from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
import pyworkflow.protocol.params as params

from pwem.objects import SetOfParticles, CTFModel, Coordinate

from xmipp3.protocols.protocol_apply_tilt_to_ctf import XmippProtApplyTiltToCtf

import matplotlib.pyplot as plt

class XmippApplyTiltToCTFViewer(ProtocolViewer):
    """ Visualization of the output of apply_tilt_to_ctf protocol
    """
    _label = 'viewer apply tilt to ctg'
    _targets = [XmippProtApplyTiltToCtf]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayLocalDefocus', params.IntParam, label='Display local defocus of micrograph:',
                      help="""Type the ID of the micrograph to see particle local defocus of the selected micrograph. 
                      It is possible that not all the micrographs are available.
                      Please check the ID of the micrographs in the output set of micrographs of this protocol.""")

    def _getVisualizeDict(self):
        return {
            'displayLocalDefocus': self._viewLocalDefocus,
        }

    def _viewLocalDefocus(self, _):
        particles: SetOfParticles = self.protocol.outputParticles
        micrographId: int = self.displayLocalDefocus.get()
        
        x = []
        y = []
        defocus = []
        for particle in particles.iterItems(where='_micId=%d' % micrographId):
            coordinate: Coordinate = particle.getCoordinate()
            ctf: CTFModel = particle.getCTF()
            
            x.append(coordinate.getX())
            y.append(coordinate.getY())
            defocus.append((ctf.getDefocusU() + ctf.getDefocusV()) / 2)
                
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, defocus)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Defocus')
        
        return [fig]
