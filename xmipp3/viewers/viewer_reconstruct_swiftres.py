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

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import LabelParam, IntParam, FloatParam

from pwem import emlib
from pwem.viewers.views import DataView

from xmipp3.protocols.protocol_reconstruct_swiftres import XmippProtReconstructSwiftres



class XmippReconstructSwiftresViewer(ProtocolViewer):
    """ Wrapper to visualize different type of data objects
    with the Xmipp program xmipp_showj
    """
    _label = 'viewer swiftres'
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]
    _targets = [XmippProtReconstructSwiftres]
    
    # --------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='FSC')
        form.addParam('showIterationFsc', IntParam, label='Display iteration FSC',
                      default=0 )
        form.addParam('showClassFsc', IntParam, label='Display class FSC',
                      default=0)
        form.addParam('showFscCutoff', FloatParam, label='Display FSC cutoff',
                      default=0.143)

    def _getVisualizeDict(self):
        return {
            'showIterationFsc': self._showIterationFsc,
            'showClassFsc': self._showClassFsc,
            'showFscCutoff': self._showFscCutoff
        }
    

    # --------------------------- UTILS functions ------------------------------
    def _getIterationCount(self) -> int:
        return self.protocol._getIterationCount()
    
    def _getClassCount(self) -> int:
        return self.protocol._getClassCount()

    def _getSamplingRate(self) -> float:
        return self.protocol._getSamplingRate()
    
    def _getFscFilename(self, iteration: int, cls: int):
        return self.protocol._getFscFilename(iteration, cls)
    
    def _readFsc(self, filename: str) -> np.ndarray:
        fscMd = emlib.MetaData(filename)
        
        values = []
        for objId in fscMd:
            freq = fscMd.getValue(emlib.MDL_RESOLUTION_FREQ, objId)
            fsc = fscMd.getValue(emlib.MDL_RESOLUTION_FRC, objId)
            values.append((freq, fsc))
         
        return np.array(values)   
    
    def _readFscCutoff(self, cls: int, cutoff: float) -> np.ndarray:
        samplingRate = self._getSamplingRate()
        
        values = []
        for it in itertools.count():
            fscFn = self._getFscFilename(it, cls)
            if os.path.exists(fscFn):
                fscMd = emlib.MetaData(fscFn)
                values.append(self.protocol._computeResolution(
                    fscMd, 
                    samplingRate, 
                    cutoff
                ))
            else:
                break

        return np.array(values)
    
    # ---------------------------- SHOW functions ------------------------------
    def _showIterationFsc(self, e):
        fig, ax = plt.subplots()
        
        it = int(self.showIterationFsc)
        for cls in itertools.count():
            fscFn = self._getFscFilename(it, cls)
            if os.path.exists(fscFn):
                fsc = self._readFsc(fscFn)
                label = f'Class {cls}'
                ax.plot(fsc[:,0], fsc[:,1], label=label)
            else:
                break

        ax.set_title('Class FSC')
        ax.set_xlabel('Resolution (1/A)')
        ax.set_ylabel('FSC')
        ax.legend()
               
        return [fig]
    
    def _showClassFsc(self, e):
        fig, ax = plt.subplots()
        
        cls = int(self.showClassFsc)
        for it in itertools.count():
            fscFn = self._getFscFilename(it, cls)
            if os.path.exists(fscFn):
                fsc = self._readFsc(fscFn)
                label = f'Iteration {it}'
                ax.plot(fsc[:,0], fsc[:,1], label=label)
            else:
                break

        ax.set_title('Class FSC')
        ax.set_xlabel('Resolution (1/A)')
        ax.set_ylabel('FSC')
        ax.legend()
               
        return [fig]
    
    def _showFscCutoff(self, e):
        fig, ax = plt.subplots()
        
        cutoff = float(self.showFscCutoff)
        for cls in range(self._getClassCount()):
            y = self._readFscCutoff(cls, cutoff)
            x = np.arange(len(y))
            label = f'Class {cls}'
            ax.plot(x, y, label=label)

        ax.set_title('Resolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Resolution (A)')
        ax.legend()
               
        return [fig]