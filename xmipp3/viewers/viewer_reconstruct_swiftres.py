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

from typing import Tuple, Iterator
import os
import itertools
import collections
import numpy as np
import matplotlib.collections
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

        form.addSection(label='Classification')
        form.addParam('showClassMigration', LabelParam, label='Display class migration diagram',
                      default=0.143)
        
    def _getVisualizeDict(self):
        return {
            'showIterationFsc': self._showIterationFsc,
            'showClassFsc': self._showClassFsc,
            'showFscCutoff': self._showFscCutoff,
            
            'showClassMigration': self._showClassMigration
        }
    

    # --------------------------- UTILS functions ------------------------------
    def _getIterationCount(self) -> int:
        return self.protocol._getIterationCount()
    
    def _getClassCount(self) -> int:
        return self.protocol._getClassCount()

    def _getSamplingRate(self) -> float:
        return self.protocol._getSamplingRate()
    
    def _getAlignmentMdFilename(self, iteration: int):
        return self.protocol._getAlignmentMdFilename(iteration)
    
    def _getFscFilename(self, iteration: int, cls: int):
        return self.protocol._getFscFilename(iteration, cls)
    
    def _readFsc(self, fscMd: emlib.MetaData) -> np.ndarray:
        values = []
        for objId in fscMd:
            freq = fscMd.getValue(emlib.MDL_RESOLUTION_FREQ, objId)
            fsc = fscMd.getValue(emlib.MDL_RESOLUTION_FRC, objId)
            values.append((freq, fsc))
         
        return np.array(values)   
    
    def _iterAlignmentMdFilenames(self):
        for it in itertools.count():
            alignmentFn = self._getAlignmentMdFilename(it)
            if os.path.exists(alignmentFn):
                yield alignmentFn
            
            else:
                break
    
    def _iterFscMdIterationFilenames(self, it: int):
        for cls in itertools.count():
            fscFn = self._getFscFilename(it, cls)
            if os.path.exists(fscFn):
                yield fscFn
            
            else:
                break

    def _iterFscMdClassFilenames(self, cls: int):
        for it in itertools.count():
            fscFn = self._getFscFilename(it, cls)
            if os.path.exists(fscFn):
                yield fscFn
            
            else:
                break
        
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
    
    def _computeClassMigrations(self, srcMd: emlib.MetaData, dstMd: emlib.MetaData) -> dict:
        result = {}

        for objId in srcMd:
            srcClassId = srcMd.getValue(emlib.MDL_REF3D, objId)
            dstClassId = dstMd.getValue(emlib.MDL_REF3D, objId)
            
            key = (srcClassId, dstClassId)
            if key in result:
                result[key] += 1
            else:
                result[key] = 1

        return result

    def _computeIterationClassMigrationSegments(self, migrations: dict) -> Tuple[np.ndarray, np.ndarray]:
        segments = np.empty((len(migrations), 2))
        counts = np.array(list(migrations.values()))

        # Write the y values form migrations
        for i, (srcClass, dstClass) in enumerate(migrations.keys()):
            segments[i,:] = (srcClass, dstClass)
            
        return segments, counts

    def _computeClassPoints(self, alignmentMd: emlib.MetaData) -> np.ndarray:
        class3d = alignmentMd.getColumnValues(emlib.MDL_REF3D)
        freq = collections.Counter(class3d)
        return np.array(list(freq.items()))
    
    def _computeClassMigrationElements(self, alignmentFilenames: Iterator[str]) -> Tuple[np.ndarray, 
                                                                                         np.ndarray,
                                                                                         np.ndarray,
                                                                                         np.ndarray]:
        segments = []
        points = []
        counts = []
        sizes = []
        
        srcAlignmentFn = next(alignmentFilenames)
        srcAlignmentMd = emlib.MetaData(srcAlignmentFn)
        iterationPoints, iterationSizes = self._computeClassFrequencies(srcAlignmentMd)
        counts.append(iterationCounts)
        sizes.append(iterationSizes)
        for iteration, dstAlignmentFn in enumerate(alignmentFilenames, start=1):
            dstAlignmentMd = emlib.MetaData(dstAlignmentFn)
            
            srcAlignmentMd.intersection(dstAlignmentMd, emlib.MDL_ITEM_ID)
            migrations = self._computeClassMigrations(srcAlignmentMd, dstAlignmentMd)
            iterationSegments, iterationCounts = self._computeIterationClassMigrationSegments(iteration, migrations)
            iterationPoints, iterationSizes = self._computeClassFrequencies(dstAlignmentMd)
            
            segments.append(iterationSegments)
            points.append(iterationPoints)
            counts.append(iterationCounts)
            sizes.append(iterationSizes)
            
            # Save for next
            srcAlignmentMd = dstAlignmentMd
            
        return np.concatenate(segments), np.concatenate(points), np.concatenate(counts), np.concatenate(sizes)

    # ---------------------------- SHOW functions ------------------------------
    def _showIterationFsc(self, e):
        fig, ax = plt.subplots()
        
        it = int(self.showIterationFsc)
        for cls, fscFn in enumerate(self._iterFscMdIterationFilenames(it), start=1):
            fscMd = emlib.MetaData(fscFn)
            fsc = self._readFsc(fscMd)
            label = f'Class {cls}'
            ax.plot(fsc[:,0], fsc[:,1], label=label)

        ax.set_title('Class FSC')
        ax.set_xlabel('Resolution (1/A)')
        ax.set_ylabel('FSC')
        ax.legend()
               
        return [fig]
    
    def _showClassFsc(self, e):
        fig, ax = plt.subplots()
        
        cls = int(self.showClassFsc)
        for it, fscFn in enumerate(self._iterFscMdClassFilenames(cls), start=1):
            fscMd = emlib.MetaData(fscFn)
            fsc = self._readFsc(fscMd)
            label = f'Iteration {it}'
            ax.plot(fsc[:,0], fsc[:,1], label=label)

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
        ax.set_ylim(bottom=0)
        ax.legend()
               
        return [fig]
    
    def _showClassMigration(self, e):
        fig, ax = plt.subplots()
        
        it = self._iterAlignmentMdFilenames()
        segments, counts = self._computeClassMigrationSegments(it)
        lines = matplotlib.collections.LineCollection(segments, array=counts)

        it = self._iterAlignmentMdFilenames()
        points, sizes = self._computeClassSizePoints(it)
                
        ax.add_collection(lines)
        sc = ax.scatter(points[:,0], points[:,1], c=sizes)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Class')
        fig.colorbar(sc, ax=ax)
        
        return [fig]
        