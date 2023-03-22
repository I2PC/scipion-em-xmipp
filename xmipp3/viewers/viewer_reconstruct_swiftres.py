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

from typing import Tuple, Iterator, Iterable
import os
import itertools
import collections
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
from pyworkflow.protocol.params import LabelParam, IntParam, FloatParam

from pwem import emlib
from pwem.viewers.views import DataView, ObjectView
from pwem.viewers import EmPlotter

import xmippLib

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
        form.addSection(label='Noise model')
        form.addParam('showNoiseModel', LabelParam, label='Display radial noise model profile')
        form.addParam('showWeights', LabelParam, label='Display radial weight profile')
        
        form.addSection(label='Classification')
        form.addParam('showClassMigration', LabelParam, label='Display class migration diagram')
        form.addParam('showClassSizes', LabelParam, label='Display class sizes')
        
        form.addSection(label='Angular difference from previous iteration')
        form.addParam('showAngleDiffMetadata', IntParam, label='Display iteration angular difference metadata',
                      default=0)
        form.addParam('showAngleDiffVecDiffHist', LabelParam, label='Display vector difference histogram')
        form.addParam('showAngleDiffShiftDiffHist', LabelParam, label='Display shift difference histogram')

        form.addSection(label='Reconstruction')
        form.addParam('showIterationAngularDistribution', IntParam, label='Display iteration angular distribution',
                      default=0)
        form.addParam('showIterationAngularDistribution3d', IntParam, label='Display iteration angular distribution 3D',
                      default=0)
        form.addParam('showIterationShiftHistogram', IntParam, label='Display iteration shift histogram',
                      default=0)
        form.addParam('showIterationVolume', IntParam, label='Display iteration volume',
                      default=0)
        
        form.addSection(label='FSC')
        form.addParam('showIterationFsc', IntParam, label='Display iteration FSC',
                      default=0 )
        form.addParam('showClassFsc', IntParam, label='Display class FSC',
                      default=0)
        form.addParam('showFscCutoff', FloatParam, label='Display FSC cutoff',
                      default=0.143)

        
    def _getVisualizeDict(self):
        return {
            'showNoiseModel': self._showNoiseModel,
            'showWeights': self._showWeights,
            
            'showClassMigration': self._showClassMigration,
            'showClassSizes': self._showClassSizes,

            'showAngleDiffMetadata': self._showAngleDiffMetadata,
            'showAngleDiffVecDiffHist': self._showAngleDiffVecDiffHistogram,
            'showAngleDiffShiftDiffHist': self._showAngleDiffShiftDiffHistogram,

            'showIterationAngularDistribution': self._showIterationAngularDistribution,
            'showIterationAngularDistribution3d': self._showIterationAngularDistribution3d,
            'showIterationShiftHistogram': self._showIterationShiftHistogram,
            'showIterationVolume': self._showIterationVolume,

            'showIterationFsc': self._showIterationFsc,
            'showClassFsc': self._showClassFsc,
            'showFscCutoff': self._showFscCutoff,
            
        }
    

    # --------------------------- UTILS functions ------------------------------
    def _getIterationParametersFilename(self, iteration: int):
        return self.protocol._getIterationParametersFilename(iteration)
            
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
    
    def _getFilteredReconstructionFilename(self, iteration: int, cls: int):
        return self.protocol._getFilteredReconstructionFilename(iteration, cls)
    
    def _getAngleDiffMdFilename(self, iteration: int):
        return self.protocol._getAngleDiffMdFilename(iteration)

    def _getAngleDiffVecDiffHistogramMdFilename(self, iteration: int):
        return self.protocol._getAngleDiffVecDiffHistogramMdFilename(iteration)

    def _getAngleDiffShiftDiffHistogramMdFilename(self, iteration: int):
        return self.protocol._getAngleDiffShiftDiffHistogramMdFilename(iteration)
    
    def _getNoiseModelFilename(self, iteration: int):
        return self.protocol._getNoiseModelFilename(iteration)
    
    def _getWeightsFilename(self, iteration: int):
        return self.protocol._getWeightsFilename(iteration)
    
    def _readFsc(self, fscMd: emlib.MetaData) -> np.ndarray:
        values = []
        for objId in fscMd:
            freq = fscMd.getValue(emlib.MDL_RESOLUTION_FREQ, objId)
            fsc = fscMd.getValue(emlib.MDL_RESOLUTION_FRC, objId)
            values.append((freq, fsc))
         
        return np.array(values)   
    
    def _iterFilenamesUntilNotExist(self, filenames: Iterable[str]):
        for filename in filenames:
            if os.path.exists(filename):
                yield filename
            
            else:
                break
            
    def _iterMetadatas(self, filenames: Iterable[str]) -> emlib.MetaData:
        md = emlib.MetaData()
        for fn in filenames:
            md.read(fn)
            yield md
    
    def _iterAlignmentMdFilenames(self):
        return self._iterFilenamesUntilNotExist(
            map(self._getAlignmentMdFilename, itertools.count())
        )

    def _iterAlignmentMds(self):
        return self._iterMetadatas(self._iterAlignmentMdFilenames())

    def _iterFscMdIterationFilenames(self, it: int):
        return self._iterFilenamesUntilNotExist(
            map(lambda cls : self._getFscFilename(it, cls), itertools.count())
        )

    def _iterFscMdClassFilenames(self, cls: int):
        return self._iterFilenamesUntilNotExist(
            map(lambda it : self._getFscFilename(it, cls), itertools.count())
        )
    
    def _iterAngleDiffVecDiffHistMdFilenames(self):
        return self._iterFilenamesUntilNotExist(
            map(self._getAngleDiffVecDiffHistogramMdFilename, itertools.count())
        )
        
    def _iterAngleDiffShiftDiffHistMdFilenames(self):
        return self._iterFilenamesUntilNotExist(
            map(self._getAngleDiffShiftDiffHistogramMdFilename, itertools.count())
        )
        
    def _iterNoiseModelFilenames(self):
        return self._iterFilenamesUntilNotExist(
            map(self._getNoiseModelFilename, itertools.count())
        )
        
    def _iterWeightsFilenames(self):
        return self._iterFilenamesUntilNotExist(
            map(self._getWeightsFilename, itertools.count())
        )
        
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
    
    def _computeFscTickLabels(self, ticks: np.ndarray) -> np.ndarray:
        samplingRate = self._getSamplingRate()
        labels = list(map('1/{0:.2f}'.format, samplingRate/ticks))
        return labels

    def _computeIterationClassSizes(self, alignmentMd: emlib.MetaData, label: int = emlib.MDL_REF3D) -> dict:
        classIds = alignmentMd.getColumnValues(label)
        return collections.Counter(classIds)

    def _computeClassSizes(self, alignmentMds: Iterable[emlib.MetaData], label: int = emlib.MDL_REF3D) -> dict:
        result = dict()
        
        for it, alignmentMd in enumerate(alignmentMds):
            counts = self._computeIterationClassSizes(alignmentMd, label)

            # Process existing elements
            for id in result.keys():
                result[id].append(counts.get(id, 0))

            # Process new elements
            for id, count in counts.items():
                if id not in result:
                    result[id] = [0]*it + [count]
        
        return { classId: np.array(counts) for classId, counts in result.items() }

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
        for i, (srcClass, dstClass) in enumerate(migrations.items()):
            segments[i,:] = (srcClass, dstClass)
            
        return segments, counts
    
    def _computeClassMigrationElements(self, alignmentFilenames: Iterator[str]) -> Tuple[LineCollection, np.ndarray]:
        srcAlignmentFn = next(alignmentFilenames)
        srcAlignmentMd = emlib.MetaData(srcAlignmentFn)
        for iteration, dstAlignmentFn in enumerate(alignmentFilenames, start=1):
            dstAlignmentMd = emlib.MetaData(dstAlignmentFn)
            
            # Obtain common elements
            srcAlignmentMd.intersection(dstAlignmentMd, emlib.MDL_ITEM_ID)

            migrations = self._computeClassMigrations(srcAlignmentMd, dstAlignmentMd)
            sizes = self._computeIterationClassSizes(dstAlignmentMd)
            segments, counts = self._computeIterationClassMigrationSegments(migrations)
            
            # Save for next
            srcAlignmentMd = dstAlignmentMd
        
        #TODO
        
    def _readAngleDiffVecHistogram(self, filename):
        md = emlib.MetaData(filename)
        diff = md.getColumnValues(emlib.MDL_ANGLE_DIFF)
        count = md.getColumnValues(emlib.MDL_COUNT)

        return diff, count
        
    def _readAngleDiffShiftHistogram(self, filename):
        md = emlib.MetaData(filename)
        diff = md.getColumnValues(emlib.MDL_SHIFT_DIFF)
        count = md.getColumnValues(emlib.MDL_COUNT)

        return diff, count

    def _computePsdRadialProfile(self, psd: np.ndarray, n: int = 128) -> np.ndarray:
        result = np.empty(n)

        # Compute the frequency grid
        freqX = np.fft.rfftfreq(len(psd))
        freqY = np.fft.fftfreq(len(psd))
        freq2 = freqX**2 + (freqY**2)[None].T

        # Select by bands
        limits = np.linspace(0, 0.5, n+1)
        limits2 = limits**2
        for i in range(len(result)):
            low2, high2 = limits2[i], limits2[i+1]
            mask = np.logical_and(low2 <= freq2, freq2 <= high2)
            result[i] = np.average(psd[mask])

        return result

    # ---------------------------- SHOW functions ------------------------------
    def _showNoiseModel(self, e):
        fig, ax = plt.subplots()

        psd = xmippLib.Image()
        freq = np.linspace(0, 0.5, 128)[:-1]
        freq += freq[1] / 2
        for iteration, filename in enumerate(self._iterNoiseModelFilenames()):
            psd.read(filename)
            profile = self._computePsdRadialProfile(psd.getData(), len(freq))
            label = f'Iteration {iteration}'
            ax.plot(freq, profile, label=label)
            
        ax.set_title('Noise model')
        ax.set_xlabel('Resolution (1/A)')
        ax.set_ylabel('$\sigma^2$')
        ax.set_xticklabels(self._computeFscTickLabels(ax.get_xticks()))
        ax.legend()
        
        return [fig]
    
    def _showWeights(self, e):
        fig, ax = plt.subplots()

        psd = xmippLib.Image()
        freq = np.linspace(0, 0.5, 128)[:-1]
        freq += freq[1] / 2
        for iteration, filename in enumerate(self._iterWeightsFilenames()):
            psd.read(filename)
            profile = self._computePsdRadialProfile(psd.getData(), len(freq))
            label = f'Iteration {iteration}'
            ax.plot(freq, profile, label=label)
            
        ax.set_title('Weights')
        ax.set_xlabel('Resolution (1/A)')
        ax.set_ylabel('Weight')
        ax.set_xticklabels(self._computeFscTickLabels(ax.get_xticks()))
        ax.legend()
        
        return [fig]

    def _showClassMigration(self, e):
        fig, ax = plt.subplots()
        
        lines, points = self._computeClassMigrationElements()
        
        ax.add_collection(lines)
        sc = ax.scatter(points[:,0], points[:,1], c=points[:,2])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Class')
        fig.colorbar(sc, ax=ax)
        
        return [fig]
        
    def _showClassSizes(self, e):
        fig, ax = plt.subplots()
        
        sizes = self._computeClassSizes(self._iterAlignmentMds())
        bottom = None
        for label, counts in sizes.items():
            x = np.arange(len(counts))
            ax.bar(x=x, height=counts, bottom=bottom, label=label)
            
            if bottom is None:
                bottom = counts.copy()
            else:
                bottom += counts
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Class size')
        ax.legend()

        return [fig]
        
    def _showAngleDiffMetadata(self, e):
        mdFilename = self._getAngleDiffMdFilename(int(self.showAngleDiffMetadata))
        v = DataView(mdFilename)
        return [v]
    
    def _showAngleDiffVecDiffHistogram(self, e):
        fig, ax = plt.subplots()
        
        for it, fn in enumerate(self._iterAngleDiffVecDiffHistMdFilenames()):
            diff, count = self._readAngleDiffVecHistogram(fn)
            label = f'Iteration {it}'
            ax.plot(diff, count, label=label)

        ax.set_title('Angle difference histogram')
        ax.set_xlabel('Angle difference [deg]')
        ax.set_ylabel('Particle count')
        ax.legend()
               
        return [fig]
        
    def _showAngleDiffShiftDiffHistogram(self, e):
        fig, ax = plt.subplots()
        
        for it, fn in enumerate(self._iterAngleDiffShiftDiffHistMdFilenames()):
            diff, count = self._readAngleDiffShiftHistogram(fn)
            label = f'Iteration {it}'
            ax.plot(diff, count, label=label)

        ax.set_title('Shift difference histogram')
        ax.set_xlabel('Shift difference [px]')
        ax.set_ylabel('Particle count')
        ax.legend()
               
        return [fig]
        
    def _showIterationVolume(self, e):
        iteration = int(self.showIterationVolume)
        filename = self._getFilteredReconstructionFilename(iteration, 0)
        return DataView(filename)
    
    def _showIterationAngularDistribution(self, e):
        iteration = int(self.showIterationAngularDistribution)
        view = EmPlotter(x=1, y=1, mainTitle="Iteration %d" % iteration, windowTitle="Angular distribution")
        axis = view.plotAngularDistributionFromMd(self._getAlignmentMdFilename(iteration), '', histogram=True)
        view.getFigure().colorbar(axis)
        return [view]    

    def _showIterationAngularDistribution3d(self, e):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Read data
        iteration = int(self.showIterationAngularDistribution3d)
        md = emlib.MetaData(self._getAlignmentMdFilename(iteration))
        rot = np.deg2rad(md.getColumnValues(emlib.MDL_ANGLE_ROT))
        tilt = np.deg2rad(md.getColumnValues(emlib.MDL_ANGLE_TILT))
        
        # Make histogram
        N = 32
        heatmap, rotEdges, tiltEdges = np.histogram2d(
            x=rot, 
            y=tilt, 
            density=True,
            range=[[-np.pi, +np.pi], [0.0, np.pi]],
            bins=N
        )
        
        # Cartesian coordinates in unit sphere
        rotEdges, tiltEdges = np.meshgrid(rotEdges, tiltEdges)
        x = np.sin(tiltEdges) * np.cos(rotEdges)
        y = np.sin(tiltEdges) * np.sin(rotEdges)
        z = np.cos(tiltEdges)
        
        # Plot surface
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.plasma(heatmap))
        ax.set_title('Angular distribution')
        
        return [fig]    

    def _showIterationShiftHistogram(self, e):
        fig, ax = plt.subplots()
        
        iteration = int(self.showIterationShiftHistogram)
        md = emlib.MetaData(self._getAlignmentMdFilename(iteration))

        shiftX = md.getColumnValues(emlib.MDL_SHIFT_X)
        shiftY = md.getColumnValues(emlib.MDL_SHIFT_Y)
        
        N = 128
        heatmap, xEdges, yEdges = np.histogram2d(
            x=shiftX, 
            y=shiftY, 
            density=True,
            bins=N
        )

        image = ax.imshow(heatmap, extent=(xEdges[0], xEdges[-1], yEdges[0], yEdges[-1]))
        ax.set_title('Shift histogram')
        ax.set_xlabel('Shift X [px]')
        ax.set_ylabel('Shift Y [px]')
        fig.colorbar(image)
               
        return [fig]
        
    def _showIterationFsc(self, e):
        fig, ax = plt.subplots()
        
        it = int(self.showIterationFsc)
        for cls, fscFn in enumerate(self._iterFscMdIterationFilenames(it), start=1):
            fscMd = emlib.MetaData(fscFn)
            fsc = self._readFsc(fscMd)
            label = f'Class {cls}'
            ax.plot(fsc[:,0], fsc[:,1], label=label)
        ax.axhline(0.5, color='black', linestyle='--')
        ax.axhline(0.143, color='black', linestyle='--')

        ax.set_title('Class FSC')
        ax.set_xlabel('Resolution (1/A)')
        ax.set_ylabel('FSC')
        ax.set_xticklabels(self._computeFscTickLabels(ax.get_xticks()))
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
        ax.axhline(0.5, color='black', linestyle='--')
        ax.axhline(0.143, color='black', linestyle='--')

        ax.set_title('Class FSC')
        ax.set_xlabel('Resolution (1/A)')
        ax.set_ylabel('FSC')
        ax.set_xticklabels(self._computeFscTickLabels(ax.get_xticks()))
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

