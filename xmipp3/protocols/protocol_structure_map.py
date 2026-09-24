# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
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


import numpy as np
import os

import xmippLib

from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfVolumes
from pyworkflow import VERSION_2_0
from pyworkflow.utils import cleanPath
from pyworkflow.utils.path import cleanPattern

from ..protocols.protocol_structure_map_zernike3d import mds
from pyworkflow import BETA, UPDATED, NEW, PROD


class XmippProtStructureMap(ProtAnalysis3D):
    """ Performs structure mapping based on correlation distance between
    volumes. This protocol helps identify similarities and differences among
    multiple structures by quantifying their spatial relationships.

    AI Generated

    ## Overview

    The Struct Map protocol compares several 3D volumes and represents their
    relationships in a low-dimensional structural map.

    When several volumes are available, for example from 3D classification,
    different reconstructions, or alternative processing branches, it is often
    useful to quantify how similar or different they are. This protocol computes a
    pairwise correlation-based distance between all input volumes. It then converts
    the resulting distance matrix into coordinate maps using multidimensional
    scaling.

    The output files allow the user to visualize the relative arrangement of the
    volumes in one, two, or three dimensions. Volumes that are close in the
    structural map are more similar by correlation distance. Volumes that are far
    apart are more different.

    ## Inputs and General Workflow

    The input is a set of volumes or a set of 3D classes.

    If the input is a set of 3D classes, the protocol uses the representative
    volume of each class. If the input is a set of volumes, each volume is used
    directly.

    The protocol first rescales all volumes to a common working sampling rate
    defined by the target resolution. It also crops them to a common box size so
    that pairwise comparisons are technically consistent.

    Then, for each ordered pair of different volumes, one volume is locally aligned
    to the other and a correlation value is computed. The protocol converts this
    correlation into a distance using:

    \[
    \text{distance} = 1 - \text{correlation}
    \]

    These distances are assembled into a correlation-distance matrix. Finally, the
    protocol applies multidimensional scaling to generate 1D, 2D, and 3D coordinate
    representations.

    ## Input Volume(s)

    The **Input volume(s)** parameter accepts either:

    - a **SetOfVolumes**;
    - a **SetOfClasses3D**.

    For a SetOfVolumes, each volume is compared with all the others.

    For a SetOfClasses3D, the representative volume of each class is used. The
    protocol also records class weights corresponding to the number of particles in
    each class.

    The volumes should represent related structures. Comparing unrelated maps is
    technically possible, but the resulting structural map may be difficult to
    interpret biologically.

    ## Target Resolution

    The **Target resolution** parameter defines the resolution used to prepare the
    volumes for comparison.

    The protocol rescales the input volumes so that the target resolution
    corresponds approximately to two thirds of the Fourier spectrum. This reduces
    the influence of high-frequency noise and focuses the comparison on the
    structural scale selected by the user.

    The default value is 8 Å, which is appropriate for comparing global shapes or
    medium-resolution structural differences.

    A lower numerical target resolution includes finer detail but may make the
    comparison more sensitive to noise, local artifacts, or alignment errors. A
    higher numerical value focuses on coarser global differences.

    ## Volume Rescaling and Cropping

    Before comparing volumes, the protocol brings them to a common scale.

    It computes a working sampling rate from the target resolution and the sampling
    rates of the input volumes. Volumes are resized when necessary and then cropped
    to the smallest box size among the input volumes.

    This ensures that all volumes can be compared on compatible grids.

    Users should still make sure that the volumes are biologically comparable and
    that important density is not lost when volumes are cropped to the smallest
    box size.

    ## Pairwise Local Alignment

    For each pair of different volumes, the protocol locally aligns one volume to
    the other before computing the correlation distance.

    This step reduces the effect of small residual translations or rotations
    between volumes. It is important because a poor relative alignment would make
    two similar maps appear artificially different.

    The alignment is local. Therefore, the volumes should already be roughly in the
    same orientation and coordinate frame. The protocol is not intended to solve
    completely arbitrary relationships between unrelated volumes.

    ## Correlation Distance

    After local alignment, the protocol computes the correlation between the two
    volumes.

    The distance used for the structural map is:

    \[
    1 - \text{correlation}
    \]

    A smaller distance means stronger similarity. A larger distance means weaker
    similarity.

    Because the protocol computes ordered pair comparisons, the distance matrix may
    reflect the alignment direction used for each pair. Users should interpret the
    matrix as a practical correlation-distance summary rather than as a perfect
    mathematical metric.

    ## Correlation Matrix

    The protocol writes the pairwise distance matrix to:

    `CorrMatrix.txt`

    This matrix contains the distance between every pair of input volumes. The
    diagonal entries are zero because each volume is identical to itself in the
    comparison matrix.

    This file is useful for quantitative inspection, clustering, plotting, or
    external analysis.

    ## Structural Coordinate Maps

    After computing the distance matrix, the protocol applies multidimensional
    scaling to obtain low-dimensional coordinates.

    It writes three coordinate files:

    - `CoordinateMatrixCorr1.txt`;
    - `CoordinateMatrixCorr2.txt`;
    - `CoordinateMatrixCorr3.txt`.

    These files contain 1D, 2D, and 3D embeddings of the input volumes based on
    their pairwise correlation distances.

    The 2D or 3D coordinate maps are usually the most useful for visualization.
    Nearby points correspond to similar volumes, while separated points correspond
    to more different volumes.

    ## Weights File

    The protocol writes a file named:

    `weights.txt`

    For input 3D classes, the weights correspond to the number of particles in each
    class. For input volumes, the weights are set to 1.

    These weights can be useful when plotting the structural map. For example,
    class points can be drawn with sizes proportional to the number of particles
    they represent.

    ## Interpretation of the Structural Map

    The structural map should be interpreted as a relative similarity map of the
    input volumes.

    Clusters of nearby volumes suggest similar structures or similar class
    representatives. Gradual arrangements may suggest continuous variability or
    progressive structural changes. Isolated points may indicate distinct classes,
    outliers, artifacts, or volumes that differ strongly from the rest.

    The map is based on correlation distance after local alignment and rescaling.
    It does not directly identify the biological cause of differences. Differences
    may arise from conformational variability, compositional heterogeneity,
    resolution differences, noise, masking, reconstruction artifacts, or alignment
    issues.

    ## Practical Recommendations

    Use this protocol when you have several related volumes and want to summarize
    their structural relationships.

    For 3D classification results, use the class representatives as input and
    inspect the weights file together with the coordinate map.

    Start with a target resolution around 8 Å when comparing global or
    medium-resolution structural differences.

    Use a lower target-resolution value only when the volumes are reliable enough
    to compare finer details.

    Make sure volumes are approximately aligned and represent comparable regions
    before running the protocol.

    Inspect the correlation matrix as well as the coordinate maps. A low-dimensional
    embedding is a simplification of the full pairwise distance matrix.

    Be cautious if one input volume has a much smaller box size than the others,
    because all volumes are cropped to the smallest dimension.

    ## Final Perspective

    Struct Map is a comparative volume-analysis protocol.

    For biological users, its main value is that it converts a collection of 3D
    maps or 3D class representatives into a quantitative structural relationship
    map. This can help interpret 3D classification results, compare alternative
    reconstructions, identify outlier volumes, or visualize structural variability
    among maps.

    The protocol should be used as an exploratory analysis tool. The structural map
    suggests relationships among volumes, but biological interpretation requires
    inspection of the original maps, class sizes, reconstruction quality, and the
    processing context.
    """
    _label = 'struct map'
    _lastUpdateVersion = VERSION_2_0
    OUTPUT_SUFFIX = '_%d_crop.vol'
    _devStatus = UPDATED

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfClasses3D, SetOfVolumes',
                      label="Input volume(s)", important=True,
                      help='Select one or more volumes (SetOfClasses3D)\n'
                           'for structure mapping.')
        form.addParam('targetResolution', params.FloatParam, label="Target resolution",
                      default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParallelSection(threads=1, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        volList = []
        dimList = []
        srList = []
        volList, dimList, srList, _ = self._iterInputVolumes(volList, dimList, srList, [])
        nVoli = 1
        depsConvert = []
        for _ in volList:
            convert = self._insertFunctionStep('convertStep', volList[nVoli - 1],
                                     dimList[nVoli - 1], srList[nVoli - 1],
                                     min(dimList), max(srList), nVoli, prerequisites=[])
            depsConvert.append(convert)
            nVoli += 1

        nVoli = 1
        deps = []
        for _ in volList:
            nVolj = 1
            for _ in volList:
                if nVolj != nVoli:
                    stepID = self._insertFunctionStep('alignStep', volList[nVoli - 1],
                                            volList[nVolj - 1], nVoli - 1,
                                            nVolj - 1, prerequisites=depsConvert)
                    deps.append(stepID)
                nVolj += 1
            nVoli += 1

        self._insertFunctionStep('correlationMatrix', volList, prerequisites=deps)
        self._insertFunctionStep('gatherResultsStepCorr')
        self._insertFunctionStep('cleanIntermediate')

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self, volFn, volDim, volSr, minDim, maxSr, nVoli):
        Xdim = volDim
        Ts = volSr
        newTs = self.targetResolution.get() * 1.0 / 3.0
        newTs = max(maxSr, newTs)
        newXdim = Xdim * Ts / newTs
        fnOut = os.path.splitext(volFn)[0]
        fnOut = self._getExtraPath(os.path.basename(fnOut + self.OUTPUT_SUFFIX % nVoli))

        ih = ImageHandler()
        if Xdim != newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s -o %s --dim %d" % (volFn, fnOut, newXdim))

        else:
            ih.convert(volFn, fnOut)

        if newXdim>minDim:
            self.runJob("xmipp_transform_window", " -i %s -o %s --crop %d" %
                        (fnOut, fnOut, (newXdim - minDim)))

    def alignStep(self, inputVolFn, refVolFn, i, j):
        inputVolFn = self._getExtraPath(os.path.basename(os.path.splitext(inputVolFn)[0]
                                                         + self.OUTPUT_SUFFIX % (i + 1)))
        refVolFn = self._getExtraPath(os.path.basename(os.path.splitext(refVolFn)[0]
                                                       + self.OUTPUT_SUFFIX % (j + 1)))
        fnOut = self._getTmpPath('vol%dAlignedTo%d.vol' % (i, j))

        params = ' --i1 %s --i2 %s --apply %s --local --dontScale' % \
                 (refVolFn, inputVolFn, fnOut)

        self.runJob("xmipp_volume_align", params)

        self.computeCorr(fnOut, refVolFn, i, j)

        cleanPath(fnOut)

    def computeCorr(self, vol1, vol2, i, j):
        vol = xmippLib.Image(vol1)
        defVol = xmippLib.Image(vol2)
        corr = vol.correlation(defVol)
        corr = 1 - corr
        outFile = self._getExtraPath('Pair_%d_%d_correlation.txt' % (i,j))
        with open(outFile, 'w') as f:
            f.write('%f' % corr)

    def correlationMatrix(self, volList):
        numVol = len(volList)
        self.corrMatrix = np.zeros((numVol, numVol))
        for i in range(numVol):
            for j in range(numVol):
                if i != j:
                    path = self._getExtraPath('Pair_%d_%d_correlation.txt' % (i,j))
                    self.corrMatrix[i,j] = np.loadtxt(path)

    def gatherResultsStepCorr(self):
        fnRoot = self._getExtraPath("CorrMatrix.txt")
        self.saveCorrelation(self.corrMatrix, fnRoot)

    def saveCorrelation(self, matrix, fnRoot, label=''):
        np.savetxt(fnRoot, matrix, "%f")
        corr = np.asarray(matrix)
        for i in range(1, 4):
            embed = mds(corr, i)
            embedExtended = np.pad(embed, ((0, 0), (0, i - embed.shape[1])),
                                   "constant", constant_values=0)
            np.savetxt(self._defineResultsName(i, label), embedExtended)

    def cleanIntermediate(self):
        cleanPattern(self._getExtraPath("Pair*correlation.txt"))
        cleanPattern(self._getExtraPath("*_crop.vol"))

    # --------------------------- UTILS functions --------------------------------------------
    def _iterInputVolumes(self, volList, dimList, srList, idList):
        """ Iterate over all the input volumes. """
        count = 1
        weight = []
        if not isinstance(self.inputVolumes.get(), SetOfVolumes):
            for cls in self.inputVolumes.get().iterItems():
                vol = cls.getRepresentative()
                volList.append(vol.getFileName())
                dimList.append(vol.getDim()[0])
                srList.append(vol.getSamplingRate())
                idList.append(count)
                weight.append(cls.getSize())
                count += 1
        else:
            for vol in self.inputVolumes.get().iterItems():
                volList.append(vol.getFileName())
                dimList.append(vol.getDim()[0])
                srList.append(vol.getSamplingRate())
                idList.append(count)
                weight.append(1)
                count += 1
        np.savetxt(self._getExtraPath('weights.txt'), np.asarray(weight))
        return volList, dimList, srList, idList

    def split(self, array, nrows, ncols):
        """Split a matrix into sub-matrices."""
        r, h = array.shape
        return (array.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    def _defineResultsName(self, i, label=''):
        return self._getExtraPath('Coordinate%sMatrixCorr%d.txt' % (label,i))
