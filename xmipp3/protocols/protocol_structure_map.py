
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
import glob
import os
import re

import xmippLib

from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfVolumes, Volume
from pyworkflow import VERSION_2_0
from pyworkflow.utils import cleanPattern


def mds(d, dimensions=2):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """

    # Distance matrix size
    (n, n) = d.shape

    # Centering Matrix
    J = np.identity(n, float) - (1 / n) * np.ones([n, n], float)

    # Center distance matrix
    B = -0.5 * J @ d @ J

    # Singular value decomposition
    [U, S, V] = np.linalg.svd(B)
    S = np.diag(S)

    # Coordinates matrix from MDS
    Y = U[:, 0:dimensions] @ np.power(S[0:dimensions, 0:dimensions], 0.5)

    return Y


class XmippProtStructureMap(ProtAnalysis3D):
    """ Protocol for structure mapping based on correlation distance. """
    _label = 'struct map'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfClasses3D',
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
        for voli in volList:
            convert = self._insertFunctionStep('convertStep', volList[nVoli - 1],
                                     dimList[nVoli - 1], srList[nVoli - 1],
                                     min(dimList), max(srList), nVoli, prerequisites=[])
            depsConvert.append(convert)
            nVoli += 1

        nVoli = 1
        deps = []
        for voli in volList:
            nVolj = 1
            for volj in volList:
                if nVolj != nVoli:
                    stepID = self._insertFunctionStep('alignStep', volList[nVoli - 1],
                                            volList[nVolj - 1], nVoli - 1,
                                            nVolj - 1, prerequisites=depsConvert)
                    deps.append(stepID)
                nVolj += 1
            nVoli += 1

        self._insertFunctionStep('computeCorr', volList)

        self._insertFunctionStep('gatherResultsStepCorr')

        cleanPattern(self._getExtraPath('*.vol'))

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self, volFn, volDim, volSr, minDim, maxSr, nVoli):
        Xdim = volDim
        Ts = volSr
        newTs = self.targetResolution.get() * 1.0 / 3.0
        newTs = max(maxSr, newTs)
        newXdim = Xdim * Ts / newTs
        fnOut = os.path.splitext(volFn)[0]
        fnOut = self._getExtraPath(os.path.basename(fnOut + '_%d_crop.vol' % nVoli))

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
        inputVolFn = self._getExtraPath(os.path.basename(os.path.splitext(inputVolFn)[0] + '_%d_crop.vol' % (i + 1)))
        refVolFn = self._getExtraPath(os.path.basename(os.path.splitext(refVolFn)[0] + '_%d_crop.vol' % (j + 1)))
        fnOut = self._getExtraPath('vol%dAlignedTo%d.vol' % (i, j))

        params = ' --i1 %s --i2 %s --apply %s --local --dontScale' % \
                 (refVolFn, inputVolFn, fnOut)

        self.runJob("xmipp_volume_align", params)

    def computeCorr(self, volList):
        ind = 0
        numVol = len(volList)
        self.corrMatrix = np.zeros((numVol, numVol))
        self.corrMatrix[numVol-1][numVol-1] = 0.0

        for item in volList:
            vol = xmippLib.Image(item)
            self.corrMatrix[ind][ind] = 0.0
            path = self._getExtraPath("*AlignedTo%d.vol" % ind)
            for fileVol in glob.glob(path):
                matches = re.findall("(\d+)", fileVol)
                ind2 = int(matches[1])
                defVol = xmippLib.Image(fileVol)
                corr = vol.correlation(defVol)
                self.corrMatrix[ind2][ind] = 1-corr
            ind += 1

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

    # --------------------------- UTILS functions --------------------------------------------
    def _iterInputVolumes(self, volList, dimList, srList, idList):
        """ Iterate over all the input volumes. """
        count = 1
        weight = []
        for cls in self.inputVolumes.get().iterItems():
            vol = cls.get()
            volList.append(vol.getFileName())
            dimList.append(vol.getDim()[0])
            srList.append(vol.getSamplingRate())
            idList.append(count)
            weight.append(cls.getSize())
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
