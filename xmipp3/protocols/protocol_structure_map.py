
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
    """ Performs structure mapping based on correlation distance between volumes. This protocol helps identify similarities and differences among multiple structures by quantifying their spatial relationships."""
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
