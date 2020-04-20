
# **************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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
import math
import os
import ntpath

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


class XmippProtStructureMapSPH(ProtAnalysis3D):
    """ Protocol for structure mapping based on spherical harmonics. """
    _label = 'sph struct map'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.MultiPointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select one or more volumes (Volume or SetOfVolumes)\n'
                           'for structure mapping.')
        form.addParam('targetResolution', params.FloatParam, label="Target resolution",
                      default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('computeDef', params.BooleanParam, label="Compute deformation",
                      default=True,
                      help="Performed and structure mapping with/without deforming the input volumes")
        form.addParam('sigma', params.NumericListParam, label="Multiresolution", default="1 2",
                      help="Perform the analysys comparing different filtered versions of the volumes")
        form.addParam('Rmax', params.IntParam, default=0,
                      label='Sphere radius',
                      experLevel=params.LEVEL_ADVANCED,
                      help='Radius of the sphere where the spherical harmonics will be computed.')
        form.addParam('depth', params.IntParam, default=3,
                      label='Harmonical depth', condition='computeDef',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Harmonical depth of the deformation=1,2,3,...')
        form.addParallelSection(threads=1, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):

        volList, dimList, srList = self._iterInputVolumes()
        self.distanceMatrix = np.zeros((len(volList), len(volList)))

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
                    stepID = self._insertFunctionStep('deformStep', volList[nVoli - 1],
                                            volList[nVolj - 1], nVoli - 1,
                                            nVolj - 1, prerequisites=depsConvert)
                    deps.append(stepID)

                else:
                    stepID = self._insertFunctionStep("extraStep", nVoli, nVolj, prerequisites=depsConvert)
                    deps.append(stepID)
                nVolj += 1
            nVoli += 1

        if self.computeDef.get():
            self._insertFunctionStep('gatherResultsStepDef', volList, prerequisites=deps)

        self._insertFunctionStep('computeCorr', prerequisites=deps)

        self._insertFunctionStep('gatherResultsStepCorr', volList)

        cleanPattern(self._getExtraPath('*.vol'))


    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self, volFn, volDim, volSr, minDim, maxSr, nVoli):
        Xdim = volDim
        Ts = volSr
        newTs = self.targetResolution.get() * 1.0 / 3.0
        newTs = max(maxSr, newTs)
        newXdim = Xdim * Ts / newTs
        newRmax = self.Rmax.get() * Ts / newTs
        self.newRmax = min(newRmax, self.Rmax.get())
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

    def deformStep(self, inputVolFn, refVolFn, i, j):
        inputVolFn = self._getExtraPath(os.path.basename(os.path.splitext(inputVolFn)[0] + '_%d_crop.vol' % (i+1)))
        refVolFn = self._getExtraPath(os.path.basename(os.path.splitext(refVolFn)[0] + '_%d_crop.vol' % (j+1)))
        fnOut = self._getExtraPath('vol%dAlignedTo%d.vol' % (i, j))
        fnOut2 = self._getExtraPath('vol%dDeformedTo%d.vol' % (i, j))

        params = ' --i1 %s --i2 %s --apply %s --local --dontScale' % \
                 (refVolFn, inputVolFn, fnOut)

        self.runJob("xmipp_volume_align", params)

        if self.computeDef.get():
            params = ' -i %s -r %s -o %s --depth %d --sigma "%s"' %\
                     (fnOut, refVolFn, fnOut2, self.depth.get(), self.sigma.get())
            if self.newRmax != 0:
                params = params + ' --Rmax %d' % self.newRmax

            self.runJob("xmipp_volume_deform_sph", params)
            distanceValue = np.loadtxt('./deformation.txt')
            self.distanceMatrix[i][j] = distanceValue

    def extraStep(self, nVoli, nVolj):
        self.distanceMatrix[nVoli - 1][nVolj - 1] = 0.0


    def gatherResultsStepDef(self, volList):
        fnRoot = self._getExtraPath("DistanceMatrix.txt")
        nVoli = 1
        for i in volList:
            nVolj = 1
            for j in volList:
                fh = open(fnRoot, "a")
                fh.write("%f\t" % self.distanceMatrix[(nVoli - 1)][(nVolj - 1)])
                fh.close()
                nVolj += 1
            fh = open(fnRoot, "a")
            fh.write("\n")
            fh.close()
            nVoli += 1

        distance = np.asarray(self.distanceMatrix)
        for i in range(1, 4):
            embed = mds(distance, i)
            embedExtended = np.pad(embed, ((0, 0), (0, i - embed.shape[1])),
                                   "constant", constant_values=0)
            np.savetxt(self._defineResultsName(i), embedExtended)

    def computeCorr(self):
        ind = 0
        volList, _, _ = self._iterInputVolumes()
        self.corrMatrix = np.zeros((len(volList), len(volList)))
        self.corrMatrix[len(volList)-1][len(volList)-1] = 0.0

        import xmippLib

        volSet = self.inputVolumes

        for item in volList:
            vol = xmippLib.Image(item)
            self.corrMatrix[ind][ind] = 0.0
            #vol = ih.read(item.get().getFileName())
            #vol = vol.getData()
            #vol = np.asarray(vol)
            if self.computeDef.get():
                path = self._getExtraPath("*DeformedTo%d.vol" % ind)
            else:
                path = self._getExtraPath("*AlignedTo%d.vol" % ind)
            for fileVol in glob.glob(path):
                base = ntpath.basename(fileVol)
                import re
                matches = re.findall("(\d+)", fileVol)
                ind2 = int(matches[1])

                #defVol = ih.read(file)
                #defVol = defVol.getData()
                #defVol = np.asarray(defVol)
                defVol = xmippLib.Image(fileVol)
                #corr = np.multiply(vol,defVol)
                corr = vol.correlation(defVol)
                self.corrMatrix[ind2][ind] = 1-corr
            ind += 1

    def gatherResultsStepCorr(self, volList):
        fnRoot = self._getExtraPath("CorrMatrix.txt")
        nVoli = 1
        for i in volList:
            nVolj = 1
            for j in volList:
                fh = open(fnRoot, "a")
                fh.write("%f\t" % self.corrMatrix[(nVoli - 1)][(nVolj - 1)])
                fh.close()
                nVolj += 1
            fh = open(fnRoot, "a")
            fh.write("\n")
            fh.close()
            nVoli += 1

        corr = np.asarray(self.corrMatrix)
        for i in range(1, 4):
            embed = mds(corr, i)
            embedExtended = np.pad(embed, ((0, 0), (0, i - embed.shape[1])),
                                   "constant", constant_values=0)
            np.savetxt(self._defineResultsName2(i), embedExtended)


    # --------------------------- UTILS functions --------------------------------------------
    def _iterInputVolumes(self):
        """ Iterate over all the input volumes. """
        volList = []
        dimList = []
        srList = []
        for pointer in self.inputVolumes:
            item = pointer.get()
            if item is None:
                break
            itemId = item.getObjId()
            if isinstance(item, Volume):
                volList.append(item.getFileName())
                dimList.append(item.getDim()[0])
                srList.append(item.getSamplingRate())
            elif isinstance(item, SetOfVolumes):
                for vol in item:
                    volList.append(vol.getFileName())
                    dimList.append(vol.getDim()[0])
                    srList.append(vol.getSamplingRate())
        return volList, dimList, srList


    def _defineResultsName(self, i):
        return self._getExtraPath('CoordinateMatrix%d.txt' % i)

    def _defineResultsName2(self, i):
        return self._getExtraPath('CoordinateMatrixCorr%d.txt' % i)




