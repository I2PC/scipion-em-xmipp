
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
from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfVolumes, Volume
from pyworkflow import VERSION_2_0
import numpy as np


def mds(d, dimensions=2):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """

    (n, n) = d.shape
    E = (-0.5 * d ** 2)

    # Use mat to get column and row means to act as column and row means.
    Er = np.mat(np.mean(E, 1))
    Es = np.mat(np.mean(E, 0))

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = np.array(E - np.transpose(Er) - Es + np.mean(E))

    [U, S, V] = np.linalg.svd(F)

    Y = U * np.sqrt(S)

    return (Y[:, 0:dimensions], S)


class XmippProtStructureMapSPH(ProtAnalysis3D):
    """ Protocol for structure mapping based on spherical harmonics. """
    _label = 'sph struct map'
    _lastUpdateVersion = VERSION_2_0

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
        form.addParam('depth', params.IntParam, default=3,
                      label='Harmonical depth',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Harmonical depth of the deformation=1,2,3,...')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):

        volList, dimList, srList = self._iterInputVolumes()
        self.distanceMatrix = np.zeros((len(volList), len(volList)))

        nVoli = 1
        for voli in volList:
            self._insertFunctionStep('convertStep', volList[nVoli - 1],
                                     dimList[nVoli - 1], srList[nVoli - 1],
                                     min(dimList), max(srList))
            nVolj = 1
            for volj in volList:
                if nVolj != nVoli:
                    self._insertFunctionStep('deformStep', volList[nVoli - 1],
                                             volList[nVolj - 1], nVoli - 1,
                                             nVolj - 1)
                else:
                    self.distanceMatrix[nVoli - 1][nVolj - 1] = 0.0
                nVolj += 1
            nVoli += 1

        self._insertFunctionStep('gatherResultsStep', volList)

    # --------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self, volFn, volDim, volSr, minDim, maxSr):
        Xdim = volDim
        Ts = volSr
        newTs = self.targetResolution.get() * 1.0 / 3.0
        newTs = max(maxSr, newTs)
        newXdim = int(Xdim * Ts / newTs)

        ih = ImageHandler()
        if Xdim != newXdim:
            self.runJob("xmipp_image_resize",
                        "-i %s --dim %d" % (volFn, newXdim))

        if newXdim>minDim:
            self.runJob("xmipp_transform_window", " -i %s --crop %d" %
                        (volFn, (newXdim - minDim)))

    def deformStep(self, inputVolFn, refVolFn, i, j):
        fnOut = self._getExtraPath('vol%dDeformedTo%d.vol' % (i, j))

        params = ' --i1 %s --i2 %s --apply %s --least_squares --local ' % \
                 (refVolFn, inputVolFn, fnOut)
        self.runJob("xmipp_volume_align", params)

        params = ' -i %s -r %s -o %s --depth %d ' %\
                 (fnOut, refVolFn, fnOut, self.depth.get())

        self.runJob("xmipp_volume_deform_sph", params)
        distanceValue = np.loadtxt('./deformation.txt')
        self.distanceMatrix[i][j] = distanceValue


    def gatherResultsStep(self, volList):
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
            embed, _ = mds(distance, i)
            embedExtended = np.pad(embed, ((0, 0), (0, i - embed.shape[1])),
                                   "constant", constant_values=0)
            np.savetxt(self._defineResultsName(i), embedExtended)

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




