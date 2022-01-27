
# **************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
# *              David Herreros Calero (dherreros@cnb.csic.es)
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
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks
from scipy.stats import entropy
from scipy.ndimage.filters import gaussian_filter
import os

import xmippLib

from pwem.protocols import ProtAnalysis3D
import pyworkflow.protocol.params as params
from pwem.emlib.image import ImageHandler
from pwem.objects import SetOfVolumes, Volume
from pyworkflow import VERSION_2_0
from pyworkflow.utils import cleanPath, copyFile, getExt


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

def rigidRegistration(X, Y):
    Xcm = np.sum(X, axis=0) / X.shape[0]
    Ycm = np.sum(Y, axis=0) / Y.shape[0]
    Xc = np.transpose(X - Xcm)
    Yc = np.transpose(Y - Ycm)
    [U, S, V] = np.linalg.svd(Xc @ Yc.T)
    R = V @ U.T
    t = Ycm - R @ Xcm
    return R, t


class XmippProtStructureMapZernike3D(ProtAnalysis3D):
    """ Protocol for structure mapping based on Zernike3D. """
    _label = 'struct map - Zernike3D'
    _lastUpdateVersion = VERSION_2_0
    OUTPUT_SUFFIX = '_%d_crop.mrc'
    ALIGNED_VOL = 'vol%dAligned.mrc'

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_PARALLEL

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                           Select the one you want to use.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")
        form.addParam('inputVolumes', params.MultiPointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select one or more volumes (Volume or SetOfVolumes)\n'
                           'for structure mapping.')
        form.addParam('twoSets', params.BooleanParam, label='Compare two sets?', default=False,
                      help='Useful when two Sets are intended to be compared independently (e.g. '
                           'comparing EMDBS and Maps coming from PDBs).')
        form.addParam('secondSet', params.MultiPointerParam, pointerClass='SetOfVolumes,Volume',
                      condition='twoSets==True', label='Second set of volumes', allowsNull=True,
                      help='Select one or more volumes (Volume or SetOfVolumes)\n'
                           'to compare to the first set.')
        form.addParam('targetResolution', params.FloatParam, label="Target resolution",
                      default=8.0,
                      help="In Angstroms, the images and the volume are rescaled so that this resolution is at "
                           "2/3 of the Fourier spectrum.")
        form.addParam('sigma', params.NumericListParam, label="Multiresolution", default="1 2",
                      help="Perform the analysis comparing different filtered versions of the volumes. The values "
                           "specified here will determine the cutoff frequency of the filter in normalized units "
                           "(normalized to 1/2).")
        form.addParam('Rmax', params.IntParam, default=0,
                      label='Sphere radius',
                      experLevel=params.LEVEL_ADVANCED,
                      help='Radius of the sphere where the spherical harmonics will be computed (in voxels).')
        form.addParam('l1', params.IntParam, default=3,
                      label='Zernike Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Zernike Polynomials of the deformation=1,2,3,...')
        form.addParam('l2', params.IntParam, default=2,
                      label='Harmonical Degree',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Degree Spherical Harmonics of the deformation=1,2,3,...')
        form.addParam('penalization', params.FloatParam, default=0.00025, label='Regularization',
                      expertLevel=params.LEVEL_ADVANCED,
                      help='Penalization to deformations (higher values penalize more the deformation).')
        form.addParallelSection(threads=1, mpi=1)

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        volList = []
        dimList = []
        srList = []
        volList, dimList, srList, _ = self._iterInputVolumes(self.inputVolumes, volList, dimList, srList, [])
        if self.twoSets.get():
            volList, dimList, srList, _ = self._iterInputVolumes(self.secondSet, volList, dimList, srList, [])

        # Resize volumes according to new sampling rate
        nVoli = 1
        depsConvert = []
        for _ in volList:
            convert = self._insertFunctionStep('convertStep', volList[nVoli - 1],
                                     dimList[nVoli - 1], srList[nVoli - 1],
                                     min(dimList), max(srList), nVoli, prerequisites=[])
            depsConvert.append(convert)
            nVoli += 1

        # Align all volumes to the first one (reference)
        nVoli = 1
        nVolj = 2
        depsAlign = []
        for _ in volList[1:]:
            if nVolj != nVoli:
                stepID = self._insertFunctionStep('alignStep', volList[nVolj - 1],
                                        volList[nVoli - 1], nVoli - 1,
                                        nVolj - 1, prerequisites=depsConvert)
                depsAlign.append(stepID)
            nVolj += 1

        # Zernikes3D
        nVoli = 1
        depsZernike = []
        count = 1
        for _ in volList:
            nVolj = 1
            for _ in volList:
                if nVolj != nVoli:
                    stepID = self._insertFunctionStep('deformStep', volList[nVoli - 1],
                                            volList[nVolj - 1], nVoli - 1,
                                            nVolj - 1, count, prerequisites=depsAlign)
                    depsZernike.append(stepID)
                nVolj += 1
                count += 1
            nVoli += 1

        self._insertFunctionStep('deformationMatrix', volList, prerequisites=depsZernike)
        self._insertFunctionStep('gatherResultsStepDef')

        self._insertFunctionStep('correlationMatrix', volList, prerequisites=depsZernike)
        self._insertFunctionStep('gatherResultsStepCorr')

        self._insertFunctionStep('entropyConsensus')

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
        fnOut = self._getExtraPath(os.path.basename(fnOut + self.OUTPUT_SUFFIX % nVoli))

        ih = ImageHandler()
        volFn = volFn if getExt(volFn) == '.vol' else volFn + ':mrc'
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
                                                         + self.OUTPUT_SUFFIX % (j+1)))
        refVolFn = self._getExtraPath(os.path.basename(os.path.splitext(refVolFn)[0]
                                                       + self.OUTPUT_SUFFIX % (i+1)))
        fnOut = self._getTmpPath(self.ALIGNED_VOL % (j + 1))
        params = ' --i1 %s --i2 %s --apply %s --local --dontScale' % \
                 (refVolFn, inputVolFn, fnOut)

        self.runJob("xmipp_volume_align", params)

    def deformStep(self, inputVolFn, refVolFn, i, j, step_id):
        if j == 0:
            refVolFn_aux = self._getExtraPath(os.path.basename(os.path.splitext(refVolFn)[0]
                                                               + self.OUTPUT_SUFFIX % (j + 1)))
        else:
            refVolFn_aux = self._getTmpPath(self.ALIGNED_VOL % (j + 1))
        if i == 0:
            fnOut_aux = self._getExtraPath(os.path.basename(os.path.splitext(inputVolFn)[0]
                                                            + self.OUTPUT_SUFFIX % (i + 1)))
        else:
            fnOut_aux = self._getTmpPath(self.ALIGNED_VOL % (i + 1))
        refVolFn = self._getTmpPath("reference_%d.mrc" % step_id)
        fnOut = self._getTmpPath("input_%d.mrc" % step_id)
        copyFile(refVolFn_aux, refVolFn)
        copyFile(fnOut_aux, fnOut)
        fnOut2 = self._getTmpPath('vol%dDeformedTo%d.mrc' % (i + 1, j + 1))

        params = ' -i %s -r %s -o %s --l1 %d --l2 %d --sigma "%s" --oroot %s --regularization %f' %\
                 (fnOut, refVolFn, fnOut2, self.l1.get(), self.l2.get(), self.sigma.get(),
                  self._getExtraPath('Pair_%d_%d' % (i, j)), self.penalization.get())
        if self.newRmax != 0:
            params = params + ' --Rmax %d' % self.newRmax

        if self.useGpu.get():
            self.runJob("xmipp_cuda_volume_deform_sph", params)
        else:
            params = params + ' --thr 1'
            self.runJob("xmipp_volume_deform_sph", params)

        self.computeCorr(fnOut2, refVolFn, i, j)

        cleanPath(fnOut)
        cleanPath(refVolFn)
        cleanPath(fnOut2)

    def deformationMatrix(self, volList):
        cleanPath(self._getTmpPath("*.mrc"))
        numVol = len(volList)
        self.distanceMatrix = np.zeros((numVol, numVol))
        for i in range(numVol):
            for j in range(numVol):
                if i != j:
                    path = self._getExtraPath('Pair_%d_%d_deformation.txt' % (i, j))
                    self.distanceMatrix[i, j] = np.loadtxt(path)

    def correlationMatrix(self, volList):
        numVol = len(volList)
        self.corrMatrix = np.zeros((numVol, numVol))
        for i in range(numVol):
            for j in range(numVol):
                if i != j:
                    path = self._getExtraPath('Pair_%d_%d_correlation.txt' % (i,j))
                    self.corrMatrix[i, j] = np.loadtxt(path)


    def gatherResultsStepDef(self):
        fnRoot = self._getExtraPath("DistanceMatrix.txt")
        self.saveDeformation(self.distanceMatrix, fnRoot)
        if self.twoSets.get():
            half = int(self.distanceMatrix.shape[0] / 2)
            subMatrixes = self.split(self.distanceMatrix, half, half)
            for idm in range(4):
                fnRoot = self._getExtraPath("DistanceSubMatrix_%d.txt" % (idm + 1))
                self.saveDeformation(subMatrixes[idm], fnRoot, 'Sub_%d_' % (idm + 1))

    def saveDeformation(self, matrix, fnRoot, label=''):
        np.savetxt(fnRoot, matrix, "%f")
        distance = np.asarray(matrix)
        for i in range(1, 4):
            embed = mds(distance, i)
            embedExtended = np.pad(embed, ((0, 0), (0, i - embed.shape[1])),
                                   "constant", constant_values=0)
            np.savetxt(self._defineResultsName(i, label), embedExtended)

    def computeCorr(self, vol1, vol2, i, j):
        vol = xmippLib.Image(vol1)
        defVol = xmippLib.Image(vol2)
        corr = vol.correlation(defVol)
        corr = 1-corr
        outFile = self._getExtraPath('Pair_%d_%d_correlation.txt' % (i, j))
        with open(outFile, 'w') as f:
            f.write('%f' % corr)

    def gatherResultsStepCorr(self):
        fnRoot = self._getExtraPath("CorrMatrix.txt")
        self.saveCorrelation(self.corrMatrix, fnRoot)
        if self.twoSets.get():
            half = int(self.distanceMatrix.shape[0] / 2)
            subMatrixes = self.split(self.corrMatrix, half, half)
            for idm in range(4):
                fnRoot = self._getExtraPath("CorrSubMatrix_%d.txt" % (idm + 1))
                self.saveCorrelation(subMatrixes[idm], fnRoot, 'Sub_%d_' % (idm + 1))

    def saveCorrelation(self, matrix, fnRoot, label=''):
        np.savetxt(fnRoot, matrix, "%f")
        corr = np.asarray(matrix)
        for i in range(1, 4):
            embed = mds(corr, i)
            embedExtended = np.pad(embed, ((0, 0), (0, i - embed.shape[1])),
                                   "constant", constant_values=0)
            np.savetxt(self._defineResultsName2(i, label), embedExtended)

    def rigidAlignmentMapping(self, i):
        X1 = np.loadtxt(self._defineResultsName(i))
        X2 = np.loadtxt(self._defineResultsName2(i))

        # Normalize the matrices
        [_, S, _] = np.linalg.svd(X1)
        X1 = X1 / np.amax(np.abs(S))
        [_, S, _] = np.linalg.svd(X2)
        X2 = X2 / np.amax(np.abs(S))

        # Register the points (taking mirrors into account)
        for idf in range(4):
            if idf == 0:
                R, t = rigidRegistration(X1, X2)
                X1 = t + np.transpose(R @ X1.T)
                cost = np.sum(pdist(X1 - X2))
            elif idf == 1:
                X1_attempt = np.fliplr(X1)
                R, t = rigidRegistration(X1_attempt, X2)
                X1_attempt = t + np.transpose(R @ X1_attempt.T)
                cost_attempt = np.sum(pdist(X1_attempt - X2))
                if cost_attempt < cost:
                    cost = cost_attempt
                    X1 = X1_attempt
            elif idf == 2:
                X1_attempt = np.flipud(X1)
                R, t = rigidRegistration(X1_attempt, X2)
                X1_attempt = t + np.transpose(R @ X1_attempt.T)
                cost_attempt = np.sum(pdist(X1_attempt - X2))
                if cost_attempt < cost:
                    cost = cost_attempt
                    X1 = X1_attempt
            elif idf == 3:
                X1_attempt = np.fliplr(X1)
                X1_attempt = np.flipud(X1_attempt)
                R, t = rigidRegistration(X1_attempt, X2)
                X1_attempt = t + np.transpose(R @ X1_attempt.T)
                cost_attempt = np.sum(pdist(X1_attempt - X2))
                if cost_attempt < cost:
                    cost = cost_attempt
                    X1 = X1_attempt
        return X1, X2

    def gaussianKernel(self, sigma, i):
        lbox = int(6 * sigma)
        if lbox % 2 == 0:
            lbox += 1
        mid = int((lbox - 1) / 2 + 1)
        if i == 2:
            kernel = np.zeros((lbox, lbox))
            kernel[mid, mid] = 1
            kernel = gaussian_filter(kernel, sigma=sigma)
        else:
            kernel = np.zeros((lbox, lbox, lbox))
            kernel[mid, mid, mid] = 1
            kernel = gaussian_filter(kernel, sigma=sigma)
        return kernel, lbox, mid

    def convolution(self, Xr, kernel, sigma, R, C, D):
        # Create the grid
        lbox = int(6 * sigma)
        if lbox % 2 == 0:
            lbox += 1
        mid = int((lbox - 1) / 2 + 1)
        S_shape = [d + lbox for d in R.shape]
        S = np.zeros(S_shape)

        # Place rounded points on the grid
        for p in range(Xr.shape[0]):
            if D is None:
                indy = np.argmin(np.abs(R[:, 0] - Xr[p, 0]))
                indx = np.argmin(np.abs(C[0, :] - Xr[p, 1]))
                S[indx:indx + 2 * mid - 1, indy:indy + 2 * mid - 1] += kernel
            else:
                indx = np.argmin(np.abs(R[:, 0, 0] - Xr[p, 0]))
                indy = np.argmin(np.abs(C[0, :, 0] - Xr[p, 1]))
                indz = np.argmin(np.abs(D[0, 0, :] - Xr[p, 2]))
                S[indx:indx + 2 * mid - 1, indy:indy + 2 * mid - 1, indz:indz + 2 * mid - 1] += kernel
        return S

    def entropyConsensus(self):
        for i in range(2, 4):
            # Rigid alignment of mappings (considering mirrors as well)
            X1, X2 = self.rigidAlignmentMapping(i)

            # Round point to place them in a grid
            Xr1 = np.round(X1, decimals=3)
            Xr2 = np.round(X2, decimals=3)
            size_grid = 2.5 * max((np.amax(Xr1), np.amax(Xr2)))

            # Parameters needed for future convolution
            grid_coords = np.linspace(-size_grid, size_grid, num=400)
            if i == 2:
                R, C = np.meshgrid(grid_coords, grid_coords, indexing='ij')
                D = None
            else:
                R, C, D = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
            sigma = R.shape[0] / 20

            # Create Gaussian Kernel
            kernel, lbox, mid = self.gaussianKernel(sigma, i)

            # Consensus
            alpha_vect = np.arange(0, 1.01, 0.01)
            entropy_vect = []
            X_matrices = []

            for alpha in alpha_vect:
                X = alpha * X1 + (1 - alpha) * X2
                X_matrices.append(X)

                # Round points to place them in a grid
                Xr = np.round(X, decimals=3)

                # Convolve grid and kernel
                S = self.convolution(Xr, kernel, sigma, R, C, D)

                # Compute the Shannon entropy associated to the convolved grid
                _, counts = np.unique(S, return_counts=True)
                entropy_vect.append(entropy(counts, base=2))

            # Find optimal entropy value (minimum)
            entropy_vect = np.asarray(entropy_vect)
            id_peaks, _ = find_peaks(-entropy_vect)
            if id_peaks.size > 1:
                peaks = entropy_vect[id_peaks]
                id_optimal = id_peaks[np.argmin(peaks)]
            elif id_peaks.size == 1:
                id_optimal = id_peaks[0]
            else:
                id_optimal = 0

            X_optimal = X_matrices[id_optimal]

            np.savetxt(self._defineResultsName3(i), X_optimal)

    # --------------------------- UTILS functions --------------------------------------------
    def _iterInputVolumes(self, volumes, volList, dimList, srList, idList):
        """ Iterate over all the input volumes. """
        count = 1
        for pointer in volumes:
            item = pointer.get()
            if item is None:
                break
            if isinstance(item, Volume):
                volList.append(item.getFileName())
                dimList.append(item.getDim()[0])
                srList.append(item.getSamplingRate())
                idList.append(count)
            elif isinstance(item, SetOfVolumes):
                for vol in item:
                    volList.append(vol.getFileName())
                    dimList.append(vol.getDim()[0])
                    srList.append(vol.getSamplingRate())
                    idList.append(count)
                    count += 1
            count += 1
        return volList, dimList, srList, idList

    def split(self, array, nrows, ncols):
        """Split a matrix into sub-matrices."""
        r, h = array.shape
        return (array.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    def _defineResultsName(self, i, label=''):
        return self._getExtraPath('Coordinate%sMatrix%d.txt' % (label, i))

    def _defineResultsName2(self, i, label=''):
        return self._getExtraPath('Coordinate%sMatrixCorr%d.txt' % (label, i))

    def _defineResultsName3(self, i):
        return self._getExtraPath('ConsensusMatrix%d.txt' % i)

    # ------------------------- VALIDATE functions -----------------------------
    def validate(self):
        """ Try to find errors on define params. """
        errors = []
        l1 = self.l1.get()
        l2 = self.l2.get()
        if (l1 - l2) < 0:
            errors.append('Zernike degree must be higher than '
                          'SPH degree.')
        return errors




