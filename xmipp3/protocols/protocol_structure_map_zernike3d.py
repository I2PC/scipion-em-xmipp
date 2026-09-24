
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

    AI Generated

    ## Overview

    The Struct Map - Zernike3D protocol compares several 3D volumes using
    deformable Zernike3D-based registration.

    The goal is to build a structural map that represents how similar or different
    a set of volumes are. Unlike simple correlation-based comparison, this protocol
    also estimates how much deformation is needed to transform one volume into
    another. Volumes that can be transformed into each other with small
    deformations are considered structurally close, while volumes requiring larger
    deformations are considered more distant.

    The protocol computes two complementary descriptions:

    - a deformation-distance map based on Zernike3D deformation;
    - a correlation-distance map after deformable fitting.

    It also computes consensus embeddings that combine the deformation-based and
    correlation-based structural maps.

    This protocol is useful for analyzing conformational variability, comparing
    sets of maps, and visualizing structural relationships among multiple 3D
    volumes.

    ## Inputs and General Workflow

    The input is one or more volumes or sets of volumes.

    The protocol first collects all input volumes and rescales them to a common
    working sampling rate determined by the target resolution. It also crops them
    to a common box size. Then, all volumes except the first are locally aligned to
    the first volume, which acts as the common reference frame.

    After this preparation, the protocol performs pairwise deformable comparisons.
    For each pair of different volumes, it estimates a Zernike3D deformation that
    maps one volume onto the other. The protocol stores both the deformation
    distance and the correlation distance after deformation.

    Finally, it converts the distance matrices into low-dimensional coordinates
    using multidimensional scaling and computes consensus mappings between the
    deformation-based and correlation-based representations.

    ## Input Volume(s)

    The **Input volume(s)** parameter accepts one or more individual volumes or
    sets of volumes.

    Each selected volume is included in the structural map. If a SetOfVolumes is
    provided, all volumes in the set are used.

    The input volumes should represent related structures. For example, they may be
    different 3D classes, different conformational states, maps from different
    processing branches, or maps from related datasets.

    The method assumes that meaningful structural relationships can be described by
    deformations between volumes. Completely unrelated maps may produce distances
    that are technically computable but biologically difficult to interpret.

    ## Compare Two Sets

    The **Compare two sets?** option enables a two-set comparison mode.

    This is useful when the user wants to compare two groups of volumes, for
    example experimental EMDB-like maps against maps generated from atomic models,
    or two different families of reconstructions.

    When this option is enabled, the user provides a **Second set of volumes**.
    The protocol combines both sets for the pairwise calculations, but it also
    writes submatrices that separate within-set and between-set comparisons.

    This helps distinguish relationships inside each group from relationships
    between the two groups.

    ## Second Set of Volumes

    The **Second set of volumes** parameter is used only when two-set comparison is
    enabled.

    It accepts one or more volumes or sets of volumes. These volumes are appended
    to the first input group and included in the same deformation and correlation
    analyses.

    For meaningful interpretation, the two sets should be comparable in scale,
    orientation, molecular content, and resolution range.

    ## Target Resolution

    The **Target resolution** parameter defines the resolution used to prepare the
    volumes for comparison.

    The protocol rescales the volumes so that this resolution is placed at
    approximately two thirds of the Fourier spectrum. This focuses the comparison
    on structural information at the selected scale and reduces the influence of
    high-frequency noise.

    The default value is 8 Å, which is suitable for comparing global shape and
    medium-resolution conformational differences.

    A lower numerical target resolution includes finer detail but may make the
    comparison more sensitive to noise or reconstruction artifacts. A higher value
    focuses on coarser structural differences.

    ## Multiresolution

    The **Multiresolution** parameter defines the filter settings used during the
    Zernike3D deformation comparison.

    The values specify cutoff frequencies in normalized units, normalized to one
    half of the Fourier spectrum. The protocol can therefore compare different
    filtered versions of the volumes during deformation estimation.

    This multiresolution strategy helps the deformation fit use information at
    different spatial scales. It can make the comparison more robust than relying
    on a single frequency band.

    The default values provide a simple two-level comparison.

    ## Sphere Radius

    The **Sphere radius** parameter defines the radius, in voxels, of the sphere
    where the spherical harmonics are computed.

    If the value is 0, the underlying deformation program uses its default behavior.

    This is an advanced parameter. It should be adjusted only when the user knows
    that the deformation support should be restricted to a specific radius.

    The radius is internally rescaled when the volumes are resized to the working
    sampling rate.

    ## Zernike Degree

    The **Zernike Degree** parameter controls the degree of the Zernike polynomials
    used to model the deformation.

    Higher degrees allow more complex deformations. Lower degrees restrict the
    deformation to smoother, simpler changes.

    The default value is intended to capture relatively smooth structural
    variability.

    Increasing this value may help describe more complex conformational changes,
    but it can also make the deformation more flexible and potentially less robust.

    ## Harmonic Degree

    The **Harmonical Degree** parameter controls the degree of the spherical
    harmonics used in the deformation model.

    Together with the Zernike degree, it determines the complexity of the allowed
    deformation field.

    The protocol validates that the Zernike degree must be greater than or equal
    to the harmonic degree. If the harmonic degree is larger, the protocol reports
    an error.

    ## Regularization

    The **Regularization** parameter penalizes deformation magnitude.

    A larger regularization value discourages large or complex deformations. A
    smaller value allows the model to deform more freely.

    Regularization is important because a deformation model that is too flexible
    may fit noise or local artifacts rather than meaningful structural
    differences. A model that is too restricted may fail to capture real
    conformational changes.

    The default value is a practical compromise for many structural mapping tasks.

    ## GPU Execution

    The protocol supports both GPU and CPU execution.

    When GPU execution is enabled, the protocol uses the CUDA implementation of the
    Zernike3D deformation program. This is usually faster and is recommended when
    available.

    When GPU execution is disabled, the CPU implementation is used.

    Because the protocol performs many pairwise deformable registrations, GPU
    execution can substantially reduce runtime for large volume sets.

    ## Volume Rescaling and Cropping

    Before pairwise comparison, all volumes are resized to a common working
    sampling rate determined by the target resolution and the original sampling
    rates.

    They are then cropped to a common box size, using the smallest box dimension
    among the input volumes.

    This makes pairwise comparison technically consistent. However, users should
    ensure that important density is not lost during cropping and that all volumes
    represent comparable molecular regions.

    ## Initial Local Alignment

    Before deformable comparison, all volumes except the first are locally aligned
    to the first volume.

    This places the volume set into a common approximate coordinate frame. The
    Zernike3D deformation step can then focus on structural differences rather than
    large rigid-body misalignments.

    This local alignment assumes that the volumes are already roughly comparable.
    The protocol is not intended to align completely unrelated maps from arbitrary
    orientations.

    ## Pairwise Zernike3D Deformation

    For each ordered pair of different volumes, the protocol estimates a
    Zernike3D deformation from one volume to the other.

    The deformation program writes files describing the pairwise deformation and
    the deformation distance. The protocol collects these values into a
    deformation-distance matrix.

    This matrix reflects how much deformation is needed to relate each volume to
    each other volume.

    ## Correlation After Deformation

    After deforming one volume toward another, the protocol computes the
    correlation between the deformed volume and the reference volume.

    It converts this value into a correlation distance:

    \[
    \text{distance} = 1 - \text{correlation}
    \]

    This provides a complementary measure of how well the deformation explains the
    relationship between the two maps.

    The protocol stores these values in a correlation-distance matrix.

    ## Deformation Distance Matrix

    The deformation-distance matrix is written to:

    `DistanceMatrix.txt`

    This matrix contains the pairwise deformation distances between all input
    volumes.

    Small values indicate that two volumes can be related by a smaller deformation.
    Large values indicate that stronger deformation is needed.

    When two-set comparison is enabled, the protocol also writes deformation
    submatrices that separate within-set and between-set comparisons.

    ## Correlation Distance Matrix

    The correlation-distance matrix is written to:

    `CorrMatrix.txt`

    This matrix contains distances derived from the correlation between deformed
    and reference volumes.

    It complements the deformation matrix. Two volumes may require a moderate
    deformation but still achieve high correlation after fitting, or they may show
    poor correlation even after deformation.

    When two-set comparison is enabled, correlation submatrices are also written.

    ## Low-Dimensional Coordinate Maps

    The protocol converts both distance matrices into coordinate representations
    using multidimensional scaling.

    For the deformation-distance matrix, it writes:

    - `CoordinateMatrix1.txt`;
    - `CoordinateMatrix2.txt`;
    - `CoordinateMatrix3.txt`.

    For the correlation-distance matrix, it writes:

    - `CoordinateMatrixCorr1.txt`;
    - `CoordinateMatrixCorr2.txt`;
    - `CoordinateMatrixCorr3.txt`.

    These files contain 1D, 2D, and 3D embeddings of the volume relationships.

    The 2D and 3D maps are usually most useful for visualization. Nearby points
    represent structurally similar volumes; distant points represent more distinct
    volumes.

    ## Consensus Structural Maps

    The protocol also computes consensus mappings between the deformation-based
    and correlation-based embeddings.

    For 2D and 3D embeddings, it aligns the two coordinate maps, considers possible
    mirror relationships, and searches for an optimal mixture between them. The
    criterion is based on an entropy measure of the point distribution.

    The consensus maps are written as:

    - `ConsensusMatrix2.txt`;
    - `ConsensusMatrix3.txt`.

    These consensus representations are intended to combine information from both
    deformation distance and correlation distance.

    ## Interpreting the Structural Map

    The structural map should be interpreted as a relative map of structural
    relationships among the input volumes.

    Clusters may indicate related conformations or similar reconstructions.
    Gradients may indicate continuous conformational changes. Outliers may
    represent distinct states, artifacts, poorly reconstructed maps, or volumes
    that are difficult to deform into the others.

    The deformation-based map emphasizes how much shape change is needed between
    volumes. The correlation-based map emphasizes how similar the volumes are after
    deformable fitting. The consensus map attempts to combine these perspectives.

    ## Practical Recommendations

    Use this protocol when several related volumes need to be compared in terms of
    structural variability.

    Use the two-set option when comparing two groups of maps, such as experimental
    maps versus model-derived maps.

    Start with the default target resolution of 8 Å for global or medium-resolution
    structural comparison.

    Use conservative Zernike and harmonic degrees at first. Increase them only if
    the expected conformational changes require more flexible deformation.

    Keep regularization enabled and avoid making it too small, especially when maps
    are noisy.

    Use GPU execution when available, because the pairwise deformation calculations
    can be computationally demanding.

    Inspect deformation-based, correlation-based, and consensus coordinate maps.
    Agreement among them supports a stable interpretation, while disagreement may
    indicate complex differences or sensitivity to the comparison metric.

    Always inspect the original volumes as well. The structural map summarizes
    relationships but does not explain their biological cause by itself.

    ## Final Perspective

    Struct Map - Zernike3D is a comparative volume-analysis protocol based on
    deformable registration.

    For biological users, its main value is that it transforms a collection of 3D
    maps into a structural landscape. This can help reveal clusters, continuous
    conformational variability, outliers, and relationships between two groups of
    maps.

    The protocol should be used as an exploratory and interpretative tool. Its
    outputs suggest structural relationships, but biological conclusions should be
    supported by visual inspection, reconstruction quality, class sizes, and the
    experimental context.
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




