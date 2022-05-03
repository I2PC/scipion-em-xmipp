# **************************************************************************
# *
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
# *              Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
"""
Protocol to split a volume in two volumes based on a set of images
"""

from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL
from pyworkflow.protocol.params import PointerParam, FloatParam, IntParam, StringParam, BooleanParam, EnumParam
from pyworkflow.protocol.params import LT, LE, GE, GT, Range
from pyworkflow.utils.path import copyFile, makePath, createLink, cleanPattern, cleanPath, moveFile

from pwem.protocols import ProtClassify3D
from pwem.objects import Volume, Image, Particle, Class2D

import xmippLib
from xmipp3.convert import writeSetOfParticles

import math
import itertools
import pywt
import PIL.Image
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from scipy import special
from scipy import sparse
from scipy import stats
from scipy.sparse import csgraph
from scipy.sparse import linalg

class XmippProtSplitvolume(ProtClassify3D):
    """Split volume in two"""
    _label = 'split volume'
    _devStatus = BETA
    
    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)
        # self.stepsExecutionMode = STEPS_PARALLEL

    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('directionalClasses', PointerParam, label="Directional classes", 
                      pointerClass='SetOfClasses2D', pointerCondition='hasRepresentatives', 
                      important=True, 
                      help='Select a set of particles with angles. Preferrably the output of a run of directional classes')
        form.addParam('mask', PointerParam, label="Mask", 
                      pointerClass='Mask', allowsNull=True,
                      help='Mask used when comparing image pairs')
        form.addParam('minClassSize', IntParam, label="Minumum class size percentile (%)", 
                      validators=[Range(0, 100)], default=50,
                      help='Required number of classes per solid angle.')
        form.addParam('minClassesPerDirection', IntParam, label="Minumum number of classes per direction", 
                      validators=[GT(0)], default=1,
                      help='Required number of classes per solid angle.')
        form.addParam('keepBestDirections', IntParam, label="Keep best directions (%)", 
                      validators=[Range(0, 100)], default=30,
                      help='Percentage of directions to be used')

        form.addSection(label='Graph building')
        form.addParam('imageDistanceMetric', EnumParam, label='Image distance metric',
                      choices=['Correlation', 'Wassertain', 'Entropy'], default=1,
                      help='Distance metric used when comparing image pairs.')
        form.addParam('maxAngularDistanceMethod', EnumParam, label='Maximum angular distance method',
                      choices=['Threshold', 'Percentile'], default=0,
                      help='Determines how the maximum angular distance is obtained.')
        form.addParam('maxAngularDistanceThreshold', FloatParam, label="Maximum angular distance threshold (deg)", 
                      validators=[Range(0, 180)], default=15, condition='maxAngularDistanceMethod==0',
                      help='Maximum angular distance value for considering the comparison of two classes. '
                      'Valid range: 0 to 180 deg')
        form.addParam('maxAngularDistancePercentile', FloatParam, label="Maximum angular distance percentile (%)", 
                      validators=[Range(0, 100)], default=10, condition='maxAngularDistanceMethod==1',
                      help='Maximum angular distance percentile for considering the comparison of two classes. '
                      'Valid range: 0 to 100%')
        form.addParam('maxNeighbors', IntParam, label="Maximum number of neighbors", 
                      validators=[GE(0)], default=12,
                      help='Maximum number of comparisons to consider for each directional class. Use 0 to disable this limit')
        form.addParam('enforceUndirected', EnumParam, label="Enforce undirected", 
                      choices=['Or', 'And', 'Average'], default=1,
                      help='Enforce a undirected graph')
        form.addParam('graphMetric', EnumParam, label="Cut metric", 
                      choices=['Graph cut', 'Ratio cut', 'Normalized cut', 'Quotient cut'], default=3,
                      help='Objective function to minimize when cutting the graph in half')

        form.addParallelSection(threads=mp.cpu_count(), mpi=0)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Create the thread pool
        if self.numberOfThreads > 1:
            self._threadPool = ThreadPoolExecutor(int(self.numberOfThreads))

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('computeAngularDistancesStep')
        self._insertFunctionStep('compareImagesStep')
        self._insertFunctionStep('computeWeightsStep')
        self._insertFunctionStep('computeFiedlerVectorStep')
        self._insertFunctionStep('partitionGraphStep')
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        # Read input data
        directionalClasses = self._convertClasses2D(self.directionalClasses.get())
        inputMask = self.mask.get()
        minClassSizePercentile = self.minClassSize.get()
        minClassesPerDirection = self.minClassesPerDirection.get()
        bestDirectionsPercentage = self.keepBestDirections.get()

        # Convert images
        images = np.array(self._convertClassRepresentatives(directionalClasses), dtype=object)
        directionIds = np.array(self._calculateDirectionIds(directionalClasses), dtype=np.uint)
        classSizes = np.array(self._calculateClassSizes(directionalClasses), dtype=np.uint)

        # Convert the compare mask
        compareMask = self._convertCompareMask(inputMask, images)
        self._writeCompareMask(compareMask)

        # Filter the input images
        selection = np.ones(len(images), dtype=bool)
        selection = self._calculateClassSizeMask(classSizes, minClassSizePercentile, selection)
        selection = self._calculateDirectionSizeSelectionMask(directionIds, minClassesPerDirection, selection)
        selection = self._calculateDirectionDisparitySelectionMask(directionIds, images, bestDirectionsPercentage, selection)

        print(f'Using {np.count_nonzero(selection)} images out of {len(images)} '\
              f'({100*np.count_nonzero(selection)/len(images)}%)')

        images = images[selection]
        directionIds = directionIds[selection]

        self._writeInputImages(images)
        self._writeDirectionIds(directionIds)

    def computeAngularDistancesStep(self):
        # Read input data
        images = self._readInputImages()

        # Perform the computation
        distances = self._calculateAngularDistanceMatrix(images)

        # Save output data
        self._writeAngularDistances(distances)

    def compareImagesStep(self):
        # Read input data
        images = self._readInputImages()
        directionIds = self._readDirectionIds()
        distances = self._readAngularDistances()
        maxAngularDistance = self._getMaxAngularDistance(distances)
        maxNeighbors = self._getMaxNeighbors()

        # Compute the image pairs to be considered
        pairs = self._calculateImagePairs(distances, directionIds, maxAngularDistance, maxNeighbors)
        print(f'Considering {len(pairs)} image pairs with an '
              f'angular distance of up to {np.degrees(maxAngularDistance)} deg')

        # Compute the comparisons
        comparisons = self._calculateComparisonMatrix(
            images, pairs, 
            self._getImageCompareFunc(), 
            self._getThreadPool()
        )

        # Get the similarity ponderation based on the distance
        ponderation = self._calculateDistancePonderation(distances, maxAngularDistance)
        comparisons *= ponderation

        # Save output data
        self._writeComparisons(comparisons)

    def computeWeightsStep(self):
        # Read input data
        comparisons = self._readComparisons()
        symmetrizeFunc = self._getSymmetrizeFunc()

        # Calculate weights from comparisons
        weights = self._calculateWeights(comparisons, symmetrizeFunc, 40, 90)

        # Print information about the graph
        nComponents, _ = csgraph.connected_components(weights)
        print(f'The unpartitioned graph has {nComponents} components')

        # Save output data
        self._writeWeights(weights)

    def computeFiedlerVectorStep(self):
        # Read input data
        adjacency = self._readWeights()
        graph = sparse.csr_matrix(adjacency)

        # Compute the fiedler vector
        fiedler = self._calculateFiedlerVector(graph)

        # Write the result
        self._writeFiedlerVector(fiedler)

    def partitionGraphStep(self):
        # Read the input data
        graph = self._readWeights()
        fiedler = self._readFiedlerVector()
        metric = self._getGraphPartitionMetricFunc()

        # Partition the graph using the Fiedler vector
        labels = self._partitionFiedlerVector(graph, fiedler, metric)

        # Write the result
        self._writeLabels(labels)

    def createOutputStep(self):
        # Read input data
        images = self._getInputImages()
        labels = self._readLabels()

        # Create the output elements
        classes = self._createOutputClasses(images, labels, 'output')
        volumes = self._createOutputVolumes(classes, 'output')

        # Define the output
        sources = [self.directionalClasses]
        outputs = {
            'outputClasses': classes,
            'outputVolumes': volumes
        }

        self._defineOutputs(**outputs)
        for src in sources:
            for dst in outputs.values():
                self._defineSourceRelation(src, dst)

    #--------------------------- INFO functions ----------------------------------------------------
    def _validate(self):
        result = []

        if self.mask.get() is not None:
            if self.mask.get().getDim() != self.directionalClasses.get().getDimensions():
                result.append('Mask and Directional Class representatives must have the same size')

        return result

    #--------------------------- UTILS functions ---------------------------------------------------
    def _getThreadPool(self):
        return getattr(self, '_threadPool', None)

    def _getImageCompareFunc(self):
        options = [
            self._compareImagesCorrelation,
            self._compareImagesWassertein,
            self._compareImagesEntropy
        ]
        selection = self.imageDistanceMetric.get()
        return options[selection]

    def _getMaxAngularDistance(self, distances):
        method = self.maxAngularDistanceMethod.get()
        if method == 0:
            return np.radians(self.maxAngularDistanceThreshold.get())
        elif method == 1:
            return np.percentile(distances, self.maxAngularDistancePercentile.get())

    def _getMaxNeighbors(self):
        count = self.maxNeighbors.get()
        return count if count > 0 else None

    def _getSymmetrizeFunc(self):
        options = [
            np.maximum, 
            np.minimum, 
            lambda x, y : (x+y)/2
        ]
        selection = self.enforceUndirected.get()
        return options[selection]

    def _getGraphPartitionMetricFunc(self):
        options = [
            self._calculateGraphCutMetric,
            self._calculateRatioCutMetric,
            self._calculateNormalizedCutMetric,
            self._calculateQuotientCutMetric
        ]
        selection = self.graphMetric.get()
        return options[selection]

    def _setInputImages(self, images):
        self._insertChild('_images', images)

    def _getInputImages(self):
        return self._images

    def _writeInputImages(self, images):
        result = self._createSetOfParticles('input')
        result.setAcquisition(images[0].getAcquisition())
        result.setSamplingRate(images[0].getSamplingRate())
        result.setHasCTF(images[0].hasCTF())
        result.setAlignmentProj()

        for image in images:
            result.append(image)
        
        self._setInputImages(result)

    def _readInputImages(self):
        return list(map(Particle.clone, self._getInputImages()))

    def _writeMatrix(self, path, x, fmt='%s'):
        # Determine the format
        np.savetxt(path, x, delimiter=',', fmt=fmt) # CSV

    def _readMatrix(self, path, dtype=float):
        return np.genfromtxt(path, delimiter=',', dtype=dtype) # CSV

    def _storeMatrix(self, attribute, x, fmt):
        """ Stores an ndarray to disk and as an attribute 
        """
        attributeName = '_' + attribute
        attributeFile = self._getExtraPath(attribute + '.csv')

        setattr(self, attributeName, x)
        self._writeMatrix(attributeFile, x, fmt=fmt)

    def _loadMatrix(self, attribute, dtype):
        """ Searches if the requested ndarray is stored as an attribute.
            If not, it loads it from disk
        """
        attributeName = '_' + attribute
        attributeFile = self._getExtraPath(attribute + '.csv')

        result = getattr(self, attributeName, None)
        if result is None:
            # Need to load it from disk
            result = self._readMatrix(attributeFile, dtype=dtype)
            setattr(self, attributeName, result) # Keep it for future calls

        assert(result is not None)
        assert(result.dtype==dtype)
        return result

    def _writeDirectionIds(self, directions):
        assert(directions.dtype==np.uint)
        self._storeMatrix('directions', directions, fmt='%d')

    def _readDirectionIds(self):
        return self._loadMatrix('directions', dtype=np.uint)

    def _writeCompareMask(self, mask):
        assert(mask.dtype==bool)
        self._storeMatrix('mask', mask, fmt='%d')
    
    def _readCompareMask(self):
        return self._loadMatrix('mask', dtype=bool)

    def _writeAngularDistances(self, distances):
        assert(distances.dtype==float)
        self._storeMatrix('distances', distances, fmt='%.8f')

    def _readAngularDistances(self):
        return self._loadMatrix('distances', dtype=float)

    def _writeComparisons(self, comparisons):
        assert(comparisons.dtype==float)
        self._storeMatrix('comparisons', comparisons, fmt='%.8f')

    def _readComparisons(self):
        return self._loadMatrix('comparisons', dtype=float)

    def _writeWeights(self, weights):
        assert(weights.dtype==float)
        self._storeMatrix('weights', weights, fmt='%.8f')
    
    def _readWeights(self):
        return self._loadMatrix('weights', dtype=float)

    def _writeFiedlerVector(self, fiedler):
        assert(fiedler.dtype==float)
        self._storeMatrix('fiedler', fiedler, fmt='%.8f')

    def _readFiedlerVector(self):
        return self._loadMatrix('fiedler', dtype=float)

    def _writeLabels(self, labels):
        assert(labels.dtype==np.uint)
        self._storeMatrix('labels', labels, fmt='%d')

    def _readLabels(self):
        return self._loadMatrix('labels', dtype=np.uint)

    def _convertClasses2D(self, classes):
        result = []
        for cls in classes:
            result.append(cls.clone())
        return result

    def _convertToXmippImages(self, images):
        """Converts a list of images into xmipp images"""
        locations = map(Image.getLocation, images)
        result = list(map(xmippLib.Image, locations))
        return result

    def _convertClassRepresentatives(self, classes):
        return list(map(Class2D.getRepresentative, classes))

    def _calculateDirectionIds(self, classes):
        reprojections = list(map(lambda cls : cls.reprojection.getLocation(), classes))

        # Create a dictionary assotiating reprojections and its ids
        uniqueReprojections = set(reprojections)
        uniqueReprojectionIds = range(len(uniqueReprojections))
        reprojectionIds = dict(zip(uniqueReprojections, uniqueReprojectionIds))

        # Map the reprojections to its ids
        ids = list(map(reprojectionIds.__getitem__, reprojections))
        return ids

    def _calculateClassSizes(self, classes):
        return list(map(len, classes))

    def _calculateClassSizeMask(self, classSizes, sizePercentile, mask):
        threshold = np.percentile(classSizes, sizePercentile)
        mask = np.logical_and(mask, classSizes >= threshold)
        return mask

    def _calculateDirectionSizeSelectionMask(self, directionIds, minClassesPerDirection, mask):
        if minClassesPerDirection > 1:
            repetitions = Counter(directionIds[mask]) # Only count where selected
            return np.array(list(map(
                lambda enable, id: enable and repetitions[id] >= minClassesPerDirection, 
                mask,
                directionIds,
            )))
        else:
            # Nothing to do
            return mask

    def _calculateDirectionDisparitySelectionMask(self, directionIds, images, keepBestDirections, mask):
        if keepBestDirections < 100:
            compareFunc = self._getImageCompareFunc()

            # For all directions compute the correlation among its members
            directionDisparities = {}
            for direction in np.unique(directionIds):
                selection = np.logical_and(mask, directionIds==direction)
                if np.count_nonzero(selection) >= 2:
                    # Compare all image pairs possible for this direction
                    directionImages = self._convertToXmippImages(images[selection])
                    imagePairs = itertools.combinations(directionImages, r=2)
                    correlations = map(compareFunc, *zip(*imagePairs))

                    # Use the maximum comparison as the disparity
                    directionDisparities[direction] = max(correlations)
                else:
                    directionDisparities[direction] = 1 # 1 is considered as "equal"

            # Compute the threshold
            threshold = np.percentile(list(directionDisparities.values()), keepBestDirections)

            # Select only the directions that have a disparity below the threshold
            return np.array(list(map(
                lambda enable, id: enable and directionDisparities[id] <= threshold, 
                mask,
                directionIds,
            )))

        else:
            # Nothing to do
            return mask

    def _convertCompareMask(self, mask, images, dtype=bool):
        if mask is not None:
            return xmippLib.Image(mask.getLocation()).getData().astype(dtype)
        else:
            dimensions = images[0].getDim()
            return np.ones(dimensions[0:2], dtype=dtype) 

    def _getProjectionDirection(self, image):
        # Multiply by the unit z vector, as projection is performed in this direction
        return image.getTransform().getRotationMatrix()[:, 2]

    def _calculateAngularDistance(self, image0, image1):
        """ Given two rotation matrices it computes the
            angle between their transformations
        """
        # Obtain the projection directions of the images
        dir0 = self._getProjectionDirection(image0)
        dir1 = self._getProjectionDirection(image1)

        # Calculate the cosine of the angle between projections. 
        # Note that dir0 and dir1 are normalized
        c = np.dot(dir0, dir1)

        # As projections in the antipodes are equivalent (but mirrored) also consider
        # the complementary angle and choose the smallest one
        c = abs(c)

        # Clip to [0, 1] as floating point errors may lead to values slightly 
        # outside of the domain of acos
        c = min(c, 1.0)

        # Obtain the angle from the cosine
        angle = math.acos(c)

        return angle

    def _calculateAngularDistanceMatrix(self, images):
        """ Given a set of images, it computes the angular
            distances among all pairs. The result is represented
            by a symmetrical matrix with the angles in radians
        """
        # Compute a symmetric matrix with the angular distances.
        # Do not compute the diagonal, as the distance with itself is zero.
        distances = np.zeros(shape=(len(images), )*2)
        for idx0, idx1 in itertools.combinations(range(len(images)), r=2):
            # Write the result on symmetrical positions
            angle = self._calculateAngularDistance(images[idx0], images[idx1])
            distances[idx0, idx1] = angle
            distances[idx1, idx0] = angle

        # Ensure that the result matrix is symmetrical and with 0s in the diagonal
        assert(np.array_equal(distances, distances.T))
        assert(np.all(np.diagonal(distances) == 0))
        return distances

    def _calculateImagePairs(self, distances, directionIds, maxDistance, maxNeighbors):
        """ Computes the indices of a matrix where to compute
            the comparisons. In order to do so, it considers
            a distance threshold and a maximum number of comparisons
            per each row
        """
        # Consider only distances closer than the threshold
        pairs = distances <= maxDistance

        # Limit the number of comparisons to be computed
        for pairRow, distanceRow in zip(pairs, distances):
            indices = np.argsort(distanceRow)
            pairRow[indices[maxNeighbors+1:]] = False

        # Remove comparisons with itself. This is redundant,
        # as it would be done when removing by reprojections.
        # Hovever is cheaper to do like this.
        np.fill_diagonal(pairs, False)

        # Remove all the comparisons corresponding to its same reprojection
        for idx0, idx1 in np.argwhere(pairs):
            if directionIds[idx0] == directionIds[idx1]:
                pairs[idx0, idx1] = False

        return np.argwhere(pairs)

    def _subtractAlignedImages(self, img0, img1):
        # Align img1 respect img0
        img1 = img0.alignConsideringMirrors(img1)
        img0 = img0.getData()
        img1 = img1.getData()

        # Mask both images to consider alignment wrapping
        mask = self._readCompareMask()
        img0 *= mask
        img1 *= mask

        # Subtract adjusting their mean so that the difference has a mean of 0
        result = img0 - img1*(np.mean(img0)/np.mean(img1))

        assert(np.isclose(np.mean(result), 0))
        return result

    def _compareImagesCorrelation(self, img0, img1):
        correlation = img0.correlationAfterAlignment(img1)

        # Clip the values lower than 0 to 0
        result = max(correlation, 0.0)

        assert(result >= 0)
        assert(result <= 1)
        return result

    def _compareImagesWassertein(self, img0, img1, wavelet='sym5', nLevels=None):
        # Based on:
        # https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/master/src/aspire/operators/wemd.py

        # Align the images and compute the difference
        diff = self._subtractAlignedImages(img0, img1)

        # Set the default Wavelet transform level count if not given
        if nLevels is None:
            nLevels = math.ceil(math.log2(max(diff.shape))) + 1

        # Compute the Wavelet transform
        dwt = pywt.wavedecn(diff, wavelet, mode="zero", level=nLevels)
        detailCoefficients = dwt[1:]
        assert(len(detailCoefficients) == nLevels)

        weightedCoefficients = []
        r = diff.ndim / 2 + 1 # 2 for 2D
        for j, details in enumerate(detailCoefficients):
            level = nLevels - j - 1 # Levels are in descending order
            weight = 2 ** (r*level) 
            for detail in details.values():
                weightedCoefficients.append(weight*detail.flatten())

        # Compute the Wasserstein distance
        allWeightedCoefficients = np.concatenate(weightedCoefficients)
        distance = np.linalg.norm(allWeightedCoefficients, ord=1) # Use the Manhattan norm

        # Map the distance to the [0, 1] range, considering 1 as equal
        factor = len(allWeightedCoefficients) * 16
        result = math.exp(-distance/factor) # TODO use a linear function

        assert(result >= 0)
        assert(result <= 1)
        return result

    def _compareImagesEntropy(self, img0, img1, nBins=(16, )*2):
        # Align the images and compute the difference
        diff = self._subtractAlignedImages(img0, img1)

        # Square the difference image to obtain the absolute value
        diff *= diff

        # Calculate the spatial probability distribution
        bins = np.array(PIL.Image.fromarray(diff).resize(nBins, PIL.Image.BOX))
        bins /= np.sum(bins)
        assert(np.all(bins >= 0))
        assert(np.all(bins <= 1))
        assert(np.isclose(np.sum(bins), 1))

        # Calculate the Shannon entropy
        entropy = -np.sum(bins*np.log2(bins))

        # The maximum entropy is obtained for a uniform distribution, this is,
        # a probability distribution with p_i = 1/N. Therefore, the maximum 
        # entropy value is -N*(1/N)*log(1/N)=-log2(1/N)=log2(N)
        # Normalize the result to [0, 1]
        result = entropy / np.log2(np.prod(bins.shape))

        # Modulate the result with the total energy
        result *= math.exp(-np.mean(diff))

        assert(result >= 0)
        assert(result <= 1)
        return result

    def _calculateComparisonMatrix(self, images, pairs, compareFunc, threadPool=None):
        """ Computes the comparisons between image pairs
            defined by pairs array
        """
        
        def minmax(iterable):
            return min(iterable), max(iterable)

        mapFunc = threadPool.map if threadPool is not None else map

        # Convert to xmipp images in order to use functions that deal with its data
        xmippImages = self._convertToXmippImages(images)
        
        # Select the image pairs to be processed.
        imagePairs = {}
        for pair in pairs:
            # Refer to the lower triangle of the matrix
            pair = minmax(pair)

            # Add it if not defined
            if pair not in imagePairs:
                imagePairs[pair] = (xmippImages[pair[0]], xmippImages[pair[1]])

        # Compute the comparisons using parallelization if requested
        computedComparisons = dict(zip(
            imagePairs.keys(),
            mapFunc(compareFunc, *zip(*imagePairs.values()))
        ))

        # Write the computed comparisons to the resulting matrix
        comparisons = np.zeros((len(images), )*2, dtype=float)
        for pair in pairs:
            comparisons[tuple(pair)] = computedComparisons[minmax(pair)]

        # Ensure that the result matrix is in [0, 1]
        assert(np.all(comparisons >= 0.0))
        assert(np.all(comparisons <= 1.0))
        return comparisons

    def _calculateDistancePonderation(self, distances, maxDistance):
        """ Gives each of the distances a ponderation to stimulate
            the similarity metric for images that are far away.
            For this purpose, the Cumulative Normal Distribution Function
            is used, so that images that are close are penalized 
            with a ponderation of 0.5, whilst images that are maxDistance 
            appart will receive a ponderation of 0.9999 (virtually 1)
        """

        # Calculate the typical deviation assuming a quasi-1 probability for distance being
        # smaller than maxDistance. PPF (Percent Point Function) is the same as Inverse CDF
        pMaxDistance = 1-1e-4 # Probability for distance being smaller than maxDistance.
        sigma = maxDistance / stats.norm.ppf(pMaxDistance, loc=0, scale=1)
        assert(np.isclose(stats.norm.cdf(maxDistance, loc=0, scale=sigma), pMaxDistance))

        # Calculate the Cumulative Normal Distribution Function for the distances.
        return stats.norm.cdf(distances, loc=0, scale=sigma)

    def _smoothStep(self, x, xMin=0, xMax=1, N=1):
        """ Performs the Nth order Hermite interpolation
            of x between xMin and xMax
        """
        if xMin == xMax:
            # In case xMin == xMax use a normal step to avoid division by zero
            result = np.where(x < xMin, 0, 1)

        else:
            # Based on:
            # https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python

            # Obtain the "progress" inside the range [xMin, xMax]. If outside the range, clamp to 0 or 1
            t = np.clip((x - xMin) / (xMax - xMin), 0, 1)
        
            # Use the Nth degree Hermite function for t
            result = np.zeros_like(t)
            for n in range(N + 1):
                result += special.comb(N + n, n) * special.comb(2*N + 1, N - n) * (-t) ** n
            result *= t ** (N + 1)

        return result

    def _calculateWeights(self, comparisons, symmetrizeFunc, lower=50, upper=80):
        """ Given a matrix of comparisons among neighboring 
            images, it computes the adjacency matrix with
            interger weights
        """

        # Perform a Hermite interpolation of the comparisons between the given percentiles
        weights = np.zeros_like(comparisons)
        for weightRow, compRow in zip(weights, comparisons):
            # Calculate lower and upper percentiles for the non zero comparisons
            values = compRow[np.nonzero(compRow)]
            if len(values) > 0:
                [minVal, maxVal] = np.percentile(values, [lower, upper])
                weightRow[:] = self._smoothStep(compRow, xMin=minVal, xMax=maxVal, N=1)

        # Ensure the graph is symmetric
        weights = symmetrizeFunc(weights, weights.T)

        assert(np.all(weights >= 0))
        assert(np.all(weights <= 1))
        return weights

    def _calculateFiedlerVector(self, graph):
        """ Given a adjacency matrix, it computes the Fiedler vector, this is,
            the eigenvector with the second smallest eigenvalue of its laplacian
            matrix
            https://people.csail.mit.edu/jshun/6886-s18/lectures/lecture13-1.pdf#page=11
            https://es.mathworks.com/help/matlab/math/partition-graph-with-laplacian-matrix.html
        """
        # In order to use eigsh, graph matrix needs to be symmetric (undirected graph)
        if not np.array_equal(graph.toarray(), graph.T.toarray()):
            raise ValueError('Input graph must be undirected')

        # Decompose the laplacian matrix of the graph into 2
        # eigenvectors with the smallest eigenvalues
        l = csgraph.laplacian(graph, normed=True)
        values, vectors = linalg.eigsh(l, k=2, which='SM')
        assert(np.array_equal(values, sorted(values)))

        # Fiedler vector corresponds to second eigenvector
        # with the smallest eigenvalue
        return vectors[:,1]

    def _calculateGraphSizes(self, labels):
        """ Computes the vertex count of each component of the graph
        """
        counter  = Counter(labels)
        return np.array(list(counter.values()))

    def _calculateGraphVolumes(self, graph, labels):
        """ Computes the volume of each component of the graph
        """
        result = np.zeros(int(np.max(labels))+1)
        degrees = np.sum(graph, axis=1)

        for i in range(len(labels)):
            result[labels[i]] += degrees[i]

        return result

    def _calculateGraphCutMetric(self, graph, labels):
        cost = 0
        for idx0, idx1 in zip(*np.nonzero(graph)):
            if labels[idx0] != labels[idx1]:
                cost += graph[idx0, idx1]
        return cost

    def _calculateRatioCutMetric(self, graph, labels):
        graphCut = self._calculateGraphCutMetric(graph, labels)
        sizes = self._calculateGraphSizes(labels)
        return graphCut * np.sum(1/sizes) # Inverse sum of sizes

    def _calculateNormalizedCutMetric(self, graph, labels):
        graphCut = self._calculateGraphCutMetric(graph, labels)
        volumes = self._calculateGraphVolumes(graph, labels)
        return graphCut * np.sum(1/volumes) # Inverse sum of volumes

    def _calculateQuotientCutMetric(self, graph, labels):
        graphCut = self._calculateGraphCutMetric(graph, labels)
        volumes = self._calculateGraphVolumes(graph, labels)
        return graphCut * 1/np.min(volumes)

    def _calculateCutLabels(self, ordering, cut):
        """ Obtains a the labels for a graph bipartition
            at cut (cut is the index of last vertex that 
            is member of the class 0) with the vertex priority 
            given by ordering
        """
        labels = np.zeros(len(ordering), dtype=np.uint)
        labels[ordering[cut+1:]] = 1
        return labels

    def _calculateMetricValues(self, graph, ordering, metric):
        """ Obtains the values of the metric function
            for each possible cut given by the vertex
            priority defined by ordering
        """
        # Make an array for all possible cuts
        values = np.zeros(len(ordering)-1)

        # Calculate the cost metric for each cut
        for i in range(len(values)):
            # Make a fictitious classification at cut
            labels = self._calculateCutLabels(ordering, i)

            # Calculate the metric for it
            values[i] = metric(graph, labels)

        return values

    def _partitionFiedlerVector(self, graph, fiedler, metric):
        """ Partition the graph into 2 components which
            minimize the given metric
        """
        # Sort the fiedler vector so that it represents the cut priority of each vertex
        indices = np.argsort(fiedler)

        # Calculate the metric function values
        metricValues = self._calculateMetricValues(graph, indices, metric)

        # Calculate the cheapest cut
        cut = np.argmin(metricValues)
        labels = self._calculateCutLabels(indices, cut)

        return labels

    def _reconstructVolume(self, path, particles):
        # Convert the particles to a xmipp metadata file
        fnParticles = path+'_particles.xmd'
        writeSetOfParticles(particles, fnParticles)

        # Reconstruct the volume
        args  = f'-i {fnParticles} '
        args += f'-o {path} '
        args += f'--max_resolution 0.25 '
        args += f'-v 0'
        self.runJob('xmipp_reconstruct_fourier', args)

        # Clear the metadata file
        cleanPattern(fnParticles)

    def _createOutputClasses(self, images, classification, suffix=''):
        result = self._createSetOfClasses3D(images, suffix)

        # Classify input particles
        repCbk = lambda cls : self._getExtraPath(f'{suffix}_volume_c{cls:02d}.vol')
        loader = XmippProtSplitvolume.ClassesLoader(classification, repCbk)
        loader.fillClasses(result)

        # Create the representatives
        for cls in result:
            representative = cls.getRepresentative()
            self._reconstructVolume(representative.getFileName(), cls)

        return result

    def _createOutputVolumes(self, classes, suffix=''):
        result = self._createSetOfVolumes(suffix)
        result.setSamplingRate(classes.getImages().getSamplingRate())

        for cls in classes:
            volume = cls.getRepresentative().clone()
            volume.setObjId(cls.getObjId())
            result.append(volume)
        
        return result

    class ClassesLoader:
        """ Helper class to produce classes
        """
        def __init__(self, classification, representativeCallback):
            self.classification = classification
            self.representativeCallback = representativeCallback

        def fillClasses(self, clsSet):
            clsSet.classifyItems(updateItemCallback=self._updateParticle,
                                itemDataIterator=iter(self.classification),
                                updateClassCallback=self._updateClass,
                                doClone=False)

        def _updateParticle(self, item, cls):
            classId = cls + 1
            item.setClassId(classId)

        def _updateClass(self, item):
            location = self.representativeCallback(item.getObjId()-1)
            item.setRepresentative(Volume(location=location))