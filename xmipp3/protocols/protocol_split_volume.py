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

from pwem import emlib
from pwem.protocols import ProtClassify3D
from pwem.objects import Volume, Image, Particle, Class2D
from pwem.constants import ALIGN_PROJ

import xmippLib
from xmipp3.convert import writeSetOfParticles, locationToXmipp

import math
import itertools
import functools
import collections
import random
import pywt
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

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
        form.addParam('minClassesPerDirection', IntParam, label="Minumum number of classes per direction", 
                      validators=[GT(0)], default=1,
                      help='Required number of classes per solid angle.')
        form.addParam('keepBestDirections', IntParam, label="Keep best directions (%)", 
                      validators=[Range(0, 100)], default=30,
                      help='Percentage of directions to be used')

        form.addSection(label='Image comparison')
        form.addParam('imageDistanceMetric', EnumParam, label='Image distance metric',
                      choices=['Pearson', 'Wasserstein'], default=1,
                      help='Distance metric used when comparing image pairs.')
        form.addParam('considerMirrors', BooleanParam, label='Consider mirrors', default=False,
                      help='Consider image mirrors when aligning the images')
        form.addParam('mask', PointerParam, label="Mask", 
                      pointerClass='Mask', allowsNull=True,
                      help='Mask used when comparing image pairs')

        form.addSection(label='Graph building')
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
        form.addParam('weightTransformFunction', EnumParam, label='Weight Transform Function',
                      choices=['Staircase', 'Likelihood'], default=0,
                      help='TODO')
        form.addParam('weightTransformFunctionLocality', EnumParam, label='Transform locality',
                      choices=['Global', 'Local', 'Combine'], default=0, 
                      help='TODO')
        group = form.addGroup('Weight transform percentiles', 
                help='TODO')
        group.addParam('weightTransformPercentileMin', FloatParam, label='Minimum',
                       validators=[Range(0, 100)], default=10)
        group.addParam('weightTransformPercentileLower', FloatParam, label='Lower',
                       validators=[Range(0, 100)], default=20)
        group.addParam('weightTransformPercentileUpper', FloatParam, label='Upper',
                       validators=[Range(0, 100)], default=80)
        group.addParam('weightTransformPercentileMax', FloatParam, label='Maximum',
                       validators=[Range(0, 100)], default=90)
        form.addParam('transformWeightsCosine', BooleanParam, label='Cosine distance', 
                       default=True,
                       help='Apply the cosine distance to image pairs')
        
        form.addSection(label='Graph cut')
        form.addParam('graphCutLevels', IntParam, label='Cut levels', 
                      validators=[GE(1)], default=2,
                      help='Number of recursive partitions to perform. This will lead to '
                      'approximately 2^(N-1) classes')
        form.addParam('graphCutPadding', FloatParam, label='Padding (%)', 
                      validators=[Range(0, 100)], default=25, expertLevel=LEVEL_ADVANCED,
                      help='Amount of extra vertices added to allow unbalanced partitions.')

        form.addParallelSection(threads=mp.cpu_count(), mpi=0)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Create the thread pool
        if self.numberOfThreads > 1:
            self._threadPool = ThreadPoolExecutor(max_workers=int(self.numberOfThreads))

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('filterInputStep')
        self._insertFunctionStep('computeAngularDistancesStep')
        self._insertFunctionStep('computeImagePairsStep')
        self._insertFunctionStep('computeImageComparisonsStep')
        self._insertFunctionStep('computeWeightsStep')
        self._insertFunctionStep('partitionGraphStep')
        self._insertFunctionStep('reconstructVolumesStep')
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        # Read input data
        directionalClasses = self.directionalClasses.get()
        inputMask = self.mask.get()

        # Convert data
        classes = self._convertClasses2D(directionalClasses)
        images = self._convertClassRepresentatives(classes)
        directionIds = self._calculateDirectionIds(classes)
        compareMask = self._convertCompareMask(inputMask, images)

        # Write direction IDs to images in ClassId field
        for image, directionId in zip(images, directionIds):
            image.setClassId(directionId)

        # Save output data
        self._writeInputImages(images)
        self._writeCompareMask(compareMask)
        self._defineSourceRelation(self.directionalClasses, self._getInputImages())

    def filterInputStep(self):
        # Read input data
        images = self._readInputImages()
        minClassesPerDirection = self.minClassesPerDirection.get()
        bestDirectionsPercentage = self.keepBestDirections.get()

        # Select the input images
        selection = np.array(list(map(Particle.isEnabled, images)), dtype=bool)
        selection = self._calculateDirectionSizeSelection(images, minClassesPerDirection, selection)
        selection = self._calculateDirectionDisparitySelection(images, bestDirectionsPercentage, selection)

        # Filter the images
        selectedImages = images[selection]
        discardedImages = images[~selection]

        print(f'Using {len(selectedImages)} projections out of {len(images)} '\
              f'({100*len(selectedImages)/len(images)}%)')
        
        # Save output data
        self._writeSelectedImages(selectedImages)
        self._writeDiscardedImages(discardedImages)
        self._defineSourceRelation(self._getInputImages(), self._getSelectedImages())
        self._defineSourceRelation(self._getInputImages(), self._getDiscardedImages())

    def computeAngularDistancesStep(self):
        # Read input data
        images = self._readSelectedImages()
        considerAntipodes = self._getConsiderMirrors()

        # Perform the computation
        distances = self._calculateAngularDistanceMatrix(images, considerAntipodes)

        # Save output data
        self._writeAngularDistances(distances)

    def computeImagePairsStep(self):
        # Read input data
        distances = self._readAngularDistances()
        maxAngularDistance = self._getMaxAngularDistance(distances)
        maxNeighbors = self._getMaxNeighbors()

        # Compute the image pairs to be considered
        pairs = self._calculateImagePairs(distances, maxAngularDistance, maxNeighbors)
        print(f'Considering {len(pairs)} image pairs with an '
              f'angular distance of up to {np.degrees(maxAngularDistance)} deg')

        # Write the result
        self._writeImagePairs(pairs)

    def computeImageComparisonsStep(self):
        # Read input data
        images = self._readSelectedImages()
        pairs = self._readImagePairs()

        # Compute the comparisons
        comparisons, differences = self._computeComparisons(images, pairs)

        # Convert to matrix form
        comparisonMatrix = self._calculateComparisonMatrix(images, pairs, comparisons)
        
        # Save output data
        self._writeComparisonDifferences(differences)
        self._writeImageComparisons(comparisonMatrix)

    def computeWeightsStep(self):
        # Read input data
        images = self._readSelectedImages()
        distances = self._readAngularDistances()
        pairs = self._readImagePairs()
        comparisons = self._readImageComparisons()
        transformFunc = self._getWeightTransformFunc()

        weights = transformFunc(comparisons)

        if self.transformWeightsCosine:
            weights = self._calculateWeightsCosine(weights)

        # Save output data
        self._writeWeights(weights)
        self._writeWeightMetaData(images, pairs, distances, comparisons, weights)

    def partitionGraphStep(self):
        # Read the input data
        images = self._readSelectedImages()
        adjacency = self._readWeights()
        directionIds = self._getDirectionIds(images)
        nLevels = self.graphCutLevels.get()
        padding = self.graphCutPadding.get() / 100
        MIN_SIZE = 32

        # Convert the adjacency to a sparse representation
        graph = sparse.csr_matrix(adjacency)
        
        # Obtain information about the graph
        nComponents, labels = csgraph.connected_components(adjacency)
        print(f'The unpartitioned graph has {nComponents} components')

        # Merge insignificant classes and do not consider them for partitioning
        labels = self._discardSmallClasses(labels, threshold=MIN_SIZE)
        
        # Partition the graph
        labels = self._partitionGraphMultilevel(
            graph, 
            labels, 
            directionIds, 
            nLevels=nLevels, 
            padding=padding, 
            minSize=MIN_SIZE
        )

        # Write the result
        self._writePartitionLabels(labels)

    def reconstructVolumesStep(self):
        # Read the input data
        images = self._readSelectedImages()
        labels = self._readPartitionLabels(-1)

        # Reconstruct a volume for each top-level partition
        for cls in np.unique(labels):
            selection = labels == cls
            partitionImages = images[selection]
            path = self._getPartitionVolumeFileName(cls)
            self._reconstructVolume(path, partitionImages)

    def createOutputStep(self):
        # Read input data
        images = self._getSelectedImages()
        labels = self._readPartitionLabels(-1)

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

    def _getMapFunc(self):
        pool = self._getThreadPool()
        return pool.map if pool is not None else map

    def _getImageAlignFunc(self):
        considerMirrors = self._getConsiderMirrors()
        mask = self._readCompareMask()
        return self.ImageAligner(considerMirrors, mask)

    def _getImageCompareFunc(self, shape):
        options = [
            self.PearsonImageComparator(),
            self.WassersteinImageComparator(math.ceil(math.log2(max(shape))) + 1, 'sym5'),
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

    def _getConsiderMirrors(self):
        return self.considerMirrors.get()

    def _getWeightInterpolationFunc(self):
        min = self.weightTransformPercentileMin.get()
        lower = self.weightTransformPercentileLower.get()
        upper = self.weightTransformPercentileUpper.get()
        max = self.weightTransformPercentileMax.get()

        options = [
            functools.partial(self._calculateWeightsInterpolationStaircase, min=min, lower=lower, upper=upper, max=max),
            self._calculateWeightInterpolationLogLikelihood,
        ]
        selection = options[self.weightTransformFunction.get()]
        return selection

    def _getWeightTransformFunc(self):
        interpFunc = self._getWeightInterpolationFunc()

        options = [
            interpFunc,
            functools.partial(self._calculateWeightsInterpolationLocal, interpFunc=interpFunc),
            functools.partial(self._calculateWeightsInterpolationMixedLocal, interpFunc=interpFunc),
        ]
        selection = options[self.weightTransformFunctionLocality.get()]

        return selection

    def _setSetOfImages(self, name, images):
        self._insertChild('_'+name, images)

    def _getSetOfImages(self, name):
        return getattr(self, '_'+name)

    def _writeSetOfImages(self, name, images):
        result = self._createSetOfParticles('_'+name)

        if len(images):
            result.setAcquisition(images[0].getAcquisition())
            result.setSamplingRate(images[0].getSamplingRate())
            result.setHasCTF(images[0].hasCTF())
            result.setAlignmentProj()

        for image in images:
            result.append(image)
        
        self._setSetOfImages(name, result)

    def _readSetOfImages(self, name):
        images = self._getSetOfImages(name)

        result = np.empty(len(images), dtype=object)
        for i, img in enumerate(images):
            result[i] = img.clone()

        return result

    def _writeInputImages(self, images):
        self._writeSetOfImages('inputImages', images)
    
    def _getInputImages(self):
        return self._getSetOfImages('inputImages')

    def _readInputImages(self):
        return self._readSetOfImages('inputImages')

    def _writeSelectedImages(self, images):
        self._writeSetOfImages('selectedImages', images)

    def _getSelectedImages(self):
        return self._getSetOfImages('selectedImages')
    
    def _readSelectedImages(self):
        return self._readSetOfImages('selectedImages')

    def _writeDiscardedImages(self, images):
        self._writeSetOfImages('discardedImages', images)
    
    def _getDiscardedImages(self):
        return self._getSetOfImages('discardedImages')
    
    def _readDiscardedImages(self):
        return self._readSetOfImages('discardedImages')

    def _writeMatrix(self, path, x, fmt='%s'):
        # Determine the format
        np.savetxt(path, x, delimiter=',', fmt=fmt) # CSV

    def _readMatrix(self, path, dtype=float):
        return np.genfromtxt(path, delimiter=',', dtype=dtype) # CSV

    def _writeNdarrayAsStack(self, path, arr, imageObj=xmippLib.Image()):
        imageObj.setData(arr)
        imageObj.write(path)

    def _readNdarrayAsStack(self, path, imageObj=xmippLib.Image()):
        imageObj.read(path)
        return imageObj.getData()

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

    def _writeImagePairs(self, pairs):
        assert(pairs.dtype==int)
        self._storeMatrix('pairs', pairs, fmt='%d')

    def _readImagePairs(self):
        return self._loadMatrix('pairs', dtype=int)

    def _getComparisonDifferencesStackFileName(self):
        return self._getExtraPath('differences.mrcs')

    def _writeComparisonDifferences(self, differences):
        # Reshape the given array to behave like a stack
        differences = np.stack(differences)
        differences.reshape(len(differences), 1, *(differences[0].shape))

        # Write it
        path = self._getComparisonDifferencesStackFileName()
        self._writeNdarrayAsStack(path, differences)

    def _writeImageComparisons(self, comparisons):
        assert(comparisons.dtype==float)
        self._storeMatrix('image_comparisons', comparisons, fmt='%.8f')
    
    def _readImageComparisons(self):
        return self._loadMatrix('image_comparisons', dtype=float)

    def _getWeightMetaDataFileName(self):
        return self._getExtraPath('weights.xmd')

    def _writeWeightMetaData(self, images, pairs, distances, comparisons, weights):
        md = emlib.MetaData()
        differencesStackFn = self._getComparisonDifferencesStackFileName()

        for i, (idx0, idx1) in enumerate(pairs):
            # Get the data for the indices
            imgDiff = Image(location=(i+1, differencesStackFn))
            img0 = images[idx0]
            img1 = images[idx1]
            distance = distances[idx0, idx1]
            comparison = comparisons[idx0, idx1]
            weight = weights[idx0, idx1]

            # Create a row with both images and the comparison
            objId = md.addObject()
            row = emlib.metadata.Row()
            row.setValue(emlib.MDL_IMAGE_REF, locationToXmipp(*img0.getLocation()))
            row.setValue(emlib.MDL_IMAGE, locationToXmipp(*img1.getLocation()))
            row.setValue(emlib.MDL_IMAGE_RESIDUAL, locationToXmipp(*imgDiff.getLocation()))
            row.setValue(emlib.MDL_ANGLE_DIFF, distance)
            row.setValue(emlib.MDL_CORRELATION_IDX, comparison)
            row.setValue(emlib.MDL_WEIGHT, weight)
            row.writeToMd(md, objId)

        # Write to file
        path = self._getWeightMetaDataFileName()
        md.write(path)

    def _writeWeights(self, weights):
        assert(weights.dtype==float)
        self._storeMatrix('weights', weights, fmt='%.8f')
    
    def _readWeights(self):
        return self._loadMatrix('weights', dtype=float)

    def _writePartitionLabels(self, labels):
        assert(labels.dtype==np.uint)
        self._storeMatrix('partitions', labels, fmt='%d')

    def _readPartitionLabels(self, level=None):
        result = self._loadMatrix('partitions', dtype=np.uint)

        # Ensure the data is 2D
        if result.ndim < 2:
            result.shape = (1, len(result))
        assert(result.ndim==2)

        # Select a level if requested
        if level is not None:
            result = result[level]

        return result

    def _getPartitionVolumeFileName(self, cls):
        return self._getExtraPath(f'partition_{cls:04d}.vol')

    def _readPartitionVolume(self, cls, imageObj=xmippLib.Image()):
        path = self._getPartitionVolumeFileName(cls)
        return self._readNdarrayAsStack(path, imageObj=imageObj)

    def _readPartitionVolumes(self, classes):
        result = np.empty(len(classes), dtype=object)

        imgObj = xmippLib.Image() # Cache to avoid reallocations
        for i, cls in enumerate(classes):
            result[i] = self._readPartitionVolume(cls, imageObj=imgObj)

        return result

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
        return np.array(ids, dtype=np.uint)

    def _calculateDirectionSizeSelection(self, images, minClassesPerDirection, mask):
        if minClassesPerDirection > 1:
            directionIds = self._getDirectionIds(images)
            repetitions = collections.Counter(directionIds[mask]) # Only count where selected
            selection = np.array(list(map(
                lambda directionId : repetitions[directionId] >= minClassesPerDirection, 
                directionIds
            )))
            mask = np.logical_and(mask, selection)
        
        return mask

    def _calculateDirectionDisparitySelection(self, images, keepBestDirections, mask):
        if keepBestDirections < 100:
            directionIds = self._getDirectionIds(images)

            def selectDirectionImages(directionId):
                selection = np.logical_and(mask, directionIds==directionId)
                return np.argwhere(selection).T[0]

            def computeDirectionImagePairs(selection):
                return list(itertools.combinations(selection, r=2))

            # Select image pairs for each direction
            uniqueDirections = np.arange(max(directionIds)+1)
            assert(np.array_equal(uniqueDirections, np.unique(directionIds)))
            directionImagePairs = list(map(computeDirectionImagePairs, map(selectDirectionImages, uniqueDirections)))

            # Flatten the image pair list
            allImagePairs = sum(directionImagePairs, [])

            # Compute comparisons among them
            comparisons, _ = self._computeComparisons(images, allImagePairs)
            assert(len(comparisons) == len(allImagePairs))

            # For all directions select the most dissimilar pair (lowest value)
            directionDisparities = np.zeros_like(uniqueDirections, dtype=float)
            for directionId, pairs in enumerate(directionImagePairs):
                disparity = 1
                for pair in pairs:
                    disparity = min(disparity, comparisons[allImagePairs.index(pair)])
                directionDisparities[directionId] = disparity
                
            # Compute the threshold
            threshold = np.percentile(directionDisparities, keepBestDirections)

            # Select only the directions that have a disparity below the threshold
            selection = np.array(list(map(
                lambda id: directionDisparities[id] <= threshold, 
                directionIds,
            )))
            mask = np.logical_and(mask, selection)

        return mask

    def _convertCompareMask(self, mask, images, dtype=bool):
        if mask is not None:
            return xmippLib.Image(mask.getLocation()).getData().astype(dtype)
        else:
            dimensions = images[0].getDim()
            return np.ones(dimensions[0:2], dtype=dtype) 

    def _getDirectionIds(self, images):
        return np.array(list(map(Particle.getClassId, images)), dtype=np.uint)

    def _calculateDirectionalLabels(self, directionIds):
        result = np.zeros_like(directionIds)

        # Obtain the frequencies of each direction
        counter = collections.Counter(directionIds)

        cyclicLabel = itertools.cycle(range(2))
        for i, directionId in enumerate(directionIds):
            if counter[directionId] > 1:
                # Count items with the same id before it
                label = np.count_nonzero(directionIds[:i] == directionId)

            else:
                # Assign a cyclic label to it
                label = next(cyclicLabel)

            result[i] = label

        return result

    def _getProjectionDirection(self, image):
        # Multiply by the unit z vector, as projection is performed in this direction
        return image.getTransform().getRotationMatrix()[:, 2]

    def _calculateAngularDistance(self, image0, image1, considerAntipodes):
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
        if considerAntipodes:
            c = abs(c)

        # Clip to [0, 1] as floating point errors may lead to values slightly 
        # outside of the domain of acos
        c = min(c, 1.0)

        # Obtain the angle from the cosine
        angle = math.acos(c)

        return angle

    def _calculateAngularDistanceMatrix(self, images, considerAntipodes):
        """ Given a set of images, it computes the angular
            distances among all pairs. The result is represented
            by a symmetrical matrix with the angles in radians
        """
        # Compute a symmetric matrix with the angular distances.
        # Do not compute the diagonal, as the distance with itself is zero.
        distances = np.zeros(shape=(len(images), )*2)
        for idx0, idx1 in itertools.combinations(range(len(images)), r=2):
            # Write the result on symmetrical positions
            angle = self._calculateAngularDistance(images[idx0], images[idx1], considerAntipodes)
            distances[idx0, idx1] = angle
            distances[idx1, idx0] = angle

        # Ensure that the result matrix is symmetrical and with 0s in the diagonal
        assert(np.array_equal(distances, distances.T))
        assert(np.all(np.diagonal(distances) == 0))
        return distances

    def _calculateImagePairs(self, distances, maxDistance, maxNeighbors):
        """ Computes the indices of a matrix where to compute
            the comparisons. In order to do so, it considers
            a distance threshold and a maximum number of comparisons
            per each row
        """
        # Consider only distances closer than the threshold
        pairs = distances <= maxDistance

        # Limit the number of comparisons to be computed
        if maxNeighbors is not None:
            indices = np.argsort(distances, axis=1)
            for pairRow, ordering in zip(pairs, indices):
                pairRow[ordering[maxNeighbors+1:]] = False

        # Remove comparisons with itself
        np.fill_diagonal(pairs, False)

        return np.argwhere(pairs)

    def _computeComparisons(self, images, pairs):
        """ Computes the comparisons between image pairs
            defined by pairs array
        """
        def minmax(iterable):
            return min(iterable), max(iterable)

        # Get the functions
        alignFunc = self._getImageAlignFunc()
        compareFunc = self._getImageCompareFunc(images[0].getDim())
        mapFunc = self._getMapFunc()

        # Convert to xmipp images in order to use functions that deal with its data
        xmippImages = self._convertToXmippImages(images)
        
        # Select unique image pairs to be processed.
        uniquePairs = set(map(minmax, pairs))
        uniqueImagePairs = map(lambda pair : (xmippImages[pair[0]], xmippImages[pair[1]]), uniquePairs)

        # Align images
        alignedImages = mapFunc(alignFunc, *zip(*uniqueImagePairs))

        # Compare images
        computedComparisons = mapFunc(compareFunc, *zip(*alignedImages))
        pairComparisons = dict(zip(uniquePairs, computedComparisons))

        # Write the computed comparisons to the resulting matrix
        comparisons = np.zeros(len(pairs), dtype=float)
        differences = np.empty(len(pairs), dtype=object)
        for i, pair in enumerate(pairs):
            comparisons[i], differences[i] = pairComparisons[minmax(pair)]

        # Ensure that the result matrix is in [0, 1]
        assert(np.all(comparisons >= 0.0))
        assert(np.all(comparisons <= 1.0))
        return comparisons, differences

    def _calculateComparisonMatrix(self, images, pairs, comparisons):
        result = np.zeros((len(images), )*2, dtype=float)
        for pair, comparison in zip(pairs, comparisons):
            result[tuple(pair)] = comparison
        return result

    def _calculateDistancePonderation(self, distances, maxDistance=None):
        """ Gives each of the distances a ponderation to stimulate
            the similarity metric for images that are far away.
            For this purpose, the Cumulative Normal Distribution Function
            is used, so that images that are close are penalized 
            with a ponderation of 0.5, whilst images that are maxDistance 
            appart will receive a ponderation of 0.9999 (virtually 1)
        """

        # Set the maxDistance if not set
        if maxDistance is None:
            maxDistance = max(distances)

        # Calculate the typical deviation assuming a quasi-1 probability for distance being
        # smaller than maxDistance. PPF (Percent Point Function) is the same as Inverse CDF
        pMaxDistance = 1-1e-4 # Probability for distance being smaller than maxDistance.
        sigma = maxDistance / stats.norm.ppf(pMaxDistance, loc=0, scale=1)
        assert(np.isclose(stats.norm.cdf(maxDistance, loc=0, scale=sigma), pMaxDistance))

        # Calculate the Cumulative Normal Distribution Function for the distances.
        result = stats.norm.cdf(distances, loc=0, scale=sigma)

        assert(np.all(result >= 0))
        assert(np.all(result <= 1))
        return result

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

    def _calculateWeightsInterpolationStaircase(self, comparisons, min, lower, upper, max):
        values = comparisons[np.nonzero(comparisons)]
        x00, x01, x10, x11 = np.percentile(values, [min, lower, upper, max])
        
        step0 = np.nonzero(np.logical_and(comparisons, comparisons<=x01))
        step1 = np.nonzero(np.logical_and(comparisons, comparisons>=x10))

        result = np.zeros_like(comparisons)
        result[step0] = -self._smoothStep(comparisons[step0], x01, x00)
        result[step1] = +self._smoothStep(comparisons[step1], x10, x11)

        return result

    def _calculateWeightInterpolationLogLikelihood(self, comparisons, mu0=0.3, mu1=0.7):
        """ Calculate the weight according to Neyman-Pearson hypothesis test. 
            Let H1 be the hypotheisis of two images being similar and H0 disimilar.
            Assuming a normal distribution of the distance function:
            C ~ N(mu0, sigma) if H0, N(mu1, sigma) otherwise
            We can define the likelihood function
            L(x) = f(x|H1) / f(x|H0)
            And test for it being >= 1 which is equivalent of testing the possitiveness
            of its logatithm:
            T(x) = ln L(x) = ln f(x|H1) - ln f(x|H0)
            which for a normal function is equivalent to a linear function defined as:
            with:
            T(x) = (mu1 - mu0) / sigma^2 * (x - (mu0 + mu1)/2)
            As we care only about the relative values of T(x) we will ignore the gain
            applied to it. 
            We will estimate (mu0 + mu1) / 2 as the median of x
        """
        result = np.zeros_like(comparisons)
        indices = np.nonzero(comparisons)

        x = comparisons[indices]
        y = x - np.mean(x)

        result[indices] = y
        return result

    def _calculateWeightsInterpolationLocal(self, comparisons, interpFunc):
        weights = np.zeros_like(comparisons)
        for weightRow, compRow in zip(weights, comparisons):
            weightRow[:] = interpFunc(compRow)
        return weights
    
    def _calculateWeightsInterpolationMixedLocal(self, comparisons, interpFunc):
        g = interpFunc(comparisons)
        l = self._calculateWeightsInterpolationLocal(comparisons, interpFunc)
        return (g+l)/2

    def _calculateWeightsCosine(self, comparisons):
        # Consider comparison with itself to be 1
        v = comparisons.copy()
        np.fill_diagonal(v, 1)

        # Normalize each row
        v /= np.linalg.norm(v, axis=1)[:,np.newaxis]
        assert(np.allclose(np.linalg.norm(v, axis=1), 1))

        # Compute the dot product (cosine of the normalzied vectors)
        result = np.dot(v, v.T)

        # Do not consider comparisons with itself
        np.fill_diagonal(result, 0)
        
        assert(np.array_equal(result, result.T))
        return result
    
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

        # In order to use eigsh, the graph laplacian must be PSD.
        # This condition is only satisfied if the graph has 
        # possitive edges
        graph = graph.maximum(0)

        # Compute the laplacian matrix, this is the degree diagonal
        # matrix minus the adjacency
        l = csgraph.laplacian(graph)
        nVertices = l.shape[0]

        # Decompose the laplacian matrix of the graph into 2
        # eigenvectors with the smallest eigenvalues.
        # The first eigenvector is known, provide it as an
        # starting point
        nMaxIter = 1e3 * nVertices
        v0 = np.full(nVertices, 1/math.sqrt(nVertices))
        values, vectors = linalg.eigsh(l, k=2, which='SM', v0=v0, maxiter=nMaxIter)

        # Eigenvalues should be in increasing magnitude order
        assert(np.array_equal(np.abs(values), sorted(np.abs(values))))

        # The first eigenvalue should be 0 and its corresponding
        # eigenvector should be 1s normalized:
        # https://youtu.be/zZae_C2BU_4?t=2489
        assert(np.isclose(values[0], 0))
        assert(np.allclose(vectors[:,0], v0))

        # Fiedler vector corresponds to second eigenvector
        # with the smallest eigenvalue
        return vectors[:,1]

    def _calculateGraphSizes(self, labels):
        """ Computes the vertex count of each component of the graph
        """
        counter = collections.Counter(labels)
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

    def _partitionFiedlerVector(self, graph, fiedler, metric=None):
        """ Partition the graph into 2 components which
            minimize the given metric
        """
        if metric is not None:
            # Sort the fiedler vector so that it represents the cut priority of each vertex
            indices = np.argsort(fiedler)

            # Calculate the metric function values
            metricValues = self._calculateMetricValues(graph, indices, metric)

            # Calculate the cheapest cut
            cut = np.argmin(metricValues)
            labels = self._calculateCutLabels(indices, cut)

        else:
            labels = fiedler >= 0

        return labels

    def _partitionGraphSpectral(self, graph, metric=None):
        nVertices = graph.shape[0]

        if nVertices > 2:
            # Compute the fiedler vector
            fiedler = self._calculateFiedlerVector(graph)

            # Partition it
            labels = self._partitionFiedlerVector(graph, fiedler, metric)

        else:
            # 2 nodes. Trivial partition
            labels = np.arange(nVertices, dtype=np.uint)

        assert(nVertices == len(labels))
        return labels

    def _calculateKerninghanLinCostReductions(self, graph, labels):
        """ The cost reduction value (D) is defined for each
            vertex as the difference between the external (E) 
            and internal (I) costs (D=E-I). The external cost
            is the sum of all edges that emanate from a 
            particular vertex and go to a vertex that does not
            belong to the same partition. Similarly, the internal
            cost is the sum of all edges that stay inside the same
            component.
            http://users.ece.northwestern.edu/~haizhou/357/lec2.pdf#page=6
        """
        signs = 2*labels - 1
        result = graph.dot(signs)
        result *= -signs

        return result

    def _calculateKerninghanLinMaximumSwapGain(self, graph, delta, classA, classB):
        """ Maximizes the objective function Da + Db - 2*Cab where
            Da is the cost a cost reduction of a vertex "a" belonging
            to classA, Db is a cost reduction of a vertex belonging to 
            classB and Cab is the weight of the edge joining them. 
            This function returns the vertex pair to swap and its cost
        """

        # Compute the cost for all possible pairs in matrix form
        m = np.meshgrid(delta[classB], delta[classA]) # deltaA in columns, deltaB in rows
        w = graph[classA,:][:,classB] # Select the sub matrix with A as rows and B as columns
        gains = np.sum(m, axis=0) - 2*w # Da + Db - 2ab in (a, b) position

        # Maximize the cost function
        a, b = np.unravel_index(np.argmax(gains), gains.shape)
        gain = gains[a, b]
        pair = (classA[a], classB[b])
        return pair, gain

    def _calculateKerninghanLinPass(self, graph, labels):
        """ Kerninghan Lin algorithm is iterative. This function
            computes one iteration of the algorithm. It takes
            the grapg itself and the previous classification
            and it returns a vector representing vertices
            to be swapped and the gain of that swap.
        """
        N = len(labels)
        V2 = 2*np.array([1, -1])

        # Create a vertex set for each class
        classA = list(np.nonzero( labels)[0])
        classB = list(np.nonzero(~labels)[0])
        if len(classA) != len(classB):
            raise ValueError('Both classes must have the same size')

        # Calculate the cost reduction for each vertex
        delta = self._calculateKerninghanLinCostReductions(graph, labels)

        # Calculate the cost of swapping all nodes
        allSwaps = np.zeros((N//2, 2), dtype=np.uint)
        allCosts = np.zeros(N//2)
        for i in range(len(allCosts)):
            (a, b), g = self._calculateKerninghanLinMaximumSwapGain(graph, delta, classA, classB)

            # Do not consider these vertices anymore
            classA.remove(a)
            classB.remove(b)

            # Update the D values for next iteration
            # Da' = Da + 2Cxa - 2Cxb for x in A \ a
            # Db' = Db + 2Cxb - 2Cxa for x in B \ b
            deltaUpdate = graph[:,[a, b]].dot(V2) # 2Cxa - 2Cxb
            delta[classA] += deltaUpdate[classA]
            delta[classB] -= deltaUpdate[classB]

            # Store the results
            allSwaps[i,:] = a, b
            allCosts[i] = g
        
        # Ensure all was consumed
        assert(len(classA) == 0)
        assert(len(classB) == 0)

        # Integrate costs and take the maximum
        costFunction = np.cumsum(allCosts)
        k = np.argmax(costFunction)

        # Select the result
        swap = allSwaps[:k+1].flatten()
        cost = costFunction[k]

        return swap, cost
        
    def _partitionGraphKernighanLin(self, graph, labels=None, padding=0.0):
        """ Partitions the graph using the Kerninghan Lin
            algorithm. A initial partition can be provided
            to iteratively improve it. If it is not given, 
            it will compute it using the Fiedler approach
            This implementation is based on these slides:
            http://users.ece.northwestern.edu/~haizhou/357/lec2.pdf
        """
        EPSILON = 1e-6
        N = graph.shape[0]

        # Ensure the initial partition is defined
        if labels is None:
            labels = np.array([False]*(N//2) + [True]*(N-N//2))
            random.shuffle(labels)
        
        nVertices = 2*math.ceil(N*(1+padding)/2)
        assert(nVertices>=N)
        if nVertices > N:
            # Padding is required
            graph = graph.copy()
            graph.resize(nVertices, nVertices)
            
            padding0 = nVertices//2 - np.count_nonzero(~labels)
            padding1 = nVertices//2 - np.count_nonzero( labels)
            labels = np.append(labels, [False]*padding0 + [True]*padding1)

        assert(graph.shape[0] == len(labels))
        assert(graph.shape[1] == len(labels))
        assert(np.count_nonzero(labels) == np.count_nonzero(~labels))

        # Iteratively swap vertices among classes until it is not beneficial
        i = 0
        while True:
            # Obtain the indices to swap and its cost
            swap, gain = self._calculateKerninghanLinPass(graph, labels)

            # Print information
            print(f'KL iteration{i} gain: {gain}')
            i+=1

            # Swap the labels if necessary
            if gain > EPSILON:
                labels[swap] = ~labels[swap]
            else:
                break

        # Convert the partition
        labels = labels[:N].astype(np.uint)

        return labels

    def _partitionGraphMultilevel(self, graph, labels, directionIds, nLevels, padding, minSize=16):
        # Initialize the result matrix
        result = np.zeros((nLevels, len(labels)), dtype=np.uint)
        result[0,:] = labels

        for level in range(1, nLevels):
            prevLabels = result[level-1]
            nextLabels = np.zeros_like(prevLabels)

            componentCounter = 1
            for component in np.unique(prevLabels):
                # Build the adjacency matrix for the component
                selection = prevLabels == component

                if component == 0:
                    # Leave the zero component untouched
                    nextLabels[selection] = 0

                elif np.count_nonzero(selection) > minSize:
                    # Partition components larger than the threshold
                    componentGraph = graph[selection, :][:, selection]
                    componentDirectionalLabels = self._calculateDirectionalLabels(directionIds[selection]).astype(bool)
                    assert(componentGraph.shape[0] == componentGraph.shape[1])
                
                    # Partition it
                    componentPartition = self._partitionGraphKernighanLin(
                        componentGraph, 
                        componentDirectionalLabels,
                        padding
                    )

                    # Compute the labels
                    nextLabels[selection] = componentCounter + componentPartition
                    componentCounter += 2
                
                else:
                    # No partition
                    nextLabels[selection] = componentCounter
                    componentCounter += 1

            # Write the result of this level
            result[level,:] = nextLabels

        return result

    def _discardSmallClasses(self, labels, threshold=16):
        counter = collections.Counter(labels)

        result = np.empty_like(labels, dtype=np.uint)
        i = 1
        for label in np.unique(labels):
            # Transform the label according to its frequency
            if counter[label] < threshold:
                result[labels==label] = 0
            else:
                result[labels==label] = i
                i += 1
        
        return result

    def _reconstructVolume(self, path, particles):
        """ Given a list of particles, with angular assignment
            it reconstructs a volume from them
        """

        # Convert the particles to a xmipp metadata file
        fnParticles = path+'_particles.xmd'
        writeSetOfParticles(particles, fnParticles, alignType=ALIGN_PROJ)

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
        repCbk = self._getPartitionVolumeFileName
        loader = XmippProtSplitvolume.ClassesLoader(classification, repCbk)
        loader.fillClasses(result)

        return result

    def _createOutputVolumes(self, classes, suffix=''):
        result = self._createSetOfVolumes(suffix)
        result.setSamplingRate(classes.getImages().getSamplingRate())

        for cls in classes:
            volume = cls.getRepresentative().clone()
            volume.setObjId(cls.getObjId())
            result.append(volume)
        
        return result

    class ImageAligner:
        def __init__(self, considerMirrors, mask):
            self.alignFunc = xmippLib.Image.alignConsideringMirrors if considerMirrors else xmippLib.Image.align
            self.mask = mask
            
        def __call__(self, img0, img1):
            # Align img1 respect img0
            img1 = self._alignImages(img0, img1)
            img0 = img0.getData()
            img1 = img1.getData()

            # Mask both images to consider alignment wrapping
            img0 *= self.mask
            img1 *= self.mask

            return img0, img1
        
        def _alignImages(self, imgRef, img):
            return self.alignFunc(imgRef, img)

    class PearsonImageComparator:
        def __call__(self, img0, img1):
            # Compute the correlation index
            correlation = np.corrcoef(img0.flat, img1.flat)
            difference = img0 - img1

            # Clip the result to [0, 1]
            result = max(correlation, 0)

            assert(result >= 0)
            assert(result <= 1)
            return result, difference

    class WassersteinImageComparator:
        def __init__(self, nLevels, wavelet='sym5'):
            self.nLevels = nLevels
            self.wavelet = wavelet

        def __call__(self, img0, img1):
            # Based on:
            # https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/master/src/aspire/operators/wemd.py

            # Subtract the input images so that the mean is zero
            difference = img0 - img1*(np.mean(img0)/np.mean(img1))
            assert(np.isclose(np.mean(difference), 0))

            # Compute the Wavelet transform
            coefficients = self._computeWaveletCoefficients(difference)

            # Apply the weights to the coefficients
            weightedCoefficients = []
            weights = self._computeWeights(difference.ndim)
            for details, weight in zip(coefficients, weights):
                for detail in details.values():
                    weightedCoefficients.append(weight*detail.flatten())

            # Compute the Wasserstein distance
            allWeightedCoefficients = np.concatenate(weightedCoefficients)
            distance = np.linalg.norm(allWeightedCoefficients, ord=1) # Use the Manhattan norm

            # Map the distance to the [0, 1] range, considering 1 as equal
            # TODO consider using other methods
            factor = len(allWeightedCoefficients) * 16
            result = math.exp(-distance/factor)

            assert(result >= 0)
            assert(result <= 1)
            return result, difference

        def _computeWaveletCoefficients(self, residual):
            dwt = pywt.wavedecn(residual, self.wavelet, mode='zero', level=self.nLevels)
            coefficients = dwt[-1:0:-1] # All except the first one reversed

            assert(len(coefficients) == self.nLevels)
            return coefficients

        def _computeWeights(self, nDim):
            r = nDim/2 + 1  # 2 for 2D
            levels = np.arange(self.nLevels) # Reversed
            exponents = r*levels
            weights = 2 ** exponents

            assert(len(weights) == self.nLevels)
            return weights

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
            item.setClassId(cls)

        def _updateClass(self, item):
            location = self.representativeCallback(item.getObjId())
            item.setRepresentative(Volume(location=location))