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
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from scipy import sparse
from scipy import stats
from scipy import special
from scipy.sparse import csgraph
from scipy.sparse import linalg


class XmippProtSplitvolume(ProtClassify3D):
    """Split volume in two"""
    _label = 'split volume'
    _devStatus = BETA
    
    def __init__(self, **args):
        ProtClassify3D.__init__(self, **args)
        # self.stepsExecutionMode = STEPS_PARALLEL
        self._createFilenames()

    def _createFilenames(self):
        """ Centralize the names of the files. """
        classFmt='c%(cls)02d'
        suffixFmt='%(suffix)s'

        myDict = {
            'directions': 'directions.csv',
            'distances': 'distances.csv',
            'correlations': 'correlations.csv',
            'weights': 'weights.csv',
            'fiedler': 'fiedler.csv',
            'labels': 'labels.csv',
            'representative': f'{suffixFmt}_volume_{classFmt}.vol'
        }
        self._updateFilenamesDict(myDict)
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('directionalClasses', PointerParam, label="Directional classes", 
                      pointerClass='SetOfClasses2D', pointerCondition='hasRepresentatives', 
                      important=True, 
                      help='Select a set of particles with angles. Preferrably the output of a run of directional classes')
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
        form.addParam('considerAntipodes', BooleanParam, label='Consider antipodes',
                      default=True,
                      help='Due to the nature of the projections, antipodes can be considered to be '
                      'equivalent. This option toggles wether to consider values larger than 90º to be '
                      'actually closer.')
        form.addParam('maxAngularDistanceMethod', EnumParam, label='Maximum angular distance method',
                      choices=['Threshold', 'Percentile'], default=0,
                      help='Determines how the maximum angular distance is obtained.')
        form.addParam('maxAngularDistanceThreshold', FloatParam, label="Maximum angular distance threshold (deg)", 
                      validators=[Range(0, 180)], default=15, condition='maxAngularDistanceMethod==0',
                      help='Maximum angular distance value for considering the correlation of two classes. '
                      'Valid range: 0 to 180 deg')
        form.addParam('maxAngularDistancePercentile', FloatParam, label="Maximum angular distance percentile (%)", 
                      validators=[Range(0, 100)], default=10, condition='maxAngularDistanceMethod==1',
                      help='Maximum angular distance percentile for considering the correlation of two classes. '
                      'Valid range: 0 to 100%')
        form.addParam('maxCorrelations', IntParam, label="Maximum number of correlations", 
                      validators=[GE(0)], default=12,
                      help='Maximum number of correlations to consider for each directional class. Use 0 to disable this limit')
        #form.addParam('maxDegree', IntParam, label="Maximum degree of each node of the graph", 
        #              validators=[GE(0)], default=8, expertLevel=LEVEL_ADVANCED,
        #              help='Maximum out-degree of each node of the graph. Use 0 to disable degree limitation')
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
        self._insertFunctionStep('computeCorrelationStep')
        self._insertFunctionStep('computeWeightsStep')
        self._insertFunctionStep('computeEigenvectorsStep')
        self._insertFunctionStep('classifyGraphStep')
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        # Read input data
        directionalClasses = self._convertClasses2D(self.directionalClasses.get())
        minClassSize = self.minClassSize.get()
        minClassesPerDirection = self.minClassesPerDirection.get()
        keepBestDirections = self.keepBestDirections.get()

        # Obtain the directions IDs for the input data and the images
        directionIds = self._calculateDirectionIds(directionalClasses)
        images = np.array(list(map(Class2D.getRepresentative, directionalClasses)))

        # Mask the input according to the input parameters
        mask = np.ones(len(directionalClasses), dtype=bool)
        mask = self._calculateClassSizeMask(directionalClasses, minClassSize, mask)
        mask = self._calculateDirectionSizeSelectionMask(directionIds, minClassesPerDirection, mask)
        mask = self._calculateDirectionSelectionMask(directionIds, images, keepBestDirections, mask)
        print(f'Using {np.count_nonzero(mask)} images out of {len(images)} '\
              f'({100*np.count_nonzero(mask)/len(images)}%)')

        # Apply mask
        directionIds = directionIds[mask]
        images = images[mask]

        # Save the output data
        self._writeInputImages(images)
        self._writeDirectionIds(directionIds)

    def computeAngularDistancesStep(self):
        # Read input data
        images = self._readInputImages()

        # Perform the computation
        distances = self._calculateAngularDistanceMatrix(images)

        # Save output data
        self._writeAngularDistances(distances)

    def computeCorrelationStep(self):
        # Read input data
        images = self._readInputImages()
        directionIds = self._readDirectionIds()
        distances = self._readAngularDistances()
        maxAngularDistance = self._getMaxAngularDistance(distances)
        maxCorrelations = self.maxCorrelations.get() if self.maxCorrelations.get() > 0 else None

        # Reflect the distance for values larger than pi/2 when considering antipodes
        if self.considerAntipodes.get():
            distances = np.minimum(distances, math.pi-distances)

        # Compute the class pairs to be considered
        pairs = self._calculateCorrelationPairs(distances, directionIds, maxAngularDistance, maxCorrelations)
        print(f'Considering {len(pairs)} image pairs with an '
              f'angular distance of up to {np.degrees(maxAngularDistance)} deg')

        # Compute the correlation
        correlations = self._calculateCorrelationMatrix(images, pairs, self._getThreadPool())

        # Save output data
        self._writeCorrelations(correlations)

    def computeWeightsStep(self):
        # Read input data
        correlations = self._readCorrelations()
        symmetrize = {
            0: np.maximum, 
            1: np.minimum, 
            2: lambda x, y : (x+y)/2
        }[self.enforceUndirected.get()]

        # Calculate weights from correlations
        weights = self._calculateWeights(correlations, symmetrize)

        # Print information about the graph
        nComponents, _ = csgraph.connected_components(weights)
        print(f'The unpartitioned graph has {nComponents} components')

        # Save output data
        self._writeWeights(weights)

    def computeEigenvectorsStep(self):
        # Read input data
        adjacency = self._readWeights()
        graph = sparse.csr_matrix(adjacency)

        # Compute the fiedler vector
        fiedler = self._calculateFiedlerVector(graph)

        # Write the result
        self._writeFiedlerVector(fiedler)

    def classifyGraphStep(self):
        # Read the input data
        graph = self._readWeights()
        fiedler = self._readFiedlerVector()
        metric = {
            0: self._calculateGraphCutMetric,
            1: self._calculateRatioCutMetric,
            2: self._calculateNormalizedCutMetric,
            3: self._calculateQuotientCutMetric
        }[self.graphMetric.get()]

        # Classify the images
        labels = self._classifyFiedlerVector(graph, fiedler, metric)

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
        return result
    
    def _citations(self):
        pass

    def _summary(self):
        pass
    
    def _methods(self):
        pass

    #--------------------------- UTILS functions ---------------------------------------------------
    def _getThreadPool(self):
        return getattr(self, '_threadPool', None)

    def _getMaxAngularDistance(self, distances):
        method = self.maxAngularDistanceMethod.get()
        if method == 0:
            return np.radians(self.maxAngularDistanceThreshold.get())
        elif method == 1:
            return np.percentile(distances, self.maxAngularDistancePercentile.get())

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

    def _writeDirectionIds(self, directions):
        assert(directions.dtype==np.uint)
        self._writeMatrix(self._getExtraPath(self._getFileName('directions')), directions, fmt='%d')

    def _readDirectionIds(self):
        return self._readMatrix(self._getExtraPath(self._getFileName('directions')), dtype=np.uint)

    def _writeAngularDistances(self, distances):
        assert(distances.dtype==float)
        self._writeMatrix(self._getExtraPath(self._getFileName('distances')), distances, fmt='%.8f')

    def _readAngularDistances(self):
        return self._readMatrix(self._getExtraPath(self._getFileName('distances')), dtype=float)

    def _writeCorrelations(self, correlations):
        assert(correlations.dtype==float)
        self._writeMatrix(self._getExtraPath(self._getFileName('correlations')), correlations, fmt='%.8f')
    
    def _readCorrelations(self):
        return self._readMatrix(self._getExtraPath(self._getFileName('correlations')), dtype=float)

    def _writeWeights(self, weights):
        assert(weights.dtype==float)
        self._writeMatrix(self._getExtraPath(self._getFileName('weights')), weights, fmt='%f')
    
    def _readWeights(self):
        return self._readMatrix(self._getExtraPath(self._getFileName('weights')), dtype=float)

    def _writeFiedlerVector(self, fiedler):
        assert(fiedler.dtype==float)
        self._writeMatrix(self._getExtraPath(self._getFileName('fiedler')), fiedler, fmt='%f')

    def _readFiedlerVector(self):
        return self._readMatrix(self._getExtraPath(self._getFileName('fiedler')), dtype=float)

    def _writeLabels(self, labels):
        assert(labels.dtype==np.uint)
        self._writeMatrix(self._getExtraPath(self._getFileName('labels')), labels, fmt='%d')

    def _readLabels(self):
        return self._readMatrix(self._getExtraPath(self._getFileName('labels')), dtype=np.uint)

    def _calculateDirectionIds(self, classes):
        reprojections = list(map(lambda cls : cls.reprojection.getLocation(), classes))

        # Create a dictionary assotiating reprojections and its ids
        uniqueReprojections = set(reprojections)
        uniqueReprojectionIds = range(len(uniqueReprojections))
        reprojectionIds = dict(zip(uniqueReprojections, uniqueReprojectionIds))

        # Map the reprojections to its ids
        ids = list(map(reprojectionIds.__getitem__, reprojections))
        return np.array(ids, dtype=np.uint)

    def _calculateClassSizeMask(self, classes, sizePercentile, mask):
        sizes = list(map(len, classes))
        threshold = np.percentile(sizes, sizePercentile)
        mask = np.logical_and(mask, sizes >= threshold)
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

    def _calculateDirectionSelectionMask(self, directionIds, images, keepBestDirections, mask, threadPool=None):
        if keepBestDirections < 100:
            # Map function used to compute correlations in parallel
            mapFunc = threadPool.map if threadPool is not None else map

            # For all directions compute the correlation among its members
            directionDisparities = {}
            for direction in np.unique(directionIds):
                selection = np.logical_and(mask, directionIds==direction)
                if np.count_nonzero(selection) >= 2:
                    # Compute all the correlations among the image pairs
                    directionImages = map(xmippLib.Image, map(Particle.getLocation, images[selection]))
                    imagePairs = itertools.combinations(directionImages, r=2)
                    correlations = mapFunc(xmippLib.Image.correlationAfterAlignment, *zip(*imagePairs))

                    # Use the minimum correlation as the disparity
                    directionDisparities[direction] = min(correlations)
                else:
                    directionDisparities[direction] = 0

            # Compute the threshold
            threshold = np.percentile(list(directionDisparities.values()), 100-keepBestDirections)

            # Select only the directions that have a disparity above threshold
            return np.array(list(map(
                lambda enable, id: enable and directionDisparities[id] >= threshold, 
                mask,
                directionIds,
            )))

        else:
            # Nothing to do
            return mask

    def _convertClasses2D(self, classes):
        result = []
        for cls in classes:
            result.append(cls.clone())
        return result

    def _calculateAngularDistanceMatrix(self, classes):
        """ Given a set of images, it computes the angular
            distances among all pairs. The result is represented
            by a symmetrical matrix with the angles in radians
        """

        # Shorthands for some variables
        nClasses = len(classes)

        # Compute a symmetric matrix with the angular distances.
        # Do not compute the diagonal, as the distance with itself is zero.
        distances = np.zeros(shape=(nClasses, nClasses))
        for idx0, class0 in enumerate(classes):
            for idx1, class1 in enumerate(itertools.islice(classes, idx0)):
                # Obtain the rotation matrices of the directional classes
                rotMtx0 = class0.getTransform().getRotationMatrix()
                rotMtx1 = class1.getTransform().getRotationMatrix()

                # Compute the angular distance. Based on:
                # http://www.boris-belousov.net/2016/12/01/quat-dist/
                # Rotation matrix is supposed to be orthonormal. Therefore, transpose is cheaper than inversion
                # Clip is used inside the arccos because floating point errors may lead to values slightly 
                # outside of its domain
                assert(np.allclose(np.linalg.inv(rotMtx1), rotMtx1.T))
                diffMtx = np.matmul(rotMtx0, rotMtx1.T)
                distance = math.acos(np.clip((np.trace(diffMtx) - 1)/2, -1.0, +1.0))

                # Write the result on symmetrical positions
                distances[idx0, idx1] = distance
                distances[idx1, idx0] = distance

        # Ensure that the result matrix is symmetrical
        assert(np.array_equal(distances, distances.T))
        return distances

    def _calculateCorrelationPairs(self, distances, directionIds, maxDistance, maxCorrelations):
        """ Computes the indices of a matrix where to compute
            the correlations. In order to do so, it considers
            a distance threshold and a maximum number of correlations
            per each row
        """
        # Consider only distances closer than the threshold
        pairs = distances <= maxDistance

        # Limit the number of correlations to be computed
        for pairRow, distanceRow in zip(pairs, distances):
            indices = np.argsort(distanceRow)
            pairRow[indices[maxCorrelations+1:]] = False

        # Remove comparisons with itself. This is redundant,
        # as it would be done when removing by reprojections.
        # Hovever is cheaper to do like this.
        np.fill_diagonal(pairs, False)

        # Remove all the comparisons corresponding to its same reprojection
        for idx0, idx1 in np.argwhere(pairs):
            if directionIds[idx0] == directionIds[idx1]:
                pairs[idx0, idx1] = False

        return np.argwhere(pairs)

    def _convertToXmippImages(self, images):
        """Converts a list of images into xmipp images"""
        locations = map(Image.getLocation, images)
        result = list(map(xmippLib.Image, locations))
        return result

    def _calculateCorrelationMatrix(self, images, pairs, threadPool=None):
        """ Computes the correlations between image pairs
            defined by pairs array
        """
        correlations = np.zeros((len(images), )*2, dtype=float)
        
        def minmax(iterable):
            return min(iterable), max(iterable)

        # Convert to xmipp images in order to use correlation functions
        xmippImages = self._convertToXmippImages(images)
        
        # Select the image pairs to be processed.
        # Diagonal is directly written as it is trivial
        imagePairs = {}
        for pair in pairs:
            if pair[0] == pair[1]:
                # Correlation with itself is 1
                correlations[tuple(pair)] = 1.0

            else:
                # Refer to the lower triangle of the matrix
                pair = minmax(pair)

                # Add it if not defined
                if pair not in imagePairs:
                    imagePairs[pair] = (xmippImages[pair[0]], xmippImages[pair[1]])

        # Compute the correlations using parallelization if requested
        mapFunc = threadPool.map if threadPool is not None else map
        computedCorrelations = dict(zip(
            imagePairs.keys(),
            mapFunc(xmippLib.Image.correlationAfterAlignment, *zip(*imagePairs.values()))
        ))

        # Write the computed correlations to the resulting matrix
        for pair in pairs:
            correlations[tuple(pair)] = computedCorrelations[minmax(pair)]

        # Ensure that the result matrix is in [0, 1]
        assert(np.all(correlations >= 0.0))
        assert(np.all(correlations <= 1.0))
        return correlations

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
        
            # Calculate
            result = np.zeros_like(t)
            for n in range(N + 1):
                result += special.comb(N + n, n) * special.comb(2*N + 1, N - n) * (-t) ** n
            result *= t ** (N + 1)

        return result

    def _calculateWeights(self, correlations, symmetrizeFun, lower=50, upper=80):
        """ Given a matrix of correlations among neighboring 
            images, it computes the adjacency matrix with
            interger weights
            TODO: improve the algorithm
        """

        # Perform a Hermite interpolation of the correlations between the given percentiles
        weights = np.zeros_like(correlations)
        for weightRow, correlationRow in zip(weights, correlations):
            # Calculate lower and upper percentiles for the non zero correlations
            values = correlationRow[np.nonzero(correlationRow)]
            [minVal, maxVal] = np.percentile(values, [lower, upper])

            weightRow[:] = self._smoothStep(correlationRow, xMin=minVal, xMax=maxVal, N=1)

        # Ensure the graph is symmetric
        weights = symmetrizeFun(weights, weights.T)

        assert(np.all(weights >= 0))
        assert(np.all(weights <= 1))
        return weights

    def _calculateFiedlerVector(self, graph):
        """ Given a matrix with the eigenvectors with the smallest 
            eigenvalues of the laplacian matrix of a network, it 
            returns a classification of each vertex using a standing 
            wave metaphore. This consists in interpreting the eigenvectors 
            with the smallest eigenvalues as harmonics. Therefore, 
            each class has a unique combination of wave phases. Then 
            each vertex is classified according to its signs for each 
            eigenvector.
            https://people.csail.mit.edu/jshun/6886-s18/lectures/lecture13-1.pdf#page=11
            https://es.mathworks.com/help/matlab/math/partition-graph-with-laplacian-matrix.html
        """
        # In order to use eigsh, graph matrix needs to be symmetric (undirected graph)
        if not np.array_equal(graph.toarray(), graph.T.toarray()):
            raise ValueError('Input graph must be undirected')

        # Decompose the laplacian matrix of the graph into N
        # eigenvectors with the smallest eigenvalues
        l = csgraph.laplacian(graph, normed=True)
        values, vectors = linalg.eigsh(l, k=2, which='SM')
        assert(np.array_equal(values, sorted(values)))

        # Fiedler vector corresponds to second eigenvector
        # with the smallest eigenvalue
        return vectors[:,1]

    def _calculateGraphVolumes(self, graph, labels):
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
        counts = np.array(list(Counter(labels).values()))
        return graphCut * np.sum(1/counts) # Inverse sum of sizes

    def _calculateNormalizedCutMetric(self, graph, labels):
        graphCut = self._calculateGraphCutMetric(graph, labels)
        volumes = self._calculateGraphVolumes(graph, labels)
        return graphCut * np.sum(1/volumes) # Inverse sum of sizes

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

    def _classifyFiedlerVector(self, graph, fiedler, metric):
        """ Partition the graph into 2 components which
            minimize the given metric
        """
        # Sort the fiedler vector so that it represents the cut priority of each vertex
        indices = np.argsort(fiedler)

        # Calculate the metric function values
        metricValues = self._calculateMetricValues(graph, indices, metric)

        # Calculate the cheapest cut
        labels = self._calculateCutLabels(indices, np.argmin(metricValues))

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
        repCbk = lambda cls : self._getExtraPath(self._getFileName('representative', suffix=suffix, cls=cls))
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