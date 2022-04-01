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
from pwem.objects import Volume, Image, Particle

import xmippLib
from xmipp3.convert import writeSetOfParticles

import math
import itertools
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from collections import Counter

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
            'labels': 'labels.csv',
            'representative': f'{suffixFmt}_volume_{classFmt}.vol'
        }
        self._updateFilenamesDict(myDict)
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('directionalClasses', PointerParam, label="Directional classes", 
                      pointerClass='SetOfClasses2D',
                      important=True, 
                      help='Select a set of particles with angles. Preferrably the output of a run of directional classes')
        form.addParam('minClassesPerDirection', IntParam, label="Minumum number of classes per direction", 
                      validators=[GT(0)], default=1, expertLevel=LEVEL_ADVANCED,
                      help='Required number of classes per solid angle.')

        form.addSection(label='Graph building')
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
        form.addParam('maxNeighbors', IntParam, label="Maximum number of neighbors", 
                      validators=[GT(0)], default=8,
                      help='Number of neighbors to consider for each directional class.')
        form.addParam('enforceUndirected', BooleanParam, label="Enforce undirected", 
                      default=True, expertLevel=LEVEL_ADVANCED,
                      help='Enforce a undirected graph')

        form.addSection(label='Graph partition')
        form.addParam('graphPartitionMethod', EnumParam, label="Graph partition method", 
                      choices=['Spectral',  'Minimum cut'], default=0,
                      help='The method used for partitioning the graph')
        form.addParam('graphPartitionCount', IntParam, label="Minimum number of components", 
                      validators=[GE(2)], default=2,
                      help='Minimum number of classes to obtain')
        form.addParallelSection(threads=mp.cpu_count(), mpi=0)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Create the thread pool
        if self.numberOfThreads > 1:
            self.threadPool = ThreadPool(processes=int(self.numberOfThreads))

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('computeAngularDistancesStep')
        self._insertFunctionStep('computeCorrelationStep')
        self._insertFunctionStep('computeWeightsStep')
        self._insertFunctionStep('graphPartitionStep')
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        directionalClasses = self.directionalClasses.get()

        # Calculate the ids of the directions of each class
        directionIds = self._calculateDirectionIds(directionalClasses)

        # Create a mask for selecting input classes
        minClassesPerDirection = self.minClassesPerDirection.get()
        selectionMask = self._calculateDirectionSelectionMask(directionIds, minClassesPerDirection)

        # Apply mask
        images = self._convertClasses2DRepresentatives(directionalClasses, selectionMask)
        if selectionMask is not None:
            directionIds = directionIds[selectionMask]
        assert(len(images) == len(directionIds))

        # Save the output data
        self._writeInputImages(directionalClasses, images)
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

        # Compute the class pairs to be considered
        pairs = self._calculatePairMatrix(distances, directionIds, maxAngularDistance)
        print(f'Considering {np.count_nonzero(pairs)} image pairs with an '
              f'angular distance of up to {np.degrees(maxAngularDistance)} deg')

        # Compute the correlation
        correlations = self._calculateCorrelationMatrix(images, pairs, self._getThreadPool())

        # Save output data
        self._writeCorrelations(correlations)

    def computeWeightsStep(self):
        # Read input data
        correlations = self._readCorrelations()

        # Calculate weights from correlations
        weights = self._calculateWeights(correlations)

        # Limit the number of neighbors
        nNeighbors = self.maxNeighbors.get()
        for row in weights:
            # Delete the lowest correlations and leave only "nNeighbors"
            indices = np.argsort(row)
            row[indices[:-nNeighbors]] = 0

        # Make the graph undirected if requested
        isUndirected = self.enforceUndirected.get()
        if isUndirected:
            weights = np.maximum(weights, weights.T)

        # Print information about the graph
        nComponents, _ = self._getGraphComponents(weights)
        print(f'The unpartitioned graph has {nComponents} components')

        # Save output data
        self._writeWeights(weights)

    def graphPartitionStep(self):
        # Read input data
        adjacency = self._readWeights()
        graph = sparse.csr_matrix(adjacency)
        
        # Partition the graph according to the selected method
        partitionFunctions = {
            0: self._spectralPartition,
            1: self._minimumCut
        }
        partitionFunction = partitionFunctions[self.graphPartitionMethod.get()]
        partitionCount = self.graphPartitionCount.get()
        labels = partitionFunction(graph, partitionCount)
        
        # Save the output
        self._writeLabels(labels)

    def createOutputStep(self):
        # Read input data
        images = self._getInputImages()
        labels = self._readLabels()

        # Create the output elements
        classes = self._createOutputClasses(images, labels, 'output')
        volumes = self._createOutputVolumes(classes, 'output')

        # Define the output
        sources = [images]
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

        if self.maxNeighbors.get() >= len(self.directionalClasses.get()):
            result.append('Comparison count should be less than the length of the input directional classes')

        return result
    
    def _citations(self):
        pass

    def _summary(self):
        pass
    
    def _methods(self):
        pass

    #--------------------------- UTILS functions ---------------------------------------------------
    def _getThreadPool(self):
        return getattr(self, 'threadPool', None)

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

    def _writeInputImages(self, classes, images):
        result = self._createSetOfParticles('input')
        result.copyInfo(classes.getImages())

        for image in images:
            result.append(image)
        
        self._setInputImages(result) # This needs to happen before
        self._defineSourceRelation(classes, result)

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
        assert(weights.dtype==np.uint)
        self._writeMatrix(self._getExtraPath(self._getFileName('weights')), weights, fmt='%d')
    
    def _readWeights(self):
        return self._readMatrix(self._getExtraPath(self._getFileName('weights')), dtype=np.uint)

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

    def _calculateDirectionSelectionMask(self, directionIds, minClassesPerDirection=1):
        if minClassesPerDirection > 1:
            repetitions = Counter(directionIds)
            return list(map(
                lambda id : repetitions[id] >= minClassesPerDirection, 
                directionIds
            ))
        else:
            return None

    def _convertClasses2DRepresentatives(self, classes, mask=None):
        """ Converts a set of directional classes 2d into its representatives 
        """
        result = []

        # Define the mask if not defined
        if mask is None:
            mask = [True]*len(classes)
        
        # Copy the class representatives where mask
        for cls, msk in zip(classes, mask):
            if msk:
                result.append(cls.getRepresentative().clone())

        assert(len(result) == np.count_nonzero(mask))
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

                # Note that antipodes are actually equivalent for projections. Therefore,
                # for distances larger than pi/2, the projections are getting closer
                # TODO: Speak with COSS about it
                distance = min(distance, math.pi-distance)

                # Write the result on symmetrical positions
                distances[idx0, idx1] = distance
                distances[idx1, idx0] = distance

        # Ensure that the result matrix is symmetrical
        assert(np.array_equal(distances, distances.T))
        return distances

    def _calculatePairMatrix(self, distances, directionIds, maxDistance):
        """ Computes a symmetrical matrix of booleans
            where image pairs set to true are closer than
            the given threshold. It also avoids comparisons
            among image pairs corresponding to the same 
            solid angle. Therefore, the diagonal is
            always set to false.
        """

        # Consider only distances closer than the threshold
        pairs = distances <= maxDistance

        # Remove comparisons with itself. This is redundant,
        # as it would be done when removing by reprojections.
        # Hovever is cheaper to do like this.
        np.fill_diagonal(pairs, False)

        # Remove all the comparisons corresponding to its same reprojection
        for idx0, idx1 in zip(*np.nonzero(pairs)):
            if directionIds[idx0] == directionIds[idx1]:
                pairs[idx0, idx1] = False

        return pairs

    def _convertToXmippImages(self, images):
        """Converts a list of images into xmipp images"""
        locations = map(Image.getLocation, images)
        result = list(map(xmippLib.Image, locations))
        return result

    def _calculateCorrelationMatrixOne(self, images, pairs):
        """ Calculates correlations among image pairs with a single thread
        """
        correlations = np.zeros_like(pairs, dtype=float)
        
        # Compute the symmetric matrix with the correlations
        for idx0, idx1 in zip(*np.nonzero(pairs)):
            if idx0 < idx1:
                # Calculate the corrrelation
                image0 = images[idx0]
                image1 = images[idx1]

                # Write it on symmetrical positions
                correlation = image0.correlationAfterAlignment(image1)
                correlations[idx0, idx1] = correlation
                correlations[idx1, idx0] = correlation

            elif idx0 == idx1:
                # Correlation with itself is 1
                correlations[idx0, idx0] = 1.0

        return correlations

    def _calculateCorrelationMatrixParallel(self, images, pairs, threadPool):
        """ Calculates correlations among image pairs with multiple threads
        """
        correlations = np.zeros_like(pairs, dtype=float)
        
        # Select the image pairs to be processed
        imagePairs = {}
        for idx0, idx1 in zip(*np.nonzero(pairs)):
            if idx0 < idx1:
                # Calculate the corrrelation
                image0 = images[idx0]
                image1 = images[idx1]

                imagePairs[(idx0, idx1)] = (image0, image1)

            elif idx0 == idx1:
                # Correlation with itself is 1
                correlations[idx0, idx0] = 1.0

        # Compute the correlations in parallel
        results = threadPool.starmap(xmippLib.Image.correlationAfterAlignment, imagePairs.values())
        assert(len(imagePairs) == len(results))

        # Write the results
        for (idx0, idx1), correlation in zip(imagePairs.keys(), results):
            correlations[idx0, idx1] = correlation
            correlations[idx1, idx0] = correlation

        return correlations

    def _calculateCorrelationMatrix(self, images, pairs, threadPool=None):
        """ Computes the correlations between image pairs
            defined by pairs mask matrix. Pairs should be 
            symmetrical. Therefore, the resulting matrix is 
            also symmetrical.
        """

        # Ensure that the pairs matrix is symmetrical
        assert(np.array_equal(pairs, pairs.T))

        # Convert to xmipp images in order to use correlation functions
        xmippImages = self._convertToXmippImages(images)

        # Compute the matrix with or without parallelization
        if threadPool:
            correlations = self._calculateCorrelationMatrixParallel(xmippImages, pairs, threadPool)
        else:
            correlations = self._calculateCorrelationMatrixOne(xmippImages, pairs)

        # Ensure that the result matrix is symmetrical and in [0, 1]
        assert(np.array_equal(correlations, correlations.T))
        assert(np.all(correlations >= 0.0))
        assert(np.all(correlations <= 1.0))
        return correlations

    def _calculateWeights(self, correlations):
        """ Given a matrix of correlations among neighboring 
            images, it computes the adjacency matrix with
            interger weights
            TODO: improve the algorithm
        """
        # Normalize respect the smallest nonzero correlation
        minVal = 0.9*np.min(correlations[np.nonzero(correlations)])
        maxVal = np.max(correlations)
        result = (correlations - minVal) / (maxVal - minVal)
        result = np.maximum(result, 0)

        # Square to emphasize large values
        result *= result

        # Scale to be representable by integers
        result *= 2**10

        assert(np.all(result >= 0))
        assert(np.all(result <= 2**10))
        return result.astype(np.uint)

    def _classifySpectrum(self, eigenVectors):
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
        nVertices, nVectors = eigenVectors.shape
        nVectorsLog2 = math.trunc(math.log2(nVectors))
        assert(2**nVectorsLog2==nVectors) # TODO generalize for non pow2 classifications

        # Iterate over the pow 2 indices, summing their weights in 
        # decreasing order (binary counting)
        labels = np.zeros(nVertices, dtype=np.uint)
        for i in range(1, nVectorsLog2+1):
            w = eigenVectors[:, 2**i-1] # Extract the eigenvector
            r = 2**(nVectorsLog2-i)
            labels += r*(w>=0).astype(np.uint)

        return labels

    def _spectralPartition(self, graph, componentCount):
        """ Performs a spectral partition of the graph into componentCount components
        """
        # Convert the graph into doubles, required for linear algebra operations
        graph = graph.astype(np.double)

        # In order to use eigsh, graph matrix needs to be symmetric (undirected graph)
        if not np.array_equal(graph.toarray(), graph.T.toarray()):
            raise ValueError('Input graph must be undirected')

        # Decompose the laplacian matrix of the graph into N
        # eigenvectors with the smallest eigenvalues. Then interpret these vectors as
        # "standing waves" in the graph and use them to partition it
        l = csgraph.laplacian(graph, normed=True)
        values, vectors = linalg.eigsh(l, k=componentCount, which='SM')
        assert(np.array_equal(values, sorted(values)))
        labels = self._classifySpectrum(vectors)

        return labels

    def _getSourceSinkVertices(self, graph, labels):
        """Selects the least connected pair of vertices in the biggest component
        TODO: Improve this algorithm
        """
        result = None

        # Partition the biggest component
        component = stats.mode(labels)[0]

        # Among all the connected vertices, get the least connected ones
        for pos in zip(*graph.nonzero()):
            if (not result or graph[pos] < graph[result]) and labels[pos[0]] == component:
                assert(labels[pos[1]] == component) # The destination vertex should also be in the same component
                result = pos

        return result

    def _getGraphComponents(self, graph):
        """ Returns the number of components of the graph
            and a array of labels which assigns a component
            to each vertex
        """
        nComponents, labels = csgraph.connected_components(graph)
        return nComponents, labels.astype(np.uint)

    def _calculateResidualGraph(self, graph, source, sink):
        """ Obtains the residual graph for the maximum
            flow analysis of the given graph between
            source and sink vertices
        """
        # Perform maximum flow analysis
        flow = csgraph.maximum_flow(graph, source, sink)

        # The flow matrix should be antisymmetric
        assert(np.array_equal(flow.residual.toarray(), -flow.residual.T.toarray()))

        # Use the residual graph to determine separated components
        # Then remove the edges connecting different components
        # http://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf
        # https://cp-algorithms.com/graph/edmonds_karp.html
        residual = graph - flow.residual # Comptute the available flow
        residual = residual.minimum(residual.T) # Use the most restrictive flow
        residual.eliminate_zeros()

        return residual

    def _minimumCut(self, graph, componentCount):
        """ Performs a minimum cut of the graph into
            componentCount elements. If componentCount
            is > 2, the cuts are done iteratively
        """

        # Copy the graph, as it will be modified
        graph = graph.copy()

        # Obtain the component labels from the graph
        nComponents, labels = self._getGraphComponents(graph)

        # Repeatedly cut the graph until the desired amount of components is obtained
        while nComponents < componentCount:
            # Perform a maximum flow analysis with the biggest component of the graph
            source, sink = self._getSourceSinkVertices(graph, labels)
            residual = self._calculateResidualGraph(graph, source, sink)

            # Partition the residual graph
            nComponents, labels = self._getGraphComponents(residual)
        
            # Only keep the edges where both vertices correspond to the same component
            for pos in zip(*graph.nonzero()):
                if labels[pos[0]] != labels[pos[1]]:
                    graph[pos] = 0
            graph.eliminate_zeros()

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