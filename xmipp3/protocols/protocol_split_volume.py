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
from pwem.objects import Volume

import xmippLib
from xmipp3.convert import writeSetOfParticles

import math
import itertools
import numpy as np

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
                      pointerClass='SetOfAverages', pointerCondition='hasAlignmentProj',
                      important=True, 
                      help='Select a set of particles with angles. Preferrably the output of a run of directional classes')

        form.addSection(label='Graph building')
        form.addParam('maxAngularDistance', FloatParam, label="Maximum angular distance (deg)", 
                      validators=[Range(0, 180)], default=15,
                      help='Maximum angular distance for considering the correlation of two classes. '
                      'Valid range: 0 to 180 deg')
        form.addParam('maxNeighbors', IntParam, label="Maximum number of neighbors", 
                      validators=[GT(0)], default=8,
                      help='Number of neighbors to consider for each directional class.')
        form.addParam('enforceUndirected', BooleanParam, label="Enforce undirected", 
                      default=False, expertLevel=LEVEL_ADVANCED,
                      help='Enforce a undirected graph')

        form.addSection(label='Graph partition')
        form.addParam('graphPartitionMethod', EnumParam, label="Graph partition method", 
                      choices=['Spectral',  'Minimum cut'], default=0,
                      help='The method used for partitioning the graph')
        form.addParam('graphPartitionCount', IntParam, label="Minimum number of components", 
                      validators=[GE(2)], default=2,
                      help='Minimum number of classes to obtain')
        form.addParallelSection()
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('computeAngularDistancesStep')
        self._insertFunctionStep('computeCorrelationStep')
        self._insertFunctionStep('computeWeightsStep')
        self._insertFunctionStep('graphPartitionStep')
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        self._particleList = self._getParticleList(self.directionalClasses.get())

    def computeAngularDistancesStep(self):
        # Read input data
        particles = self._particleList

        # Perform the computation
        distances = self._calculateAngularDistanceMatrix(particles)

        # Save output data
        self._writeAngularDistances(distances)

    def computeCorrelationStep(self):
        # Read input data
        distances = self._readAngularDistances()
        maxAngularDistance = np.radians(self.maxAngularDistance.get())
        particles = self._particleList

        # Compute the class pairs to be considered
        pairs = self._calculatePairMatrix(distances, maxAngularDistance)

        # Compute the correlation
        correlations = self._calculateCorrelationMatrix(particles, pairs)

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
        labels = self._readLabels()
        particles = self.directionalClasses.get()
        nClasses = int(labels.max()) + 1

        # Create the output elements
        classes = self._createOutputClasses(particles, nClasses, labels, 'output')
        volumes = self._createOutputVolumes(classes, 'output')

        # Define the output
        self._defineOutputs(outputClasses=classes, outputVolumes=volumes)

        # Establish source-output relations
        sources = [self.directionalClasses]
        destinations = [classes, volumes]
        for src in sources:
            for dst in destinations:
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
    def _writeMatrix(self, path, x, fmt='%s'):
        # Determine the format
        np.savetxt(path, x, delimiter=',', fmt=fmt) # CSV

    def _readMatrix(self, path, dtype=float):
        return np.genfromtxt(path, delimiter=',', dtype=dtype) # CSV

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

    def _getParticleList(self, particles):
        """Converts a set of particles into a list"""

        result = []

        for particle in particles:
            result.append(particle.clone())

        assert(len(result) == len(particles))
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

    def _calculatePairMatrix(self, distances, maxDistance):
        """ Computes a symmetrical matrix of booleans
            where image pairs set to true are closer than
            the given threshold. The diagonal is se to false
            as the comparison with itself is not meaningful
        """
        pairs = distances <= maxDistance
        np.fill_diagonal(pairs, False) # Do not compute correlations with itself
        return pairs

    def _calculateCorrelation(self, class0, class1):
        """ Computes the correlation of two images
            after aligning them
        """
        img0 = xmippLib.Image(class0.getLocation())
        img1 = xmippLib.Image(class1.getLocation())
        corr = img0.correlationAfterAlignment(img1)
        return max(corr, 0.0)

    def _calculateCorrelationMatrix(self, classes, pairs):
        """ Computes the correlations between image pairs
            defined by pairs mask matrix. Pairs should be 
            symmetrical. Therefore, the resulting matrix is 
            also symmetrical.
        """
        correlations = np.zeros_like(pairs, dtype=float)

        # Ensure that the pairs matrix is symmetrical
        assert(np.array_equal(pairs, pairs.T))

        # Compute the symmetric matrix with the correlations
        for idx0, idx1 in zip(*np.nonzero(pairs)):
            if idx0 < idx1:
                # Calculate the corrrelation
                class0 = classes[idx0]
                class1 = classes[idx1]

                # Write it on symmetrical positions
                correlation = self._calculateCorrelation(class0, class1)
                correlations[idx0, idx1] = correlation
                correlations[idx1, idx0] = correlation

            elif idx0 == idx1:
                # Correlation with itself is 1
                correlations[idx0, idx0] = 1.0

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
        minVal = np.min(correlations[np.nonzero(correlations)])
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
        component = stats.mode(labels)

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

    def _calculateResidualGraph(graph, source, sink):
        """ Obtains the residual graph for the maximum
            flow analysis of the given graph between
            source and sink vertices
        """
        # Perform maximum flow analysis
        flow = csgraph.maximum_flow(graph, source, sink)

        # Flow should be an antisymmetric matrix
        assert(flow.residual == -flow.residual.T)

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
            residual = self._calculateResidualGraph(source, sink)

            # Partition the residual graph
            nComponents, labels = self._getGraphComponents(residual)
        
            # Only keep the edges where both vertices correspond to the same component
            for pos in zip(*graph.nonzero()):
                if labels[pos[0]] != labels[pos[1]]:
                    graph[pos] = 0
            graph.eliminate_zeros()

        return labels

    def _reconstructVolume(self, path, particles, xmdSuffix=''):
        # Convert the particles to a xmipp metadata file
        fnParticles = self._getTmpPath('particles_'+xmdSuffix+'.xmd')
        writeSetOfParticles(particles, fnParticles)

        # Reconstruct the volume
        args  = f'-i {fnParticles} '
        args += f'-o {path} '
        args += f'--max_resolution 0.25 '
        args += f'-v 0'
        self.runJob('xmipp_reconstruct_fourier', args)

        # Clear the metadata file
        cleanPattern(fnParticles)

    def _createOutputClasses(self, particles, nClasses, classification, suffix=''):
        result = self._createSetOfClasses3D(particles, suffix)

        # Create a list with all the representative filenames
        representatives = []
        for i in range(nClasses):
            path = self._getExtraPath(self._getFileName('representative', suffix=suffix, cls=i))
            representatives.append(path)

        # Classify input particles
        loader = XmippProtSplitvolume.ClassesLoader(classification, representatives)
        loader.fillClasses(result)

        return result

    def _createOutputVolumes(self, classes, suffix=''):
        result = self._createSetOfVolumes(suffix)
        result.setSamplingRate(classes.getImages().getSamplingRate())
        for i, cls in enumerate(classes):
            vol = cls.getRepresentative()
            vol.setObjId(cls.getObjId())
            self._reconstructVolume(vol.getFileName(), cls, suffix+str(i))
            result.append(vol)
        
        return result

    class ClassesLoader:
        """ Helper class to produce classes
        """
        def __init__(self, classification, representatives):
            self.classification = classification
            self.representatives = representatives

        def fillClasses(self, clsSet):
            clsSet.classifyItems(updateItemCallback=self._updateParticle,
                                updateClassCallback=self._updateClass,
                                itemDataIterator=iter(self.classification),
                                doClone=False)

        def _updateParticle(self, item, cls):
            classId = cls + 1
            item.setClassId(classId)

        def _updateClass(self, item):
            classId = item.getObjId()
            classIdx = classId-1

            # Set the representative
            vol = Volume(self.representatives[classIdx])
            item.setRepresentative(vol)
