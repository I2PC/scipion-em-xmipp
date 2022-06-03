# **************************************************************************
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *              Daniel del Hoyo Gomez (daniel.delhoyo.gomez@alumnos.upm.es)
# *              Jorge Garcia Condado (jgcondado@cnb.csic.es)
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

from unittest import result
from pwem import emlib
from pwem.protocols import EMProtocol
from pwem.objects import Volume

from pyworkflow.protocol.params import MultiPointerParam, EnumParam, IntParam
from pyworkflow.protocol.params import Range, GE, GT, LE, LT
from pyworkflow.object import Float
from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import STEPS_PARALLEL
from pyworkflow.utils.path import makePath

from xmipp3.convert import setXmippAttribute

import math
import csv
import collections
import itertools
import os.path
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

class XmippProtConsensusClasses3D(EMProtocol):
    """ Compare several SetOfClasses3D.
        Return the consensus clustering based on a objective function
        that uses the similarity between clusters intersections and
        the entropy of the clustering formed.
    """
    _label = 'consensus clustering 3D'
    _devStatus = BETA

    def __init__(self, *args, **kwargs):
        EMProtocol.__init__(self, *args, **kwargs)
        self._createFileNames()
        
    def _createFileNames(self):
        iterFmt='i%(iter)06d'

        myDict = {
            'intersections': f'intersections.csv',
            'clustering': f'clusterings/{iterFmt}.csv',
            'objective_values': f'objective_values.csv',
            'elbows': f'elbows.csv',
            'reference_sizes': f'reference_sizes.csv',
            'reference_relative_sizes': f'reference_relative_sizes.csv',
        }
        self._updateFilenamesDict(myDict)

    def _defineParams(self, form):
        # Input data
        form.addSection(label='Input')
        form.addParam('inputClassifications', MultiPointerParam, pointerClass='SetOfClasses3D', label="Input classes", 
                      important=True,
                      help='Select several sets of classes where to evaluate the '
                           'intersections.')

        # Consensus
        form.addSection(label='Consensus clustering')
        form.addParam('clusteringManualCount', IntParam, label='Manual cluster count',
                      validators=[GE(2)], default=3,
                      help='Number of clusters to be obtained manually')
        form.addParam('clusteringAngle', IntParam, label='Objective function threshold angle',
                      validators=[Range(0, 90)], default=45,
                      help='Angle to determine the cluster count')
        
        # Reference random classification
        form.addSection(label='Reference random classification')
        form.addParam('randomClassificationCount', IntParam, label='Number of random classifications',
                      validators=[GE(0)], default=0,
                      help='Number of random classifications used for computing the significance of the actual '
                           'clusters. 0 for skipping this step')
        form.addParallelSection(mpi=0, threads=mp.cpu_count())

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """ Inserting one step for each intersections analysis """
        outputPrerequisites = []
        self._setupThreadPool(int(self.numberOfThreads))

        # Intersect and ensemble
        self._insertFunctionStep('intersectStep', prerequisites=[])
        self._insertFunctionStep('ensembleStep')
        outputPrerequisites.append(self._insertFunctionStep('findElbowsStep'))

        # Perform a random classification and use it as a reference
        if self.randomClassificationCount.get() > 0:
            outputPrerequisites.append(self._insertFunctionStep('checkSignificanceStep', prerequisites=[]))

        # Lastly create the output    
        self._insertFunctionStep('createOutputStep', prerequisites=outputPrerequisites)

    def intersectStep(self):
        classifications = self._getInputClassifications()

        # Compute the intersection among all classes
        intersections = self._calculateClassificationIntersections(classifications)

        # Write the result to disk
        self._writeIntersections(intersections)
        
    def ensembleStep(self, numClusters=1):
        # Read input data
        classifications = self._getInputClassifications()
        intersections = self._readIntersections()

        # Iteratively ensemble intersections until only 1 remains
        allClusterings, allObValues = self._ensembleIntersections(classifications, intersections, numClusters)

        # Reverse them so that their size increases with the index
        allClusterings.reverse()
        allObValues.reverse()

        # Save the results
        self._writeAllClusterings(allClusterings)
        self._writeObjectiveValues(allObValues)

    def findElbowsStep(self):
        obValues = self._readObjectiveValues()
        numClusters = list(range(1, 1+len(obValues)))

        # Normalize clusters and obValues
        normObValues = self._normalizeValues(obValues)
        normNumClusters = self._normalizeValues(numClusters)

        # Calculate the elbows
        elbows = {
            'origin': self._findElbowOrigin(normNumClusters, normObValues),
            'angle': self._findElbowAngle(normNumClusters, normObValues, -np.radians(self.clusteringAngle.get())),
            'pll': self._findElbowPll(obValues)
        }

        # Start counting at 1
        for key in elbows.keys():
            elbows[key] += 1

        # Save the results
        self._writeElbows(elbows)

    def checkSignificanceStep(self):
        classifications = self._getInputClassifications()
        numRand = self.randomClassificationCount.get()
        threadPool = self._getThreadPool()

        # Compute the random consensus
        consensusSizes, consensusRelativeSizes = self._calculateReferenceClassification(
            classifications, numRand, threadPool
        )

        # Sort the sizes so that percentile calculation is easy
        consensusSizes.sort()
        consensusRelativeSizes.sort()

        # Write results to disk
        self._writeReferenceClassificationSizes(consensusSizes)
        self._writeReferenceClassificationRelativeSizes(consensusRelativeSizes)

    def createOutputStep(self):
        # Read all the necessary data from disk
        manualClusterCount = self.clusteringManualCount.get()
        initialClusterCount = self._readClusteringCount()
        elbows = self._readElbows()
        consensusSizes = self._readReferenceClassificationSizes()
        consensusRelativeSizes = self._readReferenceClassificationRelativeSizes()

        # Append the manual and initial indices to the elbow
        if manualClusterCount > 0:
            elbows['manual'] = manualClusterCount
        elbows['initial'] =  initialClusterCount

        # Read selected clusters
        clusterings = dict(zip(elbows.keys(), map(self._readClustering, elbows.values())))

        # Create the output classes and define them
        particles = self._getInputParticles()
        outputClasses = dict(zip(
            map(lambda name : 'outputClasses_'+name,
                clusterings.keys() 
            ),
            map(self._createOutputClasses3D,
                itertools.repeat(particles),
                clusterings.values(),
                clusterings.keys(),
                itertools.repeat(consensusSizes),
                itertools.repeat(consensusRelativeSizes)
            )
        ))
        self._defineOutputs(**outputClasses)

        # Stablish source output relationships
        sources = list(self.inputClassifications)
        destinations = list(outputClasses.values())
        for src in sources:
            for dst in destinations:
                self._defineSourceRelation(src, dst)

    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []

        # Ensure that all classifications are made of the same images
        images = self._getInputParticles(0)
        for i in range(1, len(self.inputClassifications)):
            if self._getInputParticles(i) != images:
                errors.append(f'Classification {i} has been done with different images')

        return errors

    # --------------------------- UTILS functions ------------------------------
    def _getThreadPool(self):
        return getattr(self, '_threadPool', None)

    def _setupThreadPool(self, maxThreads):
        if maxThreads > 1:
            self._threadPool = ThreadPool(processes=maxThreads)

    def _getInputParticles(self, i=0):
        return self.inputClassifications[i].get().getImages()

    def _getInputClassifications(self):
        if not hasattr(self, '_classifications'):
            self._classifications = self._convertInputClassifications(self.inputClassifications)
        
        return self._classifications

    def _writeTable(self, path, table, fmt='%s'):
        np.savetxt(path, table, delimiter=',', fmt=fmt)

    def _readTable(self, path, dtype=str):
        return np.genfromtxt(path, delimiter=',', dtype=dtype)

    def _writeList(self, path, lst, fmt='%s'):
        self._writeTable(path, lst, fmt)

    def _readList(self, path, dtype=str):
        return self._readTable(path, dtype=dtype)
        
    def _writeClassification(self, path, classification):
        with open(path, 'w') as file:
            writer = csv.writer(file)
            
            # Write the header
            header = ['Representative index', 'Representative path', 'Source size', 'Particle ids']
            writer.writerow(header)

            # Transform the classification
            def f(cls):
                result = []
                
                # Add the location of the representative
                representative = cls.getRepresentative()
                location = representative.getLocation() if representative is not None else (0, '')
                result += list(location)

                # Add the source size
                result += [cls.getSourceSize()]

                # Add the particles
                result += list(cls.getParticleIds())

                assert(len(result) == (len(cls) + 3))
                return result
            writer.writerows(map(f, classification))

    def _readClassification(self, path):
        with open(path, 'r') as file:
            reader = csv.reader(file)

            # Ignore the header
            next(reader)

            # Transform each row to a ParticleCluster in a classification
            def f(row):
                ite = iter(row)

                # Parse the representative
                index = int(next(ite))
                path = next(ite)
                representative = Volume(location=(index, path))

                # Parse the source size
                sourceSize = int(next(ite))

                # Parse the id set
                particleIds = set(map(lambda id : int(id), ite))

                result = XmippProtConsensusClasses3D.ParticleCluster(particleIds, representative, sourceSize)
                assert(len(row) == (len(result) + 3))
                return result
            return list(map(f, reader))

    def _writeDictionary(self, path, d):
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(d.keys()) # Header
            writer.writerow(d.values()) # Values

    def _readDictionary(self, path, dtype=str):
        with open(path, 'r') as file:
            reader = csv.reader(file)
            keys = list(next(reader))
            values = list(map(lambda value : dtype(value), next(reader)))
            return dict(zip(keys, values))

    def _writeIntersections(self, intersections):
        self._writeClassification(self._getExtraPath(self._getFileName('intersections')), intersections)

    def _readIntersections(self):
        return self._readClassification(self._getExtraPath(self._getFileName('intersections')))

    def _writeClustering(self, iter, clustering):
        self._writeClassification(self._getExtraPath(self._getFileName('clustering', iter=iter)), clustering)

    def _readClustering(self, iter):
        return self._readClassification(self._getExtraPath(self._getFileName('clustering', iter=iter)))

    def _readClusteringCount(self):
        count = 0
        while os.path.exists(self._getExtraPath(self._getFileName('clustering', iter=count+1))):
            count += 1
        return count

    def _writeAllClusterings(self, clusterings):   
        # Create the path
        makePath(os.path.dirname(self._getExtraPath(self._getFileName('clustering', iter=0))))

        # Write all
        for iter, clustering in enumerate(clusterings, start=1):
            assert(len(clustering) == iter)
            self._writeClustering(iter, clustering)

    def _readAllClusterings(self):
        return list(map(self._readClustering, range(1, 1+self._readClusteringCount())))

    def _writeObjectiveValues(self, allObValues):
        self._writeList(self._getExtraPath(self._getFileName('objective_values')), allObValues, '%f')

    def _readObjectiveValues(self):
        return self._readList(self._getExtraPath(self._getFileName('objective_values')), dtype=float)

    def _writeElbows(self, elbows):
        self._writeDictionary(self._getExtraPath(self._getFileName('elbows')), elbows)

    def _readElbows(self):
        return self._readDictionary(self._getExtraPath(self._getFileName('elbows')), dtype=int)

    def _writeReferenceClassificationSizes(self, sizes):
        self._writeList(self._getExtraPath(self._getFileName('reference_sizes')), sizes, '%d')

    def _readReferenceClassificationSizes(self):
        path = self._getExtraPath(self._getFileName('reference_sizes'))
        if os.path.exists(path):
            return self._readList(path, dtype=int)
        else:
            return None

    def _writeReferenceClassificationRelativeSizes(self, sizes):
        self._writeList(self._getExtraPath(self._getFileName('reference_relative_sizes')), sizes, '%f')

    def _readReferenceClassificationRelativeSizes(self):
        path = self._getExtraPath(self._getFileName('reference_relative_sizes'))
        if os.path.exists(path):
            return self._readList(path, dtype=float)
        else:
            return None

    def _convertInputClassifications(self, classifications):
        """ Returns the list of lists of sets that stores the set of ids of each class"""
        def class3dToParticleCluster(cls):
            ids = cls.getIdSet()
            rep = cls.getRepresentative().clone() if cls.hasRepresentative() else None
            return XmippProtConsensusClasses3D.ParticleCluster(ids, rep)

        f = lambda classification : list(map(class3dToParticleCluster, classification.get()))
        result = list(map(f, classifications))
        return result
    
    def _calculateClassificationIntersections(self, classifications):
        # Start with the first classification
        result = classifications[0]

        for i in range(1, len(classifications)):
            classification = classifications[i]
            intersections = []

            # Perform the intersection with previous classes
            for cls0 in result:
                for cls1 in classification:
                    intersection = cls0.intersection(cls1)
                    if intersection:
                        intersections.append(intersection)

            # Use the new intersection for the next step
            result = intersections

        return result

    def _calculateSubsetSimilarity(self, n, c):
        """ Return the similarity of intersection subset (n) with class subset (c) """
        inter = len(n.intersection(c))
        return inter / len(n)

    def _buildSimilarityVector(self, n, cs):
        """ Build the similarity vector of n subset with each c subset in cs """
        return list(map(self._calculateSubsetSimilarity, itertools.repeat(n), cs))

    def _calculateEntropy(self, p):
        """ Return the entropy of the elements in a vector """
        result = -sum(map(lambda x : x*math.log2(x), p))
        assert(result >= 0) # Entropy cannot be negative
        return result

    def _calculateDistribution(self, ns):
        """ Return the vector of distribution (p) from the n subsets in ns
        """
        p = list(map(len, ns))
        return np.array(p) / sum(p)

    def _calculateCosineSimilarity(self, v1, v2):
        """ Return the cosine similarity of two vectors that is used as similarity """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _objectiveFunction(self, ns, indices, vs, eps=0.01):
        """Objective function to minimize based on the entropy and the similarity between the clusters to merge
        """
        p = self._calculateDistribution(ns)
        pi = self._calculateDistribution(self._ensembleIntersectionPair(ns, indices))
        h = self._calculateEntropy(p)
        hi = self._calculateEntropy(pi)

        sv = self._calculateCosineSimilarity(vs[indices[0]], vs[indices[1]])

        return (h - hi) / (sv + eps)

    def _ensembleIntersectionPair(self, intersections, indices):
        """ Return a list of subsets where the intersections[indices] have been merged """
        # Divide the input intersections into the ones referred by indices and not
        selection = [intersections[i] for i in indices]  # intersection[indices]
        result = [intersections[i] for i in range(len(intersections)) if i not in indices]  # intersections - selection

        # Merge all the selected clusters into the same one
        merged = XmippProtConsensusClasses3D.ParticleCluster.union(*selection)
        
        # Add the merged items to the result
        result.append(merged)
        return result

    def _ensembleCheapestIntersectionPair(self, allClasses, intersections):
        """ Merges the lest costly pair of intersections """
        # Compute the similarity vectors
        similarityVectors = list(map(self._buildSimilarityVector, intersections, itertools.repeat(allClasses)))

        # For each possible pair of intersections compute the cost of merging them
        obValues = {}
        for pair in itertools.combinations(range(len(intersections)), r=2):
            obValues[pair] = self._objectiveFunction(intersections, pair, similarityVectors)

        values = list(obValues.values())
        keys = list(obValues.keys())

        # Select the minimum cost pair and merge them
        obValue = min(values)
        mergePair = keys[values.index(obValue)]
        mergedIntersections = self._ensembleIntersectionPair(intersections, mergePair)

        return mergedIntersections, obValue

    def _ensembleIntersections(self, classifications, intersections, numClusters=1):
        """ Ensemble clustering by COSS that merges interactions of clusters based
            on entropy minimization """
        # Results
        allIntersections = [intersections]
        allObValues = [0.0]

        # Obtain a list with all source classes
        allClasses = list(itertools.chain(*classifications))

        # Iteratively merge intersections
        while len(allIntersections[-1]) > numClusters:
            mergedIntersections, obValue = self._ensembleCheapestIntersectionPair(allClasses, allIntersections[-1])
            allIntersections.append(mergedIntersections)
            allObValues.append(obValue)

        assert(len(allIntersections) == len(allObValues))
        return allIntersections, allObValues
            
    def _normalizeValues(self, x):
        """Normalize values to range 0 to 1"""
        return (x-np.min(x))/(np.max(x)-np.min(x))

    def _calculateGaussianMleEstimates(self, d, q):
        """ MLE estimates of guassian distributions
        with common variance parameter
        """
        p = len(d)
        d1 = d[:q]
        d2 = d[q:]
        mu1 = np.mean(d1)
        mu2 = np.mean(d2)
        var1 = np.var(d1)
        var2 = np.var(d2)
        sigma = ((q-1)*var1+(p-q-1)*var2)/(p-2)
        theta1 = [mu1, sigma]
        theta2 = [mu2, sigma]
        return theta1, theta2

    def _calculateGaussianDistribution(self, d, theta):
        """ Gaussian pdf """
        m = theta[0]
        s = theta[1]
        gauss = 1/(np.sqrt(2*np.pi*s))*np.exp(-1/(2*s)*(d-m)**2)
        return gauss

    def _calculateProfileLogLikelihood(self, d, q, theta1, theta2):
        """ Profile log likelihood for given parameters """
        log_f_q = np.log(self._calculateGaussianDistribution(d[:q], theta1))
        log_f_p = np.log(self._calculateGaussianDistribution(d[q:], theta2))
        l_q = np.sum(log_f_q) + np.sum(log_f_p)
        return l_q

    def _calculateFullProfileLogLikelihood(self, d):
        """ Calculate profile log likelihood for each partition
            of the data """
        pll = []
        for q in range(1, len(d)):
            theta1, theta2 = self._calculateGaussianMleEstimates(d, q)
            pll.append(self._calculateProfileLogLikelihood(d, q, theta1, theta2))
        return pll

    def _findElbowPll(self, obValues):
        """ Find the elbow according to the full profile likelihood
        """
        # Get profile log likelihood for log of objective values
        # Remove last obValue as it is zero and log(0) is undefined
        pll = self._calculateFullProfileLogLikelihood(np.log(obValues[:-1]))
        return np.argmax(pll)

    def _findElbowOrigin(self, x, y):
        """ Find the point closest to origin from normalied data
        """
        coors = np.column_stack((x, y))
        distances = np.linalg.norm(coors, axis=1)
        assert(len(x) == len(distances))
        return np.argmin(distances)

    def _findElbowAngle(self, x, y, angle=-math.pi/4):
        """ Find the angle the slope of the function makes and
            return the point at which the slope crosses the given angle
        """
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.arctan2(dy, dx)

        # Find the first point with an angle greater than the given one
        crossing = np.argmax(angles >= angle)

        return crossing

    def _getClassificationLengths(self, classifications):
        """ Returns a list of lists that stores the lengths of each classification """
        return list(map(lambda classification : list(map(len, classification)), classifications))

    def _calculateRandomClassification(self, C, N):
        """ Randomly classifies N element indices into groups with
        sizes defined by rows of C """
        Cp = []

        for Ci in C:
            assert(sum(Ci) == N)  # The groups should address all the elements TODO: Maybe LEQ?
            x = np.argsort(np.random.uniform(size=N)).tolist()  # Shuffles a [0, N) iota

            # Select the number random indices requested by each group
            Cip = []
            first = 0
            for s in Ci:
                Cip.append(XmippProtConsensusClasses3D.ParticleCluster(x[first:first + s]))
                first += s

            # Add the random classification to the result
            Cp.append(Cip)

        return Cp

    def _calculateRandomClassificationConsensus(self, C, N):
        """ Obtains the intersections of a consensus
            of a random classification of sizes defined in C of
            N elements. C is a list of lists, where de sum of each
            row must equal N """

        # Create random partitions of same size
        randomClassification = self._calculateRandomClassification(C, N)

        # Compute the repeated classifications
        return self._calculateClassificationIntersections(randomClassification)

    def _calculateReferenceClassification(self, classifications, numExec, threadPool=None):
        """ Create random partitions of same size to compare the quality
         of the classification """
        clusterLengths = self._getClassificationLengths(classifications)
        numParticles = sum(clusterLengths[0])

        # Repeatedly obtain a consensus of a random classification of same size
        if threadPool:
            consensus = threadPool.starmap(
                self._calculateRandomClassificationConsensus,
                itertools.repeat((clusterLengths, numParticles), numExec)
            )
        else:
            consensus = list(map(
                self._calculateRandomClassificationConsensus, 
                itertools.repeat(clusterLengths, numExec),
                itertools.repeat(numParticles, numExec)
            ))
        assert(len(consensus) == numExec)

        # Concatenate all consensuses
        consensus = list(itertools.chain(*consensus))

        # Obtain the consensus sizes
        consensusSizes = list(map(len, consensus))
        consensusSizeRatios = list(map(XmippProtConsensusClasses3D.ParticleCluster.getRelativeSize, consensus))
        return consensusSizes, consensusSizeRatios

    def _createOutputClasses3D(self, particles, clustering, name, randomConsensusSizes=None, randomConsensusRelativeSizes=None):
        outputClasses = self._createSetOfClasses3D(particles, suffix=name)

        # Create a list with the filenames of the representatives
        representatives = list(map(
            XmippProtConsensusClasses3D.ParticleCluster.getRepresentative,
            clustering
        ))

        # Fill the output
        loader = XmippProtConsensusClasses3D.ClassesLoader(
            clustering, 
            representatives,
            randomConsensusSizes,
            randomConsensusRelativeSizes
        )
        loader.fillClasses(outputClasses)

        return outputClasses
    
    class ParticleCluster:
        """ Keeps track of the information related to successive class intersections.
            It is instantiated with a single class and allows to perform intersections
            and unions with it"""
        def __init__(self, particleIds, representative=None, sourceSize=None):
            self._particleIds = set(particleIds)  # Particle ids belonging to this intersection
            self._representative = representative
            self._sourceSize = sourceSize if sourceSize is not None else len(self._particleIds)  # The size of the representative class

        def __len__(self):
            return len(self.getParticleIds())

        def __eq__(self, other):
            return self.getParticleIds() == other.getParticleIds()

        def intersection(self, *others):
            return self._combine(set.intersection, min, XmippProtConsensusClasses3D.ParticleCluster.getSourceSize, *others)

        def union(self, *others):
            return self._combine(set.union, max, len, *others)

        def getParticleIds(self):
            return self._particleIds

        def getRepresentative(self):
            return self._representative

        def getSourceSize(self):
            return self._sourceSize

        def getRelativeSize(self):
            return len(self) / self.getSourceSize()

        def _combine(self, operation, selector, selectionCriteria, *others):
            allItems = (self, ) + others
            
            # Perform the requested operation among the id sets
            allParticleIds = map(XmippProtConsensusClasses3D.ParticleCluster.getParticleIds, allItems)
            particleIds = operation(*allParticleIds)

            # Select the class with the appropriate criteria
            selection = selector(allItems, key=selectionCriteria)
            representative = selection.getRepresentative()
            sourceSize = selection.getSourceSize()

            return XmippProtConsensusClasses3D.ParticleCluster(particleIds, representative, sourceSize)

    class ClassesLoader:
        """ Helper class to produce classes
        """
        def __init__(self, clustering, representatives, randomConsensusSizes=None, randomConsensusRelativeSizes=None):
            self.clustering = clustering
            self.classification = self._createClassification(clustering)
            self.representatives = representatives
            self.randomConsensusSizes = randomConsensusSizes
            self.randomConsensusRelativeSizes = randomConsensusRelativeSizes

        def fillClasses(self, clsSet):
            clsSet.classifyItems(updateItemCallback=self._updateParticle,
                                updateClassCallback=self._updateClass,
                                itemDataIterator=iter(self.classification.items()),
                                doClone=False)

        def _updateParticle(self, item, row):
            particleId, classId = row
            assert(item.getObjId() == particleId)
            item.setClassId(classId)

        def _updateClass(self, item):
            classId = item.getObjId()
            classIdx = classId-1

            # Set the representative
            item.setRepresentative(self.representatives[classIdx])

            # Set the size p-value
            if self.randomConsensusSizes is not None:
                size = len(self.clustering[classIdx])
                percentile = self._findPercentile(self.randomConsensusSizes, size)
                pValue = 1 - percentile
                setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_SIZE_PVALUE, Float(pValue))


            # Set the relative size p-value
            if self.randomConsensusRelativeSizes is not None:
                size = self.clustering[classIdx].getRelativeSize()
                percentile = self._findPercentile(self.randomConsensusRelativeSizes, size)
                pValue = 1 - percentile
                setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_RELATIVE_SIZE_PVALUE, Float(pValue))

        def _createClassification(self, clustering):
            # Fill the classification data (particle-class mapping)
            result = {}
            for cls, data in enumerate(clustering):
                for particleId in data.getParticleIds():
                    result[particleId] = cls + 1 

            return collections.OrderedDict(sorted(result.items())) 

        def _findPercentile(self, data, value):
            """ Given an array of values (data), finds the corresponding percentile of the value.
                Percentile is returned in range [0, 1] """
            assert(np.array_equal(sorted(data), data))  # In order to iterate it in ascending order

            # Count the number of elements that are smaller than the given value
            i = 0
            while i < len(data) and data[i] < value:
                i += 1

            # Convert it into a percentage
            return float(i) / float(len(data))
