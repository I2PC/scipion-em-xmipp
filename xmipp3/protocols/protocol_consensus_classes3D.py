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

from pwem import emlib
from pwem.protocols import EMProtocol

from pyworkflow.protocol.params import MultiPointerParam, EnumParam, IntParam
from pyworkflow.protocol.params import Range, GE, GT, LE, LT
from pyworkflow.object import Float, List, Object
from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import STEPS_PARALLEL
from pyworkflow.utils.path import makePath

import math
import csv
import collections
import itertools
import multiprocessing as mp
import numpy as np
import os.path

from xmipp3.convert.convert import setXmippAttribute


class XmippProtConsensusClasses3D(EMProtocol):
    """ Compare several SetOfClasses3D.
        Return the consensus clustering based on a objective function that uses the similarity between clusters
        intersections and the entropy of the clustering formed.
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
            'clustering': f'clusterings/i{iterFmt}.csv',
            'objective_values': f'objective_values.csv'
        }
        self._updateFilenamesDict(myDict)

    def _defineParams(self, form):
        # Input data
        form.addSection(label='Input')
        form.addParam('inputClasses', MultiPointerParam, pointerClass='SetOfClasses3D', label="Input classes", 
                      important=True,
                      help='Select several sets of classes where to evaluate the '
                           'intersections.')

        # Consensus
        form.addSection(label='Consensus clustering')
        form.addParam('clusteringMethod', EnumParam, label='Clustering method',
                      choices=['Manual', 'Origin', 'Angle', 'PLL'], default=0,
                      help='Method used to automatically determine the number of classes')
        form.addParam('clusteringManualCount', IntParam, label='Cluster count',
                      condition='clusteringMethod==0', validators=[GE(2)], default=3,
                      help='Number of clusters to be obtained manually')
        form.addParam('clusteringAngle', IntParam, label='Angle',
                      condition='clusteringMethod==2', validators=[Range(0, 90)], default=45,
                      help='Angle to determine the cluster count')
        
        # Reference random classification
        form.addSection(label='Reference random classification')
        form.addParam('randomClassificationCount', IntParam, label='Number of random classifications',
                      validators=[GE(0)], default=0,
                      help='Number of random classifications used for computing the significance of the actual '
                           'clusters. 0 for skipping this step')
        form.addParallelSection()

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """ Inserting one step for each intersections analysis """
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('intersectStep')
        self._insertFunctionStep('ensembleStep')

    def convertInputStep(self):
        self.classificationMatrix = self._convertInputClassifications(self.inputClasses)
    
    def intersectStep(self):
        classifications = self.classificationMatrix

        # Compute the intersection among all classes
        intersections = self._calculateClassificationIntersections(classifications)

        # Write the result to disk
        self._writeIntersections(intersections)
        
    def ensembleStep(self, numClusters=1):
        # Read input data
        classifications = self.classificationMatrix
        intersections = self._readIntersections()

        # Iteratively ensemble intersections until only 1 remains
        allIntersections, allObValues = self._ensembleIntersections(classifications, intersections, numClusters)

        # Reverse them so that their size increases with the index
        allIntersections.reverse()
        allObValues.reverse()

        # Save the results
        self._writeEnsembledIntersections(allIntersections)
        self._writeObjectiveValues(allObValues)











    def populateStep(self, classificationIdx=0):
        """ Initializes attributes to be able to start comparing """

        # Select the given classification
        classification = self.inputMultiClasses[classificationIdx].get()

        # At the beginning, use the given classification as the intersection
        # (consider it as a intersection with the 'all' set, AKA the identity for this operation)
        intersections = []
        for cluster in classification:
            clusterId = cluster.getObjId()
            particleIds = cluster.getIdSet()

            # Build a intersection-like structure to store it on the intersection list
            intersection = XmippProtConsensusClasses3D.ClassIntersection(particleIds, classificationIdx, clusterId)

            # Do not append classes that have no elements
            if intersection:
                intersections.append(intersection)

        # Store the results
        self.intersectionList = List(intersections)

    def compareStep(self, classification1Idx):
        """ Intersects the given classification with all the previous intersections """

        # Select the given classification
        classification1 = self.inputMultiClasses[classification1Idx].get()

        print('Computing intersections between classes from classification %s and '
              'the previous ones:' % (classification1.getNameId()))

        # Intersect all the classes from the given classification with the previous intersection
        intersections = []
        for cluster1 in classification1:
            cluster1Id = cluster1.getObjId()
            particleIds1 = cluster1.getIdSet()

            # Build a intersection-like structure to intersect it against the other
            intersector = XmippProtConsensusClasses3D.ClassIntersection(particleIds1, classification1Idx, cluster1Id)

            for currIntersection in self.intersectionList:
                # Intersect the previous intersection with the intersector
                intersection = currIntersection.intersect(intersector)

                # Do not append classes that have no elements
                if intersection:
                    intersections.append(intersection)

        # Overwrite previous intersections with the new ones
        self.intersectionList = List(intersections)

    def findElbowsStep(self):
        """" Finds elbows of the COSS ensemble process """
        # Shorthands for variables
        numClusters = list(range(1, 1+len(self.ensembleObValues)))
        obValues = self.ensembleObValues

        # Get profile log likelihood for log of objective values
        # Remove last obValue as it is zero and log(0) is undefined
        pll = self._calculateFullProfileLogLikelihood(np.log(obValues[:-1]))

        # Normalize clusters and obValues
        normc = self._normalizeValues(numClusters)
        normo = self._normalizeValues(obValues)

        # Find different kinds of elbows
        elbow_idx_origin = self._findClosestPointToOrigin(normc, normo)
        elbow_idx_angle, _ = self._findElbowAngle(normc, normo)
        elbow_idx_pll = np.argmax(pll)

        # Parse all elbow info to pass to other functions
        elbows = {
            'origin': elbow_idx_origin,
            'angle': elbow_idx_angle,
            'pll': elbow_idx_pll
        }

        # Save relevant data for analysis
        self.elbows = Object(elbows)

    def checkSignificanceStep(self):
        """ Create random partitions of same size to compare the quality
         of the classification """

        # Set up multi-threading
        threadPool = mp.Pool(int(self.numberOfThreads))

        # Obtain the execution parameters
        numExec = self.numRand.get()

        # Obtain the group sizes
        clusterLengths = self._getClusterLengths()

        # Calculate the total particle count
        numParticles = sum(clusterLengths[0])

        # Repeatedly obtain a consensus of a random classification of same size
        consensus = threadPool.starmap(
            XmippProtConsensusClasses3D.RandomConsensus(),
            itertools.repeat((clusterLengths, numParticles), numExec)
        )
        threadPool.close()
        consensus = list(itertools.chain(*consensus))

        # Obtain the size and its ratio
        consensusSizes = [len(s) for s in consensus]
        consensusSizeRatios = [s.getSizeRatio() for s in consensus]
        consensusSizes.sort()
        consensusSizeRatios.sort()

        # Calculate common percentiles
        percentiles = [90, 95, 99, 100]  # Common values for percentiles. Add on your own taste. 100 is the max value
        sizePercentiles = np.percentile(consensusSizes, percentiles)
        sizeRatioPercentiles = np.percentile(consensusSizeRatios, percentiles)

        # Store the results
        self.randomConsensusSizes = List(consensusSizes)
        self.randomConsensusSizeRatios = List(consensusSizeRatios)
        self.randomConsensusSizePercentiles = Object({key: value for key, value in zip(percentiles, sizePercentiles)})
        self.randomConsensusSizeRatioPercentiles = Object({key: value for key, value in zip(percentiles, sizeRatioPercentiles)})

    def createOutputStep(self):
        """Save the output classes"""
        self._saveOutputs() # Saves data into pkl files for later visualization

        # Always output all the initial intersections
        outputClassesInitial = self._createOutput3DClassWithAttributes(self.intersectionList, 'initial')
        self._defineOutputs(outputClasses_initial=outputClassesInitial)

        for item in self.inputMultiClasses:
            self._defineSourceRelation(item, outputClassesInitial)

        # Check if the ensemble step has been performed
        if hasattr(self, 'ensembleIntersectionLists'):
            manualClusterCount = self.manualClusterCount.get()
            automaticClusterCount = self.automaticClusterCount.get()

            # Check if a manual cluster count was given
            if manualClusterCount > 0:
                i = min(manualClusterCount, len(self.ensembleIntersectionLists)) - 1  # Most restrictive one
                outputClassesManual = self._createOutput3DClassWithAttributes(self.ensembleIntersectionLists[i], 'manual')
                self._defineOutputs(outputClasses_manual=outputClassesManual)

                # Establish output relations
                for item in self.inputMultiClasses:
                    self._defineSourceRelation(item, outputClassesManual)

            # Check if automatic cluster count is enabled
            if automaticClusterCount == 0:
                elbows = self.elbows.get()
                for key, value in elbows.items():
                    outputClassesName = 'outputClasses_' + key
                    outputClasses = self._createOutput3DClassWithAttributes(self.ensembleIntersectionLists[value], key)
                    self._defineOutputs(**{outputClassesName: outputClasses})

                    # Establish output relations
                    for item in self.inputMultiClasses:
                        self._defineSourceRelation(item, outputClasses)

    # --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []

        # Show automatically obtained elbows
        if hasattr(self, 'elbows'):
            summary.append('Number of Classes')
            elbows = self.elbows.get()

            for key, value in elbows.items():
                summary.append(f'{key}: {value+1}')

        # Check if common percentiles of sizes are going to be shown
        if hasattr(self, 'randomConsensusSizePercentiles'):
            summary.append('Common consensus size percentiles')
            percentiles = self.randomConsensusSizePercentiles.get()

            # Add all the values to the summary
            for key, value in percentiles.items():
                summary.append(f'{key}%: {value}')

        # Check if common percentiles of ratios are going to be shown
        if hasattr(self, 'randomConsensusSizeRatioPercentiles'):
            summary.append('Common consensus size ratio percentiles')
            percentiles = self.randomConsensusSizeRatioPercentiles.get()

            # Add all the values to the summary
            for key, value in percentiles.items():
                summary.append(str(key) + '%: ' + str(value))

        return summary

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        errors = []
        
        max_nClusters = np.prod([len(self.inputMultiClasses[i].get()) for i in range(len(self.inputMultiClasses))])
        if self.manualClusterCount.get() > max_nClusters:
            errors.append("Too many clusters selected for output")

        particles0 = self.inputMultiClasses[0].get().getImages()
        for i in range(1, len(self.inputMultiClasses)):
            particles1 = self.inputMultiClasses[i].get().getImages()
            if particles0 != particles1:
                errors.append(f"SetOfClasses #{i} was made with different particles")

        return errors

    # --------------------------- UTILS functions ------------------------------
    def _writeClassification(self, path, classification):
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(classification)

    def _readClassification(self, path):
        with open(path, 'r') as file:
            reader = csv.reader(file)
            return list(map(lambda row : set(map(lambda id : int(id), row)), reader))
        
    def _writeIntersections(self, intersections):
        self._writeClassification(self._getExtraPath(self._getFileName('intersections')), intersections)

    def _readIntersections(self):
        return self._readClassification(self._getExtraPath(self._getFileName('intersections')))

    def _writeEnsembledIntersections(self, allIntersections):
        # Create the path
        makePath(os.path.dirname(self._getExtraPath(self._getFileName('clustering', iter=0))))

        # Write all
        for iter, intersections in enumerate(allIntersections, start=1):
            assert(len(intersections) == iter)
            self._writeClassification(self._getExtraPath(self._getFileName('clustering', iter=iter)), intersections)

    def _readEnsembledIntersections(self):
        allIntersections = []
        while os.path.exists(path := self._getExtraPath(self._getFileName('clustering', iter=len(allIntersections)))):
            intersections = self._readClassification(path)
            allIntersections.append(intersections)

        return allIntersections

    def _writeObjectiveValues(self, allObValues):
        with open(self._getExtraPath(self._getFileName('objective_values')), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(allObValues)

    def _readObjectiveValues(self):
        with open(self._getExtraPath(self._getFileName('objective_values')), 'r') as file:
            reader = csv.reader(file)
            return list(map(lambda obValue : float(obValue), next(reader)))

    def _convertInputClassifications(self, classification):
        """ Returns the list of lists of sets that stores the set of ids of each class"""
        f = lambda classification : list(map(lambda cls : cls.getIdSet(), classification.get()))
        result = list(map(f, classification))
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
                    intersection = cls0 & cls1
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
        H = 0
        for i in range(len(p)):
            H += p[i] * math.log(p[i], 2)
        return -H

    def _calculateDistribution(self, ns, indices=[]):
        """ Return the vector of distribution (p) from the n subsets in ns
            If two indices are given, these two subsets are merged in the distribution """
        p = np.array([])
        if len(indices) == 2:
            # Merging two subsets
            N = len(ns) - 1
            for i in range(len(ns)):
                if i not in indices:
                    p = np.append(p, len(ns[i]))
            p = np.append(p, len(ns[indices[0]].union(ns[indices[1]])))
        else:
            N = len(ns)
            for n in ns:
                p = np.append(p, len(n))
        return p / N

    def _calculateCosineSimilarity(self, v1, v2):
        """ Return the cosine similarity of two vectors that is used as similarity """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _objectiveFunction(self, ns, indices, vs, eps=0.01):
        """Objective function to minimize based on the entropy and the similarity between the clusters to merge
        """
        p = self._calculateDistribution(ns)
        pi = self._calculateDistribution(ns, indices)
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
        merged = set.union(*selection)
        
        # Add the merged items to the result
        result.append(merged)
        return result

    def _ensembleCheapestIntersectionPair(self, allClasses, intersections):
        """ Merges the lest costly pair of intersections """
        # Compute the similarity vectors
        similarityVectors = list(map(self._buildSimilarityVector, intersections, itertools.repeat(allClasses)))

        # For each possible pair of intersections compute the cost of merging them
        obValues = {}
        for pair in itertools.permutations(range(len(intersections)), r=2):
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
            






    def _getClusterLengths(self):
        """ Returns a list of lists that stores the lengths of each classification """
        result = []

        for classification in self.inputMultiClasses:
            classification = classification.get()

            lengths = []
            for cluster in classification:
                lengths.append(len(cluster))
            result.append(lengths)

        return result

    def _getIntersectionParticleIds(self, intersections):
        """ Return the list of sets from a list of intersections """
        result = []

        for intersection in intersections:
            result.append(intersection.particleIds)

        return result






    def _normalizeValues(self, x):
        """Normalize values to range 0 to 1"""
        return (x-np.min(x))/(np.max(x)-np.min(x))

    def _findClosestPointToOrigin(self, x, y):
        """ Find the point closest to origin from normalied data
        """
        coors = list(zip(x, y))
        distances = np.linalg.norm(coors, axis=1)
        elbow_idx = np.argmin(distances)
        return elbow_idx

    def _findElbowAngle(self, x, y):
        """ Find the angle the slope of the function makes and
            return the point at which the slope changes from > 45º
            to <45º"""
        slopes = np.diff(y)/np.diff(x)
        angles = np.arctan(-slopes)
        elbow_idx = -1
        for i in range(len(angles)):
            if angles[i] < np.pi/4:
                elbow_idx = i
                break
        return elbow_idx, angles

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

    def _saveOutputs(self):
        rmScipionListWrapper = lambda x: list(x)
        rmScipionObjWrapper = lambda x: x.get()
        self._storeAttributeIfExists(self._getFileName('clusterings'), 'ensembleIntersectionLists', rmScipionListWrapper)
        self._storeAttributeIfExists(self._getFileName('objective_function'), 'ensembleObValues', rmScipionListWrapper)
        self._storeAttributeIfExists(self._getFileName('elbows'), 'elbows', rmScipionObjWrapper)
        self._storeAttributeIfExists(self._getFileName('random_consensus_sizes'), 'randomConsensusSizes', rmScipionListWrapper)
        self._storeAttributeIfExists(self._getFileName('random_consensus_size_ratios'), 'randomConsensusSizeRatios', rmScipionListWrapper)
        self._storeAttributeIfExists(self._getFileName('size_percentiles'), 'randomConsensusSizePercentiles', rmScipionObjWrapper)
        self._storeAttributeIfExists(self._getFileName('size_ratio_percentiles'), 'randomConsensusSizeRatioPercentiles', rmScipionObjWrapper)

    def _storeAttributeIfExists(self, filename, attribute, func=None):
        """ Saves an attribute identified by its name, 
            only if it exists. Returns true if successful """
        result = hasattr(self, attribute)

        if result:
            self._storeAttribute(filename, attribute, func)

        return result

    def _storeAttribute(self, filename, attribute, func=None):
        """ Saves an attribute identified by its name"""
        self._storeObject(filename, getattr(self, attribute), func)

    def _storeObject(self, filename, obj, func=None):
        """ Saves an object """
        path = self._getExtraPath(filename)

        # Apply a transformation if necessary
        if func is not None:
            obj = func(obj)

        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def _createOutput3DClassWithAttributes(self, clustering, name):
        randomConsensusSizes = getattr(self, 'randomConsensusSizes', None)
        randomConsensusRelativeSizes = getattr(self, 'randomConsensusSizeRatios', None)
        return self._createOutput3DClass(clustering, name, randomConsensusSizes, randomConsensusRelativeSizes)

    def _createOutput3DClass(self, clustering, name, randomConsensusSizes=None, randomConsensusRelativeSizes=None):
        inputParticles = self.inputMultiClasses[0].get().getImages()
        outputClasses = self._createSetOfClasses3D(inputParticles, suffix=name)

        # Fill the output
        loader = XmippProtConsensusClasses3D.ClassesLoader(
            clustering, 
            self.inputMultiClasses,
            randomConsensusSizes,
            randomConsensusRelativeSizes
        )
        loader.fillClasses(outputClasses)

        return outputClasses

    class ClassIntersection:
        """ Keeps track of the information related to successive class intersections.
            It is instantiated with a single class and allows to perform intersections
            and unions with it"""
        def __init__(self, particleIds, classificationIdx=None, clusterId=None, clusterSize=None, maxClusterSize=None):
            self.particleIds = set(particleIds)  # Particle ids belonging to this intersection

            self.representativeClassificationIndex = classificationIdx  # Classification index of the representative class
            self.representativeClusterId = clusterId  # The id of the representative class
            self.representativeClusterSize = clusterSize if clusterSize is not None else len(self.particleIds)  # The size of the representative class

            self.maxClusterSize = maxClusterSize if maxClusterSize is not None else self.representativeClusterSize  # Size of the biggest origin class

        def __len__(self):
            return len(self.particleIds)

        def intersect(self, other):
            # Select the data from the smallest representative cluster
            return self._combine(other, 'intersection', other.representativeClusterSize < self.representativeClusterSize)

        def merge(self, other):
            # Select the data from the largest intersection
            return self._combine(other, 'union', len(other.particleIds) > len(self.particleIds))

        def getSizeRatio(self):
            return len(self.particleIds) / self.maxClusterSize

        def _combine(self, other, op, rep):
            """ Base operations when combining two intersections.
                op is the member function of set used to combine particle ids
                rep is true if the representative class belongs to other"""

            # Ensure that the type is correct           
            if not isinstance(other, XmippProtConsensusClasses3D.ClassIntersection):
                raise TypeError('other must be of type ClassIntersection')

            # Determine the representative class
            selection = other if rep is True else self

            # Combine particle sets in the defined manner by op
            particleIds = getattr(self.particleIds, op)(other.particleIds)

            # Select the data from the representative class
            classificationIdx = selection.representativeClassificationIndex
            clusterId = selection.representativeClusterId
            clusterSize = selection.representativeClusterSize

            # Record the size of the biggest origin cluster for further analysis
            maxClusterSize = max(self.maxClusterSize, other.maxClusterSize)

            # Construct the new class
            return XmippProtConsensusClasses3D.ClassIntersection(particleIds, classificationIdx, clusterId, clusterSize, maxClusterSize)

    class ClassesLoader:
        """ Helper class to produce classes
        """
        def __init__(self, classes, inputClasses, randomConsensusSizes=None, randomConsensusRelativeSizes=None):
            self.classes = classes
            self.classification = self._createClassification(classes)
            self.representatives = self._createRepresentatives(classes, inputClasses)
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
                size = len(self.classes[classIdx])
                percentile = self._findPercentile(self.randomConsensusSizes, size)
                pValue = 1 - percentile
                setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_SIZE_PVALUE, Float(pValue))


            # Set the relative size p-value
            if self.randomConsensusRelativeSizes is not None:
                size = self.classes[classIdx].getSizeRatio()
                percentile = self._findPercentile(self.randomConsensusRelativeSizes, size)
                pValue = 1 - percentile
                setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_RELATIVE_SIZE_PVALUE, Float(pValue))

        def _createClassification(self, classes):
            # Fill the classification data (particle-class mapping)
            result = {}
            for cls, data in enumerate(classes):
                for particleId in data.particleIds:
                    result[particleId] = cls + 1 

            return collections.OrderedDict(sorted(result.items())) 

        def  _createRepresentatives(self, classes, inputClasses):
            result = [None]*len(classes)

            for cls, data in enumerate(classes):
                classificationIdx = data.representativeClassificationIndex
                clusterId = data.representativeClusterId
                classification = inputClasses[classificationIdx].get()
                cluster = classification[clusterId]
                result[cls] = cluster.getRepresentative()

            return result
        
        def _findPercentile(self, data, value):
            """ Given an array of values (data), finds the corresponding percentile of value.
                Percentile is returned in range [0, 1] """
            assert(sorted(data) == data)  # In order to iterate it in ascending order

            # Count the number of elements that are smaller than the given value
            i = 0
            while i < len(data) and data[i] < value:
                i += 1

            # Convert it into a percentage
            return float(i) / float(len(data))

    class RandomConsensus:
        def __call__(self, C, N):
            """ Obtains the intersections of a consensus
                of a random classification of sizes defined in C of
                N elements. C is a list of lists, where de sum of each
                row must equal N """

            # Create random partitions of same size
            randomClassification = self._makeRandomClassification(C, N)

            # Compute the repeated classifications
            return self._makeConsensus(randomClassification)

        def _makeRandomClassification(self, C, N):
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
                    Cip.append(x[first:first + s])
                    first += s

                # Add the random classification to the result
                Cp.append(Cip)

            return Cp

        def _makeConsensus(self, C):
            """ Computes the groups of elements that are equally
                classified for all the classifications of Cp """

            assert(len(C) > 0)  # There should be at least one classification

            # Convert the classification into a intersection list
            Cp = [[XmippProtConsensusClasses3D.ClassIntersection(i) for i in c] for c in C]

            # Initialize a list of sets containing the groups of the first classification
            S = Cp[0]

            # For the remaining classifications, compute the elements that repeatedly appear in the same group
            for i in range(1, len(Cp)):
                Sp = []
                for s1 in S:
                    for s2 in Cp[i]:
                        # Obtain only the elements in common for this combination
                        news = s1.intersect(s2)

                        # A group is only formed if non-empty
                        if news:
                            Sp.append(news)
                S = Sp

            return S