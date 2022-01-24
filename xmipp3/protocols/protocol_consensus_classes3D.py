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

from xmipp3.convert import setXmippAttribute

from pwem import emlib
from pwem.protocols import EMProtocol
from pwem.objects import Class3D

from pyworkflow.protocol.params import MultiPointerParam, EnumParam, IntParam
from pyworkflow.object import Float, List, Object
from pyworkflow.constants import BETA

import math
import pickle
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt


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
        myDict = {
            'objective_function': 'objective_func.pkl',
            'clusterings': 'clusterings.pkl',
            'elbows': 'elbows.pkl',
            'size_percentiles': 'size_percentiles.pkl',
            'size_ratio_percentiles': 'relative_size_percentiles.pkl',
        }
        self._updateFilenamesDict(myDict)

    def _defineParams(self, form):
        # Input data
        form.addSection(label='Input')

        form.addParam('inputMultiClasses', MultiPointerParam, important=True,
                      label="Input Classes", pointerClass='SetOfClasses3D',
                      help='Select several sets of classes where '
                           'to evaluate the intersections.')

        # Consensus
        form.addSection(label='Consensus clustering')

        form.addParam('manualClusterCount', IntParam, default=-1,
                      label='Manual Number of Clusters',
                      help='Set the final number of clusters. Disabled if <= 0')
        form.addParam('automaticClusterCount', EnumParam,
                      choices=['Yes', 'No'], default=0,
                      label='Guess the Number of Clusters', display=EnumParam.DISPLAY_HLIST,
                      help='Deduce the number of clusters based on the shape of the objective function')

        # Reference random classification
        form.addSection(label='Reference random classification')

        form.addParam('numRand', IntParam, default=0,
                      label='Number of random classifications',
                      help='Number of random classifications used for computing the significance of the actual '
                           'clusters. 0 for skipping this step')
        form.addParallelSection(threads=mp.cpu_count(), mpi=0)

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """ Inserting one step for each intersections analysis """

        # Intersect all input classes
        self._insertFunctionStep('populateStep')
        for i in range(1, len(self.inputMultiClasses)):
            self._insertFunctionStep('compareStep', i)

        # Perform a reference random classification if requested
        if self.numRand.get() > 0:
            self._insertFunctionStep('checkSignificanceStep')

        # Determine if ensemble is needed
        if self.manualClusterCount.get() > 0 or self.automaticClusterCount.get() == 0:
            self._insertFunctionStep('cossEnsembleStep')

        # Determine if automatic clustering is enabled
        if self.automaticClusterCount.get() == 0:
            self._insertFunctionStep('findElbowsStep')

        self._insertFunctionStep('createOutputStep')

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
            intersection = ClassIntersection(particleIds, classificationIdx, clusterId)

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
            intersector = ClassIntersection(particleIds1, classification1Idx, cluster1Id)

            for currIntersection in self.intersectionList:
                # Intersect the previous intersection with the intersector
                intersection = currIntersection.intersect(intersector)

                # Do not append classes that have no elements
                if intersection:
                    intersections.append(intersection)

        # Overwrite previous intersections with the new ones
        self.intersectionList = List(intersections)

    def cossEnsembleStep(self, numClusters=1):
        """ Ensemble clustering by COSS that merges interactions of clusters based
            on entropy minimization """
        # Obtain the classification as a list of list of sets
        classifications = get_classification_particle_ids(self.inputMultiClasses)

        # Initialize with values before merging
        allIntersections = [self.intersectionList]  # Stores the intersections of all iterations
        allObValues = [0.0]
        intersectionParticleIds = get_intersection_particle_ids(self.intersectionList)

        # Iterations of merging clusters
        while len(allIntersections[-1]) > numClusters:
            # Reshape the cluster matrix (list of sets instead of list of list of sets)
            allClusters = reshape_matrix_to_list(classifications)

            # Compute the similarity vectors
            similarityVectors = []
            for intersection in intersectionParticleIds:
                similarityVectors.append(build_simvector(intersection, allClusters))

            # For each possible pair of intersections compute the cost of merging them
            n = len(intersectionParticleIds)
            obValues = {}
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Objective function to minimize for each merged pair
                    obValues[(i, j)] = ob_function(intersectionParticleIds, [i, j], similarityVectors)

            values = list(obValues.values())
            keys = list(obValues.keys())

            # Select the minimum cost pair and merge them
            minObValue = min(values)
            mergePair = keys[values.index(minObValue)]
            mergedIntersections = merge_subsets(allIntersections[-1], mergePair)

            # Update the particle ids for the next iteration
            intersectionParticleIds = get_intersection_particle_ids(mergedIntersections)

            # Save the data for the next iteration
            allIntersections.append(mergedIntersections)
            allObValues.append(minObValue)

        # Reverse objective values so they go from largest to smallest
        allIntersections.reverse()
        allObValues.reverse()

        # Save the results
        self.ensembleIntersectionLists = List(allIntersections)
        self.ensembleObValues = List(allObValues)

    def findElbowsStep(self):
        """" Finds elbows of the COSS ensemble process """
        # Shorthands for variables
        numClusters = list(range(1, 1+len(self.ensembleObValues)))
        obValues = self.ensembleObValues

        # Get profile log likelihood for log of objective values
        # Remove last obValue as it is zero and log(0) is undefined
        pll = full_profile_log_likelihood(np.log(obValues[:-1]))

        # Normalize clusters and obValues
        normc = normalize(numClusters)
        normo = normalize(obValues)

        # Find different kinds of elbows
        elbow_idx_origin = find_closest_point_to_origin(normc, normo)
        elbow_idx_angle, _ = find_elbow_angle(normc, normo)
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
        clusterLengths = get_cluster_lengths(self.inputMultiClasses)

        # Calculate the total particle count
        numParticles = sum(clusterLengths[0])

        # Repeatedly obtain a consensus of a random classification of same size
        consensus = threadPool.starmap(
            random_consensus,
            [(clusterLengths, numParticles) for i in range(numExec)]
        )
        threadPool.close()
        consensus = reshape_matrix_to_list(consensus)

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
        # self._store()

        # Always output all the initial intersections
        outputClassesInitial = self._createOutput3DClass(self.intersectionList, 'initial')
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
                outputClassesManual = self._createOutput3DClass(self.ensembleIntersectionLists[i], 'manual')
                self._defineOutputs(outputClasses_manual=outputClassesManual)

                # Establish output relations
                for item in self.inputMultiClasses:
                    self._defineSourceRelation(item, outputClassesManual)

            # Check if automatic cluster count is enabled
            if automaticClusterCount == 0:
                elbows = self.elbows.get()
                for key, value in elbows.items():
                    outputClassesName = 'outputClasses_' + key
                    outputClasses = self._createOutput3DClass(self.ensembleIntersectionLists[value], key)
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
        max_nClusters = np.prod([len(self.inputMultiClasses[i].get()) for i in range(len(self.inputMultiClasses))])
        errors = []
        if len(self.inputMultiClasses) <= 1:
            errors = ["More than one Input Classes is needed to compute the consensus."]
        elif self.numClust.get() > max_nClusters:
            errors = ["Too many clusters selected for output"]
        elif self.numClust.get() < -1 or self.numClust.get() == 0:
            errors = ["Invalid number of clusters selected"]

        return errors

    # --------------------------- UTILS functions ------------------------------
    def _saveOutputs(self):
        rmScipionListWrapper = lambda x: list(x)
        rmScipionObjWrapper = lambda x: x.get()
        self._storeAttributeIfExists(self._getFileName('clusterings'), 'ensembleIntersectionLists', rmScipionListWrapper)
        self._storeAttributeIfExists(self._getFileName('objective_function'), 'ensembleObValues', rmScipionListWrapper)
        self._storeAttributeIfExists(self._getFileName('elbows'), 'elbows', rmScipionObjWrapper)
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

    def _createOutput3DClass(self, clustering, name):

        inputParticles = self.inputMultiClasses[0].get().getImages()
        outputClasses = self._createSetOfClasses3D(inputParticles, suffix=name)

        for classItem in clustering:
            # Shorthands for dictionary items and member variables
            particleIds = classItem.particleIds
            classificationIdx = classItem.representativeClassificationIndex
            clusterId = classItem.representativeClusterId
            classification = self.inputMultiClasses[classificationIdx].get()
            cluster = classification[clusterId]

            # Create a new cluster for output based on the current one
            newClass = Class3D()
            # newClass.copyInfo(cluster)
            newClass.setAcquisition(cluster.getAcquisition())
            newClass.setRepresentative(cluster.getRepresentative())

            # Calculate the size percentile for the cluster size
            if hasattr(self, 'randomConsensusSizes'):
                percentile = find_percentile(self.randomConsensusSizes, len(classItem))
                pValue = 1 - percentile
                setXmippAttribute(newClass, emlib.MDL_CLASS_INTERSECTION_SIZE_PVALUE, Float(pValue))

            # Calculate the relative size percentile for the cluster size
            if hasattr(self, 'randomConsensusSizeRatios'):
                percentile = find_percentile(self.randomConsensusSizeRatios, classItem.getSizeRatio())
                pValue = 1 - percentile
                setXmippAttribute(newClass, emlib.MDL_CLASS_INTERSECTION_RELATIVE_SIZE_PVALUE, Float(pValue))

            # Add all the particle IDs
            outputClasses.append(newClass)
            enabledClass = outputClasses[newClass.getObjId()]
            enabledClass.enableAppend()
            for particleId in particleIds:
                enabledClass.append(inputParticles[particleId])

            outputClasses.update(enabledClass)

        return outputClasses


class ClassIntersection:
    """ Keeps track of the information related to successive class intersections.
        It is instantiated with the """
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
        return ClassIntersection(particleIds, classificationIdx, clusterId, clusterSize, maxClusterSize)


######################################
#        Helper functions            #
######################################

def reshape_matrix_to_list(m):
    """ Given a list of lists, converts it into a list with all its elements """
    result = []
    for row in m:
        result += row

    return result


def is_sorted(values):
    """ Returns true if a list is sorted or empty """
    return all(values[i] <= values[i+1] for i in range(len(values)-1))


def find_percentile(data, value):
    """ Given an array of values (data), finds the corresponding percentile of value.
        Percentile is returned in range [0, 1] """
    assert(is_sorted(data))  # In order to iterate it in ascending order

    # Count the number of elements that are smaller than the given value
    i = 0
    while i < len(data) and data[i] < value:
        i += 1

    # Convert it into a percentage
    return float(i) / float(len(data))


def get_cluster_lengths(scipionMultiClasses):
    """ Returns a list of lists that stores the lengths of each classification """
    result = []

    for i in range(len(scipionMultiClasses)):
        classification = scipionMultiClasses[i].get()

        result.append([])
        for cluster in classification:
            result[-1].append(len(cluster))

    return result


def get_intersection_particle_ids(intersections):
    """ Return the list of sets from a list of intersections """
    result = []

    for intersection in intersections:
        result.append(intersection.particleIds)

    return result


def get_classification_particle_ids(scipionMultiClasses):
    """ Returns the list of lists of sets that stores the clusters of each classification """
    result = []

    for i in range(len(scipionMultiClasses)):
        classification = scipionMultiClasses[i].get()

        result.append([])
        for cluster in classification:
            result[-1].append(cluster.getIdSet())

    return result


def merge_subsets(intersections, indices):
    """ Return a list of subsets where the intersections[indices] have been merged """
    # Divide the input intersections into the ones referred by indices and not
    selection = [intersections[i] for i in indices]  # intersection[indices]
    result = [intersections[i] for i in range(len(intersections)) if i not in indices]  # intersections - selection

    # Merge all the selected clusters into the same one
    merged = selection[0]
    for i in range(1, len(selection)):
        merged = merged.merge(selection[i])

    # Add the merged items to the result
    result.append(merged)
    return result


######################################
# COSS functions. See notes 8/4/2019 #
######################################

def nc_similarity(n, c):
    """ Return the similarity of intersection subset (n) with class subset (c) """
    inter = len(n.intersection(c))
    return inter / len(n)


def entropy(p):
    """ Return the entropy of the elements in a vector """
    H = 0
    for i in range(len(p)):
        H += p[i] * math.log(p[i], 2)
    return -H


def build_simvector(n, cs):
    """ Build the similarity vector of n subset with each c subset in cs """
    v = []
    for c in cs:
        v.append(nc_similarity(n, c))
    return v


def calc_distribution(ns, indexs=[]):
    """ Return the vector of distribution (p) from the n subsets in ns
        If two indexs are given, these two subsets are merged in the distribution """
    p = np.array([])
    if len(indexs) == 2:
        # Merging two subsets
        N = len(ns) - 1
        for i in range(len(ns)):
            if i not in indexs:
                p = np.append(p, len(ns[i]))
        p = np.append(p, len(ns[indexs[0]].union(ns[indexs[1]])))
    else:
        N = len(ns)
        for n in ns:
            p = np.append(p, len(n))
    return p / N


def vectors_similarity(v1, v2):
    """ Return the cosine similarity of two vectors that is used as similarity """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def ob_function(ns, indexs, vs, eps=0.01):
    """Objective function to minimize based on the entropy and the similarity between the clusters to merge
    """
    p = calc_distribution(ns)
    pi = calc_distribution(ns, indexs)
    h = entropy(p)
    hi = entropy(pi)

    sv = vectors_similarity(vs[indexs[0]], vs[indexs[1]])
    return (h - hi) / (sv + eps)


def normalize(x):
    """Normalize values to range 0 to 1"""
    return (x-np.min(x))/(np.max(x)-np.min(x))


def find_closest_point_to_origin(x, y):
    """ Find the point closest to origin from normalied data
    """
    coors = list(zip(x, y))
    distances = np.linalg.norm(coors, axis=1)
    elbow_idx = np.argmin(distances)
    return elbow_idx


def find_elbow_angle(x, y):
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


def mle_estimates(d, q):
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


def fg(d, theta):
    """ Gaussian pdf """
    m = theta[0]
    s = theta[1]
    gauss = 1/(np.sqrt(2*np.pi*s))*np.exp(-1/(2*s)*(d-m)**2)
    return gauss


def profile_log_likelihood(d, q, theta1, theta2):
    """ Profile log likelihood for given parameters """
    log_f_q = np.log(fg(d[:q], theta1))
    log_f_p = np.log(fg(d[q:], theta2))
    l_q = np.sum(log_f_q) + np.sum(log_f_p)
    return l_q


def full_profile_log_likelihood(d):
    """ Calculate profile log likelihood for each partition
        of the data """
    pll = []
    for q in range(1, len(d)):
        theta1, theta2 = mle_estimates(d, q)
        pll.append(profile_log_likelihood(d, q, theta1, theta2))
    return pll


def coss_random_classification(C, N):
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


def coss_consensus(C):
    """ Computes the groups of elements that are equally
        classified for all the classifications of Cp """

    assert(len(C) > 0)  # There should be at least one classification

    # Convert the classification into a intersection list
    Cp = [[ClassIntersection(i) for i in c] for c in C]

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


def random_consensus(C, N):
    """ Obtains the intersections of a consensus
        of a random classification of sizes defined in C of
        N elements. C is a list of lists, where de sum of each
        row must equal N """

    # Create random partitions of same size
    randomClassification = coss_random_classification(C, N)

    # Compute the repeated classifications
    return coss_consensus(randomClassification)


#############################
# TODO remove as deprecated #
#############################

def create_nsubset(c1, c2):
    """ Return the intersection of two sets """
    c1 = set(c1)
    c2 = set(c2)
    return c1.intersection(c2)


def adjust_index(nsol, pair_min):
    """Adjust the index given that same cluster count only as 1
    """
    isol = [0]
    for i in range(1, len(nsol)):
        if nsol[i] == nsol[i - 1]:
            isol.append(isol[-1])
        else:
            isol.append(isol[-1] + 1)
    npair = [isol.index(pair_min[0]), isol.index(pair_min[1])]
    return npair


def get_vs(us, cs):
    """Return the vector of similarities between the subgroups (us) and the clusters (cs)
    """
    list_cs = []
    for c in cs:
        list_cs += c
    vs = []
    for u in us:
        vs.append(build_simvector(u, list_cs))
    return vs


def n_clusters(all_coss_us):
    """Find number of clusters in each set"""
    nclusters = []
    for coss_us in all_coss_us:
        nclusters.append(len(coss_us))

    return nclusters


def clusterings_as_sets(rs, nclust, Y):
    """ Creating class clusters: list of lists(clusterings) of sets
        (each set is a cluster with the names of objects)
        From a list of lists(clusterings) where the position is the object
        and the value the cluster """
    cs = []
    for _ in range(len(rs)):
        cs.append([set() for _ in range(nclust)])

    for i in range(len(Y)):
        for j in range(len(cs)):
            cs[j][rs[j][i]].add(i)
    return cs


def intersection_clusters(cs):
    """ Return the intersections between the clusters of
        different clusterings """
    us = []
    for c in cs:
        if len(us) == 0:
            for setc in c:
                us.append(setc)
        else:
            aux = []
            for setc in c:
                for setu in us:
                    aux.append(setu.intersection(setc))
            us = aux

    # Removing empty sets from us
    while set() in us:
        us.remove(set())
    return us


def us2labels(us, lenX):
    """From merged us it returns the labels of the elements"""
    # Saving labels
    coss_labels = [-1 for _ in range(lenX)]
    for ic, clus in enumerate(us):
        for idx in clus:
            coss_labels[idx] = ic
    return coss_labels


def find_function_elbow(values, xs=None, ploti=False):
    """Finds the point where the maximum change in the slope of the function draw from
    some values. Returns only non-convex elbows.
    """
    if xs == None:
        xs = list(range(len(values)))

    slopes = []
    for i in range(1, len(values)):
        slopes.append((values[i] - values[i - 1]) / (xs[i] - xs[i - 1]))
    # print(slopes)
    chslopes = []
    for i in range(1, len(slopes)):
        chslopes.append((slopes[i] - slopes[i - 1]) / (xs[i] - xs[i - 1]))
    # print(chslopes)
    elbow_idx = chslopes.index(max(chslopes)) + 1
    # print(elbow)
    if ploti:
        plt.plot(xs, values)
        plt.axvline(xs[elbow_idx], color='red')
        plt.show()
    return elbow_idx

def plot_all_coss(all_coss_us, ob_values, outimage):
    """Plots the objective function (norm-log) and the rand score of all number of clusters
    of the coss greedy ensemble clustering
    """
    nclusters = []
    for coss_us in all_coss_us:
        nclusters.append(len(coss_us))

    nclusters = list(reversed(nclusters))
    ob_values = list(reversed(ob_values))

    # Plot objective funtion norm.log values
    #logob_values = np.log(ob_values)
    #normob_values = ob_values / np.linalg.norm(ob_values)
    #lognormob_values = np.log(ob_values) / np.linalg.norm(np.log(ob_values))

    plt.plot(nclusters, ob_values)
    elbow_idx = find_function_elbow(ob_values)

    plt.scatter([nclusters[elbow_idx]], [ob_values[elbow_idx]], color='green')
    plt.xlabel('Number of clusters')
    plt.ylabel('Objective values')
    plt.title('Objective values for each number of clusters')
    plt.savefig(outimage)
    #plt.show()
    return nclusters, ob_values, elbow_idx
