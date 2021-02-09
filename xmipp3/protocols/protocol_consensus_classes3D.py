# **************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *              Daniel del Hoyo Gomez (daniel.delhoyo.gomez@alumnos.upm.es)
# *              Jorge Garcia Condado (jgcondado@cnb.csic.es)
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

from pwem.protocols import EMProtocol
from pyworkflow.protocol.params import MultiPointerParam, EnumParam, IntParam
from pwem.objects import Class3D
from scipy.cluster import hierarchy

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

class XmippProtConsensusClasses3D(EMProtocol):
    """ Compare several SetOfClasses3D.
        Return the consensus clustering based on a objective function that uses the similarity between clusters
        intersections and the entropy of the clustering formed.
    """
    _label = 'consensus clustering 3D'

    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMultiClasses', MultiPointerParam, important=True,
                      label="Input Classes", pointerClass='SetOfClasses3D',
                      help='Select several sets of classes where '
                           'to evaluate the intersections.')
        form.addParam('doConsensus', EnumParam,
                      choices=['Yes', 'No'], default=0,
                      label='Get Consensus Clustering', display=EnumParam.DISPLAY_HLIST,
                      help='Use coss ensemble method for obtaing a consensus clustering')
        form.addParam('numClust', IntParam, default=-1,
                      label='Set Number of Clusters',
                      help='Set the final number of clusters. If -1, deduced by the shape of the objective function')


    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """ Inserting one step for each intersections analisis
        """

        self._insertFunctionStep('compareFirstStep',
                                 self.inputMultiClasses[0].get().getObjId(),
                                 self.inputMultiClasses[1].get().getObjId())

        if len(self.inputMultiClasses) > 2:
            for i in range(2, len(self.inputMultiClasses)):
                self._insertFunctionStep('compareOthersStep', i,
                                         self.inputMultiClasses[i].get().getObjId())
        if self.doConsensus.get() == 0:
            self._insertFunctionStep('cossEnsembleStep')

        self._insertFunctionStep('createOutputStep')

    def compareFirstStep(self, objId1, objId2):
        """ We found the intersections for the two firsts sets of classes
        """
        set1Id = 0
        set2Id = 1
        set1 = self.inputMultiClasses[set1Id].get()
        set2 = self.inputMultiClasses[set2Id].get()

        print('Computing intersections between classes form set %s and set %s:'
              % (set1.getNameId(), set2.getNameId()))

        newList = []
        for cls1 in set1:
            cls1Id = cls1.getObjId()
            ids1 = cls1.getIdSet()

            for cls2 in set2:
                cls2Id = cls2.getObjId()
                ids2 = cls2.getIdSet()

                interTuple = self.intersectClasses(set1Id, cls1Id, ids1,
                                                   set2Id, cls2Id, ids2)
                # Do not append classes that have no elements
                if interTuple[0] != 0:
                    newList.append(interTuple)

        self.intersectsList = newList

    def compareOthersStep(self, set1Id, objId):
        """ We found the intersections for the two firsts sets of classes
        """
        set1 = self.inputMultiClasses[set1Id].get()

        print('Computing intersections between classes form set %s and '
              'the previous ones:' % (set1.getNameId()))

        newList = []
        currDB = self.intersectsList
        for cls1 in set1:
            cls1Id = cls1.getObjId()
            ids1 = cls1.getIdSet()

            for currTuple in currDB:
                ids2 = currTuple[1]
                set2Id = currTuple[2]
                cls2Id = currTuple[3]
                clSize = currTuple[4]

                interTuple = self.intersectClasses(set1Id, cls1Id, ids1,
                                                   set2Id, cls2Id, ids2, clSize)

                # Do not append classes that have no elements
                if interTuple[0] != 0:
                    newList.append(interTuple)

        self.intersectsList = newList

    def cossEnsembleStep(self):
        """ Ensemble clusterig with coss method
        """
        cs = self.get_cs(self.inputMultiClasses)
        all_coss_us, ob_values = self.coss_ensemble(cs, min_nclust=1)

        # Plot dendogram
        self.plot_dendogram(ob_values, self._getExtraPath())

        # Get number of clusters at each step
        nclusters = n_clusters(all_coss_us)

        # Reverse objective calues so they go from largest to smallest
        ob_values = list(reversed(ob_values))

        # Get profile log likelihood for log of objective values
        # Remove last obvalue as it is zero and log(0) is undefined
        pll = full_profile_log_likelihood(np.log(ob_values[:-1]))

        # Normalize clusters and obvalues
        normc, normo = normalize(nclusters, ob_values)

        # Find different kinds of elbows
        elbow_idx_origin = find_closest_point_to_origin(normc, normo)
        elbow_idx_angle, _ = find_elbow_angle(normc, normo)
        elbow_idx_pll = np.argmax(pll)
        self.elbows = [[elbow_idx_origin, 'origin'], [elbow_idx_angle, 'angle'],
                  [elbow_idx_pll, 'pll']]

        for elbow in self.elbows:
            print(elbow[1]+':'+str(elbow[0]+1))

        # Plot all indexes ontop of objective function
        plot_function_and_elbows(nclusters, ob_values, self.elbows, self._getExtraPath()+'/objective_function_plot.png')

        # Store values of objective function
        self._objectiveFData = [nclusters,ob_values]
        self.nclusters = nclusters

        # TODO how do we set this
        self.elbowIdx = elbow_idx_angle

        nclust = self.numClust.get()
        if  nclust == -1:
            self.intersectsList = self.all_iLists[self.elbowIdx]
        else:
            self.intersectsList = self.all_iLists[nclusters.index(nclust)]

        print('Saving best number of clusters detected: ', nclusters[self.elbowIdx])

        self.save_outputs()

    def createOutputStep(self):
        '''Saves the outputs into the pickle files'''
        inputParticles = self.inputMultiClasses[0].get().getImages()
        outputClasses = self._createSetOfClasses3D(inputParticles)

        clustering = self.intersectsList

        for classItem in clustering:
            numOfPart = classItem[0]
            partIds = classItem[1]
            setRepId = classItem[2]
            clsRepId = classItem[3]

            setRep = self.inputMultiClasses[setRepId].get()
            clRep = setRep[clsRepId]

            newClass = Class3D()
            # newClass.copyInfo(clRep)
            newClass.setAcquisition(clRep.getAcquisition())
            newClass.setRepresentative(clRep.getRepresentative())

            outputClasses.append(newClass)

            enabledClass = outputClasses[newClass.getObjId()]
            enabledClass.enableAppend()
            for itemId in partIds:
                enabledClass.append(inputParticles[itemId])

            outputClasses.update(enabledClass)

        self._defineOutputs(outputClasses=outputClasses)
        for item in self.inputMultiClasses:
            self._defineSourceRelation(item, outputClasses)

    # --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []
        for elbow in self.elbows:
            summary.append(elbow[1]+':'+str(elbow[0]+1))
        return summary

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        errors = [] if len(self.inputMultiClasses) > 1 else \
            ["More than one Input Classes is needed to compute the consensus."]
        return errors

    # --------------------------- UTILS functions ------------------------------
    def save_outputs(self):
        self.saveClusteringsList()
        self.saveObjectiveFData()
        self.saveElbowIndex()

    def saveClusteringsList(self):
        '''Saves the list of clusterings with different number of clusters into a pickle file'''
        savepath = self._getExtraPath('clusterings.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(self.all_iLists, f)

    def saveObjectiveFData(self):
        '''Save the data of the objective function'''
        savepath = self._getExtraPath('ObjectiveFData.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(self._objectiveFData, f)

    def saveElbowIndex(self):
        '''Save the calculated number of clusters where the function has an elbow'''
        savepath = self._getExtraPath('elbowclusters.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(self.elbowIdx, f)

    def intersectClasses(self, setId1, clId1, ids1,
                         setId2, clId2, ids2, clsSize2=None):
        size1 = len(ids1)
        size2 = len(ids2) if clsSize2 is None else clsSize2

        inter = ids1.intersection(ids2)

        if size1 < size2:
            setId = setId1
            clsId = clId1
            clsSize = size1
        else:
            setId = setId2
            clsId = clId2
            clsSize = size2

        return (len(inter), inter, setId, clsId, clsSize)

    def get_cs(self, scipionMultiClasses):
        '''Returns the list of lists of sets that stores the clusters of each classification'''
        cs=[]
        for i in range(len(scipionMultiClasses)):
            clasif = scipionMultiClasses[i].get()
            cs.append([])
            for cluster in clasif:
                cs[-1].append(cluster.getIdSet())
        return cs

    def get_us(self, scipionIntersectList):
        '''Return the list of sets from the scipion list of intersections'''
        us=[]
        for cur_tuple in scipionIntersectList:
            us.append(cur_tuple[1])
        return us

    def coss_ensemble(self, cs, min_nclust=2):
        '''Ensemble clustering by COSS that merges interections of clusters based on entropy minimization
        '''
        # Creating intersection clusters (us)
        iList = self.intersectsList
        us = self.get_us(iList)

        # Initialize with values before merging
        all_us = [us]
        self.all_iLists = [iList]
        ob_values = [0.0]

        # Iterations of merging clusters
        while len(us) > min_nclust:
            # Similarity vectors
            all_cs = []
            for c in cs:
                all_cs += c
            vs = []
            for u in us:
                vs.append(build_simvector(u, all_cs))

            # For each posible pair of us
            n = len(us)
            fob = {}
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Objective function to minimize for each merged pair
                    fob[(i, j)] = ob_function(us, [i, j], vs)

            values = list(fob.values())
            keys = list(fob.keys())

            # Index of subsets that minimize the objective function by merging
            ob_values.append(min(values))
            pair_min = keys[values.index(min(values))]
            iList = merge_subsets(iList, pair_min)
            us = self.get_us(iList)
            all_us.append(us)
            self.all_iLists.append(iList)
        self.all_iLists = list(reversed(self.all_iLists))
        return all_us, ob_values

    def plot_dendogram(self, obvalues, outimage):
        ''' Plot the dendogram from the objective functions of the merge
        between the groups of images
        '''
        # Initzialize required values
        allilists = self.all_iLists.copy()
        allilists.reverse()
        linkage_matrix = np.zeros((len(allilists)-1, 4))
        set_ids = np.arange(len(allilists))
        original_num_sets = len(allilists)

        # loop over each iteration of clustering
        for i in range(len(allilists)-1):
            # find the two sets that were merged
            sets_merged = []
            for set_info, set_id in zip(allilists[i], set_ids):
                if set_info not in allilists[i+1]:
                    sets_merged.append(set_id)

            # find original number of sets within new set
            if sets_merged[0] - original_num_sets < 0:
                n1 = 1
            else:
                n1 = linkage_matrix[sets_merged[0]-original_num_sets, 3]
            if sets_merged[1] - original_num_sets < 0:
                n2 = 1
            else:
                n2 = linkage_matrix[sets_merged[1]-original_num_sets, 3]

            # create linkage matrix
            linkage_matrix[i, 0] = sets_merged[0] # set id of merged set
            linkage_matrix[i, 1] = sets_merged[1] # set id of merged set
            linkage_matrix[i, 2] = obvalues[i+1]  # objective function as distance
            linkage_matrix[i, 3] = n1+n2 # total number of oiginal sets

            # change set ids to reflect new set of clusters
            set_ids = np.delete(set_ids, np.argwhere(set_ids == sets_merged[0]))
            set_ids = np.delete(set_ids, np.argwhere(set_ids == sets_merged[1]))
            set_ids = np.append(set_ids, len(allilists)+i)

        # Plot resulting dendogram
        plt.figure()
        dn = hierarchy.dendrogram(linkage_matrix)
        plt.title('Dendogram')
        plt.xlabel('sets ids')
        plt.ylabel('objective function')
        plt.tight_layout()
        plt.savefig(outimage+'/dendogram.png')

        # Plot resulting dendogram with log scale
        plt.yscale('log')
        plt.ylim([np.min(linkage_matrix[:, 2]), np.max(linkage_matrix[:,2])])
        plt.savefig(outimage+'/dendogram_log.png')

#################################
# COSS functions. See notes 8/4/2019
def nc_similarity(n, c):
    '''Return the similarity of intersection subset (n) with class subset (c)
    '''
    inter = len(n.intersection(c))
    return inter / len(n)

def entropy(p):
    '''Return the entropy of the elements in a vector
    '''
    H = 0
    for i in range(len(p)):
        H += p[i] * math.log(p[i], 2)
    return -H

def create_nsubset(c1, c2):
    '''Return the intersection of two sets
    '''
    c1 = set(c1)
    c2 = set(c2)
    return c1.intersection(c2)

def build_simvector(n, cs):
    '''Build the similarity vector of n subset with each c subset in cs
    '''
    v = []
    for c in cs:
        v.append(nc_similarity(n, c))
    return v

def calc_distribution(ns, indexs=[]):
    '''Return the vector of distribution (p) from the n subsets in ns
    If two indexs are given, these two subsets are merged in the distribution
    '''
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
    '''Return the cosine similarity of two vectors that is used as similarity
    '''
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def merge_subsets(iList, indexs):
    '''Return a list of subsets where the subsets[index] have been merged
    '''
    niList = []
    for i in range(len(iList)):
        if i not in indexs:
            niList.append(iList[i])

    #Clusters with specified index are merged
    cl0 = iList[indexs[0]][1]
    cl1 = iList[indexs[1]][1]
    union = cl0.union(cl1)

    #Representative info is chosen from the previous bigger cluster
    if len(cl0) > len(cl1):
        rep = 0
    else:
        rep = 1
    clasId = iList[indexs[rep]][2]
    clusId = iList[indexs[rep]][3]
    prevClusLen = iList[indexs[rep]][4]

    niList.append((len(union), union, clasId, clusId, prevClusLen))

    return niList

def ob_function(ns, indexs, vs, eps=0.01):
    '''Objective function to minimize based on the entropy and the similarity between the clusters to merge
    '''
    p = calc_distribution(ns)
    pi = calc_distribution(ns, indexs)
    h = entropy(p)
    hi = entropy(pi)

    sv = vectors_similarity(vs[indexs[0]], vs[indexs[1]])
    return (h - hi) / (sv + eps)

def clusterings_as_sets(rs, nclust, Y):
    '''Creating class clusters: list of lists(clusterings) of sets
    (each set is a cluster with the names of objects)
    From a list of lists(clusterings) where the position is the object
    and the value the cluster
    '''
    cs = []
    for _ in range(len(rs)):
        cs.append([set() for _ in range(nclust)])

    for i in range(len(Y)):
        for j in range(len(cs)):
            cs[j][rs[j][i]].add(i)
    return cs

def intersection_clusters(cs):
    '''Return the intersections between the clusters of different clusterings
    '''
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
    '''From merged us it returns the labels of the elements'''
    # Saving labels
    coss_labels = [-1 for _ in range(lenX)]
    for ic, clus in enumerate(us):
        for idx in clus:
            coss_labels[idx] = ic
    return coss_labels

def adjust_index(nsol, pair_min):
    '''Adjust the index given that same cluster count only as 1
    '''
    isol = [0]
    for i in range(1, len(nsol)):
        if nsol[i] == nsol[i - 1]:
            isol.append(isol[-1])
        else:
            isol.append(isol[-1] + 1)
    npair = [isol.index(pair_min[0]), isol.index(pair_min[1])]
    return npair

def get_vs(us, cs):
    '''Return the vector of similarities between the subgroups (us) and the clusters (cs)
    '''
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

    # Reverse to have smallest to biggest
    nclusters = list(reversed(nclusters))

    return nclusters

def normalize(x, y):
    """Normalize values to range 0 to 1"""
    normx = (x-np.min(x))/(np.max(x)-np.min(x))
    normy = (y-np.min(y))/(np.max(y)-np.min(y))

    return normx, normy

def find_closest_point_to_origin(x, y):
    """ Find the point closest to origin from normalied data
    """
    coors = list(zip(x, y))
    distances = np.linalg.norm(coors, axis=1)
    elbow_idx = np.argmin(distances)
    return elbow_idx

def find_elbow_angle(x, y):
    """Find the angle the slope of the function makes and
    return the point at which the slope changes from > 45º
    to <45º"""
    slopes = np.diff(y)/np.diff(x)
    angles = np.arctan(-slopes)
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

def plot_function_and_elbows(nclusters, ob_values, elbows, outimage):
    '''Plots the objective function and elbows
    '''

    plt.figure()
    plt.plot(nclusters, ob_values)
    for elbow_idx, name in elbows:
        plt.scatter([nclusters[elbow_idx]], [ob_values[elbow_idx]], label=name + ': '+str(nclusters[elbow_idx]))
    plt.legend()
    plt.xlabel('Number of clusters')
    plt.ylabel('Objective values')
    plt.title('Objective values for each number of clusters')
    plt.tight_layout()
    plt.savefig(outimage)

# TODO remove as deprecated

def find_function_elbow(values, xs=None, ploti=False):
    '''Finds the point where the maximum change in the slope of the function draw from
    some values. Returns only non-convex elbows.
    '''
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
    '''Plots the objective function (norm-log) and the rand score of all number of clusters
    of the coss greedy ensemble clustering
    '''
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
