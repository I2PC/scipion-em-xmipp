# **************************************************************************
# *
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              Daniel Del Hoyo Gomez (daniel.delhoyo.gomez@alumnos.upm.es)
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

import os
import numpy as np
import Cluster_Ensembles as CE
from pyworkflow.em.protocol.protocol_3d import ProtAnalysis3D
from pyworkflow.protocol.params import PointerParam, StringParam, FloatParam
import sys
import tables

import sklearn.datasets as skdt
import sklearn.cluster as skcl
import sklearn.metrics as skme
from math import log
import matplotlib.pyplot as plt

#############
# Benchmark functions
def read_benchmark(fn):
    '''Return the dictionary {X:np.array(), Y:np.array, classes: number_of_classes}
    '''
    x, y = [], []
    with open(fn) as filex:
        for line in filex:
            x.append(list(map(float, line.strip().split()[:-1])))
            y.append(int(line.strip().split()[-1]))
    x = np.asarray(x)
    y = np.asarray(y)
    # Make first label=0
    mini = np.min(y)
    y -= mini
    return {'data': x, 'target': y, 'classes': len(set(y))}

def load_benchmarks(bs=[]):
    '''Return dictionary of up to 8 dictionaries with the benchmark clustering data
    {X:np.array(), Y:np.array, classes: number_of_classes}
    '''
    bm_path = '/home/daniel/Desktop/ensemble_clustering/benchmarks/'
    all_bs = os.listdir(bm_path)
    if bs == []:
        bs = all_bs

    bmark_dic = {}
    for b in all_bs:
        if b in bs:
            bmark_dic[b.replace('.txt', '')] = read_benchmark(bm_path + b)
    return bmark_dic
#################################
# COSS functions
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
        H += p[i] * log(p[i], 2)
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
    '''Return the scalar product of two vectors that is used as similarity
    '''
    return np.dot(v1, v2)

def merge_subsets(ns, indexs):
    '''Return a list of subsets where the subsets[index] have been merged
    '''
    us = []
    for i in range(len(ns)):
        if i not in indexs:
            us.append(ns[i])
    us.append(ns[indexs[0]].union(ns[indexs[1]]))
    return us

def ob_function(ns, indexs, vs, eps=0.01):
    '''Objective function to minimize
    '''
    p = calc_distribution(ns)
    pi = calc_distribution(ns, indexs)
    # print('p: ',p)
    # print('pi: ',pi)

    h = entropy(p)
    hi = entropy(pi)
    # print('h: ',h)
    # print('hi: ',hi)

    sv = vectors_similarity(vs[indexs[0]], vs[indexs[1]])
    # print('sv: ',sv)

    return (h - hi) / (sv + eps)

def clusterings_as_sets(rs,nclust,Y):
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

def coss_ensemble(rs,nclust,X,Y):
    '''Ensemble clustering by COSS that merges interections of clusters based on entropy minimization
    '''
    # Creating set based clusterings
    cs = clusterings_as_sets(rs,nclust,Y)
    # Creating intersection clusters (us)
    us = intersection_clusters(cs)

    # Iterations of merging clusters
    while len(us) > nclust:
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
        pair_min = keys[values.index(min(values))]
        us = merge_subsets(us, pair_min)

    # Saving labels
    coss_labels = [-1 for _ in range(len(X))]
    for ic, clus in enumerate(us):
        for idx in clus:
            coss_labels[idx] = ic
    return coss_labels

#############################################
#############################################


class XmippConsensusClustering(ProtAnalysis3D):
    """
    Create a consensus clustering of the volumes clusterings
    """
    _label = 'consensus clustering'

    def __init__(self, **args):
        ProtAnalysis3D.__init__(self, **args)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        pass

    ##################################
    # Plotting functions
    def plot_clusts(self, X, list_labs, b=0, nclasses=3):
        clust_titles = ['Real', 'Kmeans', 'Ward', 'Spectral', 'COSS', 'CE']
        plt.figure(figsize=(90, 90))
        col = 1
        for il, lab in enumerate(list_labs):
            plt.subplot(int('1{}{}'.format(len(list_labs), col)))
            plt.title(clust_titles[il])
            for i in range(nclasses):
                plt.plot(X[lab == i, 0], X[lab == i, 1], '.', label='class {}'.format(i))
            # plt.legend()
            col += 1
        # plt.show()
        plt.savefig(self._getPath("clustering_{}.png".format(b)))
        plt.show()

    def main(self):
        # Downloading dataset and true labels
        datasets = load_benchmarks()

        for ib,d in enumerate(datasets):
            print('\nDataset: ', d)
            sys.stdout.flush()
            X = datasets[d]['data']
            Y = datasets[d]['target']
            nclust = datasets[d]['classes']

            # Executing multiple clusterings
            rs = []
            rs.append(skcl.KMeans(nclust, random_state=0, n_init=1).fit(X).labels_)
            # rs.append(skcl.KMeans(nclust,random_state=1,n_init=1).fit(X).labels_)
            rs.append(skcl.AgglomerativeClustering(n_clusters=nclust, affinity='euclidean',
                                                   linkage='ward').fit_predict(X))
            rs.append(skcl.SpectralClustering(nclust, n_init=1, assign_labels='discretize').fit_predict(X))
            # rs.append(skcl.AgglomerativeClustering(n_clusters=nclust, affinity='euclidean',
            # linkage='single').fit_predict(X))
            ##rs.append(skcl.AgglomerativeClustering(n_clusters=nclust, affinity='euclidean',
            ##                                  linkage='average').fit_predict(X))

            ###################

            # Ensemble clustering by COSS method
            coss_labels = coss_ensemble(rs,nclust,X,Y)
            cl = CE.cluster_ensembles(np.array(rs), verbose=False)

            ##    #Silhouette score as a external measure of clustering performance
            ##    print('\nSil Real: ',skme.silhouette_score(X,Y))
            ##    for i,r in enumerate(rs):
            ##        print('Sil {}: '.format(i),skme.silhouette_score(X,r))
            ##    print('Sil coss: ',skme.silhouette_score(X,coss_labels))
            ##    print()

            # Consensus with real classification labels
            for i, r in enumerate(rs):
                print('Sim_to_real {}: '.format(i), skme.adjusted_rand_score(Y, r))
            print('Sim_to_real coss: ', skme.adjusted_rand_score(Y, coss_labels))
            print('Sim_to_real CE: ', skme.adjusted_rand_score(Y, cl))

            # Plotting clustering
            self.plot_clusts(X, [Y] + rs + [np.array(coss_labels)] + [np.array(cl)], ib, nclasses=nclust)


    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        cluster_runs = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0]])
        #self._insertFunctionStep("clusteringEnsemble", cluster_runs)
        self._insertFunctionStep("main")

    def clusteringEnsemble(self,cluster_runs):
        cluster_runs = np.random.randint(0, 3, (15, 1000))

        cl = CE.cluster_ensembles(cluster_runs,verbose=False)

        print(cl)
        sys.stdout.flush()







