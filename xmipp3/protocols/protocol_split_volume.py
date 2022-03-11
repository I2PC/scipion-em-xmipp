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
from pwem.constants import (ALIGN_NONE, ALIGN_2D, ALIGN_3D, ALIGN_PROJ)

import xmippLib
from xmipp3.convert import writeSetOfParticles, writeSetOfVolumes, readSetOfVolumes

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
        self.stepsExecutionMode = STEPS_PARALLEL
        self._createFilenames()

    def _createFilenames(self):
        """ Centralize the names of the files. """
        classFmt='c%(cls)02d'
        suffixFmt='%(suffix)s'

        myDict = {
            'distances': 'distances.csv',
            'correlations': 'correlations.csv',
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
        form.addParam('maxComparisonCount', IntParam, label="Maximum number of classes to be compared", 
                      validators=[GT(0)], default=8,
                      help='Number of classes among which the aligned correlation is computed.')

        form.addSection(label='Graph partition')
        form.addParam('graphPartitionMethod', EnumParam, label="Graph partition method", 
                      choices=['Spectral',  'Minimum cut'], default=0,
                      help='The method used for partitioning the graph')
        form.addParam('graphPartitionCount', IntParam, label="Number of components", 
                      validators=[GE(2)], default=2,
                      help='Number of graphs to obtain')
        form.addParallelSection()
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('computeAngularDistancesStep')
        self._insertFunctionStep('computeCorrelationStep')
        self._insertFunctionStep('graphPartitionStep')
        self._insertFunctionStep('createOutputStep')

    #--------------------------- STEPS functions ---------------------------------------------------
    def convertInputStep(self):
        self._particleList = self._getParticleList(self.directionalClasses.get())

    def computeAngularDistancesStep(self):
        # Perform the computation
        distances = self._getAngularDistanceMatrix(self._particleList)

        # Save output data
        self._saveMatrix(self._getExtraPath(self._getFileName('distances')), distances)

    def computeCorrelationStep(self):
        # Read input data
        distances = self._loadMatrix(self._getExtraPath(self._getFileName('distances')))
        maxAngularDistance = np.radians(self.maxAngularDistance.get())
        particles = self._particleList

        # Compute the class pairs to be considered
        pairs = self._getPairMatrix(distances, maxAngularDistance)

        # Compute the correlation
        correlations = self._getCorrelationMatrix(particles, pairs)

        # Save output data
        self._saveMatrix(self._getExtraPath(self._getFileName('correlations')), correlations)

    def graphPartitionStep(self):
        # Read input data
        correlations = self._loadMatrix(self._getExtraPath(self._getFileName('correlations')))
        graph = csgraph.csgraph_from_dense(correlations)
        
        # Partition the graph according to the selected method
        partitionFunctions = {
            0: self._spectralPartition,
            1: self._minimumCut
        }
        partitionFunction = partitionFunctions[self.graphPartitionMethod.get()]
        partitionCount = self.graphPartitionCount.get()
        labels = partitionFunction(graph, partitionCount)
        
        # Save the output
        self._saveMatrix(self._getExtraPath(self._getFileName('labels')), labels)

    def createOutputStep(self):
        # Read input data
        labels = self._loadMatrix(self._getExtraPath(self._getFileName('labels')), dtype=int)
        particles = self.directionalClasses.get()
        nClasses = max(labels) + 1

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

        if self.maxComparisonCount.get() >= len(self.directionalClasses.get()):
            result.append('Comparison count should be less than the length of the input directional classes')

        return result
    
    def _citations(self):
        pass

    def _summary(self):
        pass
    
    def _methods(self):
        pass

    #--------------------------- UTILS functions ---------------------------------------------------
    def _saveMatrix(self, path, x):
        # Determine the format
        np.savetxt(path, x, delimiter=',', fmt='%s') # CSV

    def _loadMatrix(self, path, dtype=float):
        return np.genfromtxt(path, delimiter=',', dtype=dtype) # CSV


    def _getParticleList(self, particles):
        result = []

        for particle in particles:
            result.append(particle.clone())

        assert(len(result) == len(particles))
        return result
    
    def _getAngularDistanceMatrix(self, classes):
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
                assert(np.allclose(np.linalg.inv(rotMtx1), np.transpose(rotMtx1)))
                diffMtx = np.matmul(rotMtx0, np.transpose(rotMtx1))
                distance = math.acos(np.clip((np.trace(diffMtx) - 1)/2, -1.0, +1.0))

                # Write the result on symmetrical positions
                distances[idx0, idx1] = distance
                distances[idx1, idx0] = distance

        # Ensure that the result matrix is symmetrical
        assert(np.array_equal(distances, np.transpose(distances)))
        return distances

    def _getPairMatrix(self, distances, maxDistance):
        pairs = distances <= maxDistance
        np.fill_diagonal(pairs, False)
        return pairs

    def _getCorrelation(self, class0, class1):
        img0 = xmippLib.Image(class0.getLocation())
        img1 = xmippLib.Image(class1.getLocation())
        corr = img0.correlationAfterAlignment(img1)
        return max(corr, 0.0)

    def _getCorrelationMatrix(self, classes, pairs):
        correlations = np.zeros_like(pairs, dtype=float)

        # Ensure that the pairs matrix is symmetrical
        assert(np.array_equal(pairs, np.transpose(pairs)))

        # Compute the symmetric matrix with the correlations
        for idx0, idx1 in zip(*np.nonzero(pairs)):
            if idx0 < idx1:
                # Calculate the corrrelation
                class0 = classes[idx0]
                class1 = classes[idx1]

                # Write it on symmetrical positions
                correlation = self._getCorrelation(class0, class1)
                correlations[idx0, idx1] = correlation
                correlations[idx1, idx0] = correlation

            elif idx0 == idx1:
                # Correlation on the diagonal is 1
                correlations[idx0, idx0] = 1.0

        # Ensure that the result matrix is symmetrical
        assert(np.array_equal(correlations, np.transpose(correlations)))
        return correlations

    def _classifySpectrum(self, eigenVectors):
        nVertices, nVectors = eigenVectors.shape
        nVectorsLog2 = math.trunc(math.log2(nVectors))
        assert(2**nVectorsLog2==nVectors) # TODO generalize for non pow2 classifications

        # Iterate over the pow 2 indices, summing their weights in 
        # decreasing order (binary counting)
        labels = np.zeros(nVertices)
        for i in range(1, nVectorsLog2+1):
            w = eigenVectors[:, 2**i-1] # Extract the eigenvector
            r = 2**(nVectorsLog2-i)
            labels += r*(w>=0)

        return labels

    def _spectralPartition(self, graph, componentCount):
        # Decompose the laplacian matrix of the graph into its N
        # smallest eigenvectors. Then interpret these vectors as
        # "standing waves" in the graph and use them to partition it
        l = csgraph.laplacian(graph)
        values, vectors = linalg.eigsh(l, k=componentCount, which='SM')
        assert(values == sorted(values))
        return self._classifySpectrum(vectors)

    def _getSourceSinkVertices(self, graph, labels, component):
        result = None

        # Among all the connected vertices, get the least conneted ones
        for pos in zip(*graph.nonzero()):
            if (not result or graph[pos] < graph[result]) and labels[pos[0]] == component:
                assert(labels[pos[1]] == component) # The destination vertex should also be in the same component
                result = pos

        return result

    def _removeEdges(self, graph, labels):
        result = sparse.csr_matrix(graph.shape, dtype=graph.dtype)

        # Copy the edges only if both vertices correspond to the same component
        for pos in zip(*graph.nonzero()):
            if labels[pos[0]] == labels[pos[1]]:
                result[pos] = graph[pos]
        
        return result

    def _minimumCut(self, graph, componentCount):
        # Obtain the component labels from the graph
        nComponents, labels = csgraph.connected_components(graph)

        # Repeatedly cut the graph until the desired amount of components is obtained
        while nComponents < componentCount:
            # Perform a maximum flow analysis with the biggest component of the graph
            component = stats.mode(labels)
            source, sink = self._getSourceSinkVertices(graph, labels, component)
            flow = csgraph.maximum_flow(graph, source, sink)

            # Update the labels and the graph with the residual graph
            nComponents, labels = csgraph.connected_components(flow.residual)
            graph = self._removeEdges(graph, labels)

        return labels

    def _reconstructVolume(self, path, particles, xmdSuffix=''):
        # Convert the particles to a xmimpp metadata file
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

