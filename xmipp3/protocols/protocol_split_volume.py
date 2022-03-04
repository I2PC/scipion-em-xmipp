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

from unittest import result
from pyworkflow.constants import BETA
from pyworkflow.protocol.constants import LEVEL_ADVANCED, STEPS_PARALLEL
from pyworkflow.protocol.params import PointerParam, FloatParam, IntParam, StringParam, BooleanParam
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
        myDict = {
            'distances': 'distances.csv',
            'correlations': 'correlations.csv',
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

        form.addParallelSection()
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Start converting the input
        self._insertFunctionStep('convertInputStep')

        # Compute the angular distances
        self._insertFunctionStep('computeAngularDistancesStep')

        # Compute the correlation among the closest averages
        self._insertFunctionStep('computeCorrelationStep')

        # Create the output
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

    def createOutputStep(self):
        pass

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
        np.savetxt(path, x, delimiter=',') # CSV

    def _loadMatrix(self, path, dtype=float):
        return np.genfromtxt(path, delimiter=',', dtype=dtype) # CSV

    def _calculateInDegreesFromAdjacency(self, adj):
        return np.sum(adj, axis=0)

    def _calculateOutDegreesFromAdjacency(self, adj):
        return np.sum(adj, axis=1)

    def _calculateDegreesFromAdjacency(self, adj):
        return self._calculateInDegreesFromAdjacency(adj) + self._calculateOutDegreesFromAdjacency(adj)

    def _calculateLaplacianFromAdjacency(self, adjacency):
        result = -adjacency
        degrees = self._calculateDegreesFromAdjacency(adjacency)

        for i in range(len(degrees)):
            assert(result[i, i] == 0)
            result[i, i] = degrees[i]

        return result

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
                assert(np.allclose(np.linalg.inv(rotMtx1), np.transpose(rotMtx1)))
                diffMtx = np.matmul(rotMtx0, np.transpose(rotMtx1))
                distance = math.acos((np.trace(diffMtx) - 1)/2)

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
        corr = img0.correlation(img1)
        return max(corr, 0.0)

    def _getCorrelationMatrix(self, classes, pairs):
        correlations = np.zeros_like(pairs, dtype=float)

        # Ensure that the pairs matrix is symmetrical
        assert(np.array_equal(pairs, np.transpose(pairs)))

        # Compute the symmetric matrix with the correlations
        for idx0, idx1 in zip(*np.where(pairs)):
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