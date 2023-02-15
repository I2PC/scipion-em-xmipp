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

from typing import Iterable, Sequence, Optional, List, Set
import itertools

from pwem.protocols import EMProtocol
from pwem.objects import Pointer, Object, SetOfClasses, SetOfImages
from pwem import emlib

from pyworkflow.protocol.params import Form, MultiPointerParam
from pyworkflow.constants import BETA
from pyworkflow.object import Float

from xmipp3.convert import setXmippAttribute

import numpy as np
import scipy.stats
import scipy.cluster
import scipy.spatial

class XmippProtConsensusClasses(EMProtocol):
    """ Compare several SetOfClasses.
        Return the consensus clustering based on a objective function
        that uses the similarity between clusters intersections and
        the entropy of the clustering formed.
    """
    _label = 'consensus classes'
    _devStatus = BETA

    def __init__(self, *args, **kwargs):
        EMProtocol.__init__(self, *args, **kwargs)
        
    def _defineParams(self, form: Form):
        # Input data
        form.addSection(label='Input')
        form.addParam('inputClassifications', MultiPointerParam, pointerClass='SetOfClasses', 
                      label="Input classes", important=True,
                      help='Select several sets of classes where to evaluate the '
                           'intersections.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('referenceIntersectionStep')
        self._insertFunctionStep('intersectStep')
        self._insertFunctionStep('mergeStep')
        #self._insertFunctionStep('createOutputStep')

    def referenceIntersectionStep(self):
        sizes = self._getInputClassificationSizes()
        nImages = sizes[0].sum()
        sourceProbabilities = self._calculateIntersectionSourceProbabilities(sizes)
        intersectionProbabilities = self._calculateIntersectionProbabilities(sourceProbabilities)
        normIntersectionProbabilities = self._calculateNormalizedIntersectionProbabilities(
            sourceProbabilities, 
            intersectionProbabilities
        )
        intersectionSizes = np.sort(intersectionProbabilities * nImages)
        normalizedIntersectionSizes = np.sort(normIntersectionProbabilities) # No need to multiply

        # Write
        np.save(self._getReferenceIntersectionSizeFilename(), intersectionSizes)
        np.save(self._getReferenceIntersectionNormalizedSizeFilename(), normalizedIntersectionSizes)

    def intersectStep(self):
        classifications = self._getInputClassifications()
        classes = self._getInputClassificationIds()
        intersections = self._calculateIntersections(classes)
        
        referenceSizes = np.load(self._getReferenceIntersectionSizeFilename())
        referenceRelativeSizes = np.load(self._getReferenceIntersectionNormalizedSizeFilename())
        outputClasses = self._createSetOfClasses(
            classifications,
            intersections,
            self._getMergedIntersectionSuffix(len(intersections)),
            referenceSizes,
            referenceRelativeSizes
        )
        
        self._defineOutputs(outputClasses=outputClasses)
        self._defineSourceRelation(self.inputClassifications, outputClasses)

    def mergeStep(self):
        classifications = self._getInputClassifications()
        classes = self._getInputClassificationIds()
        allClasses = list(itertools.chain(*classes))
        intersections = self._getOutputIntersectionIds()
        
        similarity = self._calculateClusterSimilarityMatrix(intersections, allClasses)
        linkage = self._calculateLinkageMatrix(similarity)
        merging = self._calculateMergedIntersections(intersections, linkage)

        # Create outputs
        np.save(self._getLinkageMatrixFilename(), linkage)

        referenceSizes = np.load(self._getReferenceIntersectionSizeFilename())
        referenceRelativeSizes = np.load(self._getReferenceIntersectionNormalizedSizeFilename())
        for merged in merging[1:]: # Skip the first one as it is empty
            outputClasses = self._createSetOfClasses(
                classifications,
                merged,
                self._getMergedIntersectionSuffix(len(merged)),
                referenceSizes,
                referenceRelativeSizes
            )

            outputClasses.write()
    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []

        # Ensure that all classifications are made of the same images
        items = self._getInputImages(0)
        for i in range(1, len(self.inputClassifications)):
            if self._getInputImages(i) != items:
                errors.append(f'Classification {i} has been done with different images')

        return errors
    
    # --------------------------- UTILS functions ------------------------------
    def _getInputClassificationCount(self) -> int:
        return len(self.inputClassifications)
    
    def _getInputClassification(self, classification: int) -> SetOfClasses:
        return self.inputClassifications[classification].get()

    def _getInputClassifications(self) -> Sequence[SetOfClasses]:
        return list(map(Pointer.get, self.inputClassifications))

    def _getInputClassificationIds(self) -> Sequence[Sequence[Set[int]]]:
        def convertClassification(classification: Pointer):
            return list(map(SetOfImages.getIdSet, classification.get()))
            
        return list(map(convertClassification, self.inputClassifications))

    def _getInputClassificationSizes(self) -> Sequence[np.ndarray]:
        def convertClassification(classification: Pointer):
            return np.array(list(map(len, classification.get())))
            
        return list(map(convertClassification, self.inputClassifications))

    def _getInputImages(self, classification: int = 0) -> SetOfImages:
        return self._getInputClassification(classification).getImages()
 
    def _getOutputIntersectionIds(self):
        return list(map(SetOfImages.getIdSet, self.outputClasses))
 
    def _getReferenceIntersectionSizeFilename(self) -> str:
        return self._getExtraPath('reference_sizes.npy')
    
    def _getReferenceIntersectionNormalizedSizeFilename(self) -> str:
        return self._getExtraPath('reference_norm_sizes.npy')
    
    def _getLinkageMatrixFilename(self) -> str:
        return self._getExtraPath('linkage.npy')

    def _getMergedIntersectionSuffix(self, numel: int) -> str:
        return 'merged_%06d' % numel
 
    def _getOutputSqliteTemplate(self) -> str:
        return 'classes_%s.sqlite'

    def _getOutputSqliteFilename(self, suffix: str = '') -> str:
        return self._getPath(self._getOutputSqliteTemplate() % suffix)
 
 
    
    def _calculateIntersections(self, 
                                classifications: Iterable[Sequence[Set[int]]]):
        # Start with the first classification
        result = classifications[0]

        for i in range(1, len(classifications)):
            classification = classifications[i]
            intersections = []

            # Perform the intersection with previous classes
            for cls0 in result:
                for cls1 in classification:
                    intersection = cls0.intersection(cls1)
                    if len(intersection) > 0:
                        intersections.append(intersection)

            # Use the new intersection for the next step
            result = intersections

        return result

    def _calculateIntersectionSourceProbabilities(self, 
                                                  classSizes: Iterable[np.ndarray]) -> np.ndarray:
        
        def calculateProbability(sizes: np.ndarray) -> np.ndarray:
            return sizes / np.sum(sizes)
        
        probabilities = map(calculateProbability, classSizes)
        
        # Consider all combinations
        prod = list(itertools.product(*probabilities))
        return np.array(prod)
    
    def _calculateIntersectionProbabilities(self, 
                                            sourceProbabilities: np.ndarray) -> np.ndarray:
        return np.product(sourceProbabilities, axis=1)

    def _calculateNormalizedIntersectionProbabilities(self, 
                                                      sourceProbabilities: np.ndarray,
                                                      probabilities: Optional[np.ndarray] = None ) -> np.ndarray:
        if probabilities is None:
            probabilities = self._calculateIntersectionProbabilities(sourceProbabilities)
            
        norm = np.min(sourceProbabilities, axis=1)
        return probabilities / norm
    
    def _calculateClassificationLengths(self, classifications: Iterable[SetOfClasses]) -> Sequence[np.ndarray]:
        def getLengths(classification: SetOfClasses) -> np.ndarray:
            return np.ndarray(list(map(len, classification)))
        
        return list(map(getLengths, classifications))
    
    def _calculateClusterSimilarity(self, x: Set[int], y: Set[int]):
        nCommon = len(x & y)
        nTotal = len(x | y)
        return nCommon / nTotal
    
    def _calculateClusterSimilarityMatrix(self, a, b):
        result = np.empty((len(a), len(b)))
        
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                result[i,j] = self._calculateClusterSimilarity(x, y)

        return result
    
    def _calculateLinkageMatrix(self, points: np.ndarray) -> np.ndarray:
        return scipy.cluster.hierarchy.linkage(
            points,
            method='single',
            metric='cosine'
        )

    def _calculateMergedIntersections(self, 
                                      intersections: List[Set[int]],
                                      linkage: np.ndarray ):
        merged = [intersections]
        clusters = intersections.copy()
        
        for idx0, idx1, _, _ in linkage:
            # Construct a new cluster joining two of the clusters
            cluster0: Set[int] = clusters[int(idx0)]
            cluster1: Set[int] = clusters[int(idx1)]
            clusterMerged = cluster0.union(cluster1)
            clusters.append(clusterMerged)

            # Copy the previous row except for the
            # merged sets. Add the new merged set
            assert(cluster0 in merged[-1])
            assert(cluster1 in merged[-1])
            mergedNew = [clusterMerged]
            for cluster in merged[-1]:
                if cluster not in (cluster0, cluster1):
                    mergedNew.append(cluster)
            
            merged.append(mergedNew)
            
        return merged
    
    def _calculatePercentile(self, data: np.ndarray, value: float):
            """ Given an array of values (data), finds the corresponding percentile of the value.
                Percentile is returned in range [0, 1] """
            assert(np.array_equal(sorted(data), data))  # In order to iterate it in ascending order

            # Count the number of elements that are smaller than the given value
            i = 0
            while i < len(data) and data[i] < value:
                i += 1

            # Convert it into a percentage
            return float(i) / float(len(data))
    
    def _calculateClusterRepresentativeClass(self,
                                             cluster: Set[int],
                                             classifications: Iterable[SetOfClasses]):
        
        allClasses = itertools.chain(*classifications)
            
        def computeSimilarity(cls: SetOfImages) -> float:
            objIds = cls.getIdSet()
            return self._calculateClusterSimilarity(cluster, objIds)
            
        return max(allClasses, key=computeSimilarity)
    
    # -------------------------- Convert functions -----------------------------
    def _createSetOfClasses(self, 
                            classifications: Sequence[SetOfClasses],
                            clustering: Sequence[Set[int]], 
                            suffix: str,
                            referenceSizes=None, 
                            referenceRelativeSizes=None ):

        # Create an empty set with the same images as the input classification
        result: SetOfClasses = self._EMProtocol__createSet( # HACK
            type(classifications[0]), 
            self._getOutputSqliteTemplate(),
            suffix
        ) 
        result.setImages(classifications[0].getImages())
    
        # Fill the output
        def updateItem(item: Object, _):
            objId: int = item.getObjId()
            
            classId = 0
            for cls, objIds in enumerate(clustering):
                if objId in objIds:
                    classId = cls + 1
                    break # Found!
                
            item.setClassId(classId)
        
        def updateClass(item: SetOfImages):
            classId: int = item.getObjId()
            classIdx = classId - 1
            cluster = clustering[classIdx]
            representativeClass = self._calculateClusterRepresentativeClass(
                cluster,
                classifications
            )
            size = len(clustering[classIdx])
            relativeSize = size / len(representativeClass)
            
            item.setRepresentative(representativeClass.getRepresentative())
            
            if referenceSizes is not None:
                sizePercentile = self._calculatePercentile(referenceSizes, size)
                pValue = 1 - sizePercentile
                setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_SIZE_PVALUE, Float(pValue))
                
            if referenceRelativeSizes is not None:
                relativeSizePercentile = self._calculatePercentile(referenceRelativeSizes, relativeSize)
                pValue = 1 - relativeSizePercentile
                setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_RELATIVE_SIZE_PVALUE, Float(pValue))

        result.classifyItems(
            updateItemCallback=updateItem,
            updateClassCallback=updateClass,
            doClone=True
        )
    
        return result
