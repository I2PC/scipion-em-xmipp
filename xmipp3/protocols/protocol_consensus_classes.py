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

from typing import Set
import collections

from pwem.protocols import EMProtocol
from pwem.objects import SetOfClasses

from pyworkflow.protocol.params import Form, MultiPointerParam
from pyworkflow.constants import BETA

from xmipp3.convert import setXmippAttribute


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
        self._insertFunctionStep('intersectStep', prerequisites=[])

    def intersectStep(self):
        classifications = self._getInputClassifications()

        # Compute the intersection among all classes
        intersections = self._calculateClassificationIntersections(classifications)

        # Create the output classes and define them
        outputClasses = self._createOutputClasses(
            self.inputClassifications[0].get(),
            intersections,
            'intersections',
            #itertools.repeat(consensusSizes),
            #itertools.repeat(consensusRelativeSizes)
        )
        self._defineOutputs(**{'outputClasses_intersections': outputClasses})

        # Stablish source output relationships
        self._defineSourceRelation(self.inputClassifications, outputClasses)

    def createOutputStep(self):
        pass
    
    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []

        # Ensure that all classifications are made of the same images
        items = self._getInputItems(0)
        for i in range(1, len(self.inputClassifications)):
            if self._getInputItems(i) != items:
                errors.append(f'Classification {i} has been done with different items')

        return errors
    
    # --------------------------- UTILS functions ------------------------------
    def _getInputClassifications(self):
        if not hasattr(self, '_classifications'):
            self._classifications = self._convertInputClassifications(self.inputClassifications)
        
        return self._classifications
    
    def _getInputItems(self, i=0):
        return self.inputClassifications[i].get().getImages()
    
    def _convertInputClassifications(self, classifications):
        """ Returns the list of lists of sets that stores the set of ids of each class"""
        def classToCluster(cls):
            ids = cls.getIdSet()
            rep = cls.getRepresentative().clone() if cls.hasRepresentative() else None
            return XmippProtConsensusClasses.Cluster(ids, rep)

        f = lambda classification : list(map(classToCluster, classification.get()))
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
    
    def _createOutputClasses(self, 
                             classification: SetOfClasses,
                             clustering, 
                             name ):
                             #randomConsensusSizes=None, 
                             #randomConsensusRelativeSizes=None ):

        # Create an empty set with the same images as the input classification
        result = self._EMProtocol__createSet( # HACK
            type(classification), 
            'classes%s.sqlite', name
        ) 
        result.setImages(classification.getImages())
    
        # Fill the output
        loader = XmippProtConsensusClasses.ClassesLoader(
            clustering, 
            #randomConsensusSizes,
            #randomConsensusRelativeSizes
        )
        loader.fillClasses(result)

        return result
    
    # ---------------------------- I/O functions -------------------------------

    # --------------------------- HELPER classes -------------------------------
    class Cluster:
        """ Keeps track of the information related to successive class intersections.
            It is instantiated with a single class and allows to perform intersections
            and unions with it"""
        def __init__(self, ids: Set[int], representative=None, sourceSize=None):
            self._ids = ids
            self._representative = representative
            self._sourceSize = sourceSize if sourceSize is not None else len(self._ids)  # The size of the representative class

        def __len__(self):
            return len(self.getIds())

        def __eq__(self, other):
            return self.getIds() == other.getIds()

        def intersection(self, *others):
            return self._combine(set.intersection, min, XmippProtConsensusClasses.Cluster.getSourceSize, *others)

        def union(self, *others):
            return self._combine(set.union, max, len, *others)

        def getIds(self):
            return self._ids

        def getRepresentative(self):
            return self._representative

        def getSourceSize(self):
            return self._sourceSize

        def getRelativeSize(self):
            return len(self) / self.getSourceSize()

        def _combine(self, operation, selector, selectionCriteria, *others):
            allItems = (self, ) + others
            
            # Perform the requested operation among the id sets
            allIds = map(XmippProtConsensusClasses.Cluster.getIds, allItems)
            ids = operation(*allIds)

            # Select the class with the appropriate criteria
            selection = selector(allItems, key=selectionCriteria)
            representative = selection.getRepresentative()
            sourceSize = selection.getSourceSize()

            return XmippProtConsensusClasses.Cluster(ids, representative, sourceSize)
    
    
    
    class ClassesLoader:
        """ Helper class to produce classes
        """
        def __init__(self, clustering):
            self._clustering = clustering
            self._classification = self._createClassification(clustering)

        def fillClasses(self, clsSet):
            clsSet.classifyItems(updateItemCallback=self._updateItem,
                                updateClassCallback=self._updateClass,
                                itemDataIterator=iter(self._classification.items()),
                                doClone=False)

        def _updateItem(self, item, row):
            itemId, classId = row
            assert(item.getObjId() == itemId)
            item.setClassId(classId)

        def _updateClass(self, item):
            classId = item.getObjId()
            classIdx = classId-1

            # Set the representative
            item.setRepresentative(self._clustering[classIdx].getRepresentative())

            # Set the size p-value
            #if self.randomConsensusSizes is not None:
            #    size = len(self.clustering[classIdx])
            #    percentile = self._findPercentile(self.randomConsensusSizes, size)
            #    pValue = 1 - percentile
            #    setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_SIZE_PVALUE, Float(pValue))
            
            # Set the relative size p-value
            #if self.randomConsensusRelativeSizes is not None:
            #    size = self.clustering[classIdx].getRelativeSize()
            #    percentile = self._findPercentile(self.randomConsensusRelativeSizes, size)
            #    pValue = 1 - percentile
            #    setXmippAttribute(item, emlib.MDL_CLASS_INTERSECTION_RELATIVE_SIZE_PVALUE, Float(pValue))

        def _createClassification(self, clustering):
            # Fill the classification data (particle-class mapping)
            result = {}
            for cls, data in enumerate(clustering):
                for id in data.getIds():
                    result[id] = cls + 1 

            return collections.OrderedDict(sorted(result.items())) 

        #def _findPercentile(self, data, value):
        #    """ Given an array of values (data), finds the corresponding percentile of the value.
        #        Percentile is returned in range [0, 1] """
        #    assert(np.array_equal(sorted(data), data))  # In order to iterate it in ascending order

            # Count the number of elements that are smaller than the given value
        #    i = 0
        #    while i < len(data) and data[i] < value:
        #        i += 1

            # Convert it into a percentage
        #    return float(i) / float(len(data))