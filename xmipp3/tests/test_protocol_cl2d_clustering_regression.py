# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from unittest.mock import patch

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_cl2d_clustering import (
    OUTPUT_AVERAGES,
    OUTPUT_CLASSES,
    XmippProtCL2DClustering,
)


class _FakeItem:
    def __init__(self, objId=None):
        self.objId = objId
        self.classId = None

    def clone(self):
        return type(self)(self.objId)

    def copyInfo(self, other):
        pass

    def setObjId(self, objId):
        self.objId = objId

    def getObjId(self):
        return self.objId

    def setClassId(self, classId):
        self.classId = classId


class _FakeAverage(_FakeItem):
    pass


class _FakeClass(_FakeItem):
    def __init__(self, objId=None, particles=None, corrupted=False):
        super().__init__(objId)
        self._particles = particles or []
        self.corrupted = corrupted

    def clone(self):
        return _FakeClass(self.objId, list(self._particles), self.corrupted)

    def iterItems(self):
        if self.corrupted:
            raise ValueError("corrupted particle stack")
        return iter(self._particles)

    def copyInfo(self, other):
        # Mirrors the real copyInfo contract: it reads metadata FROM
        # `other` (the resolved cluster-member template), so a corrupted
        # `other` is what actually raises here, not `self`.
        if getattr(other, 'corrupted', False):
            raise ValueError("corrupted class metadata")

    def enableAppend(self):
        pass

    def append(self, particle):
        self._particles.append(particle)


class _CorruptedAverage(_FakeAverage):
    def clone(self):
        raise ValueError("corrupted image data")


class _FakeAveragesSet:
    """Stands in for a SetOfAverages: not a SetOfClasses2D instance, so
    createOutputSetOfAverages takes its 'else' (averages) branch."""

    def __init__(self, items):
        self._items = items

    def __contains__(self, itemId):
        return itemId in self._items

    def getItem(self, field, value):
        return self._items[value]

    def getSamplingRate(self):
        return 1.5


class _FakeAveragesOutputSet:
    def __init__(self):
        self.items = []
        self.samplingRate = None

    def append(self, item):
        self.items.append(item)

    def setSamplingRate(self, rate):
        self.samplingRate = rate

    def getIdSet(self):
        return {item.getObjId() for item in self.items}


class _FakeClasses2DSet:
    def __init__(self):
        self._classes = {}

    def append(self, newClass):
        self._classes[newClass.getObjId()] = newClass

    def __getitem__(self, classId):
        return self._classes[classId]

    def update(self, class2D):
        self._classes[class2D.getObjId()] = class2D

    def write(self):
        pass

    def getIdSet(self):
        return set(self._classes.keys())

    def particlesOf(self, classId):
        return list(self._classes[classId]._particles)


class TestXmippCL2DClusteringRegression(BaseTest):
    """Regression tests for CL2D Clustering output-building robustness."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        return self.newProtocol(XmippProtCL2DClustering)

    def testReadClustersFromTxtParsesIdsAsIntegers(self):
        # Regression test: cluster member ids must be parsed as real ints,
        # not strings - Set.getItem("id", "36") relies on the backend
        # coercing a string to match an integer column, which SQLite does
        # only by its own type-affinity rules and a strict backend (e.g.
        # other backends) may not.
        prot = self._newProtocol()
        txtPath = self.proj.getTmpPath('cl2d_clusters.txt')
        with open(txtPath, 'w') as f:
            f.write("Cluster 0:\n6\nCluster 1:\n36\n34\n")

        clusters = prot.read_clusters_from_txt(txtPath)

        self.assertEqual({0: [6], 1: [36, 34]}, clusters)
        for ids in clusters.values():
            for classRef in ids:
                self.assertIsInstance(classRef, int)

    def testCreateOutputSetOfAveragesSkipsMissingMemberAndUsesNextValidOne(self):
        # Regression test: Set.getItem raises (UnboundLocalError) rather
        # than returning None for a row it cannot find. A cluster whose
        # first listed member is missing from the input set (e.g. a
        # best_clusters_with_names.txt / class_representatives.txt
        # mismatch) must fall back to the next resolvable member instead
        # of crashing the whole output-creation step.
        prot = self._newProtocol()
        prot.samplingRate = 1.0
        inputSet2D = _FakeAveragesSet({34: _FakeAverage(34)})  # 36 is missing
        result_dict = {1: [36, 34]}
        outputRefs = _FakeAveragesOutputSet()
        prot._createSetOfAverages = lambda: outputRefs

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Particle',
                _FakeAverage,
        ):
            output_dict = prot.createOutputSetOfAverages(
                inputSet2D, result_dict, {},
            )

        self.assertEqual({34}, output_dict[OUTPUT_AVERAGES].getIdSet())

    def testCreateOutputSetOfAveragesSkipsWholeClusterWhenNoMemberResolves(self):
        # If none of a cluster's members can be found, that cluster must be
        # skipped (with a clear error message) rather than crashing the
        # whole createOutputStep and losing every other cluster's result.
        prot = self._newProtocol()
        prot.samplingRate = 1.0
        inputSet2D = _FakeAveragesSet({34: _FakeAverage(34)})
        result_dict = {1: [99], 2: [34]}  # cluster 1 is fully unresolvable
        outputRefs = _FakeAveragesOutputSet()
        prot._createSetOfAverages = lambda: outputRefs

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Particle',
                _FakeAverage,
        ):
            output_dict = prot.createOutputSetOfAverages(
                inputSet2D, result_dict, {},
            )

        self.assertEqual({34}, output_dict[OUTPUT_AVERAGES].getIdSet())

    def testCreateOutputSetOfClassesSkipsMissingMemberButKeepsValidOnes(self):
        # Same regression, but for the Classes extraction path: a missing
        # cluster member must be skipped (with a clear log message)
        # instead of crashing, while particles from the still-resolvable
        # members of the same cluster are preserved.
        prot = self._newProtocol()
        particle1 = _FakeItem(101)
        validClass = _FakeClass(34, particles=[particle1])
        inputSet2D = _FakeAveragesSet({34: validClass})  # 36 is missing
        result_dict = {1: [36, 34]}

        classes2DSet = _FakeClasses2DSet()
        prot._createSetOfClasses2D = lambda imagesPointer: classes2DSet
        inputSet2D.getImagesPointer = lambda: None

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Class2D',
                _FakeClass,
        ):
            output_dict = prot.createOutputSetOfClasses(
                inputSet2D, result_dict, {},
            )

        self.assertEqual({34}, output_dict[OUTPUT_CLASSES].getIdSet())
        self.assertEqual([101], [p.getObjId() for p in classes2DSet.particlesOf(34)])

    def testCreateOutputSetOfClassesSkipsWholeClusterWhenNoMemberResolves(self):
        # If every member of a cluster is missing from the input set, no
        # class must be created for it (with a clear error message), but
        # other clusters must still be processed normally.
        prot = self._newProtocol()
        particle1 = _FakeItem(202)
        validClass = _FakeClass(34, particles=[particle1])
        inputSet2D = _FakeAveragesSet({34: validClass})
        result_dict = {1: [99], 2: [34]}  # cluster 1 is fully unresolvable

        classes2DSet = _FakeClasses2DSet()
        prot._createSetOfClasses2D = lambda imagesPointer: classes2DSet
        inputSet2D.getImagesPointer = lambda: None

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Class2D',
                _FakeClass,
        ):
            output_dict = prot.createOutputSetOfClasses(
                inputSet2D, result_dict, {},
            )

        self.assertEqual({34}, output_dict[OUTPUT_CLASSES].getIdSet())

    def testCreateOutputSetOfAveragesTriesNextReferenceWhenFirstFailsToResolve(self):
        # Regression test: a genuine processing failure while resolving one
        # cluster's centroid candidate (e.g. a corrupted image) must not
        # crash the whole createOutputStep. It must try the next candidate
        # reference of the same cluster instead.
        prot = self._newProtocol()
        prot.samplingRate = 1.0
        inputSet2D = _FakeAveragesSet({
            36: _CorruptedAverage(36),
            34: _FakeAverage(34),
        })
        result_dict = {1: [36, 34]}
        outputRefs = _FakeAveragesOutputSet()
        prot._createSetOfAverages = lambda: outputRefs

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Particle',
                _FakeAverage,
        ):
            output_dict = prot.createOutputSetOfAverages(
                inputSet2D, result_dict, {},
            )

        self.assertEqual({34}, output_dict[OUTPUT_AVERAGES].getIdSet())

    def testCreateOutputSetOfAveragesSkipsClusterWhenAllReferencesFailToResolve(self):
        prot = self._newProtocol()
        prot.samplingRate = 1.0
        inputSet2D = _FakeAveragesSet({
            36: _CorruptedAverage(36),
            34: _FakeAverage(34),
        })
        result_dict = {1: [36], 2: [34]}  # cluster 1's only ref is corrupted
        outputRefs = _FakeAveragesOutputSet()
        prot._createSetOfAverages = lambda: outputRefs

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Particle',
                _FakeAverage,
        ):
            output_dict = prot.createOutputSetOfAverages(
                inputSet2D, result_dict, {},
            )

        self.assertEqual({34}, output_dict[OUTPUT_AVERAGES].getIdSet())

    def testCreateOutputSetOfClassesSkipsCorruptedMemberButKeepsOthers(self):
        # Regression test: a genuine processing failure while extracting
        # particles from one cluster member (e.g. a corrupted particle
        # stack) must not crash the whole createOutputStep. The rest of
        # the cluster's resolvable members must still be processed. Here
        # the template is already built from the first (valid) member, so
        # the second member's failure is isolated to iterItems().
        prot = self._newProtocol()
        validClass = _FakeClass(34, particles=[_FakeItem(101)])
        corruptedClass = _FakeClass(36, particles=[_FakeItem(999)], corrupted=True)
        inputSet2D = _FakeAveragesSet({34: validClass, 36: corruptedClass})
        result_dict = {1: [34, 36]}

        classes2DSet = _FakeClasses2DSet()
        prot._createSetOfClasses2D = lambda imagesPointer: classes2DSet
        inputSet2D.getImagesPointer = lambda: None

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Class2D',
                _FakeClass,
        ):
            output_dict = prot.createOutputSetOfClasses(
                inputSet2D, result_dict, {},
            )

        newClassIds = output_dict[OUTPUT_CLASSES].getIdSet()
        self.assertEqual(1, len(newClassIds))
        newClassId = next(iter(newClassIds))
        particleIds = [
            p.getObjId()
            for p in classes2DSet.particlesOf(newClassId)
        ]
        self.assertEqual([101], particleIds)

    def testCreateOutputSetOfClassesRetriesTemplateWhenFirstMemberFailsToBuildIt(self):
        # Regression test: a genuine failure while building the class
        # template from the first resolvable member (e.g. corrupted class
        # metadata) must not leave the cluster without a class - the next
        # resolvable member must be tried as the template instead.
        prot = self._newProtocol()
        corruptedTemplate = _FakeClass(36, particles=[_FakeItem(999)], corrupted=True)
        validClass = _FakeClass(34, particles=[_FakeItem(101)])
        inputSet2D = _FakeAveragesSet({36: corruptedTemplate, 34: validClass})
        result_dict = {1: [36, 34]}

        classes2DSet = _FakeClasses2DSet()
        prot._createSetOfClasses2D = lambda imagesPointer: classes2DSet
        inputSet2D.getImagesPointer = lambda: None

        with patch(
                'xmipp3.protocols.protocol_cl2d_clustering.Class2D',
                _FakeClass,
        ):
            output_dict = prot.createOutputSetOfClasses(
                inputSet2D, result_dict, {},
            )

        self.assertEqual({34}, output_dict[OUTPUT_CLASSES].getIdSet())
        particleIds = [
            p.getObjId()
            for p in classes2DSet.particlesOf(34)
        ]
        self.assertEqual([101], particleIds)


if __name__ == '__main__':
    import unittest
    unittest.main()
