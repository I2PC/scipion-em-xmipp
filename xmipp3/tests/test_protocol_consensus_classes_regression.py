# **************************************************************************
# Regression tests for logical output handling in consensus classes.
# **************************************************************************

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from xmipp3.protocols.protocol_consensus_classes import XmippProtConsensusClasses


class _LogicalSet:
    def __init__(self):
        self.loaded = False

    def loadAllProperties(self):
        self.loaded = True


class TestConsensusClassesLogicalOutputs(unittest.TestCase):

    def test_MergedIntersectionsReusePersistedLogicalOutput(self):
        logicalOutput = _LogicalSet()
        prot = SimpleNamespace(merged3=logicalOutput)

        result = XmippProtConsensusClasses._obtainMergedIntersections(prot, 3)

        self.assertIs(result, logicalOutput)
        self.assertTrue(logicalOutput.loaded)

    def test_MergedIntersectionsCreateFromLogicalInputImages(self):
        images = object()
        classifications = object()
        outputClasses = object()
        definedOutputs = {}

        prot = SimpleNamespace(
            _getInputImages=lambda: images,
            _getInputClassifications=lambda: classifications,
            _getOutputIntersectionIds=lambda: [set([1]), set([2])],
            _readLinkageMatrix=lambda: object(),
            _readReferenceIntersectionSizes=lambda: (object(), object()),
            _calculateMergedIntersections=lambda intersections, linkage, stop: [[set([1, 2])]],
            _getMergedIntersectionSuffix=lambda n: 'merged_%06d' % n,
            _createSetOfClasses=Mock(return_value=outputClasses),
            _defineOutputs=lambda **kwargs: definedOutputs.update(kwargs),
            _defineSourceRelation=lambda *args: None,
            inputClassifications=object(),
        )

        result = XmippProtConsensusClasses._obtainMergedIntersections(prot, 1)

        self.assertIs(result, outputClasses)
        self.assertIs(prot._createSetOfClasses.call_args.kwargs['images'], images)
        self.assertEqual({'merged1': outputClasses}, definedOutputs)


if __name__ == "__main__":
    unittest.main()
