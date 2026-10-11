"""Regression: exact logical ID reads across different Set interfaces.

Some valid Set implementations reject membership filters but support bounded
comparisons. Direct lookups must not be used when filtered reads are supported.
"""
import re
import unittest

from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase


class _Particle:
    def __init__(self, item_id):
        self._item_id = item_id

    def getObjId(self):
        return self._item_id

    def clone(self):
        return _Particle(self._item_id)


class _RestrictedFilterSet:
    """Model a Set that supports bounded filters but not membership filters."""
    def __init__(self, ids, supports_in=False):
        self.ids = sorted(set(ids))
        self.supports_in = supports_in
        self.queries = []
        self.get_item_calls = 0
        self.refresh_calls = 0

    def loadAllProperties(self):
        self.refresh_calls += 1

    def iterItems(self, orderBy='id', direction='ASC', where=None):
        assert orderBy == 'id'
        assert direction == 'ASC'
        self.queries.append(where)

        if ' IN ' in where:
            if not self.supports_in:
                raise NotImplementedError(
                    'Unsupported Set filter expression: %s' % where
                )
            members = where.split('(', 1)[1].rsplit(')', 1)[0]
            requested = {int(text.strip()) for text in members.split(',')}
            return iter(_Particle(item_id) for item_id in self.ids
                        if item_id in requested)

        # This Set implementation accepts bounded comparisons.
        match = re.fullmatch(r'id >= (\d+) AND id <= (\d+)', where or '')
        if match is None:
            raise AssertionError('Unbounded or unexpected filter: %r' % where)
        lower, upper = (int(value) for value in match.groups())
        return iter(_Particle(item_id) for item_id in self.ids
                    if lower <= item_id <= upper)

    def getItem(self, field, item_id):
        self.get_item_calls += 1
        # Direct lookup would access an unavailable storage mapper.
        raise AttributeError("'NoneType' object has no attribute 'selectBy'")

    def __iter__(self):
        raise AssertionError('Do not load every item from a large Set')

    def getIdSet(self):
        raise AssertionError('Do not scan every id from the output Set')


class _Harness(XmippStreamingBase):
    def __init__(self, output_set=None):
        if output_set is not None:
            self.outputClasses = output_set


class TestXmippBoundedLogicalIdFallback(unittest.TestCase):
    def test_ExactInputIdsSurviveUnsupportedInWithoutAccessingClosedMapper(self):
        images = _RestrictedFilterSet([5, 7, 8, 9, 11, 12])
        particles = _Harness()._loadLogicalSetItemsByIds(
            images, [11, 7, 7], batchSize=4,
        )
        self.assertEqual([7, 11], [p.getObjId() for p in particles])
        self.assertEqual(0, images.get_item_calls)
        self.assertTrue(any(' AND ' in q for q in images.queries))

    def test_SparseInputIdsNeverTriggerWideRangeScan(self):
        images = _RestrictedFilterSet([1, 2, 3, 200, 201, 202, 50000, 50001])
        particles = _Harness()._loadLogicalSetItemsByIds(
            images, [50000, 2, 201, 2], batchSize=2,
        )
        self.assertEqual([2, 201, 50000], [p.getObjId() for p in particles])
        range_queries = [q for q in images.queries if ' AND ' in q]
        self.assertTrue(range_queries)
        for query in range_queries:
            lower, upper = map(int, re.findall(r'\d+', query))
            self.assertLessEqual(upper - lower, 4)
        self.assertEqual(0, images.get_item_calls)

    def test_ExistingInFastPathRemainsAvailable(self):
        images = _RestrictedFilterSet([2, 3, 4], supports_in=True)
        particles = _Harness()._loadLogicalSetItemsByIds(images, [4, 2])
        self.assertEqual([2, 4], [p.getObjId() for p in particles])
        self.assertEqual(['id IN (2,4)'], images.queries)
        self.assertEqual(0, images.get_item_calls)

    def test_PersistedCandidatesDoNotFallbackToClosedMapper(self):
        output = _RestrictedFilterSet([4, 6, 7, 8, 12])
        existing = _Harness(output)._getPersistedCandidateIds(
            'outputClasses', [12, 5, 7, 7],
        )
        self.assertEqual({7, 12}, existing)
        self.assertEqual(0, output.get_item_calls)
        self.assertEqual(1, output.refresh_calls)
        self.assertTrue(any(' AND ' in q for q in output.queries))

    def test_UnavailableInputIdsAreNotInventedOrReplaced(self):
        images = _RestrictedFilterSet([3, 7, 11])
        particles = _Harness()._loadLogicalSetItemsByIds(
            images, [7, 8, 11], batchSize=3,
        )
        self.assertEqual([7, 11], [p.getObjId() for p in particles])
        self.assertEqual(0, images.get_item_calls)
        # The AlignPCA caller checks these IDs against the requested batch
        # and fails explicitly if any particle is missing.


if __name__ == '__main__':
    unittest.main()
