"""Regression: logical ID reads must survive native template iteration.

Native Scipion Sets (SQLite mapper, SCIPION_MAPPER_USE_TEMPLATE) yield one
reused object per row from iterItems(). Items kept beyond their own row must
be cloned while current, or every kept reference ends up as the last row.
"""
import os
import re
import shutil
import tempfile
import unittest

from pwem import objects as emobj

from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase


class _TemplateParticle:
    def __init__(self):
        self._item_id = None

    def getObjId(self):
        return self._item_id

    def clone(self):
        copy = _TemplateParticle()
        copy._item_id = self._item_id
        return copy


class _TemplateSet:
    """Refill one object per row, as the native SQLite mapper does."""
    def __init__(self, ids, supports_in=True):
        self.ids = sorted(ids)
        self.supports_in = supports_in
        self._template = _TemplateParticle()

    def loadAllProperties(self):
        pass

    def iterItems(self, orderBy='id', direction='ASC', where=None):
        if ' IN ' in where:
            if not self.supports_in:
                raise NotImplementedError(where)
            members = where.split('(', 1)[1].rsplit(')', 1)[0]
            wanted = {int(text) for text in members.split(',')}
            selected = [i for i in self.ids if i in wanted]
        else:
            lower, upper = map(int, re.fullmatch(r'id >= (\d+) AND id <= (\d+)', where).groups())
            selected = [i for i in self.ids if lower <= i <= upper]
        for itemId in selected:
            self._template._item_id = itemId
            yield self._template


class _Harness(XmippStreamingBase):
    def __init__(self, output_set=None):
        if output_set is not None:
            self.outputClasses = output_set


class TestLogicalIdsUnderTemplateIteration(unittest.TestCase):
    def test_MembershipFilterKeepsEveryRequestedItem(self):
        items = _Harness()._loadLogicalSetItemsByIds(_TemplateSet(range(1, 1201)), range(1, 1201))
        self.assertEqual(list(range(1, 1201)), [item.getObjId() for item in items])

    def test_RangeFilterKeepsEveryRequestedItem(self):
        source = _TemplateSet([2, 3, 5, 8, 13, 21], supports_in=False)
        items = _Harness()._loadLogicalSetItemsByIds(source, [21, 3, 8], batchSize=4)
        self.assertEqual([3, 8, 21], [item.getObjId() for item in items])

    def test_PersistedCandidateIdsAreExact(self):
        existing = _Harness(_TemplateSet([4, 6, 7, 8, 12]))._getPersistedCandidateIds(
            'outputClasses', [12, 5, 7, 4],
        )
        self.assertEqual({4, 7, 12}, existing)

    def test_NativeSqliteSetKeepsEveryRequestedParticle(self):
        workDir = tempfile.mkdtemp()
        try:
            fileName = os.path.join(workDir, 'particles.sqlite')
            particles = emobj.SetOfParticles(filename=fileName)
            particles.setSamplingRate(1.0)
            for itemId in range(1, 1201):
                particle = emobj.Particle()
                particle.setObjId(itemId)
                particle.setLocation(itemId, 'particles.mrcs')
                particle.setSamplingRate(1.0)
                particles.append(particle)
            particles.write()
            particles.close()

            particles = emobj.SetOfParticles(filename=fileName)
            try:
                requested = list(range(1, 1201, 7))
                items = _Harness()._loadLogicalSetItemsByIds(particles, requested)
                self.assertEqual(requested, [item.getObjId() for item in items])
                self.assertEqual(requested, [item.getIndex() for item in items])
            finally:
                particles.close()
        finally:
            shutil.rmtree(workDir)


if __name__ == '__main__':
    unittest.main()
