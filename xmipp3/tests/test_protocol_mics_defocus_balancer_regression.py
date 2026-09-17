# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_mics_defocus_balancer import (
    XmippProtMicDefocusSampler,
    balanced_sampling,
    compute_statistics,
)


class _FakeCtfSet:
    def __init__(self, ids, streamClosed=False):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self.closed = False

    def getIdSet(self):
        return set(self._ids)

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closed = True


class _FakeItem:
    def __init__(self, itemId, micrograph=None):
        self._itemId = itemId
        self._micrograph = micrograph

    def clone(self):
        micrograph = self._micrograph.clone() if self._micrograph is not None else None
        return _FakeItem(self._itemId, micrograph)

    def getObjId(self):
        return self._itemId

    def getMicrograph(self):
        return self._micrograph


class _FakeItemSet:
    def __init__(self, items=None):
        self._items = {item.getObjId(): item for item in (items or [])}
        self.closed = False

    def __getitem__(self, itemId):
        return self._items[itemId]

    def __iter__(self):
        return iter(self._items.values())

    def getIdSet(self):
        return set(self._items)

    def append(self, item):
        self._items[item.getObjId()] = item

    def close(self):
        self.closed = True


class TestXmippMicDefocusSamplerRegression(BaseTest):
    """Regression tests for defocus-sampler streaming and sampling logic."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self, minImages=3, numImages=2):
        prot = self.newProtocol(
            XmippProtMicDefocusSampler,
            minImages=minImages,
            numImages=numImages
        )
        prot.ctfFn = 'unused.sqlite'
        prot.insertedIds = []
        prot.sampled_images = []
        prot.finished = False
        return prot

    def _prepareCheck(self, prot, ids, streamClosed=False):
        fakeSet = _FakeCtfSet(ids, streamClosed=streamClosed)
        scheduled = []
        updates = []

        prot._loadInputCtfSet = lambda _: fakeSet
        prot._getFirstJoinStep = lambda: None
        prot._insertNewCtfsSteps = lambda newIds: scheduled.append(list(newIds)) or []
        prot.updateSteps = lambda: updates.append(True)

        return fakeSet, scheduled, updates

    def testBalancedSamplingCapsSampleSizeToPopulation(self):
        imageDict = {
            1: 1000.0,
            2: 2000.0,
            3: 3000.0,
        }

        sampled = balanced_sampling(imageDict, N=10, bins=10)

        self.assertEqual(3, len(sampled))
        self.assertEqual(set(imageDict), set(sampled))

    def testBalancedSamplingSmallSampleCoversDefocusRange(self):
        imageDict = {
            imageId: float(imageId * 1000)
            for imageId in range(1, 11)
        }

        sampled = balanced_sampling(imageDict, N=2, bins=10)

        self.assertEqual({1, 10}, set(sampled))

    def testSingleValueStatisticsHaveZeroVariance(self):
        stats = compute_statistics([15000.0])

        self.assertEqual(0.0, stats['std'])
        self.assertEqual(0.0, stats['variance'])
        self.assertEqual(0.0, stats['range'])

    def testValidationRejectsNonPositiveSamplingParameters(self):
        prot = self._newProtocol(minImages=0, numImages=0)

        errors = prot._validate()

        self.assertEqual(2, len(errors))

    def testOnlyOneSamplingStepIsScheduled(self):
        prot = self._newProtocol(minImages=3, numImages=2)
        prot.insertedIds = [1, 2, 3]
        _, scheduled, updates = self._prepareCheck(
            prot,
            ids=[1, 2, 3, 4, 5, 6],
            streamClosed=False
        )

        prot._checkNewInput()

        self.assertEqual([], scheduled)
        self.assertEqual([], updates)

    def testClosedStreamSchedulesAvailableCtfsBelowMinimum(self):
        prot = self._newProtocol(minImages=100, numImages=25)
        _, scheduled, updates = self._prepareCheck(
            prot,
            ids=[1, 2, 3],
            streamClosed=True
        )

        prot._checkNewInput()

        self.assertEqual([[1, 2, 3]], [sorted(ids) for ids in scheduled])
        self.assertEqual(1, len(updates))

    def testEmptyClosedStreamFinishesWithoutScheduling(self):
        prot = self._newProtocol()
        fakeSet, scheduled, updates = self._prepareCheck(
            prot,
            ids=[],
            streamClosed=True
        )

        prot._checkNewInput()

        self.assertTrue(prot.finished)
        self.assertEqual([], scheduled)
        self.assertEqual([], updates)
        self.assertTrue(fakeSet.closed)

    def testResumeWithExistingOutputDoesNotSampleAgain(self):
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESUME
        _, scheduled, updates = self._prepareCheck(
            prot,
            ids=[1, 2, 3],
            streamClosed=True
        )
        prot._getAllDoneIds = lambda: ([1, 2], 2)

        prot._checkNewInput()

        self.assertFalse(prot.finished)
        self.assertEqual([1, 2], prot.sampled_images)
        self.assertEqual([], scheduled)
        self.assertEqual([], updates)

    def testRestartDoesNotReusePreviousOutputState(self):
        prot = self._newProtocol(minImages=3, numImages=2)
        prot._originalRunMode = MODE_RESTART
        _, scheduled, updates = self._prepareCheck(
            prot,
            ids=[1, 2, 3],
            streamClosed=False
        )
        prot._getAllDoneIds = lambda: ([1, 2], 2)

        prot._checkNewInput()

        self.assertEqual([[1, 2, 3]], [sorted(ids) for ids in scheduled])
        self.assertEqual(1, len(updates))

    def testResumeUsesPersistedSampleWithoutReadingInputAgain(self):
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESUME
        prot.sampledIds.set([2, 3])
        prot.sampled_images = list(prot.sampledIds)
        prot._loadInputCtfSet = lambda _: self.fail('Input should not be reopened when sampled ids are persisted.')

        prot._checkNewInput()

        self.assertEqual([2, 3], prot.sampled_images)

    def testFillOutputDoesNotDuplicateItemsOnResume(self):
        prot = self._newProtocol()
        mic1 = _FakeItem(101)
        mic2 = _FakeItem(102)
        ctf1 = _FakeItem(1, mic1)
        ctf2 = _FakeItem(2, mic2)
        inputSet = _FakeItemSet([ctf1, ctf2])
        ctfOutput = _FakeItemSet([ctf1.clone()])
        micOutput = _FakeItemSet([mic1.clone()])
        prot._loadInputCtfSet = lambda _: inputSet

        prot.fillOutput(ctfOutput, micOutput, [1, 2])

        self.assertEqual({1, 2}, ctfOutput.getIdSet())
        self.assertEqual({101, 102}, micOutput.getIdSet())
        self.assertTrue(inputSet.closed)

    def testResumeCanRecoverCtfIdsFromMicrographOnlyOutput(self):
        prot = self._newProtocol()
        mic1 = _FakeItem(101)
        mic2 = _FakeItem(102)
        inputSet = _FakeItemSet([_FakeItem(1, mic1), _FakeItem(2, mic2)])
        prot.outputMicrographs = _FakeItemSet([mic2.clone()])
        prot._loadInputCtfSet = lambda _: inputSet

        doneIds, sizeOutput = prot._getAllDoneIds()

        self.assertEqual([2], doneIds)
        self.assertEqual(1, sizeOutput)
        self.assertTrue(inputSet.closed)
