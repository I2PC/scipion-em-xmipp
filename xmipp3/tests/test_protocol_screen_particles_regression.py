# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from types import SimpleNamespace
from unittest.mock import patch

from pyworkflow.object import Float
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_screen_particles import XmippProtScreenParticles


class _FakeParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _FakeInputSet:
    def __init__(self, ids, streamClosed=False):
        self._particles = [_FakeParticle(objId) for objId in ids]
        self._streamClosed = streamClosed
        self.closed = False

    def loadAllProperties(self):
        pass

    def iterItems(self, orderBy='id', direction='ASC', where=None):
        return iter(self._particles)

    def isStreamClosed(self):
        return self._streamClosed

    def getSize(self):
        return len(self._particles)

    def close(self):
        self.closed = True


class _FakeOutputSet:
    def __init__(self, ids=None):
        self.ids = set(ids or [])
        self.appended = []

    def getSize(self):
        return len(self.ids)

    def getIdSet(self):
        return set(self.ids)

    def append(self, particle):
        self.ids.add(particle.getObjId())
        self.appended.append(particle.getObjId())

    def iterItems(self, orderBy='id'):
        return iter([])


class TestXmippScreenParticlesRegression(BaseTest):
    """Regression tests for Screen Particles streaming and Continue handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtScreenParticles)
        prot.fnInputMd = prot._getExtraPath('input.xmd')
        prot.fnInputOldMd = prot._getExtraPath('inputOld.xmd')
        prot.fnOutputMd = prot._getExtraPath('output.xmd')
        prot.fnProcessedIds = prot._getExtraPath('processed_ids.txt')
        return prot

    def testLoadInputUsesProcessedIdsInsteadOfCreationTimestamp(self):
        prot = self._newProtocol()
        inputSet = _FakeInputSet([1, 2, 3])
        prot.inputParticles = SimpleNamespace(get=lambda: SimpleNamespace(getFileName=lambda: 'input.sqlite'))
        prot._readProcessedIds = lambda: {1, 2}
        written = {}

        def captureWrite(items, filename, **kwargs):
            written[filename] = [item.getObjId() for item in items]

        with patch('xmipp3.protocols.protocol_screen_particles.SetOfParticles', return_value=inputSet), patch('xmipp3.protocols.protocol_screen_particles.writeSetOfParticles', side_effect=captureWrite):
            inputSize, streamClosed = prot._loadInput()

        self.assertEqual([3], written[prot.fnInputMd])
        self.assertEqual([1, 2], written[prot.fnInputOldMd])
        self.assertEqual(3, inputSize)
        self.assertFalse(streamClosed)
        self.assertTrue(inputSet.closed)

    def testNewInputDoesNotDependOnSqliteMtime(self):
        prot = self._newProtocol()
        prot._loadInput = lambda: (3, False)
        prot._insertNewPartsSteps = lambda: []
        prot._getFirstJoinStep = lambda: None
        prot.updateSteps = lambda: None

        with patch('xmipp3.protocols.protocol_screen_particles.os.path.getmtime', side_effect=AssertionError('Streaming input must not depend on SQLite mtime.')), patch('xmipp3.protocols.protocol_screen_particles.os.path.exists', return_value=False), patch('xmipp3.protocols.protocol_screen_particles.isEmpty', return_value=False):
            prot._checkNewInput()

    def testAppendNewParticlesSkipsAlreadyPersistedIds(self):
        prot = self._newProtocol()
        outputSet = _FakeOutputSet([1])
        prot._appendNewParticles(outputSet, [_FakeParticle(1), _FakeParticle(2)])
        self.assertEqual({1, 2}, outputSet.ids)
        self.assertEqual([2], outputSet.appended)

    def testAppendNewParticlesDoesNotQueryIdsOnFreshSet(self):
        prot = self._newProtocol()

        class _FreshOutputSet(_FakeOutputSet):
            def getIdSet(self):
                raise AssertionError('Fresh output Set must not query IDs before its first append.')

        outputSet = _FreshOutputSet()
        prot._appendNewParticles(outputSet, [_FakeParticle(1)])
        self.assertEqual({1}, outputSet.ids)
        self.assertEqual([1], outputSet.appended)

    def testStreamingOutputReusesLogicalSetWithoutLegacySqlite(self):
        class LogicalOutputSet:
            def __init__(self):
                self.enableAppendCalls = 0
                self.copyInfoCalls = 0

            def enableAppend(self):
                self.enableAppendCalls += 1

            def copyInfo(self, inputSet):
                self.copyInfoCalls += 1

        class FreshOutputSet:
            STREAM_OPEN = 1

            def __init__(self, filename=None):
                self.filename = filename

            def setStreamState(self, state):
                self.streamState = state

            def copyInfo(self, inputSet):
                self.inputSet = inputSet

        inputSet = object()
        prot = self._newProtocol()
        logicalOutput = LogicalOutputSet()
        prot.outputParticles = logicalOutput
        prot.inputParticles = SimpleNamespace(get=lambda: inputSet)
        prot._getPath = lambda baseName: '/tmp/' + baseName

        with patch(
            'xmipp3.protocols.protocol_screen_particles.os.path.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSet,
                'outputParticles.sqlite',
            )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming particles output must reuse the logical Set when "
            "the legacy SQLite file is absent.",
        )
        self.assertEqual(1, logicalOutput.enableAppendCalls)
        self.assertEqual(1, logicalOutput.copyInfoCalls)

    def testProcessedCheckpointIsWrittenAfterOutputUpdate(self):
        prot = self._newProtocol()
        prot.finished = False
        prot.streamClosed = False
        prot.outputSize = 1
        prot.inputSize = 2
        outputSet = _FakeOutputSet()
        events = []

        prot._loadOutputSet = lambda *args: outputSet
        prot._createSetOfParticles = lambda: [_FakeParticle(2)]
        prot._readMetadataIds = lambda _: [2]
        prot._appendNewParticles = lambda *args: None
        prot._recalculateSummaryValues = lambda *args: None
        prot._updateOutputSet = lambda *args: events.append('update')
        prot._writeProcessedIds = lambda ids: events.append('checkpoint')
        prot._readProcessedIds = lambda: {1, 2}
        prot._store = lambda *args: None

        with patch('xmipp3.protocols.protocol_screen_particles.os.path.exists', return_value=True), patch('xmipp3.protocols.protocol_screen_particles.readSetOfParticles'), patch('xmipp3.protocols.protocol_screen_particles.writeSetOfParticles'), patch('xmipp3.protocols.protocol_screen_particles.cleanPath'):
            prot._checkNewOutput()

        self.assertEqual(['update', 'checkpoint'], events)

    def testContinuePreservesVarianceThreshold(self):
        prot = self._newProtocol()
        prot.minZScore = Float(1.0)
        prot.maxZScore = Float(2.0)
        prot.sumZScore = Float(3.0)
        prot.varThreshold = Float(4.0)
        prot.isContinued = lambda: True
        prot._store = lambda *args: None
        prot._initializeZscores()
        self.assertEqual(4.0, prot.varThreshold.get())
