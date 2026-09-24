# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

import os
from datetime import datetime

from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from pwem.objects import SetOfMicrographs, SetOfMovies

from xmipp3.protocols.protocol_movie_max_shift import (
    OUTPUT_MICS_DISCARDED,
    OUTPUT_MOVIES,
    XmippProtMovieMaxShift,
)


class _FakeInputSet:
    def __init__(self, ids, streamClosed=False, items=None):
        self._ids = set(ids)
        self._streamClosed = streamClosed
        self._items = items or {}
        self.closed = False

    def getIdSet(self):
        return set(self._ids)

    def getSize(self):
        return len(self._ids)

    def isStreamClosed(self):
        return self._streamClosed

    def getItem(self, _, itemId):
        return self._items[itemId]

    def close(self):
        self.closed = True


class _FakeItem:
    def __init__(self, itemId):
        self.itemId = itemId
        self.enabled = True

    def clone(self):
        clone = _FakeItem(self.itemId)
        clone.enabled = self.enabled
        return clone

    def setEnabled(self, enabled):
        self.enabled = enabled


class _FakeOutputSet:
    def __init__(self):
        self.items = []
        self.closed = False

    def append(self, item):
        self.items.append(item)

    def getSize(self):
        return len(self.items)

    def close(self):
        self.closed = True


class TestXmippMovieMaxShiftRegression(BaseTest):
    """Regression tests for Movie Max Shift streaming and Resume handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtMovieMaxShift)
        prot.movsFn = 'unused.sqlite'
        prot.insertedIds = []
        prot.acceptedIds = []
        prot.discardedIds = []
        prot.isStreamClosed = False
        prot.alreadyLoad = True
        return prot

    def _prepareInputCheck(self, prot, ids, streamClosed=False):
        fakeSet = _FakeInputSet(ids, streamClosed=streamClosed)
        scheduled = []
        updates = []

        prot._loadInputSet = lambda _: fakeSet
        prot._getFirstJoinStep = lambda: None

        def insertSteps(newIds):
            newIds = sorted(newIds)
            scheduled.append(newIds)
            prot.insertedIds.extend(newIds)
            return []

        prot._insertNewMoviesSteps = insertSteps
        prot.updateSteps = lambda: updates.append(True)

        return fakeSet, scheduled, updates

    def testStreamingInputDoesNotDependOnSqliteMtime(self):
        prot = self._newProtocol()

        fnMovies = self.proj.getTmpPath('movie_max_shift_stream.sqlite')
        with open(fnMovies, 'w'):
            pass

        originalMtime = os.path.getmtime(fnMovies)
        prot.movsFn = fnMovies
        prot.insertedIds = [1]

        # This reproduced the old early-return condition.
        prot.lastCheck = datetime.fromtimestamp(originalMtime + 60)

        _, scheduled, updates = self._prepareInputCheck(
            prot,
            ids=[1, 2],
            streamClosed=False
        )

        prot._checkNewInput()

        self.assertEqual([[2]], scheduled)
        self.assertEqual(1, len(updates))

    def testResumeSkipsMoviesAlreadyPersistedInOutputs(self):
        prot = self._newProtocol()
        prot._originalRunMode = MODE_RESUME
        prot.runMode.set(MODE_RESUME)
        prot._getAllDoneIds = lambda: ([1], 1, [1], [])

        _, scheduled, updates = self._prepareInputCheck(
            prot,
            ids=[1, 2, 3],
            streamClosed=False
        )

        prot._checkNewInput()

        self.assertEqual([[2, 3]], scheduled)
        self.assertEqual([1, 2, 3], sorted(prot.insertedIds))
        self.assertEqual(1, len(updates))

    def testRestartDoesNotReusePreviousOutputs(self):
        prot = self._newProtocol()

        # Reproduce Protocol._runSteps(): runMode is changed to RESUME while
        # _originalRunMode keeps the action originally requested by the user.
        prot._originalRunMode = MODE_RESTART
        prot.runMode.set(MODE_RESUME)

        def failIfDoneOutputsAreRead():
            raise AssertionError('Restart must not restore previous output IDs.')

        prot._getAllDoneIds = failIfDoneOutputsAreRead

        _, scheduled, updates = self._prepareInputCheck(
            prot,
            ids=[1, 2, 3],
            streamClosed=False
        )

        prot._checkNewInput()

        self.assertEqual([[1, 2, 3]], scheduled)
        self.assertEqual(1, len(updates))

    def testDiscardedMicrographsAreDeclaredAsPossibleOutput(self):
        self.assertEqual(
            'outputMicrographsDiscarded',
            OUTPUT_MICS_DISCARDED
        )
        self.assertIn(
            'outputMicrographsDiscarded',
            XmippProtMovieMaxShift._possibleOutputs
        )
        self.assertIs(
            SetOfMicrographs,
            XmippProtMovieMaxShift._possibleOutputs[
                'outputMicrographsDiscarded'
            ]
        )

    def testStreamingMovieOutputReusesLogicalSetWithoutLegacySqlite(self):
        from unittest.mock import patch

        class LogicalOutputSet:
            def __init__(self):
                self.enableAppendCalls = 0

            def enableAppend(self):
                self.enableAppendCalls += 1

        class FreshOutputSet:
            STREAM_OPEN = 1

            def __init__(self, filename=None):
                self.filename = filename

            def setStreamState(self, state):
                self.streamState = state

            def copyInfo(self, inputSet):
                self.inputSet = inputSet

        prot = self._newProtocol()

        logicalOutput = LogicalOutputSet()
        prot.outputMovies = logicalOutput
        prot.inputMics = None
        prot._loadInputSet = (
            lambda _:
            _FakeInputSet(ids=[1])
        )
        prot._getPath = (
            lambda baseName:
            '/tmp/' + baseName
        )

        with patch(
            'xmipp3.protocols.protocol_movie_max_shift.exists',
            return_value=False,
        ):
            outputSet = prot._loadOutputSet(
                FreshOutputSet,
                'movies.sqlite',
            )

        self.assertIs(
            outputSet,
            logicalOutput,
            "Streaming movie output must reuse the logical output "
            "when the legacy SQLite file is absent.",
        )
        self.assertEqual(
            1,
            logicalOutput.enableAppendCalls,
        )

    def testMovieOutputWorksWithoutAssociatedMicrographs(self):
        prot = self._newProtocol()
        prot.acceptedIds = [1]
        prot.inputMics = None
        prot.outMicName = None
        prot._getAllDoneIds = lambda: ([], 0, [], [])
        prot._getFirstJoinStep = lambda: None
        prot._store = lambda: None
        prot._defineTransformRelation = lambda *args, **kwargs: None

        movie = _FakeItem(1)

        def loadInputSet(_):
            return _FakeInputSet(
                ids=[1],
                streamClosed=False,
                items={1: movie}
            )

        prot._loadInputSet = loadInputSet

        movieOutput = _FakeOutputSet()

        def loadOutputSet(SetClass, _):
            if SetClass is SetOfMovies:
                return movieOutput
            if SetClass is SetOfMicrographs:
                return None
            raise AssertionError('Unexpected output Set class.')

        prot._loadOutputSet = loadOutputSet

        def failIfMicsAreLoaded():
            raise AssertionError(
                'Associated micrographs must not be loaded when inputMics is None.'
            )

        prot._loadMicAssociatedInputSet = failIfMicsAreLoaded

        updatedOutputs = []
        prot._updateOutputSet = (
            lambda outputName, outputSet, streamMode:
            updatedOutputs.append((outputName, outputSet, streamMode))
        )

        prot._checkNewOutput()

        self.assertEqual(1, movieOutput.getSize())
        self.assertEqual(OUTPUT_MOVIES, updatedOutputs[0][0])

    def testMissingAssociatedMicrographSetReturnsNone(self):
        prot = self._newProtocol()
        prot.outMicName = None

        class FailMapper:
            def getParent(self, _):
                raise AssertionError(
                    'Parent protocol must not be queried without an output name.'
                )

        prot.getMapper = lambda: FailMapper()

        self.assertIsNone(prot._loadMicAssociatedInputSet())
