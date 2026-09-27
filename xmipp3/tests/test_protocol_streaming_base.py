# **************************************************************************
# *
# * Streaming-base regression tests.
# *
# **************************************************************************

import importlib
import importlib.util
import unittest

from pwem.protocols import ProtProcessMovies

from xmipp3.protocols.protocol_movie_max_shift import XmippProtMovieMaxShift


STREAMING_MODULE = "xmipp3.protocols.protocol_streaming_base"


class _Pointer:
    def __init__(self, value):
        self._value = value
        self.getCalls = 0

    def get(self):
        self.getCalls += 1
        return self._value


class _LogicalSet:
    def __init__(self):
        self.loadCalls = 0

    def loadAllProperties(self):
        self.loadCalls += 1


class _DiscoverySet:
    def __init__(self, ids):
        self.ids = list(ids)
        self.calls = []

    def getUniqueValues(self, field, where=None):
        self.calls.append((field, where))
        return list(self.ids)


class _ClosedSet:
    def __init__(self, expectedSize, visibleIds):
        self.expectedSize = expectedSize
        self.visibleIds = list(visibleIds)
        self.sizeCalls = 0
        self.uniqueCalls = []

    def getSize(self):
        self.sizeCalls += 1
        return self.expectedSize

    def getUniqueValues(self, field, where=None):
        self.uniqueCalls.append((field, where))
        return list(self.visibleIds)


class TestXmippStreamingBase(unittest.TestCase):

    def _getBaseClass(self):
        spec = importlib.util.find_spec(STREAMING_MODULE)

        self.assertIsNotNone(
            spec,
            "Xmipp needs a shared backend-independent streaming base.",
        )

        module = importlib.import_module(STREAMING_MODULE)

        self.assertTrue(
            hasattr(module, "XmippStreamingBase"),
            "protocol_streaming_base.py must expose XmippStreamingBase.",
        )

        return module.XmippStreamingBase

    def testMovieMaxShiftUsesSharedStreamingBase(self):
        baseClass = self._getBaseClass()

        self.assertTrue(
            issubclass(XmippProtMovieMaxShift, baseClass),
            "MovieMaxShift must reuse the shared Xmipp streaming helpers.",
        )

        self.assertTrue(
            issubclass(XmippProtMovieMaxShift, ProtProcessMovies),
            "MovieMaxShift must keep its native ProtProcessMovies behavior.",
        )

    def testLogicalSetLoadingUsesPointerObject(self):
        baseClass = self._getBaseClass()
        logicalSet = _LogicalSet()
        pointer = _Pointer(logicalSet)

        loaded = baseClass()._loadLogicalSet(pointer)

        self.assertIs(loaded, logicalSet)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(logicalSet.loadCalls, 1)

    def testDiscoveryQueriesOnlyIdsBeyondWatermark(self):
        baseClass = self._getBaseClass()
        logicalSet = _DiscoverySet([7, 9])

        ids, watermark = baseClass()._discoverIdsAfter(
            logicalSet,
            5,
        )

        self.assertEqual(ids, [7, 9])
        self.assertEqual(watermark, 9)
        self.assertEqual(
            logicalSet.calls,
            [("id", "id > 5")],
        )

    def testOpenStreamReconciliationDoesNotFullScan(self):
        baseClass = self._getBaseClass()
        logicalSet = _ClosedSet(
            expectedSize=100,
            visibleIds=list(range(1, 101)),
        )

        base = baseClass()
        base._lastInputId = 11

        newIds, terminalConsistent = (
            base._reconcileClosedStreamIds(
                logicalSet,
                discoveredIds=[11],
                knownIds=set(range(1, 11)),
                producerClosed=False,
            )
        )

        self.assertEqual(newIds, [11])
        self.assertTrue(terminalConsistent)
        self.assertEqual(logicalSet.sizeCalls, 0)
        self.assertEqual(logicalSet.uniqueCalls, [])

    def testClosedStreamReconciliationRecoversLateVisibleIds(self):
        baseClass = self._getBaseClass()
        logicalSet = _ClosedSet(
            expectedSize=10,
            visibleIds=list(range(1, 11)),
        )

        base = baseClass()
        base._lastInputId = 10

        newIds, terminalConsistent = (
            base._reconcileClosedStreamIds(
                logicalSet,
                discoveredIds=[9, 10],
                knownIds=set(),
                producerClosed=True,
            )
        )

        self.assertEqual(newIds, list(range(1, 11)))
        self.assertTrue(terminalConsistent)
        self.assertEqual(base._lastInputId, 10)
        self.assertEqual(logicalSet.sizeCalls, 1)
        self.assertEqual(
            logicalSet.uniqueCalls,
            [("id", None)],
        )

class _RunMode:
    def get(self):
        return 0


class _StreamingMovieSet:
    def __init__(self):
        self.loadCalls = 0
        self.closeCalls = 0
        self.uniqueCalls = []

    def loadAllProperties(self):
        self.loadCalls += 1

    def getUniqueValues(self, field, where=None):
        self.uniqueCalls.append((field, where))

        if where == "id > 3":
            return [4, 5]

        if where == "id > 5":
            return [6]

        raise AssertionError(
            "Unexpected discovery query: %r" % (where,)
        )

    def isStreamClosed(self):
        return False

    def close(self):
        self.closeCalls += 1


class TestXmippMovieMaxShiftStreamingInput(unittest.TestCase):

    def testMovieMaxShiftUsesLogicalIncrementalInputDiscovery(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        inputSet = _StreamingMovieSet()
        pointer = _Pointer(inputSet)

        class _Harness:
            inputMovies = pointer
            runMode = _RunMode()
            insertedIds = [1, 2, 3]
            _lastInputId = 3
            isStreamClosed = False

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieMaxShift streaming must not depend on "
                    "inputMovies.getFileName()."
                )

            def _loadLogicalSet(self, inputPointer):
                return XmippStreamingBase._loadLogicalSet(
                    self,
                    inputPointer,
                )

            def _discoverIdsAfter(self, logicalSet, lastId):
                return XmippStreamingBase._discoverIdsAfter(
                    self,
                    logicalSet,
                    lastId,
                )

            def _reconcileClosedStreamIds(
                    self,
                    logicalSet,
                    discoveredIds,
                    knownIds,
                    producerClosed,
            ):
                return XmippStreamingBase._reconcileClosedStreamIds(
                    self,
                    logicalSet,
                    discoveredIds,
                    knownIds,
                    producerClosed,
                )

            def _getFirstJoinStep(self):
                return None

            def _insertNewMoviesSteps(self, newIds):
                newIds = list(newIds)
                self.batches.append(newIds)
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

            def info(self, message):
                pass

        protocol = _Harness()
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtMovieMaxShift._checkNewInput(protocol)
        XmippProtMovieMaxShift._checkNewInput(protocol)

        self.assertEqual(
            protocol.batches,
            [[4, 5], [6]],
        )
        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 3"),
                ("id", "id > 5"),
            ],
        )
        self.assertEqual(protocol._lastInputId, 6)
        self.assertEqual(protocol.insertedIds, [1, 2, 3, 4, 5, 6])
        self.assertEqual(pointer.getCalls, 2)
        self.assertEqual(inputSet.loadCalls, 2)
        self.assertEqual(inputSet.closeCalls, 2)
        self.assertEqual(protocol.updateCalls, 2)

class _MovieAlignment:
    def getShifts(self):
        return [0.0, 0.0], [0.0, 0.0]


class _Movie:
    def clone(self):
        return self

    def getAlignment(self):
        return _MovieAlignment()


class _WorkerMovieSet(_LogicalSet):
    def __init__(self):
        super().__init__()
        self.closeCalls = 0
        self.getItemCalls = []

    def getItem(self, field, value):
        self.getItemCalls.append((field, value))
        return _Movie()

    def close(self):
        self.closeCalls += 1


class TestXmippMovieMaxShiftLogicalWorkerInput(unittest.TestCase):

    def testEvaluateMovieAlignUsesLogicalInputSet(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        inputSet = _WorkerMovieSet()
        pointer = _Pointer(inputSet)

        class _Harness:
            inputMovies = pointer
            samplingRate = 1.0
            rejType = XmippProtMovieMaxShift.REJ_FRAME
            REJ_FRAME = XmippProtMovieMaxShift.REJ_FRAME
            REJ_MOVIE = XmippProtMovieMaxShift.REJ_MOVIE
            REJ_AND = XmippProtMovieMaxShift.REJ_AND
            REJ_OR = XmippProtMovieMaxShift.REJ_OR

            acceptedIds = []
            discardedIds = []

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieMaxShift workers must not depend on "
                    "inputMovies.getFileName()."
                )

            def _loadLogicalSet(self, inputPointer):
                return XmippStreamingBase._loadLogicalSet(
                    self,
                    inputPointer,
                )

        protocol = _Harness()
        protocol.acceptedIds = []
        protocol.discardedIds = []

        XmippProtMovieMaxShift._evaluateMovieAlign(
            protocol,
            [4, 5],
        )

        self.assertEqual(
            inputSet.getItemCalls,
            [("id", 4), ("id", 5)],
        )
        self.assertEqual(protocol.acceptedIds, [4, 5])
        self.assertEqual(protocol.discardedIds, [])
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

class _InitializeMovieSet:
    def __init__(self):
        self.samplingCalls = 0
        self.streamCalls = 0

    def getSamplingRate(self):
        self.samplingCalls += 1
        return 1.23

    def isStreamClosed(self):
        self.streamCalls += 1
        return False

    def getFileName(self):
        raise AssertionError(
            "MovieMaxShift initialization must not depend on "
            "a file-backed input Set."
        )


class TestXmippMovieMaxShiftLogicalInitialization(unittest.TestCase):

    def testInitializeStepDoesNotRequireInputFilename(self):
        inputSet = _InitializeMovieSet()
        pointer = _Pointer(inputSet)

        class _Harness:
            inputMovies = pointer

        protocol = _Harness()

        XmippProtMovieMaxShift.initializeStep(protocol)

        self.assertEqual(protocol.samplingRate, 1.23)
        self.assertEqual(protocol.insertedIds, [])
        self.assertEqual(protocol._lastInputId, 0)
        self.assertEqual(protocol.acceptedIds, [])
        self.assertEqual(protocol.discardedIds, [])
        self.assertFalse(protocol.isStreamClosed)
        self.assertFalse(protocol.alreadyLoad)
        self.assertEqual(inputSet.samplingCalls, 1)
        self.assertEqual(inputSet.streamCalls, 1)

class _PersistedOutputSet:
    def __init__(self, ids):
        self.ids = set(ids)
        self.getIdSetCalls = 0

    def getIdSet(self):
        self.getIdSetCalls += 1
        return set(self.ids)


class TestXmippStreamingPersistedOutputState(unittest.TestCase):

    def testPersistedOutputIdsAreRestoredOnlyOnce(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        output = _PersistedOutputSet({1, 3, 5})

        class _Harness(XmippStreamingBase):
            outputMovies = output

        protocol = _Harness()

        first = protocol._restorePersistedOutputIds(
            "outputMovies",
        )
        second = protocol._restorePersistedOutputIds(
            "outputMovies",
        )

        self.assertEqual(first, {1, 3, 5})
        self.assertEqual(second, {1, 3, 5})
        self.assertEqual(output.getIdSetCalls, 1)

    def testPersistedOutputCacheTracksNewlyPublishedIds(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        output = _PersistedOutputSet({2})

        class _Harness(XmippStreamingBase):
            outputMovies = output

        protocol = _Harness()

        protocol._restorePersistedOutputIds(
            "outputMovies",
        )
        protocol._markOutputIdsPersisted(
            "outputMovies",
            [4, 6],
        )

        self.assertEqual(
            protocol._getKnownPersistedOutputIds(
                "outputMovies",
            ),
            {2, 4, 6},
        )
        self.assertEqual(output.getIdSetCalls, 1)

class _CountingOutputSet:
    def __init__(self, ids):
        self.ids = set(ids)
        self.getIdSetCalls = 0

    def getIdSet(self):
        self.getIdSetCalls += 1
        return set(self.ids)

    def getSize(self):
        return len(self.ids)


class TestXmippMovieMaxShiftDoneIdsCache(unittest.TestCase):

    def testGetAllDoneIdsDoesNotRescanPersistedOutputs(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        accepted = _CountingOutputSet({1, 3})
        discarded = _CountingOutputSet({2, 4})

        class _Harness(XmippStreamingBase):
            outputMovies = accepted
            outputMoviesDiscarded = discarded

        protocol = _Harness()

        first = XmippProtMovieMaxShift._getAllDoneIds(
            protocol,
        )
        second = XmippProtMovieMaxShift._getAllDoneIds(
            protocol,
        )

        expectedDone = [1, 3, 2, 4]
        expectedAccepted = [1, 3]
        expectedDiscarded = [2, 4]

        self.assertEqual(first[0], expectedDone)
        self.assertEqual(first[1], 4)
        self.assertEqual(first[2], expectedAccepted)
        self.assertEqual(first[3], expectedDiscarded)

        self.assertEqual(second, first)

        self.assertEqual(
            accepted.getIdSetCalls,
            1,
        )
        self.assertEqual(
            discarded.getIdSetCalls,
            1,
        )

class _OutputInfoSource:
    def __init__(self):
        self.marker = "logical-input"


class _FakeOutputSet:
    STREAM_OPEN = 1

    def __init__(self, filename=None):
        self.filename = filename
        self.streamState = None
        self.copiedFrom = None
        self.appendEnabled = False

    def setStreamState(self, state):
        self.streamState = state

    def copyInfo(self, inputSet):
        self.copiedFrom = inputSet

    def enableAppend(self):
        self.appendEnabled = True

    def loadAllProperties(self):
        pass

    def __len__(self):
        return 0


class TestXmippMovieMaxShiftLogicalOutputCreation(unittest.TestCase):

    def testMovieOutputCreationCopiesInfoFromLogicalInput(self):
        logicalInput = _OutputInfoSource()
        pointer = _Pointer(logicalInput)

        class _Harness:
            inputMovies = pointer
            inputMics = None

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieMaxShift output creation must not depend "
                    "on inputMovies.getFileName()."
                )

            def _getPath(self, baseName):
                return "/tmp/xmipp-streaming-%s" % baseName

        protocol = _Harness()

        output = XmippProtMovieMaxShift._loadOutputSet(
            protocol,
            _FakeOutputSet,
            "movies.sqlite",
        )

        self.assertIs(output.copiedFrom, logicalInput)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(output.streamState, _FakeOutputSet.STREAM_OPEN)

class _SizeOnlyMovieSet:
    def __init__(self, size):
        self.size = size
        self.sizeCalls = 0

    def getSize(self):
        self.sizeCalls += 1
        return self.size


class TestXmippMovieMaxShiftLogicalCompletion(unittest.TestCase):

    def testCheckNewOutputUsesLogicalInputSize(self):
        inputSet = _SizeOnlyMovieSet(6)
        pointer = _Pointer(inputSet)

        class _Harness:
            inputMovies = pointer
            inputMics = object()
            outMicName = None
            acceptedIds = []
            discardedIds = []
            isStreamClosed = False
            finished = False

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieMaxShift completion must not depend on "
                    "inputMovies.getFileName()."
                )

            def _getAllDoneIds(self):
                return [], 0, [], []

            def debug(self, message):
                pass

        protocol = _Harness()

        XmippProtMovieMaxShift._checkNewOutput(protocol)

        self.assertFalse(protocol.finished)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.sizeCalls, 1)

class _PublishMovie:
    def __init__(self, objId):
        self.objId = objId
        self.enabled = None

    def clone(self):
        return _PublishMovie(self.objId)

    def setEnabled(self, enabled):
        self.enabled = enabled


class _PublishInputMovies:
    def __init__(self, ids):
        self.ids = list(ids)
        self.loadCalls = 0
        self.closeCalls = 0
        self.getItemCalls = []

    def loadAllProperties(self):
        self.loadCalls += 1

    def getSize(self):
        return len(self.ids)

    def getItem(self, field, value):
        self.getItemCalls.append((field, value))
        return _PublishMovie(value)

    def close(self):
        self.closeCalls += 1


class _PublishOutputMovies:
    def __init__(self):
        self.items = []
        self.closeCalls = 0

    def getSize(self):
        return len(self.items)

    def getIdSet(self):
        return {item.objId for item in self.items}

    def append(self, item):
        self.items.append(item)

    def close(self):
        self.closeCalls += 1


class TestXmippMovieMaxShiftLogicalPublishing(unittest.TestCase):

    def testPublishingMoviesUsesLogicalInputSet(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        logicalInput = _PublishInputMovies([7])
        pointer = _Pointer(logicalInput)
        outputMovies = _PublishOutputMovies()

        class _Harness(XmippStreamingBase):
            inputMovies = pointer
            inputMics = None
            outMicName = None
            acceptedIds = [7]
            discardedIds = []
            isStreamClosed = False
            finished = False

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieMaxShift publishing must not depend on "
                    "inputMovies.getFileName()."
                )

            def setInputMics(self):
                self.inputMics = None
                self.outMicName = None

            def _getAllDoneIds(self):
                return [], 0, [], []

            def _loadOutputSet(self, SetClass, baseName):
                if baseName == "movies.sqlite":
                    return outputMovies
                return None

            def _updateOutputSet(
                    self,
                    outputName,
                    outputSet,
                    streamMode,
            ):
                pass

            def _markOutputIdsPersisted(
                    self,
                    outputName,
                    itemIds,
            ):
                pass

            def _defineTransformRelation(self, source, target):
                pass

            def _getFirstJoinStep(self):
                return None

            def _store(self):
                pass

            def debug(self, message):
                pass

            def info(self, message):
                pass

        protocol = _Harness()

        XmippProtMovieMaxShift._checkNewOutput(protocol)

        self.assertEqual(
            logicalInput.getItemCalls,
            [("id", 7)],
        )
        self.assertEqual(logicalInput.loadCalls, 1)
        self.assertEqual(logicalInput.closeCalls, 1)
        self.assertEqual(
            [item.objId for item in outputMovies.items],
            [7],
        )
        self.assertEqual(outputMovies.closeCalls, 1)

class _LateSiblingMic:
    def __init__(self, objId):
        self.objId = objId
        self.enabled = None

    def clone(self):
        return _LateSiblingMic(self.objId)

    def setEnabled(self, enabled):
        self.enabled = enabled


class _LateSiblingMicInput:
    def __init__(self, objId):
        self.objId = objId
        self.visible = False
        self.getItemCalls = []
        self.getIdSetCalls = 0
        self.closeCalls = 0

    def getIdSet(self):
        self.getIdSetCalls += 1
        raise AssertionError(
            "Late sibling discovery must not scan all input mic IDs."
        )

    def getItem(self, field, value):
        self.getItemCalls.append((field, value))

        if self.visible and value == self.objId:
            return _LateSiblingMic(value)

        return None

    def close(self):
        self.closeCalls += 1


class _TrackedPublishOutputSet:
    def __init__(self):
        self.items = []
        self.getIdSetCalls = 0
        self.closeCalls = 0

    def getSize(self):
        return len(self.items)

    def getIdSet(self):
        self.getIdSetCalls += 1
        return {item.objId for item in self.items}

    def append(self, item):
        self.items.append(item)

    def close(self):
        self.closeCalls += 1


class TestXmippMovieMaxShiftLateSiblingMicrograph(unittest.TestCase):

    def testLateSiblingMicrographIsRetriedIncrementally(self):
        from xmipp3.protocols.protocol_movie_max_shift import (
            OUTPUT_MICS,
            OUTPUT_MOVIES,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        logicalMovies = _PublishInputMovies([7])
        moviePointer = _Pointer(logicalMovies)

        inputMics = _LateSiblingMicInput(7)
        movieOutput = _TrackedPublishOutputSet()
        micOutput = _TrackedPublishOutputSet()

        class _Harness(XmippStreamingBase):
            outMicName = OUTPUT_MICS
            acceptedIds = [7]
            discardedIds = []
            isStreamClosed = False
            finished = False

            def _getAllDoneIds(self):
                accepted = sorted(
                    self._getKnownPersistedOutputIds(
                        OUTPUT_MOVIES,
                    )
                )
                return (
                    list(accepted),
                    len(accepted),
                    list(accepted),
                    [],
                )

            def _loadMicAssociatedInputSet(self):
                return self.inputMics

            def _loadOutputSet(self, SetClass, baseName):
                if baseName == "movies.sqlite":
                    return self.outputMovies

                if baseName == "micrographs.sqlite":
                    return self.outputMicrographs

                return None

            def _updateOutputSet(
                    self,
                    outputName,
                    outputSet,
                    streamMode,
            ):
                pass

            def _defineTransformRelation(self, source, target):
                pass

            def _getFirstJoinStep(self):
                return None

            def _store(self):
                pass

            def debug(self, message):
                pass

            def info(self, message):
                pass

        protocol = _Harness()
        protocol.inputMovies = moviePointer
        protocol.inputMics = inputMics
        protocol.outputMovies = movieOutput
        protocol.outputMicrographs = micOutput

        XmippProtMovieMaxShift._checkNewOutput(protocol)

        self.assertEqual(
            [item.objId for item in movieOutput.items],
            [7],
        )
        self.assertEqual(
            [item.objId for item in micOutput.items],
            [],
        )

        inputMics.visible = True

        XmippProtMovieMaxShift._checkNewOutput(protocol)

        self.assertEqual(
            [item.objId for item in movieOutput.items],
            [7],
        )
        self.assertEqual(
            [item.objId for item in micOutput.items],
            [7],
        )

        self.assertEqual(inputMics.getIdSetCalls, 0)
        self.assertLessEqual(movieOutput.getIdSetCalls, 1)
        self.assertLessEqual(micOutput.getIdSetCalls, 1)
        self.assertEqual(
            inputMics.getItemCalls,
            [("id", 7), ("id", 7)],
        )

class _ResumeMovieSet:
    def __init__(self, ids):
        self.ids = list(ids)
        self.uniqueCalls = []
        self.loadCalls = 0
        self.closeCalls = 0

    def loadAllProperties(self):
        self.loadCalls += 1

    def getUniqueValues(self, field, where=None):
        self.uniqueCalls.append((field, where))

        if where == "id > 0":
            return list(self.ids)

        if where == "id > 5":
            return []

        raise AssertionError(
            "Unexpected resume discovery query: %r" % (where,)
        )

    def isStreamClosed(self):
        return False

    def close(self):
        self.closeCalls += 1


class TestXmippMovieMaxShiftResumePersistence(unittest.TestCase):

    def testResumeSkipsPersistedMoviesAndSchedulesOnlyMissingOnes(self):
        from pyworkflow.protocol.constants import MODE_RESUME

        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        inputSet = _ResumeMovieSet([1, 2, 3, 4, 5])
        pointer = _Pointer(inputSet)

        acceptedOutput = _CountingOutputSet({1, 3})
        discardedOutput = _CountingOutputSet({2})

        class _Harness(XmippStreamingBase):
            inputMovies = pointer
            outputMovies = acceptedOutput
            outputMoviesDiscarded = discardedOutput

            insertedIds = []
            _lastInputId = 0
            isStreamClosed = False
            _originalRunMode = MODE_RESUME
            runMode = _RunMode()

            def _getFirstJoinStep(self):
                return None

            def _getAllDoneIds(self):
                return XmippProtMovieMaxShift._getAllDoneIds(
                    self,
                )

            def _insertNewMoviesSteps(self, newIds):
                newIds = list(newIds)
                self.batches.append(newIds)
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

            def info(self, message):
                self.infoMessages.append(message)

        protocol = _Harness()
        protocol.insertedIds = []
        protocol.batches = []
        protocol.updateCalls = 0
        protocol.infoMessages = []

        XmippProtMovieMaxShift._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4, 5]])
        self.assertEqual(
            set(protocol.insertedIds),
            {1, 2, 3, 4, 5},
        )
        self.assertEqual(protocol._lastInputId, 5)
        self.assertEqual(acceptedOutput.getIdSetCalls, 1)
        self.assertEqual(discardedOutput.getIdSetCalls, 1)

        XmippProtMovieMaxShift._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4, 5]])
        self.assertEqual(acceptedOutput.getIdSetCalls, 1)
        self.assertEqual(discardedOutput.getIdSetCalls, 1)
        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 0"),
                ("id", "id > 5"),
            ],
        )
