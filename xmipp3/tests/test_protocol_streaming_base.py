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

class TestXmippMovieDoseAnalysisStreamingBase(unittest.TestCase):

    def testMovieDoseAnalysisUsesSharedStreamingBase(self):
        from xmipp3.protocols.protocol_movie_dose_analysis import (
            XmippProtMovieDoseAnalysis,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        self.assertTrue(
            issubclass(
                XmippProtMovieDoseAnalysis,
                XmippStreamingBase,
            ),
            "MovieDoseAnalysis must reuse the shared Xmipp "
            "streaming helpers.",
        )

        self.assertTrue(
            issubclass(
                XmippProtMovieDoseAnalysis,
                ProtProcessMovies,
            ),
            "MovieDoseAnalysis must keep ProtProcessMovies behavior.",
        )

class _DoseStreamingMovieSet:
    def __init__(self):
        self.loadCalls = 0
        self.closeCalls = 0
        self.uniqueCalls = []
        self.sizeCalls = 0

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

    def getSize(self):
        self.sizeCalls += 1
        return 6

    def isStreamClosed(self):
        return False

    def close(self):
        self.closeCalls += 1


class TestXmippMovieDoseAnalysisStreamingInput(unittest.TestCase):

    def testMovieDoseAnalysisUsesIncrementalLogicalDiscovery(self):
        from xmipp3.protocols.protocol_movie_dose_analysis import (
            XmippProtMovieDoseAnalysis,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        inputSet = _DoseStreamingMovieSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer
            insertedIds = [1, 2, 3]
            _lastInputId = 3
            _inputSize = None
            isStreamClosed = False

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieDoseAnalysis streaming must not depend "
                    "on inputMovies.getFileName()."
                )

            def isContinued(self):
                return False

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

        from threading import Lock

        protocol = _Harness()
        protocol._lock = Lock()
        protocol.insertedIds = [1, 2, 3]
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtMovieDoseAnalysis._checkNewInput(protocol)
        XmippProtMovieDoseAnalysis._checkNewInput(protocol)

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
        self.assertEqual(
            protocol.insertedIds,
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(protocol._inputSize, 6)
        self.assertEqual(pointer.getCalls, 2)
        self.assertEqual(inputSet.loadCalls, 2)
        self.assertEqual(inputSet.closeCalls, 2)
        self.assertEqual(inputSet.sizeCalls, 2)
        self.assertEqual(protocol.updateCalls, 2)

class TestXmippMovieDoseAnalysisLogicalInitialization(unittest.TestCase):

    def testMovieDoseInitializeStepDoesNotRequireInputFilename(self):
        from xmipp3.protocols.protocol_movie_dose_analysis import (
            XmippProtMovieDoseAnalysis,
        )

        class _Acquisition:
            def getDosePerFrame(self):
                return 1.5

        class _Movie:
            def getAcquisition(self):
                return _Acquisition()

        class _LogicalMovieSet:
            def getSamplingRate(self):
                return 1.25

            def getFileName(self):
                raise AssertionError(
                    "MovieDoseAnalysis initialization must not "
                    "depend on a Set filename."
                )

            def isStreamClosed(self):
                return False

            def getFramesRange(self):
                return (1, 40, 1)

            def getFirstItem(self):
                return _Movie()

        class _Pointer:
            def __init__(self, value):
                self.value = value
                self.getCalls = 0

            def get(self):
                self.getCalls += 1
                return self.value

        class _Harness:
            inputMovies = _Pointer(_LogicalMovieSet())

            def isContinued(self):
                return False

        protocol = _Harness()

        XmippProtMovieDoseAnalysis.initializeStep(protocol)

        self.assertEqual(protocol.samplingRate, 1.25)
        self.assertEqual(protocol.framesRange, (1, 40, 1))
        self.assertEqual(protocol.dosePerFrame, 1.5)
        self.assertEqual(protocol._lastInputId, 0)
        self.assertEqual(protocol.insertedIds, [])
        self.assertEqual(protocol.processedIds, [])
        self.assertFalse(protocol.isStreamClosed)

class TestXmippMovieDoseAnalysisLogicalWorkerInput(unittest.TestCase):

    def testLoadMoviesByIdsUsesLogicalInputSet(self):
        from threading import Lock

        from xmipp3.protocols.protocol_movie_dose_analysis import (
            XmippProtMovieDoseAnalysis,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Movie:
            def __init__(self, objId):
                self.objId = objId

            def clone(self):
                return _Movie(self.objId)

            def getObjId(self):
                return self.objId

        class _LogicalMovieSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.getItemCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Movie(value)

            def close(self):
                self.closeCalls += 1

        class _Pointer:
            def __init__(self, value):
                self.value = value
                self.getCalls = 0

            def get(self):
                self.getCalls += 1
                return self.value

        logicalSet = _LogicalMovieSet()
        pointer = _Pointer(logicalSet)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieDoseAnalysis workers must not depend "
                    "on an input Set filename."
                )

        protocol = _Harness()
        protocol._lock = Lock()

        movies = XmippProtMovieDoseAnalysis._loadMoviesByIds(
            protocol,
            [3, 7],
        )

        self.assertEqual(sorted(movies), [3, 7])
        self.assertEqual(movies[3].getObjId(), 3)
        self.assertEqual(movies[7].getObjId(), 7)
        self.assertEqual(
            logicalSet.getItemCalls,
            [("id", 3), ("id", 7)],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(logicalSet.loadCalls, 1)
        self.assertEqual(logicalSet.closeCalls, 1)

class TestXmippMovieDoseAnalysisLogicalInputSize(unittest.TestCase):

    def testGetInputSizeUsesLogicalSetAndCachesResult(self):
        from threading import Lock

        from xmipp3.protocols.protocol_movie_dose_analysis import (
            XmippProtMovieDoseAnalysis,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _LogicalMovieSet:
            def __init__(self):
                self.loadCalls = 0
                self.sizeCalls = 0
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def getSize(self):
                self.sizeCalls += 1
                return 11

            def close(self):
                self.closeCalls += 1

        class _Pointer:
            def __init__(self, value):
                self.value = value
                self.getCalls = 0

            def get(self):
                self.getCalls += 1
                return self.value

        logicalSet = _LogicalMovieSet()
        pointer = _Pointer(logicalSet)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer
            _inputSize = None

            @property
            def movsFn(self):
                raise AssertionError(
                    "MovieDoseAnalysis input size must not depend "
                    "on an input Set filename."
                )

        protocol = _Harness()
        protocol._lock = Lock()

        first = XmippProtMovieDoseAnalysis._getInputSize(protocol)
        second = XmippProtMovieDoseAnalysis._getInputSize(protocol)

        self.assertEqual(first, 11)
        self.assertEqual(second, 11)
        self.assertEqual(protocol._inputSize, 11)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(logicalSet.loadCalls, 1)
        self.assertEqual(logicalSet.sizeCalls, 1)
        self.assertEqual(logicalSet.closeCalls, 1)

class TestXmippTiltAnalysisStreamingBase(unittest.TestCase):

    def testTiltAnalysisUsesSharedStreamingBase(self):
        from pwem.protocols import ProtMicrographs

        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        self.assertTrue(
            issubclass(
                XmippProtTiltAnalysis,
                XmippStreamingBase,
            ),
            "TiltAnalysis must reuse the shared Xmipp streaming helpers.",
        )

        self.assertTrue(
            issubclass(
                XmippProtTiltAnalysis,
                ProtMicrographs,
            ),
            "TiltAnalysis must keep ProtMicrographs behavior.",
        )

    def testTiltAnalysisInitializeStepDoesNotRequireInputFilename(self):
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _LogicalMicrographSet:
            def __init__(self):
                self.samplingCalls = 0
                self.streamCalls = 0

            def getSamplingRate(self):
                self.samplingCalls += 1
                return 2.5

            def getFileName(self):
                raise AssertionError(
                    "TiltAnalysis initialization must not depend "
                    "on a Set filename."
                )

            def isStreamClosed(self):
                self.streamCalls += 1
                return False

        inputSet = _LogicalMicrographSet()
        pointer = _Pointer(inputSet)

        class _Harness:
            inputMicrographs = pointer

            def getWindowSize(self):
                return 256

        protocol = _Harness()

        XmippProtTiltAnalysis.initializeStep(protocol)

        self.assertEqual(protocol.samplingRate, 2.5)
        self.assertEqual(protocol.stats, {})
        self.assertEqual(protocol.insertedIds, [])
        self.assertEqual(protocol.processedIds, [])
        self.assertEqual(protocol._lastInputId, 0)
        self.assertFalse(protocol.isStreamClosed)
        self.assertEqual(protocol.windowSize, 256)
        self.assertFalse(hasattr(protocol, "micsFn"))
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.samplingCalls, 1)
        self.assertEqual(inputSet.streamCalls, 1)

    def testTiltAnalysisUsesIncrementalLogicalInputDiscovery(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _StreamingMicrographSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.uniqueCalls = []
                self.getIdSetCalls = 0

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

            def getIdSet(self):
                self.getIdSetCalls += 1
                raise AssertionError(
                    "TiltAnalysis streaming discovery must not "
                    "scan the full input ID set."
                )

            def isStreamClosed(self):
                return False

            def close(self):
                self.closeCalls += 1

        inputSet = _StreamingMicrographSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            insertedIds = [1, 2, 3]
            _lastInputId = 3
            isStreamClosed = False

            @property
            def micsFn(self):
                raise AssertionError(
                    "TiltAnalysis streaming must not depend on "
                    "inputMicrographs.getFileName()."
                )

            def isContinued(self):
                return False

            def _getFirstJoinStep(self):
                return None

            def _insertNewMicrographSteps(self, newIds):
                newIds = list(newIds)
                self.batches.append(newIds)
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

            def info(self, message):
                pass

        protocol = _Harness()
        protocol.insertedIds = [1, 2, 3]
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtTiltAnalysis._checkNewInput(protocol)
        XmippProtTiltAnalysis._checkNewInput(protocol)

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
        self.assertEqual(inputSet.getIdSetCalls, 0)
        self.assertEqual(protocol._lastInputId, 6)
        self.assertEqual(
            protocol.insertedIds,
            [1, 2, 3, 4, 5, 6],
        )
        self.assertFalse(protocol.isStreamClosed)
        self.assertEqual(pointer.getCalls, 2)
        self.assertEqual(inputSet.loadCalls, 2)
        self.assertEqual(inputSet.closeCalls, 2)
        self.assertEqual(protocol.updateCalls, 2)

    def testTiltAnalysisRestoresPersistedOutputIdsOnlyOnce(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        accepted = _CountingOutputSet({1, 3})
        discarded = _CountingOutputSet({2, 4})

        class _Harness(XmippStreamingBase):
            outputMicrographs = accepted
            discardedMicrographs = discarded

        protocol = _Harness()

        first = XmippProtTiltAnalysis._getAllDoneIds(protocol)
        second = XmippProtTiltAnalysis._getAllDoneIds(protocol)

        self.assertEqual(set(first[0]), {1, 2, 3, 4})
        self.assertEqual(first[1], 4)
        self.assertEqual(set(first[2]), {1, 3})
        self.assertEqual(set(first[3]), {2, 4})
        self.assertEqual(second, first)

        self.assertEqual(
            accepted.getIdSetCalls,
            1,
            "Accepted persisted IDs must be restored once and cached.",
        )
        self.assertEqual(
            discarded.getIdSetCalls,
            1,
            "Discarded persisted IDs must be restored once and cached.",
        )

    def testTiltAnalysisPublishingUpdatesPersistedOutputCache(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _Value:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class _Micrograph:
            def __init__(self, objId):
                self.objId = objId

            def clone(self):
                return _Micrograph(self.objId)

            def getObjId(self):
                return self.objId

        class _InputSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.getItemCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def getSize(self):
                return 2

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Micrograph(value)

            def close(self):
                self.closeCalls += 1

        class _OutputSet:
            def __init__(self):
                self.items = []
                self.getIdSetCalls = 0

            def getSize(self):
                return len(self.items)

            def getIdSet(self):
                self.getIdSetCalls += 1
                return {item.getObjId() for item in self.items}

            def append(self, item):
                self.items.append(item)

        inputSet = _InputSet()
        pointer = _Pointer(inputSet)
        acceptedOutput = _OutputSet()
        discardedOutput = _OutputSet()

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            outputMicrographs = acceptedOutput
            discardedMicrographs = discardedOutput
            processedIds = [7, 8]
            isStreamClosed = False
            micsFn = "unused"
            meanCorr_threshold = _Value(0.5)
            stdCorr_threshold = _Value(0.1)
            stats = {
                7: {
                    "mean": 0.8,
                    "std": 0.05,
                    "min": 0.1,
                    "max": 0.9,
                },
                8: {
                    "mean": 0.2,
                    "std": 0.05,
                    "min": 0.1,
                    "max": 0.3,
                },
            }

            def _loadInputSet(self, micsFn):
                return inputSet

            def _getAllDoneIds(self):
                return XmippProtTiltAnalysis._getAllDoneIds(self)

            def getPSDs(self, micFolder, ID):
                return XmippProtTiltAnalysis.getPSDs(micFolder, ID)

            def _appendNewMicrographs(self, micSet, micrographs):
                return XmippProtTiltAnalysis._appendNewMicrographs(
                    self,
                    micSet,
                    micrographs,
                )

            def _loadOutputSet(self, SetClass, baseName):
                if baseName == "micrograph.sqlite":
                    return acceptedOutput

                if baseName == "micrographDISCARDED.sqlite":
                    return discardedOutput

                raise AssertionError(
                    "Unexpected output baseName: %s" % baseName
                )

            def _updateOutputSet(
                    self,
                    outputName,
                    outputSet,
                    streamMode,
            ):
                pass

            def _getExtraPath(self):
                return "/tmp"

            def _getFirstJoinStep(self):
                return None

            def _store(self):
                pass

        protocol = _Harness()
        protocol.processedIds = [7, 8]
        protocol.stats = dict(_Harness.stats)

        before = XmippProtTiltAnalysis._getAllDoneIds(protocol)
        self.assertEqual(before[0], [])

        XmippProtTiltAnalysis._checkNewOutput(protocol)

        after = XmippProtTiltAnalysis._getAllDoneIds(protocol)

        self.assertEqual(after[0], [7, 8])
        self.assertEqual(after[1], 2)
        self.assertEqual(after[2], [7])
        self.assertEqual(after[3], [8])
        self.assertEqual(
            [mic.getObjId() for mic in acceptedOutput.items],
            [7],
        )
        self.assertEqual(
            [mic.getObjId() for mic in discardedOutput.items],
            [8],
        )
        self.assertEqual(acceptedOutput.getIdSetCalls, 1)
        self.assertEqual(discardedOutput.getIdSetCalls, 1)

    def testTiltAnalysisCheckNewOutputUsesLogicalInputSize(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _LogicalMicrographSet:
            def __init__(self):
                self.loadCalls = 0
                self.sizeCalls = 0
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def getSize(self):
                self.sizeCalls += 1
                return 6

            def close(self):
                self.closeCalls += 1

        inputSet = _LogicalMicrographSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            processedIds = []
            isStreamClosed = False
            finished = False

            @property
            def micsFn(self):
                raise AssertionError(
                    "TiltAnalysis completion must not depend on "
                    "inputMicrographs.getFileName()."
                )

            def _getAllDoneIds(self):
                return [], 0, [], []

            def _getFirstJoinStep(self):
                return None

            def _store(self):
                pass

        protocol = _Harness()
        protocol.processedIds = []

        XmippProtTiltAnalysis._checkNewOutput(protocol)

        self.assertFalse(protocol.finished)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.sizeCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testTiltAnalysisPublishingLoadsMicrographsFromLogicalInput(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _Value:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class _Micrograph:
            def __init__(self, objId):
                self.objId = objId

            def clone(self):
                return _Micrograph(self.objId)

            def getObjId(self):
                return self.objId

        class _LogicalInputSet:
            def __init__(self):
                self.loadCalls = 0
                self.sizeCalls = 0
                self.getItemCalls = []
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def getSize(self):
                self.sizeCalls += 1
                return 1

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Micrograph(value)

            def close(self):
                self.closeCalls += 1

        class _OutputSet:
            def __init__(self):
                self.items = []

            def getSize(self):
                return len(self.items)

            def getIdSet(self):
                return {item.getObjId() for item in self.items}

            def append(self, item):
                self.items.append(item)

        inputSet = _LogicalInputSet()
        pointer = _Pointer(inputSet)
        acceptedOutput = _OutputSet()

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            outputMicrographs = acceptedOutput
            processedIds = [7]
            isStreamClosed = False
            meanCorr_threshold = _Value(0.5)
            stdCorr_threshold = _Value(0.1)
            stats = {
                7: {
                    "mean": 0.8,
                    "std": 0.05,
                    "min": 0.1,
                    "max": 0.9,
                },
            }

            @property
            def micsFn(self):
                raise AssertionError(
                    "TiltAnalysis publishing must not depend on "
                    "inputMicrographs.getFileName()."
                )

            def _getAllDoneIds(self):
                return [], 0, [], []

            def getPSDs(self, micFolder, ID):
                return XmippProtTiltAnalysis.getPSDs(
                    micFolder,
                    ID,
                )

            def _appendNewMicrographs(self, micSet, micrographs):
                return XmippProtTiltAnalysis._appendNewMicrographs(
                    self,
                    micSet,
                    micrographs,
                )

            def _loadOutputSet(self, SetClass, baseName):
                if baseName == "micrograph.sqlite":
                    return acceptedOutput

                raise AssertionError(
                    "Unexpected output baseName: %s" % baseName
                )

            def _updateOutputSet(
                    self,
                    outputName,
                    outputSet,
                    streamMode,
            ):
                pass

            def _getExtraPath(self):
                return "/tmp"

            def _getFirstJoinStep(self):
                return None

            def _store(self):
                pass

        protocol = _Harness()
        protocol.processedIds = [7]
        protocol.stats = dict(_Harness.stats)

        XmippProtTiltAnalysis._checkNewOutput(protocol)

        self.assertEqual(
            inputSet.getItemCalls,
            [("id", 7)],
        )
        self.assertEqual(
            [mic.getObjId() for mic in acceptedOutput.items],
            [7],
        )
        self.assertEqual(pointer.getCalls, 2)
        self.assertEqual(inputSet.loadCalls, 2)
        self.assertEqual(inputSet.sizeCalls, 1)
        self.assertEqual(inputSet.closeCalls, 2)

    def testTiltAnalysisWorkerLoadsBatchFromLogicalInput(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _Micrograph:
            def __init__(self, objId):
                self.objId = objId

            def clone(self):
                return _Micrograph(self.objId)

            def getObjId(self):
                return self.objId

        class _LogicalMicrographSet:
            def __init__(self):
                self.loadCalls = 0
                self.getItemCalls = []
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Micrograph(value)

            def close(self):
                self.closeCalls += 1

        inputSet = _LogicalMicrographSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer

            @property
            def micsFn(self):
                raise AssertionError(
                    "TiltAnalysis workers must not depend on "
                    "inputMicrographs.getFileName()."
                )

            def _processMicrograph(self, micrograph):
                self.processed.append(micrograph.getObjId())

        protocol = _Harness()
        protocol.processed = []

        XmippProtTiltAnalysis.processMicrographListStep(
            protocol,
            [4, 7],
        )

        self.assertEqual(
            inputSet.getItemCalls,
            [("id", 4), ("id", 7)],
        )
        self.assertEqual(protocol.processed, [4, 7])
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testTiltAnalysisResumeSkipsPersistedMicrographs(self):
        from pyworkflow.protocol.constants import MODE_RESUME

        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _ResumeMicrographSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.uniqueCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))

                if where == "id > 0":
                    return [1, 2, 3, 4, 5]

                if where == "id > 5":
                    return []

                raise AssertionError(
                    "Unexpected resume discovery query: %r" % (where,)
                )

            def isStreamClosed(self):
                return False

            def close(self):
                self.closeCalls += 1

        class _RunMode:
            def get(self):
                return 0

        inputSet = _ResumeMicrographSet()
        pointer = _Pointer(inputSet)

        acceptedOutput = _CountingOutputSet({1, 3})
        discardedOutput = _CountingOutputSet({2})

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            outputMicrographs = acceptedOutput
            discardedMicrographs = discardedOutput

            insertedIds = []
            _lastInputId = 0
            isStreamClosed = False
            _originalRunMode = MODE_RESUME
            runMode = _RunMode()

            def isContinued(self):
                return True

            def _getAllDoneIds(self):
                return XmippProtTiltAnalysis._getAllDoneIds(
                    self,
                )

            def _getFirstJoinStep(self):
                return None

            def _insertNewMicrographSteps(self, newIds):
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

        XmippProtTiltAnalysis._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4, 5]])
        self.assertEqual(
            set(protocol.insertedIds),
            {1, 2, 3, 4, 5},
        )
        self.assertEqual(protocol._lastInputId, 5)
        self.assertEqual(acceptedOutput.getIdSetCalls, 1)
        self.assertEqual(discardedOutput.getIdSetCalls, 1)

        XmippProtTiltAnalysis._checkNewInput(protocol)

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
        self.assertEqual(inputSet.loadCalls, 2)
        self.assertEqual(inputSet.closeCalls, 2)

    def testTiltAnalysisRecoversLateVisibleIdsAfterStreamCloses(self):
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )
        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _LateVisibleMicrographSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.sizeCalls = 0
                self.uniqueCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))

                if where == "id > 0":
                    return [9, 10]

                if where is None:
                    return list(range(1, 11))

                raise AssertionError(
                    "Unexpected discovery query: %r" % (where,)
                )

            def getSize(self):
                self.sizeCalls += 1
                return 10

            def isStreamClosed(self):
                return True

            def close(self):
                self.closeCalls += 1

        inputSet = _LateVisibleMicrographSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            insertedIds = []
            _lastInputId = 0
            isStreamClosed = False

            def isContinued(self):
                return False

            def _getFirstJoinStep(self):
                return None

            def _insertNewMicrographSteps(self, newIds):
                newIds = list(newIds)
                self.batches.append(newIds)
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

            def info(self, message):
                pass

        protocol = _Harness()
        protocol.insertedIds = []
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtTiltAnalysis._checkNewInput(protocol)

        self.assertEqual(
            protocol.batches,
            [list(range(1, 11))],
        )
        self.assertEqual(
            protocol.insertedIds,
            list(range(1, 11)),
        )
        self.assertEqual(protocol._lastInputId, 10)
        self.assertTrue(protocol.isStreamClosed)
        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 0"),
                ("id", None),
            ],
        )
        self.assertEqual(inputSet.sizeCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)
        self.assertEqual(protocol.updateCalls, 1)

    def testTiltAnalysisCreatesOutputThroughProtocolFactory(self):
        from pwem.objects import SetOfMicrographs

        from xmipp3.protocols.protocol_tilt_analysis import (
            XmippProtTiltAnalysis,
        )

        class _LogicalInput:
            pass

        class _OutputSet:
            STREAM_OPEN = 1

            def __init__(self):
                self.streamState = None
                self.copiedFrom = None
                self.appendEnabled = False

            def setStreamState(self, state):
                self.streamState = state

            def copyInfo(self, inputSet):
                self.copiedFrom = inputSet

            def enableAppend(self):
                self.appendEnabled = True

        logicalInput = _LogicalInput()
        pointer = _Pointer(logicalInput)
        createdOutput = _OutputSet()

        class _Harness:
            inputMicrographs = pointer

            def __init__(self):
                self.factoryCalls = []

            def _createSetOfMicrographs(self, suffix=""):
                self.factoryCalls.append(suffix)
                return createdOutput

            def _getPath(self, *args, **kwargs):
                raise AssertionError(
                    "TiltAnalysis output creation must not depend "
                    "on an output SQLite filename."
                )

        protocol = _Harness()

        output = XmippProtTiltAnalysis._loadOutputSet(
            protocol,
            SetOfMicrographs,
            "micrograph.sqlite",
        )

        self.assertIs(output, createdOutput)
        self.assertEqual(protocol.factoryCalls, [""])
        self.assertEqual(
            createdOutput.streamState,
            createdOutput.STREAM_OPEN,
        )
        self.assertIs(createdOutput.copiedFrom, logicalInput)
        self.assertEqual(pointer.getCalls, 1)

class TestXmippCTFConsensusStreamingBase(unittest.TestCase):

    def testCTFConsensusUsesSharedStreamingBase(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        self.assertTrue(
            issubclass(
                XmippProtCTFConsensus,
                XmippStreamingBase,
            ),
            "CTFConsensus must reuse the shared Xmipp streaming helpers.",
        )

    def testCTFConsensusInitializeParamsDoesNotRequireInputFilename(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )

        class _LogicalCtfSet:
            def getFileName(self):
                raise AssertionError(
                    "CTFConsensus initialization must not depend "
                    "on inputCTF.getFileName()."
                )

        logicalInput = _LogicalCtfSet()
        pointer = _Pointer(logicalInput)

        class _Harness:
            inputCTF = pointer

            def initializeRejDict(self):
                self.initializeRejDictCalls += 1

            def setSecondaryAttributes(self):
                self.setSecondaryAttributesCalls += 1

        protocol = _Harness()
        protocol.initializeRejDictCalls = 0
        protocol.setSecondaryAttributesCalls = 0

        XmippProtCTFConsensus.initializeParams(protocol)

        self.assertFalse(protocol.finished)
        self.assertFalse(protocol.isStreamClosed)
        self.assertEqual(protocol.insertedIds, [])
        self.assertEqual(protocol.acceptedIds, {})
        self.assertEqual(protocol.discardedIds, {})
        self.assertEqual(protocol._lastInputId, 0)
        self.assertFalse(hasattr(protocol, "ctfFn1"))
        self.assertEqual(protocol.initializeRejDictCalls, 1)
        self.assertEqual(protocol.setSecondaryAttributesCalls, 1)
        self.assertEqual(pointer.getCalls, 0)

    def testCTFConsensusUsesIncrementalLogicalInputDiscovery(self):
        from pyworkflow.protocol.constants import MODE_RESTART

        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _LogicalCtfSet:
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
                    "Unexpected incremental discovery query: %r" % (where,)
                )

            def getIdSet(self):
                raise AssertionError(
                    "CTFConsensus streaming discovery must not scan getIdSet()."
                )

            def isStreamClosed(self):
                return False

            def close(self):
                self.closeCalls += 1

        class _RunMode:
            def get(self):
                return MODE_RESTART

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            calculateConsensus = False
            insertedIds = [1, 2, 3]
            _lastInputId = 3
            isStreamClosed = False
            _originalRunMode = MODE_RESTART
            runMode = _RunMode()

            @property
            def ctfFn1(self):
                raise AssertionError(
                    "CTFConsensus streaming discovery must not depend "
                    "on inputCTF.getFileName()."
                )

            def _getFirstJoinStep(self):
                return None

            def _insertNewCtfsSteps(self, newIds):
                newIds = list(newIds)
                self.batches.append(newIds)
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

            def info(self, message):
                pass

        protocol = _Harness()
        protocol.insertedIds = [1, 2, 3]
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtCTFConsensus._checkNewInput(protocol)
        XmippProtCTFConsensus._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4, 5], [6]])
        self.assertEqual(protocol.insertedIds, [1, 2, 3, 4, 5, 6])
        self.assertEqual(protocol._lastInputId, 6)
        self.assertFalse(protocol.isStreamClosed)
        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 3"),
                ("id", "id > 5"),
            ],
        )
        self.assertEqual(pointer.getCalls, 2)
        self.assertEqual(inputSet.loadCalls, 2)
        self.assertEqual(inputSet.closeCalls, 2)
        self.assertEqual(protocol.updateCalls, 2)

    def testCTFConsensusMatchesIncrementalDualInputsAcrossPolls(self):
        from pyworkflow.protocol.constants import MODE_RESTART

        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _LogicalCtfSet:
            def __init__(self, responses):
                self.responses = dict(responses)
                self.loadCalls = 0
                self.closeCalls = 0
                self.uniqueCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))

                if where not in self.responses:
                    raise AssertionError(
                        "Unexpected incremental discovery query: %r"
                        % (where,)
                    )

                return list(self.responses[where])

            def getIdSet(self):
                raise AssertionError(
                    "Dual-input consensus discovery must not scan getIdSet()."
                )

            def isStreamClosed(self):
                return False

            def close(self):
                self.closeCalls += 1

        class _RunMode:
            def get(self):
                return MODE_RESTART

        inputSet1 = _LogicalCtfSet({
            "id > 3": [4, 5],
            "id > 5": [6],
        })
        inputSet2 = _LogicalCtfSet({
            "id > 3": [4],
            "id > 4": [5, 6],
        })

        pointer1 = _Pointer(inputSet1)
        pointer2 = _Pointer(inputSet2)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer1
            inputCTF2 = pointer2
            calculateConsensus = True
            insertedIds = [1, 2, 3]
            _lastInputId1 = 3
            _lastInputId2 = 3
            _pendingInputIds1 = set()
            _pendingInputIds2 = set()
            isStreamClosed = False
            _originalRunMode = MODE_RESTART
            runMode = _RunMode()

            @property
            def ctfFn1(self):
                raise AssertionError(
                    "Dual-input consensus must not depend on ctfFn1."
                )

            @property
            def ctfFn2(self):
                raise AssertionError(
                    "Dual-input consensus must not depend on ctfFn2."
                )

            def _getFirstJoinStep(self):
                return None

            def _insertNewCtfsSteps(self, newIds):
                newIds = list(newIds)
                self.batches.append(newIds)
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

            def info(self, message):
                pass

        protocol = _Harness()
        protocol.insertedIds = [1, 2, 3]
        protocol._pendingInputIds1 = set()
        protocol._pendingInputIds2 = set()
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtCTFConsensus._checkNewInput(protocol)
        XmippProtCTFConsensus._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4], [5, 6]])
        self.assertEqual(protocol.insertedIds, [1, 2, 3, 4, 5, 6])
        self.assertEqual(protocol._lastInputId1, 6)
        self.assertEqual(protocol._lastInputId2, 6)
        self.assertEqual(protocol._pendingInputIds1, set())
        self.assertEqual(protocol._pendingInputIds2, set())
        self.assertFalse(protocol.isStreamClosed)

        self.assertEqual(
            inputSet1.uniqueCalls,
            [
                ("id", "id > 3"),
                ("id", "id > 5"),
            ],
        )
        self.assertEqual(
            inputSet2.uniqueCalls,
            [
                ("id", "id > 3"),
                ("id", "id > 4"),
            ],
        )
        self.assertEqual(pointer1.getCalls, 2)
        self.assertEqual(pointer2.getCalls, 2)
        self.assertEqual(inputSet1.loadCalls, 2)
        self.assertEqual(inputSet2.loadCalls, 2)
        self.assertEqual(inputSet1.closeCalls, 2)
        self.assertEqual(inputSet2.closeCalls, 2)
        self.assertEqual(protocol.updateCalls, 2)

    def testCTFConsensusRestoresPersistedOutputIdsOnlyOnce(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        acceptedOutput = _CountingOutputSet({1, 3})
        discardedOutput = _CountingOutputSet({2, 4})

        class _Harness(XmippStreamingBase):
            outputCTF = acceptedOutput
            outputCTFDiscarded = discardedOutput

        protocol = _Harness()

        first = XmippProtCTFConsensus._getAllDoneIds(protocol)
        second = XmippProtCTFConsensus._getAllDoneIds(protocol)

        self.assertEqual(
            first,
            ([1, 3, 2, 4], 4, [1, 3], [2, 4]),
        )
        self.assertEqual(second, first)
        self.assertEqual(acceptedOutput.getIdSetCalls, 1)
        self.assertEqual(discardedOutput.getIdSetCalls, 1)

    def testCTFConsensusPublishingUpdatesPersistedOutputCache(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            OUTPUT_CTF,
            OUTPUT_CTF_DISCARDED,
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _InputCtfSet:
            def getIdSet(self):
                return {1, 2, 8, 9}

        class _PublishedCtfSet:
            def __init__(self, ids):
                self.ids = set(ids)
                self.getIdSetCalls = 0
                self.closeCalls = 0
                self.micrographs = None

            def getIdSet(self):
                self.getIdSetCalls += 1
                return set(self.ids)

            def setMicrographs(self, micrographs):
                self.micrographs = micrographs

            def close(self):
                self.closeCalls += 1

        class _PublishedMicSet:
            def __init__(self):
                self.closeCalls = 0

            def close(self):
                self.closeCalls += 1

        acceptedOutput = _PublishedCtfSet({9})
        discardedOutput = _PublishedCtfSet({8})
        acceptedMics = _PublishedMicSet()
        discardedMics = _PublishedMicSet()

        class _Harness(XmippStreamingBase):
            calculateConsensus = False
            isStreamClosed = False
            acceptedIds = {1: 'T'}
            discardedIds = {2: 'F'}
            ctfFn1 = "legacy-unused-by-this-test.sqlite"
            outputCTF = acceptedOutput
            outputCTFDiscarded = discardedOutput

            def _getAllDoneIds(self):
                return XmippProtCTFConsensus._getAllDoneIds(
                    self,
                )

            def _loadInputCtfSet(self, ctfFn):
                return _InputCtfSet()

            def _loadOutputSet(self, SetClass, baseName):
                outputs = {
                    "ctfs.sqlite": acceptedOutput,
                    "micrographs.sqlite": acceptedMics,
                    "ctfsDiscarded.sqlite": discardedOutput,
                    "micrographsDiscarded.sqlite": discardedMics,
                }
                return outputs[baseName]

            def fillOutput(self, ctfSet, micSet, newDone, label):
                pass

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                self.updatedOutputs.append(outputName)

            def _defineTransformRelation(self, *args):
                pass

            def _defineCtfRelation(self, *args):
                pass

            def _getFirstJoinStep(self):
                return None

            def _store(self):
                pass

            def _markOutputIdsPersisted(self, outputName, itemIds):
                self.markCalls.append(
                    (outputName, list(itemIds))
                )
                return super()._markOutputIdsPersisted(
                    outputName,
                    itemIds,
                )

        protocol = _Harness()
        protocol.acceptedIds = {1: 'T'}
        protocol.discardedIds = {2: 'F'}
        protocol.updatedOutputs = []
        protocol.markCalls = []

        XmippProtCTFConsensus._checkNewOutput(protocol)

        self.assertEqual(
            protocol.markCalls,
            [
                (OUTPUT_CTF, [1]),
                (OUTPUT_CTF_DISCARDED, [2]),
            ],
        )

        self.assertEqual(
            XmippProtCTFConsensus._getAllDoneIds(protocol),
            ([1, 9, 2, 8], 4, [1, 9], [2, 8]),
        )
        self.assertEqual(acceptedOutput.getIdSetCalls, 1)
        self.assertEqual(discardedOutput.getIdSetCalls, 1)

    def testCTFConsensusCompletionDoesNotReloadInputSets(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        acceptedOutput = _CountingOutputSet({1, 2})

        class _Harness(XmippStreamingBase):
            calculateConsensus = False
            isStreamClosed = True
            insertedIds = [1, 2]
            acceptedIds = {1: 'T', 2: 'T'}
            discardedIds = {}
            outputCTF = acceptedOutput

            @property
            def ctfFn1(self):
                raise AssertionError(
                    "CTFConsensus completion must not depend on ctfFn1."
                )

            def _loadInputCtfSet(self, ctfFn):
                raise AssertionError(
                    "CTFConsensus completion must not reopen the input Set."
                )

            def _getAllDoneIds(self):
                return XmippProtCTFConsensus._getAllDoneIds(
                    self,
                )

            def _getFirstJoinStep(self):
                return None

            def _store(self):
                self.storeCalls += 1

        protocol = _Harness()
        protocol.insertedIds = [1, 2]
        protocol.acceptedIds = {1: 'T', 2: 'T'}
        protocol.discardedIds = {}
        protocol.storeCalls = 0

        XmippProtCTFConsensus._checkNewOutput(protocol)

        self.assertTrue(protocol.finished)
        self.assertEqual(protocol.storeCalls, 1)
        self.assertEqual(acceptedOutput.getIdSetCalls, 1)

    def testCTFConsensusWorkerLoadsBatchFromLogicalInput(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Ctf:
            def __init__(self, objId):
                self.objId = objId

            def clone(self):
                return _Ctf(self.objId)

            def getDefocusU(self):
                return 10000

            def getDefocusV(self):
                return 9900

            def getResolution(self):
                return 3

            def isEnabled(self):
                return True

        class _LogicalCtfSet:
            def __init__(self):
                self.loadCalls = 0
                self.getItemCalls = []
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Ctf(value)

            def close(self):
                self.closeCalls += 1

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            calculateConsensus = False
            useCritXmipp = False
            acceptedIds = {}
            discardedIds = {}
            discDict = {
                'defocus': 0,
                'astigmatism': 0,
                'astigmatismPer': 0,
                'singleResolution': 0,
                '_xmipp_ctfCritFirstZero': 0,
                '_xmipp_ctfCritfirstZeroRatio': 0,
                '_xmipp_ctfCritCorr13': 0,
                '_xmipp_ctfCritIceness': 0,
                '_xmipp_ctfCritCtfMargin': 0,
                '_xmipp_ctfCritNonAstigmaticValidty': 0,
                'consensusResolution': 0,
            }

            @property
            def ctfFn1(self):
                raise AssertionError(
                    "CTFConsensus workers must not depend on ctfFn1."
                )

            def _getDefociValues(self):
                return 0, 50000

            def _getMaxAstisgmatism(self):
                return 5000

            def _getMaxAstigmatismPer(self):
                return 1

            def _getMinResol(self):
                return 10

            def _getCtfResol(self, ctf):
                return ctf.getResolution()

        protocol = _Harness()
        protocol.acceptedIds = {}
        protocol.discardedIds = {}
        protocol.discDict = dict(_Harness.discDict)

        XmippProtCTFConsensus.selectCtfStep(
            protocol,
            [4, 7],
        )

        self.assertEqual(
            inputSet.getItemCalls,
            [
                ("id", 4),
                ("id", 7),
            ],
        )
        self.assertEqual(
            protocol.acceptedIds,
            {
                4: 'T',
                7: 'T',
            },
        )
        self.assertEqual(protocol.discardedIds, {})
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testCTFConsensusFillOutputLoadsLogicalInput(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            ACCEPTED,
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Micrograph:
            def __init__(self, objId):
                self.objId = objId
                self.enabled = None

            def clone(self):
                return _Micrograph(self.objId)

            def setEnabled(self, enabled):
                self.enabled = enabled

        class _Ctf:
            def __init__(self, objId):
                self.objId = objId
                self.enabled = None
                self.micrograph = _Micrograph(objId)

            def clone(self):
                return _Ctf(self.objId)

            def getMicrograph(self):
                return self.micrograph

            def setEnabled(self, enabled):
                self.enabled = enabled

            def getDefocusU(self):
                return 10000

            def getDefocusV(self):
                return 9900

        class _LogicalCtfSet:
            def __init__(self):
                self.loadCalls = 0
                self.getItemCalls = []
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Ctf(value)

            def close(self):
                self.closeCalls += 1

        class _OutputSet:
            def __init__(self):
                self.items = []

            def append(self, item):
                self.items.append(item)

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)
        outputCtfs = _OutputSet()
        outputMics = _OutputSet()

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            calculateConsensus = False

            @property
            def ctfFn1(self):
                raise AssertionError(
                    "CTFConsensus output publishing must not depend on ctfFn1."
                )

            def _getEnable(self, ctfId, label):
                return True

        protocol = _Harness()

        XmippProtCTFConsensus.fillOutput(
            protocol,
            outputCtfs,
            outputMics,
            [4, 7],
            ACCEPTED,
        )

        self.assertEqual(
            inputSet.getItemCalls,
            [
                ("id", 4),
                ("id", 7),
            ],
        )
        self.assertEqual(
            [ctf.objId for ctf in outputCtfs.items],
            [4, 7],
        )
        self.assertEqual(
            [mic.objId for mic in outputMics.items],
            [4, 7],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testCTFConsensusOutputCreationUsesProtocolFactories(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_ctf_consensus as ctf_consensus
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _InputMicrographs:
            pass

        class _InputCtfSet:
            def __init__(self, micrographs):
                self.micrographs = micrographs

            def getMicrographs(self):
                return self.micrographs

        class _FactorySet:
            STREAM_OPEN = "open"

            def _initialize(self):
                self.streamStates = []
                self.enableAppendCalls = 0
                self.copiedInfo = None
                self.micrographs = None

            def enableAppend(self):
                self.enableAppendCalls += 1

            def setStreamState(self, state):
                self.streamStates.append(state)

            def copyInfo(self, other):
                self.copiedInfo = other

            def setMicrographs(self, micrographs):
                self.micrographs = micrographs

        class _FakeSetOfCTF(_FactorySet):
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "CTFConsensus must not construct SetOfCTF(filename=...)."
                )

        class _FakeSetOfMicrographs(_FactorySet):
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "CTFConsensus must not construct "
                    "SetOfMicrographs(filename=...)."
                )

        inputMicrographs = _InputMicrographs()
        inputCtfSet = _InputCtfSet(inputMicrographs)
        pointer = _Pointer(inputCtfSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer

            def _getPath(self, *args):
                raise AssertionError(
                    "CTFConsensus output creation must not depend on "
                    "legacy sqlite paths."
                )

            def _newFactorySet(self, SetClass):
                outputSet = object.__new__(SetClass)
                outputSet._initialize()
                return outputSet

            def _createSetOfCTF(self, suffix=''):
                self.factoryCalls.append(("ctf", suffix))
                outputSet = self._newFactorySet(_FakeSetOfCTF)
                self.createdSets.append(outputSet)
                return outputSet

            def _createSetOfMicrographs(self, suffix=''):
                self.factoryCalls.append(("micrographs", suffix))
                outputSet = self._newFactorySet(
                    _FakeSetOfMicrographs,
                )
                self.createdSets.append(outputSet)
                return outputSet

        protocol = _Harness()
        protocol.factoryCalls = []
        protocol.createdSets = []

        with patch.object(
            ctf_consensus,
            "SetOfCTF",
            _FakeSetOfCTF,
        ), patch.object(
            ctf_consensus,
            "SetOfMicrographs",
            _FakeSetOfMicrographs,
        ):
            acceptedCtf = XmippProtCTFConsensus._loadOutputSet(
                protocol,
                _FakeSetOfCTF,
                "ctfs.sqlite",
            )
            acceptedMics = XmippProtCTFConsensus._loadOutputSet(
                protocol,
                _FakeSetOfMicrographs,
                "micrographs.sqlite",
            )
            discardedCtf = XmippProtCTFConsensus._loadOutputSet(
                protocol,
                _FakeSetOfCTF,
                "ctfsDiscarded.sqlite",
            )
            discardedMics = XmippProtCTFConsensus._loadOutputSet(
                protocol,
                _FakeSetOfMicrographs,
                "micrographsDiscarded.sqlite",
            )

        self.assertEqual(
            protocol.factoryCalls,
            [
                ("ctf", ""),
                ("micrographs", ""),
                ("ctf", "Discarded"),
                ("micrographs", "Discarded"),
            ],
        )

        self.assertIs(acceptedCtf.micrographs, inputMicrographs)
        self.assertIs(discardedCtf.micrographs, inputMicrographs)
        self.assertIs(acceptedMics.copiedInfo, inputMicrographs)
        self.assertIs(discardedMics.copiedInfo, inputMicrographs)

        for outputSet in protocol.createdSets:
            self.assertEqual(
                outputSet.streamStates,
                [outputSet.STREAM_OPEN],
            )

        self.assertEqual(pointer.getCalls, 4)

    def testCTFConsensusClosedReconciliationAdvancesBothWatermarks(self):
        from pyworkflow.protocol.constants import MODE_RESTART

        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _ClosedLogicalCtfSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.uniqueCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))

                if where == "id > 3":
                    return []

                if where is None:
                    return [1, 2, 3, 4]

                raise AssertionError(
                    "Unexpected query: %r" % where
                )

            def getSize(self):
                return 4

            def isStreamClosed(self):
                return True

            def close(self):
                self.closeCalls += 1

        input1 = _ClosedLogicalCtfSet()
        input2 = _ClosedLogicalCtfSet()
        pointer1 = _Pointer(input1)
        pointer2 = _Pointer(input2)

        class _RunMode:
            def get(self):
                return MODE_RESTART

        class _Harness(XmippStreamingBase):
            inputCTF = pointer1
            inputCTF2 = pointer2
            calculateConsensus = True
            runMode = _RunMode()

            def _getFirstJoinStep(self):
                return None

            def _insertNewCtfsSteps(self, newIds):
                self.batches.append(list(newIds))
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

        protocol = _Harness()
        protocol.insertedIds = [1, 2, 3]
        protocol._lastInputId = 0
        protocol._lastInputId1 = 3
        protocol._lastInputId2 = 3
        protocol._pendingInputIds1 = set()
        protocol._pendingInputIds2 = set()
        protocol.isStreamClosed = False
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtCTFConsensus._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4]])
        self.assertEqual(protocol.insertedIds, [1, 2, 3, 4])
        self.assertEqual(protocol._lastInputId1, 4)
        self.assertEqual(protocol._lastInputId2, 4)
        self.assertEqual(protocol._pendingInputIds1, set())
        self.assertEqual(protocol._pendingInputIds2, set())
        self.assertTrue(protocol.isStreamClosed)
        self.assertEqual(protocol.updateCalls, 1)

        self.assertEqual(
            input1.uniqueCalls,
            [
                ("id", "id > 3"),
                ("id", None),
            ],
        )
        self.assertEqual(
            input2.uniqueCalls,
            [
                ("id", "id > 3"),
                ("id", None),
            ],
        )
        self.assertEqual(pointer1.getCalls, 1)
        self.assertEqual(pointer2.getCalls, 1)
        self.assertEqual(input1.loadCalls, 1)
        self.assertEqual(input2.loadCalls, 1)
        self.assertEqual(input1.closeCalls, 1)
        self.assertEqual(input2.closeCalls, 1)

    def testCTFConsensusInsertAllStepsDoesNotRequireSecondaryFilename(self):
        from xmipp3.protocols.protocol_ctf_consensus import (
            XmippProtCTFConsensus,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _LogicalCtfSet:
            def getFileName(self):
                raise AssertionError(
                    "CTFConsensus must not require the secondary input filename."
                )

        inputSet2 = _LogicalCtfSet()
        pointer2 = _Pointer(inputSet2)

        class _Harness(XmippStreamingBase):
            calculateConsensus = True
            inputCTF2 = pointer2

            def initializeParams(self):
                self.initializeCalls += 1

            def _insertFunctionStep(
                self,
                func,
                prerequisites=None,
                wait=False,
                needsGPU=False,
            ):
                self.insertedSteps.append(
                    (
                        func,
                        prerequisites,
                        wait,
                        needsGPU,
                    )
                )
                return 1

            def createOutputStep(self):
                pass

        protocol = _Harness()
        protocol.initializeCalls = 0
        protocol.insertedSteps = []

        XmippProtCTFConsensus._insertAllSteps(protocol)

        self.assertEqual(protocol.initializeCalls, 1)
        self.assertEqual(pointer2.getCalls, 0)
        self.assertEqual(len(protocol.insertedSteps), 1)

        func, prerequisites, wait, needsGPU = (
            protocol.insertedSteps[0]
        )

        self.assertEqual(func.__name__, "createOutputStep")
        self.assertEqual(prerequisites, [])
        self.assertTrue(wait)
        self.assertFalse(needsGPU)
