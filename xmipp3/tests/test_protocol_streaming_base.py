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


    def testPreprocessMicrographsUsesSharedStreamingBase(self):
        from pwem.protocols import ProtPreprocessMicrographs

        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )

        baseClass = self._getBaseClass()

        self.assertTrue(
            issubclass(XmippProtPreprocessMicrographs, baseClass),
            "PreprocessMicrographs must reuse the shared Xmipp streaming helpers.",
        )

        self.assertTrue(
            issubclass(
                XmippProtPreprocessMicrographs,
                ProtPreprocessMicrographs,
            ),
            "PreprocessMicrographs must keep its native "
            "ProtPreprocessMicrographs behavior.",
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
        protocol.newDeps = []

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

    def __contains__(self, itemId):
        return True

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

        from threading import Lock

        class _Harness:
            inputMovies = pointer
            samplingRate = 1.0
            rejType = XmippProtMovieMaxShift.REJ_FRAME
            REJ_FRAME = XmippProtMovieMaxShift.REJ_FRAME
            REJ_MOVIE = XmippProtMovieMaxShift.REJ_MOVIE
            REJ_AND = XmippProtMovieMaxShift.REJ_AND
            REJ_OR = XmippProtMovieMaxShift.REJ_OR
            MOVIE_VISIBILITY_MAX_ATTEMPTS = (
                XmippProtMovieMaxShift.MOVIE_VISIBILITY_MAX_ATTEMPTS
            )
            MOVIE_VISIBILITY_RETRY_DELAY = (
                XmippProtMovieMaxShift.MOVIE_VISIBILITY_RETRY_DELAY
            )

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

            def _loadMovieForEvaluation(self, movieId):
                return XmippProtMovieMaxShift._loadMovieForEvaluation(
                    self,
                    movieId,
                )

        protocol = _Harness()
        protocol.acceptedIds = []
        protocol.discardedIds = []
        protocol._lock = Lock()

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
        self.samplingRate = None

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

    def setSamplingRate(self, samplingRate):
        self.samplingRate = samplingRate


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

    def __contains__(self, itemId):
        return itemId in self.ids

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

    def __contains__(self, itemId):
        return self.visible and itemId == self.objId

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
        # Set.getItem raises rather than returning None for a row it
        # cannot find, so fillOutput now checks membership first and only
        # calls getItem once the mic is actually visible - the first,
        # not-yet-visible pass must not call getItem at all.
        self.assertEqual(
            inputMics.getItemCalls,
            [("id", 7)],
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
        protocol.newDeps = []

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
            runMode = _RunMode()
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
        from pyworkflow.protocol.constants import MODE_RESTART

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


        class _Harness:
            inputMovies = _Pointer(_LogicalMovieSet())
            _originalRunMode = MODE_RESTART

            def getRunMode(self):
                return MODE_RESTART

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

            def __contains__(self, movieId):
                return True

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Movie(value)

            def close(self):
                self.closeCalls += 1


        logicalSet = _LogicalMovieSet()
        pointer = _Pointer(logicalSet)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer
            MOVIE_VISIBILITY_MAX_ATTEMPTS = (
                XmippProtMovieDoseAnalysis.MOVIE_VISIBILITY_MAX_ATTEMPTS
            )
            MOVIE_VISIBILITY_RETRY_DELAY = (
                XmippProtMovieDoseAnalysis.MOVIE_VISIBILITY_RETRY_DELAY
            )

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

            def __contains__(self, itemId):
                return True

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

            def __contains__(self, itemId):
                return True

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

            def __contains__(self, itemId):
                return True

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Micrograph(value)

            def close(self):
                self.closeCalls += 1

        inputSet = _LogicalMicrographSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            MIC_VISIBILITY_MAX_ATTEMPTS = (
                XmippProtTiltAnalysis.MIC_VISIBILITY_MAX_ATTEMPTS
            )
            MIC_VISIBILITY_RETRY_DELAY = (
                XmippProtTiltAnalysis.MIC_VISIBILITY_RETRY_DELAY
            )

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
        protocol.newDeps = []

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
        protocol.newDeps = []

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
                return list(newDone)

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

            def __contains__(self, itemId):
                return True

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
            CTF_VISIBILITY_MAX_ATTEMPTS = (
                XmippProtCTFConsensus.CTF_VISIBILITY_MAX_ATTEMPTS
            )
            CTF_VISIBILITY_RETRY_DELAY = (
                XmippProtCTFConsensus.CTF_VISIBILITY_RETRY_DELAY
            )
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

            def __contains__(self, itemId):
                return True

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
        protocol.newDeps = []

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

    def testCTFConsensusStepsGeneratorDoesNotRequireSecondaryFilenameUpfront(self):
        # CTFConsensus now uses the generic ProtStreamingBase._insertAllSteps
        # (shared by every migrated protocol, which only inserts the
        # resumableStepGeneratorStep - no CTF-specific logic runs during
        # step insertion any more). The behavior this test used to pin -
        # not touching the secondary input's filename while seeding steps -
        # is exercised here at the stepsGeneratorStep level instead, which
        # is where CTFConsensus's own logic now actually starts running.
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
                self.finished = True

            def _checkNewInput(self):
                self.checkNewInputCalls += 1

            def _checkNewOutput(self):
                self.checkNewOutputCalls += 1

            def _insertFunctionStep(
                self,
                func,
                prerequisites=None,
                needsGPU=False,
            ):
                self.insertedSteps.append(
                    (func, prerequisites, needsGPU)
                )
                return 1

            def createOutputStep(self):
                pass

        protocol = _Harness()
        protocol.initializeCalls = 0
        protocol.checkNewInputCalls = 0
        protocol.checkNewOutputCalls = 0
        protocol.insertedSteps = []

        XmippProtCTFConsensus.stepsGeneratorStep(protocol)

        self.assertEqual(protocol.initializeCalls, 1)
        self.assertEqual(protocol.checkNewInputCalls, 0)
        self.assertEqual(protocol.checkNewOutputCalls, 0)
        self.assertEqual(pointer2.getCalls, 0)
        self.assertEqual(len(protocol.insertedSteps), 1)

        func, prerequisites, needsGPU = protocol.insertedSteps[0]

        self.assertEqual(func.__name__, "createOutputStep")
        self.assertEqual(prerequisites, [])
        self.assertFalse(needsGPU)


class TestXmippMicDefocusSamplerStreamingBase(unittest.TestCase):

    def testMicDefocusSamplerUsesSharedStreamingBase(self):
        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            XmippProtMicDefocusSampler,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        self.assertTrue(
            issubclass(
                XmippProtMicDefocusSampler,
                XmippStreamingBase,
            )
        )

    def testMicDefocusSamplerInitializeParamsDoesNotRequireInputFilename(self):
        from pyworkflow.protocol.constants import MODE_RESUME

        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            XmippProtMicDefocusSampler,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _LogicalCtfSet:
            def getFileName(self):
                raise AssertionError(
                    "MicDefocusSampler must not require the input filename."
                )

        class _RunMode:
            def get(self):
                return MODE_RESUME

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            runMode = _RunMode()
            _originalRunMode = MODE_RESUME
            sampledIds = [4, 7]

            def getRunMode(self):
                return MODE_RESUME

        protocol = _Harness()

        XmippProtMicDefocusSampler.initializeParams(protocol)

        self.assertFalse(protocol.finished)
        self.assertEqual(protocol.insertedIds, [])
        self.assertEqual(protocol.sampled_images, [4, 7])
        self.assertFalse(hasattr(protocol, "ctfFn"))
        self.assertEqual(pointer.getCalls, 0)

    def testMicDefocusSamplerAccumulatesIncrementalIdsUntilSamplingThreshold(self):
        from pyworkflow.protocol.constants import MODE_RESTART

        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            XmippProtMicDefocusSampler,
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

                responses = {
                    "id > 0": [1, 2],
                    "id > 2": [3],
                }

                if where not in responses:
                    raise AssertionError(
                        "Unexpected incremental query: %r" % where
                    )

                return responses[where]

            def getIdSet(self):
                raise AssertionError(
                    "MicDefocusSampler streaming discovery must not "
                    "scan getIdSet()."
                )

            def isStreamClosed(self):
                return False

            def close(self):
                self.closeCalls += 1

        class _Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class _RunMode:
            def get(self):
                return MODE_RESTART

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            minImages = _Param(3)
            runMode = _RunMode()

            @property
            def ctfFn(self):
                raise AssertionError(
                    "MicDefocusSampler streaming discovery must not "
                    "depend on ctfFn."
                )

            def _insertNewCtfsSteps(self, newIds):
                self.batches.append(list(newIds))
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

        protocol = _Harness()
        protocol.finished = False
        protocol.sampled_images = []
        protocol.insertedIds = []
        protocol._lastInputId = 0
        protocol._pendingInputIds = set()
        protocol.newDeps = []
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtMicDefocusSampler._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [])
        self.assertEqual(protocol.insertedIds, [])
        self.assertEqual(protocol._pendingInputIds, {1, 2})
        self.assertEqual(protocol._lastInputId, 2)

        XmippProtMicDefocusSampler._checkNewInput(protocol)

        self.assertEqual(
            protocol.batches,
            [[1, 2, 3]],
        )
        self.assertEqual(
            protocol.insertedIds,
            [1, 2, 3],
        )
        self.assertEqual(protocol._pendingInputIds, set())
        self.assertEqual(protocol._lastInputId, 3)
        self.assertEqual(protocol.updateCalls, 1)

        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 0"),
                ("id", "id > 2"),
            ],
        )
        self.assertEqual(pointer.getCalls, 2)
        self.assertEqual(inputSet.loadCalls, 2)
        self.assertEqual(inputSet.closeCalls, 2)

    def testMicDefocusSamplerClosedStreamRecoversLateVisibleIds(self):
        from pyworkflow.protocol.constants import MODE_RESTART

        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            XmippProtMicDefocusSampler,
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

                if where == "id > 2":
                    return []

                if where is None:
                    return [1, 2, 3]

                raise AssertionError(
                    "Unexpected query: %r" % where
                )

            def getSize(self):
                return 3

            def isStreamClosed(self):
                return True

            def close(self):
                self.closeCalls += 1

        class _Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class _RunMode:
            def get(self):
                return MODE_RESTART

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            minImages = _Param(10)
            runMode = _RunMode()

            def _insertNewCtfsSteps(self, newIds):
                self.batches.append(list(newIds))
                self.insertedIds.extend(newIds)
                return []

            def updateSteps(self):
                self.updateCalls += 1

        protocol = _Harness()
        protocol.finished = False
        protocol.sampled_images = []
        protocol.insertedIds = []
        protocol._lastInputId = 2
        protocol._pendingInputIds = {1, 2}
        protocol.newDeps = []
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtMicDefocusSampler._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[1, 2, 3]])
        self.assertEqual(protocol.insertedIds, [1, 2, 3])
        self.assertEqual(protocol._pendingInputIds, set())
        self.assertEqual(protocol._lastInputId, 3)
        self.assertEqual(protocol.updateCalls, 1)
        self.assertFalse(protocol.finished)

        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 2"),
                ("id", None),
            ],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testMicDefocusSamplerWorkerLoadsBatchFromLogicalInput(self):
        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            XmippProtMicDefocusSampler,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Ctf:
            def __init__(self, objId, defocusU):
                self.objId = objId
                self.defocusU = defocusU

            def clone(self):
                return _Ctf(self.objId, self.defocusU)

            def getDefocusU(self):
                return self.defocusU

        class _LogicalCtfSet:
            def __init__(self):
                self.loadCalls = 0
                self.getItemCalls = []
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def __contains__(self, itemId):
                return True

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Ctf(value, 10000 + value)

            def close(self):
                self.closeCalls += 1

        class _Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class _SampledIds:
            def __init__(self):
                self.values = None

            def set(self, values):
                self.values = list(values)

        class _SummaryVar:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            numImages = _Param(2)
            CTF_VISIBILITY_MAX_ATTEMPTS = (
                XmippProtMicDefocusSampler.CTF_VISIBILITY_MAX_ATTEMPTS
            )
            CTF_VISIBILITY_RETRY_DELAY = (
                XmippProtMicDefocusSampler.CTF_VISIBILITY_RETRY_DELAY
            )

            @property
            def ctfFn(self):
                raise AssertionError(
                    "MicDefocusSampler worker must not depend on ctfFn."
                )

            def _store(self):
                self.storeCalls += 1

            def info(self, message):
                self.messages.append(message)

        protocol = _Harness()
        protocol.sampledIds = _SampledIds()
        protocol.summaryVar = _SummaryVar()
        protocol.storeCalls = 0
        protocol.messages = []

        XmippProtMicDefocusSampler.extractBalancedDefocus(
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
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

        self.assertEqual(
            sorted(protocol.sampled_images),
            sorted(protocol.sampledIds.values),
        )
        self.assertEqual(len(protocol.sampled_images), 2)
        self.assertEqual(protocol.storeCalls, 1)
        self.assertIsNotNone(protocol.summaryVar.value)

    def testMicDefocusSamplerFillOutputLoadsLogicalInput(self):
        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            XmippProtMicDefocusSampler,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Micrograph:
            def __init__(self, objId):
                self.objId = objId

            def clone(self):
                return _Micrograph(self.objId)

            def getObjId(self):
                return self.objId

        class _Ctf:
            def __init__(self, objId):
                self.objId = objId
                self.micrograph = _Micrograph(objId)

            def clone(self):
                return _Ctf(self.objId)

            def getObjId(self):
                return self.objId

            def getMicrograph(self):
                return self.micrograph

        class _LogicalCtfSet:
            def __init__(self):
                self.loadCalls = 0
                self.getItemCalls = []
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def __contains__(self, itemId):
                return True

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Ctf(value)

            def close(self):
                self.closeCalls += 1

        class _OutputSet:
            def __init__(self, existingIds=None):
                self.ids = set(existingIds or [])
                self.items = []

            def getSize(self):
                return len(self.ids)

            def getIdSet(self):
                return set(self.ids)

            def append(self, item):
                self.items.append(item)
                self.ids.add(item.getObjId())

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)
        outputCtfs = _OutputSet(existingIds=[4])
        outputMics = _OutputSet(existingIds=[4])

        class _Harness(XmippStreamingBase):
            inputCTF = pointer

            @property
            def ctfFn(self):
                raise AssertionError(
                    "MicDefocusSampler fillOutput must not depend on ctfFn."
                )

        protocol = _Harness()

        XmippProtMicDefocusSampler.fillOutput(
            protocol,
            outputCtfs,
            outputMics,
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
            [ctf.getObjId() for ctf in outputCtfs.items],
            [7],
        )
        self.assertEqual(
            [mic.getObjId() for mic in outputMics.items],
            [7],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testMicDefocusSamplerOutputCreationUsesProtocolFactories(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_mics_defocus_balancer as defocus_sampler
        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            OUTPUT_CTF,
            OUTPUT_MICS,
            XmippProtMicDefocusSampler,
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
            def _initialize(self):
                self.enableAppendCalls = 0
                self.copiedInfo = None
                self.micrographs = None

            def enableAppend(self):
                self.enableAppendCalls += 1

            def copyInfo(self, other):
                self.copiedInfo = other

            def setMicrographs(self, micrographs):
                self.micrographs = micrographs

        class _FakeSetOfCTF(_FactorySet):
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "MicDefocusSampler must not construct "
                    "SetOfCTF(filename=...)."
                )

        class _FakeSetOfMicrographs(_FactorySet):
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "MicDefocusSampler must not construct "
                    "SetOfMicrographs(filename=...)."
                )

        inputMicrographs = _InputMicrographs()
        inputCtfSet = _InputCtfSet(inputMicrographs)
        pointer = _Pointer(inputCtfSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer

            def _getPath(self, *args):
                raise AssertionError(
                    "MicDefocusSampler output creation must not depend "
                    "on legacy sqlite paths."
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
            defocus_sampler,
            "SetOfCTF",
            _FakeSetOfCTF,
        ), patch.object(
            defocus_sampler,
            "SetOfMicrographs",
            _FakeSetOfMicrographs,
        ):
            ctfOutput = XmippProtMicDefocusSampler._loadOutputSet(
                protocol,
                _FakeSetOfCTF,
                OUTPUT_CTF,
            )
            micOutput = XmippProtMicDefocusSampler._loadOutputSet(
                protocol,
                _FakeSetOfMicrographs,
                OUTPUT_MICS,
            )

        self.assertEqual(
            protocol.factoryCalls,
            [
                ("ctf", ""),
                ("micrographs", ""),
            ],
        )
        self.assertIs(ctfOutput.micrographs, inputMicrographs)
        self.assertIs(micOutput.copiedInfo, inputMicrographs)
        self.assertEqual(pointer.getCalls, 2)

    def testMicDefocusSamplerResumeFallbackUsesLogicalInput(self):
        from xmipp3.protocols.protocol_mics_defocus_balancer import (
            OUTPUT_MICS,
            XmippProtMicDefocusSampler,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Micrograph:
            def __init__(self, objId):
                self.objId = objId

            def getObjId(self):
                return self.objId

        class _Ctf:
            def __init__(self, objId, micId):
                self.objId = objId
                self.micrograph = _Micrograph(micId)

            def getObjId(self):
                return self.objId

            def getMicrograph(self):
                return self.micrograph

        class _LogicalCtfSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def __iter__(self):
                return iter([
                    _Ctf(1, 10),
                    _Ctf(2, 20),
                    _Ctf(3, 30),
                ])

            def close(self):
                self.closeCalls += 1

        class _OutputMicrographs:
            def getIdSet(self):
                return {10, 30}

        inputSet = _LogicalCtfSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputCTF = pointer
            outputMicrographs = _OutputMicrographs()

            @property
            def ctfFn(self):
                raise AssertionError(
                    "MicDefocusSampler Resume fallback must not depend "
                    "on ctfFn."
                )

        protocol = _Harness()

        doneIds, sizeOutput = XmippProtMicDefocusSampler._getAllDoneIds(
            protocol,
        )

        self.assertEqual(doneIds, [1, 3])
        self.assertEqual(sizeOutput, 2)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

class TestXmippScreenParticlesStreamingBase(unittest.TestCase):

    def testScreenParticlesUsesSharedStreamingBase(self):
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        self.assertTrue(
            issubclass(
                XmippProtScreenParticles,
                XmippStreamingBase,
            )
        )

    def testScreenParticlesBuildsBatchFromIncrementalLogicalInput(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Particle:
            def __init__(self, objId):
                self.objId = objId

            def getObjId(self):
                return self.objId

        class _LogicalParticleSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.uniqueCalls = []
                self.getItemCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def __contains__(self, itemId):
                return True

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))

                if field == "id" and where == "id > 1":
                    return [2, 3]

                raise AssertionError(
                    "Unexpected incremental query: %r, %r"
                    % (field, where)
                )

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))

                if field != "id":
                    raise AssertionError(
                        "Particles must be loaded by logical id."
                    )

                return _Particle(value)

            def getSize(self):
                return 3

            def isStreamClosed(self):
                return False

            def getFileName(self):
                raise AssertionError(
                    "ScreenParticles discovery must not depend on "
                    "the input Set filename."
                )

            def getIdSet(self):
                raise AssertionError(
                    "ScreenParticles discovery must not scan getIdSet()."
                )

            def close(self):
                self.closeCalls += 1

        class _OutputParticles:
            def __init__(self):
                self.getIdSetCalls = 0

            def getIdSet(self):
                self.getIdSetCalls += 1
                return {1}


        inputSet = _LogicalParticleSet()
        outputSet = _OutputParticles()
        pointer = _Pointer(inputSet)
        writes = []

        def _captureWrite(items, filename, alignType=None):
            writes.append(
                (
                    filename,
                    [item.getObjId() for item in items],
                    alignType,
                )
            )

        class _Harness(XmippStreamingBase):
            inputParticles = pointer
            outputParticles = outputSet
            fnInputMd = "input.xmd"
            fnInputOldMd = "inputOld.xmd"
            _lastInputId = 1

            def _readProcessedIds(self):
                raise AssertionError(
                    "ScreenParticles batch discovery must use persisted "
                    "output ids, not processed_ids.txt."
                )

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )

        protocol = _Harness()

        with patch.object(
            screen_particles,
            "writeSetOfParticles",
            side_effect=_captureWrite,
        ), patch.object(
            screen_particles,
            "cleanPath",
        ) as cleanPath:
            inputSize, streamClosed = (
                XmippProtScreenParticles._loadInput(protocol)
            )

        self.assertEqual(inputSize, 3)
        self.assertFalse(streamClosed)
        self.assertEqual(protocol._lastInputId, 3)

        self.assertEqual(
            writes,
            [
                ("input.xmd", [2, 3], screen_particles.ALIGN_NONE),
                ("inputOld.xmd", [1], screen_particles.ALIGN_NONE),
            ],
        )
        cleanPath.assert_not_called()

        self.assertEqual(outputSet.getIdSetCalls, 1)
        self.assertEqual(
            inputSet.uniqueCalls,
            [("id", "id > 1")],
        )
        self.assertEqual(
            inputSet.getItemCalls,
            [
                ("id", 2),
                ("id", 3),
                ("id", 1),
            ],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testScreenParticlesUsesPersistedOutputIdsForBatchResume(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Particle:
            def __init__(self, objId):
                self.objId = objId

            def getObjId(self):
                return self.objId

        class _LogicalParticleSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.uniqueCalls = []
                self.getItemCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def __contains__(self, itemId):
                return True

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))

                if field == "id" and where == "id > 0":
                    return [1, 2, 3]

                raise AssertionError(
                    "Unexpected incremental query: %r, %r"
                    % (field, where)
                )

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))

                if field != "id":
                    raise AssertionError(
                        "Particles must be loaded by logical id."
                    )

                return _Particle(value)

            def getSize(self):
                return 3

            def isStreamClosed(self):
                return False

            def close(self):
                self.closeCalls += 1

        class _OutputParticles:
            def __init__(self):
                self.getIdSetCalls = 0

            def getIdSet(self):
                self.getIdSetCalls += 1
                return {1}


        inputSet = _LogicalParticleSet()
        outputSet = _OutputParticles()
        pointer = _Pointer(inputSet)
        writes = []

        def _captureWrite(items, filename, alignType=None):
            writes.append(
                (
                    filename,
                    [item.getObjId() for item in items],
                    alignType,
                )
            )

        class _Harness(XmippStreamingBase):
            inputParticles = pointer
            outputParticles = outputSet
            fnInputMd = "input.xmd"
            fnInputOldMd = "inputOld.xmd"
            _lastInputId = 0

            def _readProcessedIds(self):
                raise AssertionError(
                    "ScreenParticles Resume must use persisted output "
                    "ids, not processed_ids.txt."
                )

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )

        protocol = _Harness()

        with patch.object(
            screen_particles,
            "writeSetOfParticles",
            side_effect=_captureWrite,
        ), patch.object(
            screen_particles,
            "cleanPath",
        ) as cleanPath:
            inputSize, streamClosed = (
                XmippProtScreenParticles._loadInput(protocol)
            )

        self.assertEqual(inputSize, 3)
        self.assertFalse(streamClosed)
        self.assertEqual(protocol._lastInputId, 3)

        self.assertEqual(
            writes,
            [
                ("input.xmd", [2, 3], screen_particles.ALIGN_NONE),
                ("inputOld.xmd", [1], screen_particles.ALIGN_NONE),
            ],
        )
        cleanPath.assert_not_called()

        self.assertEqual(outputSet.getIdSetCalls, 1)
        self.assertEqual(
            protocol._getKnownPersistedOutputIds("outputParticles"),
            {1},
        )
        self.assertEqual(
            inputSet.uniqueCalls,
            [("id", "id > 0")],
        )
        self.assertEqual(
            inputSet.getItemCalls,
            [
                ("id", 2),
                ("id", 3),
                ("id", 1),
            ],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testScreenParticlesInitializesOutputSizeFromPersistedOutput(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _OutputParticles:
            def __init__(self):
                self.getIdSetCalls = 0

            def getIdSet(self):
                self.getIdSetCalls += 1
                return {4, 7}

        class _Harness(XmippStreamingBase):
            outputParticles = _OutputParticles()

            def _initializeZscores(self):
                pass

            def _getExtraPath(self, name):
                return name

            def isContinued(self):
                return True

            def _readProcessedIds(self):
                raise AssertionError(
                    "ScreenParticles must initialize progress from "
                    "persisted output, not processed_ids.txt."
                )

            def _loadInput(self):
                self.loadInputCalls += 1
                return 3, False

            def _insertNewPartsSteps(self):
                raise AssertionError(
                    "No batch should be inserted in this focused test."
                )

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )

        protocol = _Harness()
        protocol.loadInputCalls = 0
        protocol.newDeps = []

        with patch.object(
            screen_particles.os.path,
            "exists",
            return_value=False,
        ), patch.object(
            screen_particles,
            "isEmpty",
            return_value=True,
        ):
            XmippProtScreenParticles._prepareStreamingGenerator(protocol)

        self.assertEqual(protocol.outputSize, 2)
        self.assertEqual(protocol.inputSize, 3)
        self.assertFalse(protocol.streamClosed)
        self.assertEqual(protocol.loadInputCalls, 1)
        self.assertEqual(protocol.newDeps, [])
        self.assertEqual(
            protocol.outputParticles.getIdSetCalls,
            1,
        )
        self.assertEqual(
            protocol._getKnownPersistedOutputIds("outputParticles"),
            {4, 7},
        )

    def testScreenParticlesMarksPublishedBatchAsPersistedWithoutSidecar(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Particle:
            def __init__(self, objId):
                self.objId = objId

            def getObjId(self):
                return self.objId

        class _OutputParticles:
            def __init__(self):
                self.ids = {1}
                self.getIdSetCalls = 0

            def getSize(self):
                return len(self.ids)

            def getIdSet(self):
                self.getIdSetCalls += 1
                return set(self.ids)

            def append(self, particle):
                self.ids.add(particle.getObjId())

            def iterItems(self, orderBy=None):
                return iter([])

        outputSet = _OutputParticles()

        class _Harness(XmippStreamingBase):
            fnInputMd = "input.xmd"
            fnOutputMd = "output.xmd"
            streamClosed = False
            inputSize = 3
            outputSize = 1
            finished = False
            inputParticles = object()

            def _loadOutputSet(self, SetClass, baseName):
                return outputSet

            def _readMetadataIds(self, metadataFile):
                self.metadataReads.append(metadataFile)
                return [2, 3]

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )
            _markRejectedParticleIds = (
                XmippProtScreenParticles._markRejectedParticleIds
            )

            def _createSetOfParticles(self):
                class _BatchSet:
                    def getIdSet(self):
                        return {2, 3}

                return _BatchSet()

            def _appendNewParticles(self, outSet, particles):
                outSet.append(_Particle(2))
                outSet.append(_Particle(3))

            def _recalculateSummaryValues(self, outSet):
                self.summaryCalls += 1

            def _getPath(self, name):
                return name

            def _updateOutputSet(self, name, outSet, streamMode):
                self.events.append("update")
                self.outputParticles = outSet
                self.updatedOutputs.append((name, streamMode))

            def _defineTransformRelation(self, source, target):
                self.events.append("relation")
                self.relations.append((source, target))

            def _store(self):
                self.storeCalls += 1


        protocol = _Harness()
        protocol.metadataReads = []
        protocol.summaryCalls = 0
        protocol.updatedOutputs = []
        protocol.storeCalls = 0
        protocol.events = []
        protocol.relations = []

        with patch.object(
            screen_particles.os.path,
            "exists",
            return_value=True,
        ), patch.object(
            screen_particles,
            "readSetOfParticles",
        ), patch.object(
            screen_particles,
            "writeSetOfParticles",
        ), patch.object(
            screen_particles,
            "cleanPath",
        ) as cleanPath:
            XmippProtScreenParticles._checkNewOutput(protocol)

        self.assertEqual(
            protocol._getKnownPersistedOutputIds("outputParticles"),
            {1, 2, 3},
        )
        self.assertEqual(protocol.outputSize, 3)
        self.assertEqual(outputSet.getIdSetCalls, 1)
        self.assertEqual(protocol.metadataReads, ["input.xmd"])
        self.assertEqual(protocol.summaryCalls, 1)
        self.assertEqual(protocol.storeCalls, 1)
        self.assertEqual(protocol.events, ["update", "relation"])
        self.assertEqual(
            protocol.relations,
            [(protocol.inputParticles, outputSet)],
        )
        cleanPath.assert_called_once_with("output.xmd")

    def testScreenParticlesDoesNotCreateProcessedIdsSidecar(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Harness(XmippStreamingBase):
            def _initializeZscores(self):
                pass

            def _getExtraPath(self, name):
                self.extraPaths.append(name)
                return name

            def isContinued(self):
                return False

            def _loadInput(self):
                return 0, False

            def _insertNewPartsSteps(self):
                return []

            def _insertFunctionStep(
                self,
                funcName,
                prerequisites=None,
                wait=False,
            ):
                self.insertedSteps.append(
                    (funcName, prerequisites, wait)
                )
                return len(self.insertedSteps)

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )

        protocol = _Harness()
        protocol.extraPaths = []
        protocol.insertedSteps = []

        with patch.object(
            screen_particles,
            "cleanPath",
        ) as cleanPath, patch.object(
            screen_particles.os.path,
            "exists",
            return_value=False,
        ), patch.object(
            screen_particles,
            "isEmpty",
            return_value=True,
        ):
            XmippProtScreenParticles._prepareStreamingGenerator(protocol)

        self.assertEqual(
            protocol.extraPaths,
            [
                "input.xmd",
                "inputOld.xmd",
                "output.xmd",
            ],
        )
        self.assertFalse(
            hasattr(protocol, "fnProcessedIds")
        )
        self.assertEqual(
            [call.args[0] for call in cleanPath.call_args_list],
            [
                "input.xmd",
                "inputOld.xmd",
                "output.xmd",
            ],
        )

    def testScreenParticlesInputStatusUsesLogicalSet(self):
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _LogicalParticleSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0

            def loadAllProperties(self):
                self.loadCalls += 1

            def getSize(self):
                return 7

            def isStreamClosed(self):
                return True

            def getFileName(self):
                raise AssertionError(
                    "ScreenParticles input status must not depend on "
                    "the input Set filename."
                )

            def close(self):
                self.closeCalls += 1


        inputSet = _LogicalParticleSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputParticles = pointer

        protocol = _Harness()

        inputSize, streamClosed = (
            XmippProtScreenParticles._getInputStatus(protocol)
        )

        self.assertEqual(inputSize, 7)
        self.assertTrue(streamClosed)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testScreenParticlesOutputCreationUsesProtocolFactory(self):
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _InputParticles:
            pass


        class _FactorySet:
            STREAM_OPEN = "open"

            def __init__(self):
                self.streamStates = []
                self.copyInfoCalls = []

            def setStreamState(self, state):
                self.streamStates.append(state)

            def enableAppend(self):
                raise AssertionError(
                    "A newly created output must not be reopened."
                )

            def copyInfo(self, inputSet):
                self.copyInfoCalls.append(inputSet)

        class _LegacySetClass:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "ScreenParticles must not construct "
                    "SetOfParticles(filename=...)."
                )

        inputSet = _InputParticles()
        pointer = _Pointer(inputSet)
        outputSet = _FactorySet()

        class _Harness(XmippStreamingBase):
            inputParticles = pointer

            def _getPath(self, *args):
                raise AssertionError(
                    "ScreenParticles output creation must not depend "
                    "on a legacy sqlite path."
                )

            def _createSetOfParticles(self, suffix=''):
                self.factoryCalls.append(suffix)
                return outputSet

            def _store(self, obj=None):
                raise AssertionError(
                    "_loadOutputSet() must not store a newly created Set "
                    "before the first batch is appended."
                )

            def _defineTransformRelation(self, source, target):
                raise AssertionError(
                    "_loadOutputSet() must not define the relation before "
                    "_updateOutputSet() publishes the first batch."
                )

        protocol = _Harness()
        protocol.factoryCalls = []

        result = XmippProtScreenParticles._loadOutputSet(
            protocol,
            _LegacySetClass,
            "outputParticles.sqlite",
        )

        self.assertIs(result, outputSet)
        self.assertEqual(
            protocol.factoryCalls,
            ['_output'],
        )
        self.assertEqual(outputSet.streamStates, ["open"])
        self.assertEqual(outputSet.copyInfoCalls, [inputSet])
        self.assertEqual(pointer.getCalls, 1)

    def testScreenParticlesClosedStreamRecoversLateVisibleIds(self):
        from unittest.mock import patch

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Particle:
            def __init__(self, objId):
                self.objId = objId

            def getObjId(self):
                return self.objId

        class _LogicalParticleSet:
            def __init__(self):
                self.loadCalls = 0
                self.closeCalls = 0
                self.uniqueCalls = []
                self.getItemCalls = []

            def loadAllProperties(self):
                self.loadCalls += 1

            def __contains__(self, itemId):
                return True

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))

                if field != "id":
                    raise AssertionError(
                        "Unexpected field: %r" % field
                    )

                if where == "id > 2":
                    return []

                if where is None:
                    return [1, 2, 3]

                raise AssertionError(
                    "Unexpected query: %r" % where
                )

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Particle(value)

            def getSize(self):
                return 3

            def isStreamClosed(self):
                return True

            def close(self):
                self.closeCalls += 1

        class _OutputParticles:
            def __init__(self):
                self.getIdSetCalls = 0

            def getIdSet(self):
                self.getIdSetCalls += 1
                return {1, 2}


        inputSet = _LogicalParticleSet()
        outputSet = _OutputParticles()
        pointer = _Pointer(inputSet)
        writes = []

        def _captureWrite(items, filename, alignType=None):
            writes.append(
                (
                    filename,
                    [item.getObjId() for item in items],
                    alignType,
                )
            )

        class _Harness(XmippStreamingBase):
            inputParticles = pointer
            outputParticles = outputSet
            fnInputMd = "input.xmd"
            fnInputOldMd = "inputOld.xmd"
            _lastInputId = 2

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )

        protocol = _Harness()

        with patch.object(
            screen_particles,
            "writeSetOfParticles",
            side_effect=_captureWrite,
        ), patch.object(
            screen_particles,
            "cleanPath",
        ) as cleanPath:
            inputSize, streamClosed = (
                XmippProtScreenParticles._loadInput(protocol)
            )

        self.assertEqual(inputSize, 3)
        self.assertTrue(streamClosed)
        self.assertEqual(protocol._lastInputId, 3)

        self.assertEqual(
            writes,
            [
                ("input.xmd", [3], screen_particles.ALIGN_NONE),
                ("inputOld.xmd", [1, 2], screen_particles.ALIGN_NONE),
            ],
        )
        cleanPath.assert_not_called()

        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 2"),
                ("id", None),
            ],
        )
        self.assertEqual(
            inputSet.getItemCalls,
            [
                ("id", 3),
                ("id", 1),
                ("id", 2),
            ],
        )
        self.assertEqual(outputSet.getIdSetCalls, 1)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

    def testScreenParticlesResumeSkipsPersistedRejectedIds(self):
        from unittest.mock import patch

        from pyworkflow.object import CsvList

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Particle:
            def __init__(self, objId):
                self.objId = objId

            def getObjId(self):
                return self.objId

        class _LogicalParticleSet:
            def __init__(self):
                self.uniqueCalls = []
                self.getItemCalls = []

            def loadAllProperties(self):
                pass

            def __contains__(self, itemId):
                return True

            def getUniqueValues(self, field, where=None):
                self.uniqueCalls.append((field, where))
                if field != "id":
                    raise AssertionError(
                        "Unexpected field: %r" % field
                    )
                if where == "id > 0":
                    return [1, 2, 3]
                if where is None:
                    return [1, 2, 3]
                raise AssertionError(
                    "Unexpected query: %r" % where
                )

            def getItem(self, field, value):
                self.getItemCalls.append((field, value))
                return _Particle(value)

            def getSize(self):
                return 3

            def isStreamClosed(self):
                return True

            def close(self):
                pass

        class _OutputParticles:
            def getIdSet(self):
                # Particle 2 was processed in a previous batch but rejected.
                return {1, 3}

        class _Pointer:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        inputSet = _LogicalParticleSet()
        writes = []

        def _captureWrite(items, filename, alignType=None):
            writes.append(
                (
                    filename,
                    [item.getObjId() for item in items],
                    alignType,
                )
            )

        class _Harness(XmippStreamingBase):
            inputParticles = _Pointer(inputSet)
            outputParticles = _OutputParticles()
            fnInputMd = "input.xmd"
            fnInputOldMd = "inputOld.xmd"
            _lastInputId = 0

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )

        protocol = _Harness()
        protocol._rejectedParticleIds = CsvList(pType=int)
        protocol._rejectedParticleIds.set([2])

        with patch.object(
            screen_particles,
            "writeSetOfParticles",
            side_effect=_captureWrite,
        ), patch.object(
            screen_particles,
            "cleanPath",
        ):
            inputSize, streamClosed = (
                XmippProtScreenParticles._loadInput(protocol)
            )

        self.assertEqual(inputSize, 3)
        self.assertTrue(streamClosed)

        self.assertEqual(
            writes,
            [
                ("input.xmd", [], screen_particles.ALIGN_NONE),
                (
                    "inputOld.xmd",
                    [1, 2, 3],
                    screen_particles.ALIGN_NONE,
                ),
            ],
        )

    def testScreenParticlesPersistsRejectedIdsFromPublishedBatch(self):
        from unittest.mock import patch

        from pyworkflow.object import CsvList

        import xmipp3.protocols.protocol_screen_particles as screen_particles
        from xmipp3.protocols.protocol_screen_particles import (
            XmippProtScreenParticles,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        class _Particle:
            def __init__(self, objId):
                self.objId = objId

            def getObjId(self):
                return self.objId

        class _BatchSet:
            def getIdSet(self):
                # Particle 2 was rejected by Xmipp and removed by
                # readSetOfParticles(removeDisabled=True).
                return {3}

        class _OutputParticles:
            def __init__(self):
                self.ids = {1}

            def getIdSet(self):
                return set(self.ids)

            def getSize(self):
                return len(self.ids)

            def append(self, particle):
                self.ids.add(particle.getObjId())

            def iterItems(self, orderBy=None):
                return iter([])

        outputSet = _OutputParticles()
        batchSet = _BatchSet()

        class _Harness(XmippStreamingBase):
            fnInputMd = "input.xmd"
            fnOutputMd = "output.xmd"
            streamClosed = False
            inputSize = 3
            outputSize = 1
            finished = False
            inputParticles = object()

            _getRejectedParticleIds = (
                XmippProtScreenParticles._getRejectedParticleIds
            )
            _getKnownProcessedParticleIds = (
                XmippProtScreenParticles._getKnownProcessedParticleIds
            )
            _markRejectedParticleIds = (
                XmippProtScreenParticles._markRejectedParticleIds
            )

            def _loadOutputSet(self, SetClass, baseName):
                return outputSet

            def _readMetadataIds(self, metadataFile):
                return [2, 3]

            def _createSetOfParticles(self):
                return batchSet

            def _appendNewParticles(self, outSet, particles):
                outSet.append(_Particle(3))

            def _recalculateSummaryValues(self, outSet):
                pass

            def _getPath(self, name):
                return name

            def _updateOutputSet(self, name, outSet, streamMode):
                self.outputParticles = outSet

            def _defineTransformRelation(self, source, target):
                pass

            def _store(self):
                self.storeCalls += 1

        protocol = _Harness()
        protocol._rejectedParticleIds = CsvList(pType=int)
        protocol.storeCalls = 0

        with patch.object(
            screen_particles.os.path,
            "exists",
            return_value=True,
        ), patch.object(
            screen_particles,
            "readSetOfParticles",
        ), patch.object(
            screen_particles,
            "writeSetOfParticles",
        ), patch.object(
            screen_particles,
            "cleanPath",
        ):
            XmippProtScreenParticles._checkNewOutput(protocol)

        self.assertEqual(
            set(protocol._rejectedParticleIds),
            {2},
        )
        self.assertEqual(
            protocol._getKnownPersistedOutputIds("outputParticles"),
            {1, 3},
        )
        self.assertEqual(
            protocol._getKnownProcessedParticleIds(),
            {1, 2, 3},
        )
        self.assertEqual(protocol.outputSize, 3)
        self.assertGreaterEqual(protocol.storeCalls, 1)

class _PreprocessMicrograph:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _PreprocessMicrograph(self._objId)


class _PreprocessMicrographSet:
    def __init__(self, ids, streamClosed=False):
        self._items = [_PreprocessMicrograph(objId) for objId in ids]
        self._streamClosed = streamClosed
        self.loadCalls = 0
        self.closeCalls = 0

    def loadAllProperties(self):
        self.loadCalls += 1

    def __iter__(self):
        return iter(self._items)

    def isStreamClosed(self):
        return self._streamClosed

    def close(self):
        self.closeCalls += 1

    def getFileName(self):
        raise AssertionError(
            "PreprocessMicrographs input loading must not depend on "
            "inputMicrographs.getFileName()."
        )


class TestXmippPreprocessMicrographsLogicalInput(unittest.TestCase):

    def testLoadInputMicsUsesLogicalSet(self):
        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        inputSet = _PreprocessMicrographSet([3, 7], streamClosed=False)
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer

        protocol = _Harness()

        inputMics, streamClosed = (
            XmippProtPreprocessMicrographs._loadInputMics(protocol)
        )

        self.assertEqual(
            [mic.getObjId() for mic in inputMics],
            [3, 7],
        )
        self.assertFalse(streamClosed)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

class _IncrementalPreprocessSet:
    def __init__(self):
        self.loadCalls = 0
        self.closeCalls = 0
        self.uniqueCalls = []
        self.getItemCalls = []

    def loadAllProperties(self):
        self.loadCalls += 1

    def __contains__(self, itemId):
        return True

    def getUniqueValues(self, field, where=None):
        self.uniqueCalls.append((field, where))
        if field == "id" and where == "id > 3":
            return [4, 5]
        raise AssertionError(
            "Unexpected discovery query: %r, %r" % (field, where)
        )

    def getItem(self, field, value):
        self.getItemCalls.append((field, value))
        if field != "id" or value not in (4, 5):
            raise AssertionError(
                "Unexpected item lookup: %r, %r" % (field, value)
            )
        return _PreprocessMicrograph(value)

    def isStreamClosed(self):
        return False

    def close(self):
        self.closeCalls += 1

    def __iter__(self):
        raise AssertionError(
            "Streaming polling must not iterate the whole input Set."
        )


class TestXmippPreprocessMicrographsIncrementalDiscovery(unittest.TestCase):

    def testCheckNewInputQueriesOnlyIdsBeyondWatermark(self):
        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        inputSet = _IncrementalPreprocessSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            insertedDict = {1: None, 2: None, 3: None}
            _lastInputId = 3

            def _insertNewMicsSteps(self, insertedDict, inputMics):
                ids = [mic.getObjId() for mic in inputMics]
                self.batches.append(ids)
                for objId in ids:
                    insertedDict[objId] = objId
                return []

            def updateSteps(self):
                self.updateCalls += 1

        protocol = _Harness()
        protocol.insertedDict = {1: None, 2: None, 3: None}
        protocol.batches = []
        protocol.updateCalls = 0
        protocol.newDeps = []

        XmippProtPreprocessMicrographs._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4, 5]])
        self.assertEqual(protocol._lastInputId, 5)
        self.assertEqual(
            inputSet.uniqueCalls,
            [("id", "id > 3")],
        )
        self.assertEqual(
            inputSet.getItemCalls,
            [("id", 4), ("id", 5)],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)
        self.assertEqual(protocol.updateCalls, 1)

class _ClosedIncrementalPreprocessSet(_IncrementalPreprocessSet):
    def getUniqueValues(self, field, where=None):
        self.uniqueCalls.append((field, where))

        if field != "id":
            raise AssertionError("Unexpected field: %r" % field)

        if where == "id > 5":
            return []

        if where is None:
            return [1, 2, 3, 4, 5, 6]

        raise AssertionError(
            "Unexpected discovery query: %r, %r" % (field, where)
        )

    def getItem(self, field, value):
        self.getItemCalls.append((field, value))
        if field != "id" or value != 6:
            raise AssertionError(
                "Unexpected item lookup: %r, %r" % (field, value)
            )
        return _PreprocessMicrograph(value)

    def getSize(self):
        return 6

    def isStreamClosed(self):
        return True


class TestXmippPreprocessMicrographsTerminalReconciliation(unittest.TestCase):

    def testClosedStreamRecoversLateVisibleMicrograph(self):
        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        inputSet = _ClosedIncrementalPreprocessSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            insertedDict = {
                1: None,
                2: None,
                3: None,
                4: None,
                5: None,
            }
            _lastInputId = 5
            SetOfMicrographs = [
                _PreprocessMicrograph(objId)
                for objId in range(1, 6)
            ]

            def _insertNewMicsSteps(self, insertedDict, inputMics):
                ids = [mic.getObjId() for mic in inputMics]
                self.batches.append(ids)
                for objId in ids:
                    insertedDict[objId] = objId
                return []

            def updateSteps(self):
                self.updateCalls += 1

        protocol = _Harness()
        protocol.insertedDict = {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
        }
        protocol.SetOfMicrographs = [
            _PreprocessMicrograph(objId)
            for objId in range(1, 6)
        ]
        protocol.batches = []
        protocol.updateCalls = 0
        protocol.newDeps = []

        XmippProtPreprocessMicrographs._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[6]])
        self.assertEqual(protocol._lastInputId, 6)
        self.assertTrue(protocol.streamClosed)
        self.assertEqual(
            [mic.getObjId() for mic in protocol.SetOfMicrographs],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 5"),
                ("id", None),
            ],
        )
        self.assertEqual(
            inputSet.getItemCalls,
            [("id", 6)],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)
        self.assertEqual(protocol.updateCalls, 1)

class TestXmippPreprocessMicrographsPersistedResume(unittest.TestCase):

    def testRestoreInsertedMicsUsesPersistedOutputIds(self):
        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        outputSet = _PersistedOutputSet({1, 3})

        class _Harness(XmippStreamingBase):
            outputMicrographs = outputSet
            insertedDict = {}

            def _isMicPipelineDone(self, mic):
                return False

        protocol = _Harness()
        protocol.insertedDict = {}

        inputMics = [
            _PreprocessMicrograph(1),
            _PreprocessMicrograph(2),
            _PreprocessMicrograph(3),
        ]

        XmippProtPreprocessMicrographs._restoreInsertedMics(
            protocol,
            inputMics,
        )

        self.assertEqual(
            protocol.insertedDict,
            {
                1: None,
                3: None,
            },
        )
        self.assertEqual(outputSet.getIdSetCalls, 1)

class TestXmippPreprocessMicrographsOutputFactory(unittest.TestCase):

    def testGetOutputMicsUsesProtocolFactory(self):
        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        logicalInput = _OutputInfoSource()
        pointer = _Pointer(logicalInput)

        class _Harness(XmippStreamingBase):
            inputMicrographs = pointer
            doDownsample = False

            def _createSetOfMicrographs(self):
                self.factoryCalls += 1
                return _FakeOutputSet()

            def _defineOutputs(self, **outputs):
                for name, output in outputs.items():
                    setattr(self, name, output)

            def getPath(self, *args):
                raise AssertionError(
                    "PreprocessMicrographs output creation must use "
                    "_createSetOfMicrographs(), not a manual sqlite path."
                )

            def info(self, message):
                pass

        protocol = _Harness()
        protocol.factoryCalls = 0

        output = XmippProtPreprocessMicrographs.getOutputMics(
            protocol,
        )

        self.assertEqual(protocol.factoryCalls, 1)
        self.assertIs(output.copiedFrom, logicalInput)
        self.assertEqual(
            output.streamState,
            _FakeOutputSet.STREAM_OPEN,
        )
        self.assertIs(protocol.outputMicrographs, output)
        self.assertEqual(pointer.getCalls, 1)

class _PublishPreprocessMicrograph(_PreprocessMicrograph):
    def clone(self):
        return _PublishPreprocessMicrograph(self.getObjId())

    def getMicName(self):
        return "mic_%06d" % self.getObjId()


class _PreprocessOutputSet:
    def __init__(self, ids):
        self.ids = set(ids)
        self.appendedIds = []

    def getIdSet(self):
        return set(self.ids)

    def getSize(self):
        return len(self.ids)

    def isEmpty(self):
        return not self.ids

    def append(self, mic):
        micId = mic.getObjId()
        self.ids.add(micId)
        self.appendedIds.append(micId)


class TestXmippPreprocessMicrographsNoDoneAllSidecar(unittest.TestCase):

    def testCheckNewOutputUsesPersistedOutputInsteadOfDoneAll(self):
        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingBase,
        )

        outputSet = _PreprocessOutputSet({1})

        class _Harness(XmippStreamingBase):
            outputMicrographs = outputSet
            SetOfMicrographs = [_PublishPreprocessMicrograph(2)]
            streamClosed = False
            finished = False
            doDownsample = False

            def _isMicPipelineDone(self, mic):
                return True

            def getOutputMics(self):
                return outputSet


            def _readDoneList(self):
                raise AssertionError(
                    "DONE_all.TXT must not be used as durable "
                    "streaming publication state."
                )

            def _writeDoneList(self, micList):
                raise AssertionError(
                    "DONE_all.TXT must not be written after publication."
                )

            def _getOutputMicrograph(self, mic):
                return "/tmp/mic_%06d.mrc" % mic.getObjId()

            def _updateOutputSet(self, outputName, outSet, streamMode):
                self.updatedOutputName = outputName
                self.updatedStreamMode = streamMode

            def _refreshOutputRelation(self, outSet):
                pass

        protocol = _Harness()

        XmippProtPreprocessMicrographs._checkNewOutput(protocol)

        self.assertEqual(outputSet.appendedIds, [2])
        self.assertEqual(protocol.updatedOutputName, "outputMicrographs")

class TestXmippPreprocessMicrographsResumeCompletion(unittest.TestCase):

    def testPersistedOutputCountsAsProcessedWithoutPipelineMarker(self):
        from pyworkflow.object import Set

        from xmipp3.protocols.protocol_preprocess_micrographs import (
            XmippProtPreprocessMicrographs,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        outputSet = _PreprocessOutputSet({1})

        class _Harness(XmippStreamingBase):
            outputMicrographs = outputSet
            SetOfMicrographs = [
                _PublishPreprocessMicrograph(1),
                _PublishPreprocessMicrograph(2),
            ]
            streamClosed = True
            finished = False
            doDownsample = False

            def _isMicPipelineDone(self, mic):
                return mic.getObjId() == 2

            def getOutputMics(self):
                return outputSet

            def _getOutputMicrograph(self, mic):
                return "/tmp/mic_%06d.mrc" % mic.getObjId()

            def _updateOutputSet(self, outputName, outSet, streamMode):
                self.updatedOutputName = outputName
                self.updatedStreamMode = streamMode

            def _refreshOutputRelation(self, outSet):
                pass

        protocol = _Harness()

        XmippProtPreprocessMicrographs._checkNewOutput(protocol)

        self.assertTrue(protocol.finished)
        self.assertEqual(
            protocol.updatedStreamMode,
            Set.STREAM_CLOSED,
        )
        self.assertEqual(outputSet.appendedIds, [2])
        self.assertEqual(
            protocol._getKnownPersistedOutputIds(
                "outputMicrographs",
            ),
            {1, 2},
        )

class TestXmippMovieResizeStreamingBase(unittest.TestCase):

    def testMovieResizeUsesSharedStreamingBase(self):
        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        self.assertTrue(
            issubclass(XmippProtMovieResize, XmippStreamingBase)
        )

class TestXmippMovieResizeLogicalInput(unittest.TestCase):

    def testLoadInputListUsesLogicalSet(self):
        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        inputSet = _PreprocessMicrographSet(
            [2, 4],
            streamClosed=False,
        )
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer

        protocol = _Harness()

        XmippProtMovieResize._loadInputList(protocol)

        self.assertEqual(
            [movie.getObjId() for movie in protocol.listOfMovies],
            [2, 4],
        )
        self.assertFalse(protocol.streamClosed)
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)

class TestXmippMovieResizeIncrementalDiscovery(unittest.TestCase):

    def testCheckNewInputQueriesOnlyIdsBeyondWatermark(self):
        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        inputSet = _IncrementalPreprocessSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer
            insertedDict = {1: None, 2: None, 3: None}
            _lastInputId = 3
            listOfMovies = [
                _PreprocessMicrograph(1),
                _PreprocessMicrograph(2),
                _PreprocessMicrograph(3),
            ]

            def _getFirstJoinStep(self):
                return None

            def _insertNewMoviesSteps(self, insertedDict, inputMovies):
                ids = [movie.getObjId() for movie in inputMovies]
                self.batches.append(ids)
                for objId in ids:
                    insertedDict[objId] = objId
                return []

            def updateSteps(self):
                self.updateCalls += 1

        protocol = _Harness()
        protocol.insertedDict = {1: None, 2: None, 3: None}
        protocol.listOfMovies = [
            _PreprocessMicrograph(1),
            _PreprocessMicrograph(2),
            _PreprocessMicrograph(3),
        ]
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtMovieResize._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[4, 5]])
        self.assertEqual(protocol._lastInputId, 5)
        self.assertEqual(
            [movie.getObjId() for movie in protocol.listOfMovies],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            inputSet.uniqueCalls,
            [("id", "id > 3")],
        )
        self.assertEqual(
            inputSet.getItemCalls,
            [("id", 4), ("id", 5)],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)
        self.assertEqual(protocol.updateCalls, 1)
        self.assertFalse(protocol.streamClosed)

class TestXmippMovieResizeTerminalReconciliation(unittest.TestCase):

    def testClosedStreamRecoversLateVisibleMovie(self):
        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        inputSet = _ClosedIncrementalPreprocessSet()
        pointer = _Pointer(inputSet)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer
            insertedDict = {
                1: None,
                2: None,
                3: None,
                4: None,
                5: None,
            }
            _lastInputId = 5
            listOfMovies = [
                _PreprocessMicrograph(objId)
                for objId in range(1, 6)
            ]

            def _getFirstJoinStep(self):
                return None

            def _insertNewMoviesSteps(self, insertedDict, inputMovies):
                ids = [movie.getObjId() for movie in inputMovies]
                self.batches.append(ids)
                for objId in ids:
                    insertedDict[objId] = objId
                return []

            def updateSteps(self):
                self.updateCalls += 1

        protocol = _Harness()
        protocol.insertedDict = {
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
        }
        protocol.listOfMovies = [
            _PreprocessMicrograph(objId)
            for objId in range(1, 6)
        ]
        protocol.batches = []
        protocol.updateCalls = 0

        XmippProtMovieResize._checkNewInput(protocol)

        self.assertEqual(protocol.batches, [[6]])
        self.assertEqual(protocol._lastInputId, 6)
        self.assertTrue(protocol.streamClosed)
        self.assertEqual(
            [movie.getObjId() for movie in protocol.listOfMovies],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            inputSet.uniqueCalls,
            [
                ("id", "id > 5"),
                ("id", None),
            ],
        )
        self.assertEqual(
            inputSet.getItemCalls,
            [("id", 6)],
        )
        self.assertEqual(pointer.getCalls, 1)
        self.assertEqual(inputSet.loadCalls, 1)
        self.assertEqual(inputSet.closeCalls, 1)
        self.assertEqual(protocol.updateCalls, 1)

class TestXmippMovieResizePersistedResume(unittest.TestCase):

    def testRestoreInsertedMoviesUsesPersistedOutputIds(self):
        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        outputSet = _PersistedOutputSet({1, 3})

        class _Harness(XmippStreamingBase):
            outputMovies = outputSet
            insertedDict = {}

            def _isMovieDone(self, movie):
                return False

        protocol = _Harness()
        protocol.insertedDict = {}

        inputMovies = [
            _PreprocessMicrograph(1),
            _PreprocessMicrograph(2),
            _PreprocessMicrograph(3),
        ]

        XmippProtMovieResize._restoreInsertedMovies(
            protocol,
            inputMovies,
        )

        self.assertEqual(
            protocol.insertedDict,
            {
                1: None,
                3: None,
            },
        )
        self.assertEqual(outputSet.getIdSetCalls, 1)

class TestXmippMovieResizeOutputFactory(unittest.TestCase):

    def testGetOutputMoviesUsesProtocolFactory(self):
        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        logicalInput = _OutputInfoSource()
        pointer = _Pointer(logicalInput)

        class _Harness(XmippStreamingBase):
            inputMovies = pointer

            def _createSetOfMovies(self):
                self.factoryCalls += 1
                return _FakeOutputSet()

            def _getNewSamplingRate(self):
                return 4.5

            def _defineOutputs(self, **outputs):
                for name, output in outputs.items():
                    setattr(self, name, output)

            def _getPath(self, *args):
                raise AssertionError(
                    "MovieResize output creation must use "
                    "_createSetOfMovies(), not a manual sqlite path."
                )

        protocol = _Harness()
        protocol.factoryCalls = 0

        output = XmippProtMovieResize.getOutputMovies(
            protocol,
        )

        self.assertEqual(protocol.factoryCalls, 1)
        self.assertIs(output.copiedFrom, logicalInput)
        self.assertEqual(
            output.streamState,
            _FakeOutputSet.STREAM_OPEN,
        )
        self.assertEqual(
            output.samplingRate,
            4.5,
        )
        self.assertIs(protocol.outputMovies, output)
        self.assertEqual(pointer.getCalls, 1)

class TestXmippMovieResizeNoDoneAllSidecar(unittest.TestCase):

    def testCheckNewOutputUsesPersistedOutputInsteadOfDoneAll(self):
        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        outputSet = _PreprocessOutputSet({1})

        class _Harness(XmippStreamingBase):
            outputMovies = outputSet
            listOfMovies = [
                _PreprocessMicrograph(1),
                _PreprocessMicrograph(2),
            ]
            streamClosed = False
            finished = False

            def _isMovieDone(self, movie):
                return movie.getObjId() == 2

            def getOutputMovies(self):
                return outputSet

            def _appendNewMovies(
                self,
                imageSet,
                movies,
                samplingRate,
            ):
                for movie in movies:
                    imageSet.append(movie)

            def _getNewSamplingRate(self):
                return 4.5

            def _readDoneList(self):
                raise AssertionError(
                    "DONE_all.TXT must not be used as durable "
                    "MovieResize publication state."
                )

            def _writeDoneList(self, movieList):
                raise AssertionError(
                    "DONE_all.TXT must not be written after publication."
                )

            def _loadOutputSet(self, SetClass, baseName):
                raise AssertionError(
                    "MovieResize publication must use getOutputMovies(), "
                    "not a manual movies.sqlite output."
                )

            def _updateOutputSet(
                self,
                outputName,
                outSet,
                streamMode,
            ):
                self.updatedOutputName = outputName
                self.updatedStreamMode = streamMode

            def _getFirstJoinStep(self):
                return None

        protocol = _Harness()

        XmippProtMovieResize._checkNewOutput(protocol)

        self.assertEqual(outputSet.appendedIds, [2])
        self.assertEqual(
            protocol.updatedOutputName,
            "outputMovies",
        )
        self.assertEqual(
            protocol._getKnownPersistedOutputIds(
                "outputMovies",
            ),
            {1, 2},
        )

class TestXmippMovieResizeResumeCompletion(unittest.TestCase):

    def testPersistedOutputCountsAsProcessedWithoutMovieMarker(self):
        from pyworkflow.object import Set

        from xmipp3.protocols.protocol_preprocess.protocol_movie_resize import (
            XmippProtMovieResize,
        )
        from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase

        outputSet = _PreprocessOutputSet({1})

        class _Harness(XmippStreamingBase):
            outputMovies = outputSet
            listOfMovies = [
                _PreprocessMicrograph(1),
                _PreprocessMicrograph(2),
            ]
            streamClosed = True
            finished = False

            def _isMovieDone(self, movie):
                return movie.getObjId() == 2

            def getOutputMovies(self):
                return outputSet

            def _appendNewMovies(
                self,
                imageSet,
                movies,
                samplingRate,
            ):
                for movie in movies:
                    imageSet.append(movie)

            def _getNewSamplingRate(self):
                return 4.5

            def _updateOutputSet(
                self,
                outputName,
                outSet,
                streamMode,
            ):
                self.updatedOutputName = outputName
                self.updatedStreamMode = streamMode

            def _getFirstJoinStep(self):
                return None

        protocol = _Harness()

        XmippProtMovieResize._checkNewOutput(protocol)

        self.assertTrue(protocol.finished)
        self.assertEqual(
            protocol.updatedStreamMode,
            Set.STREAM_CLOSED,
        )
        self.assertEqual(outputSet.appendedIds, [2])
        self.assertEqual(
            protocol._getKnownPersistedOutputIds(
                'outputMovies',
            ),
            {1, 2},
        )
