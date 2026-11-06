import importlib.util
import sys
import types
import unittest
from pathlib import Path

from pwem.objects import Movie, SetOfMovies


def _loadModule(moduleName, modulePath):
    spec = importlib.util.spec_from_file_location(moduleName, str(modulePath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[moduleName] = module
    spec.loader.exec_module(module)
    return module


def _loadProtocolClass(moduleName, className):
    protocolsPath = Path(__file__).resolve().parents[1] / "protocols"
    packageName = "xmipp3.protocols"
    fullModuleName = "%s.%s" % (packageName, moduleName)
    oldPackage = sys.modules.get(packageName)
    package = types.ModuleType(packageName)
    package.__path__ = [str(protocolsPath)]
    sys.modules[packageName] = package

    try:
        _loadModule("xmipp3.protocols.protocol_streaming_base", protocolsPath / "protocol_streaming_base.py")
        module = _loadModule(fullModuleName, protocolsPath / (moduleName + ".py"))
        return getattr(module, className)
    finally:
        sys.modules.pop(fullModuleName, None)
        sys.modules.pop("xmipp3.protocols.protocol_streaming_base", None)
        if oldPackage is None:
            sys.modules.pop(packageName, None)
        else:
            sys.modules[packageName] = oldPackage


XmippProtFlexAlign = _loadProtocolClass("protocol_flexalign", "XmippProtFlexAlign")
XmippProtMovieGain = _loadProtocolClass("protocol_movie_gain", "XmippProtMovieGain")
XmippStreamingMoviesMixin = XmippProtFlexAlign.__mro__[1]


class TestStreamingBackendIndependence(unittest.TestCase):

    def test_FlexAlignStreamingDiscoveryDoesNotRequireInputSqliteFile(self):
        scheduledMovieIds = []
        updateCalls = []

        class MovieStub:
            def __init__(self, objId):
                self._objId = objId

            def getObjId(self):
                return self._objId

        class LogicalMovieSetStub:
            def __iter__(self):
                return iter([MovieStub(11), MovieStub(12)])

            def isStreamClosed(self):
                return False

            def getFileName(self):
                raise AssertionError("Streaming discovery must not depend on an input SQLite filename.")

        class PointerStub:
            def __init__(self, value):
                self._value = value

            def get(self):
                return self._value

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.inputMovies = PointerStub(LogicalMovieSetStub())
                self.insertedDict = {11: 101}
                self.listOfMovies = [MovieStub(11)]
                self.streamClosed = False

            def _loadLogicalInputMovies(self, watermark=None):
                return XmippProtFlexAlign._loadLogicalInputMovies(self)

            def debug(self, *args, **kwargs):
                pass

            def _getFirstJoinStep(self):
                return None

            def _insertNewMoviesSteps(self, insertedDict, inputMovies):
                deps = []
                for movie in inputMovies:
                    movieId = movie.getObjId()
                    if movieId in insertedDict:
                        continue

                    scheduledMovieIds.append(movieId)
                    stepId = 1000 + movieId
                    insertedDict[movieId] = stepId
                    deps.append(stepId)

                return deps

            def updateSteps(self):
                updateCalls.append(True)

        protocol = ProtocolStub()
        XmippProtFlexAlign._checkNewInput(protocol)
        self.assertEqual(scheduledMovieIds, [12])
        self.assertFalse(protocol.streamClosed)
        self.assertEqual(updateCalls, [True])

    def test_FlexAlignCompletionUsesPersistedOutputWithoutDoneSidecars(self):
        updateCalls = []
        joinStatuses = []

        class MovieStub:
            def __init__(self, objId):
                self._objId = objId

            def getObjId(self):
                return self._objId

        class OutputSetStub:
            def __iter__(self):
                return iter([MovieStub(11)])

        class JoinStepStub:
            def isWaiting(self):
                return True

            def setStatus(self, status):
                joinStatuses.append(status)

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.finished = False
                self.streamClosed = True
                self.listOfMovies = [MovieStub(11)]
                self.outputMovies = OutputSetStub()

            def _getPersistedOutputMovieIds(self):
                return XmippProtFlexAlign._getPersistedOutputMovieIds(self)

            def _getFinishedProcessMovieIds(self):
                return set()

            def debug(self, *args, **kwargs):
                pass

            def _readDoneList(self):
                raise AssertionError("Persisted output must be the durable completion truth, not DONE/all.TXT.")

            def _isMovieDone(self, movie):
                raise AssertionError("Persisted output must not require per-movie DONE marker files.")

            def _writeDoneList(self, movies):
                raise AssertionError("Completion reconciliation must not write DONE sidecars.")

            def _updateOutputSets(self, newDone, streamMode):
                updateCalls.append((list(newDone), streamMode))

            def _getFirstJoinStep(self):
                return JoinStepStub()

        protocol = ProtocolStub()
        XmippProtFlexAlign._checkNewOutput(protocol)
        self.assertTrue(protocol.finished)
        self.assertEqual(len(updateCalls), 1)
        self.assertEqual(updateCalls[0][0], [])
        self.assertEqual(len(joinStatuses), 1)

    def test_FlexAlignResumeUsesPersistedOutputInsteadOfMovieDoneMarker(self):
        processedMovieIds = []

        class ProtocolStub:
            def _getPersistedOutputMovieIds(self):
                return {11}

            def _getOutputMovieFolder(self, movie):
                return "/should/not/be/used"

            def _getMovieDone(self, movie):
                raise AssertionError("Resume must not depend on DONE/movie_xxx.TXT.")

            def _processMovie(self, movie):
                processedMovieIds.append(movie.getObjId())

            def isContinued(self):
                return True

        movie = Movie()
        movie.setObjId(11)
        movie.setFileName("/does/not/matter.mrc")
        protocol = ProtocolStub()

        XmippProtFlexAlign.processMovieStep(protocol, movie.getObjDict(includeBasic=True), False)

        self.assertEqual(processedMovieIds, [])


    def test_FlexAlignUsesSharedStreamingMoviesMixin(self):
        self.assertTrue(issubclass(XmippProtFlexAlign, XmippStreamingMoviesMixin))
        self.assertNotIn("_checkNewInput", XmippProtFlexAlign.__dict__)
        self.assertNotIn("_checkNewOutput", XmippProtFlexAlign.__dict__)
        self.assertNotIn("processMovieStep", XmippProtFlexAlign.__dict__)

    def test_FlexAlignUsesProtStreamingBaseLifecycle(self):
        from pyworkflow.protocol import ProtStreamingBase

        self.assertTrue(issubclass(XmippProtFlexAlign, ProtStreamingBase))
        self.assertIs(XmippProtFlexAlign._insertAllSteps, ProtStreamingBase._insertAllSteps)
        self.assertIs(XmippProtFlexAlign._stepsCheck, ProtStreamingBase._stepsCheck)

    def test_FlexAlignPrepareGeneratorDoesNotReinsertFinishedConversionStep(self):
        class FuncName:
            def get(self):
                return '_convertInputStep'

        class FinishedStep:
            funcName = FuncName()

            def isFinished(self):
                return True

        class InputMovies:
            def getSamplingRate(self):
                return 1.25

        class Pointer:
            def get(self):
                return InputMovies()

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.inputMovies = Pointer()
                self._steps = []
                self._prevSteps = [FinishedStep()]

            def _insertFunctionStep(self, *args, **kwargs):
                raise AssertionError(
                    'A persisted FINISHED _convertInputStep must not be inserted again on Continue.')

            def updateSteps(self):
                raise AssertionError(
                    'Continue must not update the step graph when conversion is already finished.')

        protocol = ProtocolStub()

        XmippProtFlexAlign._prepareStreamingGenerator(protocol)

        self.assertEqual(protocol.insertedDict, {})
        self.assertEqual(protocol.samplingRate, 1.25)
        self.assertFalse(protocol.streamClosed)
        self.assertFalse(protocol.finished)
        self.assertEqual(protocol.convertCIStep, [])

    def test_FlexAlignFinalizationIsPersistedAsStepInsteadOfRunningInsideGenerator(self):
        class ProtocolStub:
            _iterKnownStreamingSteps = XmippStreamingMoviesMixin._iterKnownStreamingSteps
            _isFunctionStepFinished = XmippStreamingMoviesMixin._isFunctionStepFinished

            def __init__(self):
                self._steps = []
                self._prevSteps = []
                self.inserted = []
                self.updateCalls = 0

            def createOutputStep(self):
                raise AssertionError(
                    'createOutputStep must not execute directly inside the streaming generator.')

            def _insertFunctionStep(self, *args, **kwargs):
                self.inserted.append((args, kwargs))
                return 71

            def updateSteps(self):
                self.updateCalls += 1

        protocol = ProtocolStub()

        XmippProtFlexAlign._finalizeStreamingGenerator(protocol)

        self.assertEqual(len(protocol.inserted), 1)
        args, kwargs = protocol.inserted[0]
        self.assertEqual(args[0], 'createOutputStep')
        self.assertEqual(kwargs.get('prerequisites', []), [])
        self.assertEqual(protocol.updateCalls, 1)

    def test_FlexAlignFinalizationDoesNotReinsertFinishedCreateOutputStepOnContinue(self):
        class FuncName:
            def get(self):
                return 'createOutputStep'

        class FinishedStep:
            funcName = FuncName()

            def isFinished(self):
                return True

        class ProtocolStub:
            def __init__(self):
                self._steps = []
                self._prevSteps = [FinishedStep()]

            def _insertFunctionStep(self, *args, **kwargs):
                raise AssertionError(
                    'A persisted FINISHED createOutputStep must not be inserted again on Continue.')

            def updateSteps(self):
                raise AssertionError(
                    'Continue must not update the graph when finalization is already finished.')

            _iterKnownStreamingSteps = XmippStreamingMoviesMixin._iterKnownStreamingSteps
            _isFunctionStepFinished = XmippStreamingMoviesMixin._isFunctionStepFinished

        protocol = ProtocolStub()

        XmippProtFlexAlign._finalizeStreamingGenerator(protocol)

    def test_FlexAlignResumeRepairCandidatesExcludeAlreadyPersistedMovies(self):
        candidateCalls = []

        class ProtocolStub:
            def _getPersistedCandidateIds(self, outputName, candidateIds):
                candidateCalls.append((outputName, set(candidateIds)))
                return {11}

        protocol = ProtocolStub()

        repairIds = XmippProtFlexAlign._getResumeRepairCandidateIds(protocol, {11, 12, 13})

        self.assertEqual(candidateCalls, [('outputMovies', {11, 12, 13})])
        self.assertEqual(repairIds, {12, 13})

    def test_FlexAlignResumeRepairCandidatesEmptyWhenNoFinishedIds(self):
        class ProtocolStub:
            def _getPersistedCandidateIds(self, outputName, candidateIds):
                raise AssertionError('Must not query candidates when there are no finished ids.')

        protocol = ProtocolStub()

        self.assertEqual(
            XmippProtFlexAlign._getResumeRepairCandidateIds(protocol, set()),
            set(),
        )

    def test_FlexAlignResumeRepairsOutputForHydratedFinishedMovieWithoutNewInputDelta(self):
        class MovieStub:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

        class OutputSet:
            def __init__(self):
                self.items = []

            def append(self, item):
                self.items.append(item)

            def __iter__(self):
                return iter(self.items)

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                movie = MovieStub(11)
                # Models the state right after _restoreFinishedStreamingMovieSteps
                # found movie 11's processMovieStep FINISHED in a prior run,
                # but missing from outputMovies - a repair candidate, not new
                # input discovery delta (listOfMovies/insertedDict already
                # reflect it, as real _checkNewInput reconciliation would).
                self.listOfMovies = [movie]
                self.streamClosed = True
                self.finished = False
                self._streamingMoviesById = {11: movie}
                self.insertedDict = {11: None}
                self._pendingInsertedIds = {11}
                self._finishedInsertedIds = {11}
                self._finishedInsertedIdsRestored = True
                self._streamingRestoredFinishedIds = {11}
                self.outputMovies = OutputSet()

            def _updateOutputSets(self, newDone, streamMode):
                for movie in newDone:
                    self.outputMovies.append(movie)

            def _getFirstJoinStep(self):
                return None

        protocol = ProtocolStub()

        XmippProtFlexAlign._checkNewOutput(protocol)

        self.assertEqual(
            {item.getObjId() for item in protocol.outputMovies},
            {11},
            'A movie whose step finished in a prior run but whose output '
            'was never persisted must be repaired (published) on '
            'Continue, even with no new input discovery delta.',
        )
        self.assertTrue(protocol.finished)

    def test_FlexAlignThreadValidationAccountsForReservedMainProcessThread(self):
        # Regression test: pyworkflow's own executor always reserves one
        # thread out of numberOfThreads for its own bookkeeping before
        # handing out the rest as execution slots (ThreadStepExecutor
        # is constructed with "nThreads - 1" in
        # pyworkflow/protocol/protocol.py). A real workflow run with one
        # GPU and 2 threads passed the OLDER "nGpus + 1" validation but
        # deadlocked at runtime: numberOfThreads=2 left only ONE real
        # execution slot (pyworkflow's own reservation), which the
        # streaming generator held forever, leaving none free for
        # processMovieStep ("Waiting 10 now before checking again for
        # new input" forever, no progress).
        class Value:
            def __init__(self, value):
                self._value = value

            def get(self):
                return self._value

        class ProtocolStub:
            gpuList = Value('0')
            numberOfThreads = Value(2)

        self.assertEqual(
            len(XmippProtFlexAlign._validateParallelProcessing(ProtocolStub())),
            1,
            'With one GPU, 2 threads must be rejected: pyworkflow reserves '
            'one for itself, leaving only one execution slot - not enough '
            'for the generator AND a worker. This is the exact '
            'configuration that deadlocked a real workflow run.',
        )

        ProtocolStub.numberOfThreads = Value(3)

        self.assertEqual(
            XmippProtFlexAlign._validateParallelProcessing(ProtocolStub()),
            [],
            'With one GPU, 3 threads is the correct minimum: pyworkflow '
            'reserves 1, leaving 2 execution slots - one for the '
            'generator, one for processMovieStep.',
        )

    def test_FlexAlignThreadValidationRequiresThreeThreadsWithZeroGpus(self):
        class Value:
            def __init__(self, value):
                self._value = value

            def get(self):
                return self._value

        class ProtocolStub:
            gpuList = Value('')
            numberOfThreads = Value(2)

        errors = XmippProtFlexAlign._validateParallelProcessing(ProtocolStub())

        self.assertEqual(
            len(errors), 1,
            'A CPU-only run (empty GPU list) still needs 3 threads - '
            'pyworkflow reserves 1 for itself, leaving 2 execution slots: '
            'one for the generator, one for processing.',
        )

        ProtocolStub.numberOfThreads = Value(3)

        self.assertEqual(
            XmippProtFlexAlign._validateParallelProcessing(ProtocolStub()),
            [],
            'Exactly 3 threads must be accepted for a CPU-only run.',
        )

    def test_MovieGainUsesSharedStreamingMoviesMixin(self):
        self.assertEqual(XmippProtMovieGain.__mro__[1].__name__, "XmippStreamingMoviesMixin")
        self.assertNotIn("_checkNewInput", XmippProtMovieGain.__dict__)
        self.assertNotIn("processMovieStep", XmippProtMovieGain.__dict__)

    def test_MovieGainPublishesFinishedMoviesWithoutDoneSidecars(self):
        updated = []

        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class FinishedStep:
            def isFinished(self):
                return True

        class OutputSet:
            STREAM_OPEN = 1
            STREAM_CLOSED = 2

            def __init__(self):
                self.items = []

            def getIdSet(self):
                return {item.getObjId() for item in self.items}

            def getSize(self):
                return len(self.items)

            def append(self, item):
                self.items.append(item)

        class ProtocolStub(XmippStreamingMoviesMixin):
            finished = False
            streamClosed = False
            insertedDict = {11: 1}
            _steps = [FinishedStep()]
            estimateGain = Param(False)
            estimateResidualGain = Param(False)

            def __init__(self):
                movie = Movie()
                movie.setObjId(11)
                self.listOfMovies = [movie]
                self.moviesSet = OutputSet()

            def _getAllDoneIds(self):
                return set()

            def _isMovieDone(self, movie):
                raise AssertionError('MovieGain output polling must not depend on DONE/movie_xxx.TXT.')

            def doGainProcess(self, movieId):
                return False

            def _loadOutputSet(self, SetClass, baseName, fixGain=False):
                return self.moviesSet

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                updated.append((outputName, list(outputSet.getIdSet()), streamMode))

            def _getFirstJoinStep(self):
                return None

        protocol = ProtocolStub()
        XmippProtMovieGain._checkNewOutput(protocol)

        self.assertEqual(protocol.moviesSet.getIdSet(), {11})
        self.assertEqual(updated[0][0], 'outputMovies')

    def test_MovieGainCreatesMovieOutputThroughProtocolFactory(self):
        class InputMovies:
            pass

        class InputPointer:
            def get(self):
                return InputMovies()

        class OutputSet:
            STREAM_OPEN = 1

            def setStreamState(self, state):
                self.streamState = state

            def copyInfo(self, inputMovies):
                self.inputMovies = inputMovies

            def setGain(self, gain):
                self.gain = gain

        class ProtocolStub(XmippStreamingMoviesMixin):
            inputMovies = InputPointer()

            def _getPath(self, *args):
                raise AssertionError('MovieGain must not build a local SQLite output path.')

            def _createSetOfMovies(self):
                self.created = OutputSet()
                return self.created

            def getFinalGainPath(self, tifFlipped=False):
                return 'gain.mrc'

        protocol = ProtocolStub()
        outputSet = XmippProtMovieGain._loadOutputSet(protocol, SetOfMovies, 'movies.sqlite', fixGain=True)

        self.assertIs(outputSet, protocol.created)
        self.assertEqual(outputSet.streamState, outputSet.STREAM_OPEN)
        self.assertEqual(outputSet.gain, 'gain.mrc')

    def test_MovieGainProcessingFailureIsNotReportedAsFinished(self):
        class ProtocolStub:
            def doGainProcess(self, movieId):
                return True

            def getInputGain(self):
                raise RuntimeError("gain processing failed")

            def error(self, message):
                pass

        movie = Movie()
        movie.setObjId(11)

        with self.assertRaisesRegex(RuntimeError, "gain processing failed"):
            XmippProtMovieGain._processMovie(ProtocolStub(), movie)

    def test_MovieGainNormalizationResumeDoesNotUseMarkerSidecar(self):
        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class FuncName:
            def get(self):
                return "normalizeGainStep"

        class FinishedStep:
            funcName = FuncName()

            def isFinished(self):
                return True

        class ProtocolStub(XmippStreamingMoviesMixin):
            estimateOrientation = Param(False)
            normalizeGain = Param(True)
            convertCIStep = []

            def __init__(self):
                self._steps = [FinishedStep()]

            def _getGainNormalizedMarker(self):
                raise AssertionError('Normalization resume must not depend on GAIN_NORMALIZED.TXT.')

            def _insertFunctionStep(self, *args, **kwargs):
                raise AssertionError('A finished persisted normalizeGainStep must not be inserted again.')

            def _restoreEstimatedIds(self, attrName, outputName):
                pass

        protocol = ProtocolStub()
        deps = XmippProtMovieGain._insertNewMoviesSteps(protocol, {}, [])

        self.assertEqual(deps, [])

    def test_MovieGainOrientationInsertionAcceptsLogicalMovieList(self):
        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class ProtocolStub(XmippStreamingMoviesMixin):
            estimateOrientation = Param(True)
            normalizeGain = Param(False)

            def __init__(self):
                self.convertCIStep = []

            def _isOutputAlreadyPublished(self, outputName):
                return False

            def _insertFunctionStep(self, functionName, *args, **kwargs):
                self.insertedFunction = functionName
                return 101

            def _restoreEstimatedIds(self, attrName, outputName):
                pass

            def _insertMovieStep(self, movie):
                return 102

        movie = Movie()
        movie.setObjId(11)
        movies = [movie]
        inserted = {}
        protocol = ProtocolStub()

        deps = XmippProtMovieGain._insertNewMoviesSteps(protocol, inserted, movies)

        self.assertEqual(protocol.insertedFunction, 'estimateOrientationStep')
        self.assertEqual(protocol.convertCIStep, [101])
        self.assertEqual(inserted, {11: 102})
        self.assertEqual(deps, [102])

    def test_MovieGainRepairsMissingGainOutputWhenMovieIsAlreadyPublished(self):
        updated = []

        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class FinishedStep:
            def isFinished(self):
                return True

        class OutputSet:
            STREAM_OPEN = 1
            STREAM_CLOSED = 2

            def __init__(self, items=None):
                self.items = list(items or [])

            def getIdSet(self):
                return {item.getObjId() for item in self.items}

            def getSize(self):
                return len(self.items)

            def append(self, item):
                self.items.append(item)

        movie = Movie()
        movie.setObjId(11)

        class ProtocolStub(XmippStreamingMoviesMixin):
            finished = False
            streamClosed = True
            insertedDict = {11: 1}
            _steps = [FinishedStep()]
            estimateGain = Param(True)
            estimateResidualGain = Param(False)
            estimatedDatabase = 'estimatedGains'
            residualDatabase = 'residualGains'

            def __init__(self):
                self.listOfMovies = [movie]
                self.outputMovies = OutputSet([movie])
                self.estimatedGains = OutputSet()

            def _getAllDoneIds(self):
                return self._getOutputIds(self.outputMovies)

            def doGainProcess(self, movieId):
                return True

            def _loadOutputSet(self, SetClass, baseName, fixGain=False):
                if baseName == self.estimatedDatabase:
                    return self.estimatedGains
                if baseName == 'outputMovies':
                    return self.outputMovies
                raise AssertionError('Unexpected output set: %s' % baseName)

            def updateGainsOutput(self, movie, outputSet, imageFile):
                outputSet.append(movie)
                return outputSet

            def getEstimatedGainPath(self, movieId):
                return 'movie_%06d_gain.xmp' % movieId

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                updated.append((outputName, set(outputSet.getIdSet()), streamMode))

            def _getFirstJoinStep(self):
                return None

        protocol = ProtocolStub()
        XmippProtMovieGain._checkNewOutput(protocol)

        estimatedUpdates = [ids for name, ids, _ in updated if name == 'estimatedGains']
        self.assertEqual(estimatedUpdates, [{11}])

    def test_StreamingBaseIncrementalReadUsesWhereAfterWatermark(self):
        movie = Movie()
        movie.setObjId(11)

        class InputSetStub:
            def __init__(self):
                self.calls = []

            def loadAllProperties(self):
                pass

            def iterItems(self, orderBy=None, direction=None, where=None):
                self.calls.append({"orderBy": orderBy, "direction": direction, "where": where})
                return iter([movie])

            def __iter__(self):
                raise AssertionError('Incremental polling must use iterItems(where=...)')

        inputSet = InputSetStub()
        streamingBase = XmippStreamingMoviesMixin.__mro__[1]

        items = streamingBase._loadLogicalSetItems(inputSet, 10)

        self.assertEqual([item.getObjId() for item in items], [11])
        self.assertEqual(inputSet.calls, [{"orderBy": "id", "direction": "ASC", "where": "id > 10"}])

    def test_StreamingMoviesPollingUsesInsertedWatermark(self):
        movie = Movie()
        movie.setObjId(12)
        calls = []

        class InputSetStub:
            def isStreamClosed(self):
                return False

        class PointerStub:
            def get(self):
                return InputSetStub()

        class ProtocolStub(XmippStreamingMoviesMixin):
            inputMovies = PointerStub()

            def __init__(self):
                self.insertedDict = {11: 1}
                self.insertedMovieIds = []
                self.stepsUpdated = False

            def _loadLogicalSetItems(self, inputSet, watermark=None):
                calls.append(watermark)
                return [movie]

            def _getFirstJoinStep(self):
                return None

            def _insertNewMoviesSteps(self, insertedDict, movies):
                self.insertedMovieIds = [item.getObjId() for item in movies]
                return [2]

            def updateSteps(self):
                self.stepsUpdated = True

        protocol = ProtocolStub()
        protocol._checkNewInput()

        self.assertEqual(calls, [11])
        self.assertEqual(protocol.insertedMovieIds, [12])
        self.assertTrue(protocol.stepsUpdated)

    def test_StreamingOutputPollingDoesNotRescanAcknowledgedFinishedSteps(self):
        class StepStub:
            def __init__(self, finished):
                self.finished = finished
                self.checks = 0

            def isFinished(self):
                self.checks += 1
                return self.finished

        firstStep = StepStub(True)
        secondStep = StepStub(False)

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = {11: 1, 12: 2}
                self._steps = [firstStep, secondStep]

        protocol = ProtocolStub()

        self.assertEqual(protocol._getFinishedInsertedIds(), {11})
        protocol._acknowledgeFinishedInsertedIds({11})

        secondStep.finished = True
        protocol._recordFinishedInsertedStep(2)
        self.assertEqual(protocol._getFinishedInsertedIds(), {12})
        self.assertEqual(firstStep.checks, 1)
        self.assertEqual(secondStep.checks, 1)

    def test_StreamingOutputCandidateLookupUsesWhere(self):
        calls = []

        def makeMovie(objId):
            movie = Movie()
            movie.setObjId(objId)
            return movie

        class OutputSetStub:
            def loadAllProperties(self):
                pass

            def iterItems(self, orderBy=None, direction=None, where=None):
                calls.append({"orderBy": orderBy, "direction": direction, "where": where})
                return iter([makeMovie(11), makeMovie(13)])

            def __iter__(self):
                raise AssertionError('Normal output polling must not scan the complete output set')

            def getIdSet(self):
                raise AssertionError('Normal output polling must not call getIdSet() for the complete output set')

        class ProtocolStub(XmippStreamingMoviesMixin):
            outputMovies = OutputSetStub()

        protocol = ProtocolStub()
        persisted = protocol._getPersistedCandidateIds('outputMovies', {13, 11})

        self.assertEqual(persisted, {11, 13})
        self.assertEqual(calls, [{"orderBy": "id", "direction": "ASC", "where": "id IN (11, 13)"}])

    def test_MovieGainAcknowledgesOnlyAfterCandidateOutputIsPersisted(self):
        candidateCalls = []
        acknowledged = []
        persistedMovieIds = set()

        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class OutputSet:
            def __init__(self):
                self.items = []

            def append(self, item):
                self.items.append(item)

            def getIdSet(self):
                return {item.getObjId() for item in self.items}

        movie = Movie()
        movie.setObjId(11)

        class ProtocolStub(XmippStreamingMoviesMixin):
            finished = False
            streamClosed = False
            estimateGain = Param(False)
            estimateResidualGain = Param(False)

            def __init__(self):
                self.listOfMovies = [movie]
                self.moviesSet = OutputSet()

            def _getFinishedProcessMovieIds(self):
                return {11}

            def _getPersistedOutputIds(self, outputName):
                raise AssertionError('Normal polling must not scan the complete output set')

            def _getPersistedCandidateIds(self, outputName, candidateIds):
                candidateIds = set(candidateIds)
                candidateCalls.append((outputName, candidateIds))
                return persistedMovieIds.intersection(candidateIds)

            def _acknowledgeFinishedInsertedIds(self, itemIds):
                acknowledged.extend(sorted(itemIds))

            def doGainProcess(self, movieId):
                return False

            def _loadOutputSet(self, SetClass, baseName, fixGain=False):
                return self.moviesSet

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                persistedMovieIds.update(outputSet.getIdSet())

            def _getFirstJoinStep(self):
                return None

        protocol = ProtocolStub()
        XmippProtMovieGain._checkNewOutput(protocol)

        self.assertEqual(candidateCalls, [('outputMovies', {11}), ('outputMovies', {11})])
        self.assertEqual(acknowledged, [11])
        self.assertEqual(persistedMovieIds, {11})

    def test_StreamingMoviesClosedInputReconcilesItemsBelowWatermark(self):
        calls = []

        def makeMovie(objId):
            movie = Movie()
            movie.setObjId(objId)
            return movie

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = {11: 1}
                self.listOfMovies = [makeMovie(11)]
                self.insertedMovieIds = []

            def _loadLogicalInputMovies(self, watermark=None):
                calls.append(watermark)
                if watermark is None:
                    return [makeMovie(10), makeMovie(11)], True
                return [], True

            def _insertNewMoviesSteps(self, insertedDict, movies):
                newIds = [movie.getObjId() for movie in movies if movie.getObjId() not in insertedDict]
                self.insertedMovieIds.extend(newIds)
                for movieId in newIds:
                    insertedDict[movieId] = movieId
                return newIds

            def _getFirstJoinStep(self):
                return None

            def updateSteps(self):
                pass

        protocol = ProtocolStub()
        protocol._checkNewInput()

        self.assertEqual(calls, [11, None])
        self.assertEqual(protocol.insertedMovieIds, [10])
        self.assertEqual({movie.getObjId() for movie in protocol.listOfMovies}, {10, 11})

    def test_MovieGainStreamingPersistenceHasNoSqliteSelectors(self):
        source = (Path(__file__).parents[1] / 'protocols/protocol_movie_gain.py').read_text(encoding='utf-8')
        forbidden = (
            'movies.sqlite',
            'estGains.sqlite',
            'resGains.sqlite',
            'orientedGain.sqlite',
        )

        for selector in forbidden:
            self.assertNotIn(selector, source)

    def test_StreamingMoviesDiscoveryUpdatesMovieLookupCache(self):
        def makeMovie(objId):
            movie = Movie()
            movie.setObjId(objId)
            return movie

        movie11 = makeMovie(11)
        movie12 = makeMovie(12)

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = {11: 1}
                self.listOfMovies = [movie11]
                self._streamingInputIds = {11}
                self._streamingMoviesById = {11: movie11}

            def _loadLogicalInputMovies(self, watermark=None):
                return [movie12], False

            def _insertNewMoviesSteps(self, insertedDict, movies):
                for movie in movies:
                    insertedDict[movie.getObjId()] = 2
                return [2]

            def _getFirstJoinStep(self):
                return None

            def updateSteps(self):
                pass

        protocol = ProtocolStub()
        protocol._checkNewInput()

        self.assertIs(protocol._streamingMoviesById[12], movie12)

    def test_StreamingMoviesPollingCachesInputWatermark(self):
        class InsertedDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError('Polling must not recompute max(insertedDict) every time')
                return super().__iter__()

        calls = []
        inserted = InsertedDict({11: 1})

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = inserted
                self.listOfMovies = []
                self._streamingInputIds = {11}
                self._streamingMoviesById = {}

            def _loadLogicalInputMovies(self, watermark=None):
                calls.append(watermark)
                return [], False

            def _insertNewMoviesSteps(self, insertedDict, movies):
                return []

            def _getFirstJoinStep(self):
                return None

            def updateSteps(self):
                pass

        protocol = ProtocolStub()
        protocol._checkNewInput()
        protocol._checkNewInput()

        self.assertEqual(calls, [11, 11])
        self.assertEqual(inserted.iterations, 1)

    def test_StreamingMoviesWatermarkAdvancesFromDiscoveryWithoutInsertedStep(self):
        calls = []

        def makeMovie(objId):
            movie = Movie()
            movie.setObjId(objId)
            return movie

        movie11 = makeMovie(11)
        movie12 = makeMovie(12)

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = {11: 1}
                self.listOfMovies = [movie11]
                self._streamingInputIds = {11}
                self._streamingMoviesById = {11: movie11}
                self._streamingInputWatermark = 11

            def _loadLogicalInputMovies(self, watermark=None):
                calls.append(watermark)
                return ([movie12], False) if watermark == 11 else ([], False)

            def _insertNewMoviesSteps(self, insertedDict, movies):
                return []

            def _getFirstJoinStep(self):
                return None

            def updateSteps(self):
                pass

        protocol = ProtocolStub()
        protocol._checkNewInput()
        protocol._checkNewInput()

        self.assertEqual(calls, [11, 12])
        self.assertEqual(protocol._streamingInputWatermark, 12)

    def test_StreamingMoviesClosedInputReconcilesOnlyOnce(self):
        calls = []

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = {11: 1}
                self.listOfMovies = []
                self._streamingInputIds = {11}
                self._streamingMoviesById = {}
                self._streamingInputWatermark = 11

            def _loadLogicalInputMovies(self, watermark=None):
                calls.append(watermark)
                return [], True

            def _insertNewMoviesSteps(self, insertedDict, movies):
                return []

            def _getFirstJoinStep(self):
                return None

            def updateSteps(self):
                pass

        protocol = ProtocolStub()
        protocol._checkNewInput()
        protocol._checkNewInput()

        self.assertEqual(calls, [11, None, 11])

    def test_StreamingOutputPollingCachesPendingInsertedIds(self):
        class InsertedDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError('Normal output polling must not rescan the complete insertedDict')
                return super().__iter__()

        class StepStub:
            def __init__(self, finished):
                self.finished = finished

            def isFinished(self):
                return self.finished

        firstStep = StepStub(True)
        secondStep = StepStub(False)
        inserted = InsertedDict({11: 1, 12: 2})

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = inserted
                self._steps = [firstStep, secondStep]

        protocol = ProtocolStub()

        self.assertEqual(protocol._getFinishedInsertedIds(), {11})
        protocol._acknowledgeFinishedInsertedIds({11})

        secondStep.finished = True
        protocol._recordFinishedInsertedStep(2)
        self.assertEqual(protocol._getFinishedInsertedIds(), {12})
        self.assertEqual(inserted.iterations, 1)

    def test_MovieGainClosedStreamKeepsDeltaPollingWhileStepsArePending(self):
        acknowledged = []
        persistedMovieIds = set()

        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class OutputSet:
            def __init__(self):
                self.items = []

            def append(self, item):
                self.items.append(item)

            def getIdSet(self):
                return {item.getObjId() for item in self.items}

        movie11 = Movie()
        movie11.setObjId(11)
        movie12 = Movie()
        movie12.setObjId(12)

        class ProtocolStub(XmippStreamingMoviesMixin):
            finished = False
            streamClosed = True
            estimateGain = Param(False)
            estimateResidualGain = Param(False)

            def __init__(self):
                self.listOfMovies = [movie11, movie12]
                self._pendingInsertedIds = {11, 12}
                self.moviesSet = OutputSet()

            def _getFinishedProcessMovieIds(self):
                return {11}

            def _getAllFinishedInsertedIds(self):
                raise AssertionError('Closed streaming must not rescan every step while pending work remains')

            def _getPersistedCandidateIds(self, outputName, candidateIds):
                return persistedMovieIds.intersection(candidateIds)

            def _acknowledgeFinishedInsertedIds(self, itemIds):
                itemIds = set(itemIds)
                acknowledged.extend(sorted(itemIds))
                self._pendingInsertedIds.difference_update(itemIds)

            def doGainProcess(self, movieId):
                return False

            def _loadOutputSet(self, SetClass, outputName, fixGain=False):
                return self.moviesSet

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                persistedMovieIds.update(outputSet.getIdSet())

            def _getFirstJoinStep(self):
                return None

        protocol = ProtocolStub()
        XmippProtMovieGain._checkNewOutput(protocol)

        self.assertEqual(acknowledged, [11])
        self.assertEqual(protocol._pendingInsertedIds, {12})
        self.assertEqual(persistedMovieIds, {11})
        self.assertFalse(protocol.finished)

    def test_StreamingOutputCompletionEventAvoidsPendingStepScan(self):
        class StepStub:
            def isFinished(self):
                raise AssertionError('Normal polling must not scan pending step statuses after initialization')

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.insertedDict = {11: 1, 12: 2}
                self._steps = [StepStub(), StepStub()]
                self._pendingInsertedIds = {11, 12}
                self._streamingStepToItemId = {1: 11, 2: 12}
                self._finishedInsertedIds = set()
                self._finishedInsertedIdsRestored = True

        protocol = ProtocolStub()
        protocol._recordFinishedInsertedStep(2)

        self.assertEqual(protocol._getFinishedInsertedIds(), {12})

    def test_MovieGainUsesProtStreamingBaseLifecycle(self):
        from pyworkflow.protocol import ProtStreamingBase

        self.assertTrue(issubclass(XmippProtMovieGain, ProtStreamingBase))
        self.assertIs(XmippProtMovieGain._insertAllSteps, ProtStreamingBase._insertAllSteps)
        self.assertIs(XmippProtMovieGain._stepsCheck, ProtStreamingBase._stepsCheck)


    def test_StreamingMoviesGeneratorDrivesInputAndOutput(self):
        from pyworkflow.protocol import ProtStreamingBase

        class ProtocolStub(XmippStreamingMoviesMixin, ProtStreamingBase):
            def __init__(self):
                self.finished = False
                self.calls = []

            def _checkNewInput(self):
                self.calls.append('input')

            def _checkNewOutput(self):
                self.calls.append('output')
                self.finished = True

            def _getStreamingSleepOnWait(self):
                return 0

        protocol = ProtocolStub()
        protocol.stepsGeneratorStep()

        self.assertEqual(protocol.calls, ['input', 'output'])


    def test_StreamingMoviesGeneratorDoesNotUseLegacyPolling(self):
        from pyworkflow.protocol import ProtStreamingBase
        from pwem.protocols import ProtProcessMovies

        self.assertIsNot(XmippProtMovieGain._stepsCheck, ProtProcessMovies._stepsCheck)
        self.assertIs(XmippProtMovieGain._stepsCheck, ProtStreamingBase._stepsCheck)

    def test_StreamingMoviesResumeRestoresFinishedProcessStepForOutputRepair(self):
        class FuncName:
            def get(self):
                return 'processMovieStep'

        class ArgsStr:
            def get(self, default=None):
                return '[{"object.id": 11}, false]'

        class FinishedStep:
            funcName = FuncName()
            argsStr = ArgsStr()

            def isFinished(self):
                return True

        protocol = object.__new__(XmippStreamingMoviesMixin)
        protocol._steps = []
        protocol._prevSteps = [FinishedStep()]
        protocol._acknowledgedFinishedIds = set()

        protocol._restoreFinishedStreamingMovieSteps()

        self.assertEqual(protocol._streamingRestoredFinishedIds, {11})
        self.assertEqual(protocol._finishedInsertedIds, {11})
        self.assertEqual(protocol._streamingInputWatermark, 11)

    def test_StreamingMoviesResumeHydratesFinishedMovieForOutputRepairWithoutFullInputScan(self):
        class FuncName:
            def get(self):
                return 'processMovieStep'

        class ArgsStr:
            def get(self, default=None):
                return '[{"object.id": 11}, false]'

        class FinishedStep:
            funcName = FuncName()
            argsStr = ArgsStr()

            def isFinished(self):
                return True

        class MovieStub:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

            def clone(self):
                return MovieStub(self.movieId)

        class LogicalInput:
            def __init__(self):
                self.queries = []

            def loadAllProperties(self):
                pass

            def iterItems(self, orderBy='id', direction='ASC', where=None):
                self.queries.append(where)
                if where != 'id IN (11)':
                    raise AssertionError('Resume repair must hydrate only finished candidate ids, got %r' % where)
                return iter([MovieStub(11)])

        class Pointer:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        logicalInput = LogicalInput()
        protocol = object.__new__(XmippStreamingMoviesMixin)
        protocol.inputMovies = Pointer(logicalInput)
        protocol._steps = []
        protocol._prevSteps = [FinishedStep()]
        protocol._acknowledgedFinishedIds = set()

        protocol._restoreFinishedStreamingMovieSteps()

        self.assertEqual(set(protocol._streamingMoviesById), {11})
        self.assertEqual(protocol._streamingMoviesById[11].getObjId(), 11)
        self.assertEqual(logicalInput.queries, ['id IN (11)'])

    def test_StreamingMoviesResumeFinishedMovieIsNotRescheduledAndRemainsOutputCandidate(self):
        class FuncName:
            def get(self):
                return 'processMovieStep'

        class ArgsStr:
            def get(self, default=None):
                return '[{"object.id": 11}, false]'

        class FinishedStep:
            funcName = FuncName()
            argsStr = ArgsStr()

            def isFinished(self):
                return True

        class MovieStub:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

            def clone(self):
                return MovieStub(self.movieId)

        class LogicalInput:
            def loadAllProperties(self):
                pass

            def iterItems(self, orderBy='id', direction='ASC', where=None):
                if where == 'id IN (11)':
                    return iter([MovieStub(11)])
                if where == 'id > 11':
                    return iter([])
                raise AssertionError('Unexpected resume input query: %r' % where)

            def isStreamClosed(self):
                return False

        class Pointer:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.inputMovies = Pointer(LogicalInput())
                self._steps = []
                self._prevSteps = [FinishedStep()]
                self._acknowledgedFinishedIds = set()
                self.inserted = []

            def _insertNewMoviesSteps(self, insertedDict, movies):
                movies = list(movies)
                self.inserted.extend(movie.getObjId() for movie in movies)
                if 11 in self.inserted:
                    raise AssertionError('Finished movie 11 must not be scheduled again after Continue.')
                return []

            def updateSteps(self):
                pass

        protocol = ProtocolStub()
        protocol._restoreFinishedStreamingMovieSteps()
        protocol._checkNewInput()

        self.assertEqual(protocol.inserted, [])
        self.assertEqual(protocol._getFinishedProcessMovieIds(), {11})
        self.assertEqual(protocol._streamingMoviesById[11].getObjId(), 11)

    def test_StreamingMoviesResumeSkipsHydrationWhenFinishedOutputIsAlreadyPersisted(self):
        class FuncName:
            def get(self):
                return 'processMovieStep'

        class ArgsStr:
            def get(self, default=None):
                return '[{"object.id": 11}, false]'

        class FinishedStep:
            funcName = FuncName()
            argsStr = ArgsStr()

            def isFinished(self):
                return True

        class LogicalInput:
            def loadAllProperties(self):
                raise AssertionError('A fully published finished movie must not hydrate its input row on Continue.')

            def iterItems(self, *args, **kwargs):
                raise AssertionError('A fully published finished movie must not query its input row on Continue.')

        class Pointer:
            def get(self):
                return LogicalInput()

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.inputMovies = Pointer()
                self._steps = []
                self._prevSteps = [FinishedStep()]
                self._acknowledgedFinishedIds = set()
                self.filteredIds = None

            def _getResumeRepairCandidateIds(self, finishedIds):
                self.filteredIds = set(finishedIds)
                return set()

        protocol = ProtocolStub()
        protocol._restoreFinishedStreamingMovieSteps()

        self.assertEqual(protocol.filteredIds, {11})
        self.assertIn(11, protocol.insertedDict)
        self.assertEqual(protocol._streamingRestoredFinishedIds, {11})
        self.assertEqual(protocol._pendingInsertedIds, set())
        self.assertEqual(protocol._finishedInsertedIds, set())
        self.assertEqual(protocol._acknowledgedFinishedIds, {11})
        self.assertEqual(protocol._streamingMoviesById, {})

    def test_MovieGainResumeRepairsOnlyFinishedIdsMissingMandatoryPersistedOutputs(self):
        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class ProtocolStub:
            estimateGain = Param(True)
            estimateResidualGain = Param(True)

            def __init__(self):
                self.calls = []
                self.persisted = {
                    'outputMovies': {11, 12, 13, 14},
                    'estimatedGains': {11, 13, 15},
                    'residualGains': {11, 12, 15},
                }

            def doGainProcess(self, movieId):
                return movieId != 14

            def _getPersistedCandidateIds(self, outputName, candidateIds):
                candidateIds = set(candidateIds)
                self.calls.append((outputName, candidateIds))
                return self.persisted.get(outputName, set()) & candidateIds

            def _getPersistedOutputIds(self, *args, **kwargs):
                raise AssertionError('Resume repair filtering must use candidate-only output lookups.')

        protocol = ProtocolStub()
        finishedIds = {11, 12, 13, 14, 15}

        repairIds = XmippProtMovieGain._getResumeRepairCandidateIds(protocol, finishedIds)

        self.assertEqual(repairIds, {12, 13, 15})
        self.assertEqual(
            protocol.calls,
            [
                ('outputMovies', finishedIds),
                ('estimatedGains', {11, 12, 13, 15}),
                ('residualGains', {11, 12, 13, 15}),
            ],
        )

    def test_MovieGainPrepareGeneratorDoesNotReinsertFinishedConversionStep(self):
        class FuncName:
            def get(self):
                return '_convertInputStep'

        class FinishedStep:
            funcName = FuncName()

            def isFinished(self):
                return True

        class InputMovies:
            def getSamplingRate(self):
                return 1.25

        class Pointer:
            def get(self):
                return InputMovies()

        class ProtocolStub(XmippStreamingMoviesMixin):
            def __init__(self):
                self.inputMovies = Pointer()
                self._steps = []
                self._prevSteps = [FinishedStep()]

            def _insertFunctionStep(self, *args, **kwargs):
                raise AssertionError(
                    'A persisted FINISHED _convertInputStep must not be inserted again on Continue.')

            def updateSteps(self):
                raise AssertionError(
                    'Continue must not update the step graph when conversion is already finished.')

        protocol = ProtocolStub()

        XmippProtMovieGain._prepareStreamingGenerator(protocol)

        self.assertEqual(protocol.insertedDict, {})
        self.assertEqual(protocol.samplingRate, 1.25)
        self.assertFalse(protocol.streamClosed)
        self.assertFalse(protocol.finished)
        self.assertEqual(protocol.convertCIStep, [])

    def test_StreamingMoviesMixinDoesNotImplicitlyMigrateEveryProtocolToProtStreamingBase(self):
        from pyworkflow.protocol import ProtStreamingBase
        self.assertNotIn(
            ProtStreamingBase,
            XmippStreamingMoviesMixin.__mro__[1:],
            'The shared movies mixin must remain lifecycle-neutral; concrete protocols must opt in explicitly.',
        )
        self.assertIn(
            ProtStreamingBase,
            XmippProtMovieGain.__mro__,
            'MovieGain must explicitly use ProtStreamingBase after its migration.',
        )
        self.assertIs(
            XmippProtMovieGain._insertAllSteps,
            ProtStreamingBase._insertAllSteps,
        )
        self.assertIn(
            ProtStreamingBase,
            XmippProtFlexAlign.__mro__,
            'FlexAlign has since been migrated to the full ProtStreamingBase '
            'lifecycle, same as MovieGain - this is no longer the "shares '
            'the helper mixin but kept the old lifecycle" example.',
        )

        # The general invariant ("using the mixin alone must not implicitly
        # pull in ProtStreamingBase") no longer has a real protocol example
        # to anchor on now that FlexAlign opted in too, so it is verified
        # directly against a minimal stub instead.
        from pyworkflow.protocol import Protocol

        class _MixinOnlyProtocolStub(XmippStreamingMoviesMixin, Protocol):
            pass

        self.assertIsNot(
            _MixinOnlyProtocolStub._insertAllSteps,
            ProtStreamingBase._insertAllSteps,
            'A protocol must not get ProtStreamingBase._insertAllSteps '
            'merely by using the shared movies mixin - it must inherit '
            'ProtStreamingBase explicitly.',
        )

    def test_MovieGainFinalizationIsPersistedAsStepInsteadOfRunningInsideGenerator(self):
        class ProtocolStub:
            _iterKnownStreamingSteps = XmippStreamingMoviesMixin._iterKnownStreamingSteps
            _isFunctionStepFinished = XmippStreamingMoviesMixin._isFunctionStepFinished

            def __init__(self):
                self._steps = []
                self._prevSteps = []
                self.inserted = []
                self.updateCalls = 0

            def createOutputStep(self):
                raise AssertionError(
                    'createOutputStep must not execute directly inside the streaming generator.')

            def _insertFunctionStep(self, *args, **kwargs):
                self.inserted.append((args, kwargs))
                return 71

            def updateSteps(self):
                self.updateCalls += 1

        protocol = ProtocolStub()

        XmippProtMovieGain._finalizeStreamingGenerator(protocol)

        self.assertEqual(len(protocol.inserted), 1)
        args, kwargs = protocol.inserted[0]
        self.assertEqual(args[0], 'createOutputStep')
        self.assertEqual(kwargs.get('prerequisites', []), [])
        self.assertEqual(protocol.updateCalls, 1)

    def test_MovieGainFinalizationDoesNotReinsertFinishedCreateOutputStepOnContinue(self):
        class FuncName:
            def get(self):
                return 'createOutputStep'

        class FinishedStep:
            funcName = FuncName()

            def isFinished(self):
                return True

        class ProtocolStub:
            def __init__(self):
                self._steps = []
                self._prevSteps = [FinishedStep()]

            def _insertFunctionStep(self, *args, **kwargs):
                raise AssertionError(
                    'A persisted FINISHED createOutputStep must not be inserted again on Continue.')

            def updateSteps(self):
                raise AssertionError(
                    'Continue must not update the graph when finalization is already finished.')

            _iterKnownStreamingSteps = XmippStreamingMoviesMixin._iterKnownStreamingSteps
            _isFunctionStepFinished = XmippStreamingMoviesMixin._isFunctionStepFinished

        protocol = ProtocolStub()

        XmippProtMovieGain._finalizeStreamingGenerator(protocol)

    def test_MovieGainFinalizationReinsertsUnfinishedCreateOutputStepOnContinue(self):
        class FuncName:
            def get(self):
                return 'createOutputStep'

        class UnfinishedStep:
            funcName = FuncName()

            def isFinished(self):
                return False

        class ProtocolStub:
            def __init__(self):
                self._steps = []
                self._prevSteps = [UnfinishedStep()]
                self.inserted = []
                self.updateCalls = 0

            def _insertFunctionStep(self, *args, **kwargs):
                self.inserted.append((args, kwargs))
                return 81

            def updateSteps(self):
                self.updateCalls += 1

            _iterKnownStreamingSteps = XmippStreamingMoviesMixin._iterKnownStreamingSteps
            _isFunctionStepFinished = XmippStreamingMoviesMixin._isFunctionStepFinished

        protocol = ProtocolStub()

        XmippProtMovieGain._finalizeStreamingGenerator(protocol)

        self.assertEqual(len(protocol.inserted), 1)
        args, kwargs = protocol.inserted[0]
        self.assertEqual(args[0], 'createOutputStep')
        self.assertEqual(kwargs.get('prerequisites', []), [])
        self.assertEqual(protocol.updateCalls, 1)

    def test_MovieGainResumePublishesHydratedRepairCandidateWithoutNewInputDelta(self):
        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class MovieStub:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

            def clone(self):
                return MovieStub(self.movieId)

        class OutputSet:
            def __init__(self):
                self.items = []

            def append(self, item):
                self.items.append(item)

        class ProtocolStub:
            estimateGain = Param(False)
            estimateResidualGain = Param(False)

            def __init__(self):
                # This models Continue after the input watermark has already
                # advanced past movie 11: there is no new input delta now.
                self.listOfMovies = []
                self._streamingMoviesById = {11: MovieStub(11)}
                self.streamClosed = False
                self.finished = False
                self.outputSet = OutputSet()
                self.updatedMovieIds = []
                self.acknowledgedIds = set()

            def _getAllDoneIds(self):
                return set()

            def _getFinishedProcessMovieIds(self):
                return {11}

            def _getPersistedCandidateIds(self, outputName, candidateIds):
                persistedIds = {item.getObjId() for item in self.outputSet.items}
                return persistedIds.intersection(candidateIds)

            def _acknowledgeFinishedInsertedIds(self, itemIds):
                self.acknowledgedIds.update(itemIds)

            def doGainProcess(self, movieId):
                return False

            def _loadOutputSet(self, SetClass, baseName, fixGain=False):
                return self.outputSet

            def _getOutputIds(self, outputSet):
                return {item.getObjId() for item in outputSet.items}

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                self.updatedMovieIds = [item.getObjId() for item in outputSet.items]

            def _getFirstJoinStep(self):
                return None

        protocol = ProtocolStub()

        XmippProtMovieGain._checkNewOutput(protocol)

        self.assertEqual(
            protocol.updatedMovieIds,
            [11],
            'A finished movie hydrated during Continue must be published even when the current input delta is empty.',
        )
        self.assertEqual(
            protocol.acknowledgedIds,
            {11},
            'The restored candidate must only be acknowledged after its output is persisted.',
        )

    def test_MovieGainResumeCompletionUsesKnownStreamingCandidatesNotCurrentInputDelta(self):
        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class MovieStub:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

            def clone(self):
                return MovieStub(self.movieId)

        class OutputSet:
            def __init__(self):
                self.items = []

            def append(self, item):
                self.items.append(item)

        class ProtocolStub:
            estimateGain = Param(False)
            estimateResidualGain = Param(False)

            def __init__(self):
                # Continue after discovery already advanced beyond movie 11.
                # The producer is now closed and there is no new input delta.
                self.listOfMovies = []
                self._streamingMoviesById = {11: MovieStub(11)}
                self.streamClosed = True
                self.finished = False
                self.outputSet = OutputSet()
                self.acknowledgedIds = set()
                self._pendingInsertedIds = set()

            def _getAllDoneIds(self):
                return set()

            def _getFinishedProcessMovieIds(self):
                return {11}

            def _getPersistedCandidateIds(self, outputName, candidateIds):
                persistedIds = {item.getObjId() for item in self.outputSet.items}
                return persistedIds.intersection(candidateIds)

            def _acknowledgeFinishedInsertedIds(self, itemIds):
                self.acknowledgedIds.update(itemIds)
                self._pendingInsertedIds.difference_update(itemIds)

            def _getAllFinishedInsertedIds(self):
                return {11}

            def _getPersistedOutputIds(self, outputName):
                return {item.getObjId() for item in self.outputSet.items}

            def doGainProcess(self, movieId):
                return False

            def _loadOutputSet(self, SetClass, baseName, fixGain=False):
                return self.outputSet

            def _getOutputIds(self, outputSet):
                return {item.getObjId() for item in outputSet.items}

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                pass

            def _getFirstJoinStep(self):
                return None

        protocol = ProtocolStub()

        XmippProtMovieGain._checkNewOutput(protocol)

        self.assertEqual(
            {item.getObjId() for item in protocol.outputSet.items},
            {11},
        )
        self.assertEqual(protocol.acknowledgedIds, {11})
        self.assertTrue(
            protocol.finished,
            'A closed stream must finish after the hydrated historical candidate is durably published; completion cannot depend on len(listOfMovies) for the current delta.',
        )

    def test_MovieGainNormalizeGainStepDoesNotWriteStateMarker(self):
        from unittest.mock import patch

        class ImageStub:
            def __init__(self):
                self.data = [1.0, 1.0]

            def read(self, filename):
                pass

            def getData(self):
                return self.data

            def setData(self, data):
                self.data = data

            def write(self, filename):
                pass

        class ProtocolStub:
            def getFinalGainPath(self):
                return 'gain.xmp'

            def _getGainNormalizedMarker(self):
                raise AssertionError(
                    'Normalization completion must come from persisted step state, not a filesystem marker.')

        protocol = ProtocolStub()

        with patch(
            'xmipp3.protocols.protocol_movie_gain.emlib.Image',
            return_value=ImageStub(),
        ), patch(
            'builtins.open',
            side_effect=AssertionError(
                'normalizeGainStep must not write a local completion sidecar.'
            ),
        ):
            XmippProtMovieGain.normalizeGainStep(protocol)

    def test_MovieGainTerminalRepairUsesMovieLookupInsteadOfRescanningInput(self):
        class Param:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class MovieStub:
            def __init__(self, movieId):
                self.movieId = movieId

            def getObjId(self):
                return self.movieId

        class SinglePassMovies:
            def __init__(self, movies):
                self.movies = list(movies)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations > 1:
                    raise AssertionError(
                        'Terminal repair must use _streamingMoviesById instead of rescanning the full input.'
                    )
                return iter(self.movies)

        class OutputSet:
            def __init__(self):
                self.items = []

            def append(self, item):
                self.items.append(item)

        movie11 = MovieStub(11)
        movie12 = MovieStub(12)

        class ProtocolStub:
            estimateGain = Param(False)
            estimateResidualGain = Param(False)

            def __init__(self):
                self.listOfMovies = SinglePassMovies([movie11, movie12])
                self._streamingMoviesById = {11: movie11, 12: movie12}
                self.streamClosed = True
                self.finished = False
                self._pendingInsertedIds = set()
                self.outputSet = OutputSet()
                self.updatedIds = []

            def _getFinishedProcessMovieIds(self):
                return set()

            def _getAllFinishedInsertedIds(self):
                return {11}

            def _getPersistedOutputIds(self, outputName):
                return set()

            def doGainProcess(self, movieId):
                return False

            def _loadOutputSet(self, SetClass, baseName, fixGain=False):
                return self.outputSet

            def _getOutputIds(self, outputSet):
                return {item.getObjId() for item in outputSet.items}

            def _updateOutputSet(self, outputName, outputSet, streamMode):
                self.updatedIds = [item.getObjId() for item in outputSet.items]

        protocol = ProtocolStub()

        XmippProtMovieGain._checkNewOutput(protocol)

        self.assertEqual(protocol.listOfMovies.iterations, 1)
        self.assertEqual(protocol.updatedIds, [11])
        self.assertFalse(protocol.finished)


    def test_StreamingMoviesRestartDoesNotRestorePreviousMovieSteps(self):
        # Restart must ignore processMovieStep state from the previous run.
        from pyworkflow.protocol.constants import MODE_RESTART
        from xmipp3.protocols.protocol_streaming_base import (
            XmippStreamingMoviesMixin,
        )

        class _FinishedStep:
            funcName = 'processMovieStep'
            argsStr = '[{"object.id": 11}]'

            def isFinished(self):
                return True

        class _ProtocolStub:
            def __init__(self):
                self._originalRunMode = MODE_RESTART
                self.insertedDict = {}

            def getRunMode(self):
                return MODE_RESTART

            def _iterKnownStreamingSteps(self):
                return iter([_FinishedStep()])

            def _getResumeRepairCandidateIds(self, finishedIds):
                return set()

        protocol = _ProtocolStub()

        XmippStreamingMoviesMixin._restoreFinishedStreamingMovieSteps(
            protocol
        )

        self.assertEqual(
            {},
            protocol.insertedDict,
            'Restart must not repopulate insertedDict from old movie steps.',
        )
        self.assertFalse(
            hasattr(protocol, '_streamingInputWatermark'),
            'Restart must rediscover the logical input from the beginning.',
        )

