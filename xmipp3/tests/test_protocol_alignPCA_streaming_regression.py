import threading
import unittest
from unittest.mock import patch

from pyworkflow.object import Set

from xmipp3.protocols.protocol_alignPCA_2D import XmippProtClassifyPcaStreaming
import xmipp3.protocols.protocol_alignPCA_2D as alignpca_module


class _Value:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Particle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _Particle(self._objId)


class _Batch:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def __len__(self):
        return len(self.items)

    def ids(self):
        return [item.getObjId() for item in self.items]


class _InputSet:
    def __init__(self):
        self.iterLimits = []
        self.iterCalls = 0

    def getStreamState(self):
        return Set.STREAM_OPEN

    def iterItems(self, orderBy='id', direction='ASC', where=None, limit=None):
        self.iterCalls += 1
        self.iterLimits.append(limit)

        if self.iterCalls == 1:
            return iter([_Particle(1), _Particle(2)])

        return iter([])

    def close(self):
        pass


class _Pointer:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Harness:
    def __init__(self):
        self.classificationBatch = _Value(2)
        self.inputSet = _InputSet()
        self.inputParticles = _Pointer(self.inputSet)
        self._lock = threading.Lock()
        self._originalRunMode = "normal-run"
        self.sleepCalls = 0
        self.scheduledBatches = []

    def _initialStep(self):
        self.finish = False
        self.lastInputId = 0
        self.lastInputIdProcessed = 0
        self._seenParticleIds = set()
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False

        # Simulate a previous classification round still in flight when
        # this generator starts accumulating the next batch.
        self.classificationLaunch = True

    def _loadEmptyParticleSet(self):
        return _Batch()

    def getRunMode(self):
        return "normal-run"

    def _streamingMustStop(self):
        return False

    def _loadInputParticleSet(self):
        return self.inputSet

    def _doClassification(self, particles):
        return XmippProtClassifyPcaStreaming._doClassification(self, particles)

    def _insertClassificationSteps(self, particles, lastInputId):
        self.scheduledBatches.append((particles.ids(), lastInputId))
        self.classificationLaunch = True
        # The regression only needs to observe the pending batch being
        # scheduled once the previous round becomes available.
        self.finish = True

    def info(self, _message):
        pass


class TestAlignPcaStreamingBatchSchedulingRegression(unittest.TestCase):

    def testFullPendingBatchLaunchesWithoutZeroLimitInputQuery(self):
        protocol = _Harness()

        def fakeSleep(_seconds):
            protocol.sleepCalls += 1
            if protocol.sleepCalls == 1:
                # Previous classification/output publication completes.
                protocol.classificationLaunch = False
            elif protocol.sleepCalls >= 2:
                # Safety valve for the current buggy implementation:
                # without a fresh input particle it otherwise keeps polling.
                protocol.finish = True

        with patch.object(alignpca_module.time, "sleep", side_effect=fakeSleep):
            XmippProtClassifyPcaStreaming.stepsGeneratorStep(protocol)

        self.assertNotIn(
            0,
            protocol.inputSet.iterLimits,
            "A full pending batch must not call iterItems(limit=0): "
            "Classic SQLite interprets a falsy limit as no LIMIT.",
        )
        self.assertEqual(
            protocol.scheduledBatches,
            [([1, 2], 2)],
            "A batch that became full while another classification was "
            "running must launch as soon as that slot becomes free, even "
            "if no new particle arrives.",
        )


if __name__ == "__main__":
    unittest.main()
class _MetadataHarness:
    def info(self, _message):
        pass

    def error(self, _message):
        pass


class _MetadataParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _MetadataRow:
    def __init__(self, itemId):
        self._itemId = itemId

    def get(self, key, default=None):
        from xmipp3.protocols.protocol_alignPCA_2D import XMIPPCOLUMNS

        if key == XMIPPCOLUMNS.itemId.value:
            return self._itemId
        return default


class TestAlignPcaStreamingMetadataIntegrityRegression(unittest.TestCase):

    def testMissingClassificationRowFailsInsteadOfDroppingParticle(self):
        protocol = _MetadataHarness()
        particle = _MetadataParticle(42)

        with self.assertRaisesRegex(
            RuntimeError,
            "42",
        ):
            XmippProtClassifyPcaStreaming._updateParticle(
                protocol,
                particle,
                None,
            )

    def testMismatchedClassificationRowFailsInsteadOfDroppingParticle(self):
        protocol = _MetadataHarness()
        particle = _MetadataParticle(42)
        row = _MetadataRow(43)

        with self.assertRaisesRegex(
            RuntimeError,
            "42",
        ):
            XmippProtClassifyPcaStreaming._updateParticle(
                protocol,
                particle,
                row,
            )
class _PersistedClasses:
    def __init__(self, size):
        self._size = size
        self.loadCalls = 0

    def loadAllProperties(self):
        self.loadCalls += 1

    def __len__(self):
        return self._size


class _ResumeClassCountHarness:
    CREATE_CLASSES = XmippProtClassifyPcaStreaming.CREATE_CLASSES
    UPDATE_CLASSES = XmippProtClassifyPcaStreaming.UPDATE_CLASSES

    def __init__(self, persistedClassCount):
        self.mode = _Value(self.CREATE_CLASSES)
        self.outputClasses = _PersistedClasses(persistedClassCount)

        # Simulate _initFnStep(): numberClasses is reset from the original
        # requested numberOfClasses before Continue restores streaming state.
        self.numberClasses = 50

        self.lastInputId = 0
        self.classificationRound = 0
        self.classificationStarted = False
        self.firstTimeDone = False
        self.lastInputIdProcessed = 0

    def _restoreStreamingStateFromSteps(self):
        return 200, 1


class TestAlignPcaStreamingResumeClassCountRegression(unittest.TestCase):

    def testCreateClassesResumeRestoresPersistedClassCount(self):
        protocol = _ResumeClassCountHarness(persistedClassCount=37)

        XmippProtClassifyPcaStreaming._updateVarsToContinue(protocol)

        self.assertEqual(protocol.lastInputId, 200)
        self.assertEqual(protocol.classificationRound, 1)
        self.assertTrue(protocol.classificationStarted)
        self.assertEqual(
            protocol.numberClasses,
            37,
            "Continue must use the durable number of classes already "
            "published, not the originally requested numberOfClasses.",
        )
class _LateVisibleInputSet:
    def __init__(self):
        self.phase = 0
        self.iterCalls = []

    def loadAllProperties(self):
        pass

    def getStreamState(self):
        return Set.STREAM_OPEN if self.phase == 0 else Set.STREAM_CLOSED

    def getSize(self):
        return 3

    def getUniqueValues(self, column, where=None):
        assert column == "id"

        visible = [1, 3] if self.phase == 0 else [1, 2, 3]
        if where is None:
            return visible

        watermark = int(where.split(">")[1].strip())
        return [itemId for itemId in visible if itemId > watermark]

    def iterItems(self, orderBy="id", direction="ASC", where=None, limit=None):
        visible = [1, 3] if self.phase == 0 else [1, 2, 3]

        if where is not None:
            if " IN " in where:
                idsText = where.split("(", 1)[1].rsplit(")", 1)[0]
                wanted = {
                    int(value.strip())
                    for value in idsText.split(",")
                    if value.strip()
                }
                visible = [
                    itemId for itemId in visible
                    if itemId in wanted
                ]
            elif ">" in where:
                watermark = int(where.split(">", 1)[1].strip())
                visible = [
                    itemId for itemId in visible
                    if itemId > watermark
                ]
            else:
                raise AssertionError(
                    "Unexpected test WHERE clause: %s" % where
                )

        if limit is not None:
            visible = visible[:limit]

        self.iterCalls.append((self.phase, where, limit, tuple(visible)))
        return iter(_Particle(itemId) for itemId in visible)

    def getItem(self, field, itemId):
        assert field == "id"
        visible = {1, 3} if self.phase == 0 else {1, 2, 3}
        return _Particle(itemId) if itemId in visible else None

    def close(self):
        pass


class _LateVisibleHarness:
    def __init__(self):
        self.classificationBatch = _Value(10)
        self.inputSet = _LateVisibleInputSet()
        self.inputParticles = _Pointer(self.inputSet)
        self._lock = threading.Lock()
        self._originalRunMode = "normal-run"
        self.scheduledBatches = []
        self.closeInserted = False

    def _initialStep(self):
        self.finish = False
        self.lastInputId = 0
        self.lastInputIdProcessed = 0
        self._seenParticleIds = set()
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = False

    def _loadEmptyParticleSet(self):
        return _Batch()

    def getRunMode(self):
        return "normal-run"

    def _streamingMustStop(self):
        return False

    def _loadInputParticleSet(self):
        self.inputSet.loadAllProperties()
        return self.inputSet

    def _reconcileClosedStreamIds(
            self, inputSet, discoveredIds, knownIds, producerClosed,
            watermarkAttr='_lastInputId'):
        return XmippProtClassifyPcaStreaming._reconcileClosedStreamIds(
            self,
            inputSet,
            discoveredIds,
            knownIds,
            producerClosed,
            watermarkAttr=watermarkAttr,
        )

    def _loadLogicalSetItemsByIds(self, inputSet, itemIds, batchSize=500):
        return XmippProtClassifyPcaStreaming._loadLogicalSetItemsByIds(
            self,
            inputSet,
            itemIds,
            batchSize=batchSize,
        )

    def _hydrateLogicalSetItemAcquisition(self, inputSet, item):
        return item

    def _doClassification(self, particles):
        return XmippProtClassifyPcaStreaming._doClassification(self, particles)

    def _insertClassificationSteps(self, particles, lastInputId):
        self.scheduledBatches.append((particles.ids(), lastInputId))
        # Pretend publication completes immediately so the generator can
        # finish deterministically.
        self.classificationLaunch = False

    def _insertFunctionStep(self, func, prerequisites=None, needsGPU=False):
        self.closeInserted = True
        return 1

    def closeOutputStep(self):
        pass

    def info(self, _message):
        pass


class TestAlignPcaStreamingLateVisibleIdRegression(unittest.TestCase):

    def testClosedStreamReconcilesLateVisibleIdsBelowWatermark(self):
        protocol = _LateVisibleHarness()

        def fakeSleep(_seconds):
            # After the first poll, id=2 becomes visible and the producer
            # simultaneously closes.
            protocol.inputSet.phase = 1

        with patch.object(alignpca_module.time, "sleep", side_effect=fakeSleep):
            XmippProtClassifyPcaStreaming.stepsGeneratorStep(protocol)

        classifiedIds = [
            particleId
            for batch, _lastId in protocol.scheduledBatches
            for particleId in batch
        ]

        self.assertEqual(
            sorted(classifiedIds),
            [1, 2, 3],
            "Terminal reconciliation must recover ids that become visible "
            "below the streaming watermark before the producer closes.",
        )
        self.assertEqual(
            len(classifiedIds),
            len(set(classifiedIds)),
            "Terminal reconciliation must not classify any particle twice.",
        )
        self.assertTrue(protocol.closeInserted)
class _PersistedClassItems:
    def __init__(self, classCount, particleIds):
        self._classCount = classCount
        self._particleIds = list(particleIds)

    def loadAllProperties(self):
        pass

    def __len__(self):
        return self._classCount

    def iterClassItems(self):
        return iter(_Particle(itemId) for itemId in self._particleIds)


class _AmbiguousResumeHarness:
    CREATE_CLASSES = XmippProtClassifyPcaStreaming.CREATE_CLASSES
    UPDATE_CLASSES = XmippProtClassifyPcaStreaming.UPDATE_CLASSES

    def __init__(self):
        self.mode = _Value(self.CREATE_CLASSES)
        self.outputClasses = _PersistedClassItems(
            classCount=5,
            particleIds=[1, 2, 3, 4],
        )
        self.numberClasses = 5
        self.lastInputId = 0
        self.classificationRound = 0
        self.classificationStarted = False
        self.firstTimeDone = False
        self.lastInputIdProcessed = 0
        self._seenParticleIds = set()

    def _restoreStreamingStateFromSteps(self):
        return 2, 1

    def _restoreProcessedParticleIdsFromSteps(self):
        return {1, 2}, True


class _AmbiguousInputSet:
    def __init__(self):
        self.iterCalls = []

    def loadAllProperties(self):
        pass

    def getStreamState(self):
        return Set.STREAM_CLOSED

    def getSize(self):
        return 5

    def getUniqueValues(self, column, where=None):
        assert column == "id"
        ids = [1, 2, 3, 4, 5]
        if where is None:
            return ids
        watermark = int(where.split(">", 1)[1].strip())
        return [itemId for itemId in ids if itemId > watermark]

    def iterItems(self, orderBy="id", direction="ASC", where=None, limit=None):
        ids = [1, 2, 3, 4, 5]
        if where is not None:
            if " IN " in where:
                idsText = where.split("(", 1)[1].rsplit(")", 1)[0]
                wanted = {
                    int(value.strip())
                    for value in idsText.split(",")
                    if value.strip()
                }
                ids = [itemId for itemId in ids if itemId in wanted]
            elif ">" in where:
                watermark = int(where.split(">", 1)[1].strip())
                ids = [itemId for itemId in ids if itemId > watermark]
        if limit is not None:
            ids = ids[:limit]
        self.iterCalls.append((where, limit, tuple(ids)))
        return iter(_Particle(itemId) for itemId in ids)

    def close(self):
        pass


class _AmbiguousGeneratorHarness:
    def __init__(self):
        self.classificationBatch = _Value(10)
        self.inputSet = _AmbiguousInputSet()
        self.inputParticles = _Pointer(self.inputSet)
        self._lock = threading.Lock()
        self._originalRunMode = "normal-run"
        self.scheduledBatches = []
        self.closeInserted = False

    def _initialStep(self):
        self.finish = False
        self.lastInputId = 2
        self.lastInputIdProcessed = 2
        self._seenParticleIds = {1, 2, 3, 4}
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = False

    def _loadEmptyParticleSet(self):
        return _Batch()

    def getRunMode(self):
        return "normal-run"

    def _streamingMustStop(self):
        return False

    def _loadInputParticleSet(self):
        self.inputSet.loadAllProperties()
        return self.inputSet

    def _reconcileClosedStreamIds(
            self, inputSet, discoveredIds, knownIds, producerClosed,
            watermarkAttr="_lastInputId"):
        return XmippProtClassifyPcaStreaming._reconcileClosedStreamIds(
            self,
            inputSet,
            discoveredIds,
            knownIds,
            producerClosed,
            watermarkAttr=watermarkAttr,
        )

    def _loadLogicalSetItemsByIds(self, inputSet, itemIds, batchSize=500):
        return XmippProtClassifyPcaStreaming._loadLogicalSetItemsByIds(
            self,
            inputSet,
            itemIds,
            batchSize=batchSize,
        )

    def _hydrateLogicalSetItemAcquisition(self, inputSet, item):
        return item

    def _doClassification(self, particles):
        return XmippProtClassifyPcaStreaming._doClassification(
            self,
            particles,
        )

    def _insertClassificationSteps(self, particles, lastInputId):
        self.scheduledBatches.append((particles.ids(), lastInputId))
        self.classificationLaunch = False

    def _insertFunctionStep(self, func, prerequisites=None, needsGPU=False):
        self.closeInserted = True
        return 1

    def closeOutputStep(self):
        pass

    def info(self, _message):
        pass


class TestAlignPcaStreamingAmbiguousPublicationRegression(unittest.TestCase):

    def testContinueTrustsDurableClassItemsEvenIfPublicationStepFailed(self):
        protocol = _AmbiguousResumeHarness()

        XmippProtClassifyPcaStreaming._updateVarsToContinue(protocol)

        self.assertEqual(
            protocol._seenParticleIds,
            {1, 2, 3, 4},
            "Durably persisted class members must remain consumed even when "
            "their publication step did not reach FINISHED.",
        )

    def testGeneratorDoesNotRepublishDurableIdsFromFailedPublicationStep(self):
        protocol = _AmbiguousGeneratorHarness()

        with patch.object(alignpca_module.time, "sleep", return_value=None):
            XmippProtClassifyPcaStreaming.stepsGeneratorStep(protocol)

        classifiedIds = [
            particleId
            for batch, _lastId in protocol.scheduledBatches
            for particleId in batch
        ]
        self.assertEqual(
            classifiedIds,
            [5],
            "Continue must classify only the particle that is not already "
            "present in the durable output.",
        )
        self.assertTrue(protocol.closeInserted)
class _SerializableClassificationStepHarness:
    runClassificationSteps = XmippProtClassifyPcaStreaming.runClassificationSteps
    updateOutputSetOfClasses = (
        XmippProtClassifyPcaStreaming.updateOutputSetOfClasses
    )

    def __init__(self):
        self.classificationLaunch = False
        self.imgsOrigXmd = "imagesInput_.xmd"
        self.imgsFn = "images_.mrc"
        self.newDeps = []
        self.insertedCalls = []

    def _updateFnClassification(self):
        pass

    def _insertFunctionStep(self, func, *args, **kwargs):
        self.insertedCalls.append((func.__name__, args, kwargs))
        return len(self.insertedCalls)


class TestAlignPcaSerializableClassificationStepRegression(unittest.TestCase):

    def testClassificationStepPersistsOnlyParticleIdsNotLiveBatchSet(self):
        protocol = _SerializableClassificationStepHarness()
        batch = [_Particle(7), _Particle(11)]

        XmippProtClassifyPcaStreaming._insertClassificationSteps(
            protocol,
            batch,
            11,
        )

        self.assertEqual(len(protocol.insertedCalls), 2)

        className, classArgs, _classKwargs = protocol.insertedCalls[0]
        self.assertEqual(className, "runClassificationSteps")
        self.assertEqual(
            classArgs[0],
            [7, 11],
            "The classification FunctionStep must persist serializable "
            "particle ids, never the live temporary batch Set.",
        )

        updateName, updateArgs, _updateKwargs = protocol.insertedCalls[1]
        self.assertEqual(updateName, "updateOutputSetOfClasses")
        self.assertEqual(updateArgs[2], [7, 11])
class _FirstPublicationAmbiguousResumeHarness:
    CREATE_CLASSES = XmippProtClassifyPcaStreaming.CREATE_CLASSES
    UPDATE_CLASSES = XmippProtClassifyPcaStreaming.UPDATE_CLASSES

    def __init__(self):
        self.mode = _Value(self.CREATE_CLASSES)
        self.outputClasses = _PersistedClassItems(
            classCount=5,
            particleIds=[1, 2, 3],
        )

        # Simulate _initFnStep restoring the originally requested value.
        self.numberClasses = 50

        self.lastInputId = 0
        self.classificationRound = 0
        self.classificationStarted = False
        self.firstTimeDone = False
        self.lastInputIdProcessed = 0
        self._seenParticleIds = set()

    def _restoreStreamingStateFromSteps(self):
        # The first publication step failed after outputClasses was already
        # committed, so there is no FINISHED publication round.
        return 0, 0

    def _restoreProcessedParticleIdsFromSteps(self):
        return set(), True


class TestAlignPcaFirstPublicationAmbiguousCommitRegression(unittest.TestCase):

    def testContinueRestoresInitialClassificationFromDurableOutput(self):
        protocol = _FirstPublicationAmbiguousResumeHarness()

        XmippProtClassifyPcaStreaming._updateVarsToContinue(protocol)

        self.assertTrue(
            protocol.classificationStarted,
            "A non-empty durable outputClasses must prove that the initial "
            "classification already exists even when its publication step "
            "did not reach FINISHED.",
        )
        self.assertEqual(
            protocol.numberClasses,
            5,
            "CREATE_CLASSES Continue must restore the effective class count "
            "from durable outputClasses after an ambiguous first commit.",
        )
        self.assertEqual(
            protocol._seenParticleIds,
            {1, 2, 3},
        )
class _FirstPublicationImages:
    def __init__(self, particleIds):
        self._particleIds = list(particleIds)

    def iterItems(self, **_kwargs):
        return iter(_Particle(itemId) for itemId in self._particleIds)

    def getIdSet(self):
        return set(self._particleIds)



class _FirstPublicationClasses:
    def __init__(self, particleIds):
        self._images = _FirstPublicationImages(particleIds)

    def getImages(self):
        return self._images

    def classifyItems(
            self,
            updateItemCallback=None,
            updateClassCallback=None,
            itemDataIterator=None,
            classifyDisabled=False,
            iterParams=None,
            doClone=True,
            raiseOnNextFailure=True,
            cancelNextWhenAppendIsFalse=False):
        # Reproduce the relevant SetOfClasses.classifyItems behavior:
        # when raiseOnNextFailure=False, exhausted metadata terminates the
        # loop silently before updateItemCallback receives row=None.
        for item in self._images.iterItems(**(iterParams or {})):
            try:
                row = next(itemDataIterator)
            except Exception:
                if raiseOnNextFailure:
                    raise
                return

            # We only need the third particle to exercise the missing-row
            # behavior. Earlier fake rows need not implement full Xmipp
            # alignment metadata.
            if item.getObjId() == 3:
                updateItemCallback(item, row)


class _FirstPublicationFillHarness:
    def __init__(self):
        self._lock = threading.Lock()

    def _createModelFile(self):
        pass

    def _loadClassesInfo(self, _filename):
        pass

    def _getExtraPath(self, *parts):
        return "/tmp/" + "/".join(parts)

    def _updateParticle(self, item, row):
        return XmippProtClassifyPcaStreaming._updateParticle(
            self,
            item,
            row,
        )

    def _iterRowsForParticleIds(self, rows, particleIds):
        return XmippProtClassifyPcaStreaming._iterRowsForParticleIds(
            rows,
            particleIds,
        )

    def _loadLogicalSetItemsByIds(self, _inputSet, itemIds):
        return [_Particle(itemId) for itemId in itemIds]

    def _classifyLogicalParticles(self, _clsSet, particles, mdIter):
        for particle in particles:
            row = next(mdIter)
            if particle.getObjId() == 3:
                self._updateParticle(particle, row)

    def _updateClass(self, _classItem):
        pass

    def info(self, _message):
        pass

    def error(self, _message):
        pass


class TestAlignPcaFirstPublicationMetadataExhaustionRegression(unittest.TestCase):

    def testFirstPublicationFailsWhenClassificationMetadataEndsEarly(self):
        protocol = _FirstPublicationFillHarness()
        classes = _FirstPublicationClasses([1, 2, 3])
        rows = [
            _MetadataRow(1),
            _MetadataRow(2),
        ]

        with patch.object(
            alignpca_module.emtable.Table,
            "iterRows",
            return_value=iter(rows),
        ):
            with self.assertRaisesRegex(RuntimeError, "3"):
                XmippProtClassifyPcaStreaming._fillClassesFromLevel(
                    protocol,
                    classes,
                    update=False,
                )
import threading as _alignpca_terminal_threading
from unittest.mock import patch as _alignpca_terminal_patch


class _TerminalCloseValue:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _TerminalCloseParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _TerminalCloseParticle(self._objId)


class _TerminalCloseInput:
    def __init__(self, particleIds):
        self._particleIds = list(particleIds)

    def getStreamState(self):
        return Set.STREAM_CLOSED

    def getSize(self):
        return len(self._particleIds)

    def getUniqueValues(self, attribute, where=None):
        if attribute != 'id':
            raise AssertionError("Unexpected attribute: %s" % attribute)
        ids = list(self._particleIds)
        if where:
            watermark = int(where.split('>')[1].strip())
            ids = [itemId for itemId in ids if itemId > watermark]
        return ids

    def iterItems(self, orderBy='id', direction='ASC',
                  where=None, limit=None):
        ids = sorted(self._particleIds)
        if where:
            watermark = int(where.split('>')[1].strip())
            ids = [itemId for itemId in ids if itemId > watermark]
        if limit is not None:
            ids = ids[:limit]
        return iter(_TerminalCloseParticle(itemId) for itemId in ids)

    def close(self):
        pass


class _TerminalClosePointer:
    def __init__(self, inputSet):
        self._inputSet = inputSet

    def get(self):
        return self._inputSet


class _TerminalCloseHarness:
    def __init__(self):
        self.classificationBatch = _TerminalCloseValue(2)
        self.inputParticles = _TerminalClosePointer(
            _TerminalCloseInput([1, 2])
        )
        self._lock = _alignpca_terminal_threading.Lock()
        self._originalRunMode = -999

        self.scheduledBatches = []
        self.closePrerequisites = []
        self._nextDependency = 100

    def getRunMode(self):
        return -999

    def _initialStep(self):
        self.finish = False
        self.lastInputId = 0
        self.lastInputIdProcessed = 0
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = False
        self.classificationRound = 0
        self.classificationStarted = False
        self.firstTimeDone = False
        self._seenParticleIds = set()

    def _loadEmptyParticleSet(self):
        return []

    def _loadInputParticleSet(self):
        return self.inputParticles.get()

    def _streamingMustStop(self):
        return False

    def _reconcileClosedStreamIds(
            self, inputSet, discoveredIds, knownIds, producerClosed,
            watermarkAttr='lastInputId'):
        return XmippProtClassifyPcaStreaming._reconcileClosedStreamIds(
            self,
            inputSet,
            discoveredIds,
            knownIds,
            producerClosed,
            watermarkAttr=watermarkAttr,
        )

    def _loadLogicalSetItemsByIds(self, inputSet, itemIds):
        wanted = set(itemIds)
        return [
            particle.clone()
            for particle in inputSet.iterItems()
            if particle.getObjId() in wanted
        ]

    def _doClassification(self, newParticlesSet):
        return XmippProtClassifyPcaStreaming._doClassification(
            self,
            newParticlesSet,
        )

    def _insertClassificationSteps(self, newParticlesSet, lastInputId):
        self.classificationLaunch = True
        self.scheduledBatches.append(
            [particle.getObjId() for particle in newParticlesSet]
        )
        dep = self._nextDependency
        self._nextDependency += 1
        self.newDeps.append(dep)

    def _insertFunctionStep(self, func, *args, **kwargs):
        if getattr(func, '__name__', '') == 'closeOutputStep':
            self.closePrerequisites.append(
                list(kwargs.get('prerequisites') or [])
            )
        return 999

    def closeOutputStep(self):
        pass

    def info(self, _message):
        pass


class TestAlignPcaTerminalCloseRegression(unittest.TestCase):

    def testClosedStreamSchedulesFinalCloseAfterLastFullBatch(self):
        protocol = _TerminalCloseHarness()

        with _alignpca_terminal_patch.object(
            alignpca_module.time,
            'sleep',
            return_value=None,
        ):
            XmippProtClassifyPcaStreaming.stepsGeneratorStep(protocol)

        self.assertTrue(protocol.finish)
        self.assertEqual(protocol.scheduledBatches, [[1, 2]])
        self.assertEqual(
            protocol.closePrerequisites,
            [[100]],
            "A closed stream must insert exactly one final close step "
            "after the last publication dependency.",
        )
class _TerminalInconsistentInput(_TerminalCloseInput):
    def __init__(self):
        super().__init__([])

    def getSize(self):
        # Producer reports one durable item, but no logical id is visible.
        return 1


class _TerminalInconsistentHarness(_TerminalCloseHarness):
    def __init__(self):
        super().__init__()
        self.inputParticles = _TerminalClosePointer(
            _TerminalInconsistentInput()
        )
        self.stopChecks = 0

    def _streamingMustStop(self):
        self.stopChecks += 1
        if self.stopChecks > 20:
            raise AssertionError(
                "AlignPCA kept polling a permanently inconsistent "
                "closed input instead of failing deterministically."
            )
        return False


class TestAlignPcaTerminalInconsistencyRegression(unittest.TestCase):

    def testPermanentlyInconsistentClosedInputDoesNotPollForever(self):
        protocol = _TerminalInconsistentHarness()

        with _alignpca_terminal_patch.object(
            alignpca_module.time,
            'sleep',
            return_value=None,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "closed.*input|terminal.*consistent|consistent.*terminal",
            ):
                XmippProtClassifyPcaStreaming.stepsGeneratorStep(protocol)
class _SerializableWorkerInput:
    def __init__(self):
        self.closed = False

    def getAlignment(self):
        return "2D"

    def close(self):
        self.closed = True


class _SerializableWorkerParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _SerializableWorkerHarness:
    def __init__(self):
        self.inputSet = _SerializableWorkerInput()
        self.convertCalled = False
        self.classificationCalled = False

    def _loadInputParticleSet(self):
        return self.inputSet

    def _loadLogicalSetItemsByIds(self, _inputSet, _particleIds):
        # Simulate one requested logical item not being reconstructable.
        return [_SerializableWorkerParticle(7)]

    def convertInputStep(self, *_args, **_kwargs):
        self.convertCalled = True

    def classification(self, *_args, **_kwargs):
        self.classificationCalled = True


class TestAlignPcaSerializableWorkerReconstructionRegression(unittest.TestCase):

    def testIncompleteLogicalBatchFailsBeforeScientificWork(self):
        protocol = _SerializableWorkerHarness()

        with self.assertRaisesRegex(RuntimeError, "Missing ids.*11"):
            XmippProtClassifyPcaStreaming.runClassificationSteps(
                protocol,
                [7, 11],
                "round.xmd",
                "round.mrc",
            )

        self.assertTrue(
            protocol.inputSet.closed,
            "The logical input Set must be closed even when reconstruction "
            "is incomplete.",
        )
        self.assertFalse(
            protocol.convertCalled,
            "Input conversion must not start with a partial logical batch.",
        )
        self.assertFalse(
            protocol.classificationCalled,
            "Scientific classification must not start with a partial "
            "logical batch.",
        )
from unittest.mock import patch as _alignpca_worker_patch


class _ScientificFailureValue:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _ScientificFailureInput:
    def __init__(self):
        self.closed = False

    def getAlignment(self):
        return "2D"

    def close(self):
        self.closed = True


class _ScientificFailureParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _ScientificFailureHarness:
    def __init__(self):
        self.inputSet = _ScientificFailureInput()
        self.training = _ScientificFailureValue(10)
        self.numberClasses = 3
        self.mask = _ScientificFailureValue(False)
        self.sigmaProt = 1.0
        self.resolutionPca = 8.0
        self.convertCalled = False

    def _loadInputParticleSet(self):
        return self.inputSet

    def _loadLogicalSetItemsByIds(self, _inputSet, particleIds):
        return [
            _ScientificFailureParticle(itemId)
            for itemId in particleIds
        ]

    def convertInputStep(self, _batchView, _imgsOrigXmd, _imgsFn):
        self.convertCalled = True

    def classification(self, *_args, **_kwargs):
        raise RuntimeError("scientific classification failure")


class TestAlignPcaScientificFailurePropagationRegression(unittest.TestCase):

    def testScientificFailurePropagatesAndDoesNotCleanupRoundInput(self):
        protocol = _ScientificFailureHarness()

        with _alignpca_worker_patch.object(
            alignpca_module.pwutils,
            "cleanPath",
        ) as cleanPath:
            with self.assertRaisesRegex(
                RuntimeError,
                "scientific classification failure",
            ):
                XmippProtClassifyPcaStreaming.runClassificationSteps(
                    protocol,
                    [7, 11],
                    "round.xmd",
                    "round.mrc",
                )

        self.assertTrue(protocol.convertCalled)
        self.assertTrue(protocol.inputSet.closed)
        cleanPath.assert_not_called()
class _SourceRelationOutput:
    def __len__(self):
        return 3


class _SourceRelationAmbiguityHarness:
    def __init__(self):
        self.output = _SourceRelationOutput()
        self.outputDurable = False
        self.relationAttempts = 0
        self.classificationStarted = False
        self._sourceRelationPublished = False
        self.classificationLaunch = True
        self.numberClasses = 10
        self.lastInputIdProcessed = 0
        self.classificationRound = 1

    def _loadOutputSet(self, _outputName):
        return self.output, self.outputDurable

    def _fillClassesFromLevel(self, _outputClasses, _update,
                              batchIds=None):
        pass

    def _updateOutputSet(self, _outputName, _outputClasses, _streamMode):
        self.outputDurable = True

    def _defineSourceRelation(self, _inputPointer, _outputClasses):
        self.relationAttempts += 1
        if self.relationAttempts == 1:
            raise RuntimeError("source relation publication failed")

    def _getInputPointer(self):
        return object()

    def info(self, _message):
        pass


class TestAlignPcaSourceRelationAmbiguousCommitRegression(unittest.TestCase):

    def testDurableFirstOutputRetriesMissingSourceRelation(self):
        protocol = _SourceRelationAmbiguityHarness()

        with self.assertRaisesRegex(
            RuntimeError,
            "source relation publication failed",
        ):
            XmippProtClassifyPcaStreaming.updateOutputSetOfClasses(
                protocol,
                3,
                Set.STREAM_OPEN,
                [1, 2, 3],
            )

        self.assertTrue(
            protocol.outputDurable,
            "The regression requires the first output publication to have "
            "become durable before the relation failure.",
        )

        XmippProtClassifyPcaStreaming.updateOutputSetOfClasses(
            protocol,
            3,
            Set.STREAM_OPEN,
            [1, 2, 3],
        )

        self.assertEqual(
            protocol.relationAttempts,
            2,
            "A retry after an ambiguous first publication must retry the "
            "missing source relation even though the output already exists.",
        )
class _ExactBatchImages:
    def __init__(self):
        self.iterCalls = []

    def iterItems(self, **kwargs):
        self.iterCalls.append(kwargs)
        raise AssertionError(
            "Exact batch publication must not query the input Set through "
            "mapper filter syntax."
        )


class _ExactBatchClassSet:
    def __init__(self):
        self.images = _ExactBatchImages()

    def getImages(self):
        return self.images


class _ExactBatchParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _ExactBatchPublicationHarness:
    def __init__(self):
        self.loadedIds = None
        self._lock = _alignpca_terminal_threading.Lock()

    def _createModelFile(self):
        pass

    def _loadClassesInfo(self, _filename):
        self._classesInfo = {}

    def _getExtraPath(self, name):
        return name

    def _loadLogicalSetItemsByIds(self, _inputSet, itemIds):
        self.loadedIds = list(itemIds)
        return [
            _ExactBatchParticle(itemId)
            for itemId in itemIds
        ]

    def _iterRowsForParticleIds(self, rows, particleIds):
        return iter(())

    def _classifyLogicalParticles(
            self, _clsSet, particles, _mdIter):
        self.classifiedIds = [
            particle.getObjId()
            for particle in particles
        ]

    def info(self, _message):
        pass


class TestAlignPcaExactBatchPublicationContract(unittest.TestCase):

    def testExactBatchPublicationDoesNotUseMapperFilterSyntax(self):
        protocol = _ExactBatchPublicationHarness()
        classes = _ExactBatchClassSet()

        with _alignpca_worker_patch.object(
            alignpca_module.emtable.Table,
            "iterRows",
            return_value=iter(()),
        ):
            XmippProtClassifyPcaStreaming._fillClassesFromLevel(
                protocol,
                classes,
                update=True,
                batchIds=[11, 7],
            )

        self.assertEqual(protocol.loadedIds, [7, 11])
        self.assertEqual(protocol.classifiedIds, [7, 11])
        self.assertEqual(classes.images.iterCalls, [])
class _ResumeReferenceValue:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _ResumeReferenceHarness:
    def __init__(self):
        self.mode = _ResumeReferenceValue(
            XmippProtClassifyPcaStreaming.UPDATE_CLASSES
        )
        self.correctCtf = _ResumeReferenceValue(False)
        self.initialClasses = _ResumeReferenceValue(object())
        self.numberOfMpi = _ResumeReferenceValue(1)
        self.firstTimeDone = True
        self.classificationStarted = True
        self.refXmd = "resume_references.xmd"
        self.ref = "/definitely/missing/alignpca_resume_classes.mrcs"
        self.sampling = 1.0
        self.jobs = []

    def runJob(self, program, args, **kwargs):
        self.jobs.append((program, args, kwargs))


class TestAlignPcaUpdateClassesResumeReferenceRegression(unittest.TestCase):

    def testMissingReferenceFailsWhenResumeStateWasRestored(self):
        protocol = _ResumeReferenceHarness()

        with self.assertRaisesRegex(
            RuntimeError,
            "Classification reference is missing",
        ):
            XmippProtClassifyPcaStreaming.convertInputStep(
                protocol,
                [],
                "round.xmd",
                "round.mrc",
            )

        self.assertEqual(
            protocol.jobs,
            [],
            "No scientific preparation job may start when a required "
            "durable classification reference is missing.",
        )
        self.assertTrue(protocol.firstTimeDone)

class _LegacyAmbiguousValue:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _LegacyAmbiguousParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId


class _LegacyAmbiguousOutput:
    def __init__(self, particleIds):
        self._particleIds = list(particleIds)
        self.closed = False

    def loadAllProperties(self):
        pass

    def iterClassItems(self):
        return iter(
            _LegacyAmbiguousParticle(itemId)
            for itemId in self._particleIds
        )

    def __len__(self):
        return 3

    def close(self):
        self.closed = True


class _LegacyAmbiguousInput:
    def __init__(self, particleIds):
        self._particleIds = list(particleIds)
        self.closed = False

    def getIdSet(self):
        return set(self._particleIds)

    def getUniqueValues(self, field):
        if field != 'id':
            raise AssertionError(
                "Unexpected logical field requested: %s" % field
            )
        return list(self._particleIds)

    def iterItems(self, **_kwargs):
        return iter(
            _LegacyAmbiguousParticle(itemId)
            for itemId in self._particleIds
        )

    def close(self):
        self.closed = True


class _LegacyAmbiguousResumeHarness:
    CREATE_CLASSES = XmippProtClassifyPcaStreaming.CREATE_CLASSES
    UPDATE_CLASSES = XmippProtClassifyPcaStreaming.UPDATE_CLASSES

    def __init__(self):
        self.mode = _LegacyAmbiguousValue(
            XmippProtClassifyPcaStreaming.UPDATE_CLASSES
        )
        self.outputClasses = _LegacyAmbiguousOutput([1, 2, 3, 4])
        self.inputSet = _LegacyAmbiguousInput([1, 2, 3, 4])
        self.mapper = None
        self.firstTimeDone = False
        self.classificationStarted = False
        self.classificationRound = 0
        self.lastInputId = 0
        self.lastInputIdProcessed = 0
        self._seenParticleIds = set()

    def _loadInputParticleSet(self):
        return self.inputSet

    def _restoreStreamingStateFromSteps(self):
        # One publication step is known FINISHED up to id 2.
        return 2, 1

    def _restoreProcessedParticleIdsFromSteps(self):
        # Legacy publication steps did not persist exact batchIds.
        return {1, 2}, False


class TestAlignPcaLegacyAmbiguousRoundResumeRegression(unittest.TestCase):

    def testDurableIdsBeyondLegacyFinishedWatermarkAdvanceRound(self):
        protocol = _LegacyAmbiguousResumeHarness()

        XmippProtClassifyPcaStreaming._updateVarsToContinue(protocol)

        self.assertTrue(protocol.classificationStarted)
        self.assertTrue(protocol.firstTimeDone)
        self.assertEqual(protocol.lastInputId, 2)
        self.assertEqual(protocol.lastInputIdProcessed, 2)
        self.assertEqual(protocol._seenParticleIds, {1, 2, 3, 4})
        self.assertEqual(
            protocol.classificationRound,
            2,
            "Durable ids beyond the last FINISHED publication watermark "
            "prove that one additional publication round committed even "
            "when legacy steps do not contain exact batchIds. Resume must "
            "not reuse that round number.",
        )
import threading as _alignpca_terminal_guard_threading


class _TerminalBusyValue:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _TerminalBusyPendingBatch:
    def __init__(self):
        self._items = []

    def __len__(self):
        return len(self._items)

    def append(self, item):
        self._items.append(item)


class _TerminalBusyClosedInput:
    def __init__(self):
        self.closed = False

    def getStreamState(self):
        return Set.STREAM_CLOSED

    def getSize(self):
        return 1

    def iterItems(self, **_kwargs):
        return iter(())

    def close(self):
        self.closed = True


class _TerminalBusyInputPointer:
    def __init__(self, inputSet):
        self._inputSet = inputSet

    def get(self):
        return self._inputSet


class _TerminalBusyHarness:
    def __init__(self):
        self._originalRunMode = -1
        self.inputSet = _TerminalBusyClosedInput()
        self.inputParticles = _TerminalBusyInputPointer(self.inputSet)
        self.classificationBatch = _TerminalBusyValue(2)
        self._lock = _alignpca_terminal_guard_threading.Lock()
        self.stopChecks = 0

    def _initialStep(self):
        self.finish = False
        self.lastInputId = 0
        self.lastInputIdProcessed = 0
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = True
        self.classificationStarted = True
        self._seenParticleIds = set()

    def _loadEmptyParticleSet(self):
        return _TerminalBusyPendingBatch()

    def getRunMode(self):
        return 0

    def _streamingMustStop(self):
        self.stopChecks += 1
        return self.stopChecks > 20

    def _loadInputParticleSet(self):
        return self.inputSet

    def _doClassification(self, _pendingBatch):
        return False

    def _reconcileClosedStreamIds(
            self, _inputSet, _pendingIds, _seenIds,
            _includeVisible, watermarkAttr=None):
        return [], False

    def _loadLogicalSetItemsByIds(self, _inputSet, _ids):
        return []

    def info(self, _message):
        pass


class TestAlignPcaTerminalInconsistencyBusyRoundRegression(unittest.TestCase):

    def testTerminalInconsistencyDoesNotExpireWhileRoundIsActive(self):
        protocol = _TerminalBusyHarness()

        with _alignpca_worker_patch.object(
            alignpca_module.time,
            "sleep",
            return_value=None,
        ):
            XmippProtClassifyPcaStreaming.stepsGeneratorStep(protocol)

        self.assertGreater(
            protocol.stopChecks,
            12,
            "The generator must survive more than the terminal stall limit "
            "while a classification/publication round is still active.",
        )
        self.assertTrue(protocol.classificationLaunch)
