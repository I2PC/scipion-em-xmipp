# ******************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# ******************************************************************************

import json
import threading
from unittest.mock import patch

from pyworkflow.object import Set
from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols import XmippProtClassifyPcaStreaming


class _FakeFuncName:
    def __init__(self, name):
        self._name = name

    def get(self):
        return self._name


class _FakeArgsStr:
    def __init__(self, value):
        self._value = value

    def get(self, default=None):
        return self._value


class _FakeStreamingStep:
    """Stands in for a persisted pyworkflow step graph entry, exactly as
    _restoreStreamingStateFromSteps reads it: funcName/argsStr objects with
    a .get() accessor, plus isFinished()."""

    def __init__(self, funcName, args, finished=True):
        self.funcName = _FakeFuncName(funcName)
        self.argsStr = _FakeArgsStr(json.dumps(args))
        self._finished = finished

    def isFinished(self):
        return self._finished


class _LogicalParticles:
    def __init__(self):
        self.loadCalls = 0

    def getFileName(self):
        raise AssertionError(
            "Streaming must not use the Set storage filename as its "
            "authoritative input."
        )

    def loadAllProperties(self):
        self.loadCalls += 1

    def close(self):
        pass


class _Pointer:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _EmptyParticles:
    def __init__(self):
        self.infoSource = None

    def copyInfo(self, source):
        self.infoSource = source


class _StreamingSetContractHarness:
    def __init__(self, particles):
        self._lock = __import__('threading').RLock()
        self.inputParticles = _Pointer(particles)
        self.emptyParticles = _EmptyParticles()

    def debug(self, *args, **kwargs):
        pass

    def _createSetOfParticles(self):
        return self.emptyParticles



class _BatchSizeParam:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _BatchParticle:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _BatchParticle(self._objId)


class _ClosedLogicalParticles:
    def __init__(self, count):
        self._particles = [
            _BatchParticle(index)
            for index in range(1, count + 1)
        ]

    def getStreamState(self):
        return Set.STREAM_CLOSED

    def getSize(self):
        return len(self._particles)

    def getUniqueValues(self, field):
        assert field == 'id'
        return [particle.getObjId() for particle in self._particles]

    def iterItems(
            self,
            orderBy=None,
            direction=None,
            where=None,
            limit=None,
    ):
        if where and ' IN ' in where:
            values = where.split('(', 1)[1].rsplit(')', 1)[0]
            wanted = {int(part.strip()) for part in values.split(',')}
            particles = [p for p in self._particles if p.getObjId() in wanted]
        else:
            lastId = int(where.split('>', 1)[1].strip()) if where else 0
            particles = [p for p in self._particles if p.getObjId() > lastId]

        if limit is not None and limit >= 0:
            particles = particles[:limit]

        yield from particles

    def close(self):
        pass


class _BatchAccumulator:
    def __init__(self):
        self._particles = []

    def append(self, particle):
        self._particles.append(particle)

    def __len__(self):
        return len(self._particles)


class _BatchPointer:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _ClosedInputBatchHarness:
    # Exercise the real terminal reconciliation, not a simplified replacement.
    _reconcileClosedStreamIds = XmippProtClassifyPcaStreaming._reconcileClosedStreamIds
    _recordTerminalProgress = XmippProtClassifyPcaStreaming._recordTerminalProgress
    _loadLogicalSetItemsByIds = XmippProtClassifyPcaStreaming._loadLogicalSetItemsByIds
    _hydrateLogicalSetItemAcquisition = staticmethod(
        XmippProtClassifyPcaStreaming._hydrateLogicalSetItemAcquisition
    )

    def _streamingMustStop(self):
        return False

    def _hasActiveStreamingWork(self):
        return False

    def __init__(self, particleCount, batchSize):
        self.source = _ClosedLogicalParticles(particleCount)
        self.inputParticles = _BatchPointer(self.source)
        self.classificationBatch = _BatchSizeParam(batchSize)

        self._originalRunMode = MODE_RESTART
        self.finish = False
        self.lastInputId = 0
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = False
        self.recordedBatchSizes = []
        self.closeOutputCalls = 0

        import threading
        self._lock = threading.RLock()

    def _initialStep(self):
        self.finish = False
        self.lastInputId = 0
        self._seenParticleIds = set()
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = False

    def _loadEmptyParticleSet(self):
        return _BatchAccumulator()

    def _loadInputParticleSet(self):
        return self.source

    def getRunMode(self):
        return MODE_RESTART

    def _doClassification(self, batch):
        return XmippProtClassifyPcaStreaming._doClassification(
            self,
            batch,
        )

    def _insertClassificationSteps(
            self,
            newParticlesSet,
            lastInputId,
    ):
        self.recordedBatchSizes.append(
            len(newParticlesSet)
        )

    def _insertFunctionStep(
            self,
            function,
            prerequisites=None,
            needsGPU=False,
    ):
        self.closeOutputCalls += 1
        return self.closeOutputCalls

    def closeOutputStep(self):
        pass

    def info(self, *args, **kwargs):
        pass



class _SyncParticle:
    def __init__(self, objId):
        self._objId = objId
        self._appendItem = True
        self._classId = None

    def getObjId(self):
        return self._objId

    def clone(self):
        return _SyncParticle(self._objId)

    def isEnabled(self):
        return True

    def setClassId(self, classId):
        self._classId = classId

    def getClassId(self):
        return self._classId


class _SyncRow:
    def __init__(self, itemId):
        self.itemId = itemId

    def get(self, _key):
        return self.itemId


class _SyncImages:
    def __init__(self, ids):
        self._ids = list(ids)

    def getIdSet(self):
        return set(self._ids)

    def close(self):
        pass

    def iterItems(self, **kwargs):
        where = kwargs.get('where')
        if where and ' IN ' in where:
            values = where.split('(', 1)[1].rsplit(')', 1)[0]
            allowed = {int(value.strip()) for value in values.split(',')}
        elif where:
            minimum = int(where.split('>', 1)[1].strip())
            allowed = {itemId for itemId in self._ids if itemId > minimum}
        else:
            allowed = set(self._ids)
        for objId in self._ids:
            if objId in allowed:
                yield _SyncParticle(objId)


class _SyncClass:
    def __init__(self):
        self.memberIds = []

    def append(self, particle):
        self.memberIds.append(particle.getObjId())


class _SyncClasses:
    def __init__(self, imageIds):
        self._images = _SyncImages(imageIds)
        self.pairs = []
        self.updatedMembers = []

    def getImages(self):
        return self._images

    def _getExistingItems(self):
        return {}

    def _get_or_create_class(self, classDict, classId, callback):
        if classId not in classDict:
            classDict[classId] = _SyncClass()
            callback(classDict[classId])
        return classDict[classId]

    def update(self, classItem):
        self.updatedMembers.extend(classItem.memberIds)



class _LegacyResumeInput:
    def __init__(self, ids):
        self.ids = list(ids)
        self.closed = False

    def getUniqueValues(self, field):
        assert field == 'id'
        return list(self.ids)

    def close(self):
        self.closed = True


class TestXmippClassifyPcaResume(BaseTest):
    """Regression tests for PCA2D streaming Continue/Restart handling."""

    @staticmethod
    def _attachLegacyResumeInput(protocol):
        # Step records predating exact batch IDs require one logical scan.
        source = _LegacyResumeInput([1, 10, 20, 30, 42, 999])
        protocol._lock = threading.RLock()
        protocol._loadInputParticleSet = lambda: source
        return source

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newPcaProtocol(self):
        return self.newProtocol(XmippProtClassifyPcaStreaming)

    def _prepareGeneratorProtocol(self, originalRunMode):
        prot = self._newPcaProtocol()

        # Reproduce Protocol._runSteps(): Scipion changes runMode to RESUME
        # while _originalRunMode keeps the action selected by the user.
        prot._originalRunMode = originalRunMode
        prot.runMode.set(MODE_RESUME)

        # We only want to exercise the startup/Resume decision here.
        prot.finish = True
        prot._initialStep = lambda: None
        prot._loadEmptyParticleSet = lambda: object()

        resumeCalls = []
        prot._updateVarsToContinue = lambda: resumeCalls.append(True)

        return prot, resumeCalls

    def testStreamingReadsTheLogicalInputSetInsteadOfReconstructingStorage(self):
        particles = _LogicalParticles()
        harness = _StreamingSetContractHarness(particles)

        loadedParticles = (
            XmippProtClassifyPcaStreaming._loadInputParticleSet(harness)
        )
        emptyParticles = (
            XmippProtClassifyPcaStreaming._loadEmptyParticleSet(harness)
        )

        self.assertIs(
            particles,
            loadedParticles,
            "Streaming must refresh and iterate the logical input Set.",
        )
        self.assertIs(
            particles,
            emptyParticles.infoSource,
            "The temporary batch Set must copy metadata from the logical input Set.",
        )
        self.assertGreaterEqual(
            particles.loadCalls,
            2,
            "Both streaming reads must refresh the logical Set state.",
        )

    def testClosedInputIsSplitByClassificationBatch(self):
        harness = _ClosedInputBatchHarness(
            particleCount=5,
            batchSize=2,
        )

        with patch(
            "xmipp3.protocols.protocol_alignPCA_2D.time.sleep",
            return_value=None,
        ):
            XmippProtClassifyPcaStreaming.stepsGeneratorStep(
                harness
            )

        self.assertEqual(
            [2, 2, 1],
            harness.recordedBatchSizes,
            "A closed input with all particles already available must "
            "still be processed in classificationBatch-sized chunks.",
        )
        self.assertEqual(
            1,
            harness.closeOutputCalls,
            "The output must close only after every pending particle "
            "has been scheduled.",
        )

    def testClassUpdateSynchronizesCumulativeMetadataWithFilteredInput(self):
        prot = self._newPcaProtocol()
        prot.lastInputIdProcessed = 2
        prot._lock = __import__("threading").RLock()
        prot._createModelFile = lambda: None
        prot._loadClassesInfo = lambda _filename: None
        prot._getExtraPath = lambda filename: filename

        def updateParticle(item, row):
            classes.pairs.append((item.getObjId(), row.itemId))
            if item.getObjId() != row.itemId:
                item._appendItem = False
            else:
                item.setClassId(1)

        prot._updateParticle = updateParticle
        prot._updateClass = lambda _item: None

        classes = _SyncClasses(
            imageIds=[1, 2, 3, 4, 5],
        )

        cumulativeRows = [
            _SyncRow(itemId)
            for itemId in [1, 2, 3, 4, 5]
        ]

        with patch(
            "xmipp3.protocols.protocol_alignPCA_2D.emtable.Table.iterRows",
            return_value=iter(cumulativeRows),
        ):
            prot._fillClassesFromLevel(
                classes,
                update=True,
            )

        self.assertEqual(
            [(3, 3), (4, 4), (5, 5)],
            classes.pairs,
            "Streaming class updates must align cumulative metadata rows "
            "with the same filtered input particle ids instead of pairing "
            "new particles with rows from previous batches.",
        )
        self.assertEqual([3, 4, 5], classes.updatedMembers)

    def testClassificationStepsKeepRoundSpecificInputFiles(self):
        prot = self._newPcaProtocol()
        prot.imgsOrigXmd = "/tmp/imagesInput_.xmd"
        prot.imgsXmd = "/tmp/images_.xmd"
        prot.imgsFn = "/tmp/images_.mrc"
        prot.classificationRound = 0
        prot.newDeps = []

        insertedSteps = []

        def insertFunctionStep(function, *args, **kwargs):
            insertedSteps.append((function, args, kwargs))
            return len(insertedSteps)

        prot._insertFunctionStep = insertFunctionStep

        firstBatch = [_BatchParticle(7)]
        secondBatch = [_BatchParticle(8), _BatchParticle(9)]

        prot._insertClassificationSteps(firstBatch, 7)
        prot._insertClassificationSteps(secondBatch, 9)

        firstClassStep = insertedSteps[0]
        secondClassStep = insertedSteps[2]

        self.assertEqual(
            prot.runClassificationSteps,
            firstClassStep[0],
        )
        self.assertEqual(
            prot.runClassificationSteps,
            secondClassStep[0],
        )

        self.assertEqual(
            (
                [7],
                "/tmp/imagesInput_0.xmd",
                "/tmp/images_0.mrc",
            ),
            firstClassStep[1],
            "The first classification step must keep the filenames "
            "of round 0 even after the generator schedules later rounds.",
        )
        self.assertEqual(
            (
                [8, 9],
                "/tmp/imagesInput_1.xmd",
                "/tmp/images_1.mrc",
            ),
            secondClassStep[1],
            "Each classification step must receive an immutable snapshot "
            "of its own round-specific input filenames.",
        )
        self.assertEqual((7, Set.STREAM_OPEN, [7]), insertedSteps[1][1])
        self.assertEqual((9, Set.STREAM_OPEN, [8, 9]), insertedSteps[3][1])

    def testResumeRestoresStateFromFinishedStepGraph(self):
        prot = self._newPcaProtocol()
        legacyInput = self._attachLegacyResumeInput(prot)
        prot._iterKnownStreamingSteps = lambda: [
            _FakeStreamingStep("updateOutputSetOfClasses", [10]),
            _FakeStreamingStep("updateOutputSetOfClasses", [20]),
            _FakeStreamingStep("updateOutputSetOfClasses", [30]),
            _FakeStreamingStep("updateOutputSetOfClasses", [42]),
        ]

        prot._updateVarsToContinue()

        self.assertEqual(42, prot.lastInputId)
        self.assertTrue(legacyInput.closed)
        self.assertEqual({1, 10, 20, 30, 42}, prot._seenParticleIds)
        self.assertEqual(
            4, prot.classificationRound,
            "classificationRound must equal the number of finished "
            "rounds, since _updateFnClassification already advances it "
            "exactly once per round before that round even runs.",
        )

    def testResumeIgnoresUnfinishedUpdateStep(self):
        prot = self._newPcaProtocol()
        legacyInput = self._attachLegacyResumeInput(prot)
        prot._iterKnownStreamingSteps = lambda: [
            _FakeStreamingStep("updateOutputSetOfClasses", [10]),
            _FakeStreamingStep("updateOutputSetOfClasses", [999], finished=False),
        ]

        prot._updateVarsToContinue()

        self.assertEqual(10, prot.lastInputId)
        self.assertTrue(legacyInput.closed)
        self.assertEqual({1, 10}, prot._seenParticleIds)
        self.assertEqual(1, prot.classificationRound)

    def testResumeWithoutPriorRoundsStartsFromInitialState(self):
        prot = self._newPcaProtocol()
        prot.lastInputId = 999
        prot.classificationRound = 99
        prot._iterKnownStreamingSteps = lambda: []

        prot._updateVarsToContinue()

        self.assertEqual(0, prot.lastInputId)
        self.assertEqual(0, prot.classificationRound)

    def testResumeUpdateClassesPreservesUpdatedReferences(self):
        prot = self._newPcaProtocol()
        legacyInput = self._attachLegacyResumeInput(prot)

        prot.mode.set(prot.UPDATE_CLASSES)
        prot.firstTimeDone = False
        prot._iterKnownStreamingSteps = lambda: [
            _FakeStreamingStep("updateOutputSetOfClasses", [42]),
        ]

        prot._updateVarsToContinue()

        self.assertTrue(legacyInput.closed)
        self.assertTrue(prot.firstTimeDone, "Continue in UPDATE_CLASSES mode must preserve the classes produced by previous rounds.")

    def testStepsGeneratorRestoresStateOnRealResume(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESUME)

        prot.stepsGeneratorStep()

        self.assertEqual(1, len(resumeCalls), "A real Continue/Resume must reconstruct the PCA2D streaming state from the step graph.")

    def testStepsGeneratorRestoresOnDefaultResumeWithNoOriginalRunMode(self):
        # Protocol.runMode defaults to MODE_RESUME even on a never-run
        # instance (_originalRunMode is only set once _runSteps() actually
        # executes). Calling _updateVarsToContinue in that case is safe: on
        # a protocol with no prior finished steps it just reconstructs the
        # same initial state _initialStep already set, so there is no
        # separate "has checkpoint" guard needed any more.
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESUME)
        del prot._originalRunMode

        prot.stepsGeneratorStep()

        self.assertEqual(1, len(resumeCalls))

    def testStepsGeneratorDoesNotRestoreStateOnRestart(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESTART)

        prot.stepsGeneratorStep()

        self.assertEqual(0, len(resumeCalls), "Restart must not restore the previous PCA2D streaming state.")


class _RefreshRequiredPcaClasses:
    def __init__(self):
        self.loaded = False
        self.appendEnabled = False

    def loadAllProperties(self):
        self.loaded = True

    def enableAppend(self):
        if not self.loaded:
            raise AssertionError(
                'Persisted PCA classes must be refreshed before append mode.'
            )
        self.appendEnabled = True


class TestXmippClassifyPcaLogicalOutputResume(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def testExistingOutputClassesAreRefreshedBeforeReuse(self):
        outputClasses = _RefreshRequiredPcaClasses()

        class _Harness:
            pass

        protocol = _Harness()
        protocol.outputClasses = outputClasses

        loadedOutput, update = XmippProtClassifyPcaStreaming._loadOutputSet(
            protocol,
            'outputClasses',
        )

        self.assertIs(outputClasses, loadedOutput)
        self.assertTrue(update)
        self.assertTrue(outputClasses.loaded)
        self.assertTrue(outputClasses.appendEnabled)

# AlignPCA resume contract GREEN 18
