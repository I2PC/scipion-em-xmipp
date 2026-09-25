# ******************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# ******************************************************************************

from unittest.mock import patch

from pyworkflow.object import Set
from pyworkflow.protocol.constants import MODE_RESTART, MODE_RESUME
from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols import XmippProtClassifyPcaStreaming


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
        self.inputParticles = _Pointer(particles)
        self.inputFn = "/tmp/compatibility.sqlite"
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
    def __init__(self, creation):
        self._creation = creation

    def getObjCreation(self):
        return self._creation

    def clone(self):
        return _BatchParticle(self._creation)


class _ClosedLogicalParticles:
    def __init__(self, count):
        self._particles = [
            _BatchParticle(index)
            for index in range(1, count + 1)
        ]

    def getStreamState(self):
        return Set.STREAM_CLOSED

    def iterItems(
            self,
            orderBy=None,
            direction=None,
            where=None,
            limit=None,
    ):
        lastCreation = 0

        if where:
            lastCreation = int(
                where.split('>"', 1)[1].rstrip('"')
            )

        particles = [
            particle
            for particle in self._particles
            if particle.getObjCreation() > lastCreation
        ]

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
    def __init__(self, particleCount, batchSize):
        self.source = _ClosedLogicalParticles(particleCount)
        self.inputParticles = _BatchPointer(self.source)
        self.classificationBatch = _BatchSizeParam(batchSize)

        self._originalRunMode = MODE_RESTART
        self.finish = False
        self.lastCreationTime = 0
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = False
        self.recordedBatchSizes = []
        self.closeOutputCalls = 0

        import threading
        self._lock = threading.RLock()

    def _initialStep(self):
        self.finish = False
        self.lastCreationTime = 0
        self.streamState = Set.STREAM_OPEN
        self.lastRound = False
        self.classificationLaunch = False

    def _loadEmptyParticleSet(self):
        return _BatchAccumulator()

    def _loadInputParticleSet(self):
        return self.source

    def getRunMode(self):
        return MODE_RESTART

    def _hasStreamingCheckpoint(self):
        return False

    def _doClassification(self, batch):
        return XmippProtClassifyPcaStreaming._doClassification(
            self,
            batch,
        )

    def _insertClassificationSteps(
            self,
            newParticlesSet,
            lastCreationTime,
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

    def getObjId(self):
        return self._objId


class _SyncRow:
    def __init__(self, itemId):
        self.itemId = itemId

    def get(self, _key):
        return self.itemId


class _SyncImages:
    def __init__(self, ids):
        self._ids = list(ids)

    def iterItems(self, **kwargs):
        where = kwargs.get("where")
        minimum = 0

        if where:
            minimum = int(
                where.split('>"', 1)[1].rstrip('"')
            )

        for objId in self._ids:
            if objId > minimum:
                yield _SyncParticle(objId)


class _SyncClasses:
    def __init__(self, imageIds):
        self._images = _SyncImages(imageIds)
        self.pairs = []

    def getImages(self):
        return self._images

    def classifyItems(
            self,
            updateItemCallback=None,
            updateClassCallback=None,
            itemDataIterator=None,
            iterParams=None,
            doClone=False,
            raiseOnNextFailure=False,
            **kwargs,
    ):
        for item in self._images.iterItems(
                **(iterParams or {})
        ):
            try:
                row = next(itemDataIterator)
            except StopIteration:
                break

            self.pairs.append(
                (item.getObjId(), row.itemId)
            )

            if updateItemCallback:
                updateItemCallback(item, row)



class TestXmippClassifyPcaResume(BaseTest):
    """Regression tests for PCA2D streaming Continue/Restart handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newPcaProtocol(self):
        return self.newProtocol(XmippProtClassifyPcaStreaming)

    def _prepareGeneratorProtocol(self, originalRunMode, hasCheckpoint):
        prot = self._newPcaProtocol()

        # Reproduce Protocol._runSteps(): Scipion changes runMode to RESUME
        # while _originalRunMode keeps the action selected by the user.
        prot._originalRunMode = originalRunMode
        prot.runMode.set(MODE_RESUME)

        # We only want to exercise the startup/Resume decision here.
        prot.finish = True
        prot._initialStep = lambda: None
        prot._loadEmptyParticleSet = lambda: object()
        prot._hasStreamingCheckpoint = lambda: hasCheckpoint

        resumeCalls = []
        prot._updateVarsToContinue = lambda: resumeCalls.append(True)

        return prot, resumeCalls

    def testStreamingReadsTheLogicalInputSetInsteadOfReconstructingStorage(self):
        particles = _LogicalParticles()
        harness = _StreamingSetContractHarness(particles)

        module = "xmipp3.protocols.protocol_alignPCA_2D"

        with patch(
            module + ".SetOfParticles",
            side_effect=AssertionError(
                "Streaming must not reconstruct SetOfParticles from a "
                "compatibility storage filename."
            ),
        ):
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
        prot.lastCreationTimeProcessed = 2
        prot._lock = __import__("threading").RLock()
        prot._createModelFile = lambda: None
        prot._loadClassesInfo = lambda _filename: None
        prot._getExtraPath = lambda filename: filename

        def updateParticle(item, row):
            if item.getObjId() != row.itemId:
                item._appendItem = False

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

        firstBatch = object()
        secondBatch = object()

        prot._insertClassificationSteps(firstBatch, "first")
        prot._insertClassificationSteps(secondBatch, "second")

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
                firstBatch,
                "/tmp/imagesInput_0.xmd",
                "/tmp/images_0.mrc",
            ),
            firstClassStep[1],
            "The first classification step must keep the filenames "
            "of round 0 even after the generator schedules later rounds.",
        )
        self.assertEqual(
            (
                secondBatch,
                "/tmp/imagesInput_1.xmd",
                "/tmp/images_1.mrc",
            ),
            secondClassStep[1],
            "Each classification step must receive an immutable snapshot "
            "of its own round-specific input filenames.",
        )

    def testResumeRestoresClassificationCheckpoint(self):
        prot = self._newPcaProtocol()

        prot._hasStreamingCheckpoint = lambda: True
        prot._getLastDone = lambda: "2026-09-16 10:11:12.123456"
        prot._getLastClassificationRound = lambda: 4

        prot._updateVarsToContinue()

        self.assertEqual("2026-09-16 10:11:12.123456", prot.lastCreationTime)
        self.assertEqual(5, prot.classificationRound)

    def testResumeWithoutCheckpointStartsFromInitialState(self):
        prot = self._newPcaProtocol()

        prot.lastCreationTime = "stale-value"
        prot.classificationRound = 99
        prot._hasStreamingCheckpoint = lambda: False

        prot._updateVarsToContinue()

        self.assertEqual("", prot.lastCreationTime)
        self.assertEqual(1, prot.classificationRound)

    def testResumeUpdateClassesPreservesUpdatedReferences(self):
        prot = self._newPcaProtocol()

        prot.mode.set(prot.UPDATE_CLASSES)
        prot.firstTimeDone = False
        prot._hasStreamingCheckpoint = lambda: True
        prot._getLastDone = lambda: "2026-09-16 10:11:12.123456"
        prot._getLastClassificationRound = lambda: 4

        prot._updateVarsToContinue()

        self.assertTrue(prot.firstTimeDone, "Continue in UPDATE_CLASSES mode must preserve the classes produced by previous rounds.")

    def testStepsGeneratorRestoresStateOnRealResume(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESUME, True)

        prot.stepsGeneratorStep()

        self.assertEqual(1, len(resumeCalls), "Continue with a checkpoint must restore the previous PCA2D streaming state.")

    def testStepsGeneratorDoesNotRestoreFreshDefaultResume(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESUME, False)

        prot.stepsGeneratorStep()

        self.assertEqual(0, len(resumeCalls), "A fresh protocol uses MODE_RESUME by default but must not restore nonexistent streaming state.")

    def testStepsGeneratorDoesNotRestoreStateOnRestart(self):
        prot, resumeCalls = self._prepareGeneratorProtocol(MODE_RESTART, True)

        prot.stepsGeneratorStep()

        self.assertEqual(0, len(resumeCalls), "Restart must not restore the previous PCA2D streaming state.")
