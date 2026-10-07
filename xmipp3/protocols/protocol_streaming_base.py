import json
from pyworkflow.protocol import ProtStreamingBase
import time
import os

import pyworkflow.object as pwobj
import pyworkflow.protocol.constants as cons
import pyworkflow.utils as pwutils
from pwem import emlib
from pwem.objects import Acquisition, Movie, MovieAlignment


class XmippStreamingBase:
    """Backend-agnostic helpers for Xmipp streaming protocols."""

    def _streamingMustStop(self):
        """True when the generator has to abandon its polling loop.

        A failed step makes pyworkflow mark the protocol as FAILED and the
        executor break out of its own loop - and then join every running
        thread, the generator's among them. A generator that keeps polling
        is never joined, so the whole run hangs with nothing left to do.
        The same applies once the run has been aborted.
        """
        status = getattr(self, 'status', None)
        value = status.get() if hasattr(status, 'get') else status

        return value in (cons.STATUS_FAILED, cons.STATUS_ABORTED)

    def _itemScopedPath(self, item, baseName, pathFunc=None):
        """Path for a per-item artefact, scoped by the item's own id.

        Two micrographs coming from different folders can share a base
        name (/a/mic001.mrc and /b/mic001.mrc). Naming the artefact after
        the base name alone makes the second silently overwrite the
        first, and the output then points two items at the same file.

        Runs made before this was scoped keep working: when only the old
        unscoped file exists, that one is returned.
        """
        pathFunc = pathFunc or self._getExtraPath
        itemId = item.getObjId() if hasattr(item, 'getObjId') else item
        scoped = pathFunc('%06d__%s' % (itemId, baseName))

        if not os.path.exists(scoped):
            legacy = pathFunc(baseName)

            if os.path.exists(legacy):
                return legacy

        return scoped

    @staticmethod
    def _loadLogicalSet(pointer):
        """Load and return the logical Set referenced by ``pointer``."""
        inputSet = pointer.get()
        inputSet.loadAllProperties()
        return inputSet


    def _discoverIdsAfter(self, inputSet, lastId):
        """Discover logical ids above the current streaming watermark."""
        try:
            ids = list(inputSet.getUniqueValues('id', where='id > %d' % lastId))
        except NotImplementedError:
            ids = [itemId for itemId in inputSet.getUniqueValues('id') if itemId > lastId]
        ids = sorted(ids)
        if ids:
            lastId = max(ids)
        return ids, lastId



    def _reconcileClosedStreamIds(
        self,
        inputSet,
        discoveredIds,
        knownIds,
        producerClosed,
        watermarkAttr='_lastInputId',
    ):
        """Recover late-visible ids only during terminal reconciliation."""
        discoveredIds = list(discoveredIds)

        if not producerClosed:
            return discoveredIds, True

        expectedSize = inputSet.getSize()
        knownIds = set(knownIds)
        visibleKnownIds = knownIds.union(discoveredIds)

        if len(visibleKnownIds) >= expectedSize:
            return discoveredIds, True

        reconciledIds = list(inputSet.getUniqueValues('id'))

        if reconciledIds:
            setattr(
                self,
                watermarkAttr,
                max(
                    getattr(self, watermarkAttr, 0),
                    max(reconciledIds),
                ),
            )

        visibleIds = set(discoveredIds)
        visibleIds.update(reconciledIds)

        newIds = [
            itemId
            for itemId in sorted(visibleIds)
            if itemId not in knownIds
        ]

        terminalConsistent = (
            len(knownIds.union(visibleIds)) >= expectedSize
        )

        return newIds, terminalConsistent



    def _getFirstLogicalItem(self, items):
        return items[0] if isinstance(items, (list, tuple)) else items.getFirstItem()

    @staticmethod
    def _loadLogicalSetItems(inputSet, watermark=None):
        loadAllProperties = getattr(inputSet, "loadAllProperties", None)
        if callable(loadAllProperties):
            loadAllProperties()

        iterItems = getattr(inputSet, "iterItems", None)
        if callable(iterItems) and watermark is not None:
            try:
                items = iterItems(orderBy="id", direction="ASC", where="id > %d" % watermark)
                return [item.clone() if callable(getattr(item, "clone", None)) else item for item in items]
            except NotImplementedError:
                items = iterItems(orderBy="id", direction="ASC")
                return [item.clone() if callable(getattr(item, "clone", None)) else item for item in items if item.getObjId() > watermark]

        items = inputSet
        return [item.clone() if callable(getattr(item, "clone", None)) else item for item in items]



    @staticmethod
    def _hydrateLogicalSetItemAcquisition(inputSet, item):
        getItemAcquisition = getattr(item, "getAcquisition", None)
        setItemAcquisition = getattr(item, "setAcquisition", None)
        getSetAcquisition = getattr(inputSet, "getAcquisition", None)
        if not callable(getItemAcquisition) or not callable(setItemAcquisition) or not callable(getSetAcquisition):
            return item

        setAcquisition = getSetAcquisition()
        if setAcquisition is None:
            return item

        itemAcquisition = getItemAcquisition()
        if itemAcquisition is None:
            setItemAcquisition(setAcquisition)
            return item

        for getterName, setterName in (("getMagnification", "setMagnification"), ("getVoltage", "setVoltage"), ("getSphericalAberration", "setSphericalAberration"), ("getAmplitudeContrast", "setAmplitudeContrast")):
            itemGetter = getattr(itemAcquisition, getterName, None)
            setGetter = getattr(setAcquisition, getterName, None)
            itemSetter = getattr(itemAcquisition, setterName, None)
            if callable(itemGetter) and callable(setGetter) and callable(itemSetter) and itemGetter() is None:
                value = setGetter()
                if value is not None:
                    itemSetter(value)

        return item

    def _loadLogicalSetItemsByIds(self, inputSet, itemIds, batchSize=500):
        itemIds = sorted(set(itemIds))
        if not itemIds:
            return []

        if hasattr(inputSet, "loadAllProperties"):
            inputSet.loadAllProperties()

        def cloneItem(item):
            item = self._hydrateLogicalSetItemAcquisition(inputSet, item)
            clone = getattr(item, "clone", None)
            return clone() if callable(clone) else item

        iterItems = getattr(inputSet, "iterItems", None)
        if callable(iterItems):
            items = []
            try:
                for offset in range(0, len(itemIds), batchSize):
                    batch = itemIds[offset:offset + batchSize]
                    where = "id IN (%s)" % ",".join(str(itemId) for itemId in batch)
                    for item in iterItems(orderBy="id", direction="ASC", where=where):
                        items.append(cloneItem(item))
                return items
            except NotImplementedError:
                pass

        getItem = getattr(inputSet, "getItem", None)
        if callable(getItem):
            items = []
            for itemId in itemIds:
                try:
                    item = getItem("id", itemId)
                except (UnboundLocalError, KeyError):
                    item = None
                if item is not None:
                    items.append(cloneItem(item))
            return items

        wantedIds = set(itemIds)
        return [cloneItem(item) for item in inputSet if item.getObjId() in wantedIds]

    @staticmethod
    def _getOutputIds(outputSet):
        if outputSet is None:
            return set()

        getSize = getattr(outputSet, "getSize", None)
        if callable(getSize) and getSize() == 0:
            return set()

        getIdSet = getattr(outputSet, "getIdSet", None)
        if callable(getIdSet):
            return set(getIdSet())

        return {item.getObjId() for item in outputSet}

    def _loadOrCreateOutputSet(self, outputName, SetClass, suffix="",
                               factoryArgs=()):
        outputSet = getattr(self, outputName, None)
        if outputSet is not None:
            loadAllProperties = getattr(outputSet, "loadAllProperties", None)
            if callable(loadAllProperties):
                loadAllProperties()

            enableAppend = getattr(outputSet, "enableAppend", None)
            if callable(enableAppend):
                enableAppend()

            return outputSet, False

        factory = getattr(self, "_create%s" % SetClass.__name__)
        factoryArgs = tuple(factoryArgs or ())
        if suffix:
            factoryArgs += (suffix,)
        outputSet = factory(*factoryArgs)
        outputSet.setStreamState(outputSet.STREAM_OPEN)
        return outputSet, True

    def _getPersistedOutputIds(self, outputName):
        outputSet = getattr(self, outputName, None)
        if outputSet is None:
            return set()

        loadAllProperties = getattr(outputSet, "loadAllProperties", None)
        if callable(loadAllProperties):
            loadAllProperties()

        return self._getOutputIds(outputSet)


    def _getPersistedOutputSize(self, outputName):
        outputSet = getattr(self, outputName, None)
        if outputSet is None:
            return 0

        loadAllProperties = getattr(outputSet, 'loadAllProperties', None)
        if callable(loadAllProperties):
            loadAllProperties()

        getSize = getattr(outputSet, 'getSize', None)
        if callable(getSize):
            return getSize()

        return len(self._getOutputIds(outputSet))


    def _restorePersistedOutputIds(self, outputName):
        """Restore one output's persisted ids once and cache them."""
        cache = getattr(self, '_persistedOutputIds', None)
        if cache is None:
            cache = {}
            self._persistedOutputIds = cache

        if outputName not in cache:
            cache[outputName] = set(self._getPersistedOutputIds(outputName))

        return set(cache[outputName])


    def _markOutputIdsPersisted(self, outputName, itemIds):
        """Advance the persisted-id cache after publishing items."""
        self._restorePersistedOutputIds(outputName)
        self._persistedOutputIds[outputName].update(itemIds)
        return set(self._persistedOutputIds[outputName])


    def _getKnownPersistedOutputIds(self, outputName):
        """Return cached persisted ids, restoring them on first access."""
        return self._restorePersistedOutputIds(outputName)

    def _getPersistedCandidateIds(self, outputName, candidateIds):
        candidateIds = sorted(set(candidateIds))
        if not candidateIds:
            return set()

        outputSet = getattr(self, outputName, None)
        if outputSet is None:
            return set()

        loadAllProperties = getattr(outputSet, "loadAllProperties", None)
        if callable(loadAllProperties):
            loadAllProperties()

        iterItems = getattr(outputSet, "iterItems", None)
        if callable(iterItems):
            where = "id IN (%s)" % ", ".join(str(itemId) for itemId in candidateIds)
            try:
                items = iterItems(orderBy="id", direction="ASC", where=where)
                return {item.getObjId() for item in items}
            except NotImplementedError:
                pass

        getItem = getattr(outputSet, "getItem", None)
        if callable(getItem):
            persistedIds = set()
            for itemId in candidateIds:
                try:
                    item = getItem("id", itemId)
                except UnboundLocalError:
                    # Missing logical ids may surface as UnboundLocalError
                    # from Set.__getitem__; treat them as not persisted.
                    item = None
                if item is not None:
                    persistedIds.add(itemId)
            return persistedIds

        return self._getOutputIds(outputSet).intersection(candidateIds)

    def _iterKnownStreamingSteps(self):
        """Iterate current and pre-resume steps without backend knowledge."""
        seen = set()
        for attrName in ("_steps", "_prevSteps"):
            for step in getattr(self, attrName, []) or []:
                stepKey = id(step)
                if stepKey in seen:
                    continue
                seen.add(stepKey)
                yield step

    def _isFunctionStepFinished(self, functionName):
        for step in self._iterKnownStreamingSteps():
            funcName = getattr(step, "funcName", None)
            if hasattr(funcName, "get"):
                funcName = funcName.get()
            if funcName == functionName and step.isFinished():
                return True
        return False

    def _getCurrentStreamingStep(self, stepIndex):
        if not isinstance(stepIndex, int) or stepIndex < 1:
            return None
        steps = getattr(self, "_steps", []) or []
        if stepIndex > len(steps):
            return None
        return steps[stepIndex - 1]

    def _ensureInsertedStepTracking(self):
        pendingIds = getattr(self, "_pendingInsertedIds", None)
        stepToItemId = getattr(self, "_streamingStepToItemId", None)
        finishedIds = getattr(self, "_finishedInsertedIds", None)

        if pendingIds is None:
            acknowledged = getattr(self, "_acknowledgedFinishedIds", set())
            pendingIds = set(getattr(self, "insertedDict", {})) - acknowledged
            self._pendingInsertedIds = pendingIds

        if stepToItemId is None:
            stepToItemId = {}
            for itemId in pendingIds:
                stepIndex = getattr(self, "insertedDict", {}).get(itemId)
                if self._getCurrentStreamingStep(stepIndex) is not None:
                    stepToItemId[stepIndex] = itemId
            self._streamingStepToItemId = stepToItemId

        if finishedIds is None:
            finishedIds = set()
            self._finishedInsertedIds = finishedIds

        return pendingIds, stepToItemId, finishedIds

    def _getFinishedInsertedIds(self):
        pendingIds, _, finishedIds = self._ensureInsertedStepTracking()

        # A step becomes runnable the moment it is inserted, which is
        # before the generator gets to record who owns it. One that
        # finishes inside that window is never reported by
        # _recordFinishedInsertedStep, so its item would never be
        # published and the protocol would never reach its end. Reading
        # the steps of what is still pending covers that for good; the
        # cost follows the work in flight, not the whole run.
        for itemId in set(pendingIds) - finishedIds:
            stepIndex = getattr(self, "insertedDict", {}).get(itemId)
            step = self._getCurrentStreamingStep(stepIndex)

            if step is not None and step.isFinished():
                finishedIds.add(itemId)

        return set(finishedIds)

    def _getAllFinishedInsertedIds(self):
        finishedIds = set(getattr(self, "_streamingRestoredFinishedIds", set()))
        for itemId, stepIndex in getattr(self, "insertedDict", {}).items():
            step = self._getCurrentStreamingStep(stepIndex)
            if step is not None and step.isFinished():
                finishedIds.add(itemId)
        return finishedIds

    def _acknowledgeFinishedInsertedIds(self, itemIds):
        itemIds = set(itemIds)
        if not itemIds:
            return

        pendingIds, _, finishedIds = self._ensureInsertedStepTracking()
        if not hasattr(self, "_acknowledgedFinishedIds"):
            self._acknowledgedFinishedIds = set()
        self._acknowledgedFinishedIds.update(itemIds)
        pendingIds.difference_update(itemIds)
        finishedIds.difference_update(itemIds)

    def _trackPendingInsertedIds(self, itemIds):
        itemIds = list(itemIds)
        if not itemIds:
            return

        pendingIds, stepToItemId, _ = self._ensureInsertedStepTracking()
        acknowledged = getattr(self, "_acknowledgedFinishedIds", set())
        for itemId in itemIds:
            if itemId in acknowledged or itemId not in self.insertedDict:
                continue
            pendingIds.add(itemId)
            stepToItemId[self.insertedDict[itemId]] = itemId


    def _recordFinishedInsertedStep(self, stepIndex):
        pendingIds, stepToItemId, finishedIds = self._ensureInsertedStepTracking()
        itemId = stepToItemId.get(stepIndex)

        if itemId is None:
            # The step finished before the generator recorded its owner.
            # insertedDict already knows which item it belongs to.
            for knownId, knownStep in getattr(self, "insertedDict", {}).items():
                if knownStep == stepIndex:
                    itemId = knownId
                    stepToItemId[stepIndex] = knownId
                    break

        if itemId in pendingIds:
            finishedIds.add(itemId)

class XmippStreamingMoviesMixin(XmippStreamingBase):
    """Streaming behavior shared by Xmipp protocols based on ProtProcessMovies."""

    def _cleanMovieFolder(self, movieFolder):
        """Remove a movie's working folder without going through a shell.

        pwem does this with os.system('rm -rf %s' % folder). The path
        comes from _getTmpPath, so it carries the project directory, and
        a project whose path contains a space turns that command into two
        arguments: it then deletes something else entirely and leaves the
        real folder behind. Removing the tree from Python avoids the
        quoting problem altogether, and the path is checked to be inside
        this protocol's own tmp directory before anything is deleted.
        """
        if pwutils.envVarOn('SCIPION_DEBUG_NOCLEAN'):
            self.info('Clean movie data DISABLED. '
                      'Movie folder will remain in disk!!!')
            return

        workspace = os.path.realpath(self._getTmpPath())
        target = os.path.realpath(movieFolder)

        if target != workspace and not target.startswith(workspace + os.sep):
            self.warning("Refusing to remove %s: it is outside this "
                         "protocol's working directory." % movieFolder)
            return

        self.info("Erasing.....movieFolder: %s" % movieFolder)
        pwutils.cleanPath(target)

    def _loadLogicalInputMovies(self, watermark=None):
        inputMovies = self.inputMovies.get()
        movies = self._loadLogicalSetItems(inputMovies, watermark)
        return movies, inputMovies.isStreamClosed()



    def _getResumeRepairCandidateIds(self, finishedIds):
        """Return finished ids that still need output repair after Continue."""
        return set(finishedIds)

    def _restoreFinishedStreamingMovieSteps(self):
        """Restore finished movie work from persisted steps after Continue."""
        if getattr(self, "_streamingResumeStepsRestored", False):
            return

        # Protocol._runSteps() changes runMode to MODE_RESUME while executing,
        # even when the user selected Restart. _originalRunMode preserves
        # the actual action requested by the user. Restart must never reuse
        # processMovieStep state from the previous run.
        from pyworkflow.protocol.constants import MODE_RESTART

        originalRunMode = getattr(self, "_originalRunMode", None)

        if originalRunMode == MODE_RESTART:
            self._streamingResumeStepsRestored = True
            return

        scheduledIds = set()
        finishedIds = set()
        for step in self._iterKnownStreamingSteps():
            funcName = getattr(step, "funcName", None)
            if hasattr(funcName, "get"):
                funcName = funcName.get()
            if funcName != "processMovieStep":
                continue

            argsStr = getattr(step, "argsStr", None)
            if hasattr(argsStr, "get"):
                argsStr = argsStr.get("[]")
            try:
                args = json.loads(argsStr or "[]")
            except (TypeError, ValueError):
                continue
            if not args or not isinstance(args[0], dict):
                continue

            movieDict = args[0]
            movieId = None
            for key in ("object.id", "_objId", "id"):
                if movieDict.get(key) is not None:
                    movieId = movieDict[key]
                    break
            if movieId is None:
                continue
            try:
                movieId = int(movieId)
            except (TypeError, ValueError):
                continue

            scheduledIds.add(movieId)
            if step.isFinished():
                finishedIds.add(movieId)

        acknowledged = getattr(self, "_acknowledgedFinishedIds", None)
        if acknowledged is None:
            acknowledged = set()
            self._acknowledgedFinishedIds = acknowledged

        finishedIds.difference_update(acknowledged)
        self._streamingRestoredFinishedIds = set(finishedIds)

        insertedDict = getattr(self, "insertedDict", None)
        if insertedDict is None:
            insertedDict = {}
            self.insertedDict = insertedDict
        for movieId in finishedIds:
            insertedDict.setdefault(movieId, None)

        repairIds = set(self._getResumeRepairCandidateIds(finishedIds))
        repairIds.intersection_update(finishedIds)
        acknowledged.update(finishedIds - repairIds)

        movieCache = getattr(self, "_streamingMoviesById", None)
        if movieCache is None:
            movieCache = {}
            self._streamingMoviesById = movieCache
        if repairIds and hasattr(self, "inputMovies"):
            inputSet = self.inputMovies.get()
            for movie in self._loadLogicalSetItemsByIds(inputSet, repairIds):
                movieCache[movie.getObjId()] = movie

        self._pendingInsertedIds = set(repairIds)
        self._streamingStepToItemId = {}
        self._finishedInsertedIds = set(repairIds)
        self._finishedInsertedIdsRestored = True

        unfinishedIds = scheduledIds - finishedIds
        if unfinishedIds:
            self._streamingInputWatermark = min(unfinishedIds) - 1
        elif scheduledIds:
            self._streamingInputWatermark = max(scheduledIds)

        self._streamingResumeStepsRestored = True

    def _prepareStreamingGenerator(self):
        pass


    def _finalizeStreamingGenerator(self):
        pass


    def stepsGeneratorStep(self):
        self._prepareStreamingGenerator()
        if self.isContinued():
            self._restorePersistedOutputIds("outputMovies")
        self._restoreFinishedStreamingMovieSteps()

        while not self.finished and not self.isFailed():
            self._checkNewInput()
            self._checkNewOutput()

            if self.finished or self.isFailed():
                break

            if self._getStreamingSleepOnWait() > 0:
                self._streamingSleepOnWait()
            else:
                time.sleep(1)

        if not self.isFailed():
            self._finalizeStreamingGenerator()

    def _stepFinished(self, step):
        doContinue = super()._stepFinished(step)
        if step.isFinished():
            self._recordFinishedInsertedStep(step.getIndex())
        return doContinue

    def _checkNewInput(self):
        if not hasattr(self, "_streamingInputWatermark"):
            self._streamingInputWatermark = max(self.insertedDict) if self.insertedDict else None
        watermark = self._streamingInputWatermark
        newMovies, self.streamClosed = self._loadLogicalInputMovies(watermark)
        discoveredIds = [movie.getObjId() for movie in newMovies]

        if not hasattr(self, "listOfMovies"):
            self.listOfMovies = []

        knownIds = getattr(self, "_streamingInputIds", None)
        if knownIds is None:
            knownIds = set(self.insertedDict)
            knownIds.update(movie.getObjId() for movie in self.listOfMovies)
            self._streamingInputIds = knownIds

        movieById = getattr(self, "_streamingMoviesById", None)
        if movieById is None:
            movieById = {movie.getObjId(): movie for movie in self.listOfMovies}
            self._streamingMoviesById = movieById

        moviesToInsert = []
        for movie in newMovies:
            movieId = movie.getObjId()
            movieById[movieId] = movie
            if movieId not in knownIds:
                self.listOfMovies.append(movie)
                knownIds.add(movieId)
            if movieId not in self.insertedDict:
                moviesToInsert.append(movie)

        if self.streamClosed and watermark is not None and not getattr(self, "_streamingClosedInputReconciled", False):
            allMovies, _ = self._loadLogicalInputMovies(None)
            scheduledIds = {movie.getObjId() for movie in moviesToInsert}
            discoveredIds.extend(movie.getObjId() for movie in allMovies)
            for movie in allMovies:
                movieId = movie.getObjId()
                movieById[movieId] = movie
                if movieId not in knownIds:
                    self.listOfMovies.append(movie)
                    knownIds.add(movieId)
                if movieId not in self.insertedDict and movieId not in scheduledIds:
                    moviesToInsert.append(movie)
                    scheduledIds.add(movieId)
            self._streamingClosedInputReconciled = True

        if discoveredIds:
            newestId = max(discoveredIds)
            if self._streamingInputWatermark is None or newestId > self._streamingInputWatermark:
                self._streamingInputWatermark = newestId

        deps = self._insertNewMoviesSteps(self.insertedDict, moviesToInsert)
        self._trackPendingInsertedIds(movie.getObjId() for movie in moviesToInsert if movie.getObjId() in self.insertedDict)
        if not deps:
            return

        outputStep = self._getFirstJoinStep()
        if outputStep:
            outputStep.addPrerequisites(*deps)
        self.updateSteps()

    def _getPersistedOutputMovieIds(self):
        return self._getPersistedOutputIds("outputMovies")

    def _getFinishedProcessMovieIds(self):
        return self._getFinishedInsertedIds()

    def _checkNewOutput(self):
        if getattr(self, "finished", False):
            return

        persistedIds = self._getPersistedOutputMovieIds()
        finishedIds = self._getFinishedProcessMovieIds()
        newDone = [movie for movie in self.listOfMovies if movie.getObjId() in finishedIds and movie.getObjId() not in persistedIds]
        self._firstTimeOutput = not persistedIds

        if newDone:
            self._updateOutputSets(newDone, pwobj.Set.STREAM_OPEN)
            persistedIds = self._getPersistedOutputMovieIds()

        inputIds = {movie.getObjId() for movie in self.listOfMovies}
        self.finished = self.streamClosed and inputIds.issubset(persistedIds)
        if not self.finished:
            return

        self._updateOutputSets([], pwobj.Set.STREAM_CLOSED)
        outputStep = self._getFirstJoinStep()
        if outputStep is not None and outputStep.isWaiting():
            outputStep.setStatus(cons.STATUS_NEW)

    def processMovieStep(self, movieDict, hasAlignment):
        movie = Movie()
        movie.setAcquisition(Acquisition())

        if hasAlignment:
            movie.setAlignment(MovieAlignment())

        movie.setAttributesFromDict(movieDict, setBasic=True, ignoreMissing=True)
        if self.isContinued() and movie.getObjId() in getattr(self, "_persistedOutputIds", {}).get("outputMovies", set()):
            return

        movieFolder = self._getOutputMovieFolder(movie)
        movieFn = movie.getFileName()
        movieName = os.path.basename(movieFn)

        if not self._filterMovie(movie):
            return

        # Named after the movie id, so a retry or a Continue finds what a
        # previous attempt left behind. The decompression branches below
        # skip their work when the output already exists, so a truncated
        # file from an interrupted run would be reused as if it were good.
        pwutils.cleanPath(movieFolder)
        pwutils.makePath(movieFolder)
        pwutils.createAbsLink(os.path.abspath(movieFn), os.path.join(movieFolder, movieName))

        if movieName.endswith("bz2"):
            newMovieName = movieName.replace(".bz2", "")
            if not os.path.exists(os.path.join(movieFolder, newMovieName)):
                self.runJob("bzip2", "-d -f %s" % movieName, cwd=movieFolder)
        elif movieName.endswith("tbz"):
            newMovieName = movieName.replace(".tbz", ".mrc")
            if not os.path.exists(os.path.join(movieFolder, newMovieName)):
                self.runJob("tar", "jxf %s" % movieName, cwd=movieFolder)
        elif movieName.endswith(".txt"):
            movieTxt = os.path.join(movieFolder, movieName)
            with open(movieTxt) as f:
                movieOrigin = os.path.basename(os.readlink(movieFn))
                newMovieName = movieName.replace(".txt", ".mrcs")
                ih = emlib.image.ImageHandler()
                for i, line in enumerate(f):
                    if line.strip():
                        inputFrame = os.path.join(movieOrigin, line.strip())
                        ih.convert(inputFrame, (i + 1, os.path.join(movieFolder, newMovieName)))
        else:
            newMovieName = movieName

        convertExt = self._getConvertExtension(newMovieName)
        correctGain = self._doCorrectGain()

        if convertExt or correctGain:
            inputMovieFn = os.path.join(movieFolder, newMovieName)
            if inputMovieFn.endswith(".em"):
                inputMovieFn += ":ems"

            newMovieName = pwutils.replaceExt(newMovieName, convertExt) if convertExt else "%s_corrected.%s" % os.path.splitext(newMovieName)
            outputMovieFn = os.path.join(movieFolder, newMovieName)

            if correctGain:
                self.info("Correcting gain and dark '%s' -> '%s'" % (inputMovieFn, outputMovieFn))
                gain, dark = self.getGainAndDark()
                self.correctGain(inputMovieFn, outputMovieFn, gainFn=gain, darkFn=dark)
            else:
                self.info("Converting movie '%s' -> '%s'" % (inputMovieFn, outputMovieFn))
                emlib.image.ImageHandler().convertStack(inputMovieFn, outputMovieFn)

        movie._originalFileName = pwobj.String(objDoStore=False)
        movie._originalFileName.set(movie.getFileName())
        movie.setFileName(os.path.join(movieFolder, newMovieName))
        self.info("Processing movie: %s" % movie.getFileName())
        self._processMovie(movie)

        if self._doMovieFolderCleanUp():
            self._cleanMovieFolder(movieFolder)
