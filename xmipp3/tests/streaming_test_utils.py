# **************************************************************************
# *
# * Shared helpers for streaming regression tests.
# *
# **************************************************************************


class FakeOutputSet:
    def __init__(self, ids=None):
        self.ids = set(ids or [])
        self.appended = []

    def getSize(self):
        return len(self.ids)

    def getIdSet(self):
        return set(self.ids)

    def append(self, item):
        itemId = item.getObjId()
        self.ids.add(itemId)
        self.appended.append(itemId)

    def setSamplingRate(self, samplingRate):
        self.samplingRate = samplingRate

    def isEmpty(self):
        return False

    def iterItems(self, orderBy='id'):
        return iter([])


class OutputStep:
    def __init__(self):
        self.prerequisites = []
        self.status = None

    def addPrerequisites(self, *deps):
        self.prerequisites.extend(deps)

    def isWaiting(self):
        return True

    def setStatus(self, status):
        self.status = status


class LogicalOutputSetProbe:
    def __init__(self):
        self.enableAppendCalls = 0
        self.copyInfoCalls = 0

    def enableAppend(self):
        self.enableAppendCalls += 1

    def copyInfo(self, inputSet):
        self.copyInfoCalls += 1


class FreshOutputSetProbe:
    STREAM_OPEN = 1

    def __init__(self, filename=None):
        self.filename = filename
        self.streamState = None

    def setStreamState(self, state):
        self.streamState = state

    def copyInfo(self, inputSet):
        self.inputSet = inputSet
