import unittest
from unittest.mock import patch

from xmipp3.protocols.protocol_extract_particles import XmippProtExtractParticles
from xmipp3.protocols.protocol_streaming_base import XmippStreamingBase


class _Value:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Mic:
    def __init__(self, fileName, micName="mic"):
        self._fileName = fileName
        self._micName = micName

    def getFileName(self):
        return self._fileName

    def getMicName(self):
        return self._micName


class _MissingCoordinatesHarness:
    def __init__(self):
        self.patchSize = _Value(-1)
        self.warnings = []

    def _getExtraPath(self, name):
        return "/tmp/" + name

    def _getMicPos(self, _mic):
        return "/path/that/does/not/exist/mic.pos"

    def _getExtractBoxSize(self):
        return 128

    def _getDownFactor(self):
        return 1.0

    def warning(self, message):
        self.warnings.append(message)




class _Coord:
    def __init__(self, objId, x=10, y=20):
        self._objId = objId
        self._x = x
        self._y = y

    def getObjId(self):
        return self._objId

    def getPosition(self):
        return self._x, self._y


class _OutputParts:
    def append(self, _particle):
        raise AssertionError("No particle should be appended when extracted metadata is missing.")


class _MissingMetadataHarness:
    def __init__(self):
        self.coordDict = {7: [_Coord(101)]}
        self._getPos = lambda coord: coord.getPosition()

    def getBoxScale(self):
        return 1.0

    def error(self, _message):
        pass

    def _getMicXmd(self, _mic):
        return "/path/that/does/not/exist/mic.xmd"



class _RetryCoord:
    def __init__(self, objId, x=10, y=20):
        self._objId = objId
        self._x = x
        self._y = y

    def getObjId(self):
        return self._objId

    def getPosition(self):
        return self._x, self._y

    def scale(self, factor):
        self._x *= factor
        self._y *= factor


class _RetryRow:
    def getValue(self, label):
        from pwem.emlib import metadata as md
        values = {
            md.MDL_XCOOR: 10,
            md.MDL_YCOOR: 20,
            md.MDL_IMAGE: "1@mic.mrcs",
            md.MDL_ENABLED: 1,
        }
        return values.get(label, 0)

    def containsLabel(self, _label):
        return False


class _RetryParticle:
    def copyObjId(self, coord):
        self._objId = coord.getObjId()

    def setLocation(self, _location):
        pass

    def setCoordinate(self, _coord):
        pass

    def setMicId(self, _micId):
        pass

    def setCTF(self, _ctf):
        pass

    def getObjId(self):
        return self._objId


class _RetryOutputParts:
    def __init__(self, durableIds):
        self._durableIds = set(durableIds)
        self.appendedIds = []

    def getIdSet(self):
        return set(self._durableIds)

    def append(self, particle):
        self.appendedIds.append(particle.getObjId())


class _RetryMic(_Mic):
    def __init__(self):
        super().__init__("/tmp/mic.mrc", "mic")

    def getObjId(self):
        return 7

    def getCTF(self):
        return None


class _RetryHarness:
    def __init__(self):
        self.coordDict = {7: [_RetryCoord(101)]}
        self._getPos = lambda coord: coord.getPosition()

    def getBoxScale(self):
        return 1.0

    def error(self, _message):
        pass

    def _getMicXmd(self, _mic):
        return "/tmp/mic.xmd"






class _LateMic:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def getMicName(self):
        return "mic_%03d" % self._objId

    def clone(self):
        return _LateMic(self._objId)


class _LateMicSet:
    def __init__(self, ids):
        self._items = {objId: _LateMic(objId) for objId in ids}

    def loadAllProperties(self):
        pass

    def getUniqueValues(self, attribute, where=None):
        if attribute != "id":
            raise AssertionError("Unexpected attribute: %s" % attribute)
        ids = sorted(self._items)
        if where is None:
            return ids
        prefix = "id > "
        if where.startswith(prefix):
            watermark = int(where[len(prefix):])
            return [objId for objId in ids if objId > watermark]
        raise AssertionError("Unexpected filter: %s" % where)

    def iterItems(self, orderBy="id", direction="ASC", where=None):
        ids = sorted(self._items)
        if where and where.startswith("id IN (") and where.endswith(")"):
            wanted = {int(value) for value in where[7:-1].split(",") if value}
            ids = [objId for objId in ids if objId in wanted]
        return iter([self._items[objId] for objId in ids])

    def getItem(self, _attribute, objId):
        return self._items.get(objId)

    def getSize(self):
        return len(self._items)

    def isStreamClosed(self):
        return True

    def getAcquisition(self):
        return None

    def close(self):
        pass


class _LatePointer:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _LateCoords:
    def __init__(self, micSet):
        self._pointer = _LatePointer(micSet)

    def getMicrographs(self, asPointer=False):
        return self._pointer if asPointer else self._pointer.get()


class _LateVisibleMicHarness:
    def __init__(self):
        self._micSet = _LateMicSet([1, 2, 3])
        self._coords = _LateCoords(self._micSet)
        self.micDict = {"mic_001": _LateMic(1), "mic_003": _LateMic(3)}
        self.coordDict = {}
        self._micsWatermark = 3
        self._pendingMicIds = set()
        self._otherMicsWatermark = 0
        self._pendingOtherMicIds = set()
        self._otherIdByCoordId = {}
        self._ctfWatermark = 0
        self._pendingCtfIds = set()
        self._ctfIdByCoordId = {}

    def getCoords(self):
        return self._coords

    def _loadLogicalSet(self, pointer):
        return XmippStreamingBase._loadLogicalSet(pointer)

    def _discoverIdsAfter(self, inputSet, lastId):
        return XmippStreamingBase._discoverIdsAfter(self, inputSet, lastId)

    def _reconcileClosedStreamIds(self, inputSet, discoveredIds, knownIds, producerClosed, watermarkAttr="_lastInputId"):
        return XmippStreamingBase._reconcileClosedStreamIds(self, inputSet, discoveredIds, knownIds, producerClosed, watermarkAttr)

    def _loadLogicalSetItemsByIds(self, inputSet, itemIds, batchSize=500):
        return XmippStreamingBase._loadLogicalSetItemsByIds(self, inputSet, itemIds, batchSize)

    def _hydrateLogicalSetItemAcquisition(self, inputSet, item):
        return XmippStreamingBase._hydrateLogicalSetItemAcquisition(inputSet, item)

    def _micsOther(self):
        return False

    def _useCTF(self):
        return False

    def _loadInputCoords(self, micDict):
        self.coordsClosed = True
        return micDict


class TestXmippExtractParticlesStreamingRegression(unittest.TestCase):

    def testMissingConvertedCoordinatesFailsExtractionStep(self):
        protocol = _MissingCoordinatesHarness()
        mic = _Mic("/tmp/mic.mrc", "mic")

        with self.assertRaises(RuntimeError):
            XmippProtExtractParticles._extractMicrograph(protocol, mic, False, "", False)

    def testMissingExtractedMetadataFailsPublication(self):
        protocol = _MissingMetadataHarness()
        mic = _Mic("/tmp/mic.mrc", "mic")
        mic.getObjId = lambda: 7

        with self.assertRaises(RuntimeError):
            XmippProtExtractParticles.readPartsFromMics(protocol, [mic], _OutputParts())

        self.assertIn(7, protocol.coordDict)


    def testRetrySkipsParticlesAlreadyDurableInOutput(self):
        protocol = _RetryHarness()
        outputParts = _RetryOutputParts({101})

        with patch("xmipp3.protocols.protocol_extract_particles.exists", return_value=True), \
             patch("xmipp3.protocols.protocol_extract_particles.md.iterRows", return_value=iter([_RetryRow()])), \
             patch("xmipp3.protocols.protocol_extract_particles.Particle", _RetryParticle), \
             patch("xmipp3.protocols.protocol_extract_particles.xmippToLocation", return_value=(1, "mic.mrcs")), \
             patch("xmipp3.protocols.protocol_extract_particles.setXmippAttributes"):
            XmippProtExtractParticles.readPartsFromMics(protocol, [_RetryMic()], outputParts)

        self.assertEqual(outputParts.appendedIds, [])
        self.assertIn(7, protocol.coordDict)




    def testClosedInputRecoversLateVisibleMicBelowWatermark(self):
        protocol = _LateVisibleMicHarness()

        newMics = XmippProtExtractParticles._loadInputList(protocol)

        self.assertEqual(set(newMics), {"mic_002"})
        self.assertEqual(protocol._micsWatermark, 3)
        self.assertIn(2, protocol._pendingMicIds)


if __name__ == "__main__":
    unittest.main()
