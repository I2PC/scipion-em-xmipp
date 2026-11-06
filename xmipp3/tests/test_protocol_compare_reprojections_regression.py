# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from unittest.mock import Mock, patch

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_compare_reprojections import (
    XmippProtCompareReprojections,
)


class _FakePointer:
    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj


class _FakeVolume:
    def __init__(self, objId, dim=(10, 10, 10)):
        self._objId = objId
        self._dim = dim

    def getObjId(self):
        return self._objId

    def getDim(self):
        return self._dim

    def clone(self):
        return _FakeVolume(self._objId, self._dim)


class _FakeVolSet:
    def __init__(self, items):
        self._items = items

    def __contains__(self, itemId):
        return itemId in self._items

    def getItem(self, field, value):
        return self._items[value]


class _CorruptedVolume(_FakeVolume):
    def getDim(self):
        raise ValueError("corrupted volume header")


class _FakeImageHandler:
    def convert(self, vol, fnVol):
        pass


class TestXmippCompareReprojectionsRegression(BaseTest):
    """Regression tests for Compare Reprojections getItem-visibility handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtCompareReprojections)
        prot.inputSet2D = _FakePointer(object())
        return prot

    def testConvertStepSkipsUnresolvableVolumeAndKeepsProcessingOthers(self):
        # Regression test: Set.getItem raises (UnboundLocalError) rather
        # than returning None for a row it cannot find. A volId that is
        # not resolvable in the input set must be skipped (with a clear
        # error) rather than crashing convertStep, and must also be
        # dropped from every per-volume dict so later steps do not try to
        # process a file that was never created.
        prot = self._newProtocol()
        prot.doDownSample = False
        prot._getDimensionsImages = lambda: 10
        prot.imgsOrigFn = self.proj.getTmpPath('residuals.xmd')
        volSet = _FakeVolSet({2: _FakeVolume(2)})  # volId 1 is missing
        prot.inputSet3D = _FakePointer(volSet)
        prot.fnVolDict = {1: '/tmp/volume_1.vol', 2: '/tmp/volume_2.vol'}
        prot.anglesDict = {1: '/tmp/angles_1.xmd', 2: '/tmp/angles_2.xmd'}
        prot.galleryDict = {1: '/tmp/gallery_1.stk', 2: '/tmp/gallery_2.stk'}
        prot.outputNamesDict = {1: 'reprojections_vol1', 2: 'reprojections_vol2'}

        with patch(
                'xmipp3.protocols.protocol_compare_reprojections.writeSetOfParticles',
        ), patch(
                'xmipp3.protocols.protocol_compare_reprojections.ImageHandler',
                _FakeImageHandler,
        ):
            prot.convertStep()

        self.assertEqual({2}, set(prot.fnVolDict.keys()))
        self.assertEqual({2}, set(prot.anglesDict.keys()))
        self.assertEqual({2}, set(prot.galleryDict.keys()))
        self.assertEqual({2}, set(prot.outputNamesDict.keys()))

    def testConvertStepRaisesOnlyWhenEveryVolumeIsUnresolvable(self):
        prot = self._newProtocol()
        prot.doDownSample = False
        prot._getDimensionsImages = lambda: 10
        prot.imgsOrigFn = self.proj.getTmpPath('residuals.xmd')
        volSet = _FakeVolSet({})  # nothing resolvable
        prot.inputSet3D = _FakePointer(volSet)
        prot.fnVolDict = {1: '/tmp/volume_1.vol'}
        prot.anglesDict = {1: '/tmp/angles_1.xmd'}
        prot.galleryDict = {1: '/tmp/gallery_1.stk'}
        prot.outputNamesDict = {1: 'reprojections_vol1'}

        with patch(
                'xmipp3.protocols.protocol_compare_reprojections.writeSetOfParticles',
        ), patch(
                'xmipp3.protocols.protocol_compare_reprojections.ImageHandler',
                _FakeImageHandler,
        ):
            with self.assertRaises(Exception):
                prot.convertStep()

    def testConvertStepSkipsVolumeThatFailsDuringConversionAndKeepsProcessingOthers(self):
        # Regression test: a genuine processing failure while converting
        # one volume (e.g. a corrupted volume header) must not crash the
        # whole convertStep - and hence the whole protocol via
        # pyworkflow's fail-on-any-exception step boundary. It must be
        # skipped (and dropped from every per-volume dict, like an
        # unresolvable volume) while the rest are still converted.
        prot = self._newProtocol()
        prot.doDownSample = False
        prot._getDimensionsImages = lambda: 10
        prot.imgsOrigFn = self.proj.getTmpPath('residuals.xmd')
        volSet = _FakeVolSet({1: _CorruptedVolume(1), 2: _FakeVolume(2)})
        prot.inputSet3D = _FakePointer(volSet)
        prot.fnVolDict = {1: '/tmp/volume_1.vol', 2: '/tmp/volume_2.vol'}
        prot.anglesDict = {1: '/tmp/angles_1.xmd', 2: '/tmp/angles_2.xmd'}
        prot.galleryDict = {1: '/tmp/gallery_1.stk', 2: '/tmp/gallery_2.stk'}
        prot.outputNamesDict = {1: 'reprojections_vol1', 2: 'reprojections_vol2'}

        with patch(
                'xmipp3.protocols.protocol_compare_reprojections.writeSetOfParticles',
        ), patch(
                'xmipp3.protocols.protocol_compare_reprojections.ImageHandler',
                _FakeImageHandler,
        ):
            prot.convertStep()

        self.assertEqual({2}, set(prot.fnVolDict.keys()))
        self.assertEqual({2}, set(prot.anglesDict.keys()))
        self.assertEqual({2}, set(prot.galleryDict.keys()))
        self.assertEqual({2}, set(prot.outputNamesDict.keys()))

    def testCreateOutputStepSkipsExtractionWhenBestVolumeMissing(self):
        # Regression test: the best-ranked volume id must be checked
        # against the input set before calling getItem, so a stale id
        # cannot crash extraction after ranking has already completed.
        prot = self._newProtocol()
        prot.anglesDict = {}  # skip the per-volume output-creation loop
        prot.doRanking = True
        prot.doExtraction = True
        prot._possibleOutputs = {}
        prot.computeRankingVolumes = lambda outputSetDict: 99
        prot.inputSet3D = _FakePointer(_FakeVolSet({}))  # 99 not present
        prot._extractElementsFrom3D = Mock()
        prot.writeOutputDict = lambda: None

        prot.createOutputStep()

        prot._extractElementsFrom3D.assert_not_called()

    def testCreateOutputStepExtractsWhenBestVolumeFound(self):
        prot = self._newProtocol()
        prot.anglesDict = {}
        prot.doRanking = True
        prot.doExtraction = True
        prot._possibleOutputs = {}
        prot.computeRankingVolumes = lambda outputSetDict: 7
        bestVol = _FakeVolume(7)
        prot.inputSet3D = _FakePointer(_FakeVolSet({7: bestVol}))
        prot._extractElementsFrom3D = Mock(return_value=(None, None))
        prot.writeOutputDict = lambda: None

        prot.createOutputStep()

        prot._extractElementsFrom3D.assert_called_once_with(bestVol)

    def testCreateOutputStepSkipsExtractionWhenExtractionFails(self):
        # Regression test: a genuine processing failure while extracting
        # the best-ranked volume/3D class (e.g. corrupted volume data)
        # must not crash the whole createOutputStep - the ranking outputs
        # already defined earlier in the same step must not be lost.
        prot = self._newProtocol()
        prot.anglesDict = {}
        prot.doRanking = True
        prot.doExtraction = True
        prot._possibleOutputs = {}
        prot.computeRankingVolumes = lambda outputSetDict: 7
        bestVol = _FakeVolume(7)
        prot.inputSet3D = _FakePointer(_FakeVolSet({7: bestVol}))
        prot._extractElementsFrom3D = Mock(
            side_effect=ValueError("corrupted volume data"),
        )
        prot.writeOutputDict = lambda: None

        prot.createOutputStep()  # must not raise

        prot._extractElementsFrom3D.assert_called_once_with(bestVol)


if __name__ == '__main__':
    import unittest
    unittest.main()
