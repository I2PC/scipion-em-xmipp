# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************
"""Preprocessed micrographs keep their file names: other protocols find their
files by them (e.g. Xmipp's .pos coordinates). Only a second micrograph with
the same name, from another folder, gets its id in front."""

import os

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_preprocess_micrographs import XmippProtPreprocessMicrographs


class _Mic:
    def __init__(self, objId, fileName):
        self._objId, self._fileName = objId, fileName

    def getObjId(self):
        return self._objId

    def getFileName(self):
        return self._fileName


class _PublishedOutput:
    def __init__(self, fileNames):
        self._fileNames = fileNames

    def getUniqueValues(self, attribute, where=None):
        assert attribute == '_filename'
        return list(self._fileNames)


class TestPreprocessedMicrographNames(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _protocol(self):
        return self.newProtocol(XmippProtPreprocessMicrographs, doInvert=True)

    def test_AMicrographKeepsItsFileName(self):
        protocol = self._protocol()

        outputFn = protocol._getOutputMicrograph(_Mic(1, '/data/a/BPV_1386.tif'))

        self.assertEqual('BPV_1386.mrc', os.path.basename(outputFn))
        self.assertEqual(outputFn, protocol._getOutputMicrograph(
            _Mic(1, '/data/a/BPV_1386.tif')))

    def test_OnlyASecondMicrographWithTheSameNameGetsItsId(self):
        protocol = self._protocol()

        first = protocol._getOutputMicrograph(_Mic(1, '/data/a/mic.mrc'))
        second = protocol._getOutputMicrograph(_Mic(2, '/data/b/mic.mrc'))

        self.assertEqual('mic.mrc', os.path.basename(first))
        self.assertEqual('000002__mic.mrc', os.path.basename(second))

    def test_ANamePublishedBeforeAContinueStaysTaken(self):
        protocol = self._protocol()
        protocol.outputMicrographs = _PublishedOutput(
            [protocol._getExtraPath('mic.mrc')])

        outputFn = protocol._getOutputMicrograph(_Mic(2, '/data/b/mic.mrc'))

        self.assertEqual('000002__mic.mrc', os.path.basename(outputFn))

    def test_AMicrographWrittenWithItsIdKeepsThatName(self):
        # Runs made while every output carried its id.
        protocol = self._protocol()
        scoped = protocol._getExtraPath('000001__mic.mrc')
        os.makedirs(os.path.dirname(scoped), exist_ok=True)
        open(scoped, 'w').close()

        try:
            self.assertEqual(scoped, protocol._getOutputMicrograph(
                _Mic(1, '/data/a/mic.mrc')))
        finally:
            os.remove(scoped)
