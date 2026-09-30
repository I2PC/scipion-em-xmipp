# *****************************************************************************
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# *****************************************************************************

from types import SimpleNamespace
from unittest.mock import patch

from pyworkflow.tests import BaseTest, setupTestProject

from xmipp3.protocols.protocol_particle_pick_remove_duplicates import (
    XmippProtPickingRemoveDuplicates,
)


class _FakeMic:
    def __init__(self, objId):
        self._objId = objId

    def getObjId(self):
        return self._objId

    def clone(self):
        return _FakeMic(self._objId)


class _FakeMicsDict(dict):
    """Stands in for a SetOfMicrographs: supports __contains__/__getitem__
    like a dict already does, plus the loadAllProperties() refresh call."""

    def loadAllProperties(self):
        pass


class TestXmippPickingRemoveDuplicatesRegression(BaseTest):
    """Regression tests for Remove Duplicates streaming input handling."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

    def _newProtocol(self):
        prot = self.newProtocol(XmippProtPickingRemoveDuplicates)
        prot.checkedMics = set()
        prot.processedMics = set()
        return prot

    def testCheckNewInputSchedulesAllVisibleMicrographs(self):
        prot = self._newProtocol()
        prot._restoreProcessedMics = lambda: None
        prot.getMainInput = lambda: SimpleNamespace(
            getMicrographs=lambda: _FakeMicsDict({1: _FakeMic(1), 2: _FakeMic(2)})
        )
        prot._getFirstJoinStep = lambda: None
        prot.updateSteps = lambda: None

        scheduled = []
        prot.insertNewCoorsSteps = (
            lambda mics: scheduled.extend(mic.getObjId() for mic in mics) or []
        )

        with patch(
                'xmipp3.protocols.protocol_particle_pick_remove_duplicates.getReadyMics',
                return_value=({1, 2}, False),
        ):
            prot._checkNewInput()

        self.assertEqual({1, 2}, set(scheduled))
        self.assertEqual({1, 2}, prot.checkedMics)

    def testCheckNewInputDefersMicrographNotYetVisibleWithoutPermanentLoss(self):
        # Regression test: getMainInput().getMicrographs() resolves a
        # Pointer whose cached value is never refreshed. A micId that
        # getReadyMics() (which does reopen fresh) already reports as
        # ready may still be invisible in that cached micrographs Set -
        # it must be deferred (kept out of checkedMics) so it is retried
        # on the next check, instead of crashing or being permanently
        # skipped.
        prot = self._newProtocol()
        prot._restoreProcessedMics = lambda: None
        prot.getMainInput = lambda: SimpleNamespace(
            getMicrographs=lambda: _FakeMicsDict({2: _FakeMic(2)})  # mic 1 missing
        )
        prot._getFirstJoinStep = lambda: None
        prot.updateSteps = lambda: None
        prot.warning = lambda *args, **kwargs: None

        scheduled = []
        prot.insertNewCoorsSteps = (
            lambda mics: scheduled.extend(mic.getObjId() for mic in mics) or []
        )

        with patch(
                'xmipp3.protocols.protocol_particle_pick_remove_duplicates.getReadyMics',
                return_value=({1, 2}, True),
        ):
            prot._checkNewInput()

        self.assertEqual([2], scheduled)
        self.assertEqual({2}, prot.checkedMics)


if __name__ == '__main__':
    import unittest
    unittest.main()
