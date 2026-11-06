# **************************************************************************
# Regression tests for logical output handling in manual particle picking.
# **************************************************************************

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from xmipp3.protocols.protocol_particle_pick import XmippProtParticlePicking


class _Coords:
    def getBoxSize(self):
        return 128


class TestParticlePickingLogicalOutputs(unittest.TestCase):

    @patch('xmipp3.protocols.protocol_particle_pick.launchSupervisedPickerGUI')
    @patch('xmipp3.protocols.protocol_particle_pick.exists', return_value=False)
    def test_DiscardedCoordinatesUseLogicalOutputPresence(self, mockedExists, mockedLauncher):
        process = Mock()
        mockedLauncher.return_value = process
        coords = _Coords()
        definedOutputs = {}
        sourceRelations = []

        prot = SimpleNamespace(
            saveDiscarded=True,
            doInteractive=True,
            _getExtraPath=lambda: '/tmp',
            _getPath=lambda name: '/tmp/' + name,
            createDiscardedStep=Mock(),
            getCoords=lambda: coords,
            _defineOutputs=lambda **kwargs: definedOutputs.update(kwargs),
            _defineSourceRelation=lambda source, target: sourceRelations.append((source, target)),
            inputMicrographs=SimpleNamespace(get=lambda: object()),
        )

        XmippProtParticlePicking.launchParticlePickGUIStep(prot, 'logical-input')

        process.wait.assert_called_once_with()
        prot.createDiscardedStep.assert_called_once_with()
        self.assertIn('boxsize', definedOutputs)
        self.assertEqual(128, definedOutputs['boxsize'].get())


if __name__ == '__main__':
    unittest.main()
