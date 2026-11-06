

import unittest
from unittest.mock import Mock

from xmipp3.protocols.protocol_eliminate_empty_images import (
    XmippProtEliminateEmptyBase,
)


class TestXmippEliminateEmptyImagesFinalizationRegression(unittest.TestCase):

    def testStepsGeneratorStopsImmediatelyWhenAlreadyFinished(self):
        # The old _stepsCheck's own "finished -> no-op" short-circuit is
        # now just the while-loop condition in stepsGeneratorStep.
        class _Harness:
            finished = True
            lenPartsSet = 0

            def __init__(self):
                self._checkNewInput = Mock()
                self._checkNewOutput = Mock()
                self._prepareStreamingGenerator = Mock()
                self._insertNewPartsSteps = Mock(return_value=[])
                self._insertFunctionStep = Mock(return_value=1)
                self.createOutputStep = Mock()

        protocol = _Harness()

        XmippProtEliminateEmptyBase.stepsGeneratorStep(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()
        protocol._insertFunctionStep.assert_called_once()

