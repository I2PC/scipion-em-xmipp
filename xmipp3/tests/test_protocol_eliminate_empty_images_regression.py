

import unittest
from unittest.mock import Mock

from xmipp3.protocols.protocol_eliminate_empty_images import (
    XmippProtEliminateEmptyBase,
)


class TestXmippEliminateEmptyImagesFinalizationRegression(unittest.TestCase):

    def testFinishedStepsCheckIsNoOp(self):
        class _Harness:
            finished = True

            def __init__(self):
                self._checkNewInput = Mock()
                self._checkNewOutput = Mock()

        protocol = _Harness()

        XmippProtEliminateEmptyBase._stepsCheck(protocol)

        protocol._checkNewInput.assert_not_called()
        protocol._checkNewOutput.assert_not_called()

