# **************************************************************************
# Backend-independence regression tests for XmippProtCTFDefocusGroup.
# **************************************************************************

import inspect
import unittest

from pwem.objects import SetOfDefocusGroup

from xmipp3.protocols import XmippProtCTFDefocusGroup


class TestCTFDefocusGroupBackendIndependence(unittest.TestCase):

    def test_CTFDefocusGroupDeclaresLogicalOutput(self):
        self.assertIn("outputDefocusGroups", XmippProtCTFDefocusGroup._possibleOutputs)
        self.assertIs(XmippProtCTFDefocusGroup._possibleOutputs["outputDefocusGroups"], SetOfDefocusGroup)

    def test_CTFDefocusGroupDoesNotBindOutputToSqliteFilename(self):
        source = inspect.getsource(XmippProtCTFDefocusGroup.createOutputStep)
        self.assertNotIn("SetOfDefocusGroup(filename=", source)
        self.assertNotIn("defocus_groups.sqlite", source)
        self.assertIn("SetOfDefocusGroup.create(", source)


if __name__ == "__main__":
    unittest.main()
