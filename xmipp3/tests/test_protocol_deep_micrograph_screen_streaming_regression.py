import os
import unittest

from xmipp3.protocols.protocol_deep_micrograph_screen import (
    XmippProtDeepMicrographScreen,
)


class TestDeepMicrographCleanerCoordinateNamingRegression(unittest.TestCase):

    def testInputCoordinateFilenameMatchesMicrographBasename(self):
        class _Mic:
            def getObjId(self):
                return 123

            def getFileName(self):
                return (
                    "Runs/003610_ProtMotionCorrNewStreaming/extra/"
                    "ja_115-1_0000_May14_21.14.32_aligned_mic_DW.mrc"
                )

        class _Harness:
            def _getExtraPath(self, *parts):
                return os.path.join("extra", *parts)

            def _itemScopedPath(self, item, baseName, pathFunc=None):
                pathFunc = pathFunc or self._getExtraPath
                return pathFunc(
                    "%06d__%s" % (item.getObjId(), baseName)
                )

        mic = _Mic()
        coordPath = XmippProtDeepMicrographScreen._getMicPos(
            _Harness(),
            mic,
        )

        self.assertEqual(
            os.path.basename(coordPath),
            "ja_115-1_0000_May14_21.14.32_aligned_mic_DW.pos",
            "micrograph_cleaner_em matches coordinates to micrographs "
            "strictly by basename; ID-scoped coordinate filenames are "
            "not compatible with the external cleaner.",
        )


if __name__ == "__main__":
    unittest.main()
