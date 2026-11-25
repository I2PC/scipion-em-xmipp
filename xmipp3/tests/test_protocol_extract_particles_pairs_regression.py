# **************************************************************************
# Backend-independence regression tests for XmippProtExtractParticlesPairs.
# **************************************************************************

import inspect
import unittest

from pwem.objects import ParticlesTiltPair

from xmipp3.protocols import XmippProtExtractParticlesPairs


class TestExtractParticlesPairsBackendIndependence(unittest.TestCase):

    def test_ExtractParticlesPairsDeclaresLogicalPairOutput(self):
        self.assertIn(
            "outputParticlesTiltPair",
            XmippProtExtractParticlesPairs._possibleOutputs,
        )
        self.assertIs(
            XmippProtExtractParticlesPairs._possibleOutputs[
                "outputParticlesTiltPair"
            ],
            ParticlesTiltPair,
        )

    def test_ExtractParticlesPairsDoesNotBindOutputToSqliteFilename(self):
        source = inspect.getsource(
            XmippProtExtractParticlesPairs.createOutputStep
        )

        self.assertNotIn(
            "ParticlesTiltPair(filename=",
            source,
        )
        self.assertNotIn(
            "particles_pairs.sqlite",
            source,
        )
        self.assertIn(
            "ParticlesTiltPair.create(",
            source,
        )


if __name__ == "__main__":
    unittest.main()
