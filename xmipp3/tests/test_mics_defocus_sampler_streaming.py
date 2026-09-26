from unittest import TestCase

from pyworkflow.protocol.constants import STATUS_NEW

from xmipp3.protocols.protocol_mics_defocus_balancer import (
    XmippProtMicDefocusSampler,
)


class _WaitingOutputStep:
    def __init__(self):
        self.status = None

    def isWaiting(self):
        return True

    def setStatus(self, status):
        self.status = status


class _DefocusSamplerHarness(XmippProtMicDefocusSampler):
    def __init__(self):
        self.finished = False
        self.sampled_images = [1, 2, 3]
        self.outputStep = _WaitingOutputStep()
        self.outputsCreated = False
        self.relationsUpdated = False

    def createOutputs(self, sampledIds):
        self.outputsCreated = True
        return object(), object()

    def updateRelations(self, ctfSet, micSet):
        self.relationsUpdated = True

    def _store(self, *args, **kwargs):
        pass

    def _getFirstJoinStep(self):
        return self.outputStep


class TestXmippMicDefocusSamplerStreamingFinalization(TestCase):
    def test_OutputStepIsUnlockedWhenSamplingFinishes(self):
        protocol = _DefocusSamplerHarness()

        protocol._checkNewOutput()

        self.assertTrue(protocol.outputsCreated)
        self.assertTrue(protocol.relationsUpdated)
        self.assertTrue(protocol.finished)
        self.assertEqual(
            STATUS_NEW,
            protocol.outputStep.status,
            "The waiting createOutputStep must be unlocked in the same "
            "_checkNewOutput call that marks sampling as finished.",
        )
