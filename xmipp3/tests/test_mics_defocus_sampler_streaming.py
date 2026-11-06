from unittest import TestCase

from xmipp3.protocols.protocol_mics_defocus_balancer import (
    XmippProtMicDefocusSampler,
)


class _DefocusSamplerHarness(XmippProtMicDefocusSampler):
    def __init__(self):
        self.finished = False
        self.sampled_images = [1, 2, 3]
        self.outputsCreated = False
        self.relationsUpdated = False

    def createOutputs(self, sampledIds):
        self.outputsCreated = True
        return object(), object()

    def updateRelations(self, ctfSet, micSet):
        self.relationsUpdated = True

    def _store(self, *args, **kwargs):
        pass


class TestXmippMicDefocusSamplerStreamingFinalization(TestCase):
    def test_OutputsAreCreatedAndFinishedIsSetWhenSamplingFinishes(self):
        # No join-step to unlock any more (stepsGeneratorStep's own
        # while-loop condition replaces that mechanism) - what still
        # matters is that _checkNewOutput creates the outputs, updates
        # relations and marks the protocol finished in the same call.
        protocol = _DefocusSamplerHarness()

        protocol._checkNewOutput()

        self.assertTrue(protocol.outputsCreated)
        self.assertTrue(protocol.relationsUpdated)
        self.assertTrue(protocol.finished)

    def test_CheckNewOutputIsNoOpOnceAlreadyFinished(self):
        protocol = _DefocusSamplerHarness()
        protocol.finished = True

        protocol._checkNewOutput()

        self.assertFalse(protocol.outputsCreated)
        self.assertFalse(protocol.relationsUpdated)
