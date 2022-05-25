

from pyworkflow.tests import *
from pwem.protocols import ProtImportVolumes
from xmipp3.protocols import protocol_deep_hand



class TestDeepHand(BaseTest):
    """ Test to get the hand of a volume
    """

    @classmethod
    def setUpClass(self, dataProject='xmipp_tutorial'):
        self.volumesFn = self.dataset.getFile('volumes')

        tests.setupTestProject(self)
        self.dataset = DataSet.getDataSet('relion_tutorial')
        self.volume = self.dataset.getFile(os.path.join('volumes', 'reference_masked.vol'))



    def runImportVolumes(self, emdbId):
        """ Run an Import volumes protocol. """
        self.protImport = self.newProtocol(ProtImportVolumes,
                                         importFrom=ProtImportVolumes.IMPORT_FROM_EMDB,
                                         emdbId=emdbId
                                         )
        self.launchProtocol(self.protImport)
        return self.protImport

    def test_deepHand(self):
        print("Run feepHand core analysis")

        protDeepHand = self.newProtocol(protocol_deep_hand,
                                        inputVolume=3, threshold=1,
                                        thresholdAlpha=False, thresholdHand=4)
        self.launchProtocol(self.protDeepHand)

        self.assertGreater(self.hand.get(), self.thresholdHand.get(), 'message')