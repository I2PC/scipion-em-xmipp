

from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from xmipp3.protocols import XmippProtDeepDenoising
from xmipp3.protocols import XmippProtGenerateReprojections
from pyworkflow.em.protocol import ProtImportVolumes, ProtImportParticles
from pyworkflow.em import exists

class TestDeepDenoising(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')

    def importData(self):
        self.importVolume = self.newProtocol(ProtImportVolumes,
                                filesPath=self.dsRelion.getFile(
                                    'import/refine3d/extra/relion_class001.mrc'),
                                samplingRate=1.0)
        self.launchProtocol(self.importVolume)

        self.importParticles = self.newProtocol(ProtImportParticles,
                                filesPath=self.dsRelion.getFile(
                                    'import/case2/particles.mrcs'),
                                samplingRate=1.0)
        self.launchProtocol(self.importParticles)


    def generateProjections(self):

        self.projections = self.newProtocol(XmippProtGenerateReprojections,
                                       inputSet=self.importParticles.outputParticles,
                                       inputVolume=self.importVolume.outputVolume)
        self.launchProtocol(self.projections)

    def denoiseSet(self):

        self.train = self.newProtocol(XmippProtDeepDenoising,
                                      model='0',
                                      inputProjections=self.projections.outputProjections,
                                      inputParticles=self.projections.outputParticles,
                                      imageSize=32)
        self.launchProtocol(self.train)
        self.assertTrue(exists(self.train._getExtraPath(
            'particlesDenoised.xmd')),"Denoising particles has failed")

