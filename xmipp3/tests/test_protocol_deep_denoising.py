

from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from xmipp3.protocols import XmippProtDeepDenoising
from xmipp3.protocols import XmippProtGenerateReprojections
from pwem.protocols import ProtImportVolumes, ProtImportParticles, exists


class TestDeepDenoising(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        dsRelion = DataSet.getDataSet('relion_tutorial')
        cls.relionVolFn = dsRelion.getFile('import/refine3d/extra/relion_class001.mrc')
        cls.relionPartsFn = dsRelion.getFile('import/case2/particles.mrcs')

    @classmethod
    def importData(cls):
        importVolume = cls.newProtocol(ProtImportVolumes,
                                       filesPath=cls.relionVolFn,
                                       samplingRate=1.0)
        cls.launchProtocol(importVolume)
        cls.inVolume = importVolume.outputVolume

        importParticles = cls.newProtocol(ProtImportParticles,
                                          filesPath=cls.relionPartsFn,
                                          samplingRate=1.0)
        cls.launchProtocol(importParticles)
        cls.inParticles = importParticles.outputParticles

    @classmethod
    def generateProjections(cls):
        protGenProj = cls.newProtocol(XmippProtGenerateReprojections,
                                      inputSet=cls.inParticles,
                                      inputVolume=cls.inVolume)
        cls.launchProtocol(protGenProj)
        cls.projections = protGenProj.outputProjections
        cls.particles = protGenProj.outputParticles

    def test_train_and_predict(self):
        self.importData()
        self.generateProjections()

        trainAndPredictProt = self.newProtocol(XmippProtDeepDenoising,
                                               model='0',
                                               inputProjections=self.projections,
                                               inputParticles=self.particles,
                                               imageSize=32)
        self.launchProtocol(trainAndPredictProt)
        self.assertSetSize(trainAndPredictProt.outputParticles, msg="Denoising training +  predict output seems wrong")

    def test_only_predict(self):
        self.importData()
        self.generateProjections()

        predictProt = self.newProtocol(XmippProtDeepDenoising,
                                       model='0',
                                       nEpochs=1,
                                       inputProjections=self.projections,
                                       inputParticles=self.particles,
                                       imageSize=32)
        self.launchProtocol(predictProt)

        self.assertSetSize(predictProt.outputParticles, msg="Denoising only predict output seems wrong")
