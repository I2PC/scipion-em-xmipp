

from pyworkflow.tests import BaseTest, DataSet, setupTestProject
from xmipp3.protocols import XmippProtDeepDenoising
from xmipp3.protocols import XmippProtGenerateReprojections
from pyworkflow.em.protocol import ProtImportVolumes, ProtImportParticles
from pyworkflow.em import exists

class TestDeepDenoisingBase(BaseTest):
    @classmethod
    def importData(cls):
        cls.importVolume = cls.newProtocol(ProtImportVolumes,
                                filesPath=cls.dsRelion.getFile(
                                    'import/refine3d/extra/relion_class001.mrc'),
                                samplingRate=1.0)
        cls.launchProtocol(cls.importVolume)

        cls.importParticles = cls.newProtocol(ProtImportParticles,
                                filesPath=cls.dsRelion.getFile(
                                    'import/case2/particles.mrcs'),
                                samplingRate=1.0)
        cls.launchProtocol(cls.importParticles)

    @classmethod
    def generateProjections(cls):

        cls.projections = cls.newProtocol(XmippProtGenerateReprojections,
                                       inputSet=cls.importParticles.outputParticles,
                                       inputVolume=cls.importVolume.outputVolume)
        cls.launchProtocol(cls.projections)

    @classmethod
    def denoiseSet(cls):

        cls.train = cls.newProtocol(XmippProtDeepDenoising,
                                      model='0',
                                      inputProjections=cls.projections.outputProjections,
                                      inputParticles=cls.projections.outputParticles,
                                      imageSize=32)
        cls.launchProtocol(cls.train)
        cls.assertTrue(exists(cls.train._getExtraPath(
            'particlesDenoised.xmd')),"Denoising particles has failed")

class TestDeepDenoising(TestDeepDenoisingBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')
        cls.protData = cls.importData()
        cls.projections = cls.generateProjections()
        cls.denoise = cls.denoiseSet()