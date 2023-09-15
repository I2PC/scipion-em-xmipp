# **************************************************************************
# *
# * Authors:    Adrian Sansinena Rodriguez  (adrian.sansinena@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from pyworkflow.tests import BaseTest, DataSet, setupTestProject

from pwem.protocols import (ProtImportParticles, ProtSubSet,
                            exists)

from pwem import emlib
#from xmipp3.protocols import XmippProtDeepCenter


class TestDeepCenter(BaseTest):
    @classmethod
    def runImportParticles(cls):
        """ Import Particles.
        """
        args = {'importFrom': ProtImportParticles.IMPORT_FROM_SCIPION,
                'sqliteFile': cls.particles,
                'amplitudConstrast': 0.1,
                'sphericalAberration': 2.0,
                'voltage': 200,
                'samplingRate': 0.99,
                'haveDataBeenPhaseFlipped': True
                }

        # Id's should be set increasing from 1 if ### is not in the
        # pattern
        protImport = cls.newProtocol(ProtImportParticles, **args)
        protImport.setObjLabel('import particles')
        cls.launchProtocol(protImport)
        return protImport

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

        # Data
        cls.dataset = DataSet.getDataSet('10010')
        cls.particles = cls.dataset.getFile('particles')

        cls.protImportParts = cls.runImportParticles()

    def test(self):
        subset = self.newProtocol(ProtSubSet,
                                  inputFullSet=self.protImportParts.outputParticles,
                                  chooseAtRandom=True,
                                  nElements=400)
        self.launchProtocol(subset)

        #deepCenter = self.newProtocol(XmippProtDeepCenter,
        #                           inputTrainSet=subset.outputParticles,
        #                           Xdim=128,
        #                           modelPretrain=False,
        #                           numCenModels=5,
        #                           numEpochs_cen=1,
        #                           batchSize=32,
        #                           learningRate=0.001,
        #                           sigma=8,
        #                           patience=5)
        #self.launchProtocol(deepCenter)
#
        #fnModel0 = deepCenter._getExtraPath("modelCenter0.h5")
        #fnModel2 = deepCenter._getExtraPath("modelCenter2.h5")
#
        #if not exists(fnModel0):
        #    self.assertTrue(False, fnModel0 + " does not exist")
        #if not exists(fnModel2):
        #    self.assertTrue(False, fnModel2 + " does not exist")
#
        #deepCenter2 = self.newProtocol(XmippProtDeepCenter,
        #                              inputTrainSet=subset.outputParticles,
        #                              Xdim=128,
        #                              modelPretrain=True,
        #                              pretrainedModels=deepCenter,
        #                              numCenModels=5,
        #                              numEpochs_cen=10,
        #                              batchSize=32,
        #                              learningRate=0.001,
        #                              sigma=8,
        #                              patience=5)
        #self.launchProtocol(deepCenter2)
#
        #fnModel0 = deepCenter2._getExtraPath("modelCenter0.h5")
        #if not exists(fnModel0):
        #    self.assertTrue(False, fnModel0 + " does not exist")