# ***************************************************************************
# *
# * Authors:     Carlos Oscar Sanchez Sorzano (coss@cnb.csic.es)
# *              David Herreros Calero (dherreros@cnb.csic.es)
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
# ***************************************************************************

import os

from pyworkflow.tests import BaseTest, DataSet, setupTestProject

import pyworkflow.utils as pwutils
from pwem.protocols import ProtImportParticles, ProtSubSet
from xmipp3.protocols import XmippProtDeepCenter

OUTPARTERROR = "There was a problem with particles output"
class TestXmippProtDeepCenter(BaseTest):
    """ Testing Deep center
    """
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)

        # Data
        cls.dataset = DataSet.getDataSet('10010')
        cls.particles = cls.dataset.getFile('particles')

        cls.protImportParts = cls.runImportParticles()

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

    def test(self):
        subset = self.newProtocol(ProtSubSet,
                                  inputFullSet=self.protImportParts.outputParticles,
                                  chooseAtRandom=True,
                                  nElements=400)
        self.launchProtocol(subset)

        deepCenter = self.newProtocol(XmippProtDeepCenter,
                                      inputParticles=subset.outputParticles,
                                      numEpochs=5)
        self.launchProtocol(deepCenter)
        self.assertIsNotNone(deepCenter.outputParticles, "There was a problem with deepCenter")

        fnModel = deepCenter._getExtraPath("model.h5")
        self.assertTrue(os.path.exists(fnModel), fnModel + " does not exist")

        self.assertIsNotNone(deepCenter.outputParticles,
                             "There was a problem with deepCenter")
