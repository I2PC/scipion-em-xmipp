# ***************************************************************************
# * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
# *
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
# ***************************************************************************/

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import ProtImportParticles

from xmipp3.protocols import XmippProtDeepCenterPredict
import xmippLib
       
    
class TestDeepCenterPredict(BaseTest):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')
        cls.dsXmipp = DataSet.getDataSet('xmipp_tutorial')
        
    def importFromRelionRefine3D(self):
        """ Import aligned Particles
        """
        prot = self.newProtocol(ProtImportParticles,
                                 objLabel='particles from relion (auto-refine 3d)',
                                 importFrom=ProtImportParticles.IMPORT_FROM_RELION,
                                 starFile=self.dsRelion.getFile('import/classify3d/extra/relion_it015_data.star'),
                                 magnification=10000,
                                 samplingRate=7.08,
                                 haveDataBeenPhaseFlipped=True
                                 )
        self.launchProtocol(prot)        
        self.assertIsNotNone(prot.outputParticles.getFileName(), 
                             "There was a problem with the import")
        return prot
         
    def test_deepCenter(self):
        """ Test protocol deepCenter_predict
        """
        protImportPars = self.importFromRelionRefine3D()        
        protCenter = self.newProtocol(XmippProtDeepCenterPredict,
                                objLabel='deep center predict',
                                numberOfMpi=1, numberOfThreads=4)
        protCenter.inputParticles.set(protImportPars.outputParticles)
        self.launchProtocol(protCenter)
        self.assertIsNotNone(protCenter.outputParticles.getFileName(), 
                             "There was a problem with the centering")
        

