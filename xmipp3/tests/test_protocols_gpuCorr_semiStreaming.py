# ***************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
# *              David Strelak (dstrelak@cnb.csic.es)
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

import time

from pwem.protocols import ProtImportAverages

from xmipp3.tests.test_protocols_gpuCorr_fullStreaming import GpuCorrCommon
from xmipp3.protocols.protocol_classification_gpuCorr_semi import *

class TestGpuCorrSemiStreaming(GpuCorrCommon):
    def test_semi_streaming(self):
        self.run_common_workflow()

    def importAverages(self):
        prot = self.newProtocol(ProtImportAverages,
                                filesPath=self.dsRelion.getFile(
                                    'import/averages.mrcs'),
                                samplingRate=1.0)
        self.launchProtocol(prot)

        return prot

    def runClassify(self, inputParts, inputAvgs):
        protClassify = self.newProtocol(XmippProtStrGpuCrrSimple,
                                        useAsRef=REF_AVERAGES)

        protClassify.inputParticles.set(inputParts)
        protClassify.inputRefs.set(inputAvgs)
        self.proj.launchProtocol(protClassify, wait=False)

        return protClassify
       

    def verify_classification(self):
        protImportAvgs = self.importAverages()
        self.assertFalse(protImportAvgs.isFailed())
        protClassify = self.runClassify(self.protExtract.outputParticles,
                                        protImportAvgs.outputAverages)

        self._waitOutput(protClassify, "outputClasses", timeOut=200)

        self.assertTrue(protClassify.hasAttribute('outputClasses'),
                        'GL2D-static has no outputClasses at the end')
        self.assertEqual(protClassify.outputClasses.getSize(),
                         protImportAvgs.outputAverages.getSize(),
                         'GL2D-static returned a wrong number of classes at the end.')