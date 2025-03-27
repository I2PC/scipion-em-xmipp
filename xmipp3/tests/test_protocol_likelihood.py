# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     James Krieger (jmkrieger@cnb.csic.es)
# *
# * Centro Nacional de Biotecnologia, CSIC
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

import os
from os.path import exists

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pwem.protocols import (ProtImportParticles, ProtImportVolumes, ProtSubSet,
                            ProtUnionSet)

from xmipp3.protocols import (XmippProtComputeLikelihood, XmippProtCropResizeVolumes,
                              XmippProtPreprocessParticles, XmippProtPreprocessVolumes)

class TestXmippComputeLikelihood(BaseTest):
    """ Test protocol for compute log likelihood protocol. """
    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')
        cls.doImports()

    @classmethod
    def checkOutput(cls, prot, outputName, conditions=[]):
        """ Check that an ouput was generated and
        the condition is valid. 
        """
        o = getattr(prot, outputName, None)
        locals()[outputName] = o 
        cls.assertIsNotNone(o, "Output: %s is None" % outputName)
        for cond in conditions:
            cls.assertTrue(eval(cond), 'Condition failed: ' + cond)

    @classmethod
    def importRibosomeVolume(cls):
        prot = cls.newProtocol(ProtImportVolumes, 
                                objLabel='import ribo volume', 
                                filesPath=cls.dsRelion.getFile('volumes/reference.mrc'),
                                samplingRate=7.08)
        cls.launchProtocol(prot)


        prot = cls.newProtocol(XmippProtPreprocessVolumes, 
                                objLabel='norm ribo volume',
                                doNormalize=True,
                                inputVolumes=prot.outputVolume)
        cls.launchProtocol(prot)
        
        return prot

    @classmethod
    def importSpikeVolume(cls):
        prot = cls.newProtocol(ProtImportVolumes, 
                                objLabel='import spike volume',
                                importFrom=1,
                                emdbId='12229')
        cls.launchProtocol(prot)

        prot = cls.newProtocol(XmippProtCropResizeVolumes, 
                                objLabel='resize spike volume',
                                doResize=True, resizeSamplingRate=7.08,
                                doWindow=True, windowSize=60,
                                inputVolumes=prot.outputVolume)
        cls.launchProtocol(prot)

        prot = cls.newProtocol(XmippProtPreprocessVolumes, 
                                objLabel='norm spike volume',
                                doNormalize=True,
                                inputVolumes=prot.outputVol)
        cls.launchProtocol(prot)
        
        return prot

    @classmethod
    def joinVolumes(cls):
        prot = cls.newProtocol(ProtUnionSet, 
                                objLabel='join volumes',
                                inputType=3,
                                inputSets=[cls.protRiboVol.outputVol,
                                           cls.protSpikeVol.outputVol])
        cls.launchProtocol(prot)
        
        return prot

    @classmethod
    def importParticles(cls, numberOfParticles, path):
        """ Import an EMX file with Particles and defocus
        """
        prot = cls.newProtocol(ProtImportParticles,
                                 objLabel='from relion (auto-refine 3d)',
                                 importFrom=ProtImportParticles.IMPORT_FROM_RELION,
                                 starFile=cls.dsRelion.getFile(path),
                                 magnification=10000,
                                 samplingRate=7.08,
                                 haveDataBeenPhaseFlipped=True
                                 )
        cls.launchProtocol(prot)
        cls.checkOutput(prot, 'outputParticles', 
                         ['outputParticles.hasAlignmentProj()',
                          'outputParticles.isPhaseFlipped()'])
        # We are going to make a subset to speed-up following processes
        protSubset = cls.newProtocol(ProtSubSet, 
                                      objLabel='subset of particles',
                                      chooseAtRandom=True,
                                      nElements=numberOfParticles)
        protSubset.inputFullSet.set(prot.outputParticles)
        
        cls.launchProtocol(protSubset)
        cls.checkOutput(protSubset, 'outputParticles', 
                         ['outputParticles.getSize() == %d' 
                          % numberOfParticles])
        

        protNorm = cls.newProtocol(XmippProtPreprocessParticles, 
                                objLabel='norm ribo particles',
                                doNormalize=True,
                                inputParticles=protSubset.outputParticles)
        cls.launchProtocol(protNorm)

        return protNorm

    @classmethod
    def doImports(cls):
        cls.protRiboVol = cls.importRibosomeVolume()
        cls.protSpikeVol = cls.importSpikeVolume()
        cls.protJoinedVols = cls.joinVolumes()

        pathNoCTF = 'import/refine3d/extra/relion_it001_data.star'
        cls.protImportPars = cls.importParticles(20, pathNoCTF)

    def testDefault(self):
        prot = self.newProtocol(XmippProtComputeLikelihood, 
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protJoinedVols.outputSet)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts

    def testRad(self):
        prot = self.newProtocol(XmippProtComputeLikelihood, 
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protJoinedVols.outputSet,
                                particleRadius=29, noiseRadius=30)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts

    def testOneVol(self):
        prot = self.newProtocol(XmippProtComputeLikelihood, 
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protRiboVol.outputVolume)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts

    def testOldProg(self):
        prot = self.newProtocol(XmippProtComputeLikelihood,
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protJoinedVols.outputSet,
                                newProg=False)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts

    def testBinThreads(self):
        prot = self.newProtocol(XmippProtComputeLikelihood, 
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protJoinedVols.outputSet,
                                binThreads=3)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts

    def testMPI(self):
        prot = self.newProtocol(XmippProtComputeLikelihood, 
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protJoinedVols.outputSet,
                                mpi=2)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts

    def testOldIgnoreCTF(self):
        prot = self.newProtocol(XmippProtComputeLikelihood, 
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protJoinedVols.outputSet,
                                newProg=False, ignoreCTF=True)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts

    def testIgnoreCTF(self):
        prot = self.newProtocol(XmippProtComputeLikelihood, 
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protJoinedVols.outputSet,
                                ignoreCTF=True)
        self.launchProtocol(prot)
        # TODO: add output checks and asserts
