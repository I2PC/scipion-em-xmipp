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
from pwem.protocols import (ProtImportParticles, ProtImportVolumes,ProtSubSet,
                            ProtUnionSet)

from xmipp3.protocols import (XmippProtComputeLikelihood,XmippProtCropResizeVolumes,
                              XmippProtPreprocessParticles, XmippProtPreprocessVolumes)

class TestXmippComputeLikelihood(BaseTest):
    """ Test protocol for compute log likelihood protocol. """
    _numberOfParticles = 20

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
        cls.protRiboIni = cls.newProtocol(ProtImportVolumes,
                                objLabel='import ribo volume', 
                                filesPath=cls.dsRelion.getFile('volumes/reference.mrc'),
                                samplingRate=7.08)
        cls.launchProtocol(cls.protRiboIni)


        prot = cls.newProtocol(XmippProtPreprocessVolumes,
                                objLabel='norm ribo volume',
                                doNormalize=True,
                                inputVolumes=cls.protRiboIni.outputVolume)
        cls.launchProtocol(prot)
        
        return prot

    @classmethod
    def importSpikeVolume(cls):
        cls.protSpike0 = cls.newProtocol(ProtImportVolumes,
                                objLabel='import spike volume',
                                importFrom=1,
                                emdbId='12229')
        cls.launchProtocol(cls.protSpike0)

        cls.protSpikeIni = cls.newProtocol(XmippProtCropResizeVolumes,
                                objLabel='resize spike volume',
                                doResize=True, resizeSamplingRate=7.08,
                                doWindow=True, windowSize=60,
                                inputVolumes=cls.protSpike0.outputVolume)
        cls.launchProtocol(cls.protSpikeIni)

        prot = cls.newProtocol(XmippProtPreprocessVolumes,
                                objLabel='norm spike volume',
                                doNormalize=True,
                                inputVolumes=cls.protSpikeIni.outputVol)
        cls.launchProtocol(prot)

        return prot

    @classmethod
    def joinVolumes(cls):
        prot = cls.newProtocol(ProtUnionSet, 
                                objLabel='join volumes norm',
                                inputType=3,
                                inputSets=[cls.protRiboVol.outputVol,
                                           cls.protSpikeVol.outputVol])
        cls.launchProtocol(prot)

        return prot

    @classmethod
    def joinVolumesIni(cls):
        prot = cls.newProtocol(ProtUnionSet,
                                objLabel='join volumes ini',
                                inputType=3,
                                inputSets=[cls.protRiboIni.outputVolume,
                                           cls.protSpikeIni.outputVol])
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

        # We are going to make a subset to speed-up following processes
        cls.protSubset = cls.newProtocol(ProtSubSet,
                                      objLabel='subset of particles',
                                      chooseAtRandom=True,
                                      nElements=numberOfParticles)
        cls.protSubset.inputFullSet.set(prot.outputParticles)
        cls.launchProtocol(cls.protSubset)

        protNorm = cls.newProtocol(XmippProtPreprocessParticles, 
                                objLabel='norm ribo particles',
                                doNormalize=True,
                                inputParticles=cls.protSubset.outputParticles)
        cls.launchProtocol(protNorm)

        return protNorm

    @classmethod
    def doImports(cls):
        cls.protRiboVol = cls.importRibosomeVolume()
        cls.protSpikeVol = cls.importSpikeVolume()
        cls.protJoinedVols = cls.joinVolumes()
        cls.protJoinedVolsIni = cls.joinVolumesIni()

        pathNoCTF = 'import/refine3d/extra/relion_it001_data.star'
        cls.protImportPars = cls.importParticles(cls._numberOfParticles,
                                                 pathNoCTF)

    def testDefault(self):
        self.runProtLikelihood(particleRadius=-1, noiseRadius=-1)

    def testOneVol(self):
        prot = self.newProtocol(XmippProtComputeLikelihood,
                                objLabel='log likelihood',
                                inputParticles=self.protImportPars.outputParticles,
                                inputRefs=self.protRiboVol.outputVol,
                                particleRadius=29, noiseRadius=30)
        self.launchProtocol(prot)
        self.checkOutput(prot, 'reprojections',
                         ['reprojections.getSize() == %d'
                          % self._numberOfParticles])
        self.checkOutput(prot, 'outputClasses',
                         ['outputClasses.getSize() == %d' % 1])
        self.checkOutput(prot, 'outputClasses',
                         ['outputClasses.getFirstItem().getSize() == %d'
                          % self._numberOfParticles])

    def testOldProg(self):
        self.runProtLikelihood(newProg=False)

    def testOldIgnoreCTF(self):
        self.runProtLikelihood(newProg=False, ignoreCTF=True)

    def testIgnoreCTF(self):
        self.runProtLikelihood(ignoreCTF=True)

    def testNoNorm(self):
        self.runProtLikelihood(badResults=True, normedInputs=False)

    def testDoNormNeeded(self):
       self.runProtLikelihood(doNorm=True, normedInputs=False)

    def testDoOldNormNeeded(self):
       self.runProtLikelihood(doNorm=True, normedInputs=False,
                              normType=0)

    def testDoNormUnneeded(self):
       self.runProtLikelihood(doNorm=True, normedInputs=True)


    def runProtLikelihood(self, particleRadius=29, noiseRadius=30,
                          doNorm=False, ignoreCTF=False, newProg=True,
                          normedInputs=True, badResults=False,
                          normType=1):
        """Run the protocol and check the outputs"""
        if normedInputs:
            inputParticles=self.protImportPars.outputParticles
            inputRefs=self.protJoinedVols.outputSet
        else:
            inputParticles=self.protSubset.outputParticles
            inputRefs=self.protJoinedVolsIni.outputSet

        prot = self.newProtocol(XmippProtComputeLikelihood,
                                objLabel='log likelihood',
                                inputParticles=inputParticles,
                                inputRefs=inputRefs,
                                particleRadius=particleRadius,
                                noiseRadius=noiseRadius,
                                doNorm=doNorm, normType=normType,
                                newProg=newProg,
                                ignoreCTF=ignoreCTF)
        self.launchProtocol(prot)
        self.checkOutput(prot, 'reprojections',
                         ['reprojections.getSize() == %d'
                          % (self._numberOfParticles * 2)])
        self.checkOutput(prot, 'outputClasses',
                         ['outputClasses.getSize() == %d' % 2])
        
        if badResults:
            self.checkOutput(prot, 'outputClasses',
                            ['outputClasses.getFirstItem().getSize() != %d'
                            % self._numberOfParticles]) # bad class assignment
        else:
            self.checkOutput(prot, 'outputClasses',
                            ['outputClasses.getFirstItem().getSize() == %d'
                            % self._numberOfParticles])
