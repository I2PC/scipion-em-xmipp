# **************************************************************************
# *
# * Authors:    Laura del Cano (ldelcano@cnb.csic.es)
# *             Josue Gomez Blanco (josue.gomez-blanco@mcgill.ca)
# *             Jose Gutierrez (jose.gutierrez@cnb.csic.es)
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

from __future__ import print_function

import os.path

from pyworkflow.tests.test_utils import wait
from pyworkflow.utils import greenStr, magentaStr
from pyworkflow.plugin import Domain
from pyworkflow.tests import *
import pwem.protocols as emprot

import xmipp3
from xmipp3.base import *
from xmipp3.convert import *
from xmipp3.constants import *
from xmipp3.protocols import *
from xmipp3.protocols import XmippFilterHelper as xfh, XmippProtCL2D
from xmipp3.protocols import XmippResizeHelper as xrh
from xmipp3.protocols import OP_DOTPRODUCT, OP_MULTIPLY, OP_SQRT

MSG_WRONG_SIZE = "There was a problem with the size of the output "
MSG_WRONG_OUTPUT = "There was a problem with the output "
MSG_WRONG_IMPORT = "There was a problem with the import of the "
MSG_WRONG_PROTOCOL = "There was a problem with the protocol: "
MSG_WRONG_DIM = "There was a problem with the dimensions of the output "
MSG_WRONG_SAMPLING = "There was a problem with the sampling rate value of the output "

# Some utility functions to import particles that are used
# in several tests.
class TestXmippBase(BaseTest):
    @classmethod
    def setData(cls, dataProject='xmipp_tutorial'):
        cls.dataset = DataSet.getDataSet(dataProject)
        cls.particlesFn = cls.dataset.getFile('particles')
        cls.particlesDir = cls.dataset.getFile('particlesDir')
        cls.volumesFn = cls.dataset.getFile('volumes')
        cls.volumesDir = cls.dataset.getFile('volumesDir')
        cls.averagesFn = cls.dataset.getFile('averages')
        cls.averagesDir = cls.dataset.getFile('averagesDir')
    
    @classmethod
    def runImportParticles(cls, pattern, samplingRate, checkStack=False,
                           phaseFlip=False):
        """ Run an Import particles protocol. """
        cls.protImport = cls.newProtocol(emprot.ProtImportParticles,
                                         filesPath=pattern,
                                         samplingRate=samplingRate,
                                         checkStack=checkStack,
                                         haveDataBeenPhaseFlipped=phaseFlip)
        print('_label: ', cls.protImport._label)
        cls.launchProtocol(cls.protImport)
        # check that input images have been imported (a better way to do this?)
        if cls.protImport.outputParticles is None:
            raise Exception('Import of images: %s, failed. outputParticles is None.' % pattern)
        return cls.protImport
    
    @classmethod
    def runImportAverages(cls, pattern, samplingRate, checkStack=False):
        """ Run an Import particles protocol. """
        cls.protImportAvg = cls.newProtocol(emprot.ProtImportAverages,
                                            filesPath=pattern,
                                            samplingRate=samplingRate,
                                            checkStack=checkStack)
        print('_label: ', cls.protImportAvg._label)
        cls.launchProtocol(cls.protImportAvg)
        # check that input images have been imported (a better way to do this?)
        if cls.protImportAvg.outputAverages is None:
            raise Exception('Import of averages: %s, failed. outputAverages is None.' % pattern)
        return cls.protImportAvg
    
    @classmethod
    def runImportVolume(cls, pattern, samplingRate, checkStack=False):
        """ Run an Import particles protocol. """
        cls.protImport = cls.newProtocol(emprot.ProtImportVolumes,
                                         filesPath=pattern,
                                         samplingRate=samplingRate,
                                         checkStack=checkStack)
        print('_label: ', cls.protImport._label)
        cls.launchProtocol(cls.protImport)
        # check that input images have been imported (a better way to do this?)
        if cls.protImport.outputVolume is None:
            raise Exception('Import of volume: %s, failed. outputVolume is None.' % pattern)
        return cls.protImport

    @classmethod
    def runResizeParticles(cls, particles, doResize, resizeOption, resizeDim):
        cls.protResize = cls.newProtocol(XmippProtCropResizeParticles,
                                         doResize=doResize,
                                         resizeOption=resizeOption,
                                         resizeDim=resizeDim)
        cls.protResize.inputParticles.set(particles)
        cls.launchProtocol(cls.protResize)
        return cls.protResize
    
    @classmethod
    def runCL2DAlign(cls, particles):
        cls.CL2DAlign = cls.newProtocol(XmippProtCL2DAlign, 
                                        maximumShift=2, numberOfIterations=2,
                                        numberOfMpi=4, numberOfThreads=1, useReferenceImage=False)
        cls.CL2DAlign.inputParticles.set(particles)
        cls.launchProtocol(cls.CL2DAlign)
        return cls.CL2DAlign
    
    @classmethod
    def runClassify(cls, particles):
        cls.ProtClassify = cls.newProtocol(XmippProtML2D, 
                                           numberOfClasses=4, maxIters=3, doMlf=False,
                                           numberOfMpi=3, numberOfThreads=2)
        cls.ProtClassify.inputParticles.set(particles)
        cls.launchProtocol(cls.ProtClassify)
        return cls.ProtClassify

    @classmethod
    def runCreateMask(cls, samplingRate, size):
        cls.protMask = cls.newProtocol(XmippProtCreateMask2D,
                                     samplingRate = samplingRate,
                                     size= size,
                                     geo=0, radius=-1 )
        cls.protMask.setObjLabel('circular mask')
        cls.launchProtocol(cls.protMask)
        return cls.protMask


class TestXmippCreateMask2D(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport = cls.runImportParticles(cls.particlesFn, 1.237, True)
        cls.samplingRate = cls.protImport.outputParticles.getSamplingRate()
        cls.size = 20
        
    def testCreateCircularMask(self):
        print("Run create circular mask for particles")
        protMask1 = self.newProtocol(XmippProtCreateMask2D,
                                     samplingRate = self.samplingRate, 
                                     size= self.size, 
                                     geo=0, radius=-1 )
        protMask1.setObjLabel('circular mask')
        self.launchProtocol(protMask1)
        self.assertIsNotNone(protMask1.outputMask,
                             "There was a problem with create circular mask "
                             "for particles")
    
    def testCreateBoxMask(self):
        print("Run create box mask for particles")
        protMask2 = self.newProtocol(XmippProtCreateMask2D,
                                     samplingRate = self.samplingRate, 
                                     size= self.size, 
                                     geo=1, boxSize=-1 )
        protMask2.setObjLabel('box mask')
        print("launching protMask2")
        self.launchProtocol(protMask2)
        print("assert....")
        self.assertIsNotNone(protMask2.outputMask,
                             "There was a problem with create boxed mask "
                             "for particles")
    
    def testCreateCrownMask(self):
        print("Run create crown mask for particles")
        protMask3 = self.newProtocol(XmippProtCreateMask2D,
                                     samplingRate = self.samplingRate, 
                                     size= self.size, 
                                     geo=2, innerRadius=2, outerRadius=12 )
        protMask3.setObjLabel('crown mask')
        self.launchProtocol(protMask3)
        self.assertIsNotNone(protMask3.outputMask,
                             "There was a problem with create crown mask "
                             "for particles")
    
    def testCreateGaussianMask(self):
        print("Run create gaussian mask for particles")
        protMask4 = self.newProtocol(XmippProtCreateMask2D,
                                     samplingRate = self.samplingRate, 
                                     size= self.size, 
                                     geo=3, sigma=-1 )
        protMask4.setObjLabel('gaussian mask')
        self.launchProtocol(protMask4)
        self.assertIsNotNone(protMask4.outputMask,
                             "There was a problem with create gaussian mask "
                             "for particles")
    
    def testCreateRaisedCosineMask(self):
        print("Run create raised cosine mask for particles")
        protMask5 = self.newProtocol(XmippProtCreateMask2D,
                                     samplingRate = self.samplingRate, 
                                     size= self.size,
                                     geo=4, innerRadius=2, outerRadius=12)
        protMask5.setObjLabel('raised cosine mask')
        self.launchProtocol(protMask5)
        self.assertIsNotNone(protMask5.outputMask,
                             "There was a problem with create raised cosine "
                             "mask for particles")
    
    def testCreateRaisedCrownMask(self):
        print("Run create raised crown mask for particles")
        protMask6 = self.newProtocol(XmippProtCreateMask2D,
                                     samplingRate = self.samplingRate, 
                                     size= self.size, 
                                     geo=5, innerRadius=2, outerRadius=12,
                                     borderDecay=2)
        protMask6.setObjLabel('raised crown mask')
        self.launchProtocol(protMask6)
        self.assertIsNotNone(protMask6.outputMask,
                             "There was a problem with create raised crown "
                             "mask for particles")
    

class TestXmippApplyMask2D(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
        cls.protImport = cls.runImportParticles(cls.particlesFn, 1.237, True)

    def testApplyCircularMask(self):
        print("Run apply circular mask for particles")
        protMask1 = self.newProtocol(XmippProtMaskParticles,
                                     source=0, geo=0, radius=-1,
                                     fillType=0, fillValue=5 )
        protMask1.inputParticles.set(self.protImport.outputParticles)
        protMask1.setObjLabel('circular mask')
        self.launchProtocol(protMask1)
        self.assertAlmostEquals(protMask1.outputParticles.getSamplingRate(), 
                                self.protImport.outputParticles.getSamplingRate(),
                                msg="There was a problem with the sampling rate "
                                    "value for the apply user custom mask for "
                                    "particles")
        self.assertIsNotNone(protMask1.outputParticles,
                             "There was a problem with apply circular mask "
                             "for particles")
    
    def testApplyBoxMask(self):
        print("Run apply box mask for particles")
        protMask2 = self.newProtocol(XmippProtMaskParticles,
                                     source=0, geo=1, boxSize=-1,
                                     fillType=1 )
        protMask2.inputParticles.set(self.protImport.outputParticles)
        protMask2.setObjLabel('box mask')
        self.launchProtocol(protMask2)
        self.assertAlmostEquals(protMask2.outputParticles.getSamplingRate(), 
                                self.protImport.outputParticles.getSamplingRate(),
                                msg="There was a problem with the sampling rate "
                                    "value for the apply user custom mask for "
                                    "particles")
        self.assertIsNotNone(protMask2.outputParticles,
                             "There was a problem with apply boxed mask for particles")
    
    def testApplyCrownMask(self):
        print("Run apply crown mask for particles")
        protMask3 = self.newProtocol(XmippProtMaskParticles,
                                     source=0, geo=2, innerRadius=2,
                                     outerRadius=12,
                                     fillType=2)
        protMask3.inputParticles.set(self.protImport.outputParticles)
        protMask3.setObjLabel('crown mask')
        self.launchProtocol(protMask3)
        self.assertAlmostEquals(protMask3.outputParticles.getSamplingRate(), 
                                self.protImport.outputParticles.getSamplingRate(),
                                msg="There was a problem with the sampling rate "
                                    "value for the apply user custom mask for "
                                    "particles")
        self.assertIsNotNone(protMask3.outputParticles,
                             "There was a problem with apply crown mask "
                             "for particles")
        
    def testApplyGaussianMask(self):
        print("Run apply gaussian mask for particles")
        protMask4 = self.newProtocol(XmippProtMaskParticles,
                                     source=0, geo=3, sigma=-1,
                                     fillType=3 )
        protMask4.inputParticles.set(self.protImport.outputParticles)
        protMask4.setObjLabel('gaussian mask')
        self.launchProtocol(protMask4)
        self.assertAlmostEquals(protMask4.outputParticles.getSamplingRate(), 
                                self.protImport.outputParticles.getSamplingRate(),
                                msg="There was a problem with the sampling rate "
                                    "value for the apply user custom mask for "
                                    "particles")
        self.assertIsNotNone(protMask4.outputParticles,
                             "There was a problem with apply gaussian mask "
                             "for particles")
        
    def testApplyRaisedCosineMask(self):
        print("Run apply raised cosine mask for particles")
        protMask5 = self.newProtocol(XmippProtMaskParticles,
                                     source=0, geo=4, innerRadius=2, outerRadius=12,
                                     fillType=0, fillValue=5 )
        protMask5.inputParticles.set(self.protImport.outputParticles)
        protMask5.setObjLabel('raised cosine mask')
        self.launchProtocol(protMask5)
        self.assertAlmostEquals(protMask5.outputParticles.getSamplingRate(), 
                                self.protImport.outputParticles.getSamplingRate(),
                                msg="There was a problem with the sampling rate "
                                    "value for the apply user custom mask for "
                                    "particles")
        self.assertIsNotNone(protMask5.outputParticles,
                             "There was a problem with apply raised cosine "
                             "mask for particles")
        
    def testApplyRaisedCrownMask(self):
        print("Run apply raised crown mask for particles")
        protMask6 = self.newProtocol(XmippProtMaskParticles,
                                     source=0, geo=5, innerRadius=2, outerRadius=12, borderDecay=2,
                                     fillType=1 )
        protMask6.inputParticles.set(self.protImport.outputParticles)
        protMask6.setObjLabel('raised crown mask')
        self.launchProtocol(protMask6)
        self.assertAlmostEquals(protMask6.outputParticles.getSamplingRate(), 
                                self.protImport.outputParticles.getSamplingRate(),
                                msg="There was a problem with the sampling rate"
                                    " value for the apply user custom mask for "
                                    "particles")
        
        self.assertIsNotNone(protMask6.outputParticles,
                             "There was a problem with apply raised crown mask "
                             "for particles")

    def testApplyUserMask(self):
        print("Run apply user mask for particles")
        # Create MASK
        protMask01 = self.newProtocol(XmippProtCreateMask2D,
                                     samplingRate=1.237, 
                                     size=500, 
                                     geo=0, radius=225)
        protMask01.setObjLabel('circular mask')
        self.launchProtocol(protMask01)
        self.assertIsNotNone(protMask01.outputMask,
                             "There was a problem with apply user custom mask "
                             "for particles")
        # Apply MASK
        protMask02 = self.newProtocol(XmippProtMaskParticles,
                                     source=1,
                                     fillType=1 )
        protMask02.inputParticles.set(self.protImport.outputParticles)
        protMask02.inputMask.set(protMask01.outputMask)
        protMask02.setObjLabel('user custom mask')
        self.launchProtocol(protMask02)
        self.assertAlmostEquals(protMask02.outputParticles.getSamplingRate(), 
                                self.protImport.outputParticles.getSamplingRate(),
                                msg="There was a problem with the sampling rate "
                                    "value for the apply user custom mask for "
                                    "particles")
        
        self.assertIsNotNone(protMask02.outputParticles,
                             "There was a problem with apply user custom mask "
                             "for particles")
        


class TestXmippScreenParticles(TestXmippBase):
    """This class check if the protocol to classify particles by their
    similarity to discard outliers work properly"""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 1.237, True)
        cls.samplingRate = cls.protImport.outputParticles.getSamplingRate()
        cls.size = 20

    def _updateProtocol(self, prot):
        prot2 = getProtocolFromDb(prot.getProject().path,
                                  prot.getDbPath(),
                                  prot.getObjId())
        # Close DB connections
        prot2.getProject().closeMapper()
        prot2.closeMappers()
        return prot2

    def test_screenPart(self):
        try:
            from itertools import izip
        except ImportError:
            izip = zip

        print('Running Screen particles test')
        xpsp = XmippProtScreenParticles  # short notation
        # First test for check I/O. Input and Output SetOfParticles must
        # be equal sized if not rejection is selected
        print('--> Running Screen without rejection')
        protScreenNone = self.newProtocol(xpsp, autoParRejection=xpsp.REJ_NONE)
        protScreenNone.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(protScreenNone)
        self.assertIsNotNone(protScreenNone.outputParticles,
                             'Output has not been produced')
        print('\t --> Output is not None')
        self.assertEqual(len(protScreenNone.outputParticles),
                         len(self.protImport.outputParticles),
                         "Input and Output Set Of Particles don't have same size")
        print('\t --> Input/Output sets sizes are equal (%s)' % len(
            protScreenNone.outputParticles))
        
        for x, y in izip(self.protImport.outputParticles,
                         protScreenNone.outputParticles):
            # print("\t      compare %s with %s" % (x, y))
            self.assertEqual(x.getObjId(), y.getObjId(), "Particles differ")
            self.assertEqual(x.getSamplingRate(), y.getSamplingRate(),
                             "Particle sampling rate differ")
        print('\t --> Input/Output sets contain the same particles')

        # Test summary zScores
        self.assertIsNotEmpty(protScreenNone.minZScore)
        self.assertIsNotEmpty(protScreenNone.maxZScore)
        self.assertIsNotEmpty(protScreenNone.sumZScore)

        # test zScore
        self.assertAlmostEqual(0.823016, protScreenNone.minZScore.get())
        self.assertAlmostEqual(3.276184, protScreenNone.maxZScore.get())
        self.assertAlmostEqual(133.033408, protScreenNone.sumZScore.get())

        # After this, we check for errors in method with particle rejection
        # by ZScore
        print("--> Running Screen with rejection to maxZScore upper than 2.5")
        protScreenZScore = self.newProtocol(xpsp,
                                            autoParRejection=xpsp.REJ_MAXZSCORE,
                                            maxZscore=2.5)
        protScreenZScore.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(protScreenZScore)
        self.assertIsNotNone(protScreenZScore.outputParticles,
                             "Output has not been produced")
        print('\t --> Output is not None')
        self.assertEqual(len(protScreenZScore.outputParticles), 69,
                         "Output Set Of Particles must be 69, but %s found" %
                         len(protScreenZScore.outputParticles))
        print('\t --> Output set size is correct (%s)' % len(
            protScreenZScore.outputParticles))
        
        # test values not equal to previous run
        self.assertAlmostEqual(0.823016, protScreenZScore.minZScore.get())
        self.assertAlmostEqual(2.460981, protScreenZScore.maxZScore.get())
        self.assertAlmostEqual(113.302433, protScreenZScore.sumZScore.get())

        for x in protScreenZScore.outputParticles:
            self.assertLess(x._xmipp_zScore.get(), 2.5,
                            "Particle with id (%s) has a ZScore of %s, "
                            "upper than supposed threshold %s"
                            % (x.getObjId(), x._xmipp_zScore.get(), 2.5))
        print('\t --> Output particles are below the ZScore threshold')
        
        # We check for errors in method with particle rejection by percentage
        print(
        "--> Running Screen with rejection of the 5% particles with the lowest ZScore")
        protScreenPercentage = self.newProtocol(xpsp,
                                                autoParRejection=xpsp.REJ_PERCENTAGE)
        protScreenPercentage.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(protScreenPercentage)
        self.assertIsNotNone(protScreenPercentage.outputParticles,
                             "Output has not been produced")
        print('\t --> Output is not None')
        self.assertEqual(len(protScreenPercentage.outputParticles), 69,
                         "Output Set Of Particles must be 69, but %s found"
                         % len(protScreenPercentage.outputParticles))
        print('\t --> Output set size is correct (%s)'
              % len(protScreenPercentage.outputParticles))
        
        for x, y in izip(protScreenZScore.outputParticles, protScreenPercentage.outputParticles):
            # print("\t      compare %s with %s" % (x, y))
            self.assertEqual(x.getObjId(), y.getObjId(), "Particles differ")
        print('\t --> Particles rejected using maxZScore(2.5) '
              'method and percentage(5%) one are the same')

        print("Start Streaming Particles")
        protStream = self.newProtocol(emprot.ProtCreateStreamData, setof=3,
                                      creationInterval=5, nDim=76,
                                      groups=10)
        protStream.inputParticles.set(self.protImport.outputParticles)
        self.proj.launchProtocol(protStream, wait=False)


        print("Run Screen Particles")
        protScreen = self.newProtocol(xpsp)
        protScreen.inputParticles.set(protStream)
        protScreen.inputParticles.setExtended("outputParticles")
        self.proj.scheduleProtocol(protScreen)

        wait(lambda: not self._updateProtocol(protScreen).isFinished(), timeout=300)
        protScreen = self._updateProtocol(protScreen)
        self.assertEqual(protScreen.outputParticles.getSize(), 76)


class TestXmippPreprocessParticles(TestXmippBase):
    """This class check if the protocol to preprocess particles in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)
        
    def test_preprocessPart(self):
        print("Run Preprocess particles")
        protPreproc = self.newProtocol(XmippProtPreprocessParticles, 
                                      doRemoveDust=True, doNormalize=True, 
                                      backRadius=48, doInvert=True,
                                      doThreshold=True, thresholdType=1)
        
        protPreproc.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(protPreproc)
        
        if self.protImport.outputParticles.hasAlignment():
            from itertools import izip
            for x, y in izip(self.protImport.outputParticles.get(),
                             protPreproc.outputParticles.get()):
                print("compare ", x, " with ", y)
                self.assertEquals(x.getAlignment(), y.getAlignment(),
                                  "Alignment wrong")
                
        self.assertAlmostEquals(protPreproc.outputParticles.getSamplingRate(),
                                self.protImport.outputParticles.getSamplingRate(),
                                'There was a problem with the sampling rate '
                                ' in the preprocess particles')

        self.assertIsNotNone(protPreproc.outputParticles,
                             "There was a problem with preprocess particles")

from pyworkflow.protocol import getProtocolFromDb
class TestXmippTriggerParticles(TestXmippBase):
    """This class check if the protocol to trigger particles in Xmipp works properly."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)

    def _updateProtocol(self, prot):
        prot2 = getProtocolFromDb(prot.getProject().path,
                                  prot.getDbPath(),
                                  prot.getObjId())
        # Close DB connections
        prot2.getProject().closeMapper()
        prot2.closeMappers()
        return prot2

    def test_triggerPart(self):
        print("Start Streaming Particles")
        protStream = self.newProtocol(emprot.ProtCreateStreamData, setof=3,
                                      creationInterval=12, nDim=76, groups=10)
        protStream.inputParticles.set(self.protImport.outputParticles)
        self.proj.launchProtocol(protStream, wait=False)


        print("Run Trigger Particles")
        protTrigger = self.newProtocol(XmippProtTriggerData,
                                       objLabel="Trigger - static output",
                                       allImages=False, outputSize=50,
                                       checkInterval=10)
        protTrigger.inputImages.set(protStream)
        protTrigger.inputImages.setExtended("outputParticles")
        self.proj.scheduleProtocol(protTrigger)

        protTrigger1b = self.newProtocol(XmippProtTriggerData,
                                         objLabel="Trigger - static output (at the end)",
                                         allImages=False, outputSize=500,
                                         checkInterval=10)
        protTrigger1b.inputImages.set(protStream)
        protTrigger1b.inputImages.setExtended("outputParticles")
        self.proj.scheduleProtocol(protTrigger1b)

        protTrigger2 = self.newProtocol(XmippProtTriggerData,
                                        objLabel="Trigger - streaming",
                                        allImages=True, outputSize=50,
                                        checkInterval=10)
        protTrigger2.inputImages.set(protStream)
        protTrigger2.inputImages.setExtended("outputParticles")
        self.proj.scheduleProtocol(protTrigger2)

        protTrigger2b = self.newProtocol(XmippProtTriggerData,
                                         objLabel="Trigger - streaming (at the end)",
                                         allImages=True, outputSize=500,
                                         checkInterval=10)
        protTrigger2b.inputImages.set(protStream)
        protTrigger2b.inputImages.setExtended("outputParticles")
        self.proj.scheduleProtocol(protTrigger2b)

        protTrigger3 = self.newProtocol(XmippProtTriggerData,
                                        objLabel="Trigger - in batches",
                                        allImages=True, splitImages=True,
                                        outputSize=50, checkInterval=10)
        protTrigger3.inputImages.set(protStream)
        protTrigger3.inputImages.setExtended("outputParticles")
        self.proj.scheduleProtocol(protTrigger3)

        protTrigger3b = self.newProtocol(XmippProtTriggerData,
                                         objLabel="Trigger - in batches (at the end)",
                                        allImages=True, splitImages=True,
                                        outputSize=500, checkInterval=10)
        protTrigger3b.inputImages.set(protStream)
        protTrigger3b.inputImages.setExtended("outputParticles")
        self.proj.scheduleProtocol(protTrigger3b)

        # when first trigger (non-streaming mode) finishes must have 50 parts.
        self.checkResults(protTrigger, 50, "Static trigger fails")
        # when second trigger (full-streaming mode) finishes must have all parts.
        self.checkResults(protTrigger2, 76, "Full streaming trigger fails")

        # When all have finished, the third (spliting mode) must have two outputs
        time.sleep(10)  # sometimes is not ready yet
        protTrigger3 = self._updateProtocol(protTrigger3)
        self.assertSetSize(protTrigger3.outputParticles1, 50, "First batch fails")
        self.assertSetSize(protTrigger3.outputParticles2, 26, "Final batch fails")

        # When input closes, trigger must release what is in input.
        self.checkFinalTrigger(protTrigger1b, msg="Final trigger in static mode fails")
        self.checkFinalTrigger(protTrigger2b, msg="Final trigger in streaming mode fails")
        self.checkFinalTrigger(protTrigger3b, "outputParticles1", "Final trigger in batches mode fails")

    def checkFinalTrigger(self, prot, outputName="outputParticles", msg=''):
        prot = self._updateProtocol(prot)
        output = getattr(prot, outputName, None)
        self.assertIsNotNone(output, msg)
        self.assertSetSize(output, 76, msg)


    def checkResults(self, prot, size, msg=''):
        t0 = time.time()
        while not prot.isFinished():
            # Time out 4 minutes, just in case
            tdelta = time.time() - t0
            if tdelta > 4 * 60:
                break
            prot = self._updateProtocol(prot)
            time.sleep(2)

        self.assertSetSize(prot.outputParticles, size, msg)

class TestXmippCropResizeParticles(TestXmippBase):
    """Check protocol crop/resize particles from Xmipp."""
    @classmethod
    def setUpClass(cls):
        print(
        "\n", greenStr(" Crop/Resize Set Up - Collect data ".center(75, '-')))
        setupTestProject(cls)
        TestXmippBase.setData('xmipp_tutorial')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 1.237, True)
        cls.acquisition = cls.protImport.outputParticles.getAcquisition()

    def launch(self, **kwargs):
        "Launch XmippProtCropResizeParticles and return output particles."
        print(magentaStr("\n==> Crop/Resize input params: %s" % kwargs))
        prot = self.newProtocol(XmippProtCropResizeParticles, **kwargs)
        prot.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(prot)
        self.assertTrue(
            hasattr(prot, "outputParticles") and prot.outputParticles is not None,
            "There was a problem applying resize/crop to the particles")
        self.assertAlmostEqual(prot.outputParticles.getAcquisition().getVoltage(),
                               self.acquisition.getVoltage())
        return prot.outputParticles  # for more tests

    def test_newSizeAndCrop(self):
        inP = self.protImport.outputParticles  # short notation
        newSize = 128
        outP = self.launch(doResize=True, resizeOption=xrh.RESIZE_DIMENSIONS,
                           resizeDim=newSize,
                           doWindow=True, windowOperation=xrh.WINDOW_OP_CROP)

        self.assertEqual(newSize, outP.getDim()[0],
                         "Output particles dimension should be equal to %d" % newSize)
        self.assertAlmostEqual(outP.getSamplingRate(),
                               inP.getSamplingRate() * (inP.getDim()[0] / float(newSize)))

        # All other attributes remain the same. For the set:
        self.assertTrue(outP.equalAttributes(
            inP, ignore=['_mapperPath', '_samplingRate', '_firstDim'], verbose=True))
        # And for its individual particles too:
        self.assertTrue(outP.equalItemAttributes(
            inP, ignore=['_filename', '_index', '_samplingRate'], verbose=True))

    def test_factorAndWindow(self):
        inP = self.protImport.outputParticles  # short notation
        outP = self.launch(doResize=True, resizeOption=xrh.RESIZE_FACTOR,
                           resizeFactor=0.5,
                           doWindow=True, windowOperation=xrh.WINDOW_OP_WINDOW,
                           windowSize=500)

        # Since the images were resized by a factor 0.5 (downsampled), the new
        # pixel size (painfully called "sampling rate") should be 2x.
        self.assertAlmostEqual(outP.getSamplingRate(), inP.getSamplingRate() * 2)
        # After the window operation, the dimensions should be the same.
        self.assertEqual(inP.getDim(), outP.getDim())

        # All other attributes remain the same. For the set:
        self.assertTrue(outP.equalAttributes(
            inP, ignore=['_mapperPath', '_samplingRate'], verbose=True))
        # And for its individual particles too:
        self.assertTrue(outP.equalItemAttributes(
            inP, ignore=['_filename', '_index', '_samplingRate'], verbose=True))
    
    def test_pyramid(self):
        inP = self.protImport.outputParticles  # short notation
        outP = self.launch(doResize=True, resizeOption=xrh.RESIZE_PYRAMID,
                           resizeLevel=1)

        # Since the images were expanded by 2**resizeLevel (=2) the new
        # pixel size (painfully called "sampling rate") should be 0.5x.
        self.assertAlmostEqual(outP.getSamplingRate(), inP.getSamplingRate() * 0.5)
        # We did no window operation, so the dimensions will have doubled.
        self.assertAlmostEqual(outP.getDim()[0], inP.getDim()[0] * 2)

        # All other attributes remain the same. For the set:
        self.assertTrue(outP.equalAttributes(
            inP, ignore=['_mapperPath', '_samplingRate', '_firstDim'], verbose=True))
        # And for its individual particles too:
        self.assertTrue(outP.equalItemAttributes(
            inP, ignore=['_filename', '_index', '_samplingRate'], verbose=True))
        

class TestXmippCropResizeWAngles(TestXmippBase):
    """Check protocol crop/resize particles from Xmipp."""
    @classmethod
    def setUpClass(cls):
        print(
        "\n", greenStr(" Crop/Resize Set Up - Collect data ".center(75, '-')))
        setupTestProject(cls)
        TestXmippBase.setData('relion_tutorial')
    
    def launch(self, **kwargs):
        "Launch XmippProtCropResizeParticles and return output particles."
        print(magentaStr("\n==> Crop/Resize input params: %s" % kwargs))
        prot = self.newProtocol(XmippProtCropResizeParticles, **kwargs)
#         prot.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(prot)
        self.assertTrue(
            hasattr(prot, "outputParticles") and prot.outputParticles is not None,
            "There was a problem applying resize/crop to the particles")
        return prot.outputParticles  # for more tests

    def test_CropResizeWAngles(self):
        print("Import Set of particles with angles")
        prot1 = self.newProtocol(emprot.ProtImportParticles,
                                 objLabel='from scipion (to-reconstruct)',
                                 importFrom=emprot.ProtImportParticles.IMPORT_FROM_SCIPION,
                                 sqliteFile=self.dataset.getFile('import/case2/particles.sqlite'),
                                 magnification=10000,
                                 samplingRate=7.08
                                 )
        self.launchProtocol(prot1)
        
        inP = prot1.outputParticles  # short notation
        newSize = 30
        factor = (inP.getDim()[0] / float(newSize))
        outP = self.launch(doResize=True, resizeOption=xrh.RESIZE_DIMENSIONS,
                           resizeDim=newSize,inputParticles=inP,
                           doWindow=True, windowOperation=xrh.WINDOW_OP_CROP)

        self.assertEqual(newSize, outP.getDim()[0],
                         "Output particles dimension should be equal to %d"
                         % newSize)
        self.assertAlmostEqual(outP.getSamplingRate(),
                               inP.getSamplingRate() * factor)

        # All other attributes remain the same. For the set:
        ignoreList = ['_mapperPath', '_samplingRate', '_firstDim']
        self.assertTrue(outP.equalAttributes(inP, ignore=ignoreList,
                                             verbose=True))

        # Check the scale factor is correctly applied to coordinates and
        # transform matrix
        for inPart, outPart in izip(inP, outP):
            coordIn = inPart.getCoordinate().getX()
            coordOut = outPart.getCoordinate().getX()
            self.assertAlmostEqual(coordIn, coordOut*factor, delta=2)

            tIn = inPart.getTransform()
            tOut = outPart.getTransform()
            tOut.scaleShifts(factor)
            mIn = tIn.getMatrix()
            mOut = tOut.getMatrix()
            self.assertTrue(np.allclose(mIn, mOut),
                            msg='Matrices not equal: %s, %s' % (mIn, mOut))


class TestXmippFilterParticles(TestXmippBase):
    """Check the proper behavior of Xmipp's filter particles protocol."""

    @classmethod
    def setUpClass(cls):
        print("\n", greenStr(" Set Up - Collect data ".center(75, '-')))
        setupTestProject(cls)
        TestXmippBase.setData('xmipp_tutorial')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 1.237,
                                                True, True)

    def test_filterParticles(self):
        print("\n", greenStr(" Filter Particles ".center(75, '-')))

        def test(parts=self.protImport.outputParticles, **kwargs):
            "Launch XmippProtFilterParticles on parts and check results."
            print(magentaStr("\n==> Input params: %s" % kwargs))
            prot = self.newProtocol(XmippProtFilterParticles, **kwargs)
            prot.inputParticles.set(parts)
            self.launchProtocol(prot)
            self.assertIsNotNone(prot.outputParticles,
                                 "There was a problem with filter particles")
            self.assertTrue(prot.outputParticles.equalAttributes(
                parts, ignore=['_mapperPath'], verbose=True))
            # Compare the individual particles too.
            self.assertTrue(prot.outputParticles.equalItemAttributes(
                parts, ignore=['_filename', '_index'], verbose=True))

        # Check a few different cases.
        test(filterSpace=FILTER_SPACE_FOURIER, lowFreq=0.1, highFreq=0.25)
        test(filterSpace=FILTER_SPACE_REAL, filterModeReal=xfh.FM_MEDIAN)
        # For wavelets, we need the input's size to be a power of 2
        print(magentaStr("\n==> Resizing particles to 256 pixels"))
        protResize = self.newProtocol(XmippProtCropResizeParticles,
                                      doResize=True,
                                      resizeOption=xrh.RESIZE_DIMENSIONS,
                                      resizeDim=256)
        protResize.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(protResize)
        test(parts=protResize.outputParticles, filterSpace=FILTER_SPACE_WAVELET,
             filterModeWavelets=xfh.FM_DAUB12, waveletMode=xfh.FM_REMOVE_SCALE)


class TestXmippOperateParticles(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        print("\n", greenStr(" Set Up - Collect data ".center(75, '-')))
        setupTestProject(cls)
        TestXmippBase.setData('xmipp_tutorial')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 1.237,
                                                True, True)

    def launchSet(self, **kwargs):
        "Launch XmippProtImageOperateParticles and return output volumes."
        print(
        magentaStr("\n==> Operate set of volumes input params: %s" % kwargs))
        prot = XmippProtImageOperateParticles()
        prot.operation.set(kwargs.get('operation', 1))
        prot.inputParticles.set(self.protImport.outputParticles)
        prot.setObjLabel(kwargs.get('objLabel', None))
        prot.isValue.set(kwargs.get('isValue', False))
        prot.inputParticles2.set(kwargs.get('particles2', None))
        prot.value.set(kwargs.get('value', None))
        prot.intValue.set(kwargs.get('intValue', None))
        
        self.proj.launchProtocol(prot, wait=True)
        self.assertTrue(hasattr(prot, "outputParticles") and
                        prot.outputParticles is not None,
                        "There was a problem producing the output")
        return prot.outputParticles
        
    def testMultiplyVolSets(self):
        part2 = self.protImport.outputParticles  # short notation
        prot1 = self.launchSet(operation=OP_MULTIPLY,
                               objLabel='Multiply two SetOfParticles',
                               particles2=part2)

    def testMultiplyValue(self):
        prot2 = self.launchSet(operation=OP_MULTIPLY,
                               isValue=True,
                               objLabel='Multiply by a Value',
                               value=2.5)
    
    def testDotProduct(self):
        part2 = self.protImport.outputParticles  # short notation
        prot3 = self.launchSet(operation=OP_DOTPRODUCT,
                               objLabel='Dot Product',
                               particles2=part2)

    def testSqrt(self):
        prot4 = self.launchSet(operation=OP_SQRT,
                               objLabel='Sqrt')


class TestXmippML2D(TestXmippBase):
    """ This class check if the protocol to classify with ML2D in Xmipp works
    properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)
    
    def test_ml2d(self):
        print("Run ML2D")
        protML2D = self.newProtocol(XmippProtML2D, 
                                   numberOfClasses=2, maxIters=3,
                                   numberOfMpi=2, numberOfThreads=2)
        protML2D.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(protML2D)        
        
        self.assertIsNotNone(protML2D.outputClasses, "There was a problem with ML2D")  


class TestXmippCL2D(TestXmippBase):
    """This class check if the protocol to classify with CL2D in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)
        cls.protImportAvgs = cls.runImportAverages(os.path.dirname(cls.particlesFn) +
                                                   '/img00007[1-4].spi', 3.5)
    
    def test_cl2d(self):
        print("Run CL2D")
        # Run CL2D with random class and core analysis
        protCL2DRandomCore = self.newProtocol(XmippProtCL2D,
                                   numberOfClasses=8, numberOfInitialClasses=1,
                                   numberOfIterations=4, numberOfMpi=2)
        protCL2DRandomCore.inputParticles.set(self.protImport.outputParticles)
        protCL2DRandomCore.setObjLabel("CL2D with random class and core analysis")
        self.launchProtocol(protCL2DRandomCore)
        self.assertIsNotNone(protCL2DRandomCore.outputClasses, "There was a problem with CL2D with random class and core analysis")

        # Run CL2D with random class and no core analysis
        protCL2DRandomNoCore = self.newProtocol(XmippProtCL2D,
                                   numberOfClasses=2, numberOfInitialClasses=1,
                                   doCore=False, numberOfIterations=4, numberOfMpi=2)
        protCL2DRandomNoCore.inputParticles.set(self.protImport.outputParticles)
        protCL2DRandomNoCore.setObjLabel("CL2D with random class and no core analysis")
        self.launchProtocol(protCL2DRandomNoCore)
        self.assertIsNotNone(protCL2DRandomNoCore.outputClasses, "There was a problem with CL2D with random class and no core analysis")

        # Run CL2D with initial classes and core analysis
        protCL2DInitialCore = self.newProtocol(XmippProtCL2D,
                                   numberOfClasses=4, randomInitialization=False,
                                   numberOfIterations=4, numberOfMpi=2)
        protCL2DInitialCore.inputParticles.set(self.protImport.outputParticles)
        protCL2DInitialCore.initialClasses.set(self.protImportAvgs.outputAverages)
        protCL2DInitialCore.setObjLabel("CL2D with initial class and core analysis")
        self.launchProtocol(protCL2DInitialCore)
        self.assertIsNotNone(protCL2DInitialCore.outputClasses, "There was a problem with CL2D with initial class and core analysis")


class TestXmippCL2DCoreAnalysis(TestXmippBase):
    """This class check if the protocol to core Analysis with CL2D in Xmipp works properly."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)

    def test_cl2dCoreAnalysis(self):
        print("Run CL2D core analysis")
        # Run CL2D with random class and no core analysis
        protCL2DRandomNoCore = self.newProtocol(XmippProtCL2D,
                                                numberOfClasses=3, numberOfInitialClasses=1,
                                                doCore=False, numberOfIterations=4, numberOfMpi=2)
        protCL2DRandomNoCore.inputParticles.set(self.protImport.outputParticles)
        protCL2DRandomNoCore.setObjLabel("CL2D with random class and no core analysis")
        self.launchProtocol(protCL2DRandomNoCore)
        self.assertIsNotNone(protCL2DRandomNoCore.outputClasses,
                             "There was a problem with CL2D with random class and no core analysis")

        # Run Core Analysis
        protCL2DCoreAnalysis = self.newProtocol(XmippProtCoreAnalysis,
                                                numberOfClasses=3, thZscore=1.5,
                                                thPCAZscore=1.5, numberOfMpi=2)
        protCL2DCoreAnalysis.inputClasses.set(protCL2DRandomNoCore.outputClasses)
        protCL2DCoreAnalysis.setObjLabel("Core analysis")
        self.launchProtocol(protCL2DCoreAnalysis)
        self.assertIsNotNone(protCL2DCoreAnalysis.outputClasses_core,
                             "There was a problem with CL2D Core Analysis")



class TestXmippProtCL2DAlign(TestXmippBase):
    """This class check if the protocol to align particles in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)
    
    def test_xmippProtCL2DAlign(self):
        print("Run Only Align")
        # Run test without image reference
        CL2DAlignNoRef = self.newProtocol(XmippProtCL2DAlign,
                                    maximumShift=5, numberOfIterations=5,
                                    numberOfMpi=4, numberOfThreads=1, useReferenceImage=False)
        CL2DAlignNoRef.setObjLabel("CL2D Align without reference")
        CL2DAlignNoRef.inputParticles.set(self.protImport.outputParticles)
        self.launchProtocol(CL2DAlignNoRef)
        # Check that output is generated
        self.assertIsNotNone(CL2DAlignNoRef.outputParticles, "There was a problem generating output particles")
        # Check that it has alignment matrix
        self.assertTrue(CL2DAlignNoRef.outputParticles.hasAlignment2D(), "Output particles do not have alignment 2D")

        CL2DAlignRef = self.newProtocol(XmippProtCL2DAlign,
                                    maximumShift=5, numberOfIterations=5,
                                    numberOfMpi=4, numberOfThreads=1, useReferenceImage=True)
        CL2DAlignRef.setObjLabel("CL2D Align with reference")
        CL2DAlignRef.inputParticles.set(self.protImport.outputParticles)
        CL2DAlignRef.referenceImage.set(CL2DAlignNoRef.outputAverage)
        self.launchProtocol(CL2DAlignRef)
        # Check that output is generated
        self.assertIsNotNone(CL2DAlignRef.outputParticles, "There was a problem generating output particles")
        # Check that it has alignment matrix
        self.assertTrue(CL2DAlignRef.outputParticles.hasAlignment2D(), "Output particles do not have alignment 2D")


class TestXmippDenoiseParticles(TestXmippBase):
    """Check protocol Denoise Particles"""
    @classmethod
    def setUpClass(cls):
        # To denoise particles we need to import the particles and the
        # classes, and particles must be aligned with classes. As this
        # is the usual situation after a CL2D, we just run that protocol.

        # Set project and data
        setupTestProject(cls)
        cls.setData('mda')

        # Import particles
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.50)

        # Normalize them
        cls.protNormalize = cls.newProtocol(XmippProtPreprocessParticles,
                                            doNormalize=True, backRadius=42)
        cls.protNormalize.inputParticles.set(cls.protImport.outputParticles)
        cls.launchProtocol(cls.protNormalize)
        cls.protXmipp2DClass = cls.newProtocol(XmippProtCL2D,
                                               numberOfClasses=4, numberOfInitialClasses=1,
                                               doCore=False, numberOfIterations=3, numberOfMpi=1,
                                               numberOfThreads=1)
        cls.protXmipp2DClass.inputParticles.set(cls.protNormalize.outputParticles)

        cls.launchProtocol(cls.protXmipp2DClass)

    def test_denoiseparticles(self):
        print("""
*****************************
| Note: This part of the test may last for several minutes,
|       building a PCA basis for denoising is time expensive.
*****************************
""")
        protDenoise = self.newProtocol(XmippProtDenoiseParticles)
        protDenoise.inputParticles.set(self.protImport.outputParticles)
        protDenoise.inputClasses.set(self.protXmipp2DClass.outputClasses)
        self.launchProtocol(protDenoise)
        # We check that protocol generates output
        self.assertIsNotNone(protDenoise.outputParticles,
                             "There was a problem generating output particles")


class TestXmippApplyAlignment(TestXmippBase):
    """This class checks if the protocol Apply Alignment works properly"""
    @classmethod
    def setUpClass(cls):
        # For apply alignment we need to import particles that have alignment 2D information
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)
        cls.align2D = cls.runCL2DAlign(cls.protImport.outputParticles)

    def test_apply_alignment(self):
        protApply = self.newProtocol(XmippProtApplyAlignment)
        protApply.inputParticles.set(self.align2D.outputParticles)
        self.launchProtocol(protApply)
        # We check that protocol generates output
        self.assertIsNotNone(protApply.outputParticles, "There was a problem generating output particles")
        # Check that output particles do not have alignment information
        self.assertFalse(protApply.outputParticles.hasAlignment(), "Output particles should not have alignment information")


#TODO: Check with JM if this test should go in here since it is not a Xmipp protocol.
class TestAlignmentAssign(TestXmippBase):
    """This class checks if the protocol Alignment Assign works properly"""
    @classmethod
    def setUpClass(cls):
        # For alignment assign we need a set of particles without alignment 2D information and other set who has alignment information
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)
        cls.protResize = cls.runResizeParticles(cls.protImport.outputParticles, True, xrh.RESIZE_DIMENSIONS, 50)
        cls.align2D = cls.runCL2DAlign(cls.protImport.outputParticles)

    def test_alignment_assign_samesize(self):
        protAssign = self.newProtocol(emprot.ProtAlignmentAssign)
        protAssign.setObjLabel("Assign alignment of same size")
        protAssign.inputParticles.set(self.protImport.outputParticles)
        protAssign.inputAlignment.set(self.align2D.outputParticles)
        self.launchProtocol(protAssign)
        # We check that protocol generates output
        self.assertIsNotNone(protAssign.outputParticles, "There was a problem generating output particles")
        # Check that output particles do not have alignment information
        self.assertTrue(protAssign.outputParticles.hasAlignment(), "Output particles should have alignment information")
        # Check the scaling between the input and the output translation in the roto-translation transformation matrix
        self._checkTranslationScaling(protAssign, self.protImport)

    def test_alignment_assign_othersize(self):
        protAssign = self.newProtocol(emprot.ProtAlignmentAssign)
        protAssign.setObjLabel("Assign alignment of different size")
        protAssign.inputParticles.set(self.protResize.outputParticles)
        protAssign.inputAlignment.set(self.align2D.outputParticles)
        self.launchProtocol(protAssign)
        # We check that protocol generates output
        self.assertIsNotNone(protAssign.outputParticles, "There was a problem generating output particles")
        # Check the scaling between the input and the output translation in the roto-translation transformation matrix
        self._checkTranslationScaling(protAssign, self.protResize)

    def _checkTranslationScaling(self, protAssign, protParticles, shiftsAppliedBefore=False):
        inputAlignFirstPartTransMat = self.align2D.outputParticles.getFirstItem().getTransform().getMatrix()
        outputAlignFirstPartTransMat = protAssign.outputParticles.getFirstItem().getTransform().getMatrix()
        scale = self.align2D.outputParticles.getSamplingRate() / protParticles.outputParticles.getSamplingRate()
        outTranslation = [outputAlignFirstPartTransMat[0, 3],  # X trans
                          outputAlignFirstPartTransMat[1, 3],  # Y trans
                          outputAlignFirstPartTransMat[2, 3]]  # Z trans

        inTranslation = [inputAlignFirstPartTransMat[0, 3],  # X translation
                         inputAlignFirstPartTransMat[1, 3],  # Y translation
                         inputAlignFirstPartTransMat[2, 3]]  # Z translation

        self.assertFalse(all(v == 0 for v in inTranslation))
        self.assertFalse(all(v == 0 for v in outTranslation))
        [self.assertAlmostEqual(inT * scale, outT) for inT, outT in zip(inTranslation, outTranslation)]


class TestXmippKerdensom(TestXmippBase):
    """This class check if the protocol to calculate the kerdensom from particles in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImport = cls.runImportParticles(cls.particlesFn, 3.5)
        cls.align2D = cls.runCL2DAlign(cls.protImport.outputParticles)

    def test_kerdensom(self):
        print("Run Kerdensom")
        xmippProtKerdensom = self.newProtocol(XmippProtKerdensom, SomXdim=2, SomYdim=2)
        xmippProtKerdensom.inputParticles.set(self.align2D.outputParticles)
        self.launchProtocol(xmippProtKerdensom)
        self.assertIsNotNone(xmippProtKerdensom.outputClasses, "There was a problem with Kerdensom")

    def test_kerdensomMask(self):
        print("Run Kerdensom with a mask")
        protMask = self.runCreateMask(3.5, 100)
        xmippProtKerdensom = self.newProtocol(XmippProtKerdensom, SomXdim=2, SomYdim=2)
        xmippProtKerdensom.inputParticles.set(self.align2D.outputParticles)
        xmippProtKerdensom.useMask.set(True)
        xmippProtKerdensom.Mask.set(protMask.outputMask)
        self.launchProtocol(xmippProtKerdensom)
        self.assertIsNotNone(xmippProtKerdensom.outputClasses, "There was a problem with Kerdensom")


class TestXmippCompareReprojections(TestXmippBase):
    """This class check if the protocol compare reprojections in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')
        cls.protImportPart = cls.runImportParticles(cls.particlesFn, 3.5)
        cls.protImportAvgs = cls.runImportAverages(cls.particlesFn, 3.5)
        cls.protImportVol = cls.runImportVolume(cls.volumesFn, 3.5)
        cls.protImport2D = cls.importFromRelion2D()
        cls.protImport3D = cls.importFromRelionRefine3D()
        cls.protCreateMask = cls.newProtocol(XmippProtCreateMask3D, threshold=0.01)
        cls.protCreateMask.inputVolume.set(cls.protImportVol.outputVolume)
        cls.launchProtocol(cls.protCreateMask)
        cls.assertIsNotNone(cls.protCreateMask.getFiles(), "There was a problem with the 3D mask")
        cls.protClassify = cls.runClassify(cls.protImportPart.outputParticles)
        cls.protProjMatch = cls.newProtocol(XmippProtProjMatch,
                                            doCTFCorrection=False,
                                            numberOfIterations=1,
                                            outerRadius=50,
                                            angSamplingRateDeg=5,
                                            symmetry="d6",
                                            numberOfMpi=4,
                                            useGpu=False)
        cls.protProjMatch.inputParticles.set(cls.protImportAvgs.outputAverages)
        cls.protProjMatch.input3DReferences.set(cls.protImportVol.outputVolume)
        cls.launchProtocol(cls.protProjMatch)

    @classmethod
    def importFromRelionRefine3D(cls):
        """ Import aligned Particles
        """
        protImport3D = cls.newProtocol(emprot.ProtImportParticles,
                                           objLabel='particles from relion (auto-refine 3d)',
                                           importFrom=emprot.ProtImportParticles.IMPORT_FROM_RELION,
                                           starFile=
                                           cls.dsRelion.getFile('import/classify3d/extra/relion_it015_data.star'),
                                           magnification=10000,
                                           samplingRate=7.08,
                                           haveDataBeenPhaseFlipped=True)
        cls.launchProtocol(protImport3D)
        return protImport3D

    @classmethod
    def importFromRelion2D(cls):
        """
        Import an EMX file with Particles and defocus
        """
        protImport2D = cls.newProtocol(emprot.ProtImportParticles,
                                           objLabel='from relion (classify 2d)',
                                           importFrom=emprot.ProtImportParticles.IMPORT_FROM_RELION,
                                           starFile=cls.dsRelion.getFile(
                                             'import/classify2d/extra/relion_it015_data.star'),
                                           magnification=10000,
                                           samplingRate=7.08,
                                           haveDataBeenPhaseFlipped=True)
        cls.launchProtocol(protImport2D)
        return protImport2D

    def test_particles1(self):
        print("Run Compare Reprojections from classes")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                        symmetryGroup="d6", numberOfMpi=5, doRanking=False)
        prot.inputSet2D.set(self.protClassify.outputClasses)
        prot.inputSet3D.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol203, "There was a problem with Compare Reprojections from classes")

    def test_particles2(self):
        print("Run Compare Reprojections from averages")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                        symmetryGroup="d6", numberOfMpi=5, doRanking=False)
        prot.inputSet2D.set(self.protImportAvgs.outputAverages)
        prot.inputSet3D.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol203, "There was a problem with Compare Reprojections from averages")

    def test_particles3(self):
        print("Run Compare Reprojections from projections with angles")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                        symmetryGroup="d6", numberOfMpi=5, doRanking=False)
        prot.inputSet2D.set(self.protProjMatch.outputParticles)
        prot.inputSet3D.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol203,
                             "There was a problem with Compare Reprojections from projections with angles")

    def test_particles4(self):
        print("Run Compare Reprojections from classes evaluating residuals")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                symmetryGroup="d6", numberOfMpi=5, doEvaluateResiduals=True, doRanking=False)
        prot.inputSet2D.set(self.protClassify.outputClasses)
        prot.inputSet3D.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol203,
                             "There was a problem with Compare Reprojections from classes evaluating residuals")

    def test_particles5(self):
        print("Run Compare Reprojections from averages evaluating residuals")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                symmetryGroup="d6", numberOfMpi=5, doEvaluateResiduals=True, doRanking=False)
        prot.inputSet2D.set(self.protImportAvgs.outputAverages)
        prot.inputSet3D.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol203,
                             "There was a problem with Compare Reprojections from averages evaluating residuals")

    def test_particles6(self):
        print("Run Compare Reprojections from projections with angles evaluating residuals")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                symmetryGroup="d6", numberOfMpi=5, doEvaluateResiduals=True, doRanking=False)
        prot.inputSet2D.set(self.protProjMatch.outputParticles)
        prot.inputSet3D.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol203,
                             "There was a problem with Compare Reprojections from projections"
                             " with angles evaluating residuals")

    def test_particles7(self):
        print("Run Compare Reprojections from projections with angles evaluating residuals without user mask")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                symmetryGroup="d6", numberOfMpi=5, doEvaluateResiduals=True, doRanking=False)
        prot.inputSet2D.set(self.protProjMatch.outputParticles)
        prot.inputSet3D.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol203,
                             "There was a problem with Compare Reprojections from projections"
                             " with angles evaluating residuals without user mask")

    def test_ranking3D(self):
        print("Run Compare Reprojections from 3d and 2d classes and then make a ranking with the best volume")
        prot = self.newProtocol(XmippProtCompareReprojections,
                                symmetryGroup="d6", numberOfMpi=5)
        prot.inputSet2D.set(self.protImport2D.outputClasses)
        prot.inputSet3D.set(self.protImport3D.outputClasses)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.reprojections_vol1, "There was a problem with Compare Reprojections vol 1")
        self.assertIsNotNone(prot.reprojections_vol2, "There was a problem with Compare Reprojections vol 2")
        self.assertIsNotNone(prot.reprojections_vol3, "There was a problem with Compare Reprojections vol 3")
        self.assertSetSize(prot.particles_bestVol, size=899)
        self.assertIsNotNone(prot.bestVolume, "There is a problem with the ranking the best volume does not exist")

class TestXmippCreateGallery(TestXmippBase):
    """This class check if the protocol create gallery in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImportVol = cls.runImportVolume(cls.volumesFn, 3.5)
    
    def _createGallery(self, step, projections):
        prot = self.newProtocol(XmippProtCreateGallery,
                                symmetryGroup="d6",
                                rotStep=step, tiltStep=step)
        prot.inputVolume.set(self.protImportVol.outputVolume)
        self.launchProtocol(prot)
        outSet = getattr(prot, 'outputReprojections', None)
        self.assertIsNotNone(outSet, "There was a problem with create gallery")
        self.assertEqual(projections, outSet.getSize())

        return prot

    def test_step5(self):
        self._createGallery(step=5, projections=131)

    def test_step10(self):
        self._createGallery(step=10, projections=32)


class TestXmippBreakSym(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
    
    def test_AngBreakSymmetry(self):
        from tempfile import NamedTemporaryFile
        import pwem.emlib.metadata as md
        
        fileTmp = NamedTemporaryFile(delete=False, suffix='.sqlite')
        partSet = SetOfParticles(filename=fileTmp.name)
        partSet.setAlignment(ALIGN_PROJ)
        # Populate the SetOfParticles with  images
        # taken from images.mrc file
        # and setting the previous alignment parameters
        
        m = np.array([[ 0.71461016, 0.63371837, -0.29619813, 15],
                      [ -0.61309201, 0.77128059, 0.17101008, 25],
                      [ 0.33682409, 0.059391174, 0.93969262, 35],
                      [ 0,          0,           0,           1]])
        p = Particle()
        p.setLocation(1, "kk.mrc")

        p.setTransform(Transform(m))
        partSet.append(p)
        partSet.write()

        print("import particles")
        protImport = self.newProtocol(emprot.ProtImportParticles,
                                         sqliteFile=fileTmp.name, samplingRate=1, importFrom=4,
                                         checkStack=False, haveDataBeenPhaseFlipped=False)
        self.launchProtocol(protImport)

        print("Run AngBreakSymmetry particles")
        protBreakSym = self.newProtocol(XmippProtAngBreakSymmetry, symmetryGroup="i2")
        protBreakSym.inputParticles.set(protImport.outputParticles)
        self.launchProtocol(protBreakSym)
        os.chdir(protBreakSym._getPath())
        from pyworkflow.utils import runJob
        runJob(None, 'xmipp_angular_distance',
               "--ang1 images.xmd --ang2 input_particles.xmd --sym i2 --oroot kk",
               env=xmipp3.Plugin.getEnviron())
        mdRober = md.MetaData("kk_vec_diff_hist.txt")
        objId = mdRober.firstObject()
        count = mdRober.getValue(md.MDL_COUNT, objId)
        
        self.assertEqual(count, 1, "There was a problem with break symmetry")
        os.unlink(fileTmp.name)


class TestXmippCorrectWiener2D(TestXmippBase):
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData()
    
    def test_CorrectWiener(self):
        prot1 = self.newProtocol(emprot.ProtImportParticles,
                                 importFrom=emprot.ProtImportParticles.IMPORT_FROM_XMIPP3,
                                 mdFile=self.dataset.getFile('particles/sphere_128.xmd'),
                                 magnification=10000,
                                 samplingRate=1,
                                 haveDataBeenPhaseFlipped=False
                                 )
        self.launchProtocol(prot1)
        print("Run CTFCorrectWiener2D particles")
        protCorrect = self.newProtocol(XmippProtCTFCorrectWiener2D)
        protCorrect.inputParticles.set(prot1.outputParticles)
        self.launchProtocol(protCorrect)
        self.assertIsNotNone(protCorrect.outputParticles, "There was a problem with Wiener Correction")

class TestXmippPickNoise(TestXmippBase):
    """This class checks if the protocol pick noise in Xmipp works properly."""
    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.dataset = DataSet.getDataSet('xmipp_tutorial')
        cls.micsFn = cls.dataset.getFile('micrographs/BPV_1386.mrc')
        cls.coordFn = cls.dataset.getFile('pickingXmipp/pickedAll/BPV_1386.pos')

    def testXmippPickNoise(self):
        # Import micrographs
        protImportMics = self.newProtocol(emprot.ProtImportMicrographs,
                                          filesPath=self.micsFn,
                                          samplingRate=1.237,
                                          voltage=300)
        self.launchProtocol(protImportMics)
        self.assertIsNotNone(protImportMics.outputMicrographs, (MSG_WRONG_IMPORT, "micrographs"))
        # Import coordinates
        protImportCoords = self.newProtocol(emprot.ProtImportCoordinates,
                                            filesPath=self.coordFn,
                                            inputMicrographs=protImportMics.outputMicrographs,
                                            boxSize=110)
        self.launchProtocol(protImportCoords)
        self.assertIsNotNone(protImportCoords.outputCoordinates, (MSG_WRONG_IMPORT, "coordinates"))
        # Protocol Pick Noise (default values)
        protPickNoise1 = self.newProtocol(XmippProtPickNoise,
                                         inputCoordinates=protImportCoords.outputCoordinates)
        self.launchProtocol(protPickNoise1)
        self.assertIsNotNone(protPickNoise1.getFiles(), (MSG_WRONG_PROTOCOL, "pick noise"))
        self.assertIsNotNone(protPickNoise1.outputCoordinates, (MSG_WRONG_OUTPUT, "coordinates"))
        # Protocol Pick Noise (extract noise number)
        protPickNoise2 = self.newProtocol(XmippProtPickNoise,
                                         inputCoordinates=protImportCoords.outputCoordinates,
                                         extractNoiseNumber=140)
        self.launchProtocol(protPickNoise2)
        self.assertIsNotNone(protPickNoise2.getFiles(), (MSG_WRONG_PROTOCOL, "pick noise"))
        self.assertIsNotNone(protPickNoise2.outputCoordinates, (MSG_WRONG_OUTPUT, "coordinates"))
        # Check if the number of noisy particles is right
        self.assertEquals(protPickNoise1.outputCoordinates.getSize(), 143, (MSG_WRONG_SIZE, "noisy particles"))
        self.assertEquals(protPickNoise2.outputCoordinates.getSize(), 140, (MSG_WRONG_SIZE, "noisy particles"))


class TestXmippClassifyPca(TestXmippBase):
    """This class check if the protocol Classify PCA (static and in streaming) in Xmipp works properly."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.dsRelion = DataSet.getDataSet('relion_tutorial')
        cls.protImport = cls.importParticles()

    @classmethod
    def importParticles(cls):
        pathToFile = 'import/case2/relion_it015_data.star'
        importProt = cls.newProtocol(emprot.ProtImportParticles,
                                     objLabel='from relion (auto-refine 3d)',
                                     importFrom=emprot.ProtImportParticles.IMPORT_FROM_RELION,
                                     starFile=cls.dsRelion.getFile(pathToFile),
                                     magnification=10000,
                                     samplingRate=7.08,
                                     haveDataBeenPhaseFlipped=True
                                     )
        cls.launchProtocol(importProt)

        return importProt

    def _updateProtocol(self, prot):
        prot2 = getProtocolFromDb(prot.getProject().path,
                                  prot.getDbPath(),
                                  prot.getObjId())
        # Close DB connections
        prot2.getProject().closeMapper()
        prot2.closeMappers()
        return prot2

    def test_ClassifyPCAStreaming(self):
        print("Run 1st Classify PCA Static")
        protPCA1 = self.newProtocol(XmippProtClassifyPca,
                                    objLabel="Classify Pca - static version",
                                    numberOfClasses=5, numberOfMpi=4, numberOfThreads=1)
        protPCA1.inputParticles.set(self.protImport.outputParticles)
        self.proj.scheduleProtocol(protPCA1)

        print("Start Streaming Particles")
        protStream = self.newProtocol(emprot.ProtCreateStreamData, setof=3,
                                      creationInterval=5, samplingRate=7.08, nDim=6000, groups=500)
        protStream.inputParticles.set(self.protImport.outputParticles)
        self.proj.scheduleProtocol(protStream, prerequisites=[protPCA1.getObjId()])

        print("Run 2nd Classify PCA Streaming")
        protPCA2 = self.newProtocol(XmippProtClassifyPcaStreaming,
                                    objLabel="Classify Pca streaming - update classes",
                                    training=2000,
                                    correctCtf=False,
                                    mode=XmippProtClassifyPcaStreaming.UPDATE_CLASSES
                                    )
        protPCA2.initialClasses.set(protPCA1)
        protPCA2.initialClasses.setExtended("outputAverages")

        protPCA2.inputParticles.set(protStream)
        protPCA2.inputParticles.setExtended("outputParticles")

        self.proj.scheduleProtocol(protPCA2)

        # when the first 2D Classification (static mode) finishes must have 5 classes.
        self.checkResults(protPCA1, 5, "Static mode classification")
        # when the 2D update Classification (streaming mode) finishes must have 5 classes.
        self.checkResults(protPCA2, 5, "Streaming mode update initial classification")

        time.sleep(5)  # sometimes is not ready yet

        # Check a final update
        self.checkFinal2DClasses1st(protPCA1, msg="Initial classification mode fails")
        self.checkFinal2DClasses2nd(protPCA2, msg="Update classification in streaming mode fails")

    def checkFinal2DClasses1st(self, prot, outputName="outputClasses", msg=''):
        prot = self._updateProtocol(prot)
        output = getattr(prot, outputName, None)
        self.assertIsNotNone(output, msg)
        self.assertSetSize(output, 5, msg)
        self.assertSetSize(output.getImages(), 5236, msg)

    def checkFinal2DClasses2nd(self, prot, outputName="outputClasses", msg=''):
        prot = self._updateProtocol(prot)
        output = getattr(prot, outputName, None)
        self.assertIsNotNone(output, msg)
        self.assertSetSize(output, 5, msg)

    def checkResults(self, prot, size, msg=''):
        t0 = time.time()
        while not prot.isFinished():
            # Time out 4 minutes, just in case
            tdelta = time.time() - t0
            if tdelta > 4 * 60:
                break
            prot = self._updateProtocol(prot)
            time.sleep(2)

        self.assertSetSize(prot.outputClasses, size, msg)


class TestXmippProtCL2DClustering(TestXmippBase):
    """This class check if the protocol clustering 2d classes in Xmipp works properly."""

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        TestXmippBase.setData('mda')
        cls.protImportAvgs = cls.runImportAverages(cls.averagesFn, 3.5)

    def test_clustering(self):
        print("Run clustering 2D classes from 2D averages")
        prot = self.newProtocol(XmippProtCL2DClustering,
                                min_cluster=3, max_cluster=-1, extractOption=1)
        prot.inputSet2D.set(self.protImportAvgs.outputAverages)
        self.launchProtocol(prot)
        self.assertIsNotNone(prot.outputAverages,
                             "There was a problem with Clustering 2D Classes")
        self.assertSetSize(prot.outputAverages, size=3)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        className = sys.argv[1]
        cls = globals().get(className, None)
        if cls:
            suite = unittest.TestLoader().loadTestsFromTestCase(cls)
            unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            print("Test: '%s' not found." % className)
    else:
        unittest.main()
