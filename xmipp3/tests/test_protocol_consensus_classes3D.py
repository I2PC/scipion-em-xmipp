#!/usr/bin/env python3

# ***************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
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

import random

from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pyworkflow.utils import greenStr
from pwem.objects import Class3D
from pwem.protocols import ProtClassify3D
from pwem.protocols import ProtImportParticles

from xmipp3.protocols import XmippProtConsensusClasses3D


class TestConsensusClasses3D(BaseTest):
    @classmethod
    def setUpClass(cls):
        """Prepare the data that we will use later on."""

        print("\n", greenStr(" Set Up - Collect data ".center(75, '-')))

        setupTestProject(cls)  # defined in BaseTest, creates cls.proj

        cls.xmippDataTest = DataSet.getDataSet('xmipp_tutorial')


        def _createClasses(label, particles, numClasses, randomSeed, numPart=None):
            """ We import a set of classes with an alterate import particles
                because there is not a import classes protocol
            """
            # Create a dummy class with the particles
            pImpClasses = cls.proj.newProtocol(ProtClassify3D)
            pImpClasses.setObjLabel(label)
            cls.proj.launchProtocol(pImpClasses, wait=True)

            setOfClasses = pImpClasses._createSetOfClasses3D(particles)

            numOfPart = particles.getSize() if numPart is None else numPart
            partIds = list(particles.getIdSet())
            m = int(numOfPart/numClasses)
            
            # random shuffle with a certain seed to get always the same classes
            random.seed(randomSeed)
            random.shuffle(partIds)
            for clInx in list(range(numClasses)):
                currIds = partIds[clInx*m:(clInx+1)*m]

                newClass = Class3D()
                newClass.copyInfo(particles)
                newClass.setAcquisition(particles.getAcquisition())
                # newClass.setRepresentative(clRep.getRepresentative())

                setOfClasses.append(newClass)

                enabledClass = setOfClasses[newClass.getObjId()]
                enabledClass.enableAppend()
                for itemId in currIds:
                    item = particles[itemId]
                    enabledClass.append(item)

                setOfClasses.update(enabledClass)

            # including the outputClasses as protocol output by hand 
            pImpClasses._defineOutputs(outputClasses=setOfClasses)

            # relaunching the protocol to get the new output
            cls.proj.launchProtocol(pImpClasses, wait=True)

            return pImpClasses.outputClasses

        # Import particles
        pImpParticles = cls.proj.newProtocol(ProtImportParticles,
                            filesPath=cls.xmippDataTest.getFile('particles'),
                            samplingRate=3.5)
        pImpParticles.setObjLabel('Import particles')

        # Launch the protocol to obtain the particles
        cls.proj.launchProtocol(pImpParticles, wait=True)

        # fake importing classes handmade
        particles = pImpParticles.outputParticles

        cls.set1 = _createClasses('3 Classes I', particles, 3, 65)
        cls.set2 = _createClasses('3 Classes II', particles, 3, 123)
        cls.set3 = _createClasses('3 Classes III', particles, 3, 256)
        cls.set4 = _createClasses('5 Classes I', particles, 5, 745)
        cls.set5 = _createClasses('5 Classes II', particles, 5, 1025)


    def checkIntersections(self, setOfIntersections, classId, partIds):
        """ Some assertions for a certain class in the setOfIntersections
        """
        classItem = setOfIntersections[classId]
        self.assertEqual(classItem.getSize(), len(partIds), 
                         "The size of the class %d is wrong" % classId)
        self.assertEqual(sorted(partIds), sorted(list(classItem.getIdSet())),
                         "The particles in the class %d are wrong." % classId)

    def checkPopulation(self, setOfIntersections, population):
        numOfPart = 0
        for classItem in setOfIntersections:
            numOfPart += classItem.getSize()

        self.assertEqual(numOfPart, population,
                         "The total number of particles is wrong")


    # The tests themselves.
    #
    def testConsensus(self):
        print("\n", greenStr(" Test Consensus ".center(75, '-')))

        # preparing and launching the protocol
        pConsClass = self.proj.newProtocol(XmippProtConsensusClasses3D,
                                           inputClassifications=[self.set1, 
                                                                 self.set2, 
                                                                 self.set3])
        self.proj.launchProtocol(pConsClass, wait=True)
        setOfIntersections = pConsClass.outputClasses_initial

        # some general assertions
        self.assertIsNotNone(setOfIntersections,
                             "There was some problem with the output")
        self.assertEqual(setOfIntersections.getSize(), 26,
                         "The number of the outputClasses is wrong")
        self.checkPopulation(setOfIntersections, 75)


    def testConsensus2(self):
        print("\n", greenStr(" Test Consensus with different set sizes".center(75, '-')))

        # preparing and launching the protocol
        pConsClass = self.proj.newProtocol(XmippProtConsensusClasses3D,
                                           inputClassifications=[self.set1, 
                                                                 self.set2, 
                                                                 self.set4])
        self.proj.launchProtocol(pConsClass, wait=True)
        setOfIntersections = pConsClass.outputClasses_initial

        # some general assertions
        self.assertIsNotNone(setOfIntersections,
                             "There was some problem with the output")
        self.assertEqual(setOfIntersections.getSize(), 38,
                         "The number of the outputClasses is wrong")
        self.checkPopulation(setOfIntersections, 75)


    def testConsensus3(self):
        print("\n", greenStr(" Test Consensus with different set sizes 2".center(75, '-')))

        # preparing and launching the protocol
        pConsClass = self.proj.newProtocol(XmippProtConsensusClasses3D,
                                           inputClassifications=[self.set4, 
                                                                 self.set2, 
                                                                 self.set1])
        self.proj.launchProtocol(pConsClass, wait=True)
        setOfIntersections = pConsClass.outputClasses_initial

        # some general assertions
        self.assertIsNotNone(setOfIntersections,
                             "There was some problem with the output")
        self.assertEqual(setOfIntersections.getSize(), 38,
                         "The number of the outputClasses is wrong")
        self.checkPopulation(setOfIntersections, 75)


    def testConsensus4(self):
        print("\n", greenStr(" Test Consensus with different class sizes".center(75, '-')))

        # preparing and launching the protocol
        pConsClass = self.proj.newProtocol(XmippProtConsensusClasses3D,
                                           inputClassifications=[self.set5, 
                                                                 self.set2, 
                                                                 self.set1])
        self.proj.launchProtocol(pConsClass, wait=True)
        setOfIntersections = pConsClass.outputClasses_initial

        # some general assertions
        self.assertIsNotNone(setOfIntersections,
                             "There was some problem with the output")
        self.assertEqual(setOfIntersections.getSize(), 36,
                         "The number of the outputClasses is wrong")
        self.checkPopulation(setOfIntersections, 75)

    def testConsensus6(self):
        print("\n", greenStr(" Test Consensus with four sets".center(75, '-')))

        # preparing and launching the protocol
        pConsClass = self.proj.newProtocol(XmippProtConsensusClasses3D,
                                           inputClassifications=[self.set1,
                                                                 self.set2,
                                                                 self.set3,
                                                                 self.set5 ])
        self.proj.launchProtocol(pConsClass, wait=True)
        setOfIntersections = pConsClass.outputClasses_initial

        # some general assertions
        self.assertIsNotNone(setOfIntersections,
                             "There was some problem with the output")
        self.assertEqual(setOfIntersections.getSize(), 61,
                         "The number of the outputClasses is wrong")
        self.checkPopulation(setOfIntersections, 75)

    def testConsensus7(self):
        print("\n", greenStr("Test Consensus with four sets and a random reference classification".center(75, '-')))

        # preparing and launching the protocol
        pConsClass = self.proj.newProtocol(XmippProtConsensusClasses3D,
                                           inputClassifications=[self.set1,
                                                                 self.set2,
                                                                 self.set3,
                                                                 self.set5 ],
                                           randomClassificationCount=1000 )
        self.proj.launchProtocol(pConsClass, wait=True)
        setOfIntersections = pConsClass.outputClasses_initial

        # Ensure that the intersection size has been written
        for cls in setOfIntersections:
            self.assertIsNotNone(cls._xmipp_classIntersectionSizePValue)
            self.assertIsNotNone(cls._xmipp_classIntersectionRelativeSizePValue)