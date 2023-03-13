# **************************************************************************
# *
# * Authors:    Jeison Mendez (jmendez@utp.edu.co)
# 
# * Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas -- IIMAS
# * Universidad Nacional Autónoma de México -UNAM
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

import sys, time
import numpy as np
from os.path import exists, join
from shutil import copy

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, FloatParam, BooleanParam,
                                        IntParam, StringParam, LEVEL_ADVANCED,
                                        USE_GPU, GPU_LIST)

from pyworkflow.project import Manager
from pyworkflow.protocol import getProtocolFromDb
import pyworkflow.object as pwobj

from pwem.protocols import EMProtocol, ProtAlignmentAssign
from pwem.objects import Volume

from pyworkflow.plugin import Domain

#from xmipp3.protocols import XmippMetaProtCreateOutput # ya no está
from xmipp3.protocols import XmippMetaProtCreateSubset

from xmipp3.protocols import XmippProtAngularGraphConsistency
from xmipp3.protocols import XmippProtReconstructHighRes

from pwem.emlib.metadata import iterRows
from xmipp3.convert import readSetOfParticles, readSetOfImages
from pwem import emlib

class XmippMetaProtDiscreteGraphHeterogeneity2(EMProtocol):
    """ Metaprotocol to run together some protocols in order to deal with heterogeneity
        in a dataset, using low resolution validation based on graph analysis
     """
    _label = 'metaprotocol heterogeneity - graph validation 2'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **kwargs):
        EMProtocol.__init__(self, **kwargs)
        self._runIds = pwobj.CsvList(pType=int)
        self.childs = []

    def setAborted(self):
        for child in self.childs:
            if child.isRunning() or child.isScheduled():
                # child.setAborted()
                self.getProject().stopProtocol(child)

        EMProtocol.setAborted(self)

    def setFailed(self, errorMsg):
        for child in self.childs:
            if child.isRunning() or child.isScheduled():
                # child.setFailed(errorMsg)
                self.getProject().stopProtocol(child)

        EMProtocol.setFailed(self, errorMsg)

    # --------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addHidden(USE_GPU, BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation.\
                       Select the one you want to use.")

        form.addHidden(GPU_LIST, StringParam, default='0',
                       expertLevel=LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used")

        form.addSection(label='Input')

        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input volume",
                      help='Select the input volume.')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      label="Input particles",
                      help='Select the input experimental images')
        form.addParam('maxNumClasses', IntParam, default=4,
                      label='Maximum number of classes to calculate',
                      help='Maximum number of classes to calculate'),
        form.addParam('maxNumIter', IntParam, default=4,
                      label='Maximum number of iterations',
                      help='Maximum number of iterations')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help="See http://xmipp.cnb.csic.es/twiki/bin/view/Xmipp/Symmetry"
                           " for a description of the symmetry groups format in Xmipp.\n"
                           "If no symmetry is present, use _c1_.")
        form.addParam('particleRadius', IntParam, default=-1,
                      label='Radius of particle (px)',
                      help='This is the radius (in pixels) of the spherical mask '
                           'covering the particle in the input images')
        form.addParam('targetResolution', FloatParam, default=8,
                      label='Target resolution',
                      help='Target resolution to solve for the heterogeneity')



        form.addParallelSection(threads=1, mpi=8)

    # --------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self.classListProtocols = []
        self.classListSizes = []
        self.classListIds = []
        self.finished = False
        self._insertFunctionStep('monitorStep')

    # --------------------------- STEPS functions ----------------------------
    def monitorStep(self):

        self._runPrerequisites = []
        manager = Manager()
        project = manager.loadProject(self.getProject().getName())
        
        self.numIter = self.maxNumIter.get()   
        
        print('convertInput')
        sys.stdout.flush()
        iter = 0
        self.convertInputStep(iter)
        
        print('RelionClassify3D_Protocol')
        sys.stdout.flush()
        ProtRelionClassify3D = Domain.importFromPlugin('relion.protocols', 
                                                       'ProtRelionClassify3D', 
                                                       doRaise=True)
        
        relionClassify3DProt = project.newProtocol(
                            ProtRelionClassify3D,
                            objLabel='relion classify 3D - iter %d' % iter,
                            symmetryGroup=self.symmetryGroup.get(),
                            numberOfClasses=self.maxNumClasses.get(),
                            numberOfIterations=15, 
                            numberOfMpi=2, 
                            numberOfThreads=2,
                            doCTF=True, 
                            initialLowPassFilterA=30,
                            doGpu=self.useGpu.get(),
                            gpusToUse=self.gpuList.get()
                            )  
        # reference files  
        # gedit ./.scipion3env/lib/python3.6/site-packages/relion/protocols/protocol_base.py
        # gedit ./.scipion3env/lib/python3.6/site-packages/relion/tests/test_protocols_relion.py
        # gedit ./.scipion3env/lib/python3.6/site-packages/relion/protocols/protocol_classify3d.py    
         
        previousProtPart = self
        relionClassify3DProt.inputParticles.set(previousProtPart)
        relionClassify3DProt.inputParticles.setExtended('outputParticlesInit') # is important this name to match
         
        previousProtVol = self
        relionClassify3DProt.referenceVolume.set(previousProtVol)
        relionClassify3DProt.referenceVolume.setExtended('outputVolumesInit') # is important this name to match
                        
        project.launchProtocol(relionClassify3DProt)
        # Next schedule will be after this one
        self._runPrerequisites.append(relionClassify3DProt.getObjId())
        self.childs.append(relionClassify3DProt)
        
        if not self.finished:
            finishedIter = False
            while finishedIter == False:
                time.sleep(15)
                relionClassify3DProt = self._updateProtocol(relionClassify3DProt)
                if relionClassify3DProt.isFailed() or relionClassify3DProt.isAborted():
                    raise Exception('ProtRelionClassify3D has failed')
                if relionClassify3DProt.isFinished():
                    finishedIter = True
                    for classItem in relionClassify3DProt.outputClasses:
                        self.classListSizes.append(classItem.getSize())
                        self.classListIds.append(classItem.getObjId())
                        self.classListProtocols.append(relionClassify3DProt)
    
        numIter = self.maxNumIter.get() + 1
        highresProtocols = []
        validationProtocols = []
        cIter = -1
        for iter in range(1, numIter):
            cIter += 1
            highresProtocols.append([])
            validationProtocols.append([])
            cItem = -1
            for classItem in relionClassify3DProt.outputClasses:
                classVar = classItem.getObjId()
                cItem += 1

                if iter == 1:
                    previousProtPart = self
                    partName = 'outputParticlesInit'
                    previousProtVol = relionClassify3DProt
                    volumeName = 'outputVolumes.%d'%classItem.getObjId()
                else:
                    prevIter = cIter - 1
                    # previousProtPart = highresProtocols[prevIter][cItem]
                    # partName = 'outputParticles'
                    previousProtPart = self
                    partName = 'outputParticlesInit'
                    previousProtVol = highresProtocols[prevIter][cItem]
                    volumeName = 'outputVolume'
                
                print('Highres (one class) - iter', iter,'class',classVar)
                print('particlesName:',partName,'\tvolumeName:',volumeName)
                sys.stdout.flush()
                
                newHighres = project.newProtocol(
                        XmippProtReconstructHighRes,
                        objLabel='xmipp Highres - iter %d class %d' % (iter,classVar),
                        symmetryGroup=self.symmetryGroup.get(),
                        numberOfIterations=1, 
                        particleRadius = self.particleRadius.get(),
                        maximumTargetResolution = self.targetResolution.get(),
                        #multiresolution=False,
                        numberOfMpi=self.numberOfMpi.get(),
                        alignmentMethod=XmippProtReconstructHighRes.GLOBAL_ALIGNMENT,
                        useGpu=self.useGpu.get(), # si algo cambiar a false
                        gpuList = self.gpuList.get()
                        )    
                newHighres.inputParticles.set(previousProtPart)
                newHighres.inputParticles.setExtended(partName) 
                 
                newHighres.inputVolumes.set(previousProtVol)
                newHighres.inputVolumes.setExtended(volumeName) 
                
                project.scheduleProtocol(newHighres, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(newHighres.getObjId())
                self.childs.append(newHighres)                               
                # self._finishedProt(newHighres)

                # append to list
                highresProtocols[cIter].append(newHighres)
                print("Iter %d, Class %d"%(cIter,cItem))
                print("TestInfo newHighres.getObjId(): ", highresProtocols[cIter][cItem].getObjId())
        
                # angular graph validation                
                print('angular graph validation - iter',iter,'class',classVar)
                sys.stdout.flush()
                
                validationProt = project.newProtocol(
                    XmippProtAngularGraphConsistency,
                    objLabel='graphValidation - iter  %d class %d' % (iter,classVar),
                    symmetryGroup=self.symmetryGroup.get(),
                    maximumTargetResolution=10,
                    numberOfMpi=self.numberOfMpi.get()
                    )
                previousProtPart = newHighres
                partName = 'outputParticles'
                validationProt.inputParticles.set(previousProtPart)
                validationProt.inputParticles.setExtended(partName) # todo no es así !!
                
                validationProt.inputVolume.set(previousProtVol)
                validationProt.inputVolume.setExtended(volumeName) 
                
                project.scheduleProtocol(validationProt, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(validationProt.getObjId())
                self.childs.append(validationProt)     
                self._finishedProt(validationProt)
                
                # append to list
                validationProtocols[cIter].append(validationProt)
                print("Iter %d, Class %d"%(cIter,cItem))
                print("TestInfo validationProt.getObjId(): ", validationProtocols[cIter][cItem].getObjId())
        
                    

                # copy outputdata from validation protocol to metaprotocol extraPath folder
                # in order to get info about
                # _xmipp_maxCCprevious, _xmipp_maxCCprevious, _xmipp_distance2MaxGraphPrevious
                fnOutParticles = validationProt._getPath('angles.xmd')
                fnInputTest = self._getExtraPath('partGraphValidated_Class_%s.xmd' % (classVar))
                copy(fnOutParticles, fnInputTest)

            
            # assign each particle to a particular class
            self.separateParticles()

            # remove disabled particles
            # class1
            i=0
            fnInputNext_1 = self._getPath('inputHighres%s.xmd'%str(i+1)) # TODO cambiar por chr(65+i)
            copy(self.fnAuxPart1_copy, fnInputNext_1)
            params = '-i %s --query select "enabled==1"' % (fnInputNext_1)
            self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)
            self.subsets = []
            self.subsets.append(self._createSetOfParticles(str(i)))
            self.subsets[i].copyInfo(self.inputParticles.get())
            readSetOfParticles(fnInputNext_1, self.subsets[i])
            result = {'outputParticlesClass%s'%str(i+1) : self.subsets[i]}
            self._defineOutputs(**result)
            self._store(self.subsets[i])
            # # al definir estas salidas el metaprotocolo se pone en FINISHED
            # outputinputSetOfParticles = self._createSetOfParticles()
            # outputinputSetOfParticles.copyInfo(self.inputParticles.get())
            # readSetOfParticles(fnInputNext_1, outputinputSetOfParticles)
            # self._defineOutputs(outputParticlesClass1=outputinputSetOfParticles)
            # self._store(outputinputSetOfParticles)

            # class2
            i=1
            fnInputNext_2 = self._getPath('inputHighres%s.xmd'%str(i+1))
            copy(self.fnAuxPart2_copy, fnInputNext_2)
            params = '-i %s --query select "enabled==1"' % (fnInputNext_2)
            self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)
            self.subsets.append(self._createSetOfParticles(str(i)))
            self.subsets[i].copyInfo(self.inputParticles.get())
            readSetOfParticles(fnInputNext_2, self.subsets[i])
            result = {'outputParticlesClass%s'%str(i+1) : self.subsets[i]}
            self._defineOutputs(**result)
            self._store(self.subsets[i])
            # # # al definir estas salidas el metaprotocolo se pone en FINISHED
            # outputinputSetOfParticles = self._createSetOfParticles()
            # outputinputSetOfParticles.copyInfo(self.inputParticles.get())
            # readSetOfParticles(fnInputNext_2, outputinputSetOfParticles)
            # self._defineOutputs(outputParticlesClass2=outputinputSetOfParticles)
            # self._store(outputinputSetOfParticles)

            # reconstruct new references using highres
            #for each class
            cItem = -1
            for classItem in relionClassify3DProt.outputClasses:
                cItem += 1
                classVar = classItem.getObjId()
                # TODO las líneas que están antes de este for debo ponerlas acá dentro
                print('Highres (one class) - iter', iter,'class',classVar)
                newHighres = project.newProtocol(
                        XmippProtReconstructHighRes,
                        objLabel='xmipp Highres 2 - iter %d class %d' % (iter,classVar),
                        symmetryGroup=self.symmetryGroup.get(),
                        numberOfIterations=1, 
                        particleRadius = self.particleRadius.get(),
                        maximumTargetResolution = self.targetResolution.get(),
                        #multiresolution=False,
                        numberOfMpi=self.numberOfMpi.get(),
                        alignmentMethod=XmippProtReconstructHighRes.GLOBAL_ALIGNMENT,
                        useGpu=self.useGpu.get(), # si algo cambiar a false
                        gpuList = self.gpuList.get()
                        )   
                
                previousProtPart = self
                partName = 'outputParticlesClass%s'%classItem.getObjId()
                previousProtVol = highresProtocols[cIter][cItem]
                volumeName = 'outputVolume'
                newHighres.inputParticles.set(previousProtPart)
                newHighres.inputParticles.setExtended(partName) 
                 
                newHighres.inputVolumes.set(previousProtVol)
                newHighres.inputVolumes.setExtended(volumeName) 

                project.scheduleProtocol(newHighres, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(newHighres.getObjId())
                self.childs.append(newHighres)                               
                self._finishedProt(newHighres)

                # replace highres protocol
                highresProtocols[cIter][cItem] = newHighres


             #clear prerequisites after 1st iter
            self._runPrerequisites.clear()  


            ## this approach hasn't worked
            ## for a large set of particles 
            ## exceeds my laptop's memmory

            # fnAuxPart1 = self._getExtraPath('partGraphValidated_Class_%s.xmd' % str(1))
            # mdParticles1 = emlib.MetaData(fnAuxPart1)
            # fnAuxPart2 = self._getExtraPath('partGraphValidated_Class_%s.xmd' % str(2))
            # mdParticles2 = emlib.MetaData(fnAuxPart2)
            # nParticles = 0
            # c1 = 0
            # c2 = 0
            # count = 0
            # for row1 in iterRows(mdParticles1):
            #     count += 1
            #     maxCC_p_1 = row1.getValue(emlib.MDL_MAXCC_PREVIOUS)
            #     graphCCPrev_1 = row1.getValue(emlib.MDL_GRAPH_CC_PREVIOUS)
            #     graphDistMaxGraphPrev_1 = row1.getValue(emlib.MDL_GRAPH_DISTANCE2MAX_PREVIOUS)
            #     # objId_1 = row1.getValue(emlib.MDL_PARTICLE_ID)
            #     print("count = %d" % count)
            #     objId_1 = row1.getObjId()
            #     sys.stdout.flush()
                
            #     for row2 in iterRows(mdParticles2):
            #         # objId_2 = row2.getValue(emlib.MDL_PARTICLE_ID)
            #         objId_2 = row2.getObjId()
            #         if objId_1 == objId_2:
            #             # print("same particle: %d , %d" % ( objId_1, objId_2) )
            #             maxCC_p_2 = row2.getValue(emlib.MDL_MAXCC_PREVIOUS)
            #             graphCCPrev_2 = row2.getValue(emlib.MDL_GRAPH_CC_PREVIOUS)
            #             graphDistMaxGraphPrev_2 = row2.getValue(emlib.MDL_GRAPH_DISTANCE2MAX_PREVIOUS)
            #             # print(maxCC_p_1, graphCCPrev_1, graphDistMaxGraphPrev_1)
            #             # print(maxCC_p_2, graphCCPrev_2, graphDistMaxGraphPrev_2)

            #             if ( maxCC_p_1 > maxCC_p_2 and  graphCCPrev_1 > graphCCPrev_2 ):
            #                 #deactivate within class2
            #                 mdParticles2.setValue(emlib.MDL_ENABLED, -1, objId_2)
            #                 # print("CLASS 1")
            #                 c1 += 1
            #             elif ( maxCC_p_2 > maxCC_p_1 and  graphCCPrev_2 > graphCCPrev_1 ):
            #                 #deactivate within class1
            #                 mdParticles1.setValue(emlib.MDL_ENABLED, -1, objId_1)
            #                 # print("CLASS 2")
            #                 c2 += 1
            #             elif ( graphDistMaxGraphPrev_1 < graphDistMaxGraphPrev_2 ):
            #                 #deactivate within class2
            #                 mdParticles2.setValue(emlib.MDL_ENABLED, -1, objId_2)
            #                 # print("CLASS 1")
            #                 c1 += 1
            #             else:
            #                 #deactivate within class1
            #                 mdParticles1.setValue(emlib.MDL_ENABLED, -1, objId_1)
            #                 # print("CLASS 2")
            #                 c2 += 1
            # print("double iteration finished")
            # fnAuxPart1_copy = self._getExtraPath('deact_partGraphValidated_Class_1.xmd')
            # fnAuxPart2_copy = self._getExtraPath('deact_partGraphValidated_Class_2.xmd')
            # mdParticles1.write(fnAuxPart1_copy)
            # mdParticles2.write(fnAuxPart2_copy)
            # print("totales c1 = %d, c2 = %d" % (c1, c2))


            # una mejor idea para la aplicación más adelante:
            # me copio las columas ordenadas según el ItemId/particleID
            # con los datos que voy a comparar y uso np.array o cualquier otra 
            # cosa de python para saber cuál de las columnas tiene 
            # lo mejor de los datos y la coloco en una columna al final
            # un identificador de la clase a la que va a pertener esa imagen
            # las demás las puede desactivar,
            # luego creo nuevos metadatas con las que quedaron activadas
            # hago una reconstrucción con highres (y su asignación)
            # para que sean los nuevos volumenes referencia para la siguiente iteración

         
        


    # --------------------------- STEPS functions -------------------------------

    def separateParticles(self):
            # get column values 
            print("Assigning particles to each class")                      
            fnAuxPart1 = self._getExtraPath('partGraphValidated_Class_%s.xmd' % str(1))
            self.runJob('xmipp_metadata_utilities','-i %s --operate sort particleId'%fnAuxPart1,numberOfMpi=1)
            mdParticles1 = emlib.MetaData(fnAuxPart1)
            maxCC_p_1 = np.array( mdParticles1.getColumnValues(emlib.MDL_MAXCC_PREVIOUS) )
            ccAssignedDir_1 = np.array( mdParticles1.getColumnValues(emlib.MDL_ASSIGNED_DIR_REF_CC) )
            graphCCPrev_1 = np.array( mdParticles1.getColumnValues(emlib.MDL_GRAPH_CC_PREVIOUS) )
            graphDistMaxGraphPrev_1 = np.array( mdParticles1.getColumnValues(emlib.MDL_GRAPH_DISTANCE2MAX_PREVIOUS) )
            
            fnAuxPart2 = self._getExtraPath('partGraphValidated_Class_%s.xmd' % str(2))
            self.runJob('xmipp_metadata_utilities','-i %s --operate sort particleId'%fnAuxPart2,numberOfMpi=1)
            mdParticles2 = emlib.MetaData(fnAuxPart2)
            maxCC_p_2 = np.array( mdParticles2.getColumnValues(emlib.MDL_MAXCC_PREVIOUS) )
            ccAssignedDir_2 = np.array( mdParticles1.getColumnValues(emlib.MDL_ASSIGNED_DIR_REF_CC) )
            graphCCPrev_2 = np.array( mdParticles2.getColumnValues(emlib.MDL_GRAPH_CC_PREVIOUS) )
            graphDistMaxGraphPrev_2 = np.array( mdParticles2.getColumnValues(emlib.MDL_GRAPH_DISTANCE2MAX_PREVIOUS) )
            
            # logic for 2 clasess -- True for class 1
            biggerCC = maxCC_p_1 > maxCC_p_2
            biggerCC_Dir = ccAssignedDir_1 > ccAssignedDir_2
            biggerGraphCC = graphCCPrev_1 > graphCCPrev_2
            smallerDistance = graphDistMaxGraphPrev_1 < graphDistMaxGraphPrev_2

            # enableClass1 = (biggerCC & biggerCC_Dir & biggerGraphCC) | smallerDistance
            enableClass1 = (biggerCC & biggerCC_Dir) | smallerDistance
            enableClass2 = ~(enableClass1)

            # translate that npArray to enable class 1
            # need to be sure that metadata is ordered by itemId before
            # print("\n===class 1")
            ii = 0
            for row in iterRows(mdParticles1):
                objId = row.getObjId()
                if ~enableClass1[ii]:
                    # print(enableClass1[ii], "\t desactivar %d"%objId)
                    mdParticles1.setValue(emlib.MDL_ENABLED, -1, objId)
                ii += 1

            ii = 0
            # print("\n===class 2")
            for row in iterRows(mdParticles2):
                objId = row.getObjId()
                if ~enableClass2[ii]:
                    # print(enableClass2[ii], "\t desactivar %d"%objId)
                    mdParticles2.setValue(emlib.MDL_ENABLED, -1, objId)
                ii += 1

            # print("double iteration finished")
            self.fnAuxPart1_copy = self._getExtraPath('deact_partGraphValidated_Class_1.xmd')
            self.fnAuxPart2_copy = self._getExtraPath('deact_partGraphValidated_Class_2.xmd')
            mdParticles1.write(self.fnAuxPart1_copy)
            mdParticles2.write(self.fnAuxPart2_copy)

    
    def _finishedProt(self, protocol):
        # Next schedule will be after this one
        self._runPrerequisites.append(protocol.getObjId())
        self.childs.append(protocol)
        if not self.finished:
            finishedIter = False
            while finishedIter == False:
                time.sleep(15)
                protocol = self._updateProtocol(protocol)
                if protocol.isFailed() or protocol.isAborted():
                    raise Exception('protocol: %s has failed' % protocol.getObjLabel())
                if protocol.isFinished():
                    finishedIter = True
                    self.classListProtocols.append(protocol) 
        print('protocol: %s has finished' % protocol.getObjLabel())
        sys.stdout.flush()
        time.sleep(15)
         

    def _updateProtocol(self, protocol):
        """ Retrieve the updated protocol
            """
        prot2 = getProtocolFromDb(protocol.getProject().path,
                                  protocol.getDbPath(),
                                  protocol.getObjId())

        # Close DB connections
        # prot2.getProject().closeMapper()
        prot2.closeMappers()
        return prot2

    def convertInputStep(self, iter):
        
        if iter == 0:
            outputParticles = self._createSetOfParticles()
            outputParticles.copyInfo(self.inputParticles.get())
            outputParticles.copyItems(self.inputParticles.get())
            self._defineOutputs(outputParticlesInit=outputParticles)
            self._store(outputParticles)
            # outputParticles.close()
    
            outputVolumes = Volume()
            outputVolumes.setFileName(self.inputVolume.get().getFileName())
            outputVolumes.setSamplingRate(self.inputVolume.get().getSamplingRate())
            self._defineOutputs(outputVolumesInit=outputVolumes)
            self._store(outputVolumes)
            # outputVolumes.close()
    
            # self.splitInputVol = outputVolumes
            # self.splitInputParts = outputParticles
            # self.signifInputParts = outputParticles

    def checkOutputsStep(self, newHighres, validationProt, iter):
        pass


    def createOutputStep(self, project):
        pass


    # --------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []
        return errors

    def _summary(self):
        summary = []
        return summary
