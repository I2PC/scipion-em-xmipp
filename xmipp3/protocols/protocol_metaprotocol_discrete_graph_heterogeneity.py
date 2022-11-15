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
from os.path import exists

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

from xmipp3.protocols import XmippMetaProtCreateOutput
from xmipp3.protocols import XmippMetaProtCreateSubset

from xmipp3.protocols import XmippProtAngularGraphConsistency
from xmipp3.protocols import XmippProtReconstructHighRes

from pwem.emlib.metadata import iterRows
from xmipp3.convert import readSetOfParticles, readSetOfImages
from pwem import emlib

class XmippMetaProtDiscreteGraphHeterogeneity(EMProtocol):
    """ Metaprotocol to run together some protocols in order to deal with heterogeneity
        in a dataset, using low resolution validation based on graph analysis
     """
    _label = 'metaprotocol heterogeneity - graph validation'
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
        
        for classItem in relionClassify3DProt.outputClasses:
            
            classVar = classItem.getObjId() 
            print('classItem-ObjId',classItem.getObjId())  
            
            for iter in range(1,numIter):
                
                if iter == 1:
                    previousProtPart = self
                    partName = 'outputParticlesInit'
                    previousProtVol = relionClassify3DProt
                    volumeName = 'outputVolumes.%d'%classItem.getObjId()
                else:
                    previousProtPart = highresNoAlignProt
                    partName = 'outputParticles'
                    previousProtVol = highresNoAlignProt
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
        
                # angular graph validation
                print('angular graph validation - iter',iter,'class',classVar)
                sys.stdout.flush()
                
                validationProt = project.newProtocol(
                    XmippProtAngularGraphConsistency,
                    objLabel='graphValidation - iter %d' % iter,
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
                # self._finishedProt(validationProt)
                
                # angular assignment from newHigresProt
                print('angular assignment from Highres - iter',iter,'class',classVar)
                sys.stdout.flush()
                
                angAssignProt = project.newProtocol(
                    ProtAlignmentAssign,
                    objLabel='assignment from highres',
                    )
                previousProtPart = validationProt
                partName = 'outputParticlesAux'
                angAssignProt.inputParticles.set(validationProt)
                angAssignProt.inputParticles.setExtended(partName)
                   
                angAssignProt.inputAlignment.set(newHighres)
                angAssignProt.inputAlignment.setExtended('outputParticles')
                
                project.scheduleProtocol(angAssignProt, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(angAssignProt.getObjId())
                self.childs.append(angAssignProt)    
                # self._finishedProt(angAssignProt)
                            
                # reconstruction of new volume for reference using only correctly assigned particles
                print('Highres reconstruction No-Alignment - iter',iter,'class',classVar)
                sys.stdout.flush()
                
                highresNoAlignProt = project.newProtocol(
                        XmippProtReconstructHighRes,
                        objLabel='xmipp Highres - No alignment',
                        symmetryGroup=self.symmetryGroup.get(),
                        numberOfIterations=1, 
                        particleRadius = self.particleRadius.get(),
                        maximumTargetResolution = self.targetResolution.get(),
                        numberOfMpi=self.numberOfMpi.get(),
                        alignmentMethod=XmippProtReconstructHighRes.NO_ALIGNMENT,
                        useGpu=self.useGpu.get(),
                        gpuList = self.gpuList.get()
                        )    
                previousProtPart = angAssignProt
                highresNoAlignProt.inputParticles.set(previousProtPart)
                highresNoAlignProt.inputParticles.setExtended('outputParticles') # is important this name to match
                 
                highresNoAlignProt.inputVolumes.set(previousProtVol)
                highresNoAlignProt.inputVolumes.setExtended(volumeName) # is important this name to match
                
                project.scheduleProtocol(highresNoAlignProt, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(angAssignProt.getObjId())
                self.childs.append(angAssignProt)     
                self._finishedProt(highresNoAlignProt)                
                #clear prerequisites after 1st iter
                self._runPrerequisites.clear()
        
        
#         for iter in range(1, numIter):
#             print('iter: ',iter)
#             
#             # se debe actualizar las partículas que van a pasar a cada iteración
#             # para iter 1 particles es self
#             # para iter>1 particles es el subset  de newhighres que cumple con criterios de calidad
#             
#             # en cuanto al volumen
#             # en iter == 1
#             # el volumen es relionClassify3DProt
#             # en iter > 1
#             # el volumen es la reconstruccion que se haga con las partículas buenas 
#             # que salen de la validación con graph consistence
#             # para este volumen se puede usar la asignación para esas partículas hecha por newhigres
#                   
#             for classItem in relionClassify3DProt.outputClasses:
#                 classVar = classItem.getObjId()
#                 print('classItem-ObjId',classItem.getObjId())
#                 # highres with all particles
#                 print('Highres (one class) - iter', iter,'class',classVar)
#                 sys.stdout.flush()
#                 
#                 newHighres = project.newProtocol(
#                         XmippProtReconstructHighRes,
#                         objLabel='xmipp Highres - iter %d class %d' % (iter,classVar),
#                         symmetryGroup=self.symmetryGroup.get(),
#                         numberOfIterations=1, 
#                         particleRadius = self.particleRadius.get(),
#                         maximumTargetResolution = self.targetResolution.get(),
#                         numberOfMpi=self.numberOfMpi.get(),
#                         alignmentMethod=XmippProtReconstructHighRes.GLOBAL_ALIGNMENT,
#                         useGpu=self.useGpu.get(),
#                         gpuList = self.gpuList.get()
#                         )    
#                 previousProtPart = self
#                 newHighres.inputParticles.set(previousProtPart)
#                 newHighres.inputParticles.setExtended('outputParticlesInit') # is important this name to match
#                  
#                 previousProtVol = relionClassify3DProt
#                 newHighres.inputVolumes.set(previousProtVol)
#                 volumeName = 'outputVolumes.%d'%classItem.getObjId()
#                 newHighres.inputVolumes.setExtended(volumeName) # is important this name to match
#                 
#                 project.scheduleProtocol(newHighres, self._runPrerequisites)                               
#                 self._finishedProt(newHighres)
#         
#                 # angular graph validation
#                 print('angular graph validation - iter',iter,'class',classVar)
#                 sys.stdout.flush()
#                 
#                 validationProt = project.newProtocol(
#                     XmippProtAngularGraphConsistency,
#                     objLabel='graphValidation - iter %d' % iter,
#                     symmetryGroup=self.symmetryGroup.get(),
#                     maximumTargetResolution=10,
#                     numberOfMpi=self.numberOfMpi.get()
#                     )
#                 previousProtPart = newHighres
#                 validationProt.inputParticles.set(previousProtPart)
#                 validationProt.inputParticles.setExtended('outputParticles') # todo no es así !!
#                 
#                 previousProtVol = relionClassify3DProt
#                 validationProt.inputVolume.set(previousProtVol)
#                 volumeName = 'outputVolume.%d'%classItem.getObjId()
#                 validationProt.inputVolume.setExtended(volumeName) 
#                 
#                 project.scheduleProtocol(validationProt, self._runPrerequisites) 
#                 self._finishedProt(validationProt)
#                 
#                 # angular assignment from newHigresProt
#                 print('angular assignment from Highres - iter',iter,'class',classVar)
#                 sys.stdout.flush()
#                 
#                 angAssignProt = project.newProtocol(
#                     ProtAlignmentAssign,
#                     objLabel='assignment from highres',
#                     )
#                 previousProtPart = validationProt
#                 partName = 'outputParticlesAux'
#                 angAssignProt.inputParticles.set(validationProt)
#                 angAssignProt.inputParticles.setExtended(partName)
#                    
#                 angAssignProt.inputAlignment.set(newHighres)
#                 angAssignProt.inputAlignment.setExtended('outputParticles')
#                 
#                 project.scheduleProtocol(angAssignProt, self._runPrerequisites) 
#                 self._finishedProt(angAssignProt)
#                             
#                 # reconstruction of new volume for reference using only correctly assigned particles
#                 print('Highres reconstruction No-Alignment - iter',iter,'class',classVar)
#                 sys.stdout.flush()
#                 
#                 highresNoAlignProt = project.newProtocol(
#                         XmippProtReconstructHighRes,
#                         objLabel='xmipp Highres - No alignment',
#                         symmetryGroup=self.symmetryGroup.get(),
#                         numberOfIterations=1, 
#                         particleRadius = self.particleRadius.get(),
#                         maximumTargetResolution = self.targetResolution.get(),
#                         numberOfMpi=self.numberOfMpi.get(),
#                         alignmentMethod=XmippProtReconstructHighRes.NO_ALIGNMENT,
#                         useGpu=self.useGpu.get(),
#                         gpuList = self.gpuList.get()
#                         )    
#                 previousProtPart = angAssignProt
#                 highresNoAlignProt.inputParticles.set(previousProtPart)
#                 highresNoAlignProt.inputParticles.setExtended('outputParticles') # is important this name to match
#                  
#                 previousProtVol = relionClassify3DProt
#                 highresNoAlignProt.inputVolumes.set(previousProtVol)
#                 volumeName = 'outputVolumes.%d'%classItem.getObjId()
#                 highresNoAlignProt.inputVolumes.setExtended(volumeName) # is important this name to match
#                 
#                 project.scheduleProtocol(highresNoAlignProt, self._runPrerequisites) 
#                 self._finishedProt(highresNoAlignProt)
                    
    # --------------------------- STEPS functions -------------------------------
    
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

