# **************************************************************************
# *
# * Authors:     Amaya Jimenez (ajimenez@cnb.csic.es)
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

import sys, time

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, FloatParam, BooleanParam,
                                        IntParam, StringParam, LEVEL_ADVANCED)
from pyworkflow.em.protocol import ProtMonitor
from pyworkflow.project import Manager
from pyworkflow.em.data import Volume
from pyworkflow.protocol import getProtocolFromDb
import pyworkflow.object as pwobj

from xmipp3.protocols import XmippProtSplitVolumeHierarchical
from xmipp3.protocols import XmippProtReconstructHeterogeneous
from xmipp3.protocols import XmippMetaProtCreateOutput
from xmipp3.protocols import XmippMetaProtCreateSubset


class XmippMetaProtDiscreteHeterogeneityScheduler(ProtMonitor):
    """ Metaprotocol to run together all the protocols to discover discrete
    heterogeneity in a set of particles
     """
    _label = 'metaprotocol heterogeneity'
    _lastUpdateVersion = VERSION_2_0

    def __init__(self, **kwargs):
        ProtMonitor.__init__(self, **kwargs)
        self._runIds = pwobj.CsvList(pType=int)
        self.childs=[]

    def setAborted(self):
        #print("en el setaborted")
        #sys.out.flush()
        for child in self.childs:
            if child.isRunning() or child.isScheduled():
                #child.setAborted()
                self.getProject().stopProtocol(child)

        ProtMonitor.setAborted(self)


    def setFailed(self, errorMsg):
        #print("en el setfailed")
        #sys.out.flush()
        for child in self.childs:
            if child.isRunning() or child.isScheduled():
                #child.setFailed(errorMsg)
                self.getProject().stopProtocol(child)

        ProtMonitor.setFailed(self, errorMsg)

    #--------------------------- DEFINE param functions ------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
         
        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input volume",
                      help='Select the input volume.')
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles',
                      pointerCondition='hasAlignmentProj',
                      label="Input particles",
                      help='Select the input experimental images with an '
                           'angular assignment.')
        form.addParam('maxNumClasses', IntParam, default=4,
                      label='Maximum number of classes to calculate',
                      help='Maximum number of classes to calculate')
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
        form.addParam('maxShift', FloatParam, default=15,
                      expertLevel=LEVEL_ADVANCED,
                      label='Maximum shift',
                      help="In pixels")
        form.addParam('useGpu', BooleanParam, default=False, label="Use GPU")


        form.addSection(label='Split volume')

        form.addParam('directionalClasses', IntParam, default=2,
                      label='Number of directional classes',
                      help="By default only one class will be computed for "
                           "each projection direction. More classes could be"
                           "computed and this is needed for protocols "
                           "split-volume. ")
        form.addParam('homogeneize', IntParam, default=-1,
                      label='Homogeneize groups',
                      condition="directionalClasses==1",
                      help="Set to -1 for no homogeneization. Set to 0 for homogeneizing "
                           "to the minimum of class size. Set to any other number to "
                           "homogeneize to that particular number")
        form.addParam('cl2dIterations', IntParam, default=5,
                      expertLevel=LEVEL_ADVANCED,
                      condition="directionalClasses > 1",
                      label='Number of CL2D iterations')
        form.addParam('splitVolume', BooleanParam, label="Split volume",
                      condition="directionalClasses > 1", default=True,
                      help='If desired, the protocol can use the directional classes calculated in this protocol to divide the input volume '
                           'into 2 distinct 3D classes as measured by PCA. If the PCA component is just noise, it means that the algorithm '
                           'does not find a difference between the 2D classes')
        form.addParam('Niter', IntParam,
                      label="Number of iterations", default=5000,
                      condition="splitVolume",
                      expertLevel=LEVEL_ADVANCED,
                      help="Number of iterations to perform the volume splitting.")
        form.addParam('Nrec', IntParam,
                      label="Number of reconstructions", default=5,
                      condition="splitVolume",
                      expertLevel=LEVEL_ADVANCED,
                      help="Number of reconstructions to perform the hierarchical clustering.")
        form.addParam('angularSampling', FloatParam, default=5,
                      label='Angular sampling',
                      expertLevel=LEVEL_ADVANCED, help="In degrees")
        form.addParam('angularDistance', FloatParam, default=10,
                      expertLevel=LEVEL_ADVANCED,
                      label='Angular distance',
                      help="In degrees. An image belongs to a group if its "
                           "distance is smaller than this value")


        form.addSection(label='Reconstruct')

        form.addParam('numberOfIterations', IntParam, default=3,
                      label='Number of iterations')
        form.addParam('nextMask', PointerParam, label="Mask",
                      pointerClass='VolumeMask', allowsNull=True,
                      help='The mask values must be between 0 (remove these pixels) and 1 (let them pass). Smooth masks are recommended.')
        form.addParam('angularMaxShift', FloatParam, label="Max. shift (%)",
                      default=10,
                      help='Maximum shift as a percentage of the image size')
        line = form.addLine('Tilt angle:',
                            help='0 degrees represent top views, 90 degrees represent side views',
                            expertLevel=LEVEL_ADVANCED)
        line.addParam('angularMinTilt', FloatParam, label="Min.", default=0,
                      expertLevel=LEVEL_ADVANCED)
        line.addParam('angularMaxTilt', FloatParam, label="Max.", default=90,
                      expertLevel=LEVEL_ADVANCED)
        form.addParam('numberOfReplicates', IntParam,
                      label="Max. Number of Replicates", default=1,
                      expertLevel=LEVEL_ADVANCED,
                      help="Significant alignment is allowed to replicate each image up to this number of times")
        form.addParam("numberVotes", IntParam, label="Number of votes", default=3,
                      expertLevel=LEVEL_ADVANCED,
                      help="Number of votes for classification (maximum 5)")
        form.addParam('stochastic', BooleanParam, label="Stochastic",
                      default=False,
                      help="Stochastic optimization")
        form.addParam("stochasticAlpha", FloatParam, label="Relaxation factor",
                      default=0.1, condition="stochastic",
                      expertLevel=LEVEL_ADVANCED,
                      help="Relaxation factor (between 0 and 1). Set it closer to 0 if the random subset size is small")
        form.addParam("stochasticN", IntParam, label="Subset size", default=200,
                      condition="stochastic", expertLevel=LEVEL_ADVANCED,
                      help="Number of images in the random subset")

        form.addParallelSection(threads=1, mpi=8)
            
         
    #--------------------------- INSERT steps functions ------------------------
    def _insertAllSteps(self):
        self.classListProtocols=[]
        self.classListSizes=[]
        self.classListIds=[]
        self.finished=False
        self._insertFunctionStep('monitorStep')

    # --------------------------- STEPS functions ----------------------------
    def monitorStep(self):

        self._runPrerequisites = []
        manager = Manager()
        project = manager.loadProject(self.getProject().getName())

        self.numIter=self.maxNumClasses.get()
        for iter in range(1,self.numIter):

            if iter==1:
                self.convertInputStep()

                newSplitProt = project.newProtocol(
                    XmippProtSplitVolumeHierarchical,
                    objLabel='split volume hierarchical - iter %d'%iter,
                    symmetryGroup=self.symmetryGroup.get(),
                    angularSampling=self.angularSampling.get(),
                    angularDistance=self.angularDistance.get(),
                    maxShift=self.maxShift.get(),
                    directionalClasses=self.directionalClasses.get(),
                    homogeneize=self.homogeneize.get(),
                    targetResolution=self.targetResolution.get(),
                    cl2dIterations=self.cl2dIterations.get(),
                    splitVolume=self.splitVolume.get(),
                    Niter=self.Niter.get(),
                    Nrec=self.Nrec.get())

                previousSplitProt = self
                newSubsetNameSplitVol = 'outputVolumesInit'
                newSplitProt.inputVolume.set(previousSplitProt)
                newSplitProt.inputVolume.setExtended(newSubsetNameSplitVol)

                newSubsetNameSplitParts = 'outputParticlesInit'
                newSplitProt.inputParticles.set(previousSplitProt)
                newSplitProt.inputParticles.setExtended(newSubsetNameSplitParts)

                project.scheduleProtocol(newSplitProt, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(newSplitProt.getObjId())
                self.childs.append(newSplitProt)

                newSignificantProt = project.newProtocol(
                    XmippProtReconstructHeterogeneous,
                    objLabel='significant heterogeneity - iter %d' % iter,
                    symmetryGroup=self.symmetryGroup.get(),
                    particleRadius=self.particleRadius.get(),
                    targetResolution=self.targetResolution.get(),
                    useGpu=self.useGpu.get(),
                    numberOfIterations=self.numberOfIterations.get(),
                    nxtMask=self.nextMask.get(),
                    angularMinTilt=self.angularMinTilt.get(),
                    angularMaxTilt=self.angularMaxTilt.get(),
                    numberOfReplicates=self.numberOfReplicates.get(),
                    angularMaxShift=self.angularMaxShift.get(),
                    numberVotes=self.numberVotes.get(),
                    stochastic=self.stochastic.get(),
                    stochasticAlpha=self.stochasticAlpha.get(),
                    stochasticN=self.stochasticN.get())

                previousSignifProt = newSplitProt
                newSubsetNameSignifParts = 'outputParticlesInit'
                newSignificantProt.inputParticles.set(previousSplitProt)
                newSignificantProt.inputParticles.setExtended(newSubsetNameSignifParts)

                newSubsetNameSignifVol = 'outputVolumes'
                newSignificantProt.inputVolumes.set(previousSignifProt)
                newSignificantProt.inputVolumes.setExtended(newSubsetNameSignifVol)

                project.scheduleProtocol(newSignificantProt, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(newSignificantProt.getObjId())
                self.childs.append(newSignificantProt)


            elif iter>1 and not self.finished:

                newSplitProt = project.newProtocol(
                    XmippProtSplitVolumeHierarchical,
                    objLabel='split volume hierarchical - iter %d' % iter,
                    symmetryGroup=self.symmetryGroup.get(),
                    angularSampling=self.angularSampling.get(),
                    angularDistance=self.angularDistance.get(),
                    maxShift=self.maxShift.get(),
                    directionalClasses=self.directionalClasses.get(),
                    homogeneize=self.homogeneize.get(),
                    targetResolution=self.targetResolution.get(),
                    cl2dIterations=self.cl2dIterations.get(),
                    splitVolume=self.splitVolume.get(),
                    Niter=self.Niter.get(),
                    Nrec=self.Nrec.get())

                newSubsetNameSplitVol = 'outputAuxVolumes'
                newSplitProt.inputVolume.set(previousSplitProt)
                newSplitProt.inputVolume.setExtended(newSubsetNameSplitVol)

                newSubsetNameSplitParts = 'outputAuxParticles'
                newSplitProt.inputParticles.set(previousSplitProt)
                newSplitProt.inputParticles.setExtended(newSubsetNameSplitParts)

                project.scheduleProtocol(newSplitProt, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(newSplitProt.getObjId())
                self.childs.append(newSplitProt)

                newSignificantProt = project.newProtocol(
                    XmippProtReconstructHeterogeneous,
                    objLabel='significant heterogeneity - iter %d' % iter,
                    symmetryGroup=self.symmetryGroup.get(),
                    particleRadius=self.particleRadius.get(),
                    targetResolution=self.targetResolution.get(),
                    useGpu=self.useGpu.get(),
                    numberOfIterations=self.numberOfIterations.get(),
                    nxtMask=self.nextMask.get(),
                    angularMinTilt=self.angularMinTilt.get(),
                    angularMaxTilt=self.angularMaxTilt.get(),
                    numberOfReplicates=self.numberOfReplicates.get(),
                    angularMaxShift=self.angularMaxShift.get(),
                    numberVotes=self.numberVotes.get(),
                    stochastic=self.stochastic.get(),
                    stochasticAlpha=self.stochasticAlpha.get(),
                    stochasticN=self.stochasticN.get())

                previousSignifProt = newSplitProt
                newSubsetNameSignifParts = 'outputAuxParticles'
                newSignificantProt.inputParticles.set(previousSplitProt)
                newSignificantProt.inputParticles.setExtended(newSubsetNameSignifParts)

                newSubsetNameSignifVol = 'outputVolumes'
                newSignificantProt.inputVolumes.set(previousSignifProt)
                newSignificantProt.inputVolumes.setExtended(newSubsetNameSignifVol)

                project.scheduleProtocol(newSignificantProt, self._runPrerequisites)
                # Next schedule will be after this one
                self._runPrerequisites.append(newSignificantProt.getObjId())
                self.childs.append(newSignificantProt)

            if not self.finished:
                finishedIter = False
                while finishedIter == False:
                    #print("ESPERA 1 Iter", iter)
                    #sys.stdout.flush()
                    time.sleep(15)
                    newSplitProt = self._updateProtocol(newSplitProt)
                    newSignificantProt = self._updateProtocol(newSignificantProt)
                    if newSplitProt.isFailed() or newSplitProt.isAborted():
                        #print("XmippProtSplitVolumeHierarchical has failed", iter)
                        #sys.out.flush()
                        raise Exception('XmippProtSplitVolumeHierarchical has failed')
                    if newSignificantProt.isFailed() or newSignificantProt.isAborted():
                        #print("XmippProtReconstructHeterogeneous has failed", iter)
                        #sys.out.flush()
                        raise Exception('XmippProtReconstructHeterogeneous has failed')
                    if newSignificantProt.isFinished():
                        finishedIter = True
                        for classItem in newSignificantProt.outputClasses:
                            self.classListSizes.append(classItem.getSize())
                            self.classListIds.append(classItem.getObjId())
                            self.classListProtocols.append(newSignificantProt)

                finishedSubset = False
                finishedLast = False
                if iter != self.numIter - 1:
                    subsetProt = self.checkOutputsStep(project, iter)
                    while finishedSubset == False and not self.finished:
                        time.sleep(5)
                        subsetProt = self._updateProtocol(subsetProt)
                        if subsetProt.isFailed() or subsetProt.isAborted():
                            raise Exception('XmippMetaProtCreateSubset has failed')
                        if subsetProt.isFinished():
                            finishedSubset = True
                    previousSplitProt = subsetProt
                elif iter == self.numIter-1:
                    outputMetaProt = self.createOutputStep(project)
                    while finishedLast == False:
                        time.sleep(5)
                        outputMetaProt = self._updateProtocol(outputMetaProt)
                        if outputMetaProt.isFailed():
                            raise Exception('XmippMetaProtCreateOutput has failed')
                        if outputMetaProt.isFinished():
                            finishedLast = True

            if self.finished and iter == self.numIter-1:
                finishedLast = False
                outputMetaProt = self.createOutputStep(project)
                while finishedLast == False:
                    time.sleep(5)
                    outputMetaProt = self._updateProtocol(outputMetaProt)
                    if outputMetaProt.isFailed():
                        raise Exception('XmippMetaProtCreateOutput has failed')
                    if outputMetaProt.isFinished():
                        finishedLast = True

            if self.finished and iter < self.numIter - 1:
                continue



    #--------------------------- STEPS functions -------------------------------

    def _updateProtocol(self, protocol):
        """ Retrieve the updated protocol
            """
        prot2 = getProtocolFromDb(protocol.getProject().path,
                                  protocol.getDbPath(),
                                  protocol.getObjId())

        # Close DB connections
        #prot2.getProject().closeMapper()
        prot2.closeMappers()
        return prot2

    def convertInputStep(self):
        outputParticles = self._createSetOfParticles()
        outputParticles.copyInfo(self.inputParticles.get())
        outputParticles.copyItems(self.inputParticles.get())
        self._defineOutputs(outputParticlesInit=outputParticles)
        self._store(outputParticles)
        #outputParticles.close()

        outputVolumes = Volume()
        outputVolumes.setFileName(self.inputVolume.get().getFileName())
        outputVolumes.setSamplingRate(self.inputVolume.get().getSamplingRate())
        self._defineOutputs(outputVolumesInit=outputVolumes)
        self._store(outputVolumes)
        #outputVolumes.close()

        # self.splitInputVol = outputVolumes
        # self.splitInputParts = outputParticles
        # self.signifInputParts = outputParticles


    def checkOutputsStep(self, project, iter):

        maxSize = max(self.classListSizes)
        minMax = float((self.inputParticles.get().getSize()/len(self.classListSizes))*0.2)
        newSubsetProt = None

        if(maxSize<minMax or maxSize<100):
            self.finished=True

        if not self.finished:
            idx = self.classListSizes.index(maxSize)
            signifProt = self.classListProtocols[idx]
            idMaxSize = self.classListIds[idx]

            newSubsetProt = project.newProtocol(
                XmippMetaProtCreateSubset,
                objLabel='metaprotocol subset - iter %d' % iter,
                #inputSetOfVolumes=signifProt.outputVolumes,
                #inputSetOfClasses3D=signifProt.outputClasses,
                idx = idMaxSize
            )
            # nameVol = 'outputVolumes'
            # newSubsetProt.inputSetOfVolumes.set(signifProt)
            # newSubsetProt.inputSetOfVolumes.setExtended(nameVol)
            nameClasses = 'outputClasses'
            newSubsetProt.inputSetOfClasses3D.set(signifProt)
            newSubsetProt.inputSetOfClasses3D.setExtended(nameClasses)

            project.scheduleProtocol(newSubsetProt, self._runPrerequisites)
            # Next schedule will be after this one
            self._runPrerequisites.append(newSubsetProt.getObjId())
            self.childs.append(newSubsetProt)


            self.classListIds.pop(idx)
            self.classListProtocols.pop(idx)
            self.classListSizes.pop(idx)

        return newSubsetProt

    def createOutputStep(self, project):

        fnOutFile = self._getExtraPath('auxOutputFile.txt')
        outFile = open(fnOutFile, 'a')
        classListProtocolsEx = []
        for i, item in enumerate(self.classListProtocols):
            if item not in classListProtocolsEx:
                classListProtocolsEx.append(item)
            outFile.write(str(item._objLabel) + "\n")
            outFile.write(str(self.classListIds[i]) + "\n")
            outFile.write(str(self.classListSizes[i]) + "\n")
        outFile.close()

        outputMetaProt = project.newProtocol(
            XmippMetaProtCreateOutput,
            inputMetaProt=self,
            inputSignifProts=classListProtocolsEx)
        project.scheduleProtocol(outputMetaProt, self._runPrerequisites)
        # Next schedule will be after this one
        self._runPrerequisites.append(outputMetaProt.getObjId())
        return outputMetaProt






    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []
        return errors
    
    def _summary(self):
        summary = []
        return summary


