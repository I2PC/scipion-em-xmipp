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
import random
from os.path import exists, join
from shutil import copy
import numpy as np
import math

from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, FloatParam, BooleanParam,
                                        IntParam, StringParam, LEVEL_ADVANCED,
                                        EnumParam, NumericListParam, GPU_LIST,
                                        USE_GPU)
from pyworkflow import BETA, UPDATED, NEW, PROD

from pyworkflow.project import Manager
import pyworkflow.object as pwobj
from pyworkflow.protocol import getProtocolFromDb
from pwem.protocols import EMProtocol
from pwem.objects import Volume
from xmipp3.convert import readSetOfParticles
from pwem import emlib
from xmipp3.protocols import XmippProtReconstructHighRes


class XmippMetaProtGoldenHighRes(EMProtocol):
    """ Metaprotocol to run golden version of highres"""
    _label = 'metaprotocol golden highres'
    _lastUpdateVersion = VERSION_2_0
    _devStatus = BETA

    SPLIT_HT = 0
    SPLIT_OTSU = 1

    def __init__(self, **kwargs):
        EMProtocol.__init__(self, **kwargs)
        self._runIds = pwobj.CsvList(pType=int)
        self.childs=[]

    def setAborted(self):
        for child in self.childs:
            if child.isRunning() or child.isScheduled():
                #child.setAborted()
                self.getProject().stopProtocol(child)

        EMProtocol.setAborted(self)


    def setFailed(self, errorMsg):
        for child in self.childs:
            if child.isRunning() or child.isScheduled():
                #child.setFailed(errorMsg)
                self.getProject().stopProtocol(child)

        EMProtocol.setFailed(self, errorMsg)

    #--------------------------- DEFINE param functions ------------------------
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
        form.addParam('inputParticles', PointerParam, label="Full-size Images",
                      important=True,
                      pointerClass='SetOfParticles', allowsNull=True,
                      help='Select a set of images at full resolution')
        form.addParam('inputVolumes', PointerParam, label="Initial volumes",
                      important=True,
                      pointerClass='Volume, SetOfVolumes',
                      help='Select a set of volumes with 2 volumes or a single volume')
        form.addParam('particleRadius', IntParam, default=-1,
                      label='Radius of particle (px)',
                      help='This is the radius (in pixels) of the spherical mask covering the particle in the input images')
        form.addParam('symmetryGroup', StringParam, default="c1",
                      label='Symmetry group',
                      help='If no symmetry is present, give c1')
        form.addParam('minimumTargetResolution', FloatParam,
                      label="Initial resolution", default=15,
                      help="In Angstroms. The minimum resolution to be used in the first step of the protocol. "
                           "Then, the resolution will be automatically adjusted.")
        form.addParam('maximumTargetResolution', FloatParam,
                      label = "Maximum resolution", default = -1,
                      help = "In Angstroms. The maximum resolution to be used along the protocol. Set it to -1 for automatic adjustment.")
        form.addParam('discardParticles', BooleanParam, label="Discard particles?", default=False,
                      help="Discard particles when two distributions are estimated?")
        form.addParam('splitMethod', EnumParam, label='Split method', choices=['Hypothesis test','Otsu'], default=self.SPLIT_HT,
                      expertLevel=LEVEL_ADVANCED, help='When discard particles is allowed, choose the method between '
                      'hypothesis testing or Otsu thresholding', condition='discardParticles')
        form.addParam('adaptiveHT', BooleanParam, label="Adaptive threshold HT", default=False, condition="splitMethod==0",
                           help='Choose if you want to use an adaptive threshold for the hypothesis testing.')

        form.addSection(label='Angular assignment')
        form.addParam('maxShift', FloatParam, label="Max. shift (%)", default=10, expertLevel=LEVEL_ADVANCED,
                      help='Maximum shift as a percentage of the image size')
        line=form.addLine('Tilt angle:', help='0 degrees represent top views, 90 degrees represent side views', expertLevel=LEVEL_ADVANCED)
        line.addParam('angularMinTilt', FloatParam, label="Min.", default=0, expertLevel=LEVEL_ADVANCED,
                      help="Side views are around 90 degrees, top views around 0")
        line.addParam('angularMaxTilt', FloatParam, label="Max.", default=180, expertLevel=LEVEL_ADVANCED,
                      help="You may generate redudant galleries by setting this angle to 180, this may help if c1 symmetry is considered")

        form.addSection(label='Post-processing')
        form.addParam('postAdHocMask', PointerParam, label="Mask", pointerClass='VolumeMask', allowsNull=True,
                      help='The mask values must be between 0 (remove these pixels) and 1 (let them pass). Smooth masks are recommended.')
        groupSymmetry = form.addGroup('Symmetry', expertLevel=LEVEL_ADVANCED)
        groupSymmetry.addParam('postSymmetryWithinMask', BooleanParam, label="Symmetrize volume within mask?", default=False)
        groupSymmetry.addParam('postSymmetryWithinMaskType', StringParam, label="Mask symmetry", default="i1", condition="postSymmetryWithinMask",
                           help='If no symmetry is present, give c1')
        groupSymmetry.addParam('postSymmetryWithinMaskMask', PointerParam, label="Mask", pointerClass='VolumeMask', allowsNull=True, condition="postSymmetryWithinMask",
                               help='The mask values must be between 0 (remove these pixels) and 1 (let them pass). Smooth masks are recommended.')
        groupSymmetry.addParam('postSymmetryHelical', BooleanParam, label="Apply helical symmetry?", default=False)
        groupSymmetry.addParam('postSymmetryHelicalRadius', IntParam, label="Radius", default=-1, condition='postSymmetryHelical',
                               help="In Angstroms")
        groupSymmetry.addParam('postSymmetryHelicalDihedral', BooleanParam, label="Dihedral symmetry", default=False,
                               condition='postSymmetryHelical')
        groupSymmetry.addParam('postSymmetryHelicalMinRot', FloatParam, label="Min. Rotation", default=0, condition='postSymmetryHelical',
                               help="In degrees")
        groupSymmetry.addParam('postSymmetryHelicalMaxRot', FloatParam, label="Max. Rotation", default=360, condition='postSymmetryHelical',
                               help="In degrees")
        groupSymmetry.addParam('postSymmetryHelicalMinZ', FloatParam, label="Min. Z shift", default=0, condition='postSymmetryHelical',
                               help="In angstroms")
        groupSymmetry.addParam('postSymmetryHelicalMaxZ', FloatParam, label="Max. Z shift", default=40, condition='postSymmetryHelical',
                               help="In angstroms")
        form.addParam('postScript', StringParam, label="Post-processing command", default="", expertLevel=LEVEL_ADVANCED,
                      help='A command template that is used to post-process the reconstruction. The following variables can be used ' 
                           '%(sampling)s %(dim)s %(volume)s %(iterDir)s. The command should read Spider volumes and modify the input volume.'
                           'the command should be accessible either from the PATH or provide the absolute path.\n'
                           'Examples: \n'
                           'xmipp_transform_filter -i %(volume)s --fourier low_pass 15 --sampling %(sampling)s\n' 
                           '/home/joe/myScript %(volume)s sampling=%(sampling)s dim=%(dim)s')
        form.addParam('postSignificantDenoise', BooleanParam, label="Significant denoising Real space", expertLevel=LEVEL_ADVANCED, default=True)
        form.addParam('postFilterBank', BooleanParam, label="Significant denoising Fourier space", expertLevel=LEVEL_ADVANCED, default=True)
        form.addParam('postLaplacian', BooleanParam, label="Laplacian denoising", expertLevel=LEVEL_ADVANCED, default=True,
                      help="It can only be used if there is a mask")
        form.addParam('postDeconvolve', BooleanParam, label="Blind deconvolution", expertLevel=LEVEL_ADVANCED, default=True)
        form.addParam('postSoftNeg', BooleanParam, label="Attenuate undershooting", expertLevel=LEVEL_ADVANCED, default=True)
        form.addParam('postSoftNegK', FloatParam, label="Attenuate undershooting (K)", expertLevel=LEVEL_ADVANCED, default=9,
                      help="Values below avg-K*sigma are attenuated")
        form.addParam('postDifference', BooleanParam, label="Evaluate difference", expertLevel=LEVEL_ADVANCED, default=True)


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

        percentage = [1.14, 2.29, 3.44, 5.74, 9.19, 14.94, 24.13, 39.08]
        numGlobalIters = len(percentage)+2
        self.consecutiveBimodal=2
        self.listConsecutiveBimodal=[]

        targetResolution = self.minimumTargetResolution.get()

        #Global iterations
        for i in range(numGlobalIters):

            self.convertInputStep(percentage, i)

            print("Target resolution - group %s: %f " %(chr(65+i), float(targetResolution)))
            sys.stdout.flush()

            if i ==0:
                previousProtVol = self
                namePreviousVol = 'outputVolumesInit'
            else:
                previousProtVol = newHighRes
                namePreviousVol = 'outputVolume'

            newHighRes = project.newProtocol(
                XmippProtReconstructHighRes,
                objLabel='HighRes - group %s'%chr(65+i),
                symmetryGroup=self.symmetryGroup.get(),
                numberOfIterations=1,
                particleRadius = self.particleRadius.get(),
                maximumTargetResolution = targetResolution,
                alignmentMethod=XmippProtReconstructHighRes.GLOBAL_ALIGNMENT,
                angularMaxShift = self.maxShift.get(),
                angularMinTilt = self.angularMinTilt.get(),
                angularMaxTilt = self.angularMaxTilt.get(),
                postAdHocMask = self.postAdHocMask.get(),
                postSymmetryWithinMask = self.postSymmetryWithinMask.get(),
                postSymmetryWithinMaskType = self.postSymmetryWithinMaskType.get(),
                postSymmetryWithinMaskMask = self.postSymmetryWithinMaskMask.get(),
                postSymmetryHelical = self.postSymmetryHelical.get(),
                postSymmetryHelicalRadius = self.postSymmetryHelicalRadius.get(),
                postSymmetryHelicalDihedral = self.postSymmetryHelicalDihedral.get(),
                postSymmetryHelicalMinRot = self.postSymmetryHelicalMinRot.get(),
                postSymmetryHelicalMaxRot= self.postSymmetryHelicalMaxRot.get(),
                postSymmetryHelicalMinZ = self.postSymmetryHelicalMinZ.get(),
                postSymmetryHelicalMaxZ = self.postSymmetryHelicalMaxZ.get(),
                postScript = self.postScript.get(),
                postSignificantDenoise = self.postSignificantDenoise.get(),
                postFilterBank = self.postFilterBank.get(),
                postLaplacian = self.postLaplacian.get(),
                postDeconvolve = self.postDeconvolve.get(),
                postSoftNeg = self.postSoftNeg.get(),
                postSoftNegK = self.postSoftNegK.get(),
                postDifference = self.postDifference.get(),
                numberOfMpi=self.numberOfMpi.get(),
                useGpu=self.useGpu.get(),
                gpuList = self.gpuList.get()
            )

            previousProtPart = self
            namePreviousParticles = 'outputParticles%s' % chr(65+i)
            newHighRes.inputParticles.set(previousProtPart)
            newHighRes.inputParticles.setExtended(namePreviousParticles)
            newHighRes.inputVolumes.set(previousProtVol)
            newHighRes.inputVolumes.setExtended(namePreviousVol)

            project.scheduleProtocol(newHighRes)

            # Next schedule will be after this one
            self._runPrerequisites.append(newHighRes.getObjId())
            self.childs.append(newHighRes)

            finishedIter = False
            while finishedIter == False:
                time.sleep(15)
                newHighRes = self._updateProtocol(newHighRes)
                if newHighRes.isFailed() or newHighRes.isAborted():
                    raise Exception('XmippProtReconstructHighRes has failed')
                if newHighRes.isFinished():
                    finishedIter = True

            fnDir = newHighRes._getExtraPath("Iter%03d"%1)
            fnFSCs = open(self._getExtraPath('fnFSCs.txt'), 'a')
            fnFSCs.write(join(fnDir,"fsc.xmd") + " \n")
            fnFSCs.close()
            targetResolution = self.checkOutputsStep(newHighRes, i, False)
            targetResolution = max(targetResolution, self.maximumTargetResolution.get())


            if i>=7: #We are in the last three iterations
                #Check the output particles and remove all the disabled ones
                fnOutParticles = newHighRes._getPath('angles.xmd')
                params = '-i %s --query select "enabled==1"' % (fnOutParticles)
                self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)
                fnFinal = self._getExtraPath('inputLocalHighRes1.xmd')
                if i==7:
                    copy(fnOutParticles, fnFinal)
                else:
                    params = ' -i %s --set union %s -o %s' % (fnFinal,fnOutParticles, fnFinal)
                    self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)

                    if i==9:
                        outputinputSetOfParticles = self._createSetOfParticles()
                        outputinputSetOfParticles.copyInfo(self.inputParticles.get())
                        readSetOfParticles(fnFinal, outputinputSetOfParticles)
                        self._defineOutputs(outputParticlesLocal1=outputinputSetOfParticles)
                        self._store(outputinputSetOfParticles)

        #Local iterations
        numLocalIters = 5
        for i in range(numLocalIters):

            if i>2:
                minPrevRes = prevTargetResolution
                if targetResolution>minPrevRes:
                    print("Target resolution is stuck")
                    sys.stdout.flush()
                    break

            prevTargetResolution = targetResolution

            print("Target resolution - INPUT local %d: %f " % ((i+1), float(targetResolution)))
            sys.stdout.flush()
            previousProtVol = newHighRes
            namePreviousVol = 'outputVolume'
            #calling highres local with the new input set
            newHighRes = project.newProtocol(
                XmippProtReconstructHighRes,
                objLabel='HighRes - local %d' % (i+1),
                symmetryGroup=self.symmetryGroup.get(),
                numberOfIterations=1,
                particleRadius=self.particleRadius.get(),
                maximumTargetResolution=targetResolution,
                alignmentMethod=XmippProtReconstructHighRes.LOCAL_ALIGNMENT,
                angularMaxShift=self.maxShift.get(),
                angularMinTilt=self.angularMinTilt.get(),
                angularMaxTilt=self.angularMaxTilt.get(),
                postAdHocMask=self.postAdHocMask.get(),
                postSymmetryWithinMask=self.postSymmetryWithinMask.get(),
                postSymmetryWithinMaskType=self.postSymmetryWithinMaskType.get(),
                postSymmetryWithinMaskMask=self.postSymmetryWithinMaskMask.get(),
                postSymmetryHelical=self.postSymmetryHelical.get(),
                postSymmetryHelicalRadius=self.postSymmetryHelicalRadius.get(),
                postSymmetryHelicalDihedral=self.postSymmetryHelicalDihedral.get(),
                postSymmetryHelicalMinRot=self.postSymmetryHelicalMinRot.get(),
                postSymmetryHelicalMaxRot=self.postSymmetryHelicalMaxRot.get(),
                postSymmetryHelicalMinZ=self.postSymmetryHelicalMinZ.get(),
                postSymmetryHelicalMaxZ=self.postSymmetryHelicalMaxZ.get(),
                postScript=self.postScript.get(),
                postSignificantDenoise=self.postSignificantDenoise.get(),
                postFilterBank=self.postFilterBank.get(),
                postLaplacian=self.postLaplacian.get(),
                postDeconvolve=self.postDeconvolve.get(),
                postSoftNeg=self.postSoftNeg.get(),
                postSoftNegK=self.postSoftNegK.get(),
                postDifference=self.postDifference.get(),
                numberOfMpi=self.numberOfMpi.get(),
                useGpu=self.useGpu.get(),
                gpuList=self.gpuList.get()
            )
            newHighRes.inputParticles.set(self)
            namePreviousParticles = 'outputParticlesLocal%d' % (i+1)
            newHighRes.inputParticles.setExtended(namePreviousParticles)
            newHighRes.inputVolumes.set(previousProtVol)
            newHighRes.inputVolumes.setExtended(namePreviousVol)

            project.scheduleProtocol(newHighRes, self._runPrerequisites)
            # Next schedule will be after this one
            self._runPrerequisites.append(newHighRes.getObjId())
            self.childs.append(newHighRes)

            finishedIter = False
            while finishedIter == False:
                time.sleep(15)
                newHighRes = self._updateProtocol(newHighRes)
                if newHighRes.isFailed() or newHighRes.isAborted():
                    raise Exception('XmippProtReconstructHighRes has failed')
                if newHighRes.isFinished():
                    finishedIter = True

            fnDir = newHighRes._getExtraPath("Iter%03d" % 1)
            fnFSCs = open(self._getExtraPath('fnFSCs.txt'), 'a')
            fnFSCs.write(join(fnDir,"fsc.xmd") + " \n")
            fnFSCs.close()
            targetResolution = self.checkOutputsStep(newHighRes, numGlobalIters+i, True)
            targetResolution = max(targetResolution, self.maximumTargetResolution.get())

            #Check the output particles and remove all the disabled ones
            fnOutParticles = newHighRes._getPath('angles.xmd')
            params = '-i %s --query select "enabled==1"' % (fnOutParticles)
            self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)
            fnFinal = self._getExtraPath('inputLocalHighRes%d.xmd'%(i+2))
            copy(fnOutParticles, fnFinal)

            if i>1:
                #including the number of particles as stoppping criteria
                mdFinal = emlib.MetaData(fnFinal)
                Nfinal = mdFinal.size()
                Ninit = self.inputParticles.get().getSize()
                if Nfinal<Ninit*0.3:
                    print("Image set size too small", Nfinal, Ninit)
                    sys.stdout.flush()
                    break


            outputinputSetOfParticles = self._createSetOfParticles()
            outputinputSetOfParticles.copyInfo(self.inputParticles.get())
            readSetOfParticles(fnFinal, outputinputSetOfParticles)
            result = {'outputParticlesLocal%d' % (i+2): outputinputSetOfParticles}
            self._defineOutputs(**result)
            self._store(outputinputSetOfParticles)
            #self = self._updateProtocol(self)

        self.createOutputStep(project)



    #--------------------------- STEPS functions -------------------------------

    def _updateProtocol(self, protocol):
        """ Retrieve the updated protocol
            """
        prot2 = getProtocolFromDb(protocol.getProject().path,
                                  protocol.getDbPath(),
                                  protocol.getObjId())

        # Close DB connections
        prot2.closeMappers()
        return prot2

    def convertInputStep(self, percentage, i):

        if i==0:
            outputVolumes = Volume()
            outputVolumes.setFileName(self.inputVolumes.get().getFileName())
            outputVolumes.setSamplingRate(self.inputVolumes.get().getSamplingRate())
            self._defineOutputs(outputVolumesInit=outputVolumes)
            self._store(outputVolumes)

        #to divide the input particles following Fibonacci percentages
        inputFullSet = self.inputParticles.get()
        if i==0:
            self.subsets = []
            self.input = []
            for item in inputFullSet:
                self.input.append(item.getObjId())
            self.origLenInput = len(self.input)
            self.pp=[]
            for p in percentage:
                self.pp.append(int((p/100.)*float(self.origLenInput)))

        if i<len(percentage):
            p=self.pp[i]
            self.subsets.append(self._createSetOfParticles(str(i)))
            self.subsets[i].copyInfo(inputFullSet)
            chosen = random.sample(range(len(self.input)), p)
            for j in chosen:
                id = self.input[j]
                self.subsets[i].append(inputFullSet[id])
            result = {'outputParticles%s'%chr(65+i) : self.subsets[i]}
            self._defineOutputs(**result)
            self._store(self.subsets[i])
            self.input = [i for j, i in enumerate(self.input) if j not in chosen]

        if i == len(percentage):
            self.subsets.append(self._createSetOfParticles(str(i)))
            self.subsets[i].copyInfo(inputFullSet)
            for item in self.subsets[3]: #5.75
                self.subsets[i].append(item)
            for item in self.subsets[6]: #24.14
                self.subsets[i].append(item)
            result = {'outputParticles%s' % chr(65 + i): self.subsets[i]}
            self._defineOutputs(**result)
            self._store(self.subsets[i])

        if i == len(percentage)+1:
            self.subsets.append(self._createSetOfParticles(str(i)))
            self.subsets[i].copyInfo(inputFullSet)
            for item in self.subsets[0]: #1.15
                self.subsets[i].append(item)
            for item in self.subsets[1]: #2.3
                self.subsets[i].append(item)
            for item in self.subsets[2]: #3.45
                self.subsets[i].append(item)
            for item in self.subsets[4]: #9.2
                self.subsets[i].append(item)
            for item in self.subsets[5]: #15
                self.subsets[i].append(item)
            result = {'outputParticles%s' % chr(65 + i): self.subsets[i]}
            self._defineOutputs(**result)
            self._store(self.subsets[i])

    def otsu(self, ccList):
        cc_number = len(ccList)
        mean_weigth = 1.0 / cc_number
        his, bins = np.histogram(ccList, int((max(ccList)-min(ccList))/0.01))
        final_thresh = -1
        final_value = -1
        cc_arr = np.arange(min(ccList),max(ccList),0.01)
        if len(cc_arr)==(len(his)+1):
            cc_arr=cc_arr[:-1]
        for t, j in enumerate(bins[1:-1]):
            idx = t+1
            pcb = np.sum(his[:idx])
            pcf = np.sum(his[idx:])
            Wb = pcb * mean_weigth
            Wf = pcf * mean_weigth

            mub = np.sum(cc_arr[:idx] * his[:idx]) / float(pcb)
            muf = np.sum(cc_arr[idx:] * his[idx:]) / float(pcf)
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = bins[idx]
                final_value = value

        return final_thresh

    def checkOutputsStep(self, newHighRes, iter, flagLocal):

        if iter>6 and self.discardParticles.get() is True:
            from pwem.emlib.metadata import iterRows

            fnOutParticles = newHighRes._getPath('angles.xmd')
            mdParticles = emlib.MetaData(fnOutParticles)

            #cleaning with maxcc (to distinguish if there are several sample populations)
            if flagLocal is False:
                ccList = mdParticles.getColumnValues(emlib.MDL_MAXCC)
            else:
                ccList = mdParticles.getColumnValues(emlib.MDL_COST)
            minCC = min(ccList)
            ccArray = np.asanyarray(ccList)
            ccArray = ccArray.reshape(-1, 1)

            #Gaussian mixture models to distinguish sample populations
            from sklearn import mixture
            clf1 = mixture.GaussianMixture(n_components=1)
            clf1.fit(ccArray)
            aic1 = clf1.aic(ccArray)
            bic1 = clf1.bic(ccArray)

            clf2 = mixture.GaussianMixture(n_components=2)
            clf2.fit(ccArray)
            aic2 = clf2.aic(ccArray)
            bic2 = clf2.bic(ccArray)

            # print("Obtained results for gaussian mixture models:")
            # print("aic1 = ", aic1)
            # print("bic1 = ", bic1)
            # print("mean1 = ", clf1.means_)
            # print("cov1 = ", clf1.covars_)
            # print(" ")
            # print("aic2 = ", aic2)
            # print("bic2 = ", bic2)
            # print("mean2 = ", clf2.means_)
            # print("cov2 = ", clf2.covars_)
            # print("weights2 = ", clf2.weights_)
            # sys.stdout.flush()

            if aic2<aic1 and bic2<bic1:

                if self.splitMethod == self.SPLIT_OTSU:
                    #Otsu thresholding
                    thOtsu=self.otsu(ccList)
                    for row in iterRows(mdParticles):
                        objId = row.getObjId()
                        if flagLocal is False:
                            z = row.getValue(emlib.MDL_MAXCC)
                        else:
                            z = row.getValue(emlib.MDL_COST)
                        if z<thOtsu:
                            mdParticles.setValue(emlib.MDL_ENABLED, 0, objId)
                    mdParticles.write(fnOutParticles)

                if self.splitMethod == self.SPLIT_HT:
                    #Hypothesis testing

                    #This is a way to make an adaptive hypothesis testing
                    #if several times we are obtaining different distributions
                    #then, we move the threshold to higher values, easier to discard particles
                    if self.adaptiveHT:
                        self.listConsecutiveBimodal.append(True)
                        if len(self.listConsecutiveBimodal)>1:
                            if self.listConsecutiveBimodal[-1] is True and self.listConsecutiveBimodal[-2] is True:
                                self.consecutiveBimodal = self.consecutiveBimodal+1
                            else:
                                self.consecutiveBimodal = self.consecutiveBimodal-1
                        if self.consecutiveBimodal<2:
                            self.consecutiveBimodal=2


                    if clf2.means_[0,0]<clf2.means_[1,0]:
                        idx0 = 0
                        idx1 = 1
                    else:
                        idx0 = 1
                        idx1 = 0
                    p0 = clf2.weights_[idx0]/(clf2.weights_[idx0]+clf2.weights_[idx1])
                    p1 = clf2.weights_[idx1]/(clf2.weights_[idx0]+clf2.weights_[idx1])
                    mu0=clf2.means_[idx0,0]
                    mu1=clf2.means_[idx1,0]
                    var0 = clf2.covariances_[idx0,0]
                    var1 = clf2.covariances_[idx1,0]
                    std0 = math.sqrt(var0)
                    std1 = math.sqrt(var1)
                    termR = math.log(p0)-math.log(p1)-math.log(std0/std1)
                    # Moving the threshold to higher values, easier to discard particles
                    if self.adaptiveHT:
                        # adaptive term
                        termR = termR +(self.consecutiveBimodal*(mu1-minCC))
                    else:
                        # fix term
                        termR = termR + (2 * (mu1 - minCC))
                    for row in iterRows(mdParticles):
                        objId = row.getObjId()
                        if flagLocal is False:
                            z = row.getValue(emlib.MDL_MAXCC)
                        else:
                            z = row.getValue(emlib.MDL_COST)
                        termL = (-((z-mu1)**2)/(2*(var1))) + (((z-mu0)**2)/(2*(var0)))
                        if termL<termR:
                            mdParticles.setValue(emlib.MDL_ENABLED, 0, objId)
                    mdParticles.write(fnOutParticles)

            else:
                if self.adaptiveHT:
                    self.listConsecutiveBimodal.append(False)


            #AJ cleaning with shift (removing particles with shift values higher than +-4sigma)
            #also cleaning with negative correlations
            shiftXList = mdParticles.getColumnValues(emlib.MDL_SHIFT_X)
            shiftYList = mdParticles.getColumnValues(emlib.MDL_SHIFT_Y)
            stdShiftX = np.std(np.asanyarray(shiftXList))
            stdShiftY = np.std(np.asanyarray(shiftYList))
            for row in iterRows(mdParticles):
                objId = row.getObjId()
                x = row.getValue(emlib.MDL_SHIFT_X)
                y = row.getValue(emlib.MDL_SHIFT_Y)
                if iter<9:
                    cc = row.getValue(emlib.MDL_MAXCC)
                else:
                    cc = 1.0
                if x>4*stdShiftX or x<-4*stdShiftX or y>4*stdShiftY or y<-4*stdShiftY or cc<0.0:
                    mdParticles.setValue(emlib.MDL_ENABLED, 0, objId)
            mdParticles.write(fnOutParticles)

        iteration = 1
        fn = newHighRes._getExtraPath("Iter%03d"%iteration)
        if exists(fn):
            resolution = 0.9*newHighRes.readInfoField(fn, "resolution", emlib.MDL_RESOLUTION_FREQREAL)
        else:
            raise Exception('XmippProtReconstructHighRes has not generated properly the output')
        return resolution


    def createOutputStep(self, project):
        pass



    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []
        if not self.inputParticles.hasValue():
            errors.append("You must provide input particles")
        if not self.inputParticles.get().isPhaseFlipped():
            errors.append("The input particles must be phase flipped")
        return errors
    
    def _summary(self):
        summary = []
        return summary



