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
from pyworkflow import VERSION_2_0
from pyworkflow.protocol.params import (PointerParam, FloatParam, BooleanParam,
                                        IntParam, StringParam, LEVEL_ADVANCED,
                                        EnumParam, NumericListParam)
from pyworkflow.em.protocol import ProtMonitor
from pyworkflow.project import Manager
from pyworkflow.em.data import Volume
from pyworkflow.protocol import getProtocolFromDb
import pyworkflow.object as pwobj
from xmipp3.convert import readSetOfParticles
from os.path import exists, join
from shutil import copy
import xmippLib
import numpy as np
import math

from xmipp3.protocols import XmippProtReconstructHighRes


class XmippMetaProtGoldenHighRes(ProtMonitor):
    """ Metaprotocol to run golden version of highres"""
    _label = 'metaprotocol golden highres'
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
                      help='See http://xmipp.cnb.uam.es/twiki/bin/view/Xmipp/Symmetry for a description of the symmetry groups format'
                           'If no symmetry is present, give c1')
        form.addParam('maximumTargetResolution', NumericListParam,
                      label="Max. Target Resolution", default="15",
                      help="In Angstroms. The actual maximum resolution will be the maximum between this number of 0.5 * previousResolution, meaning that"
                           "in a single step you cannot increase the resolution more than 1/2")

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

        percentage = [1.15, 2.3, 3.45, 5.75, 9.2, 15, 24.14, 39.1]
        numIters = len(percentage)+2
        fnFSCs=open(self._getExtraPath('fnFSCs.txt'),'a')

        self.convertInputStep(percentage)

        targetResolution = self.maximumTargetResolution.get()

        #Global iterations
        for i in range(numIters):

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
                maximumTargetResolution = targetResolution
            )

            previousProtPart = self
            namePreviousParticles = 'outputParticles%s' % chr(65+i)
            newHighRes.inputParticles.set(previousProtPart)
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

            fnDir = newHighRes._getExtraPath("Iter%03d"%1)
            fnFSCs.write(join(fnDir,"fsc.xmd") + " \n")
            targetResolution = self.checkOutputsStep(newHighRes, i, False)

            if i>=7: #We are in the last three iterations
                #Check the output particles and remove all the disabled ones
                fnOutParticles = newHighRes._getPath('angles.xmd')
                params = '-i %s --query select "enabled==1"' % (fnOutParticles)
                self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)
                fnFinal = self._getExtraPath('inputLocalHighRes.xmd')
                if i==7:
                    copy(fnOutParticles, fnFinal)
                else:
                    params = ' -i %s --set union %s -o %s' % (fnFinal,fnOutParticles, fnFinal)
                    self.runJob("xmipp_metadata_utilities", params, numberOfMpi=1)

                    if i==9:
                        outputinputSetOfParticles = self._createSetOfParticles()
                        outputinputSetOfParticles.copyInfo(self.inputParticles.get())
                        readSetOfParticles(fnFinal, outputinputSetOfParticles)
                        self._defineOutputs(outputParticlesLocal=outputinputSetOfParticles)
                        self._store(outputinputSetOfParticles)
                        #self = self._updateProtocol(self)

        #Local iterations
        print("Target resolution - INPUT local 1: %f " % (float(targetResolution)))
        sys.stdout.flush()
        previousProtVol = newHighRes
        namePreviousVol = 'outputVolume'
        #call highres local with the new input set
        newHighResLocal = project.newProtocol(
            XmippProtReconstructHighRes,
            objLabel='HighRes - local %i' % 1,
            symmetryGroup=self.symmetryGroup.get(),
            numberOfIterations=1,
            particleRadius=self.particleRadius.get(),
            maximumTargetResolution=targetResolution,
            alignmentMethod = XmippProtReconstructHighRes.LOCAL_ALIGNMENT
        )
        newHighResLocal.inputParticles.set(self)
        newHighResLocal.inputParticles.setExtended('outputParticlesLocal')
        newHighResLocal.inputVolumes.set(previousProtVol)
        newHighResLocal.inputVolumes.setExtended(namePreviousVol)

        project.scheduleProtocol(newHighResLocal, self._runPrerequisites)
        # Next schedule will be after this one
        self._runPrerequisites.append(newHighResLocal.getObjId())
        self.childs.append(newHighResLocal)

        finishedIter = False
        while finishedIter == False:
            time.sleep(15)
            newHighResLocal = self._updateProtocol(newHighResLocal)
            if newHighResLocal.isFailed() or newHighResLocal.isAborted():
                raise Exception('XmippProtReconstructHighRes has failed')
            if newHighResLocal.isFinished():
                finishedIter = True

        fnDir = newHighResLocal._getExtraPath("Iter%03d" % 1)
        fnFSCs.write(join(fnDir, "fsc.xmd") + " \n")
        targetResolution = self.checkOutputsStep(newHighResLocal, numIters, True)
        print("Target resolution - OUT local 1: %f " % (float(targetResolution)))
        sys.stdout.flush()

        fnFSCs.close()
        self.createOutputStep(project)



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

    def convertInputStep(self, percentage):
        outputVolumes = Volume()
        outputVolumes.setFileName(self.inputVolumes.get().getFileName())
        outputVolumes.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(outputVolumesInit=outputVolumes)
        self._store(outputVolumes)

        #AJ to divide the input particles following Fibonacci percentages
        inputFullSet = self.inputParticles.get()
        input=[]
        for item in inputFullSet:
            input.append(item.getObjId())
        #input=range(1, len(inputFullSet)+1)
        self.subsets=[]
        for i, p in enumerate(percentage):
            self.subsets.append(self._createSetOfParticles(str(i)))
            self.subsets[i].copyInfo(inputFullSet)
            chosen = random.sample(xrange(len(input)), int((p/100.)*float(len(input))))
            for j in chosen:
                id = input[j]
                self.subsets[i].append(inputFullSet[id])
            result = {'outputParticles%s'%chr(65+i) : self.subsets[i]}
            self._defineOutputs(**result)
            self._store(self.subsets[i])
            input = [i for j, i in enumerate(input) if j not in chosen]

        # percentage = [1.15, 2.3, 3.45, 5.75, 9.2, 15, 24.14, 39.1]
        idx = len(percentage)
        self.subsets.append(self._createSetOfParticles(str(idx)))
        self.subsets[idx].copyInfo(inputFullSet)
        for item in self.subsets[3]: #5.75
            self.subsets[idx].append(item)
        for item in self.subsets[6]: #24.14
            self.subsets[idx].append(item)
        result = {'outputParticles%s' % chr(65 + idx): self.subsets[idx]}
        self._defineOutputs(**result)
        self._store(self.subsets[idx])

        idx = len(percentage)+1
        self.subsets.append(self._createSetOfParticles(str(idx)))
        self.subsets[idx].copyInfo(inputFullSet)
        for item in self.subsets[0]: #1.15
            self.subsets[idx].append(item)
        for item in self.subsets[1]: #2.3
            self.subsets[idx].append(item)
        for item in self.subsets[2]: #3.45
            self.subsets[idx].append(item)
        for item in self.subsets[4]: #9.2
            self.subsets[idx].append(item)
        for item in self.subsets[5]: #15
            self.subsets[idx].append(item)
        result = {'outputParticles%s' % chr(65 + idx): self.subsets[idx]}
        self._defineOutputs(**result)
        self._store(self.subsets[idx])



    def checkOutputsStep(self, newHighRes, iter, flagLocal):

        if iter>6:
            print("Checking the cc and shift conditions for the output particles")
            sys.stdout.flush()
            from pyworkflow.em.metadata.utils import iterRows

            fnOutParticles = newHighRes._getPath('angles.xmd')
            mdParticles = xmippLib.MetaData(fnOutParticles)

            #AJ cleaning with maxcc (distinguish if there are several sample populations)
            if flagLocal is False:
                ccList = mdParticles.getColumnValues(xmippLib.MDL_MAXCC)
            else:
                ccList = mdParticles.getColumnValues(xmippLib.MDL_COST)
            ccArray = np.asanyarray(ccList)
            ccArray = ccArray.reshape(-1, 1)

            #Gaussian mixture models to distinguish sample populations
            from sklearn import mixture
            clf1 = mixture.GMM(n_components=1)
            clf1.fit(ccArray)
            labels1 = clf1.fit_predict(ccArray)
            aic1 = clf1.aic(ccArray)
            bic1 = clf1.bic(ccArray)

            clf2 = mixture.GMM(n_components=2)
            clf2.fit(ccArray)
            labels2 = clf2.fit_predict(ccArray)
            aic2 = clf2.aic(ccArray)
            bic2 = clf2.bic(ccArray)

            print("Obtained results for gaussian mixture models:")
            print("aic1 = ", aic1)
            print("bic1 = ", bic1)
            print("mean1 = ", clf1.means_)
            print("cov1 = ", clf1.covars_)
            print(" ")
            print("aic2 = ", aic2)
            print("bic2 = ", bic2)
            print("mean2 = ", clf2.means_)
            print("cov2 = ", clf2.covars_)
            print("weights2 = ", clf2.weights_)
            print("labels2 = ", labels2)
            sys.stdout.flush()

            if aic2<aic1 and bic2<bic1:
                print("TENEMOS DOS POBLACIONES")
                sys.stdout.flush()
                # labels2 del fit_predict
                # i=0
                # for row in iterRows(mdParticles):
                #     objId = row.getObjId()
                #     if labels2[i] is 1:
                #         mdParticles.setValue(xmippLib.MDL_ENABLED, 0, objId)
                #     i=i+1
                # mdParticles.write(fnOutParticles)

                #test de hipotesis ????????
                p0 = clf2.weights_[0,0]/(clf2.weights_[0,0]+clf2.weights_[1,0])
                p1 = clf2.weights_[1,0]/(clf2.weights_[0,0]+clf2.weights_[1,0])
                mu0=clf2.means_[0,0]
                mu1=clf2.means_[1,0]
                std0 = clf2.covars_[0,0]
                std1 = clf2.covars_[1,0]
                termR = math.log(p0)-math.log(p1)-math.log(std0/std1)
                for row in iterRows(mdParticles):
                    objId = row.getObjId()
                    if flagLocal is False:
                        z = row.getValue(xmippLib.MDL_MAXCC)
                    else:
                        z = row.getValue(xmippLib.MDL_COST)
                    termL = (((z-mu1)**2)/(2*(std1**2))) + (((z-mu0)**2)/(2*(std0**2)))
                    if termL<termR:
                        mdParticles.setValue(xmippLib.MDL_ENABLED, 0, objId)
                mdParticles.write(fnOutParticles)


            #AJ cleaning with shift (removing particles with shift values higher than +-3sigma)
            #also cleaning with negative correlations
            shiftXList = mdParticles.getColumnValues(xmippLib.MDL_SHIFT_X)
            shiftYList = mdParticles.getColumnValues(xmippLib.MDL_SHIFT_Y)
            stdShiftX = np.std(np.asanyarray(shiftXList))
            stdShiftY = np.std(np.asanyarray(shiftYList))
            for row in iterRows(mdParticles):
                objId = row.getObjId()
                x = row.getValue(xmippLib.MDL_SHIFT_X)
                y = row.getValue(xmippLib.MDL_SHIFT_Y)
                cc = row.getValue(xmippLib.MDL_MAXCC)
                if x>4*stdShiftX or x<-4*stdShiftX or y>4*stdShiftY or y<-4*stdShiftY or cc<0: #poner 4 stds y 0 en cc
                    print("Shift or negative CC condition", objId)
                    sys.stdout.flush()
                    mdParticles.setValue(xmippLib.MDL_ENABLED, 0, objId)
            mdParticles.write(fnOutParticles)

        iteration = 1
        fn = newHighRes._getExtraPath("Iter%03d"%iteration)
        if exists(fn):
            resolution = 0.9*newHighRes.readInfoField(fn, "resolution", xmippLib.MDL_RESOLUTION_FREQREAL)
        else:
            raise Exception('XmippProtReconstructHighRes has not generated properly the output')
        return resolution


    def createOutputStep(self, project):

        pass
        # fnOutFile = self._getExtraPath('auxOutputFile.txt')
        # outFile = open(fnOutFile, 'a')
        # classListProtocolsEx = []
        # for i, item in enumerate(self.classListProtocols):
        #     if item not in classListProtocolsEx:
        #         classListProtocolsEx.append(item)
        #     outFile.write(str(item._objLabel) + "\n")
        #     outFile.write(str(self.classListIds[i]) + "\n")
        #     outFile.write(str(self.classListSizes[i]) + "\n")
        # outFile.close()
        #
        # outputMetaProt = project.newProtocol(
        #     XmippMetaProtCreateOutput,
        #     inputMetaProt=self,
        #     inputSignifProts=classListProtocolsEx)
        # project.scheduleProtocol(outputMetaProt, self._runPrerequisites)
        # # Next schedule will be after this one
        # self._runPrerequisites.append(outputMetaProt.getObjId())
        # return outputMetaProt



    #--------------------------- INFO functions --------------------------------
    def _validate(self):
        errors = []
        return errors
    
    def _summary(self):
        summary = []
        return summary


