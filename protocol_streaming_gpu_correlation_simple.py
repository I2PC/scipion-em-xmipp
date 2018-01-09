# ******************************************************************************
# *
# * Authors:     Amaya Jimenez Moreno (ajimenez@cnb.csic.es)
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
# ******************************************************************************

from pyworkflow.em import ALIGN_NONE, SetOfParticles, SetOfClasses2D, ALIGN_2D
from pyworkflow.em.protocol import ProtAlign2D
import pyworkflow.em.metadata as md
import pyworkflow.protocol.params as params
from pyworkflow.em.metadata.utils import iterRows, getSize
from xmipp import MD_APPEND
from pyworkflow.em.packages.xmipp3.convert import writeSetOfParticles, \
    xmippToLocation, rowToAlignment, writeSetOfClasses2D, readSetOfImages, \
    rowToImage, readSetOfParticles
from shutil import copy
from os.path import join, exists, getmtime
from os import mkdir, remove
from datetime import datetime
from pyworkflow.utils import prettyTime, cleanPath
from pyworkflow.object import Set
from pyworkflow.protocol.constants import STATUS_NEW
import sys
from pyworkflow.em.metadata.classes import Row
from time import time
from math import floor
import pyworkflow.utils as pwutils


class XmippProtStrGpuCrrSimple(ProtAlign2D):
    """ Aligns a set of particles in streaming using the GPU Correlation algorithm. """
    _label = 'align with GPU Correlation in streaming'


    # --------------------------- DEFINE param functions -----------------------
    def _defineAlignParams(self, form):
        form.addParam('inputClasses', params.PointerParam,
                      pointerClass='SetOfClasses2D', important=True,
                      allowsNull=False,
                      label="Set of reference classes",
                      help='Set of classes that will serve as reference for '
                           'the classification')
        form.addParam('maximumShift', params.IntParam, default=10,
                      label='Maximum shift (px):')
        form.addParam('keepBest', params.IntParam, default=2,
                      label='Number of best images:',
                      help='Number of the best images to keep for every class')
        form.addParam('numberOfClassifyIterations', params.IntParam, default=1,
                      label='Number of iterations in classify stage:',
                      help='Maximum number of iterations when the classification '
                           'of the whole image set is carried out')
        form.addParallelSection(threads=0, mpi=4)


    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """" Mainly prepare the command line for calling cuda corrrelation program"""

        self.lastIdProcessed = 0
        self.percentStopClassify = 5

        self.listContinueClass=[]
        self.iterReturnSplit = 0
        self.iterReturnClass = 0
        self.imgsExp = self._getExtraPath('imagesExp.xmd')
        self.imgsRef = self._getExtraPath('imagesRef.xmd')
        self.level=0
        #self.countStep = 0
        self.listNumImgs = []
        self.listNameImgs = []
        self.listRefImgs = []
        self.blockToProc=300
        self.listToProc=[]
        self.totalInList=0
        self.lastIDsaved = 0
        self.classesDone = 0

        self.last_time = time()

        self.insertedListFlag = {}

        self._loadInputList()
        numOfImages = len(self.listOfParticles) - self.totalInList
        print("new", numOfImages, len(self.listOfParticles), self.totalInList)
        numOfBlk = floor(numOfImages/self.blockToProc)
        lastBlk = numOfImages%self.blockToProc
        self.listToProc += [self.blockToProc]*int(numOfBlk)
        self.listToProc.append(lastBlk)
        self.totalInList += self.blockToProc*numOfBlk + lastBlk
        fDeps=[]
        for i in range(int(numOfBlk + 1)):
            if i==0:
                fDeps += self._insertStepsForParticles(True)
            else:
                fDeps += self._insertStepsForParticles(False)

        self._insertFunctionStep('createOutputStep', prerequisities=fDeps, wait=True)


    def _insertStepsForParticles(self, firstTime):

        deps = []
        stepConvert = self._insertFunctionStep('_convertSet', self.imgsExp,
                                               self.imgsRef, firstTime,
                                               prerequisites = [])
        deps.append(stepConvert)

        stepIdInputmd = self._insertFunctionStep\
            ('_generateInputMd', prerequisites=[stepConvert])
        deps.append(stepIdInputmd)

        expImgMd = self._getExtraPath('inputImagesExp.xmd')
        stepIdClassify = self._insertFunctionStep\
            ('_classifyStep', expImgMd, 0, prerequisites=[stepIdInputmd])
        deps.append(stepIdClassify)

        stepIdLevelUp = self._insertFunctionStep\
            ('levelUp', expImgMd, prerequisites=[stepIdClassify])
        deps.append(stepIdLevelUp)

        return deps


    # --------------------------- STEPS functions --------------------------

    def _stepsCheck(self):
        self._checkNewInput()
        self._checkNewOutput()

    #AJ hacer prints de la hora a la que entra y a la que sale para ver cuanto tiempo se lleva
    def _checkNewInput(self):
        """ Check if there are new particles to be processed and add the necessary
        steps."""
        initial_time = time()
        particlesFile = self.inputParticles.get().getFileName()

        now = datetime.now()
        self.lastCheck = getattr(self, 'lastCheck', now)
        mTime = datetime.fromtimestamp(getmtime(particlesFile))
        self.debug('Last check: %s, modification: %s'
                   % (prettyTime(self.lastCheck),
                      prettyTime(mTime)))

        # If the input have not changed since our last check,
        # it does not make sense to check for new input data
        if self.lastCheck > mTime and hasattr(self, 'listOfParticles'):
            return None

        self.lastCheck = now
        outputStep = self._getFirstJoinStep()

        # Open input and close it as soon as possible
        self._loadInputList()
        #self.isStreamClosed = self.inputParticles.get().getStreamState()
        #self.listOfParticles = self.inputParticles.get()


        if len(self.listOfParticles) > 0 and self.listOfParticles[0] == self.lastIdProcessed + 1:
            numOfImages = len(self.listOfParticles) - self.totalInList
            #print("new", numOfImages, len(self.listOfParticles), self.totalInList)
            numOfBlk = floor(numOfImages / self.blockToProc)
            lastBlk = numOfImages % self.blockToProc
            self.listToProc += [self.blockToProc]*int(numOfBlk)
            if lastBlk!=0:
                self.listToProc.append(lastBlk)
                stepsToAppend = int(numOfBlk+1)
            else:
                stepsToAppend = int(numOfBlk)
            self.totalInList += self.blockToProc * numOfBlk + lastBlk
            fDeps=[]
            print("stepsToAppend", stepsToAppend)
            init2 = time()
            for i in range(stepsToAppend):
                fDeps += self._insertStepsForParticles(False)
            finish2 = time()
            init3 = time()
            if outputStep is not None:
                outputStep.addPrerequisites(*fDeps)
            self.updateSteps()
        finish3=time()
        print("Calculo 1 exec time", finish2-init2)
        print("Calculo 2 exec time", finish3 - init3)

        final_time = time()
        exec_time = final_time-initial_time
        print("_checkNewInput exec_time", exec_time)


    def _checkNewOutput(self):
        """ Check for already done files and update the output set. """
        # Load previously done items (from text file)
        initial_time = time()
        particlesListId = self._readParticlesId()
        if particlesListId<100000:
            interval = 180.0
        else:
            interval = 300.0
        print("interval", interval, getSize(self.imgsExp))
        if initial_time<self.last_time+interval:
            print("salgo")
            return
        else:
            self.last_time = initial_time
            print("last_time",self.last_time)

        print("hago cosas")
        doneList = self._readDoneList()

        # Check for newly done items
        particlesListId = self._readParticlesId()
        #AJ antes:
        #newDone = [particlesId for particlesId in particlesListId
        #                   if particlesId not in doneList]
        if len(doneList)==0:
            newDone = range(0, particlesListId[0] + 1)
        else:
            newDone = range(doneList[0], particlesListId[0]+1)
        #firstTime = len(doneList) == 0
        #AJ antes:
        # allDone = len(doneList) + len(newDone)
        if len(doneList) == 0:
            allDone = len(newDone)
        else:
            allDone = doneList[0] + len(newDone)

        if len(doneList) == 0:
            print("En _checkNewOutput", 0, particlesListId[0], newDone[0], newDone[len(newDone) - 1])
        else:
            print("En _checkNewOutput", doneList[0], particlesListId[0], newDone[0], newDone[len(newDone)-1])
        sys.stdout.flush()

        # We have finished when there is not more inputs (stream closed)
        # and the number of processed particles is equal to the number of inputs
        self.finished = (self.isStreamClosed == Set.STREAM_CLOSED
                         and len(self.listOfParticles)==0)
        #AJ antes: and allDone == len(self.listOfParticles)
        streamMode = Set.STREAM_CLOSED if self.finished else Set.STREAM_OPEN

        print("self.finished", self.finished)
        sys.stdout.flush()

        if len(newDone)>1:
            #AJ before
            #for particleId in newDone:
            #    self._writeDoneList(particleId)
            self._writeDoneList(newDone[len(newDone)-1])
        elif not self.finished:
            # If we are not finished and no new output have been produced
            # it does not make sense to proceed and updated the outputs
            # so we exit from the function here
            return

        outSet = self._loadOutputSet(SetOfClasses2D, 'classes2D.sqlite')

        if (exists(self._getPath('classes2D.sqlite'))):
            if(exists(self._getExtraPath('last_classes.xmd'))
               and exists(self._getExtraPath('last_images.xmd'))):
                self._updateOutputSetOfClasses('outputClasses', outSet, streamMode)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

        if (exists(self._getPath('classes2D.sqlite'))):
            outSet.close()



        #outSetImg = self._loadOutputSet(SetOfParticles, 'outputParticles.sqlite')

        #if exists(self._getExtraPath('last_images.xmd')):
        #    #newOutSet = SetOfParticles(filename=self._getExtraPath('last_images.sqlite'))
        #    readSetOfParticles(self._getExtraPath('last_images.xmd'), outSetImg)
        #    #copy(self._getExtraPath('last_images.sqlite'), self._getPath('outputParticles.sqlite'))
        #    #remove(self._getExtraPath('last_images.sqlite'))

        #if (exists(self._getPath('outputParticles.sqlite'))):
        #    if exists(self._getExtraPath('last_images.xmd')):
        #        self._updateOutputSet('outputParticles', outSetImg, streamMode)

        #if self.finished:  # Unlock createOutputStep if finished all jobs
        #    outputStep = self._getFirstJoinStep()
        #    if outputStep and outputStep.isWaiting():
        #        outputStep.setStatus(STATUS_NEW)

        #if (exists(self._getPath('outputParticles.sqlite'))):
        #    outSetImg.close()

        final_time = time()
        exec_time = final_time - initial_time
        print("_checkNewOutput exec_time", exec_time)


    def _convertSet(self, imgsFileNameImages, imgsFileNameClasses, firstTime):

        setOfImg = self.inputParticles
        setOfClasses = self.inputClasses

        if not firstTime and setOfImg is None:
            return
        if firstTime and setOfImg is None or setOfClasses is None:
            return

        if firstTime:
            self.num_classes=setOfClasses.get().__len__()
            writeSetOfClasses2D(setOfClasses.get(),
                                imgsFileNameClasses,
                                writeParticles=True)
            self.classesToImages()
            print("self.lastIdProcessed",self.lastIdProcessed)
            writeSetOfParticles(setOfImg.get(), imgsFileNameImages, alignType=ALIGN_NONE,
                                firstId=self.lastIdProcessed)
        else:
            print("self.lastIdProcessed", self.lastIdProcessed)
            writeSetOfParticles(setOfImg.get(), imgsFileNameImages, alignType=ALIGN_NONE,
                                firstId=self.lastIdProcessed)


    def _generateInputMd(self):
        doneList = self._readDoneList()
        metadataItem = md.MetaData(self.imgsExp)
        mdSize = md.getSize(self.imgsExp)
        print("Len metadataItem", mdSize)
        metadataInput = md.MetaData()

        fn = self._getExtraPath('particles.txt')
        count=0
        print("self.listToProc",self.listToProc)
        print("self.listOfParticles", self.listOfParticles)
        firstIdx = self.listOfParticles[0]
        rows = iterRows(metadataItem)
        for item in rows:
            objId = item.getValue(md.MDL_ITEM_ID)
            print("Aqui",objId)
            if len(doneList) == 0 or objId > doneList[0]:
                if objId == self.listOfParticles[0]:
                    #row.readFromMd(metadataItem, objId)
                    item.addToMd(metadataInput)
                    self.lastIdProcessed = objId
                    with open(fn, 'w') as f:  # AJ antes: 'a'
                        f.write('%d\n' % objId)
                    count += 1
                    self.listOfParticles.pop(0)
                    if count==firstIdx+self.listToProc[0]:
                        break

        #for i in range(int(self.listToProc[0])):
        #    objId = self.listOfParticles[0]
        #    if len(doneList)==0 or objId>doneList[0]:
        #        if count==0:
        #            print("En _generateInputMd, First. ", objId)
        #        row.readFromMd(metadataItem, objId)
        #        print(row)
        #        row.addToMd(metadataInput)
        #        self.lastIdProcessed = objId
        #        with open(fn, 'w') as f:  # AJ antes: 'a'
        #            f.write('%d\n' % objId)
        #        count+=1
        #        self.listOfParticles.pop(0)

        self.totalInList -= self.listToProc[0]
        self.listToProc.pop(0)
        print("En _generateInputMd, Last. ", objId)

        metadataInput.write(self._getExtraPath('inputImagesExp.xmd'), MD_APPEND)



    def _classifyStep(self, expImgMd, iterReturnClass):

        if getSize(expImgMd) == 0:
            return

        level = self.level
        refImgMd = self.imgsRef
        iterReturnClass = self.classifyWholeSetStep(refImgMd, expImgMd,
                                                    level, iterReturnClass)
        self.generateOutputMD(level)
        self.generateOutputClasses(level)


    def classesToImages(self):


        first = True
        ref=1

        mdNewClasses = md.MetaData()
        mdClass = md.MetaData("classes@" + self.imgsRef)
        rows = iterRows(mdClass)
        for row in rows:
            row.setValue(md.MDL_REF, ref)
            row.addToMd(mdNewClasses)
            ref += 1
        mdNewClasses.write('classes@'
                           + self._getExtraPath('aux.xmd'),
                           MD_APPEND)

        ref=1
        mdAll = md.MetaData()
        blocks = md.getBlocksInMetaDataFile(self.imgsRef)
        for block in blocks:
            if block.startswith('class00'):
                # Read the list of images in this class
                mdImgsInClass = md.MetaData(block + "@" + self.imgsRef)
                mdImgsInClass.fillConstant(md.MDL_REF,ref)
                mdImgsInClass.fillConstant(md.MDL_WEIGHT, 0.0)
                mdImgsInClass.write(block + "@" + self._getExtraPath('aux.xmd'),
                                    MD_APPEND)
                mdAll.unionAll(mdImgsInClass)
                ref += 1

        blocks2 = md.getBlocksInMetaDataFile(self._getExtraPath('aux.xmd'))
        for block in blocks2:
            if block.startswith('class00') and first:
                self._params = {'auxMd': block + "@" + self._getExtraPath('aux.xmd'),
                                'outMd': self._getExtraPath('first_images.xmd')}
                args = ('-i %(auxMd)s -o %(outMd)s')
                self.runJob("xmipp_metadata_utilities",
                            args % self._params, numberOfMpi=1)

                first = False
                continue

            if block.startswith('class00') and not first:
                self._params = {'auxMd': block + "@" + self._getExtraPath('aux.xmd'),
                                'outMd': self._getExtraPath('first_images.xmd')}

                args = ('-i %(auxMd)s --set union_all %(outMd)s '
                        '-o %(outMd)s --mode append')
                self.runJob("xmipp_metadata_utilities",
                            args % self._params, numberOfMpi=1)

        self._params = {'inputMd': self._getExtraPath('first_images.xmd'),
                        'outMd': self._getExtraPath('first_images.xmd')}
        args = ('-i %(inputMd)s --fill maxCC constant 0.0 '
                '-o %(outMd)s')
        self.runJob("xmipp_metadata_utilities",
                    args % self._params, numberOfMpi=1)

        #remove(self._getExtraPath('aux.xmd'))
        copy(self._getExtraPath('aux.xmd'), self._getExtraPath('last_classes.xmd'))
        copy(self._getExtraPath('first_images.xmd'), self._getExtraPath('last_images.xmd'))


    def generateOutputClasses(self, level):

        if not exists(self._getExtraPath('last_classes.xmd')):
            copy(self._getExtraPath(join('level%03d' % level,
                                         'general_level%03d' % level +
                                         '_classes.xmd')),
                 self._getExtraPath('last_classes.xmd'))
            return

        finalMetadata = self._getExtraPath('final_classes.xmd')
        lastMetadata = self._getExtraPath('last_classes.xmd')
        newMetadata = self._getExtraPath(join('level%03d' % level,
                                              'general_level%03d' % level +
                                              '_classes.xmd'))

        mdNew = md.MetaData('classes@'+newMetadata)
        mdLast = md.MetaData('classes@'+lastMetadata)
        mdRef = md.MetaData('classes@'+self.imgsRef)
        numList=[]
        mdAll = md.MetaData()
        for item, itemNew, itemRef in zip(mdNew, mdLast, mdRef):
            particles_total = mdLast.getValue(md.MDL_CLASS_COUNT, item) + \
                              mdNew.getValue(md.MDL_CLASS_COUNT, itemNew)
            imageAux = mdRef.getValue(md.MDL_IMAGE, itemRef)
            numAux=mdRef.getValue(md.MDL_REF, itemRef)
            numList.append(numAux)
            row = md.Row()
            row.setValue(md.MDL_REF, numAux)
            row.setValue(md.MDL_IMAGE, imageAux)
            row.setValue(md.MDL_CLASS_COUNT, particles_total)
            row.addToMd(mdAll)
        mdAll.write('classes@' + finalMetadata, MD_APPEND)

        total = self.num_classes
        for i in range(total):
            self._params = {'lastMd': 'class%06d_images@' % numList[i] +
                                      lastMetadata,
                            'newMd': 'class%06d_images@' % numList[i] +
                                     newMetadata,
                            'outMd': 'class%06d_images@' % numList[i] +
                                     finalMetadata}
            args = ('-i %(lastMd)s --set union_all %(newMd)s '
                    '-o %(outMd)s --mode append')
            self.runJob("xmipp_metadata_utilities",
                        args % self._params, numberOfMpi=1)

        copy(self._getExtraPath('final_classes.xmd'),
             self._getExtraPath('last_classes.xmd'))


    def generateOutputMD(self, level):

        if not exists(self._getExtraPath('last_images.xmd')):
            copy(self._getExtraPath(join('level%03d' % level,
                                         'general_images_level%03d' % level +
                                         '.xmd')),
                 self._getExtraPath('last_images.xmd'))
            return

        self._params = {'lastMd': self._getExtraPath('last_images.xmd'),
                        'newMd': self._getExtraPath(join('level%03d' % level,
                                                         'general_images_level%03d'
                                                         % level + '.xmd')),
                        'outMd': self._getExtraPath('final_images.xmd')}
        args = ('-i %(lastMd)s --set union %(newMd)s -o %(outMd)s')
        self.runJob("xmipp_metadata_utilities",
                    args % self._params, numberOfMpi=1)

        copy(self._getExtraPath('final_images.xmd'),
             self._getExtraPath('last_images.xmd'))



    def classifyWholeSetStep(self, refImgMd, expImgMd, level, iterReturnClass):

        i=iterReturnClass
        if i == self.numberOfClassifyIterations:
            i -= 1
        while i <self.numberOfClassifyIterations:
            self.iterationStep(refImgMd, expImgMd, i, False, False, level)
            refImgMd = self._getExtraPath(join('level%03d' % level,
                                               'general_level%03d' % level +
                                               '_classes.xmd'))

            #if self.checkContinueClassification(level, i):
            #    return

            if i+1 < self.numberOfClassifyIterations:
                finalMetadata = self._getExtraPath(
                    join('level%03d' % level, 'general_level%03d' % level +
                         '_classes.xmd'))
                if exists(self._getExtraPath('final_classes.xmd')):
                    lastMetadata = self._getExtraPath('final_classes.xmd')
                else:
                    if level<2:
                        lastMetadata = self._getExtraPath('last_classes.xmd')
                    else:
                        lastMetadata = self._getExtraPath(
                            join('level%03d' % (level-1), 'general_level%03d'
                                 % (level-1) + '_classes.xmd'))
                newMetadata = self._getExtraPath(
                    join('level%03d' % level, 'general_level%03d' % level +
                         '_classes.xmd'))
                self.averageClasses(finalMetadata, lastMetadata, newMetadata, True)
            i += 1

        return iterReturnClass

    def checkContinueClassification(self, level, iter):

        diff=0
        i=0
        metadata = md.MetaData(self._getExtraPath(
            join('level%03d' % level, 'general_images_level%03d' % level + '.xmd')))

        for item in metadata:
            refImg = metadata.getValue(md.MDL_REF, item)
            nameImg = metadata.getValue(md.MDL_IMAGE, item)
            if iter==0:
                self.listContinueClass.append(nameImg)
                self.listContinueClass.append(refImg)
            else:
                if nameImg in self.listContinueClass:
                    idx = self.listContinueClass.index(nameImg) + 1
                    if refImg!=self.listContinueClass[idx]:
                        diff+=1
                    self.listContinueClass[idx]=refImg
                else:
                    diff += 1
                    self.listContinueClass.append(nameImg)
                    self.listContinueClass.append(refImg)
            i+=1
        num=(diff*100/i)
        if num<self.percentStopClassify and iter>0:
            return True
        else:
            return False


    def iterationStep (self, refSet, imgsExp, iter, flag_split, flag_attraction, level):

        if not exists(join(self._getExtraPath(), 'level%03d' % level)):
            mkdir(join(self._getExtraPath(), 'level%03d' % level))

        if iter==0 and flag_split==True:

            # First step: divide the metadata input file to generate
            # a couple of references
            if level==0:
                if not flag_attraction:
                    outDirName = imgsExp[0:imgsExp.find('extra')+6] + \
                                 'level%03d' % level + \
                                 imgsExp[imgsExp.find('extra')+5:-4]
                else:
                    outDirName = imgsExp[0:imgsExp.find('extra') + 6] + \
                                 'level%03d' % level + \
                                 imgsExp[imgsExp.find('level%03d' % level) + 8:-4]
            else:
                if not flag_attraction:
                    outDirName = imgsExp[0:imgsExp.find('extra') + 6] + \
                                 'level%03d' % level + \
                                 imgsExp[imgsExp.find('level%03d' % (level-1)) + 8:-4]
                else:
                    outDirName = imgsExp[0:imgsExp.find('extra') + 6] + \
                                 'level%03d' % level + \
                                 imgsExp[imgsExp.find('level%03d' % level) + 8:-4]
            self._params = {'imgsExp': imgsExp,
                            'outDir': outDirName}
            args = ('-i %(imgsExp)s -n 2 --oroot %(outDir)s')
            self.runJob("xmipp_metadata_split", args % self._params, numberOfMpi=1)

            # Second step: calculate the means of the previous metadata
            expSet1 = outDirName + '000001.xmd'
            avg1 = outDirName + '_000001'
            expSet2 = outDirName + '000002.xmd'
            avg2 = outDirName + '_000002'
            self._params = {'imgsSet': expSet1,
                            'outputAvg': avg1}
            args = ('-i %(imgsSet)s --save_image_stats %(outputAvg)s -v 0')
            self.runJob("xmipp_image_statistics", args % self._params, numberOfMpi=1)

            self._params = {'imgsSet': expSet2,
                            'outputAvg': avg2}
            args = ('-i %(imgsSet)s --save_image_stats %(outputAvg)s -v 0')
            self.runJob("xmipp_image_statistics", args % self._params, numberOfMpi=1)

            # Third step: generate a single metadata with the two previous averages
            refSet = self._getExtraPath(join('level%03d' % level,'refSet.xmd'))
            self._params = {'avg1': avg1 + 'average.xmp',
                            'avg2': avg2 + 'average.xmp',
                            'outputMd': refSet}
            args = ('-i %(avg1)s --set union %(avg2)s -o %(outputMd)s')
            self.runJob("xmipp_metadata_utilities", args % self._params, numberOfMpi=1)

        # Fourth step: calling program xmipp_cuda_correlation
        if flag_split:
            filename = 'level%03d' % level+'_classes.xmd'
            self._params = {'imgsRef': refSet,
                            'imgsExp': imgsExp,
                            'outputFile': 'images_level%03d' % level+'.xmd',
                            'tmpDir': join(self._getExtraPath(),'level%03d' % level),
                            'keepBest': self.keepBest.get(),
                            'maxshift': self.maximumShift.get(),
                            'outputClassesFile': filename,
                            }
        else:
            filename = 'general_level%03d' % level + '_classes.xmd'
            self._params = {'imgsRef': refSet,
                            'imgsExp': imgsExp,
                            'outputFile': 'general_images_level%03d' % level + '.xmd',
                            'tmpDir': join(self._getExtraPath(),'level%03d' % level),
                            'keepBest': self.keepBest.get(),
                            'maxshift': self.maximumShift.get(),
                            'outputClassesFile': filename,
                            }
        if not flag_split:
            args = '-i_ref %(imgsRef)s -i_exp %(imgsExp)s -o %(outputFile)s '\
                   '--odir %(tmpDir)s --keep_best %(keepBest)d '\
                   '--maxShift %(maxshift)d --simplifiedMd ' \
                   '--classify %(outputClassesFile)s'
            self.runJob("xmipp_cuda_correlation", args % self._params, numberOfMpi=1)
        else:
            Nrefs = getSize(refSet)
            args = '-i %(imgsExp)s --ref0 %(imgsRef)s --nref %(Nrefs)d ' \
                   '--iter 1 --distance correlation --classicalMultiref ' \
                   '--maxShift %(maxshift)d --odir %(cl2dDir)s'
            self._params['Nrefs']=Nrefs
            self._params['cl2dDir'] = self._getExtraPath(join('level%03d' % level))
            self.runJob("xmipp_classify_CL2D", args % self._params)
            copy(self._getExtraPath(
                join('level%03d' % level,"level_00","class_classes.xmd")),
                 self._getExtraPath(
                     join('level%03d' % level,'level%03d' % level+'_classes.xmd')))
            copy(self._getExtraPath(join('level%03d' % level,"images.xmd")),
                 self._getExtraPath(
                     join('level%03d' % level,'images_level%03d' % level + '.xmd')))


    def createOutputStep(self):

        outSet = self._loadOutputSet(SetOfClasses2D, 'classes2D.sqlite')

        if (exists(self._getPath('classes2D.sqlite'))):
            if (exists(self._getExtraPath('last_classes.xmd'))
                    and exists(self._getExtraPath('last_images.xmd'))):
                self._updateOutputSetOfClasses('outputClasses', outSet,
                                               Set.STREAM_CLOSED)

        if self.finished:  # Unlock createOutputStep if finished all jobs
            outputStep = self._getFirstJoinStep()
            if outputStep and outputStep.isWaiting():
                outputStep.setStatus(STATUS_NEW)

        if (exists(self._getPath('classes2D.sqlite'))):
            outSet.close()


    # --------------------------- UTILS functions -------------------------------

    def _loadInputList(self):
        """ Load the input set of ctfs and create a list. """
        initial_time = time()
        particlesSet = self._loadInputParticleSet()
        self.isStreamClosed = particlesSet.getStreamState()
        self.listOfParticles = [m.getObjId() for m in
                                particlesSet.iterItems(where="id>%d"
                                                    %self.lastIdProcessed)]
        #AJ antes: m.getObjId() for m in particlesSet]
        particlesSet.close()
        self.debug("Closed db.")
        final_time = time()
        print("_loadInputList exec time",final_time-initial_time)


    def _loadInputParticleSet(self):
        initial_time = time()
        particlesFile = self.inputParticles.get().getFileName()
        self.debug("Loading input db: %s" % particlesFile)
        particlesSet = SetOfParticles(filename=particlesFile)
        particlesSet.loadAllProperties()
        final_time = time()
        print("_loadInputParticleSet exec time", final_time - initial_time)
        return particlesSet

    def _getFirstJoinStep(self):
        for s in self._steps:
            if s.funcName == 'createOutputStep':
                return s
        return None

    def _readDoneList(self):
        """ Read from a text file the id's of the items that have been done. """
        DoneFile = self._getExtraPath('DONE.TXT')
        DoneList = []
        # Check what items have been previously done
        if exists(DoneFile):
            with open(DoneFile) as f:
                DoneList += [int(line.strip()) for line in f]
        return DoneList

    def _writeDoneList(self, particleId):
        """ Write to a text file the items that have been done. """
        AcceptedFile = self._getExtraPath('DONE.TXT')
        with open(AcceptedFile, 'w') as f: #AJ antes: 'a'
            f.write('%d\n' % particleId)

    def _readParticlesId(self):
        fn = self._getExtraPath('particles.txt')
        particlesList = []
        # Check what items have been previously done
        if exists(fn):
            with open(fn) as f:
                particlesList += [int(line.strip().split()[0]) for line in f]
        return particlesList

    def _loadOutputSet(self, SetClass, baseName):
        setFile = self._getPath(baseName)

        if exists(setFile):
            pwutils.path.cleanPath(setFile)

        #if exists(setFile):
        #    outputSet = SetClass(filename=setFile)
        #    outputSet.loadAllProperties()
        #    outputSet.enableAppend()
        #else:
        outputSet = SetClass(filename=setFile)
        outputSet.setStreamState(outputSet.STREAM_OPEN)

        inputs = self.inputParticles.get()
        outputSet.copyInfo(inputs)
        if SetClass is SetOfClasses2D:
            outputSet.setImages(inputs)
        return outputSet


    def _updateOutputSetOfClasses(self, outputName, outputSet, state=Set.STREAM_OPEN):

        outputSet.setStreamState(state)
        # if self.hasAttribute(outputName):
        #    print("En el if")
        #    self._fillClassesFromLevel(outputSet)
        #    outputSet.write()  # Write to commit changes
        #    outputAttr = getattr(self, outputName)
        #    # Copy the properties to the object contained in the protocol
        #    outputAttr.copy(outputSet, copyId=False)
        #    # Persist changes
        #    self._store(outputAttr)
        # else:
        print("En el else")
        inputs = self.inputParticles.get()
        # Here the defineOutputs function will call the write() method
        outputSet = self._createSetOfClasses2D(inputs)
        self._fillClassesFromLevel(outputSet)
        self._defineOutputs(**{outputName: outputSet})
        self._defineSourceRelation(self.inputParticles, outputSet)
        self._store(outputSet)

        # Close set databaset to avoid locking it
        outputSet.close()



    def _updateOutputSet(self, outputName, outputSet, state=Set.STREAM_OPEN):

        outputSet.setStreamState(state)
        if self.hasAttribute(outputName):
            print("En el if")
            #self._fillClassesFromLevel(outputSet)
            outputSet.write()  # Write to commit changes
            outputAttr = getattr(self, outputName)
            # Copy the properties to the object contained in the protocol
            outputAttr.copy(outputSet, copyId=False)
            # Persist changes
            self._store(outputAttr)
        else:
            print("En el else")
            inputs = self.inputParticles.get()
            # Here the defineOutputs function will call the write() method
            outputSet = self._createSetOfParticles()
            #self._fillClassesFromLevel(outputSet)
            self._defineOutputs(**{'outputParticles': outputSet})
            self._defineSourceRelation(self.inputParticles, outputSet)
            self._store(outputSet)
        # Close set databaset to avoid locking it
        outputSet.close()


    def _updateOutputSetClasses(self, outputName, outputSet, state=Set.STREAM_OPEN):

        outputSet.setStreamState(state)
        if self.hasAttribute(outputName):
            print("En el if")
            #self._fillClassesFromLevel(outputSet)
            outputSet.write()  # Write to commit changes
            outputAttr = getattr(self, outputName)
            # Copy the properties to the object contained in the protocol
            outputAttr.copy(outputSet, copyId=False)
            # Persist changes
            self._store(outputAttr)
        else:
            print("En el else")
            inputs = self.inputParticles.get()
            # Here the defineOutputs function will call the write() method
            outputSet = self._createSetOfClasses2D(inputs)
            #self._fillClassesFromLevel(outputSet)
            self._defineOutputs(**{'outputClasses': outputSet})
            self._defineSourceRelation(self.inputParticles, outputSet)
            self._store(outputSet)
        # Close set databaset to avoid locking it
        outputSet.close()

    def _updateParticle(self, item, row):
        item.setClassId(row.getValue(md.MDL_REF))
        item.setTransform(rowToAlignment(row, ALIGN_2D))

    def _updateClass(self, item):
        classId = item.getObjId()
        if classId in self._classesInfo:
            index, fn, _ = self._classesInfo[classId]
            item.setAlignment2D()
            rep = item.getRepresentative()
            rep.setLocation(index, fn)
            rep.setSamplingRate(self.inputParticles.get().getSamplingRate())


    def _loadClassesInfo(self, filename):
        """ Read some information about the produced 2D classes
        from the metadata file.
        """
        self._classesInfo = {}  # store classes info, indexed by class id
        mdClasses = md.MetaData(filename)
        for classNumber, row in enumerate(md.iterRows(mdClasses)):
            index, fn = xmippToLocation(row.getValue(md.MDL_IMAGE))
            self._classesInfo[classNumber + 1] = (index, fn, row.clone())


    def _fillClassesFromLevel(self, clsSet):
        """ Create the SetOfClasses2D from a given iteration. """
        myFileParticles = self._getExtraPath('last_images.xmd')
        myFileClasses = self._getExtraPath('last_classes.xmd')
        self._loadClassesInfo(myFileClasses)
        #blocks = md.getBlocksInMetaDataFile(myFileClasses)
        #for __, block in enumerate(blocks):
        #    if block.startswith('class0'):
        #        xmpMd = block + "@" + myFileClasses
        #        iterator = md.SetMdIterator(xmpMd, sortByLabel=md.MDL_ITEM_ID,
        #                                    updateItemCallback=self._updateParticle,
        #                                    skipDisabled=True)
        #        # itemDataIterator is not necessary because, the class SetMdIterator
        #        # contain all the information about the metadata
        #        clsSet.classifyItems(updateItemCallback=iterator.updateItem,
        #                             updateClassCallback=self._updateClass)
        xmpMd = myFileParticles
        iterator = md.SetMdIterator(xmpMd, sortByLabel=md.MDL_ITEM_ID,
                                    updateItemCallback=self._updateParticle,
                                    skipDisabled=True)

        # itemDataIterator is not necessary because, the class SetMdIterator
        # contain all the information about the metadata
        clsSet.classifyItems(updateItemCallback=iterator.updateItem,
                             updateClassCallback=self._updateClass)

    def levelUp(self, expImgMd):
        if getSize(expImgMd) == 0:
            return
        self.level += 1


    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        errors = []
        refImage = self.inputClasses.get()
        [x1, y1, z1] = refImage.getDimensions()
        [x2, y2, z2] = self.inputParticles.get().getDim()
        if x1 != x2 or y1 != y2 or z1 != z2:
            errors.append('The input images and the reference images '
                          'have different sizes')
        return errors


    def _summary(self):
        summary = []
        if not hasattr(self, 'outputClasses'):
            summary.append("Output alignment not ready yet.")
        else:
            summary.append("Input Particles: %s"
                           % self.inputParticles.get().getSize())
            summary.append("Aligned with reference images: %s"
                           % self.inputClasses.get().getSize())
        return summary

    def _citations(self):
        return ['Sorzano2010a']

    def _methods(self):
        methods = []
        if not hasattr(self, 'outputClasses'):
            methods.append("Output alignment not ready yet.")
        else:
            methods.append(
                "We aligned images %s with respect to the reference image set "
                "%s using Xmipp CUDA correlation"
                % (self.getObjectTag('inputParticles'),
                   self.getObjectTag('inputClasses')))

        return methods

